"""Prepare matched development datasets without training or final outcome access."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from companyx_readonly import query
from reseller_churn.config.project_paths import get_run_directory
from reseller_churn.config.validation_settings import EXPECTED_LABELED_COUNTS
from reseller_churn.modeling.development import (
    file_sha256, repository_identity, validate_labeled_development_data, write_json,
)

SOURCE = ROOT / "results/debugging/companyx_features"
CUTOFF = pd.Timestamp("2013-04-01")
REVENUE_COLUMNS = ["revenue_3m", "revenue_12m"]
HEADER_COLUMNS = [
    "SalesOrderID", "StoreID", "OrderDate", "SubTotal", "TaxAmt", "Freight", "TotalDue",
]


def reconcile_headers(clean_orders, raw_headers):
    """Match accepted historical orders and verify the monetary source contract."""
    for name, frame, columns in [
        ("clean orders", clean_orders, HEADER_COLUMNS[:4]),
        ("headers", raw_headers, HEADER_COLUMNS),
    ]:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if frame[HEADER_COLUMNS[:4]].isna().any().any():
            raise ValueError(f"{name}: missing order identity/date/subtotal")
        if frame.SalesOrderID.duplicated().any():
            raise ValueError(f"{name}: duplicate order IDs")
    clean = clean_orders.copy()
    raw = raw_headers.copy()
    clean["OrderDate"] = pd.to_datetime(clean.OrderDate, errors="raise")
    raw["OrderDate"] = pd.to_datetime(raw.OrderDate, errors="raise")
    if raw.OrderDate.ge(CUTOFF).any():
        raise ValueError("Headers reach the excluded feature cutoff")
    clean = clean.loc[clean.OrderDate.lt(CUTOFF), HEADER_COLUMNS[:4]]
    paired = clean.merge(
        raw[HEADER_COLUMNS], on="SalesOrderID", how="left", validate="one_to_one",
        suffixes=("_clean", ""), indicator=True,
    )
    if not paired._merge.eq("both").all():
        raise ValueError("Missing accepted historical orders in headers")
    for column in ["StoreID", "OrderDate"]:
        if not paired[column].eq(paired[f"{column}_clean"]).all():
            raise ValueError(f"Source mismatch: {column}")
    amounts = paired[["SubTotal", "TaxAmt", "Freight", "TotalDue"]].to_numpy(dtype=float)
    if not np.isfinite(amounts).all():
        raise ValueError("Non-finite monetary values")
    if (paired[["SubTotal", "TotalDue"]] <= 0).any().any():
        raise ValueError("Accepted amounts must be positive")
    if (paired[["TaxAmt", "Freight"]] < 0).any().any():
        raise ValueError("Tax and freight must be nonnegative")
    if not np.allclose(paired.SubTotal, paired.SubTotal_clean, atol=1e-6, rtol=0):
        raise ValueError("Source mismatch: SubTotal")
    if not np.allclose(
        paired.TotalDue, paired.SubTotal + paired.TaxAmt + paired.Freight,
        atol=1e-4, rtol=0,
    ):
        raise ValueError("TotalDue formula mismatch")
    return paired[HEADER_COLUMNS].sort_values("SalesOrderID").reset_index(drop=True)


def build_alternative(baseline, headers):
    """Replace only revenue totals; first reproduce the existing SubTotal totals."""
    alternative = baseline.copy()
    for snapshot, cases in baseline.groupby("snapshot"):
        endpoint = pd.Timestamp(snapshot)
        for months, column in [(3, "revenue_3m"), (12, "revenue_12m")]:
            history = headers[
                headers.OrderDate.ge(endpoint - pd.DateOffset(months=months))
                & headers.OrderDate.lt(endpoint)
            ]
            totals = history.groupby("StoreID")[["SubTotal", "TotalDue"]].sum()
            subtotal = cases.StoreID.map(totals.SubTotal).fillna(0)
            if not np.allclose(subtotal, cases[column], atol=1e-8, rtol=1e-12):
                raise ValueError(f"Baseline revenue mismatch: {snapshot} {column}")
            alternative.loc[cases.index, column] = cases.StoreID.map(totals.TotalDue).fillna(0)
    unchanged = baseline.columns.difference(REVENUE_COLUMNS)
    pd.testing.assert_frame_equal(baseline[unchanged], alternative[unchanged], check_exact=True)
    return alternative


def prepare(run_id):
    """Validate existing references, query historical headers, and save a new run."""
    # Reuse the supported run-ID validation without writing in the supported run path.
    get_run_directory(run_id)
    output = ROOT / "results/debugging/subtotal_vs_totaldue" / run_id
    if output.exists():
        raise FileExistsError(f"Choose a new run ID; directory exists: {output}")
    baseline_path = SOURCE / "development_features.csv"
    manifest_path = SOURCE / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_hash = manifest["input_sha256"][str(baseline_path.relative_to(ROOT))]
    if file_sha256(baseline_path) != expected_hash:
        raise ValueError("Baseline hash differs from the recorded experiment")
    baseline = validate_labeled_development_data(pd.read_csv(baseline_path))
    baseline = baseline.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)
    for snapshot, (rows, churners) in EXPECTED_LABELED_COUNTS.items():
        cases = baseline[baseline.snapshot.eq(snapshot)]
        if len(cases) != rows or int(cases.churn.sum()) != churners:
            raise ValueError(f"Unexpected development counts: {snapshot}")
    clean_path = ROOT / "datasets/orders_clean.csv"
    clean = pd.read_csv(clean_path, parse_dates=["OrderDate"])
    raw = query("""
SELECT h.SalesOrderID, c.StoreID,
       CONVERT(varchar(23), h.OrderDate, 121) AS OrderDate,
       h.SubTotal, h.TaxAmt, h.Freight, h.TotalDue
FROM Sales.SalesOrderHeader AS h
JOIN Sales.Customer AS c ON c.CustomerID = h.CustomerID
WHERE c.StoreID IS NOT NULL AND h.OrderDate < '20130401'
ORDER BY h.SalesOrderID;
""")
    headers = reconcile_headers(clean, raw)
    alternative = build_alternative(baseline, headers)
    protocol_path = ROOT / "docs/debugging.md"
    protocol = protocol_path.read_text().split(
        "## SubTotal versus TotalDue: planned grouped comparison — 2026-10-14", 1
    )[1]
    output.mkdir(parents=True, exist_ok=False)
    frames = {
        "historical_headers_raw.csv": raw,
        "accepted_historical_headers.csv": headers,
        "development_subtotal.csv": baseline,
        "development_totaldue.csv": alternative,
    }
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    (output / "protocol.md").write_text("# SubTotal versus TotalDue protocol\n" + protocol)
    # Check serialized artifacts as well as the in-memory pair.
    saved_baseline = pd.read_csv(output / "development_subtotal.csv", float_precision="round_trip")
    saved_alternative = pd.read_csv(output / "development_totaldue.csv", float_precision="round_trip")
    unchanged = baseline.columns.difference(REVENUE_COLUMNS)
    pd.testing.assert_frame_equal(saved_baseline[unchanged], saved_alternative[unchanged], check_exact=True)
    write_json(output / "preparation_manifest.json", {
        "run_id": run_id,
        "stage": "matched_data_preparation_only",
        "development_rows": len(baseline), "development_churners": int(baseline.churn.sum()),
        "reseller_identities": baseline.StoreID.nunique(),
        "raw_historical_orders": len(raw), "accepted_historical_orders": len(headers),
        "excluded_raw_order_ids": sorted(set(raw.SalesOrderID) - set(headers.SalesOrderID)),
        "exclusive_source_cutoff": str(CUTOFF.date()),
        "changed_columns": REVENUE_COLUMNS,
        "final_test_labels_accessed": False, "models_fitted": 0,
        "checks": {
            "baseline_hash": "passed", "snapshot_counts": "passed",
            "source_reconciliation": "passed", "subtotal_reconstruction": "passed",
            "unchanged_columns_exact_parity": "passed", "serialized_parity": "passed",
        },
        "input_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in [baseline_path, manifest_path, clean_path]},
        "code_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in [Path(__file__), ROOT / "debug/companyx_readonly.py"]},
        "output_sha256": {name: file_sha256(output / name) for name in [*frames, "protocol.md"]},
        "repository": repository_identity(),
    })
    print(f"Prepared {len(baseline)} matched rows under {output}")
    print(f"Accepted historical orders: {len(headers)}; models fitted: 0")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    prepare(parser.parse_args().run_id)
