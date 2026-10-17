"""Investigate persistent errors using development-only pre-snapshot histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.history_comparison import (
    DESCRIPTORS,
    comparison_groups,
    history_descriptors,
    match_cases,
    matched_differences,
    summarize_differences,
)
from comparisons.baseline_diagnosis.inputs import read_csv, require, verify_baseline
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)
from reseller_churn.config.frozen_model import FEATURES
from reseller_churn.data.preparation import build_features, load_sources

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    require(not args.output.exists(), "Output directory exists")
    features, predictions, audit = verify_baseline(args.source, args.orders)
    verify_manifest(args.diagnosis)
    cases = read_csv(args.diagnosis / "account_stability.csv")
    cases = cases[cases.policy.eq("inner_selected")].copy()
    # Reconstruct decisions independently of saved recurrence columns.
    original = predictions[predictions.strategy.eq("original_baseline")].copy()
    original["alert"] = original.score.ge(original.cutoff)
    counts = original.groupby(["StoreID", "snapshot"]).alert.sum().rename("recomputed")
    checked = cases.merge(counts, on=["StoreID", "snapshot"], validate="one_to_one")
    require(checked.alerts.eq(checked.recomputed).all(), "Recurrence counts differ")
    cases = cases[["StoreID", "snapshot", "alerts"]].merge(
        features, on=["StoreID", "snapshot"], validate="one_to_one"
    )
    source_path = args.inputs / "source_manifest.json"
    source_manifest = json.loads(source_path.read_text())
    preparation = json.loads(
        (ROOT / "results/development-precision-diagnosis/manifest.json").read_text()
    )
    require(
        file_sha256(source_path) == preparation["input_hashes"]["source_manifest"],
        "Changed source manifest",
    )
    paths = {
        "orders": args.orders,
        "stores": ROOT / "datasets/stores_clean.csv",
        "headers": args.inputs / "order_headers.csv",
        "lines": args.inputs / "order_lines.csv",
    }
    for name in ("headers", "lines"):
        require(
            file_sha256(paths[name])
            == source_manifest["exports"][paths[name].name]["sha256"],
            "Changed source export",
        )
    coverage = source_manifest["coverage"]
    sources = load_sources(
        paths, coverage["from"], coverage["through_exclusive"], features.snapshot.max()
    )
    for snapshot, frame in features.groupby("snapshot"):
        rebuilt = (
            build_features(sources, snapshot).set_index("StoreID").loc[frame.StoreID]
        )
        np.testing.assert_allclose(
            rebuilt[list(FEATURES)],
            frame[list(FEATURES)],
            rtol=1e-10,
            atol=1e-9,
            equal_nan=True,
        )
    history, trajectories = history_descriptors(
        cases[["StoreID", "snapshot"]], sources.orders, sources.lines
    )
    original_history = read_csv(Path(audit["inputs"]["original_development"]["path"]))
    for feature in (
        "n_orders_12m",
        "mean_gap",
        "gap_cv",
        "overdue_ratio",
        "revenue_ratio_3m",
        "orders_ratio_3m",
    ):
        aligned = original_history[["StoreID", "snapshot", feature]].merge(
            history,
            on=["StoreID", "snapshot"],
            validate="one_to_one",
            suffixes=("_saved", ""),
        )
        np.testing.assert_allclose(
            aligned[f"{feature}_saved"],
            aligned[feature],
            rtol=1e-10,
            atol=1e-9,
            equal_nan=True,
        )
    cases = cases[["StoreID", "snapshot", "churn", "alerts", *FEATURES]].merge(
        history, on=["StoreID", "snapshot"], validate="one_to_one"
    )
    matches, scales, coverage_table = match_cases(cases)
    differences = matched_differences(cases, matches)
    summary = summarize_differences(differences)
    raw = []
    for name, (targets, controls) in comparison_groups(cases).items():
        for role, mask in [("target", targets), ("control", controls)]:
            for snapshot, frame in cases[mask].groupby("snapshot"):
                for feature in DESCRIPTORS:
                    raw.append(
                        {
                            "comparison": name,
                            "role": role,
                            "snapshot": snapshot,
                            "feature": feature,
                            "cases": len(frame),
                            "resellers": frame.StoreID.nunique(),
                            "missing": int(frame[feature].isna().sum()),
                            "median": frame[feature].median(),
                            "q25": frame[feature].quantile(0.25),
                            "q75": frame[feature].quantile(0.75),
                        }
                    )
    out = new_output_directory(args.output)
    protocol = ROOT / "comparisons/persistent_error_investigation.md"
    (out / "protocol.md").write_bytes(protocol.read_bytes())
    for name, table in {
        "case_histories": cases,
        "order_trajectories": trajectories,
        "matches": matches,
        "matching_scales": scales,
        "matching_coverage": coverage_table,
        "matched_differences": differences,
        "summary": summary,
        "raw_groups": pd.DataFrame(raw),
    }.items():
        table.to_csv(out / f"{name}.csv", index=False)
    verification = {
        "status": "passed",
        "development_cases": len(cases),
        "resellers": cases.StoreID.nunique(),
        "historical_orders": len(sources.orders),
        "historical_lines": len(sources.lines),
        "baseline_features_reconciled": 10,
        "saved_history_descriptors_reconciled": 6,
        "recurrence_cases_checked": len(checked),
        "models_fitted": 0,
        "cutoffs_selected": 0,
        "october_outcomes_accessed": False,
    }
    write_json(out / "verification.json", verification)
    implementation = [
        Path(__file__),
        Path(__file__).parent / "baseline_diagnosis/history_comparison.py",
    ]
    write_manifest(
        out,
        {
            "stage": "persistent_error_investigation",
            "audit": audit,
            "inputs": input_identity(
                {
                    **paths,
                    "source_manifest": source_path,
                    "diagnosis_manifest": args.diagnosis / "manifest.json",
                    "protocol": protocol,
                }
            ),
            "implementation": input_identity({p.name: p for p in implementation}),
            **verification,
        },
    )
    print(json.dumps(verification, indent=2))
    print(
        coverage_table[coverage_table.k.eq(3)]
        .groupby("comparison")
        .agg(targets=("StoreID", "size"), matched=("matched_controls", "min"))
        .to_string()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/repeated-grouped-precision"
    )
    parser.add_argument(
        "--diagnosis",
        type=Path,
        default=ROOT / "results/baseline-error-diagnosis-20261016-linux-audit",
    )
    parser.add_argument(
        "--inputs", type=Path, default=ROOT / "inputs/regeneration-20261016"
    )
    parser.add_argument(
        "--orders",
        type=Path,
        default=ROOT
        / "results/research-provenance-20261016/orders_clean_historical_bytes.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
