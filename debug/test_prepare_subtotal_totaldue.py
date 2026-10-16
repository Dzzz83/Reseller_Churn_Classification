"""Boundary and source-integrity checks for the matched revenue experiment."""

import pandas as pd
import pytest

from prepare_subtotal_totaldue import build_alternative, prepare, reconcile_headers


def example_headers():
    return pd.DataFrame({
        "SalesOrderID": [1, 2, 3, 4, 5],
        "StoreID": [1, 1, 1, 1, 2],
        "OrderDate": pd.to_datetime([
            "2011-12-31", "2012-01-01", "2012-10-01", "2013-01-01", "2012-07-01",
        ]),
        "SubTotal": [1000., 20., 30., 2000., 40.],
        "TaxAmt": [80., 2., 3., 160., 4.],
        "Freight": [20., 1., 1., 40., 2.],
        "TotalDue": [1100., 23., 34., 2200., 46.],
    })


def example_baseline():
    return pd.DataFrame({
        "StoreID": [1, 2], "snapshot": ["2013-01-01"] * 2,
        "revenue_3m": [30., 0.], "revenue_12m": [50., 40.],
        "share_bikes": [.4, .5], "weighted_discount_3m": [.1, float("nan")],
        "churn": [1, 0],
    })


def test_half_open_windows_and_no_recent_orders():
    baseline = example_baseline()
    result = build_alternative(baseline, example_headers())
    assert result.revenue_3m.tolist() == [34., 0.]
    assert result.revenue_12m.tolist() == [57., 46.]
    unchanged = ["StoreID", "snapshot", "share_bikes", "weighted_discount_3m", "churn"]
    pd.testing.assert_frame_equal(baseline[unchanged], result[unchanged], check_exact=True)
    assert baseline.revenue_12m.tolist() == [50., 40.]


def test_rejects_incorrect_baseline_reconstruction():
    baseline = example_baseline()
    baseline.loc[0, "revenue_3m"] = 31.
    with pytest.raises(ValueError, match="Baseline revenue mismatch"):
        build_alternative(baseline, example_headers())


def test_excludes_raw_orders_not_in_clean_population():
    raw = example_headers()
    clean = raw.iloc[1:][["SalesOrderID", "StoreID", "OrderDate", "SubTotal"]]
    result = reconcile_headers(clean, raw)
    assert result.SalesOrderID.tolist() == [2, 3, 4, 5]


@pytest.mark.parametrize("problem,message", [
    ("missing", "Missing accepted"), ("duplicate", "duplicate order"),
    ("store", "Source mismatch: StoreID"), ("date", "Source mismatch: OrderDate"),
    ("subtotal", "Source mismatch: SubTotal"), ("formula", "formula mismatch"),
    ("nonfinite", "Non-finite"), ("cutoff", "excluded feature cutoff"),
    ("negative", "must be positive"),
])
def test_rejects_invalid_headers(problem, message):
    raw = example_headers()
    clean = raw[["SalesOrderID", "StoreID", "OrderDate", "SubTotal"]].copy()
    if problem == "missing":
        raw = raw.iloc[1:]
    elif problem == "duplicate":
        raw = pd.concat([raw, raw.iloc[:1]], ignore_index=True)
    elif problem == "store":
        raw.loc[0, "StoreID"] = 99
    elif problem == "date":
        raw.loc[0, "OrderDate"] = pd.Timestamp("2011-12-30")
    elif problem == "subtotal":
        raw.loc[0, "SubTotal"] += 1
    elif problem == "formula":
        raw.loc[0, "TotalDue"] += 1
    elif problem == "nonfinite":
        raw.loc[0, "TaxAmt"] = float("nan")
    elif problem == "cutoff":
        raw.loc[0, "OrderDate"] = pd.Timestamp("2013-04-01")
    elif problem == "negative":
        raw.loc[0, "TotalDue"] = -1
    with pytest.raises(ValueError, match=message):
        reconcile_headers(clean, raw)


def test_refuses_to_overwrite_existing_run(monkeypatch, tmp_path):
    monkeypatch.setattr("prepare_subtotal_totaldue.ROOT", tmp_path)
    (tmp_path / "results/debugging/subtotal_vs_totaldue/existing").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Choose a new run ID"):
        prepare("existing")
