"""Diagnose one saved development operating point without fitting a model.

Run from the repository root with bi_venv/bin/python. Outputs are descriptive
development evidence; docs/debugging.md records the interpretation and limits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "codex-luna-verification-20261010-1"
RUN = ROOT / "results" / RUN_ID
OUTPUT = ROOT / "results/debugging/april_2013_precision_030"
SNAPSHOT = pd.Timestamp("2013-04-01")
NEAR_THRESHOLD_DISTANCE = 0.10
LOW_SCORE_CUTOFF = 0.30
DECLINE_RATIO = 0.75
MATCH_FEATURES = ["recency_days", "n_orders_12m", "revenue_12m"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_band(score: float, threshold: float) -> str:
    """Fixed descriptive bands, not additional prediction thresholds."""
    if score >= threshold:
        return "alerted"
    if score >= threshold - NEAR_THRESHOLD_DISTANCE:
        return "near_threshold_miss"
    if score < LOW_SCORE_CUTOFF:
        return "low_score_miss"
    return "other_miss"


def load_cases() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, Path]]:
    paths = {
        "selection": RUN / "development/selection/precision_03_0.3.json",
        "predictions": RUN / "development/validation_predictions.csv",
        "features": RUN / "preparation/labeled_development.csv",
        "orders": ROOT / "datasets/orders_clean.csv",
    }
    selection = json.loads(paths["selection"].read_text())
    if selection["status"] != "selected":
        raise ValueError("The saved operating point has no selected candidate.")
    if selection["candidate"]["features"] != MATCH_FEATURES:
        raise ValueError("This diagnosis expects the documented three RFM inputs.")
    if selection["prediction_policy"]["threshold_comparison"] != ">=":
        raise ValueError("Unexpected threshold policy.")
    if sha256(paths["features"]) != selection["source_data_file_sha256"]:
        raise ValueError("Development features differ from the selected run.")

    features = pd.read_csv(paths["features"], parse_dates=["snapshot"])
    if (features["snapshot"] >= pd.Timestamp("2013-10-01")).any():
        raise ValueError("Protected rows are forbidden in this diagnosis.")
    predictions = pd.read_csv(paths["predictions"], parse_dates=["snapshot"])
    predictions = predictions[
        predictions["candidate_id"].eq(selection["candidate"]["candidate_id"])
        & predictions["snapshot"].eq(SNAPSHOT)
    ].copy()
    expected_ids = set(features.loc[features["snapshot"].eq(SNAPSHOT), "StoreID"])
    if set(predictions["StoreID"]) != expected_ids:
        raise ValueError("Saved predictions do not cover exactly the validation cohort.")
    if set(predictions["fold"]) != {"fold_2"}:
        raise ValueError("Unexpected validation fold in saved predictions.")
    cases = features[features["snapshot"].eq(SNAPSHOT)].merge(
        predictions, on=["StoreID", "snapshot"], validate="one_to_one"
    )
    if not cases["churn"].equals(cases["actual_churn"]):
        raise ValueError("Saved outcomes disagree with development labels.")
    if not cases["churn_probability"].between(0, 1).all():
        raise ValueError("Invalid saved probability scores.")

    threshold = selection["threshold"]
    cases["predicted_churn"] = cases["churn_probability"].ge(threshold).astype(int)
    cases["threshold_margin"] = cases["churn_probability"] - threshold
    cases["score_rank"] = cases["churn_probability"].rank(
        ascending=False, method="min"
    ).astype(int)
    cases["result_type"] = np.select(
        [
            cases["churn"].eq(1) & cases["predicted_churn"].eq(0),
            cases["churn"].eq(1) & cases["predicted_churn"].eq(1),
            cases["churn"].eq(0) & cases["predicted_churn"].eq(1),
        ],
        ["missed_churn", "detected_churn", "false_alert"],
        default="correct_non_churn",
    )
    # Band names apply only to actual churners; retained accounts remain controls.
    cases["miss_band"] = [
        score_band(score, threshold) if churn == 1 else "retained"
        for score, churn in zip(cases["churn_probability"], cases["churn"])
    ]
    cases["activity_group"] = np.where(
        cases["n_orders_3m"].gt(0), "purchased_in_3m", "no_purchase_in_3m"
    )
    return cases, features, selection, paths


def add_history(
    cases: pd.DataFrame, orders_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict order history before aggregation; never construct future labels."""
    orders = pd.read_csv(
        orders_path,
        usecols=["StoreID", "SalesOrderID", "OrderDate", "SubTotal"],
        parse_dates=["OrderDate"],
    )
    history_start = SNAPSHOT - pd.DateOffset(months=12)
    history = orders[
        orders["OrderDate"].ge(history_start)
        & orders["OrderDate"].lt(SNAPSHOT)
        & orders["StoreID"].isin(cases["StoreID"])
    ].copy()
    if history["SalesOrderID"].duplicated().any():
        raise ValueError("Order history contains duplicate order IDs.")
    history["month"] = history["OrderDate"].dt.to_period("M").astype(str)
    months = pd.period_range(history_start, SNAPSHOT - pd.Timedelta(days=1), freq="M")
    grid = pd.MultiIndex.from_product(
        [sorted(cases["StoreID"]), months.astype(str)], names=["StoreID", "month"]
    )
    monthly_totals = history.groupby(["StoreID", "month"]).agg(
        order_count=("SalesOrderID", "nunique"), revenue=("SubTotal", "sum")
    )
    monthly = monthly_totals.reindex(grid, fill_value=0).reset_index()
    last_orders = history.groupby("StoreID")["OrderDate"].max()
    cases = cases.copy()
    cases["last_order_date"] = cases["StoreID"].map(last_orders)
    reconstructed_recency = (SNAPSHOT - cases["last_order_date"]).dt.days
    np.testing.assert_array_equal(reconstructed_recency, cases["recency_days"])
    for months_back in (3, 6, 12):
        period_start = SNAPSHOT - pd.DateOffset(months=months_back)
        recent = history[history["OrderDate"].ge(period_start)]
        counts = recent.groupby("StoreID")["SalesOrderID"].nunique()
        revenues = recent.groupby("StoreID")["SubTotal"].sum()
        np.testing.assert_array_equal(
            cases["StoreID"].map(counts).fillna(0), cases[f"n_orders_{months_back}m"]
        )
        np.testing.assert_allclose(
            cases["StoreID"].map(revenues).fillna(0), cases[f"revenue_{months_back}m"],
            rtol=1e-10, atol=1e-5,
        )

    cases["previous_3m_orders"] = cases["n_orders_6m"] - cases["n_orders_3m"]
    cases["previous_3m_revenue"] = cases["revenue_6m"] - cases["revenue_3m"]
    for metric, current, previous in [
        ("order", "n_orders_3m", "previous_3m_orders"),
        ("revenue", "revenue_3m", "previous_3m_revenue"),
    ]:
        valid_previous = cases[previous].where(cases[previous].gt(0))
        ratio_column = f"recent_{metric}_ratio"
        cases[ratio_column] = cases[current] / valid_previous
        cases[f"{metric}_decline_25pct"] = cases[ratio_column].le(DECLINE_RATIO)
    cases["gap_missing"] = cases["mean_gap"].isna()
    cases["sparse_history"] = cases["n_orders_12m"].le(2)
    return cases, monthly


def match_retained_accounts(
    cases: pd.DataFrame, features: pd.DataFrame, selection: dict
) -> pd.DataFrame:
    """Three closest retained controls per miss in training-scaled RFM space.

    Controls can recur. Matching is descriptive, not a causal or performance
    estimate. Log transforms prevent dollar magnitude dominating the distance.
    """
    fold = next(
        fold for fold in selection["folds"]
        if fold["validation_snapshot"] == str(SNAPSHOT.date())
    )
    dates = pd.to_datetime(fold["training_snapshots"])
    if any(date + pd.DateOffset(months=6) > SNAPSHOT for date in dates):
        raise ValueError("Matching scale would use unavailable training labels.")
    training = features[features["snapshot"].isin(dates)]

    def coordinates(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[MATCH_FEATURES].astype(float).copy()
        for column in ["n_orders_12m", "revenue_12m"]:
            result[column] = np.log1p(result[column])
        return result

    train_coordinates = coordinates(training)
    scale = train_coordinates.quantile(.75) - train_coordinates.quantile(.25)
    scale = scale.where(scale.gt(0), 1.0)
    center = train_coordinates.median()
    scaled = (coordinates(cases) - center) / scale
    controls = cases[cases["churn"].eq(0)].sort_values("StoreID")
    comparison_columns = [
        "churn_probability", *MATCH_FEATURES, "n_orders_3m", "revenue_3m",
        "recent_order_ratio", "recent_revenue_ratio", "mean_gap", "overdue_ratio",
    ]
    rows = []
    for index, missed in cases[cases["result_type"].eq("missed_churn")].iterrows():
        differences = scaled.loc[controls.index] - scaled.loc[index]
        distances = np.sqrt((differences ** 2).sum(axis=1))
        nearest_indices = distances.sort_values(kind="stable").head(3).index
        for neighbor_rank, control_index in enumerate(nearest_indices, 1):
            control = cases.loc[control_index]
            row = {
                "missed_StoreID": int(missed["StoreID"]),
                "retained_StoreID": int(control["StoreID"]),
                "neighbor_rank": neighbor_rank,
                "rfm_distance": float(distances.loc[control_index]),
            }
            for column in comparison_columns:
                row[f"missed_{column}"] = missed[column]
                row[f"retained_{column}"] = control[column]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("overall", "all", cases)]
    group_columns = [
        "activity_group", "result_type", "miss_band", "revenue_decline_25pct",
        "order_decline_25pct", "sparse_history", "gap_missing",
    ]
    for column in group_columns:
        groups.extend((column, str(value), group) for value, group in cases.groupby(column))
    for dimension, value, group in groups:
        positive = group["churn"].eq(1)
        alert = group["predicted_churn"].eq(1)
        tp = int((positive & alert).sum())
        fp = int((~positive & alert).sum())
        fn = int((positive & ~alert).sum())
        rows.append({
            "dimension": dimension, "group": value, "rows": len(group),
            "churners": int(positive.sum()), "tp": tp, "fp": fp, "fn": fn,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "median_score": group["churn_probability"].median(),
            **{f"median_{name}": group[name].median() for name in [
                "recency_days", "n_orders_12m", "n_orders_3m", "revenue_3m",
                "recent_order_ratio", "recent_revenue_ratio", "mean_gap", "overdue_ratio",
            ]},
        })
    return pd.DataFrame(rows)


def main() -> None:
    cases, features, selection, paths = load_cases()
    cases, monthly = add_history(cases, paths["orders"])
    cases = cases.sort_values(
        ["churn_probability", "StoreID"]
    ).reset_index(drop=True)
    matches = match_retained_accounts(cases, features, selection)
    summary = summarize(cases)
    overall = summary.iloc[0]
    expected = selection["selected_fold_metrics"]["fold_2"]
    for name in ["tp", "fp", "fn"]:
        if int(overall[name]) != expected[name]:
            raise ValueError(f"Recomputed {name} disagrees with the saved manifest.")
    if len(monthly) != len(cases) * 12:
        raise ValueError("Monthly history does not include every zero-order month.")
    if len(matches) != int(overall["fn"]) * 3:
        raise ValueError("Missing retained-account comparisons.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "validation_cases.csv": cases,
        "missed_churners.csv": cases[cases["result_type"].eq("missed_churn")],
        "monthly_purchase_history.csv": monthly,
        "matched_retained_accounts.csv": matches,
        "cohort_summary.csv": summary,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT / name, index=False, float_format="%.10g")
    metadata = {
        "run_id": RUN_ID, "snapshot": str(SNAPSHOT.date()),
        "candidate": selection["candidate"], "threshold": selection["threshold"],
        "minimum_precision": selection["minimum_precision"],
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in paths.values()
        },
        "diagnostic_code_sha256": sha256(Path(__file__)),
        "history_window": "[2012-04-01, 2013-04-01)",
        "near_threshold_distance": NEAR_THRESHOLD_DISTANCE,
        "low_score_cutoff": LOW_SCORE_CUTOFF, "decline_ratio": DECLINE_RATIO,
        "matching": "3 retained neighbors; Euclidean distance in RFM with log1p counts/revenue and training-only median/IQR scaling; controls may recur",
        "models_fitted": False, "protected_labels_accessed": False,
        "checks": "input hash, unique complete joins, label agreement, score bounds, historical feature reconciliation, saved confusion counts, 12 months/account, 3 controls/miss",
        "output_sha256": {name: sha256(OUTPUT / name) for name in tables},
    }
    (OUTPUT / "diagnostic_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    key_groups = summary["dimension"].isin(["overall", "activity_group", "miss_band"])
    print(summary[key_groups].to_string(index=False))
    print(f"Saved diagnostic artifacts to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
