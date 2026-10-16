"""Controlled two-feature-set development experiment; no final-test access."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reseller_churn.config.validation_settings import DEVELOPMENT_FOLDS
from reseller_churn.data.temporal_dataset import TemporalDataset
from reseller_churn.evaluation.metrics import (
    calculate_ranking_metrics,
    calculate_threshold_metrics,
)
from reseller_churn.evaluation.selection import (
    SELECTION_RULE,
    THRESHOLDS,
    evaluate_threshold_tradeoffs,
    select_for_precision_floor,
)
from reseller_churn.modeling.development import (
    dataframe_sha256,
    file_sha256,
    fit_and_predict_candidate,
    repository_identity,
    validate_labeled_development_data,
    validate_selection_manifest,
    write_json,
)


SOURCE = ROOT / "results/codex-luna-verification-20261010-1"
OUTPUT = ROOT / "results/debugging/rfm_vs_seven_features_precision_030"
PRECISION_FLOOR = 0.30


def compare_predictions(predictions: pd.DataFrame, selections: dict) -> pd.DataFrame:
    """Pair the same cases and expose gains, regressions, and alert changes."""
    key = ["fold", "StoreID", "snapshot"]
    paired = None
    for feature_set in ("rfm", "pruned"):
        selection = selections[feature_set]
        if selection["status"] != "selected":
            raise ValueError("Case comparison requires two feasible operating points.")
        subset = predictions[predictions["feature_set"].eq(feature_set)].copy()
        subset[f"{feature_set}_score"] = subset["churn_probability"]
        subset[f"{feature_set}_alert"] = subset["churn_probability"].ge(
            selection["threshold"]
        ).astype(int)
        subset = subset[
            key + ["actual_churn", f"{feature_set}_score", f"{feature_set}_alert"]
        ]
        if paired is None:
            paired = subset
        else:
            paired = paired.merge(
                subset, on=key, how="outer", validate="one_to_one",
                suffixes=("", "_seven"), indicator=True,
            )
            if not paired["_merge"].eq("both").all():
                raise ValueError("Feature sets were evaluated on different cases.")
            if not paired["actual_churn"].equals(paired["actual_churn_seven"]):
                raise ValueError("Feature sets have different evaluation labels.")
            paired = paired.drop(columns=["actual_churn_seven", "_merge"])

    churn = paired["actual_churn"].eq(1)
    gained_alert = paired["rfm_alert"].eq(0) & paired["pruned_alert"].eq(1)
    removed_alert = paired["rfm_alert"].eq(1) & paired["pruned_alert"].eq(0)
    paired["change_type"] = np.select(
        [
            churn & gained_alert, churn & removed_alert,
            ~churn & gained_alert, ~churn & removed_alert,
        ],
        [
            "newly_caught_churn", "newly_missed_churn",
            "added_false_alert", "removed_false_alert",
        ],
        default="unchanged",
    )
    return paired.sort_values(key).reset_index(drop=True)


def operating_metrics(predictions: pd.DataFrame, selections: dict) -> pd.DataFrame:
    rows = []
    for (feature_set, fold), group in predictions.groupby(["feature_set", "fold"]):
        selection = selections[feature_set]
        if selection["status"] != "selected":
            continue
        groups = [
            ("overall", group),
            ("purchased_in_3m", group[group["n_orders_3m"].gt(0)]),
            ("no_purchase_in_3m", group[group["n_orders_3m"].eq(0)]),
        ]
        for subgroup, cases in groups:
            if cases.empty:
                continue
            target = cases["actual_churn"].to_numpy()
            scores = cases["churn_probability"].to_numpy()
            rows.append({
                "feature_set": feature_set, "fold": fold, "subgroup": subgroup,
                "threshold": selection["threshold"], "rows": len(cases),
                "churners": int(target.sum()),
                **calculate_threshold_metrics(target, scores, selection["threshold"]),
                **calculate_ranking_metrics(target, scores),
            })
    return pd.DataFrame(rows)


def check_baseline(predictions: pd.DataFrame, saved_path: Path, candidate_id: str) -> float:
    """Fail rather than attribute a changed baseline to the feature experiment."""
    saved = pd.read_csv(saved_path)
    saved = saved[saved["candidate_id"].eq(candidate_id)]
    baseline = predictions[predictions["feature_set"].eq("rfm")]
    key = ["fold", "StoreID", "snapshot"]
    paired = baseline.merge(
        saved[key + ["actual_churn", "churn_probability"]],
        on=key, how="outer", validate="one_to_one",
        suffixes=("", "_saved"), indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Reproduced RFM cases differ from the source run.")
    np.testing.assert_array_equal(paired["actual_churn"], paired["actual_churn_saved"])
    np.testing.assert_allclose(
        paired["churn_probability"], paired["churn_probability_saved"],
        rtol=0, atol=1e-10,
    )
    return float((paired["churn_probability"] - paired["churn_probability_saved"]).abs().max())


def main() -> None:
    paths = {
        "data": SOURCE / "preparation/labeled_development.csv",
        "selection": SOURCE / "development/selection/precision_03_0.3.json",
        "saved_predictions": SOURCE / "development/validation_predictions.csv",
    }
    source_selection = json.loads(paths["selection"].read_text())
    if file_sha256(paths["data"]) != source_selection["source_data_file_sha256"]:
        raise ValueError("Source development data has changed.")
    data = validate_labeled_development_data(pd.read_csv(paths["data"]))
    baseline = validate_selection_manifest(source_selection, dataframe_sha256(data))
    if (
        baseline.architecture != "logistic_regression"
        or baseline.imbalance_strategy != "balanced_weights"
        or baseline.feature_set != "rfm"
        or source_selection["minimum_precision"] != PRECISION_FLOOR
    ):
        raise ValueError("Source candidate does not match the agreed experiment.")
    candidates = [baseline, replace(baseline, feature_set="pruned")]
    temporal = TemporalDataset(data)
    prediction_tables = []
    tradeoff_tables = []
    selections = {}
    training_counts = []
    for candidate in candidates:
        fold_predictions = {}
        for fold in DEVELOPMENT_FOLDS:
            prepared = temporal.prepare_fold(fold)
            training = prepared.training_data
            validation = prepared.validation_data
            scores, _ = fit_and_predict_candidate(
                candidate, training, training["churn"], validation
            )
            target = validation["churn"].astype(int).to_numpy()
            fold_predictions[fold.name] = {"target": target, "probabilities": scores}
            predictions = validation.copy()
            predictions["fold"] = fold.name
            predictions["feature_set"] = candidate.feature_set
            predictions["candidate_id"] = candidate.candidate_id
            predictions["actual_churn"] = target
            predictions["churn_probability"] = scores
            prediction_tables.append(predictions)
            training_counts.append({
                "feature_set": candidate.feature_set, "fold": fold.name,
                "training_rows": len(training),
                "training_churners": int(training["churn"].sum()),
                "validation_rows": len(validation),
                "validation_churners": int(target.sum()),
            })
        tradeoffs = evaluate_threshold_tradeoffs(candidate.candidate_id, fold_predictions)
        tradeoffs["feature_set"] = candidate.feature_set
        tradeoff_tables.append(tradeoffs)
        selections[candidate.feature_set] = select_for_precision_floor(
            tradeoffs, PRECISION_FLOOR
        )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    baseline_error = check_baseline(
        predictions, paths["saved_predictions"], baseline.candidate_id
    )
    if selections["rfm"]["threshold"] != source_selection["threshold"]:
        raise ValueError("Reproduced RFM threshold differs from the source run.")
    metrics = operating_metrics(predictions, selections)
    tables = {
        "validation_predictions.csv": predictions,
        "threshold_tradeoffs.csv": pd.concat(tradeoff_tables, ignore_index=True),
        "fold_and_subgroup_metrics.csv": metrics,
        "selection_summary.csv": pd.DataFrame([
            {"feature_set": name, **selection} for name, selection in selections.items()
        ]),
        "training_counts.csv": pd.DataFrame(training_counts),
    }
    if all(selection["status"] == "selected" for selection in selections.values()):
        paired = compare_predictions(predictions, selections)
        context = data[["StoreID", "snapshot", "recency_days", "n_orders_3m", "revenue_3m"]]
        paired = paired.merge(context, on=["StoreID", "snapshot"], validate="one_to_one")
        tables["paired_cases.csv"] = paired
        tables["changed_cases.csv"] = paired[paired["change_type"].ne("unchanged")]
        tables["prediction_change_counts.csv"] = (
            paired.groupby(["fold", "change_type"]).size().rename("accounts").reset_index()
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUTPUT / name, index=False)
    manifest = {
        "purpose": "Controlled RFM versus seven-feature comparison",
        "precision_floor": PRECISION_FLOOR, "selection_rule": SELECTION_RULE,
        "thresholds": list(THRESHOLDS), "threshold_policy": ">=",
        "candidates": [candidate.to_dict() for candidate in candidates],
        "selections": selections, "training_counts": training_counts,
        "rfm_reproduction_max_absolute_score_error": baseline_error,
        "input_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in paths.values()
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in [
                Path(__file__), ROOT / "reseller_churn/modeling/development.py",
                ROOT / "reseller_churn/modeling/model_factory.py",
                ROOT / "reseller_churn/evaluation/selection.py",
                ROOT / "reseller_churn/config/feature_sets.py",
            ]
        },
        "repository": repository_identity(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "models_fitted": 4, "protected_final_test_labels_accessed": False,
        "limitations": [
            "Already inspected development periods also select thresholds",
            "One architecture and fixed settings; not an independent evaluation",
            "Feature bundle replaces annual count; no isolated feature attribution",
        ],
        "output_sha256": {name: file_sha256(OUTPUT / name) for name in tables},
    }
    import sklearn
    manifest["versions"]["scikit_learn"] = sklearn.__version__
    write_json(OUTPUT / "experiment_manifest.json", manifest)
    print(metrics[metrics["subgroup"].eq("overall")].to_string(index=False))
    print(tables["selection_summary.csv"][[
        "feature_set", "status", "threshold", "mean_recall", "worst_fold_recall",
        "mean_precision",
    ]].to_string(index=False))
    print(f"Saved outputs to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
