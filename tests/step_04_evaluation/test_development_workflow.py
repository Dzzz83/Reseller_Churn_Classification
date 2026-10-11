import json

import pandas as pd

from pipeline import development_workflow
from reseller_churn.config.feature_sets import FEATURE_SETS
from reseller_churn.config.validation_settings import LABELED_SNAPSHOTS
from reseller_churn.evaluation.selection import evaluate_threshold_tradeoffs
from reseller_churn.modeling.development import load_selected_predictor


def make_labeled_development_data() -> pd.DataFrame:
    features = sorted({feature for values in FEATURE_SETS.values() for feature in values})
    rows = []
    for snapshot_index, snapshot in enumerate(LABELED_SNAPSHOTS):
        for store_id in range(1, 9):
            row = {
                "StoreID": store_id,
                "snapshot": snapshot,
                "churn": int(store_id in {1, 2}),
            }
            row.update(
                {
                    feature: float(store_id + snapshot_index)
                    for feature in features
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_workflow_writes_feasible_and_no_feasible_floor_manifests_once(
    tmp_path,
    monkeypatch,
):
    import reseller_churn.modeling.development as model_development

    run_id = "workflow-smoke"
    run_directory = tmp_path / "results" / run_id
    preparation_directory = run_directory / "preparation"
    development_directory = run_directory / "development"
    preparation_directory.mkdir(parents=True)
    development_directory.mkdir()
    data = make_labeled_development_data()
    data.to_csv(preparation_directory / "labeled_development.csv", index=False)

    monkeypatch.setattr(
        development_workflow,
        "get_run_directory",
        lambda value: run_directory,
    )
    monkeypatch.setattr(
        development_workflow,
        "MODELS_DIR",
        tmp_path / "models",
    )
    monkeypatch.setattr(development_workflow, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_development, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        development_workflow,
        "repository_identity",
        lambda: {"revision": "test", "has_local_changes": False},
    )

    def collect_one_candidate(candidates, _temporal_data):
        candidate = candidates[0]
        fold_predictions = {
            "fold_1": {
                "target": [1, 0],
                "probabilities": [0.9, 0.1],
            },
            "fold_2": {
                "target": [1, 0],
                "probabilities": [0.6, 0.9],
            },
        }
        tradeoffs = evaluate_threshold_tradeoffs(
            candidate.candidate_id,
            fold_predictions,
        )
        tradeoffs["architecture"] = candidate.architecture
        tradeoffs["feature_set"] = candidate.feature_set
        tradeoffs["imbalance_strategy"] = candidate.imbalance_strategy
        predictions = pd.DataFrame(
            [
                {
                    "candidate_id": candidate.candidate_id,
                    "architecture": candidate.architecture,
                    "feature_set": candidate.feature_set,
                    "imbalance_strategy": candidate.imbalance_strategy,
                    "fold": "fold_1",
                    "StoreID": 1,
                    "snapshot": "2013-01-01",
                    "actual_churn": 1,
                    "churn_probability": 0.9,
                },
                {
                    "candidate_id": candidate.candidate_id,
                    "architecture": candidate.architecture,
                    "feature_set": candidate.feature_set,
                    "imbalance_strategy": candidate.imbalance_strategy,
                    "fold": "fold_1",
                    "StoreID": 2,
                    "snapshot": "2013-01-01",
                    "actual_churn": 0,
                    "churn_probability": 0.1,
                },
                {
                    "candidate_id": candidate.candidate_id,
                    "architecture": candidate.architecture,
                    "feature_set": candidate.feature_set,
                    "imbalance_strategy": candidate.imbalance_strategy,
                    "fold": "fold_2",
                    "StoreID": 1,
                    "snapshot": "2013-04-01",
                    "actual_churn": 1,
                    "churn_probability": 0.6,
                },
                {
                    "candidate_id": candidate.candidate_id,
                    "architecture": candidate.architecture,
                    "feature_set": candidate.feature_set,
                    "imbalance_strategy": candidate.imbalance_strategy,
                    "fold": "fold_2",
                    "StoreID": 2,
                    "snapshot": "2013-04-01",
                    "actual_churn": 0,
                    "churn_probability": 0.9,
                },
            ]
        )
        return pd.DataFrame(), predictions, tradeoffs

    monkeypatch.setattr(
        development_workflow,
        "_collect_candidate_results",
        collect_one_candidate,
    )

    development_workflow.run_development(
        run_id,
        precision_floors=(0.401, 0.404, 0.6),
        architectures=("logistic_regression",),
        feature_sets=("rfm",),
        imbalance_strategies=("none",),
        seeds=(42,),
    )

    manifest_paths = sorted((development_directory / "selection").glob("*.json"))
    assert [path.name for path in manifest_paths] == [
        "precision_01_0.401.json",
        "precision_02_0.404.json",
        "precision_03_0.6.json",
    ]
    selected_manifests = [
        json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths[:2]
    ]
    no_feasible = json.loads(manifest_paths[2].read_text(encoding="utf-8"))
    assert all(manifest["status"] == "selected" for manifest in selected_manifests)
    assert no_feasible["status"] == "no_feasible_candidate"
    assert no_feasible["candidate"] is None
    assert no_feasible["model_bundle"] is None
    assert selected_manifests[0]["model_bundle"] == selected_manifests[1]["model_bundle"]
    assert selected_manifests[0]["search_space"]["subset_search"] is True
    assert selected_manifests[0]["precision_floor_status"] == (
        "sensitivity_value_not_approved_business_minimum"
    )

    predictor_manifest, predictor = load_selected_predictor(manifest_paths[0])
    assert predictor_manifest["candidate"]["candidate_id"] == predictor.candidate_id
    assert len(list((tmp_path / "models" / "provisional" / run_id).glob("*.joblib"))) == 1
