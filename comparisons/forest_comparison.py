"""Fixed RF10 and HGB/RF blend, matched to verified cached HGB predictions."""

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.fixed_blend import combine_components
from comparisons.matched_evaluation import evaluate_predictions
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)
from reseller_churn.config.model_settings import BASELINE_RANDOM_FOREST
from reseller_churn.modeling.model_factory import ModelFactory

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]
FOREST = "random_forest"
BLEND = "forest_blend"


def fit_forest(
    train, validation_features, seeds=study.MODEL_SEEDS, settings=BASELINE_RANDOM_FOREST
):
    """Training-only imputation/weights; validation labels are not accepted."""
    columns = study.FEATURES
    require(train.churn.nunique() == 2, "Training requires both classes")
    weights = compute_sample_weight("balanced", train.churn)
    content = pd.util.hash_pandas_object(
        train[[*KEYS, "churn", *columns]], index=False
    ).values.tobytes()
    outputs, identities = [], []
    with threadpool_limits(limits=1):
        for seed in seeds:
            model = ModelFactory.create_random_forest(settings, seed)
            model.set_params(model__n_jobs=1)
            model.fit(
                train[columns], train.churn.astype(int), model__sample_weight=weights
            )
            outputs.append(model.predict_proba(validation_features[columns])[:, 1])
            imputer = model.named_steps["imputer"]
            identities.append(
                {
                    "candidate": FOREST,
                    "seed": seed,
                    "training_sha256": hashlib.sha256(content).hexdigest(),
                    "estimator_hash": joblib.hash(model),
                    "imputer_statistics": [
                        float(x) if np.isfinite(x) else None
                        for x in imputer.statistics_
                    ],
                    "missing_indicator_columns": imputer.indicator_.features_.tolist(),
                    "forest_parameters": model.named_steps["model"].get_params(),
                    "training_rows": len(train),
                    "training_positives": int(train.churn.sum()),
                }
            )
    spread = float(np.ptp(outputs, axis=0).max())
    for identity in identities:
        identity["seed_score_range"] = spread
    return np.mean(outputs, axis=0), identities


def evaluate_fold(features, split, source_fold, folder):
    train, valid = validate_partition(features, split)
    verify_manifest(source_fold)
    saved_inner = read_csv(source_fold / "inner_scores.csv")
    baseline_inner = saved_inner[saved_inner.candidate.eq("baseline")].copy()
    assignment = baseline_inner.set_index(KEYS).inner_fold.reindex(
        pd.MultiIndex.from_frame(train[KEYS])
    )
    require(assignment.notna().all(), "Incomplete saved inner membership")
    inner_parts, identities = [], []
    for inner_fold in sorted(assignment.unique()):
        ti, vi = (
            np.flatnonzero(assignment.to_numpy() != inner_fold),
            np.flatnonzero(assignment.to_numpy() == inner_fold),
        )
        require(
            set(train.iloc[ti].StoreID).isdisjoint(train.iloc[vi].StoreID),
            "Inner reseller leakage",
        )
        score, fitted = fit_forest(train.iloc[ti], train.iloc[vi][study.FEATURES])
        rows = train.iloc[vi][KEYS + ["churn"]].copy()
        rows["score"], rows["candidate"], rows["inner_fold"] = (
            score,
            FOREST,
            int(inner_fold),
        )
        inner_parts.append(rows)
        identities.extend(
            {**item, "stage": "inner", "inner_fold": int(inner_fold)} for item in fitted
        )
    forest_inner = pd.concat(inner_parts, ignore_index=True)
    inner = pd.concat([baseline_inner, forest_inner], ignore_index=True)
    inner_blend = combine_components(
        inner.rename(columns={"candidate": "strategy"}), FOREST, BLEND
    ).rename(columns={"strategy": "candidate"})
    inner = pd.concat([inner, inner_blend], ignore_index=True)
    selections = []
    for arm in ("baseline", FOREST, BLEND):
        rows = inner[inner.candidate.eq(arm)]
        selections.append(
            {
                "candidate": arm,
                **study.select_threshold(rows.churn.to_numpy(), rows.score.to_numpy()),
            }
        )
    saved_outer = read_csv(source_fold / "predictions.csv")
    baseline = saved_outer[saved_outer.strategy.eq("baseline")].copy()
    baseline_cutoff = next(
        row["cutoff"] for row in selections if row["candidate"] == "baseline"
    )
    require(baseline.cutoff.eq(baseline_cutoff).all(), "Cached baseline cutoff differs")
    scores, fitted = fit_forest(train, valid[study.FEATURES])
    identities.extend({**row, "stage": "outer"} for row in fitted)
    forest = valid[KEYS + ["churn", "n_orders_3m"]].copy()
    forest["score"], forest["strategy"] = scores, FOREST
    forest["repeat"], forest["outer_fold"] = split["repeat"], split["outer_fold"]
    forest["cutoff"] = next(
        row["cutoff"] for row in selections if row["candidate"] == FOREST
    )
    outer = pd.concat([baseline, forest], ignore_index=True)
    blend = combine_components(outer, FOREST, BLEND)
    blend["cutoff"] = next(
        row["cutoff"] for row in selections if row["candidate"] == BLEND
    )
    outer = pd.concat([outer, blend], ignore_index=True)
    folder.mkdir(parents=True, exist_ok=False)
    inner.to_csv(folder / "inner_scores.csv", index=False)
    outer.to_csv(folder / "predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(folder / "inner_selections.csv", index=False)
    write_json(folder / "model_identities.json", identities)
    shutil.copyfile(source_fold / "split.json", folder / "split.json")
    cached_identities = [
        row
        for row in json.loads((source_fold / "model_identities.json").read_text())
        if row["candidate"] == "baseline"
    ]
    write_json(folder / "cached_hgb_identities.json", cached_identities)
    write_manifest(
        folder,
        {
            "new_forest_fits": len(identities),
            "reused_hgb_fits": len(cached_identities),
            "inputs": input_identity(
                {"source_manifest": source_fold / "manifest.json"}
            ),
        },
    )
    return outer


def run(source, output):
    require(not output.exists(), "Preserve existing output")
    verify_manifest(source)
    source_protocol = json.loads((source / "protocol.json").read_text())
    require(
        source_protocol["repeat_seeds"] == [301, 919, 1201],
        "Unexpected source repetitions",
    )
    features = read_csv(source / "development_input.csv")
    history_dir = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(history_dir)
    history = read_csv(history_dir / "case_histories.csv")
    profiles = history[[*KEYS, "n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    baseline_spec = source_protocol["arms"][0]
    forest_spec = {
        "id": FOREST,
        "features": study.FEATURES,
        "parameters": asdict(BASELINE_RANDOM_FOREST),
        "preprocessing": "training median imputer with missing indicators",
        "class_weight": None,
        "training_weights": "balanced",
        "n_jobs": 1,
    }
    blend_spec = {
        "id": BLEND,
        "features": study.FEATURES,
        "components": [baseline_spec, forest_spec],
        "weights": [0.5, 0.5],
    }
    out = new_output_directory(output)
    protocol = ROOT / "comparisons/forest_comparison_protocol.md"
    (out / "protocol.md").write_bytes(protocol.read_bytes())
    shutil.copyfile(source / "development_input.csv", out / "development_input.csv")
    write_json(
        out / "protocol.json",
        {
            "arms": [baseline_spec, forest_spec, blend_spec],
            "model_seeds": list(study.MODEL_SEEDS),
            "splits": source_protocol["splits"],
            "repeat_seeds": [301, 919, 1201],
            "interval_tail_probability": 0.05 / 38,
            "new_fits": 300,
            "cached_hgb_fits_reused": 300,
            "october_outcomes_accessed": False,
        },
    )
    parts = []
    for split in source_protocol["splits"]:
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        parts.append(evaluate_fold(features, split, source / relative, out / relative))
        print(
            f"Completed repeat {split['repeat']} fold {split['outer_fold']}", flush=True
        )
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    assessment_manifests = {}
    for spec in (forest_spec, blend_spec):
        arm = spec["id"]
        assessment = new_output_directory(out / arm)
        shutil.copyfile(
            out / "development_input.csv", assessment / "development_input.csv"
        )
        shutil.copyfile(protocol, assessment / "protocol.md")
        write_json(
            assessment / "protocol.json",
            {
                "description": "Fixed RF10"
                if arm == FOREST
                else "Fixed equal-weight HGB/RF10 blend",
                "arms": [baseline_spec, spec],
                "splits": source_protocol["splits"],
                "repeat_seeds": [301, 919, 1201],
                "interval_tail_probability": 0.05 / 38,
            },
        )
        subset = predictions[predictions.strategy.isin(["baseline", arm])]
        subset.to_csv(assessment / "validation_predictions.csv", index=False)
        decision = evaluate_predictions(
            features, subset, profiles, assessment, arm, 0.05 / 38
        )
        for split in source_protocol["splits"]:
            relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
            folder = new_output_directory(assessment / relative)
            root_fold = out / relative
            inner = read_csv(root_fold / "inner_scores.csv")
            inner = inner[inner.candidate.isin(["baseline", arm])]
            inner.to_csv(folder / "inner_scores.csv", index=False)
            selections = pd.read_csv(
                root_fold / "inner_selections.csv", float_precision="round_trip"
            )
            selections[selections.candidate.isin(["baseline", arm])].to_csv(
                folder / "inner_selections.csv", index=False
            )
            shutil.copyfile(
                root_fold / "model_identities.json",
                folder / "component_model_identities.json",
            )
            write_manifest(folder, {"new_fits": 0, "reference_forest_fit_count": 20})
        write_json(
            assessment / "baseline_fidelity.json",
            {
                "status": "unchanged cached HGB scores and inner-cutoff objective",
                "changed_baseline_decisions": 0,
            },
        )
        write_manifest(
            assessment,
            {
                "stage": "forest_comparison_assessment",
                "new_fits": 0,
                "decision": decision,
            },
        )
        assessment_manifests[arm] = input_identity(
            {"manifest": assessment / "manifest.json"}
        )["manifest"]
        print(arm, json.dumps(decision), flush=True)
    write_json(out / "assessment_manifests.json", assessment_manifests)
    write_manifest(
        out,
        {
            "stage": "forest_comparison",
            "new_forest_fits": 300,
            "reused_hgb_fits": 300,
            "inputs": input_identity(
                {
                    "cached_study_manifest": source / "manifest.json",
                    "protocol": protocol,
                }
            ),
            "implementation": input_identity(
                {
                    "runner": Path(__file__),
                    "factory": ROOT / "reseller_churn/modeling/model_factory.py",
                    "blend": ROOT / "comparisons/fixed_blend.py",
                    "analysis": ROOT / "comparisons/matched_evaluation.py",
                }
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/cadence-stability-20261016"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
