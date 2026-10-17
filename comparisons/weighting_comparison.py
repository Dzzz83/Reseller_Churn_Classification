"""Unit-weight ablation of fixed HGB10 and RF10 against cached balanced fits."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.matched_evaluation import publish_assessment
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
METHODS = ("unweighted_hgb", "unweighted_forest")


def fit_unweighted(
    method,
    train,
    validation_features,
    seeds=study.MODEL_SEEDS,
    forest_settings=BASELINE_RANDOM_FOREST,
):
    require(
        method in METHODS and train.churn.nunique() == 2,
        "Invalid unit-weight training request",
    )
    content = pd.util.hash_pandas_object(
        train[[*KEYS, "churn", *study.FEATURES]], index=False
    ).values.tobytes()
    outputs = []
    identities = []
    with threadpool_limits(limits=1):
        for seed in seeds:
            if method == "unweighted_hgb":
                model = HistGradientBoostingClassifier(
                    **study.BASE_PARAMETERS, random_state=seed
                )
            else:
                model = ModelFactory.create_random_forest(forest_settings, seed)
                model.set_params(model__n_jobs=1)
            # Unit weights are intentional; no sample_weight is passed to either model.
            model.fit(train[study.FEATURES], train.churn)
            outputs.append(
                model.predict_proba(validation_features[study.FEATURES])[:, 1]
            )
            estimator = (
                model if method == "unweighted_hgb" else model.named_steps["model"]
            )
            identity = {
                "candidate": method,
                "seed": seed,
                "training_sha256": hashlib.sha256(content).hexdigest(),
                "estimator_hash": joblib.hash(model),
                "training_weighting": "unit/no sample_weight",
                "parameters": estimator.get_params(),
                "training_rows": len(train),
                "training_positives": int(train.churn.sum()),
            }
            if method == "unweighted_forest":
                imputer = model.named_steps["imputer"]
                identity["imputer_statistics"] = [
                    float(x) if np.isfinite(x) else None for x in imputer.statistics_
                ]
                identity["missing_indicator_columns"] = (
                    imputer.indicator_.features_.tolist()
                )
            identities.append(identity)
    spread = float(np.ptp(outputs, axis=0).max())
    for identity in identities:
        identity["seed_score_range"] = spread
    return np.mean(outputs, axis=0), identities


def run(source, forest_source, output):
    require(not output.exists(), "Preserve prior output")
    verify_manifest(source)
    verify_manifest(forest_source)
    previous = json.loads((source / "protocol.json").read_text())
    require(previous["repeat_seeds"] == [301, 919, 1201], "Unexpected source repeats")
    data = read_csv(source / "development_input.csv")
    pd.testing.assert_frame_equal(
        data, read_csv(forest_source / "development_input.csv")
    )
    history_dir = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(history_dir)
    history = read_csv(history_dir / "case_histories.csv")
    profiles = history[[*KEYS, "n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    out = new_output_directory(output)
    protocol_path = ROOT / "comparisons/weighting_comparison_protocol.md"
    shutil.copyfile(protocol_path, out / "protocol.md")
    shutil.copyfile(source / "development_input.csv", out / "development_input.csv")
    baseline_spec = previous["arms"][0]
    specs = [
        {
            "id": "unweighted_hgb",
            "features": study.FEATURES,
            "parameters": study.BASE_PARAMETERS,
            "training_weighting": "unit",
            "description": "Fixed unit-weight original HGB10",
        },
        {
            "id": "unweighted_forest",
            "features": study.FEATURES,
            "parameters": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": None,
            },
            "training_weighting": "unit",
            "description": "Fixed unit-weight RF10",
        },
    ]
    write_json(
        out / "protocol.json",
        {
            "arms": [baseline_spec, *specs],
            "splits": previous["splits"],
            "model_seeds": list(study.MODEL_SEEDS),
            "repeat_seeds": [301, 919, 1201],
            "interval_tail_probability": 0.05 / 44,
            "new_fits": 600,
            "october_outcomes_accessed": False,
        },
    )
    parts = []
    for split in previous["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        source_fold = source / relative
        verify_manifest(source_fold)
        verify_manifest(forest_source / relative)
        folder = new_output_directory(out / relative)
        inner = read_csv(source_fold / "inner_scores.csv")
        baseline_inner = inner[inner.candidate.eq("baseline")].copy()
        assignment = baseline_inner.set_index(KEYS).inner_fold.reindex(
            pd.MultiIndex.from_frame(train[KEYS])
        )
        require(assignment.notna().all(), "Incomplete inner assignment")
        inner_parts = [baseline_inner]
        outer_parts = []
        identities = []
        choices = []
        base = read_csv(source_fold / "predictions.csv")
        base = base[base.strategy.eq("baseline")].copy()
        outer_parts.append(base)
        rf = read_csv(forest_source / relative / "predictions.csv")
        rf = rf[rf.strategy.eq("random_forest")].copy()
        rf["strategy"] = "balanced_forest_reference"
        outer_parts.append(rf)
        for method in METHODS:
            method_inner = []
            for inner_fold in sorted(assignment.unique()):
                ti, vi = (
                    np.flatnonzero(assignment.to_numpy() != inner_fold),
                    np.flatnonzero(assignment.to_numpy() == inner_fold),
                )
                require(
                    set(train.iloc[ti].StoreID).isdisjoint(train.iloc[vi].StoreID),
                    "Inner reseller overlap",
                )
                score, fitted = fit_unweighted(
                    method, train.iloc[ti], train.iloc[vi][study.FEATURES]
                )
                rows = train.iloc[vi][KEYS + ["churn"]].copy()
                rows["candidate"], rows["score"], rows["inner_fold"] = (
                    method,
                    score,
                    int(inner_fold),
                )
                method_inner.append(rows)
                identities.extend(
                    {**row, "stage": "inner", "inner_fold": int(inner_fold)}
                    for row in fitted
                )
            method_inner = pd.concat(method_inner, ignore_index=True)
            inner_parts.append(method_inner)
            selection = study.select_threshold(
                method_inner.churn.to_numpy(), method_inner.score.to_numpy()
            )
            choices.append({"candidate": method, **selection})
            scores, fitted = fit_unweighted(method, train, valid[study.FEATURES])
            identities.extend({**row, "stage": "outer"} for row in fitted)
            rows = valid[KEYS + ["churn", "n_orders_3m"]].copy()
            rows["score"], rows["cutoff"], rows["strategy"] = (
                scores,
                selection["cutoff"],
                method,
            )
            rows["repeat"], rows["outer_fold"] = split["repeat"], split["outer_fold"]
            outer_parts.append(rows)
        selection = study.select_threshold(
            baseline_inner.churn.to_numpy(), baseline_inner.score.to_numpy()
        )
        require(
            base.cutoff.eq(selection["cutoff"]).all(), "Cached baseline cutoff changed"
        )
        choices.append({"candidate": "baseline", **selection})
        inner = pd.concat(inner_parts, ignore_index=True)
        outer = pd.concat(outer_parts, ignore_index=True)
        parts.append(outer)
        inner.to_csv(folder / "inner_scores.csv", index=False)
        outer.to_csv(folder / "predictions.csv", index=False)
        pd.DataFrame(choices).to_csv(folder / "inner_selections.csv", index=False)
        write_json(folder / "model_identities.json", identities)
        cached = [
            row
            for row in json.loads((source_fold / "model_identities.json").read_text())
            if row["candidate"] == "baseline"
        ]
        write_json(folder / "cached_hgb_identities.json", cached)
        shutil.copyfile(source_fold / "split.json", folder / "split.json")
        write_manifest(
            folder,
            {
                "new_unit_weight_fits": 40,
                "inputs": input_identity(
                    {
                        "hgb_manifest": source_fold / "manifest.json",
                        "forest_manifest": forest_source / relative / "manifest.json",
                    }
                ),
            },
        )
        print(
            f"Completed repeat {split['repeat']} fold {split['outer_fold']}", flush=True
        )
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    references = {}
    for spec in specs:
        assessment_protocol = {
            "description": spec["description"],
            "arms": [baseline_spec, spec],
            "splits": previous["splits"],
            "repeat_seeds": [301, 919, 1201],
            "interval_tail_probability": 0.05 / 44,
        }
        decision = publish_assessment(
            out / spec["id"], out, data, profiles, predictions, assessment_protocol
        )
        references[spec["id"]] = input_identity(
            {"manifest": out / spec["id"] / "manifest.json"}
        )["manifest"]
        print(spec["id"], json.dumps(decision), flush=True)
    secondary = []
    for method, reference in [
        ("unweighted_hgb", "baseline"),
        ("unweighted_forest", "balanced_forest_reference"),
    ]:
        for repeat in [301, 919, 1201]:
            values = {}
            for arm in [method, reference]:
                part = predictions[
                    predictions.repeat.eq(repeat) & predictions.strategy.eq(arm)
                ]
                a, y = part.score.ge(part.cutoff), part.churn.eq(1)
                values[arm] = (int((a & y).sum()), int((a & ~y).sum()))
            secondary.append(
                {
                    "candidate": method,
                    "reference": reference,
                    "repeat": repeat,
                    "tp_difference": values[method][0] - values[reference][0],
                    "fp_reduction": values[reference][1] - values[method][1],
                    "scope": "descriptive within-architecture contrast; no promotion from weaker reference",
                }
            )
    pd.DataFrame(secondary).to_csv(out / "weight_effects.csv", index=False)
    write_json(out / "assessment_manifests.json", references)
    write_manifest(
        out,
        {
            "stage": "unit_weight_comparison",
            "new_fits": 600,
            "inputs": input_identity(
                {
                    "cached_hgb_manifest": source / "manifest.json",
                    "cached_forest_manifest": forest_source / "manifest.json",
                    "protocol": protocol_path,
                }
            ),
            "implementation": input_identity(
                {
                    "runner": Path(__file__),
                    "factory": ROOT / "reseller_churn/modeling/model_factory.py",
                    "analysis": ROOT / "comparisons/matched_evaluation.py",
                }
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/cadence-stability-20261016"
    )
    parser.add_argument(
        "--forest-source",
        type=Path,
        default=ROOT / "results/forest-comparison-20261016",
    )
    args = parser.parse_args()
    run(args.source, args.forest_source, args.output)
