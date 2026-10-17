"""Matched raw/control and held-training-reference percentile ensembles."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
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

ROOT = Path(__file__).resolve().parents[1]
KEYS = study.KEYS
FEATURES = study.FEATURES + ["overdue_ratio", "gap_cv"]
RAW = "crossfit_raw"
NORMALIZED = "crossfit_percentile"
COMPONENT = "crossfit_components"
TAIL = 0.05 / 54


def percentile_scores(reference, scores):
    require(
        len(reference) > 0
        and np.isfinite(reference).all()
        and np.isfinite(scores).all(),
        "Invalid reference scores",
    )
    knots, counts = np.unique(reference, return_counts=True)
    ranks = (np.cumsum(counts) - counts / 2) / len(reference)
    return np.interp(scores, knots, ranks, left=0.0, right=1.0)


def key_records(frame):
    keys = frame[KEYS].copy()
    keys["snapshot"] = keys.snapshot.dt.strftime("%Y-%m-%d")
    return keys.to_dict("records")


def fit_components(train, validation_features):
    raw, normalized, identities = [], [], []
    spec = dict(
        id=COMPONENT,
        features=FEATURES,
        parameters={
            **study.BASE_PARAMETERS,
            "max_leaf_nodes": 7,
            "l2_regularization": 1.0,
        },
    )
    with threadpool_limits(limits=1):
        for half, (ti, ri) in enumerate(study.grouped_splits(train, 2, 42)):
            fitting, reference = train.iloc[ti], train.iloc[ri]
            require(
                set(fitting.StoreID).isdisjoint(reference.StoreID), "Reference leakage"
            )
            _, fitted, spread, models = study.fit_scores(
                fitting, validation_features[FEATURES], spec, retain_models=True
            )
            for model, identity in zip(models, fitted):
                reference_score = model.predict_proba(reference[FEATURES])[:, 1]
                score = model.predict_proba(validation_features[FEATURES])[:, 1]
                mapped = percentile_scores(reference_score, score)
                raw.append(score)
                normalized.append(mapped)
                identity.update(
                    half=half,
                    features=FEATURES,
                    seed_score_range=spread,
                    reference_cases=key_records(reference),
                    reference_scores=reference_score.tolist(),
                    validation_cases=key_records(validation_features),
                    validation_raw_scores=score.tolist(),
                    validation_percentiles=mapped.tolist(),
                )
                identities.append(identity)
    return {
        RAW: np.mean(raw, axis=0),
        NORMALIZED: np.mean(normalized, axis=0),
    }, identities


def run(output):
    source = ROOT / "results/cadence-stability-20261016"
    verify_manifest(source)
    previous = json.loads((source / "protocol.json").read_text())
    data = read_csv(source / "development_input.csv")
    histories = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(histories)
    history = read_csv(histories / "case_histories.csv")
    profiles = history[KEYS + ["n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    out = new_output_directory(output)
    protocol = ROOT / "comparisons/crossfit_percentile_protocol.md"
    shutil.copyfile(protocol, out / "protocol.md")
    data.to_csv(out / "development_input.csv", index=False)
    specs = [
        dict(
            id=arm,
            features=FEATURES,
            description=(
                "Fixed symmetric-half raw-probability ensemble"
                if arm == RAW
                else "Fixed held-training-reference percentile ensemble"
            ),
            components=[
                dict(
                    id=COMPONENT,
                    features=FEATURES,
                    parameters={
                        **study.BASE_PARAMETERS,
                        "max_leaf_nodes": 7,
                        "l2_regularization": 1.0,
                    },
                )
            ],
        )
        for arm in [RAW, NORMALIZED]
    ]
    baseline_spec = previous["arms"][0]
    write_json(
        out / "protocol.json",
        dict(
            arms=[baseline_spec, *specs],
            splits=previous["splits"],
            repeat_seeds=previous["repeat_seeds"],
            interval_tail_probability=TAIL,
            new_fits=600,
            october_outcomes_accessed=False,
        ),
    )
    parts = []
    for split in previous["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        source_fold = source / relative
        verify_manifest(source_fold)
        folder = new_output_directory(out / relative)
        cached_inner = read_csv(source_fold / "inner_scores.csv")
        baseline_inner = cached_inner[cached_inner.candidate.eq("baseline")]
        assignment = baseline_inner.set_index(KEYS).inner_fold.reindex(
            pd.MultiIndex.from_frame(train[KEYS])
        )
        require(assignment.notna().all(), "Incomplete inner assignment")
        inner_parts, identities = [baseline_inner], []
        for fold in sorted(assignment.unique()):
            fitting = train.iloc[np.flatnonzero(assignment.to_numpy() != fold)]
            held = train.iloc[np.flatnonzero(assignment.to_numpy() == fold)]
            require(set(fitting.StoreID).isdisjoint(held.StoreID), "Inner leakage")
            scores, fitted = fit_components(fitting, held[KEYS + FEATURES])
            identities.extend(
                dict(**item, stage="inner", inner_fold=int(fold)) for item in fitted
            )
            for arm, score in scores.items():
                rows = held[KEYS + ["churn"]].copy()
                rows["score"], rows["candidate"], rows["inner_fold"] = (
                    score,
                    arm,
                    int(fold),
                )
                inner_parts.append(rows)
        inner = pd.concat(inner_parts, ignore_index=True)
        choices = []
        for arm in ["baseline", RAW, NORMALIZED]:
            rows = inner[inner.candidate.eq(arm)]
            choices.append(
                dict(
                    candidate=arm,
                    **study.select_threshold(
                        rows.churn.to_numpy(), rows.score.to_numpy()
                    ),
                )
            )
        cached_outer = read_csv(source_fold / "predictions.csv")
        baseline = cached_outer[cached_outer.strategy.eq("baseline")]
        require(
            baseline.cutoff.eq(choices[0]["cutoff"]).all(), "Cached baseline changed"
        )
        scores, fitted = fit_components(train, valid[KEYS + FEATURES])
        identities.extend(dict(**item, stage="outer") for item in fitted)
        outer_parts = [baseline]
        for arm, score in scores.items():
            rows = valid[KEYS + ["churn", "n_orders_3m"]].copy()
            rows["score"], rows["strategy"] = score, arm
            rows["cutoff"] = next(
                item["cutoff"] for item in choices if item["candidate"] == arm
            )
            rows["repeat"], rows["outer_fold"] = split["repeat"], split["outer_fold"]
            outer_parts.append(rows)
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
            dict(
                new_fits=len(identities),
                inputs=input_identity(
                    dict(source_manifest=source_fold / "manifest.json")
                ),
            ),
        )
        print(
            f"Completed repeat {split['repeat']} fold {split['outer_fold']}", flush=True
        )
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    references = {}
    for spec in specs:
        assessment_protocol = dict(
            description=spec["description"],
            arms=[baseline_spec, spec],
            splits=previous["splits"],
            repeat_seeds=previous["repeat_seeds"],
            interval_tail_probability=TAIL,
        )
        decision = publish_assessment(
            out / spec["id"], out, data, profiles, predictions, assessment_protocol
        )
        references[spec["id"]] = input_identity(
            dict(manifest=out / spec["id"] / "manifest.json")
        )["manifest"]
        print(spec["id"], json.dumps(decision), flush=True)
    write_json(out / "assessment_manifests.json", references)
    write_manifest(
        out,
        dict(
            stage="crossfit_percentile",
            fits=600,
            inputs=input_identity(
                dict(
                    source=source / "manifest.json",
                    history=histories / "manifest.json",
                    protocol=protocol,
                )
            ),
            implementation=input_identity(
                dict(
                    runner=Path(__file__),
                    fitting=Path(study.__file__),
                    assessment=ROOT / "comparisons/matched_evaluation.py",
                )
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
