"""Calibrate immutable development component scores on disjoint reference resellers."""

import argparse
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.breadth_reporting import markdown_table
from comparisons.crossfit_percentile import FEATURES, RAW
from comparisons.development_checks import validate_development_cases
from comparisons.development_reporting import cutoff_transfer_tables
from comparisons.matched_evaluation import publish_assessment
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "retained_logistic"
TAIL = 0.00025
LIMITS = {"max_recall_loss": 0.05, "max_fp_increase": 0}
KEYS = study.KEYS


def score_logits(scores):
    values = np.asarray(scores, dtype=float)
    require(
        np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
        "Invalid saved probabilities",
    )
    return logit(np.clip(values, 1e-12, 1 - 1e-12)).reshape(-1, 1)


def calibrate(reference_scores, reference_labels, held_scores):
    """Only the reference labels enter this fit; held labels are not accepted."""
    y = np.asarray(reference_labels)
    require(
        len(y) == len(reference_scores) and set(y) == {0, 1},
        "Calibration reference requires both binary classes",
    )
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(score_logits(reference_scores), y)
    mapped = model.predict_proba(score_logits(held_scores))[:, 1]
    return mapped, {
        "slope": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "iterations": int(model.n_iter_[0]),
        "reference_labels": y.tolist(),
        "reference_churners": int(y.sum()),
        "mapped_scores": mapped.tolist(),
    }


def keyed_cases(records):
    frame = pd.DataFrame(records)
    frame["snapshot"] = pd.to_datetime(frame.snapshot)
    return frame


def component_populations(base, held, records):
    """Verify saved fit/reference identities against a fresh grouped reconstruction."""
    require(set(base.StoreID).isdisjoint(held.StoreID), "Held reseller leakage")
    halves = list(
        StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42).split(
            base, base.churn, base.StoreID
        )
    )
    require(len(records) == 10, "Expected two halves and five seeds")
    for half, (ti, ri) in enumerate(halves):
        fitting, reference = base.iloc[ti], base.iloc[ri]
        require(set(fitting.StoreID).isdisjoint(reference.StoreID), "Reference leakage")
        subset = [row for row in records if row["half"] == half]
        require(
            len(subset) == 5
            and {row["seed"] for row in subset} == set(study.MODEL_SEEDS),
            "Component seed membership differs",
        )
        content = pd.util.hash_pandas_object(
            fitting[KEYS + ["churn"] + FEATURES], index=False
        ).values.tobytes()
        digest = hashlib.sha256(content).hexdigest()
        for record in subset:
            pd.testing.assert_frame_equal(
                keyed_cases(record["reference_cases"]),
                reference[KEYS].reset_index(drop=True),
            )
            pd.testing.assert_frame_equal(
                keyed_cases(record["validation_cases"]),
                held[KEYS].reset_index(drop=True),
            )
            require(
                record["training_sha256"] == digest, "Component fitting hash differs"
            )
            require(
                len(record["reference_scores"]) == len(reference)
                and len(record["validation_raw_scores"]) == len(held),
                "Score population mismatch",
            )
            yield record, reference


def mapped_components(base, held, records):
    mapped, identities = [], []
    for record, reference in component_populations(base, held, records):
        scores, identity = calibrate(
            record["reference_scores"], reference.churn, record["validation_raw_scores"]
        )
        mapped.append(scores)
        identities.append(
            {
                **record,
                **identity,
                "fitting_rows": len(base) - len(reference),
                "fitting_resellers": base[
                    ~base.StoreID.isin(reference.StoreID)
                ].StoreID.nunique(),
                "reference_rows": len(reference),
                "reference_resellers": reference.StoreID.nunique(),
            }
        )
    return np.mean(mapped, axis=0), identities


def intersection_gate(directory):
    pairs = pd.read_csv(directory / "paired_operating_metrics.csv")
    rows = pairs[
        pairs.scope.eq("repeat_pooled") & pairs.slice.eq("snapshot_activity")
    ].copy()
    rows["passed"] = (
        rows.recall_difference.notna()
        & rows.recall_difference.ge(-0.05)
        & rows.false_alert_reduction.ge(0)
    )
    rows.to_csv(directory / "intersection_safeguards.csv", index=False)
    decision = json.loads((directory / "decision.json").read_text())
    decision["checks"]["frozen_intersection_limits"] = bool(rows.passed.all())
    decision["accepted"] = all(decision["checks"].values())
    write_json(directory / "decision.json", decision)
    return decision


def run(output):
    require(not output.exists(), "Preserve existing run")
    source = ROOT / "results/crossfit-percentile-20261017"
    verify_manifest(source)
    previous = json.loads((source / "protocol.json").read_text())
    data = read_csv(source / "development_input.csv")
    validate_development_cases(data, FEATURES)
    require(
        (len(data), int(data.churn.sum()), data.StoreID.nunique()) == (1375, 212, 422),
        "Unexpected canonical population",
    )
    histories = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(histories)
    history = read_csv(histories / "case_histories.csv")
    profiles = history[KEYS + ["n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    out = new_output_directory(output)
    protocol_path = ROOT / "comparisons/retained_calibration_protocol.md"
    shutil.copyfile(protocol_path, out / "protocol.md")
    shutil.copyfile(Path(__file__), out / "as_run_retained_calibration.py")
    data.to_csv(out / "development_input.csv", index=False)
    spec = {
        "id": CANDIDATE,
        "description": "Retained disjoint-reference logistic calibration",
        "features": FEATURES,
        "components": previous["arms"][1]["components"],
        "calibrator": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 1e-8,
            "input": "logit(clipped probability)",
            "clip": 1e-12,
        },
    }
    protocol = {
        **previous,
        "arms": [previous["arms"][0], spec],
        "interval_tail_probability": TAIL,
        "period_limits": LIMITS,
        "new_fits": 0,
        "calibrator_fits": 600,
        "alpha_spending_cycle": 1,
    }
    write_json(out / "protocol.json", protocol)
    write_json(
        out / "prefit_identity.json",
        {
            "inputs": input_identity(
                {"source": source / "manifest.json", "protocol": protocol_path}
            ),
            "implementation": input_identity(
                {
                    "runner": Path(__file__),
                    "assessment": ROOT / "comparisons/matched_evaluation.py",
                }
            ),
        },
    )
    parts, transfers = [], []
    for split in previous["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        original = source / relative
        verify_manifest(original)
        folder = new_output_directory(out / relative)
        saved_inner = read_csv(original / "inner_scores.csv")
        baseline_inner = saved_inner[saved_inner.candidate.eq("baseline")]
        assignment = baseline_inner.set_index(KEYS).inner_fold.reindex(
            pd.MultiIndex.from_frame(train[KEYS])
        )
        require(assignment.notna().all(), "Missing inner assignment")
        cached = json.loads((original / "model_identities.json").read_text())
        inners, identities = [baseline_inner], []
        for fold in sorted(assignment.unique()):
            fitting = train.iloc[np.flatnonzero(assignment.to_numpy() != fold)]
            held = train.iloc[np.flatnonzero(assignment.to_numpy() == fold)]
            records = [
                r for r in cached if r["stage"] == "inner" and r["inner_fold"] == fold
            ]
            score, fitted = mapped_components(fitting, held[KEYS], records)
            identities.extend(fitted)
            inners.append(
                held[KEYS + ["churn"]].assign(
                    score=score, candidate=CANDIDATE, inner_fold=int(fold)
                )
            )
        inner = pd.concat(inners, ignore_index=True)
        choices = [
            dict(candidate=arm, **study.select_threshold(rows.churn, rows.score))
            for arm, rows in inner.groupby("candidate")
        ]
        outer_records = [r for r in cached if r["stage"] == "outer"]
        score, fitted = mapped_components(train, valid[KEYS], outer_records)
        identities.extend(fitted)
        original_outer = read_csv(original / "predictions.csv")
        baseline = original_outer[original_outer.strategy.eq("baseline")]
        cutoff = next(r["cutoff"] for r in choices if r["candidate"] == CANDIDATE)
        outer = pd.concat(
            [
                baseline,
                valid[KEYS + ["churn", "n_orders_3m"]].assign(
                    score=score,
                    cutoff=cutoff,
                    strategy=CANDIDATE,
                    repeat=split["repeat"],
                    outer_fold=split["outer_fold"],
                ),
            ],
            ignore_index=True,
        )
        parts.append(outer)
        transfers.append(cutoff_transfer_tables(inner, outer, data))
        inner.to_csv(folder / "inner_scores.csv", index=False)
        outer.to_csv(folder / "predictions.csv", index=False)
        pd.DataFrame(choices).to_csv(folder / "inner_selections.csv", index=False)
        write_json(folder / "model_identities.json", identities)
        shutil.copyfile(
            original / "cached_hgb_identities.json",
            folder / "cached_hgb_identities.json",
        )
        shutil.copyfile(original / "split.json", folder / "split.json")
        write_manifest(
            folder, {"source": input_identity({"fold": original / "manifest.json"})}
        )
        print(
            f"Completed repeat {split['repeat']} fold {split['outer_fold']}", flush=True
        )
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    # Reuse the established complete assessment; all new calibration is already saved.
    child = out / CANDIDATE
    publish_assessment(child, out, data, profiles, predictions, protocol)
    decision = intersection_gate(child)
    pd.concat(transfers, ignore_index=True).rename(
        columns={
            "training_rows_min": "ensemble_training_pool_rows_min",
            "training_rows_max": "ensemble_training_pool_rows_max",
            "training_resellers_min": "ensemble_training_pool_resellers_min",
            "training_resellers_max": "ensemble_training_pool_resellers_max",
        }
    ).to_csv(out / "cutoff_transfer.csv", index=False)
    control = read_csv(source / "validation_predictions.csv")
    control[control.strategy.isin(["baseline", RAW])].to_csv(
        out / "raw_control.csv", index=False
    )
    all_ids = [
        r
        for split in previous["splits"]
        for r in json.loads(
            (
                out
                / f"repeat_{split['repeat']}/fold_{split['outer_fold']}/model_identities.json"
            ).read_text()
        )
    ]
    pd.DataFrame(
        [
            {
                k: r[k]
                for k in [
                    "stage",
                    "half",
                    "seed",
                    "slope",
                    "intercept",
                    "iterations",
                    "reference_rows",
                    "reference_resellers",
                    "fitting_rows",
                    "fitting_resellers",
                ]
            }
            for r in all_ids
        ]
    ).to_csv(out / "calibration_sizes.csv", index=False)
    metrics = pd.read_csv(child / "operating_metrics.csv")
    overall = metrics[metrics.scope.eq("repeat_pooled") & metrics.slice.eq("overall")]
    (out / "report.md").write_text(
        "# Retained calibration development comparison\n\nAcceptance: "
        + str(decision["accepted"])
        + ". No HGB refits; 600 logistic calibrators.\n\n"
        + markdown_table(
            overall[
                [
                    "strategy",
                    "repeat",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                    "recall",
                    "precision",
                    "alert_count",
                ]
            ]
        )
        + "\n\nComplete decisions: retained_logistic/decision.json. All folds, periods, activity groups and intersections: retained_logistic/operating_metrics.csv.\n"
        + "\nTransfer table sizes describe the ensemble pool, not individual components; calibration_sizes.csv records actual fitting/reference sizes. Conditional intervals omit retraining, adaptive research and future populations. Repeated seeds are dependent.\n"
    )
    write_json(
        child / "completion_manifest.json",
        {
            "stage": "retained_calibration_assessment",
            "decision": decision,
            "outputs": {
                p.name: input_identity({"file": p})["file"]["sha256"]
                for p in child.iterdir()
                if p.is_file()
            },
        },
    )
    write_manifest(
        out,
        {
            "stage": "retained_calibration",
            "decision": decision,
            "inputs": input_identity({"source": source / "manifest.json"}),
            "implementation": input_identity({"runner": Path(__file__)}),
        },
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
