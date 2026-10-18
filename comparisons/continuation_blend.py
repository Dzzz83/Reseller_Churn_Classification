"""One frozen combination of two failed but complementary development procedures."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.breadth_reporting import markdown_table
from comparisons.development_reporting import (
    cutoff_transfer_tables,
    paired_operating_tables,
)
from comparisons.matched_evaluation import publish_assessment
from comparisons.retained_calibration import LIMITS, intersection_gate
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = study.KEYS
CANDIDATE = "complementary_mean"
CONTROL = "corrected_experts_control"
TAIL = 0.000015625


def undo_balanced_odds(scores, prevalence):
    """Known training-weight odds correction; no target labels or prior estimates."""
    q = np.asarray(scores, dtype=float)
    require(
        0 < prevalence < 1 and np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all(),
        "Invalid training prior or scores",
    )
    return q * prevalence / (q * prevalence + (1 - q) * (1 - prevalence))


def combine_rows(calibrated, experts, fitting):
    keys = KEYS + ["churn"]
    paired = calibrated[keys + ["score"]].merge(
        experts[keys + ["score"]],
        on=keys,
        suffixes=("_calibrated", "_expert"),
        validate="one_to_one",
    )
    require(len(paired) == len(calibrated) == len(experts), "Unmatched components")
    require(set(fitting.StoreID).isdisjoint(paired.StoreID), "Held label prior leakage")
    prevalence = float(fitting.churn.mean())
    paired["corrected_expert"] = undo_balanced_odds(paired.score_expert, prevalence)
    paired["blended_score"] = (paired.score_calibrated + paired.corrected_expert) / 2
    paired["fitting_prevalence"] = prevalence
    paired["fitting_rows"] = len(fitting)
    return paired


def run(output):
    require(not output.exists(), "Preserve prior run")
    calibrated_source = ROOT / "results/retained-calibration-20261018"
    expert_source = ROOT / "results/activity-experts-20261018-02"
    for source in [calibrated_source, expert_source]:
        verify_manifest(source)
    previous = json.loads((calibrated_source / "protocol.json").read_text())
    other = json.loads((expert_source / "protocol.json").read_text())
    require(previous["splits"] == other["splits"], "Different outer partitions")
    data = read_csv(calibrated_source / "development_input.csv")
    expert_data = read_csv(expert_source / "development_input.csv")
    pd.testing.assert_frame_equal(
        data[KEYS + ["churn"] + study.FEATURES],
        expert_data[KEYS + ["churn"] + study.FEATURES],
    )
    history_source = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(history_source)
    histories = read_csv(history_source / "case_histories.csv")
    profiles = histories[KEYS + ["n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    out = new_output_directory(output)
    path = ROOT / "comparisons/continuation_blend_protocol.md"
    shutil.copyfile(path, out / "protocol.md")
    shutil.copyfile(Path(__file__), out / "as_run_continuation_blend.py")
    data.to_csv(out / "development_input.csv", index=False)
    spec = dict(
        id=CANDIDATE,
        description="Fixed complementary calibrated/odds-corrected expert mean",
        features=previous["arms"][1]["features"],
        components=[dict(id="crossfit_components"), dict(id="activity_experts")],
    )
    protocol = dict(
        arms=[previous["arms"][0], spec],
        splits=previous["splits"],
        repeat_seeds=previous["repeat_seeds"],
        period_limits=LIMITS,
        interval_tail_probability=TAIL,
        alpha_spending_cycle=5,
        new_fits=0,
        control=dict(id=CONTROL, eligible_for_selection=False),
    )
    write_json(out / "protocol.json", protocol)
    write_json(
        out / "prefit_identity.json",
        dict(
            inputs=input_identity(
                {
                    "calibration": calibrated_source / "manifest.json",
                    "experts": expert_source / "manifest.json",
                    "protocol": path,
                }
            ),
            implementation=input_identity(
                {
                    "runner": Path(__file__),
                    "assessment": ROOT / "comparisons/matched_evaluation.py",
                }
            ),
        ),
    )
    predictions, transfers, assemblies = [], [], []
    for split in previous["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        left, right = calibrated_source / relative, expert_source / relative
        verify_manifest(left)
        verify_manifest(right)
        folder = new_output_directory(out / relative)
        a, b = read_csv(left / "inner_scores.csv"), read_csv(right / "inner_scores.csv")
        baseline = a[a.candidate.eq("baseline")]
        pd.testing.assert_frame_equal(
            baseline.reset_index(drop=True),
            b[b.candidate.eq("baseline")].reset_index(drop=True),
        )
        inside = [baseline]
        for fold in sorted(baseline.inner_fold.unique()):
            held = baseline[baseline.inner_fold.eq(fold)]
            fitting = train[~train.StoreID.isin(held.StoreID)]
            combined = combine_rows(
                a[a.candidate.eq("retained_logistic") & a.inner_fold.eq(fold)],
                b[b.candidate.eq("activity_experts") & b.inner_fold.eq(fold)],
                fitting,
            )
            combined["stage"] = "inner"
            combined["inner_fold"] = int(fold)
            assemblies.append(
                combined.assign(repeat=split["repeat"], outer_fold=split["outer_fold"])
            )
            for arm, column in [
                (CANDIDATE, "blended_score"),
                (CONTROL, "corrected_expert"),
            ]:
                inside.append(
                    combined[KEYS + ["churn"]].assign(
                        score=combined[column], candidate=arm, inner_fold=int(fold)
                    )
                )
        inner = pd.concat(inside, ignore_index=True)
        choices = [
            dict(candidate=arm, **study.select_threshold(part.churn, part.score))
            for arm, part in inner.groupby("candidate")
        ]
        a, b = read_csv(left / "predictions.csv"), read_csv(right / "predictions.csv")
        baseline = a[a.strategy.eq("baseline")]
        pd.testing.assert_frame_equal(
            baseline.reset_index(drop=True),
            b[b.strategy.eq("baseline")].reset_index(drop=True),
        )
        combined = combine_rows(
            a[a.strategy.eq("retained_logistic")],
            b[b.strategy.eq("activity_experts")],
            train,
        )
        combined["stage"] = "outer"
        combined["inner_fold"] = -1
        assemblies.append(
            combined.assign(repeat=split["repeat"], outer_fold=split["outer_fold"])
        )
        combined = combined.merge(
            valid[KEYS + ["n_orders_3m"]], on=KEYS, validate="one_to_one"
        )
        outside = [baseline]
        for arm, column in [
            (CANDIDATE, "blended_score"),
            (CONTROL, "corrected_expert"),
        ]:
            outside.append(
                combined[KEYS + ["churn", "n_orders_3m"]].assign(
                    score=combined[column],
                    strategy=arm,
                    cutoff=next(c["cutoff"] for c in choices if c["candidate"] == arm),
                    repeat=split["repeat"],
                    outer_fold=split["outer_fold"],
                )
            )
        outer = pd.concat(outside, ignore_index=True)
        predictions.append(outer)
        transfers.append(
            cutoff_transfer_tables(
                inner[inner.candidate.ne(CONTROL)],
                outer[outer.strategy.ne(CONTROL)],
                data,
            )
        )
        inner.to_csv(folder / "inner_scores.csv", index=False)
        outer.to_csv(folder / "predictions.csv", index=False)
        pd.DataFrame(choices).to_csv(folder / "inner_selections.csv", index=False)
        identities = json.loads(
            (left / "model_identities.json").read_text()
        ) + json.loads((right / "model_identities.json").read_text())
        write_json(folder / "model_identities.json", identities)
        write_manifest(
            folder,
            dict(
                new_fits=0,
                inputs=input_identity(
                    {
                        "calibration": left / "manifest.json",
                        "experts": right / "manifest.json",
                    }
                ),
            ),
        )
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    pd.concat(assemblies, ignore_index=True).to_csv(
        out / "score_assembly.csv", index=False
    )
    pd.concat(transfers, ignore_index=True).to_csv(
        out / "cutoff_transfer.csv", index=False
    )
    child = out / CANDIDATE
    publish_assessment(child, out, data, profiles, predictions, protocol)
    decision = intersection_gate(child)
    # Preserve the initial assessment manifest and bind its finalized decision separately.
    write_json(
        child / "completion_manifest.json",
        dict(
            outputs={
                p.name: input_identity({"file": p})["file"]["sha256"]
                for p in child.iterdir()
                if p.is_file()
            }
        ),
    )
    control = predictions[predictions.strategy.isin(["baseline", CONTROL])]
    operating, paired, worst = paired_operating_tables(control, CONTROL)
    control_dir = new_output_directory(out / CONTROL)
    operating.to_csv(control_dir / "operating_metrics.csv", index=False)
    paired.to_csv(control_dir / "paired_operating_metrics.csv", index=False)
    worst.to_csv(control_dir / "worst_slices.csv", index=False)
    rank = []
    for (arm, repeat, fold), part in control.groupby(
        ["strategy", "repeat", "outer_fold"]
    ):
        rank.append(
            dict(
                strategy=arm,
                repeat=int(repeat),
                outer_fold=int(fold),
                average_precision=float(
                    average_precision_score(part.churn, part.score)
                ),
                roc_auc=float(roc_auc_score(part.churn, part.score))
                if part.churn.nunique() == 2
                else None,
            )
        )
    pd.DataFrame(rank).to_csv(control_dir / "ranking.csv", index=False)
    write_manifest(
        control_dir,
        dict(eligible_for_selection=False, description="Mechanism control only"),
    )
    metrics = pd.read_csv(child / "operating_metrics.csv")
    rows = metrics[metrics.scope.eq("repeat_pooled") & metrics.slice.eq("overall")]
    (out / "report.md").write_text(
        "# Fixed complementary DEVELOPMENT ensemble\n\nAcceptance: "
        + str(decision["accepted"])
        + ". Zero new classifier/calibrator fits.\n\n"
        + markdown_table(
            rows[
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
        + "\n\nFull frozen gates/conditional intervals: complementary_mean/decision.json. All folds/periods/activity/intersections "
        "and undefined flags retained. Mechanism control is ineligible for selection. Known-weight normalization assumes "
        "weighted posterior scores;HGB calibration is not guaranteed. No repeat/seed is independent confirmation.\n"
    )
    write_manifest(
        out,
        dict(
            stage="fixed_complementary_ensemble",
            decision=decision,
            inputs=input_identity(
                {
                    "calibration": calibrated_source / "manifest.json",
                    "experts": expert_source / "manifest.json",
                }
            ),
        ),
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
