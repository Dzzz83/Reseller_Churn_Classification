"""Readable complete summaries of immutable saved breadth-study evidence."""

import argparse
import json
from pathlib import Path

import pandas as pd

from reseller_churn.artifacts import file_sha256, verify_manifest, write_json


def markdown_table(frame):
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                f"{value:.4f}" if isinstance(value, float) else str(value)
                for value in row
            )
            + " |"
        )
    return "\n".join(rows)


def load_decision(directory):
    """Honor a linked confirmation without rescuing failed primary evidence."""
    decision = json.loads((directory / "decision.json").read_text())
    confirmation = directory / "final_decision.json"
    if confirmation.exists():
        final = json.loads(confirmation.read_text())
        if final["primary_decision_sha256"] != file_sha256(directory / "decision.json"):
            raise ValueError("Confirmation refers to a different primary decision")
        if final["decision"]["accepted"] and not decision["accepted"]:
            raise ValueError("Confirmation cannot rescue a failed primary decision")
        decision = final["decision"]
    return decision


def report(directory):
    verify_manifest(directory)
    destination = directory / "report.md"
    if destination.exists():
        raise FileExistsError("Preserve the existing report")
    metrics = pd.read_csv(directory / "metrics.csv")
    decision = load_decision(directory)
    fidelity = json.loads((directory / "baseline_fidelity.json").read_text())
    verification = json.loads((directory / "independent_verification.json").read_text())
    for identity in verification.get("derived_outputs", {}).values():
        if file_sha256(Path(identity["path"])) != identity["sha256"]:
            raise ValueError("Derived verification output changed before reporting")
    protocol = json.loads((directory / "protocol.json").read_text())
    feature = protocol.get("description", protocol["arms"][1]["features"][-1])
    overall = metrics[(metrics.scope == "repeat_pooled") & (metrics.slice == "overall")]
    columns = [
        "policy",
        "repeat",
        "tp",
        "fp",
        "fn",
        "tn",
        "recall",
        "precision",
        "false_positive_rate",
    ]
    subgroup = metrics[
        (metrics.scope == "repeat_pooled")
        & metrics.slice.isin(["activity", "snapshot_slice"])
    ]
    subgroup = (
        subgroup.groupby(["policy", "slice", "group"])[
            [
                "rows",
                "churners",
                "tp",
                "fp",
                "recall",
                "precision",
                "false_positive_rate",
            ]
        ]
        .mean()
        .reset_index()
    )
    ranking = metrics[
        (metrics.scope == "outer_fold") & metrics.slice.isin(["overall", "activity"])
    ]
    ranking = (
        ranking.groupby(["policy", "slice", "group"])[["average_precision", "roc_auc"]]
        .mean()
        .reset_index()
    )
    burden = pd.read_csv(directory / "high_recall_burden.csv")
    burden = (
        burden[burden.scope.eq("outer_fold")]
        .groupby(["policy", "subgroup", "target_recall"])[
            ["fp", "precision", "achieved_recall"]
        ]
        .mean()
        .reset_index()
    )
    cases = pd.read_csv(directory / "paired_cases.csv")
    changes = []
    for repeat, frame in cases.groupby("repeat"):
        gained = ~frame.alert_baseline & frame.alert_candidate
        lost = frame.alert_baseline & ~frame.alert_candidate
        changes.append(
            {
                "repeat": repeat,
                "gained_churn": int((gained & frame.churn.eq(1)).sum()),
                "lost_churn": int((lost & frame.churn.eq(1)).sum()),
                "added_fp": int((gained & frame.churn.eq(0)).sum()),
                "removed_fp": int((lost & frame.churn.eq(0)).sum()),
            }
        )
    cutoffs, seed_ranges = [], []
    for path in sorted(directory.glob("repeat_*/fold_*/inner_selections.csv")):
        cutoffs.append(pd.read_csv(path))
        identity_path = path.parent / "model_identities.json"
        if not identity_path.exists():
            identity_path = path.parent / "component_model_identities.json"
        for identity in json.loads(identity_path.read_text()):
            seed_ranges.append(identity["seed_score_range"])
    cutoffs = (
        pd.concat(cutoffs)
        .groupby("candidate")
        .cutoff.agg(["min", "median", "max"])
        .reset_index()
    )
    tail = decision.get("interval_tail_probability", 0.025)
    draws = decision.get("bootstrap_draws", 5000)
    lines = [
        f"# Development trial: {feature}",
        "",
        "Exploratory nested grouped development reuse: 1,375 account/snapshot cases, 212 churners, 422 resellers; three dependent repetitions of five grouped folds. Both arms use identical natural populations and cutoff-selection objectives, with training-only component fitting/preprocessing. Each procedure is frozen in protocol.json. No October outcomes, final evaluation, full-data model, deployment or cutoff retuning.",
        "",
        f"**Acceptance: {'met' if decision['accepted'] else 'not met'}.** This decision includes every frozen operational, practical, uncertainty and subgroup requirement. Five-percent FP and five-point subgroup margins are research defaults, not supplied business requirements.",
        "",
        "## Paired operating results",
        "",
        markdown_table(overall[columns]),
        "",
        f"Mean FP reduction (positive is better): {decision['mean_fp_reduction']:.2f} ({decision['mean_fp_reduction_fraction']:.2%}).",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {"requirement": key, "passed": value}
                    for key, value in decision["checks"].items()
                ]
            )
        ),
        "",
        f"Conditional paired reseller-cluster intervals ({100 * (1 - 2 * tail):.1f}% central; {draws:,} draws, seed 2026): recall difference {decision['recall_difference_interval']}; FP reduction {decision['fp_reduction_interval']}. Reseller snapshots and repeats move together. These intervals omit retraining, reselection, adaptive research and population shift; they are not independent confirmation.",
        "",
        "## Activity and snapshot reporting",
        "",
        "Arithmetic means of repeat-level metrics. Counts are mean counts per repeat, not additional independent cases; snapshot slices are not temporal evaluation.",
        "",
        markdown_table(subgroup),
        "",
        "## Ranking near high recall",
        "",
        "Mean within-fold metrics (folds equally weighted, dependent):",
        "",
        markdown_table(ranking),
        "",
        "Mean within-fold FP burden and precision at outcome-derived recall points, preserving ties. Descriptive only; these are never selected operational cutoffs. Pooled mixed-model curves are separately labelled in ranking_curves.csv.",
        "",
        markdown_table(burden),
        "",
        "## Changed cases and cutoff sensitivity",
        "",
        markdown_table(pd.DataFrame(changes)),
        "",
        "Inner-training-selected outer-fold cutoff distributions:",
        "",
        markdown_table(cutoffs),
        "",
        f"Largest seed-score range: {max(seed_ranges):.8g}. Identical seed predictions do not supply independent robustness evidence. Baseline fidelity: {fidelity}.",
        "",
        "## Verification and full evidence",
        "",
        f"Read-only independent audit: {verification}.",
        "",
        "All fold/repeat and annual-count/span/discount-missingness metrics, sample sizes and small-group flags are retained in metrics.csv. See paired_cases.csv, inner_scores.csv/inner_selections.csv within every fold, model_identities.json, high_recall_burden.csv, ranking_curves.csv, bootstrap.csv, decision.json and manifests. No adverse fold or subgroup is omitted from those outputs.",
        "",
        "Associations from matched histories did not by themselves justify success. Scores, AP or descriptive curve improvements do not replace the frozen same-split recall/false-alert decision. This study cannot certify future-period performance or that transaction histories contain no other useful signal.",
    ]
    secondary = directory / "cadence_reference_comparison.csv"
    if secondary.exists():
        lines.extend(
            [
                "",
                "## Secondary procedure references",
                "",
                "These references describe incremental changes. They do not replace the original baseline in the acceptance decision. Cutoffs remain the saved training-only choices.",
                "",
                markdown_table(pd.read_csv(secondary)),
            ]
        )
    projection_path = directory / "projection_summary.csv"
    if projection_path.exists():
        projection = pd.read_csv(projection_path).drop_duplicates(
            ["fold", "stage", "inner_fold"]
        )
        projection_summary = (
            projection.groupby("stage")
            .agg(
                vocabulary_min=("vocabulary_size", "min"),
                vocabulary_mean=("vocabulary_size", "mean"),
                vocabulary_max=("vocabulary_size", "max"),
                variance_min=("explained_variance_fraction", "min"),
                variance_mean=("explained_variance_fraction", "mean"),
                variance_max=("explained_variance_fraction", "max"),
            )
            .reset_index()
        )
        lines.extend(
            [
                "",
                "## Training purchase-mix projection",
                "",
                "Five components are fitted on annual training shares only, then applied to annual/latest profiles. Captured training variance is descriptive representation fidelity, not predictive validation. Unseen product mass stays explicit.",
                "",
                markdown_table(projection_summary),
                "",
                "The root protocol enumerates PCs before unseen shares; actual model column order interleaves unseen by scope and is saved in each training identity. The verifier checks the same feature set and reconstructs hashes in that recorded order.",
            ]
        )
    with destination.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    write_json(
        directory / "report_identity.json",
        {
            "report_sha256": file_sha256(destination),
            "study_manifest_sha256": file_sha256(directory / "manifest.json"),
            "reporter_sha256": file_sha256(Path(__file__)),
            "verification_sha256": file_sha256(
                directory / "independent_verification.json"
            ),
        },
    )
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    report(parser.parse_args().directory)


if __name__ == "__main__":
    main()
