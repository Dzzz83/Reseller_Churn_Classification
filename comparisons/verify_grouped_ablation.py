"""Audit ablation predictions and report contextual component contributions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from comparisons import grouped_ablation as ablation
from comparisons import grouped_precision_recall as study


def verify_choices(inner, choices, grid):
    for candidate in grid:
        saved = inner[inner.candidate.eq(candidate["id"])]
        if (
            saved.groupby("StoreID").inner_fold.nunique().gt(1).any()
            or saved.duplicated(study.KEYS).any()
        ):
            raise ValueError("Inner grouping or prediction uniqueness failed")
        positive = saved.churn.eq(1).to_numpy()
        scores = saved.score.to_numpy()
        options = []
        for cutoff in np.unique(np.r_[0.0, scores]):
            alert = scores >= cutoff
            caught = int((positive & alert).sum())
            recall = caught / int(positive.sum())
            precision = caught / max(1, int(alert.sum()))
            if recall > 0.9:
                options.append((precision, recall, float(cutoff)))
        chosen = choices[choices.candidate.eq(candidate["id"])].iloc[0]
        np.testing.assert_allclose(
            [chosen.precision, chosen.recall, chosen.cutoff],
            max(options),
            atol=0,
            rtol=0,
        )


def audit(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        protocol["test_data_used"]
        or protocol["temporal_validation_used"]
        or protocol["model_changed"]
    ):
        raise ValueError("Expected a development-only grouped ablation")
    if (
        study.file_hash(Path(ablation.__file__)) != protocol["code_sha256"]
        or study.file_hash(Path(study.__file__)) != protocol["shared_code_sha256"]
    ):
        raise ValueError("Executed source changed")
    for name, expected in manifest["outputs"].items():
        path = (directory / name).resolve()
        if (
            not path.is_relative_to(directory.resolve())
            or study.file_hash(path) != expected
        ):
            raise ValueError(f"Output checksum failed: {name}")
    predictions = study.read_csv(directory / "validation_predictions.csv")
    data = study.read_csv(directory / "development_input.csv")
    data.snapshot = pd.to_datetime(data.snapshot)
    ablation.validate_predictions(predictions, data, protocol["repeat_seeds"])
    reference = data[study.KEYS + ["churn"]].copy()
    reference.snapshot = reference.snapshot.dt.strftime("%Y-%m-%d")
    joined = predictions.merge(
        reference, on=study.KEYS, validate="many_to_one", suffixes=("", "_reference")
    )
    if not joined.churn.eq(joined.churn_reference).all():
        raise ValueError("Validation labels differ from development")
    folds_checked = 0
    for repeat in protocol["repeat_seeds"]:
        seen = np.zeros(len(data), dtype=int)
        for fold in range(protocol["outer_folds"]):
            folder = directory / f"repeat_{repeat}" / f"fold_{fold}"
            split = json.loads((folder / "split.json").read_text())
            ti, vi = split["train_rows"], split["validation_rows"]
            if set(ti) & set(vi) or set(ti) | set(vi) != set(range(len(data))):
                raise ValueError("Outer row partition failed")
            if set(data.iloc[ti].StoreID) & set(data.iloc[vi].StoreID):
                raise ValueError("Outer group leakage")
            seen[vi] += 1
            inner = study.read_csv(folder / "inner_scores.csv")
            choices = study.read_csv(folder / "inner_selections.csv")
            verify_choices(inner, choices, protocol["grid"])
            expected = set(map(tuple, reference.iloc[ti][study.KEYS].to_numpy()))
            for candidate in protocol["grid"]:
                saved = inner[inner.candidate.eq(candidate["id"])]
                if set(map(tuple, saved[study.KEYS].to_numpy())) != expected:
                    raise ValueError("Inner population differs from training")
                matched = saved.merge(
                    reference.iloc[ti],
                    on=study.KEYS,
                    validate="one_to_one",
                    suffixes=("", "_reference"),
                )
                if not matched.churn.eq(matched.churn_reference).all():
                    raise ValueError("Inner labels differ from outer-training labels")
                if set(saved.inner_fold) != set(range(protocol["inner_folds"])):
                    raise ValueError("Unexpected inner fold assignments")
                part = predictions[
                    predictions.repeat.eq(repeat)
                    & predictions.outer_fold.eq(fold)
                    & predictions.strategy.eq(candidate["id"])
                ]
                cutoff = choices[choices.candidate.eq(candidate["id"])].cutoff.item()
                if not part.cutoff.eq(cutoff).all():
                    raise ValueError("Outer decisions do not use inner-selected cutoff")
            source = (
                (
                    directory / "synthetic_source"
                    if protocol["smoke"]
                    else Path(protocol["source_directory"])
                )
                / f"repeat_{repeat}"
                / f"fold_{fold}"
            )
            source_predictions = study.read_csv(source / "predictions.csv")
            for mask, name in ablation.REUSE.items():
                old = source_predictions[
                    source_predictions.strategy.eq(name)
                ].sort_values(study.KEYS)
                new = predictions[
                    predictions.repeat.eq(repeat)
                    & predictions.outer_fold.eq(fold)
                    & predictions.strategy.eq(f"f{mask:02d}")
                ].sort_values(study.KEYS)
                np.testing.assert_array_equal(old.score, new.score)
                np.testing.assert_array_equal(old.cutoff, new.cutoff)
            identities = json.loads((folder / "model_identities.json").read_text())
            expected_new = (
                10 * (protocol["inner_folds"] + 1) * len(protocol["model_seeds"])
            )
            expected_reused = (
                4 * (protocol["inner_folds"] + 1) * len(protocol["model_seeds"])
            )
            if (
                sum(not item["reused"] for item in identities) != expected_new
                or sum(item["reused"] for item in identities) != expected_reused
            ):
                raise ValueError("Fit/reuse accounting failed")
            folds_checked += 1
        if not np.all(seen == 1):
            raise ValueError("Each row must be held out once per repetition")
    metric_rows = 0
    for filename, keys in (
        ("fold_metrics.csv", ["repeat", "outer_fold", "strategy"]),
        ("repeat_metrics.csv", ["repeat", "strategy"]),
    ):
        metrics = study.read_csv(directory / filename)
        for row in metrics.itertuples():
            part = predictions
            for key in keys:
                part = part[part[key].eq(getattr(row, key))]
            if row.subgroup == "active":
                part = part[part.n_orders_3m > 0]
            if row.subgroup == "inactive":
                part = part[part.n_orders_3m == 0]
            y = part.churn.eq(1).to_numpy()
            alert = part.score.ge(part.cutoff).to_numpy()
            tp, fp, fn, tn = [
                int(v.sum()) for v in (y & alert, ~y & alert, y & ~alert, ~y & ~alert)
            ]
            if (tp, fp, fn, tn) != (row.tp, row.fp, row.fn, row.tn):
                raise ValueError("Confusion counts differ")
            np.testing.assert_allclose(
                [row.precision, row.recall],
                [tp / max(1, tp + fp), tp / max(1, tp + fn)],
                atol=1e-15,
                rtol=0,
            )
            if y.any():
                np.testing.assert_allclose(
                    row.average_precision,
                    average_precision_score(y, part.score),
                    atol=1e-15,
                    rtol=0,
                )
            metric_rows += 1
    summary = study.read_csv(directory / "configuration_summary.csv").set_index(
        "configuration"
    )
    contributions = study.read_csv(directory / "contributions.csv").set_index(
        "contrast"
    )
    for definition in protocol["contrasts"]:
        row = contributions.loc[definition["id"]]
        for metric in ("precision", "recall"):
            expected = sum(
                summary.loc[arm, metric] * weight
                for arm, weight in definition["weights"].items()
            )
            np.testing.assert_allclose(
                row[metric + "_delta"], expected, atol=1e-15, rtol=0
            )
        if (
            row.precision_family_low > row.precision_marginal_low
            or row.precision_family_high < row.precision_marginal_high
        ):
            raise ValueError(
                "Family adjustment interval is narrower than marginal interval"
            )
    result = dict(
        status="passed",
        outer_folds=folds_checked,
        metric_rows=metric_rows,
        inner_cutoffs_recomputed=True,
        reused_predictions_exact=True,
        all_populations_matched=True,
        new_fits=manifest["new_fits"],
        reused_fits=manifest["reused_fits"],
        test_data_used=False,
        model_changed=False,
    )
    render_report(directory, protocol, predictions, summary, contributions, result)
    study.write_json(directory / "verification.json", result)
    names = [
        "report.md",
        "verification.json",
        "component_effects.csv",
        "cutoff_effect.csv",
        "paired_changes.csv",
    ]
    study.write_json(
        directory / "report_manifest.json",
        dict(
            verifier_sha256=study.file_hash(__file__),
            source_manifest_sha256=study.file_hash(directory / "manifest.json"),
            outputs={name: study.file_hash(directory / name) for name in names},
        ),
    )
    print(json.dumps(result, indent=2))


def render_report(
    directory, protocol, predictions, summary, contributions, verification
):
    # Filled from completed predictions, never by looking up final-test outcomes.
    definitions = {item["id"]: item for item in protocol["contrasts"]}
    effects = []
    for name, row in contributions.iterrows():
        ap = sum(
            summary.loc[arm, "average_precision"] * weight
            for arm, weight in definitions[name]["weights"].items()
        )
        effects.append(
            dict(
                contrast=name,
                kind=row.kind,
                precision_delta=row.precision_delta,
                recall_delta=row.recall_delta,
                average_precision_delta=ap,
                precision_family_low=row.precision_family_low,
                precision_family_high=row.precision_family_high,
            )
        )
    effects = pd.DataFrame(effects)
    effects.to_csv(directory / "component_effects.csv", index=False)
    pairs = []
    for repeat in protocol["repeat_seeds"]:
        before = predictions[
            predictions.repeat.eq(repeat) & predictions.strategy.eq("f00")
        ]
        keys = study.KEYS + ["churn"]
        for arm in summary.index:
            after = predictions[
                predictions.repeat.eq(repeat) & predictions.strategy.eq(arm)
            ]
            joined = before[keys + ["score", "cutoff"]].merge(
                after[keys + ["score", "cutoff"]],
                on=keys,
                validate="one_to_one",
                suffixes=("_before", "_after"),
            )
            y = joined.churn.eq(1)
            old = joined.score_before.ge(joined.cutoff_before)
            new = joined.score_after.ge(joined.cutoff_after)
            pairs.append(
                dict(
                    repeat=repeat,
                    configuration=arm,
                    newly_caught=int((y & ~old & new).sum()),
                    newly_missed=int((y & old & ~new).sum()),
                    false_alerts_removed=int((~y & old & ~new).sum()),
                    false_alerts_added=int((~y & ~old & new).sum()),
                )
            )
    pd.DataFrame(pairs).to_csv(directory / "paired_changes.csv", index=False)
    cutoff_rows = []
    for repeat in protocol["repeat_seeds"]:
        baseline = predictions[
            predictions.repeat.eq(repeat) & predictions.strategy.eq("f00")
        ]
        fixed = study.decision_metrics(baseline.churn, baseline.score, 0.06)
        selected = study.decision_metrics(
            baseline.churn, baseline.score, baseline.cutoff
        )
        cutoff_rows.append(
            dict(
                repeat=repeat,
                fixed_recall=fixed["recall"],
                selected_recall=selected["recall"],
                fixed_precision=fixed["precision"],
                selected_precision=selected["precision"],
                average_precision_delta=0.0,
            )
        )
    cutoff_frame = pd.DataFrame(cutoff_rows)
    cutoff_frame.to_csv(directory / "cutoff_effect.csv", index=False)
    table = [
        "| Configuration | Changes | Recall | Precision | AP | Repeats >90% recall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, row in summary.iterrows():
        table.append(
            f"| {arm} | {row.changes} | {row.recall:.2%} | {row.precision:.2%} | {row.average_precision:.4f} | {int(row.repeats_above_90)}/{len(protocol['repeat_seeds'])} |"
        )
    effect_table = [
        "| Contrast: positive means component helps | Precision difference (points) | Adjusted interval (points) | Recall difference (points) | AP difference |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for row in effects.itertuples():
        effect_table.append(
            f"| {row.contrast} | {row.precision_delta * 100:+.2f} | [{row.precision_family_low * 100:+.2f}, {row.precision_family_high * 100:+.2f}] | {row.recall_delta * 100:+.2f} | {row.average_precision_delta:+.4f} |"
        )
    positive = contributions[contributions.precision_family_low > 0].index.tolist()
    negative = contributions[contributions.precision_family_high < 0].index.tolist()
    consistent_components = [
        name
        for name in ablation.FACTORS
        if contributions.loc["add_" + name, "precision_family_low"] > 0
        and contributions.loc["remove_" + name, "precision_family_low"] > 0
    ]
    component_table = [
        "| Component | Precision effect added alone (points) | Precision effect included in full context (points) | Recall effect included in full context (points) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ablation.FACTORS:
        added = contributions.loc["add_" + name]
        included = contributions.loc["remove_" + name]
        component_table.append(
            f"| {name} | {added.precision_delta * 100:+.2f} | {included.precision_delta * 100:+.2f} | {included.recall_delta * 100:+.2f} |"
        )
    supported_singles = []
    for index, name in enumerate(ablation.FACTORS):
        arm = f"f{1 << index:02d}"
        if (
            summary.loc[arm, "repeats_above_90"] == len(protocol["repeat_seeds"])
            and contributions.loc["add_" + name, "precision_family_low"] > 0
        ):
            supported_singles.append(name)
    doc = f"""# Grouped-validation component ablation

This is a controlled exploratory follow-up on development data. It separates
individual additions, removal from the complete candidate, and selected
interactions. It does not assign causal or uniquely additive percentages of
the previous adaptive procedure's gain to individual components.

## Design and data boundary

Fourteen predefined configurations cover five binary changes: overdue ratio,
annual order count, purchase-gap variation, seven tree leaves instead of 31,
and L2=1 instead of zero. They include baseline, each single addition, the
complete candidate, each one-component removal, cadence-only and the two
regularization changes together. The complete candidate here is the previously
frozen 13-feature cadence/regularization configuration, not the earlier
16-feature spending-plus-cadence candidate.

Same approved development rows, reseller grouping, three repetition seeds,
five outer folds, three inner folds, five estimator seeds, and training-only
weights are preserved. Each configuration selects its own cutoff using inner
OOF predictions, maximizing precision subject to recall strictly >90%.
Outer validation outcomes do not tune cutoffs or fitted parameters. Thus
decision differences include the configuration's training-only cutoff
adaptation. AP is reported separately as a ranking measure; no outer cutoff
is chosen to force matched observed recall.

Four compatible configurations reuse saved predictions and model identities
after source, protocol, population, and exact split checks. Ten additional
configurations require {verification["new_fits"]} new estimator fits;
{verification["reused_fits"]} estimator fits are represented by reused evidence.
Only development artifacts are inputs. No test scoring rows, outcomes, or
evaluation are used. The operational model and frozen research candidate are
unchanged; this study recommends rather than automatically replaces anything.

## Configuration results

Each value averages repeat-level pooled outer metrics, with fold-specific
training-selected cutoffs. Do not treat repeat predictions as independent
new customers. Recall feasibility is assessed in every repetition, not just
the displayed mean. Full repeat/fold/subgroup results are retained in CSVs.

{chr(10).join(table)}

## Main findings

Components with adjusted positive precision intervals both alone and in the
full context: `{consistent_components}`. These are contextual operational
effects under the shared training-only cutoff rule, not causal purchasing
effects or uniquely additive fractions of the earlier improvement.

{chr(10).join(component_table)}

The simplest individual additions showing an adjusted precision gain and
meeting >90% recall in every repetition are `{supported_singles}`. This is a
follow-up simplification hypothesis, not proof of equivalence to the full
candidate and not a new automatically selected replacement.

For the complete candidate, the three cadence features and both regularization
changes yield {summary.loc["f31", "recall"]:.2%} recall and
{summary.loc["f31", "precision"]:.2%} precision. Baseline values are
{summary.loc["f00", "recall"]:.2%} and {summary.loc["f00", "precision"]:.2%}.
These are fixed-configuration ablations; the earlier 29.74%/91.51% result
describes adaptive configuration selection and is a different estimand.

Fewer leaves have an observed full-context recall contribution of
{contributions.loc["remove_seven_leaves", "recall_delta"] * 100:+.2f} points;
without that change the full-minus-leaves configuration has mean recall
{summary.loc["f23", "recall"]:.2%}, reaching >90% in only
{int(summary.loc["f23", "repeats_above_90"])}/{len(protocol["repeat_seeds"])}
repetitions. Precision contribution is
{contributions.loc["remove_seven_leaves", "precision_delta"] * 100:+.2f} points.
Thus recall feasibility and precision contribution must be assessed separately.
Recall differences are descriptive secondary evidence here.

Annual count and L2 should not be called useless from intervals crossing zero.
Those intervals also do not establish that removing them preserves useful
performance. None of the tested interaction precision intervals excludes zero
in the canonical run, so there is no adjusted evidence for a specific synergy.

## Contributions and interactions

"add" compares a single addition with baseline. "remove" reports complete
candidate minus candidate without that change: a positive number means the
component helps when restored to the full context. It is not the effect of
deletion with reversed sign. Bundle contrasts hold the other bundle fixed.
Interactions are difference-in-differences, not individual-feature importance.

{chr(10).join(effect_table)}

Precision intervals use {protocol["bootstrap_draws"]:,} paired whole-reseller
bootstrap draws, carrying all snapshots and repetitions together. Percentile
tails use alpha/(2*17), a Bonferroni adjustment for the 17 predefined precision
contrasts. These are approximate family-adjusted conditional intervals, not
exact finite-sample coverage guarantees. They exclude refitting/reselection
uncertainty and do not correct for earlier exploratory development inspection.
Recall intervals in CSVs are marginal 95% exploratory intervals; AP differences
are point estimates without significance claims. No claim of reliable AP or
recall improvement is made from an unadjusted interval.

Adjusted intervals entirely above zero: `{positive}`.
Adjusted intervals entirely below zero: `{negative}`.
Intervals crossing zero mean insufficient evidence in that context; they do
not prove the component has no effect. Different single-addition and removal
effects indicate context dependence. Contributions cannot be summed as unique
shares of the earlier adaptive selection procedure's improvement.

## Cutoff contribution

On unchanged baseline model scores, replacing original 0.06 with the common
inner high-recall selection rule changes mean recall from
{cutoff_frame.fixed_recall.mean():.2%} to {cutoff_frame.selected_recall.mean():.2%},
and precision from {cutoff_frame.fixed_precision.mean():.2%} to
{cutoff_frame.selected_precision.mean():.2%}. AP is unchanged because score
rankings are unchanged. This demonstrates the cost of the high-recall operating
rule, not a ranking improvement. Feature/regularization comparisons above all
use the same selection rule, so they are not merely compared with the old
0.06 operating point.

## Interpretation and verification

Use per-repeat recall feasibility together with precision, ranking AP, error
tradeoffs, and adjusted uncertainty. A component that improves precision by
sacrificing the >90% objective is not automatically useful. Retaining the
previous candidate while inspecting simpler hypotheses does not validate
those hypotheses prospectively; a new winner chosen from these outer scores
would need a separate, appropriately frozen development comparison.

Independent checks verified {verification["metric_rows"]} metric rows, all
inner cutoff choices, outer/inner reseller isolation, matched populations,
exact reuse of scores/cutoffs, new/reused fit accounting, and artifact hashes.
The source preparation and shared fitting code remain unchanged. Synthetic
tests/smoke establish execution correctness, not model quality. The full
repository suite remains blocked by the previously documented config
module/package collision; focused research tests are run independently.

No final-test evaluation, automatic model replacement, push, or merge occurred.
"""
    (directory / "report.md").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory)
