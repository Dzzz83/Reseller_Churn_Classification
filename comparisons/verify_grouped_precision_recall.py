"""Independently audit saved grouped-study evidence and render its report."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def counts(group):
    positive = group.churn.to_numpy() == 1
    alert = group.score.to_numpy() >= group.cutoff.to_numpy()
    return tuple(
        int(values.sum())
        for values in (
            positive & alert,
            ~positive & alert,
            positive & ~alert,
            ~positive & ~alert,
        )
    )


def audit(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if protocol["test_data_used"] or protocol["temporal_validation_used"]:
        raise ValueError("This audit expects grouped development-only evaluation")
    for name, expected in manifest["outputs"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError(f"Output integrity failed: {name}")
    if (
        digest(Path(__file__).with_name("grouped_precision_recall.py"))
        != protocol["code_sha256"]
    ):
        raise ValueError("Study runner differs from executed code")
    data = load_csv(directory / "development_input.csv")
    allowed = set(protocol["snapshots"])
    if set(data.snapshot) != allowed or data.duplicated(["StoreID", "snapshot"]).any():
        raise ValueError("Unexpected development population")
    predictions = load_csv(directory / "validation_predictions.csv")
    row_keys = ["repeat", "strategy", "StoreID", "snapshot"]
    if predictions.duplicated(row_keys).any():
        raise ValueError(
            "A development row was validated more than once per strategy/repeat"
        )
    expected_keys = set(map(tuple, data[["StoreID", "snapshot"]].to_numpy()))
    for _, group in predictions.groupby(["repeat", "strategy"]):
        if set(map(tuple, group[["StoreID", "snapshot"]].to_numpy())) != expected_keys:
            raise ValueError("Validation population differs between comparisons")
        joined = group.merge(
            data[["StoreID", "snapshot", "churn"]],
            on=["StoreID", "snapshot"],
            validate="one_to_one",
            suffixes=("", "_reference"),
        )
        if not joined.churn.eq(joined.churn_reference).all():
            raise ValueError("Saved outcomes differ from approved development input")
    fit_count, audited_folds = 0, 0
    identities_seen = []
    for repeat in protocol["repeat_seeds"]:
        outer_seen = np.zeros(len(data), dtype=int)
        for fold in range(protocol["outer_folds"]):
            folder = directory / f"repeat_{repeat}" / f"fold_{fold}"
            split = json.loads((folder / "split.json").read_text())
            ti, vi = split["train_rows"], split["validation_rows"]
            train, valid = data.iloc[ti], data.iloc[vi]
            if set(ti) & set(vi) or set(ti) | set(vi) != set(range(len(data))):
                raise ValueError("Outer indices do not partition development")
            if set(train.StoreID) & set(valid.StoreID):
                raise ValueError("Outer group leakage")
            outer_seen[vi] += 1
            inner = load_csv(folder / "inner_scores.csv")
            choices = load_csv(folder / "inner_selections.csv")
            recomputed = []
            for candidate in protocol["grid"]:
                scores = inner[inner.candidate.eq(candidate["id"])]
                if set(map(tuple, scores[["StoreID", "snapshot"]].to_numpy())) != set(
                    map(tuple, train[["StoreID", "snapshot"]].to_numpy())
                ):
                    raise ValueError("Inner OOF population differs from outer training")
                if scores.duplicated(["StoreID", "snapshot"]).any():
                    raise ValueError("Duplicate inner prediction")
                if scores.groupby("StoreID").inner_fold.nunique().gt(1).any():
                    raise ValueError("Inner group leakage")
                options = []
                positive = scores.churn.eq(1).to_numpy()
                probabilities = scores.score.to_numpy()
                for cutoff in np.unique(np.r_[0.0, probabilities]):
                    alerted = probabilities >= cutoff
                    tp = int((positive & alerted).sum())
                    recall = tp / int(positive.sum())
                    precision = tp / max(1, int(alerted.sum()))
                    if recall > 0.9:
                        options.append((precision, recall, float(cutoff)))
                precision, recall, cutoff = max(options)
                recorded = choices[choices.candidate.eq(candidate["id"])].iloc[0]
                np.testing.assert_allclose(
                    [recorded.precision, recorded.recall, recorded.cutoff],
                    [precision, recall, cutoff],
                    atol=0,
                    rtol=0,
                )
                recomputed.append(
                    dict(
                        candidate=candidate["id"],
                        precision=precision,
                        recall=recall,
                        cutoff=cutoff,
                        feature_count=len(candidate["features"]),
                    )
                )
            winner = sorted(
                recomputed,
                key=lambda row: (
                    -row["precision"],
                    -row["recall"],
                    row["feature_count"],
                    row["candidate"],
                ),
            )[0]
            selected = json.loads((folder / "selected.json").read_text())
            if winner != selected:
                raise ValueError(
                    "Configuration was not selected solely from inner predictions"
                )
            outer = load_csv(folder / "predictions.csv")
            adaptive = outer[outer.strategy.eq("adaptive_selection")].sort_values(
                ["StoreID", "snapshot"]
            )
            chosen = outer[outer.strategy.eq(winner["candidate"])].sort_values(
                ["StoreID", "snapshot"]
            )
            np.testing.assert_array_equal(adaptive.score, chosen.score)
            np.testing.assert_array_equal(adaptive.cutoff, chosen.cutoff)
            if not adaptive.cutoff.eq(winner["cutoff"]).all():
                raise ValueError("Outer cutoff differs from inner selection")
            fitted = json.loads((folder / "model_identities.json").read_text())
            expected = (
                len(protocol["grid"])
                * (protocol["inner_folds"] + 1)
                * len(protocol["model_seeds"])
            )
            if len(fitted) != expected:
                raise ValueError("Unexpected estimator fit count")
            fit_count += len(fitted)
            identities_seen.extend(fitted)
            audited_folds += 1
        if not np.all(outer_seen == 1):
            raise ValueError("Outer folds do not validate every row exactly once")
    audited_metric_rows = 0
    for filename, group_keys in (
        ("fold_metrics.csv", ["repeat", "outer_fold", "strategy"]),
        ("repeat_metrics.csv", ["repeat", "strategy"]),
    ):
        reported = load_csv(directory / filename)
        for row in reported.itertuples():
            group = predictions
            for key in group_keys:
                group = group[group[key].eq(getattr(row, key))]
            if row.subgroup == "active":
                group = group[group.n_orders_3m > 0]
            if row.subgroup == "inactive":
                group = group[group.n_orders_3m == 0]
            actual = counts(group)
            if actual != (row.tp, row.fp, row.fn, row.tn) or len(group) != row.rows:
                raise ValueError("Confusion counts do not reconcile")
            tp, fp, fn, _ = actual
            np.testing.assert_allclose(
                [row.precision, row.recall],
                [tp / max(1, tp + fp), tp / max(1, tp + fn)],
                atol=1e-15,
                rtol=0,
            )
            audited_metric_rows += 1
    final_fit_count = 0
    decision = json.loads((directory / "decision.json").read_text())
    if decision["model_frozen"]:
        final_choices = load_csv(directory / "final_inner_selections.csv")
        final_inner = load_csv(directory / "final_inner_scores.csv")
        for row in final_choices.itertuples():
            group = final_inner[final_inner.candidate.eq(row.candidate)]
            if group.groupby("StoreID").inner_fold.nunique().gt(1).any():
                raise ValueError("Final selection has inner group leakage")
            if (
                set(map(tuple, group[["StoreID", "snapshot"]].to_numpy()))
                != expected_keys
            ):
                raise ValueError("Final inner population differs")
            feasible = []
            y = group.churn.eq(1).to_numpy()
            scores = group.score.to_numpy()
            for threshold in np.unique(np.r_[0.0, scores]):
                alerted = scores >= threshold
                caught = int((y & alerted).sum())
                recall = caught / int(y.sum())
                precision = caught / max(1, int(alerted.sum()))
                if recall > 0.9:
                    feasible.append((precision, recall, float(threshold)))
            np.testing.assert_allclose(
                [row.precision, row.recall, row.cutoff], max(feasible), atol=0, rtol=0
            )
        selected = final_choices.sort_values(
            ["precision", "recall", "feature_count", "candidate"],
            ascending=[False, False, True, True],
        ).iloc[0]
        frozen = json.loads((directory / "frozen_candidate.json").read_text())
        if (
            frozen["candidate"]["id"] != selected.candidate
            or frozen["cutoff"] != selected.cutoff
        ):
            raise ValueError(
                "Frozen model differs from full-development inner selection"
            )
        configured = next(
            item for item in protocol["grid"] if item["id"] == selected.candidate
        )
        if frozen["candidate"] != configured:
            raise ValueError("Frozen configuration differs from the predeclared grid")
        bundle = joblib.load(directory / "research_candidate.joblib")
        if len(bundle["models"]) != len(protocol["model_seeds"]):
            raise ValueError("Frozen ensemble seed count differs")
        # joblib.hash on an estimator object is not a serialization-stable
        # identity for its NumPy tree state. The bundle file SHA is verified
        # above; independently refit and compare predictions after reload.
        training_path = Path(protocol["inputs"]["development"])
        if digest(training_path) != protocol["inputs"]["development_sha256"]:
            raise ValueError("Final refit input checksum differs")
        training = pd.read_csv(training_path)
        training["snapshot"] = pd.to_datetime(training.snapshot)
        training = training.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)
        weights = compute_sample_weight("balanced", training.churn)
        features = configured["features"]
        training_hash = hashlib.sha256(
            pd.util.hash_pandas_object(
                training[["StoreID", "snapshot", "churn", *features]], index=False
            ).values.tobytes()
        ).hexdigest()
        bundle_checks = []
        for model, seed, identity in zip(
            bundle["models"], protocol["model_seeds"], frozen["fitted_models"]
        ):
            if training_hash != identity["training_sha256"]:
                raise ValueError("Final estimator training content differs")
            if list(model.feature_names_in_) != configured["features"]:
                raise ValueError("Frozen feature order differs")
            parameters = {**configured["parameters"], "random_state": seed}
            if any(
                model.get_params()[name] != value for name, value in parameters.items()
            ):
                raise ValueError("Frozen estimator parameters differ")
            with threadpool_limits(limits=1):
                refit = HistGradientBoostingClassifier(**parameters)
                refit.fit(
                    training[features],
                    training.churn.astype(int),
                    sample_weight=weights,
                )
                np.testing.assert_array_equal(
                    model.predict_proba(training[features]),
                    refit.predict_proba(training[features]),
                )
            bundle_checks.append(
                dict(
                    seed=seed,
                    training_content_verified=True,
                    all_development_predictions_exact_after_independent_refit=True,
                    reloaded_object_hash=joblib.hash(model),
                    original_in_memory_hash=identity["estimator_hash"],
                )
            )
        write_bundle = dict(
            bundle_file_sha256=digest(directory / "research_candidate.joblib"),
            checks=bundle_checks,
            note="In-memory joblib object hashes are not assumed stable across serialization; file SHA and exact deterministic refit predictions verify the bundle",
        )
        (directory / "bundle_verification.json").write_text(
            json.dumps(write_bundle, indent=2) + "\n"
        )
        final_identities = json.loads(
            (directory / "final_training_identities.json").read_text()
        )
        final_fit_count = len(final_identities)
        if final_fit_count != (
            len(protocol["grid"]) * protocol["inner_folds"] + 1
        ) * len(protocol["model_seeds"]):
            raise ValueError("Final inner-selection/refit count differs")
    result = dict(
        status="passed",
        outer_folds=audited_folds,
        outer_estimator_fits=fit_count,
        final_selection_and_refit_estimator_fits=final_fit_count,
        frozen_bundle_reloaded_and_verified=decision["model_frozen"],
        bundle_fidelity_refits=len(protocol["model_seeds"])
        if decision["model_frozen"]
        else 0,
        metric_rows=audited_metric_rows,
        all_output_hashes_verified=True,
        all_outer_and_inner_groups_disjoint=True,
        inner_cutoffs_and_winners_independently_recomputed=True,
        all_populations_matched=True,
        test_data_used=False,
        max_within_ensemble_score_range=max(
            item["seed_score_range"] for item in identities_seen
        ),
    )
    write_report(directory, protocol, predictions, result)
    (directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "report_manifest.json").write_text(
        json.dumps(
            dict(
                verifier_sha256=digest(Path(__file__)),
                inputs=dict(manifest=digest(directory / "manifest.json")),
                outputs={
                    name: digest(directory / name)
                    for name in (
                        "report.md",
                        "verification.json",
                        "paired_changes.csv",
                        "bundle_verification.json",
                    )
                    if (directory / name).exists()
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(result, indent=2))


def write_report(directory, protocol, predictions, verification):
    metrics = load_csv(directory / "repeat_metrics.csv")
    overall = metrics[metrics.subgroup.eq("overall")]
    decision = json.loads((directory / "decision.json").read_text())
    uncertainty = json.loads((directory / "bootstrap.json").read_text())
    changes = []
    for repeat in protocol["repeat_seeds"]:
        keys = ["StoreID", "snapshot", "churn"]
        part = predictions[predictions.repeat.eq(repeat)]
        before = part[part.strategy.eq("original_baseline")]
        for strategy in ("adaptive_selection", "full_baseline"):
            after = part[part.strategy.eq(strategy)]
            if after.empty:
                continue
            paired = before[keys + ["score", "cutoff"]].merge(
                after[keys + ["score", "cutoff"]],
                on=keys,
                validate="one_to_one",
                suffixes=("_before", "_after"),
            )
            positive = paired.churn.eq(1)
            old = paired.score_before.ge(paired.cutoff_before)
            new = paired.score_after.ge(paired.cutoff_after)
            changes.append(
                dict(
                    repeat=repeat,
                    strategy=strategy,
                    newly_caught=int((positive & ~old & new).sum()),
                    newly_missed=int((positive & old & ~new).sum()),
                    false_alerts_removed=int((~positive & old & ~new).sum()),
                    false_alerts_added=int((~positive & ~old & new).sum()),
                )
            )
    pd.DataFrame(changes).to_csv(directory / "paired_changes.csv", index=False)
    table = [
        "| Repeat | Procedure | Recall | Precision | Caught | Missed | False alerts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (
        overall[
            overall.strategy.isin(
                ["original_baseline", "full_baseline", "adaptive_selection"]
            )
        ]
        .sort_values(["repeat", "strategy"])
        .itertuples()
    ):
        table.append(
            f"| {row.repeat} | {row.strategy} | {row.recall:.2%} | {row.precision:.2%} | {row.tp} | {row.fn} | {row.fp} |"
        )
    exploratory = [
        "| Configuration | Mean repeat recall | Mean repeat precision | Repeats above 90% recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for strategy, group in overall.groupby("strategy"):
        exploratory.append(
            f"| {strategy} | {group.recall.mean():.2%} | {group.precision.mean():.2%} | {int(group.recall.gt(0.9).sum())}/{len(group)} |"
        )
    subgroup = [
        "| Procedure | Subgroup | Mean repeat recall | Mean repeat precision |",
        "| --- | --- | ---: | ---: |",
    ]
    for (strategy, name), group in metrics[
        metrics.strategy.isin(["original_baseline", "adaptive_selection"])
        & metrics.subgroup.ne("overall")
    ].groupby(["strategy", "subgroup"]):
        subgroup.append(
            f"| {strategy} | {name} | {group.recall.mean():.2%} | {group.precision.mean():.2%} |"
        )
    selected = []
    for path in sorted(directory.glob("repeat_*/fold_*/selected.json")):
        selected.append(json.loads(path.read_text())["candidate"])
    frequencies = pd.Series(selected).value_counts().to_dict()
    fold_metrics = load_csv(directory / "fold_metrics.csv")
    adaptive_folds = fold_metrics[
        fold_metrics.strategy.eq("adaptive_selection")
        & fold_metrics.subgroup.eq("overall")
    ]
    weakest_repeat = overall[overall.strategy.eq("adaptive_selection")].recall.min()
    final_specification = "No final candidate was frozen."
    if decision["model_frozen"]:
        frozen = json.loads((directory / "frozen_candidate.json").read_text())
        final_specification = f"Full-development inner selection chose `{frozen['candidate']['id']}` with features `{', '.join(frozen['candidate']['features'])}` and cutoff `{frozen['cutoff']}`. This full-data refit was selected by the evaluated procedure; the table is not a separate validation of this single fixed model."
    low, high = uncertainty["precision_delta_95"]
    conclusion = (
        "The predeclared adaptive selection procedure meets the repeat-level recall target and has a positive paired bootstrap precision interval. A research candidate was frozen using development only; it is not an independently tested or deployed replacement."
        if decision["strong_support"]
        else "The predeclared evidence criteria do not support a reliable replacement. No final candidate was frozen and no test evaluation was run. Exploratory configurations below must not be promoted using these same outer scores as an independent performance estimate."
    )
    doc = f"""# Repeated nested grouped precision study

{conclusion}

## Protocol

The target is maximum precision while recall stays strictly above 90%.
This run uses {protocol["rows"]} approved development account/snapshot rows,
{protocol["resellers"]} resellers, and {protocol["churners"]} churn cases.
Only July/October 2012 and January/April 2013 are allowed. Final testing data
is not an input, no test features or outcomes are opened, and no temporal
validation is performed. Earlier inspection of this development dataset makes
the study exploratory rather than a new independent final evaluation.

The predefined split seeds {protocol["repeat_seeds"]} each make
{protocol["outer_folds"]} outer grouped folds. Within each outer training set,
{protocol["inner_folds"]} inner grouped folds use seed 78.
All snapshots of one reseller stay in the same fold at both levels. All
candidates share identical splits and populations. Class weights use the
current fit's training labels only. Missing discounts and undefined ratios
remain native missing values. No preprocessing is learned on outer rows.

Eight configurations cross four feature sets (original ten; plus spending
ratios; plus cadence; plus all six additions) with two settings (original
boosting parameters; or seven leaves and L2=1). Five model seeds and all other
model parameters are preserved. Inner OOF predictions select the cutoff that
maximizes precision subject to pooled inner recall >90%; ties favor recall
then higher cutoff. Configuration ties favor recall, fewer features, then
stable configuration ID. The complete grid, inputs, code hash, split seeds,
and decision rules were saved to `protocol.json` before any fitting.

The primary comparison is **adaptive_selection** (configuration and cutoff
chosen inside training) against **original_baseline** (original ten-feature
model, cutoff chosen by the same inner rule). **full_baseline** is the earlier
six-feature addition with original parameters. **original_fixed_006** is a
reference only. Each outer fold freezes its own configuration/cutoff before
scoring; pooled results do not describe one deployment cutoff.

## Held-out grouped validation results

{chr(10).join(table)}

Primary mean precision gain: **{decision["mean_precision_gain"] * 100:+.2f} percentage points**.
Paired reseller-cluster 95% percentile interval: **{low * 100:+.2f} to {high * 100:+.2f} points**.
All repetitions above 90% recall: **{decision["all_repetitions_above_90_recall"]}**.
Predeclared eligibility: **{decision["eligible"]}**.
Strong-support rule (eligible plus positive lower precision bound): **{decision["strong_support"]}**.

The {protocol["bootstrap_draws"]:,} bootstrap draws resample whole resellers, carrying every snapshot
and repeat prediction together. This interval conditions on the already
fitted models and selected cutoffs; it does not include retraining/reselection
uncertainty. Repetitions overlap and are dependent. It is a marginal interval
for the single predeclared precision comparison, not simultaneous evidence
for every exploratory configuration.

## Exploratory configuration summaries

{chr(10).join(exploratory)}

These summaries are diagnostic. Selecting a fixed winner after inspecting
their outer scores introduces additional development selection. The primary
adaptive procedure already evaluates model selection inside training folds.
Do not sum the three repetitions as independent accounts or compute a larger
sample size from repeated predictions.

Adaptive configuration frequencies across outer folds: `{json.dumps(frequencies, sort_keys=True)}`.
Paired gained/lost churners and removed/added false alerts are in
`paired_changes.csv`; fold-specific outcomes and subgroup metrics are in
`fold_metrics.csv` and `repeat_metrics.csv`.

{chr(10).join(subgroup)}

## Final state and limitations

Research candidate frozen: **{decision["model_frozen"]}**.
Final testing evaluated: **False**. Operational model changed: **False**.
Main branch changed: **False**. No push or merge is performed.

{final_specification}

The >90% recall rule is pooled per repetition. Only
{int(adaptive_folds.recall.gt(0.9).sum())}/{len(adaptive_folds)} individual outer
folds exceed it; minimum fold recall is {adaptive_folds.recall.min():.2%}.
The weakest repeat has {weakest_repeat:.2%} recall. This must not be presented
as an every-fold or future guarantee.

The study answers generalization to held-out reseller identities. Source
availability, source completeness, small development population, repeated
snapshot dependence, and prior exploratory inspection remain limitations.
Outer recall below the target is a failure of the operating objective even
when inner recall exceeded 90%. A training constraint is not an outer
validation guarantee. No new independent final performance is claimed.

## Verification

Independent saved-output checks verified all file hashes, every population
join, outer and inner reseller separation, inner cutoff optimization and
configuration selection, adaptive-to-selected prediction identity, and
{verification["metric_rows"]} metric rows. All {verification["outer_estimator_fits"]}
outer-study estimator fits completed. Final full-development selection/refitting
adds {verification["final_selection_and_refit_estimator_fits"]} study fits;
{verification["bundle_fidelity_refits"]} additional deterministic refits check
saved-bundle fidelity without estimating quality. Maximum observed seed score range is
{verification["max_within_ensemble_score_range"]}; identical seed predictions
are not independent robustness confirmations.

Eight focused synthetic tests and a standalone synthetic end-to-end smoke
passed before this study; all 14 study/diagnosis tests passed afterward.
During execution, the user requested repair of the duplicate TOML sections.
Those sections and the package initializer's committed conflict markers were
fixed on the research branch. Full pytest now gets further but is blocked by
a separate module/package collision: the research `reseller_churn.config`
package shadows the merged final pipeline's `config.py`, so its CLI cannot
import `TRAINING_SNAPSHOTS`. The standalone study does not import that package.
Synthetic smoke results are execution evidence, not model-quality evidence.

Input/code/runtime specifications, fold assignments, inner and outer saved
predictions, model hashes and training-content identities are retained in
this ignored run directory. Verification is reproducible using:

```powershell
.venv/Scripts/python.exe comparisons/verify_grouped_precision_recall.py {directory.as_posix()}
```
"""
    (directory / "report.md").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory)
