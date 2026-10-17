"""Matched saved-prediction assessment shared by fixed research procedures."""

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import require


def paired_bootstrap(predictions, draws=5000, seed=2026, candidate_id="basket_breadth"):
    """Carry every reseller's snapshots/repeats together, conditional on fits."""
    data = predictions.copy()
    data["tp"] = data.churn.eq(1) & data.score.ge(data.cutoff)
    data["fp"] = data.churn.eq(0) & data.score.ge(data.cutoff)
    stores = np.sort(data.StoreID.unique())
    counts = {}
    repeats = sorted(data.repeat.unique())
    for arm in ("baseline", candidate_id):
        for repeat in repeats:
            part = data[data.strategy.eq(arm) & data.repeat.eq(repeat)]
            grouped = (
                part.groupby("StoreID")[["tp", "fp", "churn"]]
                .sum()
                .reindex(stores, fill_value=0)
            )
            counts[arm, repeat] = grouped.to_numpy(dtype=float)
    weights = np.random.default_rng(seed).multinomial(
        len(stores), np.full(len(stores), 1 / len(stores)), size=draws
    )
    recall_changes, fp_reductions = [], []
    for repeat in repeats:
        baseline = weights @ counts["baseline", repeat]
        candidate = weights @ counts[candidate_id, repeat]
        require(
            np.array_equal(baseline[:, 2], candidate[:, 2]),
            "Bootstrap outcome populations differ",
        )
        recall_changes.append(
            np.divide(
                candidate[:, 0] - baseline[:, 0],
                baseline[:, 2],
                out=np.full(draws, np.nan),
                where=baseline[:, 2] > 0,
            )
        )
        fp_reductions.append(baseline[:, 1] - candidate[:, 1])
    return pd.DataFrame(
        {
            "draw": np.arange(draws),
            "recall_difference": np.mean(recall_changes, axis=0),
            "fp_reduction": np.mean(fp_reductions, axis=0),
        }
    )


def acceptance(metrics, bootstrap, tail=0.025, candidate_id="basket_breadth"):
    pooled = metrics[(metrics.scope == "repeat_pooled") & (metrics.slice == "overall")]
    paired = pooled[pooled.policy.eq("baseline")].merge(
        pooled[pooled.policy.eq(candidate_id)],
        on="repeat",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    require(len(paired) == 3, "Missing repeat for acceptance")
    fp_reduction = (paired.fp_baseline - paired.fp_candidate).mean()
    practical_fraction = fp_reduction / paired.fp_baseline.mean()
    slices = metrics[
        (metrics.scope == "repeat_pooled")
        & metrics.slice.isin(["activity", "snapshot_slice"])
    ]
    subgroup = (
        slices.groupby(["policy", "slice", "group"], observed=True)
        .recall.mean()
        .unstack("policy")
    )
    recall_lower, recall_upper = np.nanquantile(
        bootstrap.recall_difference, [tail, 1 - tail]
    )
    fp_lower, fp_upper = np.nanquantile(bootstrap.fp_reduction, [tail, 1 - tail])
    checks = {
        "recall_preserved_every_repeat": bool(
            paired.tp_candidate.ge(paired.tp_baseline).all()
        ),
        "fewer_fp_every_repeat": bool(paired.fp_candidate.lt(paired.fp_baseline).all()),
        "at_least_five_percent_mean_fp_reduction": bool(practical_fraction >= 0.05),
        "recall_uncertainty_lower_nonnegative": bool(recall_lower >= 0),
        "fp_reduction_uncertainty_lower_positive": bool(fp_lower > 0),
        "activity_snapshot_mean_recall_loss_at_most_five_points": bool(
            (subgroup[candidate_id] - subgroup.baseline).ge(-0.05).all()
        ),
    }
    return (
        {
            "accepted": all(checks.values()),
            "checks": checks,
            "mean_fp_reduction": float(fp_reduction),
            "mean_fp_reduction_fraction": float(practical_fraction),
            "recall_difference_interval": [float(recall_lower), float(recall_upper)],
            "fp_reduction_interval": [float(fp_lower), float(fp_upper)],
            "interval_tail_probability": tail,
            "uncertainty": "paired reseller-cluster bootstrap; conditional on fits/selection; exploratory reuse",
        },
        paired,
        subgroup,
    )


def evaluate_predictions(
    features, predictions, profiles, output, candidate_id, tail, *, period_limits=None
):
    """Write the same assessment for fixed fitted models or fixed score blends."""
    from comparisons.baseline_diagnosis.analysis import (
        build_cases,
        metric_tables,
        ranking_tables,
    )
    from comparisons.development_checks import validate_development_cases
    from comparisons.development_reporting import (
        assess_period_safeguards,
        paired_operating_tables,
        validate_period_limits,
    )
    from reseller_churn.artifacts import write_json

    validate_development_cases(features)
    if period_limits is not None:
        validate_period_limits(**period_limits)
    operating, period_pairs, worst = paired_operating_tables(predictions, candidate_id)
    keys = ["StoreID", "snapshot"]
    require(
        set(predictions.strategy) == {"baseline", candidate_id},
        "Expected two matched arms",
    )
    require(
        not predictions.duplicated([*keys, "repeat", "strategy"]).any(),
        "Duplicate paired prediction",
    )
    expected = (
        features[keys + ["churn", "n_orders_3m"]]
        .sort_values(["snapshot", "StoreID"])
        .reset_index(drop=True)
    )
    for _, frame in predictions.groupby(["repeat", "strategy"]):
        actual = (
            frame[keys + ["churn", "n_orders_3m"]]
            .sort_values(["snapshot", "StoreID"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(actual, expected)
    operating.to_csv(output / "operating_metrics.csv", index=False)
    period_pairs.to_csv(output / "paired_operating_metrics.csv", index=False)
    worst.to_csv(output / "worst_slices.csv", index=False)
    all_cases, all_curves, all_burden = [], [], []
    for arm in ("baseline", candidate_id):
        rows = predictions[predictions.strategy.eq(arm)].assign(
            strategy="original_baseline"
        )
        cases = build_cases(features, rows, profiles)
        curves, burden = ranking_tables(cases)
        all_curves.append(curves.assign(policy=arm))
        all_burden.append(burden.assign(policy=arm))
        all_cases.append(cases.assign(policy=arm))
    metrics, rank_bands = metric_tables(pd.concat(all_cases, ignore_index=True))
    metrics.to_csv(output / "metrics.csv", index=False)
    rank_bands.to_csv(output / "rank_bands.csv", index=False)
    pd.concat(all_curves).to_csv(output / "ranking_curves.csv", index=False)
    pd.concat(all_burden).to_csv(output / "high_recall_burden.csv", index=False)
    bootstrap = paired_bootstrap(predictions, candidate_id=candidate_id)
    bootstrap.to_csv(output / "bootstrap.csv", index=False)
    decision, paired, subgroup = acceptance(metrics, bootstrap, tail, candidate_id)
    if period_limits is not None:
        passed, checks = assess_period_safeguards(period_pairs, **period_limits)
        checks.to_csv(output / "period_safeguards.csv", index=False)
        decision["period_limits"] = period_limits
        decision["checks"]["frozen_period_activity_limits"] = passed
        decision["accepted"] = decision["accepted"] and passed
    paired.to_csv(output / "paired_repeat_metrics.csv", index=False)
    subgroup.to_csv(output / "subgroup_recall_check.csv")
    write_json(output / "decision.json", decision)
    paired_cases = predictions[predictions.strategy.eq("baseline")].merge(
        predictions[predictions.strategy.eq(candidate_id)],
        on=[*keys, "repeat", "outer_fold", "churn", "n_orders_3m"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    require(
        len(paired_cases) * 2 == len(predictions), "Paired arms have different folds"
    )
    for arm in ("baseline", "candidate"):
        paired_cases[f"alert_{arm}"] = paired_cases[f"score_{arm}"].ge(
            paired_cases[f"cutoff_{arm}"]
        )
    paired_cases.to_csv(output / "paired_cases.csv", index=False)
    return decision


def publish_assessment(
    directory, source_folds, features, profiles, predictions, protocol
):
    """Publish a two-arm assessment from already saved fitting artifacts."""
    import json
    import shutil

    from comparisons.development_checks import validate_development_cases
    from reseller_churn.artifacts import (
        input_identity,
        new_output_directory,
        write_json,
        write_manifest,
    )

    validate_development_cases(features)
    directory = new_output_directory(directory)
    candidate = protocol["arms"][1]
    candidate_id = candidate["id"]
    features.to_csv(directory / "development_input.csv", index=False)
    shutil.copyfile(source_folds / "protocol.md", directory / "protocol.md")
    write_json(directory / "protocol.json", protocol)
    subset = predictions[predictions.strategy.isin(["baseline", candidate_id])]
    subset.to_csv(directory / "validation_predictions.csv", index=False)
    decision = evaluate_predictions(
        features,
        subset,
        profiles,
        directory,
        candidate_id,
        protocol["interval_tail_probability"],
        period_limits=protocol.get("period_limits"),
    )
    component_ids = {"baseline", candidate_id}
    component_ids.update(item["id"] for item in candidate.get("components", []))
    for split in protocol["splits"]:
        relative = f"repeat_{split['repeat']}/fold_{split['outer_fold']}"
        source = source_folds / relative
        folder = new_output_directory(directory / relative)
        inner = pd.read_csv(
            source / "inner_scores.csv",
            parse_dates=["snapshot"],
            float_precision="round_trip",
        )
        inner[inner.candidate.isin(["baseline", candidate_id])].to_csv(
            folder / "inner_scores.csv", index=False
        )
        choices = pd.read_csv(
            source / "inner_selections.csv", float_precision="round_trip"
        )
        choices[choices.candidate.isin(["baseline", candidate_id])].to_csv(
            folder / "inner_selections.csv", index=False
        )
        identities = json.loads((source / "model_identities.json").read_text())
        cached = source / "cached_hgb_identities.json"
        if cached.exists():
            identities.extend(json.loads(cached.read_text()))
        identities = [row for row in identities if row["candidate"] in component_ids]
        write_json(folder / "component_model_identities.json", identities)
        write_manifest(
            folder,
            {
                "new_fits": 0,
                "inputs": input_identity(
                    {"component_fold_manifest": source / "manifest.json"}
                ),
            },
        )
    write_json(
        directory / "baseline_fidelity.json",
        {
            "status": "unchanged cached original HGB scores and cutoff objective",
            "changed_baseline_decisions": 0,
        },
    )
    write_manifest(
        directory,
        {
            "stage": "matched_saved_score_assessment",
            "new_fits": 0,
            "decision": decision,
        },
    )
    return decision
