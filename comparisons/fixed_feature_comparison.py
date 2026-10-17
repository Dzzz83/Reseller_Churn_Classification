"""One fixed numeric-feature HGB procedure against unchanged cached baseline."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.matched_evaluation import evaluate_predictions
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

KEYS = study.KEYS


def evaluate_fixed_fold(data, split, cached, output, spec):
    """Fit one candidate with cached inner groups; outer labels only assess."""
    from comparisons.development_checks import validate_development_cases

    validate_development_cases(data, spec["features"])
    train, valid = validate_partition(data, split)
    verify_manifest(cached)
    folder = new_output_directory(output)
    inner = read_csv(cached / "inner_scores.csv")
    baseline_inner = inner[inner.candidate.eq("baseline")].copy()
    assignment = baseline_inner.set_index(KEYS).inner_fold.reindex(
        pd.MultiIndex.from_frame(train[KEYS])
    )
    require(assignment.notna().all(), "Incomplete cached inner assignment")
    inner_parts = [baseline_inner]
    identities = [
        row
        for row in json.loads((cached / "model_identities.json").read_text())
        if row["candidate"] == "baseline"
    ]
    for fold in sorted(assignment.unique()):
        fitting = train.iloc[np.flatnonzero(assignment.to_numpy() != fold)]
        held = train.iloc[np.flatnonzero(assignment.to_numpy() == fold)]
        require(set(fitting.StoreID).isdisjoint(held.StoreID), "Inner reseller leakage")
        scores, fitted, spread, _ = study.fit_scores(
            fitting, held[spec["features"]], spec
        )
        identities.extend(
            dict(**row, stage="inner", inner_fold=int(fold), seed_score_range=spread)
            for row in fitted
        )
        rows = held[KEYS + ["churn"]].copy()
        rows["score"], rows["candidate"], rows["inner_fold"] = (
            scores,
            spec["id"],
            int(fold),
        )
        inner_parts.append(rows)
    inner = pd.concat(inner_parts, ignore_index=True)
    choices = []
    for arm in ["baseline", spec["id"]]:
        rows = inner[inner.candidate.eq(arm)]
        choices.append(
            dict(
                candidate=arm,
                **study.select_threshold(rows.churn.to_numpy(), rows.score.to_numpy()),
            )
        )
    cached_outer = read_csv(cached / "predictions.csv")
    baseline = cached_outer[cached_outer.strategy.eq("baseline")]
    require(
        baseline.cutoff.eq(choices[0]["cutoff"]).all(),
        "Cached baseline cutoff changed",
    )
    scores, fitted, spread, _ = study.fit_scores(train, valid[spec["features"]], spec)
    identities.extend(
        dict(**row, stage="outer", seed_score_range=spread) for row in fitted
    )
    predicted = valid[KEYS + ["churn", "n_orders_3m"]].copy()
    predicted["score"], predicted["cutoff"], predicted["strategy"] = (
        scores,
        choices[1]["cutoff"],
        spec["id"],
    )
    predicted["repeat"], predicted["outer_fold"] = (
        split["repeat"],
        split["outer_fold"],
    )
    outer = pd.concat([baseline, predicted], ignore_index=True)
    inner.to_csv(folder / "inner_scores.csv", index=False)
    outer.to_csv(folder / "predictions.csv", index=False)
    pd.DataFrame(choices).to_csv(folder / "inner_selections.csv", index=False)
    write_json(folder / "model_identities.json", identities)
    shutil.copyfile(cached / "split.json", folder / "split.json")
    write_manifest(
        folder,
        dict(
            new_fits=20,
            cached_fits=20,
            inputs=input_identity(dict(source=cached / "manifest.json")),
        ),
    )
    return outer


def run_fixed_hgb(
    data,
    profiles,
    source,
    output,
    spec,
    protocol_path,
    tail,
    extra_inputs,
    *,
    period_limits=None,
):
    """Reuse baseline fits and memberships; every candidate choice is already fixed."""
    from comparisons.development_checks import validate_development_cases
    from comparisons.development_reporting import validate_period_limits

    validate_development_cases(data, spec["features"])
    if period_limits is not None:
        validate_period_limits(**period_limits)
    verify_manifest(source)
    previous = json.loads((source / "protocol.json").read_text())
    out = new_output_directory(output)
    shutil.copyfile(protocol_path, out / "protocol.md")
    data.to_csv(out / "development_input.csv", index=False)
    write_json(
        out / "protocol.json",
        dict(
            description=spec["description"],
            arms=[previous["arms"][0], spec],
            splits=previous["splits"],
            repeat_seeds=previous["repeat_seeds"],
            interval_tail_probability=tail,
            **({"period_limits": period_limits} if period_limits is not None else {}),
        ),
    )
    parts = []
    for split in previous["splits"]:
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        parts.append(
            evaluate_fixed_fold(data, split, source / relative, out / relative, spec)
        )
        print(
            f"Completed repeat {split['repeat']} fold {split['outer_fold']}", flush=True
        )
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    decision = evaluate_predictions(
        data, predictions, profiles, out, spec["id"], tail, period_limits=period_limits
    )
    write_json(
        out / "baseline_fidelity.json",
        dict(status="unchanged cached HGB", changed_baseline_decisions=0),
    )
    reference = read_csv(source / "validation_predictions.csv")
    reference = reference[reference.strategy.eq("cadence_no_count")]
    rows = []
    for (repeat, arm), part in pd.concat(
        [reference, predictions[predictions.strategy.eq(spec["id"])]]
    ).groupby(["repeat", "strategy"]):
        for activity, subset in [
            ("overall", part),
            ("recent_purchase", part[part.n_orders_3m.gt(0)]),
            ("no_recent_purchase", part[part.n_orders_3m.eq(0)]),
        ]:
            a, y = subset.score.ge(subset.cutoff), subset.churn.eq(1)
            tp, fp = int((a & y).sum()), int((a & ~y).sum())
            rows.append(
                dict(
                    repeat=int(repeat),
                    policy=arm,
                    activity=activity,
                    tp=tp,
                    fp=fp,
                    recall=tp / y.sum(),
                    precision=tp / max(1, tp + fp),
                )
            )
    pd.DataFrame(rows).to_csv(out / "cadence_reference_comparison.csv", index=False)
    write_manifest(
        out,
        dict(
            stage="fixed_numeric_feature",
            new_fits=300,
            cached_fits=300,
            decision=decision,
            inputs=input_identity(
                dict(
                    source=source / "manifest.json",
                    protocol=protocol_path,
                    **extra_inputs,
                )
            ),
            implementation=input_identity(
                dict(runner=Path(__file__), fitting=Path(study.__file__))
            ),
        ),
    )
    print(json.dumps(decision, indent=2))
