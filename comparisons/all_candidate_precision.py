"""Prepare all frozen candidates, then evaluate their development precision-floor rules."""

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from comparisons import grouped_precision_recall as study
from comparisons.activity_experts import fit_experts
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.continuation_blend import undo_balanced_odds
from comparisons.development_checks import validate_development_cases
from comparisons.evaluate_frozen_october import verified_artifact
from comparisons.guarded_component import choose_component
from comparisons.period_weighting import period_weights
from comparisons.positive_floor_october_evaluation import join_supplied_outcomes
from comparisons.precision_floor_october import (
    FLOORS,
    cutoff_scan,
    independent_choice,
    select_floor,
)
from comparisons.rbf_cadence import fit_margin
from comparisons.reseller_weighting import fit_weighted
from comparisons.smooth_cadence import fit_smooth
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    source_identity,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
VETO = ROOT / "results/positive-floor-october-evaluation-20261018"
UNIT = ROOT / "results/unit-revenue-october-evaluation-20261017"
OLD = ROOT / "results/repeated-grouped-precision"
PROTOCOL = ROOT / "comparisons/all_candidate_precision_protocol.md"
KEYS = study.KEYS
STUDIES = dict(
    smooth_cadence="smooth-cadence-20261018-02",
    reseller_weighted="reseller-weighting-20261018",
    activity_experts="activity-experts-20261018-02",
    period_class_weighted="period-weighting-20261018",
    rbf_cadence="rbf-cadence-20261018",
    early_quarter_share="early-quarter-share-20261018",
    cadence_regularized="repeated-grouped-precision",
)
NAMES = [
    "baseline",
    "retained_logistic",
    "smooth_cadence",
    "reseller_weighted",
    "activity_experts",
    "complementary_mean",
    "period_class_weighted",
    "rbf_cadence",
    "early_quarter_share",
    "guarded_component",
    "positive_floor_veto",
    "cadence_regularized",
    "cadence_unit_revenue",
]


def ordered(frame):
    return frame.sort_values(KEYS).reset_index(drop=True)


def historical_share(keys, orders):
    """Apply the fixed historical-quarter formula; future orders never contribute."""
    allowed = {*study.SNAPSHOTS, "2013-10-01"}
    require(
        set(keys.snapshot.dt.strftime("%Y-%m-%d")).issubset(allowed),
        "Unapproved scoring snapshot",
    )
    require(
        not orders.duplicated(["StoreID", "SalesOrderID"]).any(),
        "Duplicate accepted orders",
    )
    records = []
    for row in keys[KEYS].itertuples(index=False):
        dates = orders.loc[orders.StoreID.eq(row.StoreID), "OrderDate"]
        dates = dates[
            (dates >= row.snapshot - pd.DateOffset(months=12)) & (dates < row.snapshot)
        ]
        require(len(dates) > 0, "Eligible reseller has no historical orders")
        count = int((dates < row.snapshot - pd.DateOffset(months=9)).sum())
        records.append(
            dict(
                StoreID=row.StoreID,
                snapshot=row.snapshot,
                early_quarter_order_share=count / len(dates),
                annual_orders=len(dates),
            )
        )
    return pd.DataFrame(records)


def cached_scores(path, column, value):
    frame = read_csv(path)
    return ordered(frame[frame[column].eq(value)])


def fit_spec(train, held, spec, retain=False):
    """One fixed existing learner; held input contains no outcomes."""
    require("churn" not in held, "Held outcomes passed to learner")
    name = spec["id"]
    if name == "smooth_cadence":
        return fit_smooth(train, held, spec, retain_models=retain)
    if name == "reseller_weighted":
        return fit_weighted(train, held, spec, retain_models=retain)
    if name == "period_class_weighted":
        return fit_weighted(
            train, held, spec, weight_function=period_weights, retain_models=retain
        )
    if name == "activity_experts":
        return fit_experts(train, held, spec, retain_models=retain)
    if name == "rbf_cadence":
        return fit_margin(train, held, spec, retain_models=retain)
    return study.fit_scores(train, held, spec, retain_models=retain)


def score_models(bundle, features):
    """Reproduce every newly retained full-fit score without fitting."""
    name = bundle["spec"]["id"]
    models = bundle["models"]
    columns = bundle["spec"]["features"]
    if name == "rbf_cadence":
        return expit(models[0].decision_function(features[columns]))
    if name == "activity_experts":
        values = []
        for seed in study.MODEL_SEEDS:
            score = np.full(len(features), np.nan)
            for pair in [m for m in models if m["seed"] == seed]:
                mask = features.n_orders_3m.gt(0).eq(pair["recent_purchase"]).to_numpy()
                if mask.any():
                    score[mask] = pair["model"].predict_proba(
                        features.loc[mask, pair["features"]]
                    )[:, 1]
            require(np.isfinite(score).all(), "Unscored expert case")
            values.append(score)
        return np.mean(values, axis=0)
    return np.mean([m.predict_proba(features[columns])[:, 1] for m in models], axis=0)


def prepare(output, cache=None):
    require(not output.exists(), "Preserve previous run")
    for source in [VETO, UNIT]:
        verify_manifest(source)
    out = new_output_directory(output)
    shutil.copyfile(PROTOCOL, out / "protocol.md")
    source_dir = new_output_directory(out / "source_snapshot")
    code = [
        "all_candidate_precision",
        "precision_floor_october",
        "smooth_cadence",
        "reseller_weighting",
        "activity_experts",
        "period_weighting",
        "rbf_cadence",
        "guarded_component",
        "continuation_blend",
        "grouped_precision_recall",
    ]
    paths = {name: ROOT / "comparisons" / f"{name}.py" for name in code}
    for path in paths.values():
        shutil.copyfile(path, source_dir / path.name)
    write_json(
        out / "preexecution_identity.json",
        dict(
            code=input_identity(paths),
            inputs=input_identity(
                {
                    "protocol": PROTOCOL,
                    "veto_preparation": VETO / "manifest.json",
                    "unit_preparation": UNIT / "manifest.json",
                }
            ),
            runtime=source_identity()["versions"],
        ),
    )
    data = ordered(read_csv(VETO / "development_input.csv"))
    validate_development_cases(data)
    require(
        (len(data), int(data.churn.sum()), data.StoreID.nunique()) == (1375, 212, 422),
        "Development identity differs",
    )
    require(
        max(data.snapshot + pd.DateOffset(months=6)) <= pd.Timestamp("2013-10-01"),
        "Training labels not matured",
    )
    unit_data = ordered(
        read_csv(ROOT / "results/cadence-unit-revenue-20261017/development_input.csv")
    )
    pd.testing.assert_frame_equal(data, unit_data[data.columns])
    data["last_order_unit_revenue"] = unit_data.last_order_unit_revenue
    earlier = ordered(read_csv(OLD / "development_input.csv"))
    pd.testing.assert_frame_equal(
        data[KEYS + ["churn"] + study.FEATURES],
        earlier[KEYS + ["churn"] + study.FEATURES],
    )
    data["n_orders_12m"] = earlier.n_orders_12m
    test = ordered(read_csv(UNIT / "unlabeled_features.csv"))
    saved = ROOT / "results/current-model-diagnosis"
    verified_artifact(saved / "manifest.json", saved / "unlabeled_october_scores.csv")
    history = ordered(read_csv(saved / "unlabeled_october_scores.csv"))
    pd.testing.assert_frame_equal(test[KEYS], history[KEYS])
    np.testing.assert_allclose(
        test[study.FEATURES],
        history[study.FEATURES],
        rtol=1e-12,
        atol=1e-9,
        equal_nan=True,
    )
    test["n_orders_12m"] = history.n_orders_12m
    unit_protocol = json.loads((UNIT / "protocol.json").read_text())
    order_identity = unit_protocol["inputs"]["orders"]
    order_path = Path(order_identity["path"])
    require(
        study.file_hash(order_path) == order_identity["sha256"],
        "Historical accepted-order source changed",
    )
    orders = pd.read_csv(
        order_path,
        usecols=["StoreID", "SalesOrderID", "OrderDate"],
        parse_dates=["OrderDate"],
    )
    counts = ordered(
        historical_share(pd.concat([data[KEYS], test[KEYS]], ignore_index=True), orders)
    )
    for frame in [data, test]:
        addition = frame[KEYS].merge(counts, on=KEYS, validate="one_to_one")
        np.testing.assert_array_equal(addition.annual_orders, frame.n_orders_12m)
        frame["early_quarter_order_share"] = addition.early_quarter_order_share
    early_data = ordered(
        read_csv(ROOT / "results/early-quarter-share-20261018/development_input.csv")
    )
    np.testing.assert_array_equal(
        data.early_quarter_order_share, early_data.early_quarter_order_share
    )
    counts.to_csv(out / "historical_quarter_audit.csv", index=False)
    write_json(
        out / "historical_source_identity.json",
        input_identity(
            {
                "orders": order_path,
                "older_development": OLD / "development_input.csv",
                "earliest_share_study": ROOT
                / "results/early-quarter-share-20261018/manifest.json",
            }
        ),
    )
    data.to_csv(out / "development_input.csv", index=False)
    test.to_csv(out / "unlabeled_features.csv", index=False)
    inner = {}
    outer = {}
    unit_inner = read_csv(UNIT / "development_oof.csv")
    unit_test = read_csv(UNIT / "unlabeled_predictions.csv")
    veto_inner = read_csv(VETO / "development_oof.csv")
    veto_test = read_csv(VETO / "unlabeled_predictions.csv")
    for name, inside in [
        ("baseline", unit_inner),
        ("cadence_unit_revenue", unit_inner),
        ("retained_logistic", veto_inner),
        ("positive_floor_veto", veto_inner),
    ]:
        rows = ordered(inside[inside.candidate.eq(name)])
        pd.testing.assert_frame_equal(rows[KEYS + ["churn"]], data[KEYS + ["churn"]])
        inner[name] = rows.score.to_numpy()
        if name == "retained_logistic":
            scored = ordered(read_csv(VETO / "component_scores.csv"))
        else:
            table = veto_test if name == "positive_floor_veto" else unit_test
            scored = ordered(table[table.strategy.eq(name)])
        pd.testing.assert_frame_equal(scored[KEYS], test[KEYS])
        outer[name] = scored.score.to_numpy()
    splits = study.grouped_splits(data, 3, 78)
    write_json(
        out / "inner_splits.json",
        dict(
            folds=[
                dict(fold=i, train=ti.tolist(), validation=vi.tolist())
                for i, (ti, vi) in enumerate(splits)
            ]
        ),
    )
    for fold, (_, vi) in enumerate(splits):
        cached = ordered(
            unit_inner[
                unit_inner.candidate.eq("baseline") & unit_inner.inner_fold.eq(fold)
            ]
        )
        pd.testing.assert_frame_equal(cached[KEYS], ordered(data.iloc[vi][KEYS]))
    # The earlier cadence13 bundle has an incompatible runtime. Its exact
    # specification and three-fold OOF are refitted below in the active runtime.
    write_json(
        out / "cached_sources.json",
        input_identity(
            {
                "veto_manifest": VETO / "manifest.json",
                "unit_manifest": UNIT / "manifest.json",
                "earlier_manifest": OLD / "manifest.json",
            }
        ),
    )
    for name, study_dir in STUDIES.items():
        source = ROOT / "results" / study_dir
        if name == "cadence_regularized":
            # Legacy manifests contain Windows nested paths; verify the exact
            # root-level frozen inputs used here rather than reinterpret paths.
            for artifact in ["protocol.json", "frozen_candidate.json"]:
                verified_artifact(source / "manifest.json", source / artifact)
        else:
            verify_manifest(source)
        protocol = json.loads((source / "protocol.json").read_text())
        spec = (
            json.loads((source / "frozen_candidate.json").read_text())["candidate"]
            if name == "cadence_regularized"
            else protocol["arms"][1]
        )
        require(spec["id"] == name, "Candidate specification differs")
        folder = new_output_directory(out / name)
        shutil.copyfile(source / "protocol.json", folder / "source_protocol.json")
        values = np.full(len(data), np.nan)
        identities = []
        cached_folder = cache / name if cache is not None else None
        cached = (
            cached_folder is not None and (cached_folder / "manifest.json").is_file()
        )
        if cached:
            verify_manifest(cached_folder)
            bundle = joblib.load(cached_folder / "frozen_models.joblib")
            require(bundle["spec"] == spec, "Cached procedure differs")
            prior_records = json.loads(
                (cached_folder / "training_identities.json").read_text()
            )["fits"]
        for fold, (ti, vi) in enumerate(splits):
            train, held = data.iloc[ti], data.iloc[vi]
            require(
                set(train.StoreID).isdisjoint(held.StoreID), "Inner reseller leakage"
            )
            previous = (
                [
                    r
                    for r in prior_records
                    if r["stage"] == "inner" and r["inner_fold"] == fold
                ]
                if cached
                else []
            )
            if cached and (cached_folder / "inner_scores.csv").is_file():
                saved_inner = read_csv(cached_folder / "inner_scores.csv").set_index(
                    KEYS
                )
                score = saved_inner.reindex(
                    pd.MultiIndex.from_frame(held[KEYS])
                ).score.to_numpy()
                require(np.isfinite(score).all(), "Missing cached inner predictions")
                fitted = previous
                retained_path = cached_folder / f"inner_fold_{fold}_models.joblib"
                if retained_path.exists():
                    shutil.copyfile(retained_path, folder / retained_path.name)
            elif cached and name in ["smooth_cadence", "rbf_cadence"]:
                require(len(previous) == 1, "Missing reconstructable identity")
                if name == "smooth_cadence":
                    from comparisons.verify_smooth_cadence import reconstruct

                    score = reconstruct(held, previous[0])
                else:
                    from comparisons.verify_rbf_cadence import margins

                    score = expit(margins(held, previous[0]))
                fitted = previous
            else:
                score, records, _, inner_models = fit_spec(
                    train, held[KEYS + spec["features"]], spec, retain=True
                )
                fitted = [{**r, "stage": "inner", "inner_fold": fold} for r in records]
                if cached:
                    # Exact metadata agreement establishes unchanged recovery fits.
                    require(fitted == previous, "Recovered inner fit identity differs")
                joblib.dump(
                    dict(spec=spec, models=inner_models),
                    folder / f"inner_fold_{fold}_models.joblib",
                )
            values[vi] = score
            identities.extend(fitted)
        data[KEYS + ["churn"]].assign(score=values).to_csv(
            folder / "inner_scores.csv", index=False
        )
        if cached:
            score = pd.read_csv(
                cached_folder / "full_scores.csv", float_precision="round_trip"
            ).score.to_numpy()
            identities.extend([r for r in prior_records if r["stage"] == "full"])
            shutil.copyfile(
                cached_folder / "frozen_models.joblib", folder / "frozen_models.joblib"
            )
        else:
            score, records, _, models = fit_spec(
                data, test[KEYS + spec["features"]], spec, retain=True
            )
            bundle = dict(spec=spec, models=models)
            identities.extend([{**r, "stage": "full"} for r in records])
            joblib.dump(bundle, folder / "frozen_models.joblib")
        require(
            np.isfinite(values).all() and np.isfinite(score).all(),
            "Missing candidate scores",
        )
        np.testing.assert_allclose(
            score, score_models(bundle, test), rtol=0, atol=1e-15
        )
        write_json(folder / "training_identities.json", dict(fits=identities))
        pd.DataFrame(dict(score=score)).to_csv(folder / "full_scores.csv", index=False)
        write_manifest(
            folder,
            dict(
                inputs=input_identity(
                    {
                        "development_study": source / "manifest.json",
                        **(
                            {"recovered_full_model": cached_folder / "manifest.json"}
                            if cached
                            else {}
                        ),
                    }
                ),
                outcomes_read=False,
                cached_full_models=cached,
            ),
        )
        inner[name] = values
        outer[name] = score
        print("Prepared fixed procedure:", name, flush=True)

    # Fixed complementary formula uses only its own fitting partition's prevalence.
    def combine(calibrated, expert, prevalence):
        return (calibrated + undo_balanced_odds(expert, prevalence)) / 2

    blend = np.empty(len(data))
    for ti, vi in splits:
        blend[vi] = combine(
            inner["retained_logistic"][vi],
            inner["activity_experts"][vi],
            float(data.iloc[ti].churn.mean()),
        )
    inner["complementary_mean"] = blend
    outer["complementary_mean"] = combine(
        outer["retained_logistic"], outer["activity_experts"], float(data.churn.mean())
    )
    reference = data[KEYS + ["churn", "n_orders_3m"]].assign(score=inner["baseline"])
    component = reference.assign(score=inner["retained_logistic"])
    choice, scan, groups = choose_component(reference, component)
    write_json(out / "guarded_component_selection.json", choice)
    scan.to_csv(out / "guarded_inner_scan.csv", index=False)
    groups.to_csv(out / "guarded_reference_groups.csv", index=False)
    inner["guarded_component"] = inner[choice["component"]].copy()
    outer["guarded_component"] = outer[choice["component"]].copy()
    require(set(inner) == set(outer) == set(NAMES), "Missing planned candidate")
    selections = []
    oof_parts = []
    test_parts = []
    status = []
    for name in NAMES:
        oof = data[KEYS + ["churn"]].assign(candidate=name, score=inner[name])
        oof_parts.append(oof)
        scan = cutoff_scan(data.churn, inner[name])
        scan.to_csv(out / (name + "_cutoff_scan.csv"), index=False)
        for floor in FLOORS:
            chosen = select_floor(scan, floor)
            direct = independent_choice(data.churn, inner[name], floor)
            require(
                (chosen is None and direct is None)
                or (chosen is not None and chosen["cutoff"] == direct),
                "Independent threshold selection differs",
            )
            record = dict(
                candidate=name,
                minimum_precision_percent=floor,
                feasible=chosen is not None,
            )
            if chosen is not None:
                record.update(chosen)
                part = test[KEYS + ["n_orders_3m"]].assign(
                    strategy=name + "__" + str(floor),
                    candidate=name,
                    minimum_precision_percent=floor,
                    score=outer[name],
                    cutoff=chosen["cutoff"],
                )
                part["predicted_churn"] = part.score.ge(part.cutoff).astype(int)
                test_parts.append(part)
            selections.append(record)
        status.append(dict(candidate=name, status="prepared", new_fits=name in STUDIES))
    pd.concat(oof_parts, ignore_index=True).to_csv(
        out / "development_oof.csv", index=False
    )
    pd.DataFrame(selections).to_csv(out / "development_selections.csv", index=False)
    pd.DataFrame(status).to_csv(out / "candidate_status.csv", index=False)
    pd.concat(test_parts, ignore_index=True).to_csv(
        out / "unlabeled_predictions.csv", index=False
    )
    write_manifest(
        out,
        dict(
            stage="all_candidate_predictions_checkpointed",
            outcomes_read=False,
            candidates=NAMES,
            precision_floors=FLOORS,
        ),
    )
    print(
        "ALL cutoffs and unlabeled predictions checkpointed; outcomes not read.",
        flush=True,
    )


def evaluate(output):
    stage = verify_manifest(output)
    require(
        stage["stage"] == "all_candidate_predictions_checkpointed"
        and not stage["outcomes_read"],
        "Wrong checkpoint",
    )
    destination = output / "evaluation"
    require(not destination.exists(), "Preserve completed evaluation")
    outcomes = ROOT / "results/current-model-diagnosis/october_cases.csv"
    verified_artifact(outcomes.parent / "manifest.json", outcomes)
    labels = pd.read_csv(outcomes, usecols=KEYS + ["churn"])
    predictions = read_csv(output / "unlabeled_predictions.csv")
    require(
        predictions.predicted_churn.eq(predictions.score.ge(predictions.cutoff)).all(),
        "Changed frozen decisions",
    )
    cases = join_supplied_outcomes(predictions, labels)
    out = new_output_directory(destination)
    cases.to_csv(out / "evaluated_cases.csv", index=False)
    rows = []
    for (name, floor), part in cases.groupby(
        ["candidate", "minimum_precision_percent"]
    ):
        for group, subset in [
            ("overall", part),
            ("recent_purchase", part[part.n_orders_3m.gt(0)]),
            ("no_recent_purchase", part[part.n_orders_3m.eq(0)]),
        ]:
            m = study.decision_metrics(subset.churn, subset.score, subset.cutoff)
            y = subset.churn.eq(1)
            a = subset.predicted_churn.eq(1)
            require(
                [m[k] for k in ["tp", "fp", "fn", "tn"]]
                == [
                    int((y & a).sum()),
                    int((~y & a).sum()),
                    int((y & ~a).sum()),
                    int((~y & ~a).sum()),
                ],
                "Independent confusion differs",
            )
            precision_defined = bool(a.any())
            recall_defined = bool(y.any())
            if not precision_defined:
                m["precision"] = np.nan
            if not recall_defined:
                m["recall"] = np.nan
            rows.append(
                dict(
                    candidate=name,
                    minimum_precision_percent=int(floor),
                    subgroup=group,
                    precision_defined=precision_defined,
                    recall_defined=recall_defined,
                    **m,
                )
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    selected = pd.read_csv(
        output / "development_selections.csv", float_precision="round_trip"
    )
    overall = metrics[metrics.subgroup.eq("overall")].drop(columns="subgroup")
    summary = selected.merge(
        overall,
        on=["candidate", "minimum_precision_percent"],
        how="left",
        suffixes=("_development", "_test"),
        validate="one_to_one",
    )
    summary.to_csv(out / "comparison.csv", index=False)
    report = "# Candidate-wide development precision-floor evaluation on inspected October\n\n"
    report += "All listed candidates and all20–100%targets retained. Each cutoff maximizes development OOF recall subject to precision>=target; ties precision then cutoff. No recall floor. Natural matched cases; no outcome tuning or winner selection. October previously inspected, so descriptive reuse. Development acceptance failures remain; no deployment.\n\n"
    for name in NAMES:
        part = summary[summary.candidate.eq(name)]
        report += (
            "## " + name + "\n\n```text\n" + part.to_string(index=False) + "\n```\n\n"
        )
    report += "Undefined precision is NaN, never a successful target. Extreme targets with very few development alerts are fragile. Threshold selection on OOF labels remains adaptive development use. Full-fit scores can differ from OOF operating rates; test precision is not guaranteed by its development target. Supplied outcomes reuse historical accepted-order reconstruction; source completeness and historical availability remain assumptions. No new independent validation or training/selection uncertainty claim.\n"
    (out / "report.md").write_text(report)
    write_manifest(
        out,
        dict(
            stage="all_candidates_evaluated",
            models_fitted=0,
            cutoffs_selected_on_test=0,
            candidates=len(NAMES),
            target_comparisons=len(summary),
            independent_confusion_rows=len(metrics),
            inputs=input_identity(
                {
                    "prediction_checkpoint": output / "manifest.json",
                    "supplied_outcomes": outcomes,
                }
            ),
        ),
    )
    print(
        summary[summary.minimum_precision_percent.isin([30, 40, 50])][
            [
                "candidate",
                "minimum_precision_percent",
                "precision_test",
                "recall_test",
                "tp_test",
                "fp_test",
                "alerts_test",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "evaluate"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output, args.cache)
    else:
        evaluate(args.output)
