"""Two fixed product-engagement features under period/class weighting, development only."""

import argparse
import copy
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.development_checks import validate_development_cases
from comparisons.development_reporting import cutoff_transfer_tables
from comparisons.exact_bootstrap_signs import sign_audit
from comparisons.matched_evaluation import (
    acceptance,
    evaluate_predictions,
    paired_bootstrap,
)
from comparisons.period_weighting import period_weights
from comparisons.precision_floor_october import cutoff_scan, select_floor
from comparisons.reseller_weighting import fit_weighted
from comparisons.retained_calibration import LIMITS, intersection_gate
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    new_output_directory,
    source_identity,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = study.KEYS
REFERENCE = "period_class_weighted"
ARMS = [
    ("period_product_breadth", "unique_products_12m", 11),
    ("period_product_density", "unique_products_per_order_12m", 12),
]


def build_features(data, histories, profiles, orders):
    validate_development_cases(data)
    require(
        not orders.duplicated(KEYS + ["SalesOrderID"]).any(), "Duplicate case/order"
    )
    require(
        orders.OrderDate.lt(orders.snapshot).all()
        and orders.OrderDate.ge(orders.snapshot - pd.DateOffset(months=12)).all(),
        "Future/out-of-window order",
    )
    annual = profiles[profiles.scope.eq("annual")]
    require(set(profiles.scope) == {"annual", "latest_order"}, "Unknown profile scope")
    require(
        set(profiles.snapshot.dt.strftime("%Y-%m-%d")).issubset(study.SNAPSHOTS),
        "Nondevelopment profile",
    )
    products = annual.groupby(KEYS).ProductID.nunique().rename("unique_products_12m")
    count = orders.groupby(KEYS).SalesOrderID.nunique().rename("n_orders_12m")
    added = products.to_frame().join(count, how="outer").reset_index()
    require(
        added.notna().all().all() and added.n_orders_12m.gt(0).all(),
        "Missing historical exposure",
    )
    check = added.merge(
        histories[KEYS + ["unique_products_12m", "n_orders_12m"]],
        on=KEYS,
        validate="one_to_one",
        suffixes=("_built", "_saved"),
    )
    for name in ["unique_products_12m", "n_orders_12m"]:
        np.testing.assert_array_equal(check[name + "_built"], check[name + "_saved"])
    result = data[KEYS + ["churn"] + study.FEATURES].merge(
        added, on=KEYS, validate="one_to_one"
    )
    require(len(result) == len(data), "Historical population changed")
    result["unique_products_per_order_12m"] = (
        result.unique_products_12m / result.n_orders_12m
    )
    return result


def weighted_rank_metrics(labels, scores, weights):
    """Exact weighted tied-score AP/AUC for a matrix of reseller-bootstrap weights."""
    y = np.asarray(labels, dtype=int)
    score = np.asarray(scores, dtype=float)
    require(
        np.isin(y, [0, 1]).all() and np.isfinite(score).all(), "Invalid ranking inputs"
    )
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    require(
        w.shape[1] == len(y) and np.isfinite(w).all() and (w >= 0).all(),
        "Invalid ranking weights",
    )
    order = np.argsort(-score, kind="stable")
    y = y[order]
    score = score[order]
    w = w[:, order]
    starts = np.r_[0, np.flatnonzero(score[1:] != score[:-1]) + 1]
    positive = np.add.reduceat(w * y, starts, axis=1)
    negative = np.add.reduceat(w * (1 - y), starts, axis=1)
    tp = np.cumsum(positive, axis=1)
    fp = np.cumsum(negative, axis=1)
    total_positive = tp[:, -1]
    total_negative = fp[:, -1]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    ap = np.divide(
        (positive * precision).sum(axis=1),
        total_positive,
        out=np.full(len(w), np.nan),
        where=total_positive > 0,
    )
    numerator = (positive * (total_negative[:, None] - fp + 0.5 * negative)).sum(axis=1)
    auc = np.divide(
        numerator,
        total_positive * total_negative,
        out=np.full(len(w), np.nan),
        where=(total_positive * total_negative) > 0,
    )
    return ap, auc


def ranking_assessment(predictions, output, candidate):
    metrics = pd.read_csv(output / "metrics.csv", float_precision="round_trip")
    recent = metrics[
        metrics.scope.eq("outer_fold")
        & metrics["slice"].eq("activity")
        & metrics["group"].eq("recent_purchase")
    ]
    paired = recent[recent.policy.eq("baseline")].merge(
        recent[recent.policy.eq(candidate)],
        on=["repeat", "outer_fold"],
        suffixes=("_base", "_new"),
        validate="one_to_one",
    )
    require(len(paired) == 15, "Missing ranking fold")
    paired["ap_gain"] = paired.average_precision_new - paired.average_precision_base
    paired["auc_gain"] = paired.roc_auc_new - paired.roc_auc_base
    paired.to_csv(output / "paired_recent_ranking.csv", index=False)
    by_repeat = paired.groupby("repeat")[["ap_gain", "auc_gain"]].mean()
    by_repeat.to_csv(output / "recent_ranking_by_repeat.csv")
    checks = dict(
        recent_ap_gain_at_least_one_point=bool(paired.ap_gain.mean() >= 0.01),
        recent_ap_nondecreasing_every_repeat=bool(by_repeat.ap_gain.ge(0).all()),
        recent_auc_nondecreasing_mean=bool(paired.auc_gain.mean() >= 0),
        all_recent_fold_rank_metrics_defined=bool(
            paired[["ap_gain", "auc_gain"]].notna().all().all()
        ),
    )
    stores = np.sort(predictions.StoreID.unique())
    weights = np.random.default_rng(2026).multinomial(
        len(stores), np.repeat(1 / len(stores), len(stores)), size=5000
    )
    positions = {store: i for i, store in enumerate(stores)}
    ap_changes = []
    auc_changes = []
    for _, part in predictions[predictions.n_orders_3m.gt(0)].groupby(
        ["repeat", "outer_fold"]
    ):
        old = (
            part[part.strategy.eq("baseline")].sort_values(KEYS).reset_index(drop=True)
        )
        new = part[part.strategy.eq(candidate)].sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(old[KEYS + ["churn"]], new[KEYS + ["churn"]])
        case_weights = weights[:, [positions[s] for s in old.StoreID]]
        a, b = weighted_rank_metrics(old.churn, old.score, case_weights)
        c, d = weighted_rank_metrics(new.churn, new.score, case_weights)
        ap_changes.append(c - a)
        auc_changes.append(d - b)
    draws = pd.DataFrame(
        dict(ap_gain=np.mean(ap_changes, axis=0), auc_gain=np.mean(auc_changes, axis=0))
    )
    draws.to_csv(output / "ranking_bootstrap.csv", index=False)
    valid = draws.notna().all(axis=1)
    result = dict(
        checks=checks,
        mean_recent_ap_gain=float(paired.ap_gain.mean()),
        mean_recent_auc_gain=float(paired.auc_gain.mean()),
        descriptive_ap_gain_95=np.quantile(
            draws.loc[valid, "ap_gain"], [0.025, 0.975]
        ).tolist(),
        descriptive_auc_gain_95=np.quantile(
            draws.loc[valid, "auc_gain"], [0.025, 0.975]
        ).tolist(),
        undefined_draws=int((~valid).sum()),
        uncertainty="Paired reseller-cluster; within-fold rank differences averaged. Descriptive 95%, not familywise/adaptive confidence; omits refitting/calendar/future uncertainty.",
    )
    write_json(output / "ranking_decision.json", result)
    return result


def confirm(output, candidate, tail):
    primary = json.loads((output / "decision.json").read_text())
    final = copy.deepcopy(primary)
    stages = dict(
        confirmation="not_run_primary_failed", exact_signs="not_run_primary_failed"
    )
    if primary["accepted"]:
        predictions = read_csv(output / "validation_predictions.csv")
        metrics = pd.read_csv(output / "metrics.csv", float_precision="round_trip")
        folder = new_output_directory(output / "confirmation")
        draws = paired_bootstrap(
            predictions, draws=50000, seed=2026, candidate_id=candidate
        )
        draws.to_csv(folder / "bootstrap.csv", index=False)
        decision, _, _ = acceptance(metrics, draws, tail, candidate)
        write_json(folder / "decision.json", decision)
        final["checks"]["fifty_thousand_confirmation"] = decision["accepted"]
        stages["confirmation"] = "passed" if decision["accepted"] else "failed"
        stages["exact_signs"] = "not_run_confirmation_failed"
        if decision["accepted"]:
            audit = sign_audit(predictions, candidate, tail)
            write_json(folder / "exact_signs.json", audit)
            final["checks"]["exact_empirical_signs"] = audit[
                "conditional_sign_gates_passed"
            ]
            stages["exact_signs"] = (
                "passed" if audit["conditional_sign_gates_passed"] else "failed"
            )
        write_manifest(folder, dict(stage="additional_confirmation", models_fitted=0))
    final["accepted"] = all(final["checks"].values())
    write_json(
        output / "final_decision.json",
        dict(
            primary_decision_sha256=file_sha256(output / "decision.json"),
            decision=final,
            stages=stages,
        ),
    )
    return final


def fit_arm(data, profiles, source, split, spec, folder):
    train, held = validate_partition(data, split)
    relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
    cached = source / relative
    verify_manifest(cached)
    original_inner = read_csv(cached / "inner_scores.csv")
    reference = original_inner[original_inner.candidate.eq(REFERENCE)].copy()
    assignment = reference.set_index(KEYS).inner_fold.reindex(
        pd.MultiIndex.from_frame(train[KEYS])
    )
    require(assignment.notna().all(), "Missing cached inner membership")
    pieces = [reference.assign(candidate="baseline")]
    identities = []
    for fold in sorted(assignment.unique()):
        fitting = train.iloc[np.flatnonzero(assignment.to_numpy() != fold)]
        validation = train.iloc[np.flatnonzero(assignment.to_numpy() == fold)]
        require(
            set(fitting.StoreID).isdisjoint(validation.StoreID), "Inner reseller leak"
        )
        values, records, _, models = fit_weighted(
            fitting,
            validation[KEYS + spec["features"]],
            spec,
            weight_function=period_weights,
            retain_models=True,
        )
        identities.extend(
            [{**r, "stage": "inner", "inner_fold": int(fold)} for r in records]
        )
        part = validation[KEYS + ["churn"]].assign(
            candidate=spec["id"], score=values, inner_fold=int(fold)
        )
        part.to_csv(folder / f"inner_{fold}_scores.csv", index=False)
        joblib.dump(models, folder / f"inner_{fold}_models.joblib")
        pieces.append(part)
    inner = pd.concat(pieces, ignore_index=True)
    inner.to_csv(folder / "inner_scores.csv", index=False)
    selections = []
    for name, part in inner.groupby("candidate"):
        chosen = select_floor(cutoff_scan(part.churn, part.score), 40)
        selections.append(
            dict(
                candidate=name,
                feasible=chosen is not None,
                **(chosen if chosen is not None else {}),
            )
        )
    selection_table = pd.DataFrame(selections)
    selection_table.to_csv(folder / "inner_selections.csv", index=False)
    # Outer features carry no labels into fitting; outer labels only assess later.
    values, records, _, models = fit_weighted(
        train,
        held[KEYS + spec["features"]],
        spec,
        weight_function=period_weights,
        retain_models=True,
    )
    identities.extend([{**r, "stage": "outer"} for r in records])
    joblib.dump(models, folder / "outer_models.joblib")
    write_json(
        folder / "model_identities.json", dict(specification=spec, fits=identities)
    )
    cached_outer = read_csv(cached / "predictions.csv")
    reference_outer = cached_outer[cached_outer.strategy.eq(REFERENCE)].copy()
    reference_outer["strategy"] = "baseline"
    candidate = held[KEYS + ["churn", "n_orders_3m"]].assign(
        score=values,
        strategy=spec["id"],
        repeat=split["repeat"],
        outer_fold=split["outer_fold"],
    )
    pieces = []
    for part in [reference_outer, candidate]:
        choice = selection_table[
            selection_table.candidate.eq(part.strategy.iloc[0])
        ].iloc[0]
        part["cutoff"] = choice.cutoff if choice.feasible else np.nan
        pieces.append(part)
    predicted = pd.concat(pieces, ignore_index=True)
    predicted.to_csv(folder / "predictions.csv", index=False)
    shutil.copyfile(
        cached / "model_identities.json", folder / "cached_reference_identities.json"
    )
    if selection_table.feasible.all():
        cutoff_transfer_tables(inner, predicted, data).to_csv(
            folder / "cutoff_transfer.csv", index=False
        )
    write_manifest(
        folder,
        dict(
            source_inputs=input_identity(
                {"period_reference": cached / "manifest.json"}
            ),
            new_fits=len(identities),
            all_policies_feasible=bool(selection_table.feasible.all()),
        ),
    )
    return predicted, bool(selection_table.feasible.all())


def run(output, *, arms=ARMS, protocol_path=None, entrypoint=None):
    protocol_path = (
        protocol_path or ROOT / "comparisons/period_product_ranking_protocol.md"
    )
    source = ROOT / "results/period-weighting-20261018"
    history_source = ROOT / "results/persistent-error-investigation-20261016"
    product_source = ROOT / "results/product-mix-diagnosis-20261016"
    for path in [source, history_source, product_source]:
        verify_manifest(path)
    history = read_csv(history_source / "case_histories.csv")
    products = read_csv(product_source / "case_sku_shares.csv")
    orders = read_csv(history_source / "order_trajectories.csv")
    orders.OrderDate = pd.to_datetime(orders.OrderDate)
    data = build_features(
        read_csv(source / "development_input.csv"), history, products, orders
    )
    require(
        (len(data), int(data.churn.sum()), data.StoreID.nunique()) == (1375, 212, 422),
        "Wrong approved development population",
    )
    missing = [feature for _, feature, _ in arms if feature not in data]
    if missing:
        require(
            all(feature in history for feature in missing),
            "Unavailable prespecified historical feature",
        )
        data = data.merge(history[KEYS + missing], on=KEYS, validate="one_to_one")
    profiles = history[KEYS + ["n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    previous = json.loads((source / "protocol.json").read_text())
    out = new_output_directory(output)
    data.to_csv(out / "development_input.csv", index=False)
    shutil.copyfile(protocol_path, out / "protocol.md")
    names = [
        "period_product_ranking",
        "reseller_weighting",
        "period_weighting",
        "precision_floor_october",
        "matched_evaluation",
        "exact_bootstrap_signs",
        "retained_calibration",
    ]
    snapshot = new_output_directory(out / "source_snapshot")
    if entrypoint is not None:
        shutil.copyfile(entrypoint, snapshot / Path(entrypoint).name)
    for name in names:
        shutil.copyfile(ROOT / "comparisons" / f"{name}.py", snapshot / f"{name}.py")
    write_json(
        out / "prefit_identity.json",
        dict(
            code=input_identity({n: ROOT / "comparisons" / f"{n}.py" for n in names}),
            inputs=input_identity(
                {
                    "reference": source / "manifest.json",
                    "histories": history_source / "manifest.json",
                    "products": product_source / "manifest.json",
                    "protocol": protocol_path,
                }
            ),
            runtime=source_identity()["versions"],
        ),
    )
    specs = [
        dict(
            id=name,
            features=study.FEATURES + [feature],
            parameters=study.BASE_PARAMETERS,
            alpha_spending_cycle=cycle,
            interval_tail_probability=0.001 / 2 ** (cycle - 1) / 4,
        )
        for name, feature, cycle in arms
    ]
    write_json(
        out / "protocol.json",
        dict(
            arms=specs,
            reference=REFERENCE,
            reference_parameters=study.BASE_PARAMETERS,
            splits=previous["splits"],
            precision_floor=0.4,
            october_inputs=False,
        ),
    )
    for spec in specs:
        arm = new_output_directory(out / spec["id"])
        parts = []
        feasible = True
        for split in previous["splits"]:
            folder = new_output_directory(
                arm / f"repeat_{split['repeat']}/fold_{split['outer_fold']}"
            )
            predictions, okay = fit_arm(data, profiles, source, split, spec, folder)
            parts.append(predictions)
            feasible &= okay
            print(
                spec["id"], split["repeat"], split["outer_fold"], "complete", flush=True
            )
        predictions = pd.concat(parts, ignore_index=True)
        predictions.to_csv(arm / "validation_predictions.csv", index=False)
        if not feasible:
            write_json(
                arm / "decision.json",
                dict(
                    accepted=False,
                    checks={"all_policies_feasible": False},
                    reason="At least one inner precision policy infeasible; raw scores retained, no invented alerts",
                ),
            )
            write_manifest(arm, dict(stage="infeasible_comparison", models_fitted=300))
            continue
        evaluate_predictions(
            data,
            predictions,
            profiles,
            arm,
            spec["id"],
            spec["interval_tail_probability"],
            period_limits=LIMITS,
        )
        primary = intersection_gate(arm)
        ranking = ranking_assessment(predictions, arm, spec["id"])
        primary["checks"].update(ranking["checks"])
        primary["accepted"] = all(primary["checks"].values())
        write_json(arm / "decision.json", primary)
        final = confirm(arm, spec["id"], spec["interval_tail_probability"])
        write_manifest(
            arm,
            dict(
                stage="matched_development_assessment",
                new_fits=300,
                accepted=final["accepted"],
                primary_reference=REFERENCE,
            ),
        )
        print(spec["id"], json.dumps(final), flush=True)
    write_manifest(
        out,
        dict(
            stage="fixed_feature_ranking_trials",
            october_inputs=False,
            new_fits=300 * len(specs),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
