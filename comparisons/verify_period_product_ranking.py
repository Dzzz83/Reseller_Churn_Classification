"""No-fit audit of both frozen product-engagement comparisons."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    verify_manifest,
    write_json,
)

KEYS = ["StoreID", "snapshot"]


def ordered(frame):
    return frame.sort_values(KEYS).reset_index(drop=True)


def threshold(part):
    options = []
    y = part.churn.eq(1).to_numpy()
    score = part.score.to_numpy()
    for cut in sorted(set([0.0, *score])):
        alert = score >= cut
        tp = int((y & alert).sum())
        fp = int((~y & alert).sum())
        if tp + fp > 0 and 5 * tp >= 2 * (tp + fp):
            options.append((tp, Fraction(tp, tp + fp), float(cut)))
    return None if not options else max(options)[2]


def verify(directory):
    verify_manifest(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    pre = json.loads((directory / "prefit_identity.json").read_text())
    for identity in pre["code"].values():
        require(
            file_sha256(directory / "source_snapshot" / Path(identity["path"]).name)
            == identity["sha256"],
            "Frozen code identity changed",
        )
    data = read_csv(directory / "development_input.csv")
    history_path = directory.parent / "persistent-error-investigation-20261016"
    h = read_csv(history_path / "case_histories.csv")
    direct = data[
        KEYS + ["unique_products_12m", "n_orders_12m", "unique_products_per_order_12m"]
    ].merge(
        h[KEYS + ["unique_products_12m", "n_orders_12m"]],
        on=KEYS,
        validate="one_to_one",
        suffixes=("_built", "_original"),
    )
    require(len(direct) == 1375, "Feature population differs")
    for name in ["unique_products_12m", "n_orders_12m"]:
        np.testing.assert_array_equal(
            direct[name + "_built"], direct[name + "_original"]
        )
    np.testing.assert_array_equal(
        direct.unique_products_per_order_12m,
        direct.unique_products_12m_built / direct.n_orders_12m_built,
    )
    if "overdue_ratio" in data:
        trajectories = read_csv(history_path / "order_trajectories.csv")
        trajectories.OrderDate = pd.to_datetime(trajectories.OrderDate)
        rebuilt = []
        for (store, date), group in trajectories.groupby(KEYS):
            dates = np.sort(group.OrderDate.unique())
            gaps = np.diff(dates).astype("timedelta64[D]").astype(float)
            value = (
                (pd.Timestamp(date) - pd.Timestamp(dates[-1])).days / gaps.mean()
                if len(gaps)
                else np.nan
            )
            rebuilt.append(dict(StoreID=store, snapshot=date, expected=value))
        checked = data.merge(pd.DataFrame(rebuilt), on=KEYS, validate="one_to_one")
        np.testing.assert_allclose(
            checked.overdue_ratio,
            checked.expected,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    summaries = []
    for spec in protocol["arms"]:
        arm = directory / spec["id"]
        verify_manifest(arm)
        checked_models = 0
        parts = []
        for split in protocol["splits"]:
            train, held = validate_partition(data, split)
            folder = arm / f"repeat_{split['repeat']}/fold_{split['outer_fold']}"
            verify_manifest(folder)
            inner = read_csv(folder / "inner_scores.csv")
            pred = read_csv(folder / "predictions.csv")
            assignment = (
                inner[inner.candidate.eq("baseline")]
                .set_index(KEYS)
                .inner_fold.reindex(pd.MultiIndex.from_frame(train[KEYS]))
            )
            require(assignment.notna().all(), "Incomplete inner population")
            records = json.loads((folder / "model_identities.json").read_text())["fits"]
            for stage, fold in [("inner", i) for i in sorted(assignment.unique())] + [
                ("outer", None)
            ]:
                if stage == "outer":
                    fitting, validation = train, held
                    selected = [r for r in records if r["stage"] == "outer"]
                    model_path = folder / "outer_models.joblib"
                else:
                    fitting = train.iloc[np.flatnonzero(assignment.to_numpy() != fold)]
                    validation = train.iloc[
                        np.flatnonzero(assignment.to_numpy() == fold)
                    ]
                    selected = [
                        r
                        for r in records
                        if r["stage"] == "inner" and r["inner_fold"] == fold
                    ]
                    model_path = folder / f"inner_{fold}_models.joblib"
                require(
                    set(fitting.StoreID).isdisjoint(validation.StoreID),
                    "Reseller leakage",
                )
                content = pd.util.hash_pandas_object(
                    fitting[KEYS + ["churn"] + spec["features"]], index=False
                ).values.tobytes()
                digest = hashlib.sha256(content).hexdigest()
                cells = fitting.groupby(["snapshot", "churn"]).size()
                weights = np.array(
                    [
                        len(fitting) / (8 * cells.loc[(r.snapshot, r.churn)])
                        for r in fitting.itertuples()
                    ]
                )
                models = joblib.load(model_path)
                require(len(models) == len(selected) == 5, "Ensemble count differs")
                scores = []
                for model, record in zip(models, selected):
                    require(
                        record["training_sha256"] == digest,
                        "Fitting feature/label identity differs",
                    )
                    np.testing.assert_array_equal(weights, record["weights"])
                    require(
                        all(
                            model.get_params()[k] == v
                            for k, v in {
                                **spec["parameters"],
                                "random_state": record["seed"],
                            }.items()
                        ),
                        "Model parameters differ",
                    )
                    require(
                        list(model.feature_names_in_) == spec["features"],
                        "Predictor schema differs",
                    )
                    scores.append(
                        model.predict_proba(validation[spec["features"]])[:, 1]
                    )
                    checked_models += 1
                expected = validation[KEYS + ["churn"]].assign(
                    score=np.mean(scores, axis=0)
                )
                saved = (
                    pred[pred.strategy.eq(spec["id"])]
                    if stage == "outer"
                    else inner[
                        inner.candidate.eq(spec["id"]) & inner.inner_fold.eq(fold)
                    ]
                )
                pd.testing.assert_frame_equal(
                    ordered(expected), ordered(saved[KEYS + ["churn", "score"]])
                )
            for name, part in inner.groupby("candidate"):
                cut = threshold(part)
                saved = pred[pred.strategy.eq(name)]
                require(
                    cut is not None and saved.cutoff.eq(cut).all(),
                    "Independent precision40 selection differs/infeasible",
                )
                pd.testing.assert_frame_equal(
                    ordered(part[KEYS + ["churn"]]), ordered(train[KEYS + ["churn"]])
                )
                require(
                    part.groupby("StoreID").inner_fold.nunique().eq(1).all(),
                    "Inner reseller split",
                )
            # Reference scores/cases are byte-preserved numerically; only cutoff policy changes.
            cached = (
                directory.parent
                / "period-weighting-20261018"
                / f"repeat_{split['repeat']}/fold_{split['outer_fold']}"
            )
            base = read_csv(cached / "predictions.csv")
            base = base[base.strategy.eq("period_class_weighted")]
            np.testing.assert_array_equal(
                ordered(base).score, ordered(pred[pred.strategy.eq("baseline")]).score
            )
            parts.append(pred)
        predictions = read_csv(arm / "validation_predictions.csv")
        pd.testing.assert_frame_equal(pd.concat(parts, ignore_index=True), predictions)
        # Direct counts for every fold/repeat/snapshot/activity intersection.
        metrics = pd.read_csv(
            arm / "operating_metrics.csv", float_precision="round_trip"
        )
        for _, row in metrics.iterrows():
            x = predictions[
                predictions.strategy.eq(row.strategy)
                & predictions.repeat.eq(row["repeat"])
            ]
            if row.scope == "outer_fold":
                x = x[x.outer_fold.eq(row.outer_fold)]
            if row["slice"] in ["snapshot", "snapshot_activity"]:
                x = x[x.snapshot.eq(row["group"].split("/")[0])]
            if row["slice"] in ["activity", "snapshot_activity"]:
                x = x[
                    x.n_orders_3m.gt(0).eq(
                        row["group"].split("/")[-1] == "recent_purchase"
                    )
                ]
            y = x.churn.eq(1)
            a = x.score.ge(x.cutoff)
            require(
                [row[k] for k in ["tp", "fp", "fn", "tn"]]
                == [
                    int((y & a).sum()),
                    int((~y & a).sum()),
                    int((y & ~a).sum()),
                    int((~y & ~a).sum()),
                ],
                "Confusion counts differ",
            )
        # Independent within-fold recent AP/AUC point gates.
        rank = []
        for (repeat, fold), x in predictions[predictions.n_orders_3m.gt(0)].groupby(
            ["repeat", "outer_fold"]
        ):
            values = {}
            for name, part in x.groupby("strategy"):
                values[name] = (
                    average_precision_score(part.churn, part.score),
                    roc_auc_score(part.churn, part.score),
                )
            rank.append(
                dict(
                    repeat=repeat,
                    fold=fold,
                    ap=values[spec["id"]][0] - values["baseline"][0],
                    auc=values[spec["id"]][1] - values["baseline"][1],
                )
            )
        rank = pd.DataFrame(rank)
        primary = json.loads((arm / "decision.json").read_text())
        checks = primary["checks"]
        require(
            checks["recent_ap_gain_at_least_one_point"] == bool(rank.ap.mean() >= 0.01),
            "AP practical gate differs",
        )
        require(
            checks["recent_ap_nondecreasing_every_repeat"]
            == bool(rank.groupby("repeat").ap.mean().ge(0).all()),
            "AP repeat gate differs",
        )
        require(
            checks["recent_auc_nondecreasing_mean"] == bool(rank.auc.mean() >= 0),
            "AUC gate differs",
        )
        # Full independent operating bootstrap, preserving reseller/repeat dependence.
        base = predictions[predictions.strategy.eq("baseline")]
        new = predictions[predictions.strategy.eq(spec["id"])]
        pair = base.merge(
            new,
            on=KEYS + ["churn", "repeat", "outer_fold", "n_orders_3m"],
            suffixes=("_base", "_new"),
            validate="one_to_one",
        )
        for name in ["base", "new"]:
            alert = pair["score_" + name].ge(pair["cutoff_" + name])
            pair["tp_" + name] = alert & pair.churn.eq(1)
            pair["fp_" + name] = alert & pair.churn.eq(0)
        pooled = pair.groupby("repeat")[
            ["tp_base", "tp_new", "fp_base", "fp_new", "churn"]
        ].sum()
        require(
            checks["recall_preserved_every_repeat"]
            == bool(pooled.tp_new.ge(pooled.tp_base).all()),
            "Recall point gate differs",
        )
        require(
            checks["fewer_fp_every_repeat"]
            == bool(pooled.fp_new.lt(pooled.fp_base).all()),
            "FP point gate differs",
        )
        require(
            checks["at_least_five_percent_mean_fp_reduction"]
            == bool(
                (pooled.fp_base - pooled.fp_new).mean() / pooled.fp_base.mean() >= 0.05
            ),
            "Practical FP gate differs",
        )
        stores = np.sort(pair.StoreID.unique())
        weights = np.random.default_rng(2026).multinomial(
            len(stores), np.repeat(1 / len(stores), len(stores)), size=5000
        )
        changes = []
        reductions = []
        for _, x in pair.groupby("repeat"):
            c = (
                x.groupby("StoreID")[
                    ["tp_base", "tp_new", "fp_base", "fp_new", "churn"]
                ]
                .sum()
                .reindex(stores, fill_value=0)
                .to_numpy(dtype=float)
            )
            w = weights @ c
            changes.append((w[:, 1] - w[:, 0]) / w[:, 4])
            reductions.append(w[:, 2] - w[:, 3])
        recall = np.quantile(
            np.mean(changes, axis=0),
            [spec["interval_tail_probability"], 1 - spec["interval_tail_probability"]],
        )
        fp = np.quantile(
            np.mean(reductions, axis=0),
            [spec["interval_tail_probability"], 1 - spec["interval_tail_probability"]],
        )
        np.testing.assert_array_equal(recall, primary["recall_difference_interval"])
        np.testing.assert_array_equal(fp, primary["fp_reduction_interval"])
        require(
            checks["recall_uncertainty_lower_nonnegative"] == bool(recall[0] >= 0),
            "Recall uncertainty gate differs",
        )
        require(
            checks["fp_reduction_uncertainty_lower_positive"] == bool(fp[0] > 0),
            "FP uncertainty gate differs",
        )
        require(
            checks["all_recent_fold_rank_metrics_defined"]
            == bool(rank[["ap", "auc"]].notna().all().all()),
            "Undefined ranking gate differs",
        )
        # Recount each period/activity safeguard, including all intersections.
        safeguards = []
        for (repeat, dimension), columns in [
            ((r, dim), cols)
            for r in sorted(pair.repeat.unique())
            for dim, cols in [
                ("activity", ["recent"]),
                ("snapshot", ["snapshot"]),
                ("intersection", ["snapshot", "recent"]),
            ]
        ]:
            subset = pair[pair.repeat.eq(repeat)].assign(
                recent=lambda f: f.n_orders_3m.gt(0)
            )
            for identity, x in subset.groupby(columns):
                positives = int(x.churn.sum())
                delta = (
                    (x.tp_new.sum() - x.tp_base.sum()) / positives
                    if positives
                    else np.nan
                )
                safeguards.append(
                    dict(
                        repeat=repeat,
                        dimension=dimension,
                        group=str(identity),
                        recall=delta,
                        fp=int(x.fp_base.sum() - x.fp_new.sum()),
                    )
                )
        safe = pd.DataFrame(safeguards)
        p = safe[safe.dimension.ne("intersection")]
        i = safe[safe.dimension.eq("intersection")]
        require(
            checks["frozen_period_activity_limits"]
            == bool((p.recall.ge(-0.05) & p.fp.ge(0)).all()),
            "Period safeguard differs",
        )
        require(
            checks["frozen_intersection_limits"]
            == bool((i.recall.ge(-0.05) & i.fp.ge(0)).all()),
            "Intersection safeguard differs",
        )
        require(
            checks["activity_snapshot_mean_recall_loss_at_most_five_points"]
            == bool(p.groupby(["dimension", "group"]).recall.mean().ge(-0.05).all()),
            "Mean subgroup gate differs",
        )
        require(primary["accepted"] == all(checks.values()), "Acceptance differs")
        # Independent sklearn check of the first17 joint ranking bootstrap draws.
        saved = pd.read_csv(arm / "ranking_bootstrap.csv", float_precision="round_trip")
        ap = []
        auc = []
        for draw in range(17):
            da = []
            du = []
            for _, x in predictions[predictions.n_orders_3m.gt(0)].groupby(
                ["repeat", "outer_fold"]
            ):
                values = {}
                for name, part in x.groupby("strategy"):
                    w = weights[draw, np.searchsorted(stores, part.StoreID.to_numpy())]
                    values[name] = (
                        average_precision_score(
                            part.churn, part.score, sample_weight=w
                        ),
                        roc_auc_score(part.churn, part.score, sample_weight=w),
                    )
                da.append(values[spec["id"]][0] - values["baseline"][0])
                du.append(values[spec["id"]][1] - values["baseline"][1])
            ap.append(np.mean(da))
            auc.append(np.mean(du))
        np.testing.assert_allclose(saved.ap_gain.iloc[:17], ap, rtol=0, atol=1e-14)
        np.testing.assert_allclose(saved.auc_gain.iloc[:17], auc, rtol=0, atol=1e-14)
        final = json.loads((arm / "final_decision.json").read_text())
        require(
            final["primary_decision_sha256"] == file_sha256(arm / "decision.json"),
            "Final binding differs",
        )
        if not primary["accepted"]:
            require(
                final["decision"] == primary
                and set(final["stages"].values()) == {"not_run_primary_failed"},
                "Failed primary rescued",
            )
        else:
            raise ValueError(
                "Provisional winner requires independent 50k/exact confirmation audit"
            )
        summaries.append(
            dict(
                candidate=spec["id"],
                status="passed",
                models_reloaded=checked_models,
                cutoffs=30,
                operating_rows=len(metrics),
                operating_bootstrap_draws=5000,
                ranking_bootstrap_draws_spotchecked=17,
                accepted=primary["accepted"],
            )
        )
    destination = directory / "independent_verification.json"
    require(not destination.exists(), "Preserve audit")
    result = dict(
        status="passed",
        arms=summaries,
        models_fitted=0,
        verifier=input_identity({"script": Path(__file__)}),
    )
    write_json(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
