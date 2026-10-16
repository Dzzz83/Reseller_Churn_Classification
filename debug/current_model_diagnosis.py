"""Diagnose development errors and test precision at recall strictly above 90%.

Standalone by design: the experimental branch currently has broken package
imports. No research search code or operational model is changed. Candidate
features and cutoffs are evaluated on development only.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ("2012-07-01", "2012-10-01", "2013-01-01", "2013-04-01")
FEATURES = [
    "n_orders_3m",
    "revenue_3m",
    "recency_days",
    "share_bikes",
    "share_accessories",
    "share_clothing",
    "revenue_12m",
    "weighted_discount_3m",
    "discount_change_3m",
    "discounted_line_share_12m",
]
EXTRA = [
    "revenue_ratio_3m",
    "orders_ratio_3m",
    "overdue_ratio",
    "n_orders_12m",
    "gap_cv",
    "last_order_revenue_ratio",
]
PARAMETERS = dict(
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping=False,
)
SEEDS = (42, 78, 88, 1034, 2026)
MODEL_IDENTITIES = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"
    )


def development_labels(orders, keys, snapshot):
    """Construct labels only for the four approved development snapshots."""
    t = pd.Timestamp(snapshot)
    assert str(t.date()) in SNAPSHOTS, "Only approved development snapshots are allowed"
    future = orders[
        orders.OrderDate.ge(t) & orders.OrderDate.lt(t + pd.DateOffset(months=6))
    ]
    result = keys.copy()
    result["churn"] = (~result.StoreID.isin(future.StoreID)).astype(int)
    return result


def snapshot_features(orders, details, snapshot):
    t = pd.Timestamp(snapshot)
    history = orders[
        orders.OrderDate.ge(t - pd.DateOffset(months=12)) & orders.OrderDate.lt(t)
    ]
    eligible = history.loc[history.OrderDate.ge(t - pd.DateOffset(months=6)), "StoreID"]
    history = history[history.StoreID.isin(eligible)]
    base = history.groupby("StoreID").agg(
        last_order=("OrderDate", "max"),
        revenue_12m=("SubTotal", "sum"),
        n_orders_12m=("SalesOrderID", "nunique"),
        rev_bikes=("rev_bikes", "sum"),
        rev_accessories=("rev_accessories", "sum"),
        rev_clothing=("rev_clothing", "sum"),
    )
    base["snapshot"] = t
    base["recency_days"] = (t - base.last_order).dt.days
    for name in ("bikes", "accessories", "clothing"):
        base["share_" + name] = base["rev_" + name] / base.revenue_12m
    for months in (3, 6):
        recent = history[history.OrderDate.ge(t - pd.DateOffset(months=months))]
        base[f"n_orders_{months}m"] = (
            recent.groupby("StoreID")
            .SalesOrderID.nunique()
            .reindex(base.index, fill_value=0)
        )
        base[f"revenue_{months}m"] = (
            recent.groupby("StoreID").SubTotal.sum().reindex(base.index, fill_value=0)
        )
    rates = []
    for start, end in (
        (t - pd.DateOffset(months=3), t),
        (t - pd.DateOffset(months=6), t - pd.DateOffset(months=3)),
        (t - pd.DateOffset(months=12), t),
    ):
        rows = details[details.OrderDate.ge(start) & details.OrderDate.lt(end)]
        gross = rows.OrderQty * rows.UnitPrice
        rates.append(
            (
                (gross * rows.UnitPriceDiscount).groupby(rows.StoreID).sum()
                / gross.groupby(rows.StoreID).sum(),
                rows.UnitPriceDiscount.gt(0).groupby(rows.StoreID).mean(),
            )
        )
    base["weighted_discount_3m"] = rates[0][0]
    base["discount_change_3m"] = rates[0][0] - rates[1][0]
    base["discounted_line_share_12m"] = rates[2][1]
    dates = (
        history[["StoreID", "OrderDate"]]
        .drop_duplicates()
        .sort_values(["StoreID", "OrderDate"])
    )
    dates["gap"] = dates.groupby("StoreID").OrderDate.diff().dt.days
    gaps = dates.groupby("StoreID").gap.agg(["mean", "std"])
    base["mean_gap"] = gaps["mean"]
    base["overdue_ratio"] = base.recency_days / gaps["mean"].where(gaps["mean"].gt(0))
    base["gap_cv"] = gaps["std"] / gaps["mean"].where(gaps["mean"].gt(0))
    base["previous_3m_revenue"] = base.revenue_6m - base.revenue_3m
    base["previous_3m_orders"] = base.n_orders_6m - base.n_orders_3m
    base["revenue_ratio_3m"] = base.revenue_3m / base.previous_3m_revenue.where(
        base.previous_3m_revenue.gt(0)
    )
    base["orders_ratio_3m"] = base.n_orders_3m / base.previous_3m_orders.where(
        base.previous_3m_orders.gt(0)
    )
    last = (
        history.sort_values(["OrderDate", "SalesOrderID"])
        .groupby("StoreID")
        .tail(1)
        .set_index("StoreID")
    )
    base["last_order_revenue_ratio"] = last.SubTotal / (
        base.revenue_12m / base.n_orders_12m
    )
    return base.reset_index()


def score(train, test, columns):
    weights = compute_sample_weight("balanced", train.churn)
    outputs = []
    for seed in SEEDS:
        model = HistGradientBoostingClassifier(**PARAMETERS, random_state=seed)
        model.fit(train[columns], train.churn.astype(int), sample_weight=weights)
        outputs.append(model.predict_proba(test[columns])[:, 1])
        training_hash = hashlib.sha256(
            pd.util.hash_pandas_object(
                train[["StoreID", "snapshot", "churn", *columns]], index=False
            ).values.tobytes()
        ).hexdigest()
        MODEL_IDENTITIES.append(
            dict(
                features=list(columns),
                seed=seed,
                training_sha256=training_hash,
                fitted_estimator_hash=joblib.hash(model),
            )
        )
    return np.mean(outputs, axis=0), float(np.max(np.ptp(outputs, axis=0)))


def metrics(frame, cutoff=0.06):
    y = frame.churn.eq(1)
    alert = frame.score.ge(cutoff) if np.isscalar(cutoff) else frame.score.ge(cutoff)
    tp, fp, fn, tn = [
        int(v.sum()) for v in (y & alert, ~y & alert, y & ~alert, ~y & ~alert)
    ]
    return dict(
        rows=len(frame),
        churners=int(y.sum()),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        recall=tp / max(1, tp + fn),
        precision=tp / max(1, tp + fp),
        alerts=tp + fp,
        average_precision=float(average_precision_score(y, frame.score))
        if y.any()
        else None,
    )


def select_cutoff(y, scores, minimum_recall=0.90):
    """Maximize precision with recall strictly > target on training OOF only."""
    thresholds = np.unique(np.r_[0.0, scores])
    choices = []
    for threshold in thresholds:
        alert = scores >= threshold
        tp = int(((y == 1) & alert).sum())
        recall = tp / max(1, int((y == 1).sum()))
        precision = tp / max(1, int(alert.sum()))
        if recall > minimum_recall:
            choices.append((precision, recall, float(threshold)))
    if not choices:
        raise ValueError("No feasible training OOF cutoff")
    return max(choices)[2]


def validate_source(orders, headers, lines):
    assert not orders.SalesOrderID.duplicated().any()
    assert not headers.SalesOrderID.duplicated().any()
    assert not lines.SalesOrderDetailID.duplicated().any()
    assert lines.SalesOrderID.isin(headers.SalesOrderID).all()
    history = orders[orders.OrderDate.lt("2013-10-01")]
    assert history.SalesOrderID.isin(headers.SalesOrderID).all()
    headers = headers[headers.SalesOrderID.isin(history.SalesOrderID)]
    lines = lines[lines.SalesOrderID.isin(history.SalesOrderID)]
    joined = history.merge(
        headers, on="SalesOrderID", suffixes=("", "_source"), validate="one_to_one"
    )
    for name in ("StoreID", "OrderDate"):
        np.testing.assert_array_equal(joined[name], joined[name + "_source"])
    np.testing.assert_allclose(
        joined.SubTotal, joined.SubTotal_source, atol=1e-6, rtol=1e-10
    )
    assert lines.UnitPriceDiscount.between(0, 1).all()
    np.testing.assert_allclose(
        lines.OrderQty * lines.UnitPrice * (1 - lines.UnitPriceDiscount),
        lines.LineTotal,
        atol=1e-5,
        rtol=1e-8,
    )
    totals = lines.groupby("SalesOrderID").agg(
        qty=("OrderQty", "sum"),
        n_lines=("SalesOrderDetailID", "size"),
        revenue=("LineTotal", "sum"),
    )
    checked = history.set_index("SalesOrderID").join(totals, rsuffix="_source")
    for name in ("qty", "n_lines"):
        np.testing.assert_array_equal(checked[name], checked[name + "_source"])
    np.testing.assert_allclose(checked.SubTotal, checked.revenue, atol=0.02, rtol=1e-8)
    return lines.merge(
        headers[["SalesOrderID", "StoreID", "OrderDate"]],
        on="SalesOrderID",
        validate="many_to_one",
    )


def classify(frame):
    frame = frame.copy()
    y, a = frame.churn.eq(1), frame.score.ge(0.06)
    frame["result"] = np.select(
        [y & a, y & ~a, ~y & a],
        ["caught_churn", "missed_churn", "false_alert"],
        default="correct_non_churn",
    )
    frame["active"] = frame.n_orders_3m.gt(0)
    return frame


def development_experiment(data, output):
    """Nested cutoff selection, comparing fixed model and behavioral additions."""
    cases, rows, splits = [], [], []
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    evaluations = [
        ("grouped", str(i), tr, va)
        for i, (tr, va) in enumerate(outer.split(data, data.churn, data.StoreID))
    ]
    for snapshot in ("2013-01-01", "2013-04-01"):
        t = pd.Timestamp(snapshot)
        available = data.snapshot.map(lambda d: d + pd.DateOffset(months=6)).le(t)
        evaluations.append(
            (
                "temporal",
                snapshot,
                np.flatnonzero(available),
                np.flatnonzero(data.snapshot.eq(t)),
            )
        )
    for method, fold, ti, vi in evaluations:
        train, valid = data.iloc[ti].copy(), data.iloc[vi].copy()
        assert set(ti).isdisjoint(vi)
        if method == "grouped":
            assert set(train.StoreID).isdisjoint(valid.StoreID)
        else:
            assert (
                train.snapshot.map(lambda d: d + pd.DateOffset(months=6))
                <= valid.snapshot.min()
            ).all()
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=78)
        inner_splits = list(inner.split(train, train.churn, train.StoreID))
        for candidate, columns in (
            ("current", FEATURES),
            ("behavioral", FEATURES + EXTRA),
        ):
            inner_scores = np.full(len(train), np.nan)
            for it, iv in inner_splits:
                assert set(train.iloc[it].StoreID).isdisjoint(train.iloc[iv].StoreID)
                inner_scores[iv], _ = score(train.iloc[it], train.iloc[iv], columns)
            assert np.isfinite(inner_scores).all()
            threshold = select_cutoff(train.churn.to_numpy(), inner_scores)
            scores, seed_spread = score(train, valid, columns)
            for policy, cutoff in (
                ("inner_recall_over_90", threshold),
                ("fixed_006", 0.06),
            ):
                if candidate == "behavioral" and policy == "fixed_006":
                    continue
                saved = valid.copy()
                saved["score"], saved["cutoff"] = scores, cutoff
                saved["method"], saved["fold"] = method, fold
                saved["candidate"], saved["policy"] = candidate, policy
                cases.append(saved)
                for subgroup, subset in (
                    ("overall", saved),
                    ("active", saved[saved.n_orders_3m.gt(0)]),
                    ("inactive", saved[saved.n_orders_3m.eq(0)]),
                ):
                    rows.append(
                        dict(
                            method=method,
                            fold=fold,
                            candidate=candidate,
                            policy=policy,
                            cutoff=cutoff,
                            subgroup=subgroup,
                            **metrics(subset, cutoff),
                        )
                    )
            splits.append(
                dict(
                    method=method,
                    fold=fold,
                    candidate=candidate,
                    train_rows=len(train),
                    validation_rows=len(valid),
                    cutoff=threshold,
                    inner_recall_target="strictly >0.90",
                    max_seed_score_range=seed_spread,
                )
            )
        print("Completed", method, fold, flush=True)
    pd.DataFrame(rows).to_csv(output / "development_metrics.csv", index=False)
    all_cases = pd.concat(cases, ignore_index=True)
    all_cases.to_csv(output / "development_predictions.csv", index=False)
    pooled = []
    for (method, candidate, policy), group in all_cases.groupby(
        ["method", "candidate", "policy"]
    ):
        for subgroup, subset in (
            ("overall", group),
            ("active", group[group.n_orders_3m.gt(0)]),
            ("inactive", group[group.n_orders_3m.eq(0)]),
        ):
            pooled.append(
                dict(
                    method=method,
                    candidate=candidate,
                    policy=policy,
                    subgroup=subgroup,
                    **metrics(subset, subset.cutoff),
                )
            )
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(output / "development_pooled_metrics.csv", index=False)
    write_json(output / "splits.json", splits)
    print(pooled[pooled.subgroup.eq("overall")].to_string(index=False), flush=True)


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    paths = dict(
        orders=ROOT / "datasets/orders_clean.csv",
        headers=args.headers,
        lines=args.lines,
        source_manifest=args.source_manifest,
        reference=ROOT / "datasets/processed/ml_labeled_snapshots.csv",
        script=Path(__file__),
    )
    source = json.loads(args.source_manifest.read_text())
    assert source["coverage"]["from"] <= "2011-07-01"
    assert source["coverage"]["through_exclusive"] >= "2013-10-01"
    assert source["source"]["export_cutoff_exclusive"] == "2013-10-01"
    for name in ("headers", "lines"):
        expected = source["exports"][f"order_{name}.csv"]["sha256"]
        assert sha(paths[name]) == expected, "Source export checksum differs"
    orders = pd.read_csv(paths["orders"], parse_dates=["OrderDate"])
    # No October scoring population or October outcome is constructed or read.
    # This cutoff includes all approved development label windows only.
    orders = orders[orders.OrderDate.lt("2013-10-01")].copy()
    headers = pd.read_csv(paths["headers"], parse_dates=["OrderDate"])
    lines = pd.read_csv(paths["lines"])
    assert headers.OrderDate.lt("2013-10-01").all()
    details = validate_source(orders, headers, lines)
    reference = pd.read_csv(paths["reference"], parse_dates=["snapshot"])
    assert set(reference.snapshot.dt.strftime("%Y-%m-%d")) == set(SNAPSHOTS)
    parts = []
    for snapshot in SNAPSHOTS:
        built = snapshot_features(orders, details, snapshot)
        original = (
            reference[reference.snapshot.eq(pd.Timestamp(snapshot))]
            .sort_values("StoreID")
            .reset_index(drop=True)
        )
        np.testing.assert_array_equal(built.StoreID, original.StoreID)
        for name in FEATURES[:7]:
            np.testing.assert_allclose(
                built[name], original[name], atol=1e-7, rtol=1e-10
            )
        labels = development_labels(orders, built[["StoreID"]], snapshot)
        built["churn"] = labels.churn.to_numpy()
        np.testing.assert_array_equal(built.churn, original.churn)
        parts.append(built)
    development = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["snapshot", "StoreID"])
        .reset_index(drop=True)
    )
    assert len(development) == 1375 and development.churn.sum() == 212
    development.to_csv(output / "development_features.csv", index=False)
    development = pd.read_csv(
        output / "development_features.csv", parse_dates=["snapshot"]
    )
    protocol = dict(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        main_revision=subprocess.check_output(
            ["git", "rev-parse", "main"], text=True
        ).strip(),
        input_hashes={name: sha(path) for name, path in paths.items()},
        runtime=dict(
            python=platform.python_version(),
            sklearn=sklearn.__version__,
            pandas=pd.__version__,
            numpy=np.__version__,
            joblib=joblib.__version__,
        ),
        features=FEATURES,
        candidate_additions=EXTRA,
        parameters=PARAMETERS,
        seeds=SEEDS,
        source_coverage=source["coverage"],
        development_rows=len(development),
        october_test_population_constructed=False,
        october_test_outcomes_used=False,
        independent_evaluation=False,
        outer_grouped_seed=42,
        inner_grouped_seed=78,
        development_selection="Training-only grouped inner OOF: highest precision with recall strictly >90%; ties favor recall then higher cutoff",
        limitations=[
            "Development-only exploratory comparison; no final model selected",
            "October was inspected before this task and earlier in this task; no independent test claim",
            "One outer grouped partition; temporal evaluation has two dates and small training sets",
            "Inner folds mix training periods; outer temporal training labels all mature before validation",
            "The inherited .06 cutoff was historically selected using all development; temporal baseline is retrospective",
            "Source completeness and order-time fact availability remain assumptions",
        ],
    )
    write_json(output / "development_protocol.json", protocol)
    development_experiment(development, output)
    pd.DataFrame(MODEL_IDENTITIES).to_json(
        output / "model_identities.json", orient="records", indent=2
    )
    cases = pd.read_csv(output / "development_predictions.csv")
    current = cases[
        cases.method.eq("grouped")
        & cases.candidate.eq("current")
        & cases.policy.eq("fixed_006")
    ].copy()
    current = classify(current)
    current.to_csv(output / "current_development_cases.csv", index=False)
    current[current.result.eq("missed_churn")].to_csv(
        output / "development_missed_churners.csv", index=False
    )
    columns = ["score", *FEATURES, *EXTRA, "previous_3m_revenue", "mean_gap"]
    profiles = []
    for (active, result), group in current.groupby(["active", "result"]):
        for feature in columns:
            values = group[feature]
            profiles.append(
                dict(
                    active=bool(active),
                    result=result,
                    rows=len(group),
                    feature=feature,
                    missing=int(values.isna().sum()),
                    median=values.median(),
                    q25=values.quantile(0.25),
                    q75=values.quantile(0.75),
                )
            )
    pd.DataFrame(profiles).to_csv(output / "development_profiles.csv", index=False)
    write_json(
        output / "manifest.json",
        {
            **protocol,
            "outputs": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--lines", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
