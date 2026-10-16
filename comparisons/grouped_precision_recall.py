"""Repeated nested grouped development study; never opens final-test data."""

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

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
SPENDING = ["revenue_ratio_3m", "orders_ratio_3m"]
CADENCE = ["overdue_ratio", "n_orders_12m", "gap_cv"]
EXTRA = SPENDING + CADENCE + ["last_order_revenue_ratio"]
MODEL_SEEDS = (42, 78, 88, 1034, 2026)
REPEAT_SEEDS = (42, 78, 88)
BASE_PARAMETERS = dict(
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping=False,
)
KEYS = ["StoreID", "snapshot"]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def candidates():
    result = []
    for name, additions in (
        ("original", []),
        ("spending", SPENDING),
        ("cadence", CADENCE),
        ("full", EXTRA),
    ):
        for regularization in ("baseline", "regularized"):
            parameters = dict(BASE_PARAMETERS)
            if regularization == "regularized":
                parameters.update(max_leaf_nodes=7, l2_regularization=1.0)
            result.append(
                dict(
                    id=f"{name}_{regularization}",
                    features=FEATURES + additions,
                    parameters=parameters,
                )
            )
    return result


def validate_development(frame, canonical=True):
    required = [*KEYS, "churn", *FEATURES, *EXTRA]
    if set(required) - set(frame):
        raise ValueError("Missing development columns")
    data = frame[required].copy()
    data["snapshot"] = pd.to_datetime(data.snapshot, errors="raise")
    if (
        data.snapshot.dt.tz is not None
        or not data.snapshot.eq(data.snapshot.dt.normalize()).all()
    ):
        raise ValueError("Invalid snapshot dates")
    if set(data.snapshot.dt.strftime("%Y-%m-%d")) != set(SNAPSHOTS):
        raise ValueError("Only the four approved development snapshots are permitted")
    if data[KEYS].isna().any().any() or data.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate development keys")
    if not data.churn.isin([0, 1]).all() or data.churn.nunique() != 2:
        raise ValueError("Development requires both binary outcome classes")
    values = data[[*FEATURES, *EXTRA]].to_numpy(dtype=float)
    if (
        np.isinf(values).any()
        or not np.isfinite(data[FEATURES[:7]].to_numpy(dtype=float)).all()
    ):
        raise ValueError(
            "Invalid features; missing discounts/undefined ratios remain native missing"
        )
    if (
        not data.StoreID.gt(0).all()
        or not data.StoreID.eq(np.floor(data.StoreID)).all()
    ):
        raise ValueError("StoreID must be a positive integer")
    if canonical and (
        len(data) != 1375 or data.churn.sum() != 212 or data.StoreID.nunique() != 422
    ):
        raise ValueError("Unexpected canonical development population")
    return data.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)


def grouped_splits(data, count, seed):
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    result = list(splitter.split(data, data.churn, data.StoreID))
    coverage = np.zeros(len(data), dtype=int)
    for train, valid in result:
        if set(data.iloc[train].StoreID) & set(data.iloc[valid].StoreID):
            raise ValueError("Reseller overlap across training and validation")
        coverage[valid] += 1
        if data.iloc[train].churn.nunique() != 2:
            raise ValueError("Training fold lacks both classes")
    if not np.all(coverage == 1):
        raise ValueError("Validation folds must cover each row once")
    return result


def decision_metrics(y, score, cutoff):
    y = np.asarray(y) == 1
    alert = np.asarray(score) >= np.asarray(cutoff)
    tp, fp, fn, tn = [
        int(v.sum()) for v in (y & alert, ~y & alert, y & ~alert, ~y & ~alert)
    ]
    return dict(
        rows=len(y),
        churners=int(y.sum()),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        recall=tp / max(1, tp + fn),
        precision=tp / max(1, tp + fp),
        alerts=tp + fp,
        average_precision=float(average_precision_score(y, score)) if y.any() else None,
    )


def select_threshold(y, score):
    """Training OOF only; strict >90% recall, ties precision/recall/cutoff."""
    y, score = np.asarray(y), np.asarray(score)
    if (
        not np.isfinite(score).all()
        or not np.isin(y, [0, 1]).all()
        or not (y == 1).any()
    ):
        raise ValueError("Invalid training OOF scores or labels")
    options = []
    for cutoff in np.unique(np.r_[0.0, score]):
        alert = score >= cutoff
        caught = int(((y == 1) & alert).sum())
        recall = caught / int((y == 1).sum())
        precision = caught / max(1, int(alert.sum()))
        if recall > 0.90:
            options.append((precision, recall, float(cutoff)))
    precision, recall, cutoff = max(options)
    return dict(precision=precision, recall=recall, cutoff=cutoff)


def choose_configuration(selection_rows):
    """Use inner results only; prefer fewer features, then stable config ID."""
    feasible = [row for row in selection_rows if row["recall"] > 0.90]
    if not feasible:
        raise ValueError("No inner configuration meets the recall target")
    return sorted(
        feasible,
        key=lambda row: (
            -row["precision"],
            -row["recall"],
            row["feature_count"],
            row["candidate"],
        ),
    )[0]


def fit_scores(
    train, validation_features, candidate, seeds=MODEL_SEEDS, retain_models=False
):
    """No validation outcomes enter fitting; weights derive from train only."""
    columns = candidate["features"]
    weights = compute_sample_weight("balanced", train.churn)
    content = pd.util.hash_pandas_object(
        train[[*KEYS, "churn", *columns]], index=False
    ).values.tobytes()
    training_hash = hashlib.sha256(content).hexdigest()
    predictions, identities, models = [], [], []
    with threadpool_limits(limits=1):
        for seed in seeds:
            model = HistGradientBoostingClassifier(
                **candidate["parameters"], random_state=seed
            )
            model.fit(train[columns], train.churn.astype(int), sample_weight=weights)
            predictions.append(model.predict_proba(validation_features[columns])[:, 1])
            identities.append(
                dict(
                    candidate=candidate["id"],
                    seed=seed,
                    training_sha256=training_hash,
                    estimator_hash=joblib.hash(model),
                )
            )
            if retain_models:
                models.append(model)
    spread = float(np.ptp(predictions, axis=0).max())
    return np.mean(predictions, axis=0), identities, spread, models


def inner_selection(train, grid, inner_count, inner_seed, seeds):
    splits = grouped_splits(train, inner_count, inner_seed)
    rows, identities, saved_scores = [], [], []
    assignment = np.full(len(train), -1)
    for inner_fold, (_, valid_indices) in enumerate(splits):
        assignment[valid_indices] = inner_fold
    for candidate in grid:
        scores = np.full(len(train), np.nan)
        for inner_fold, (ti, vi) in enumerate(splits):
            predicted, fitted, spread, _ = fit_scores(
                train.iloc[ti], train.iloc[vi][candidate["features"]], candidate, seeds
            )
            scores[vi] = predicted
            identities.extend(
                [
                    {
                        **item,
                        "stage": "inner",
                        "inner_fold": inner_fold,
                        "seed_score_range": spread,
                    }
                    for item in fitted
                ]
            )
        threshold = select_threshold(train.churn.to_numpy(), scores)
        rows.append(
            dict(
                candidate=candidate["id"],
                feature_count=len(candidate["features"]),
                **threshold,
            )
        )
        saved = train[KEYS + ["churn"]].copy()
        saved["candidate"], saved["score"], saved["inner_fold"] = (
            candidate["id"],
            scores,
            assignment,
        )
        saved_scores.append(saved)
    return rows, identities, pd.concat(saved_scores, ignore_index=True)


def evaluate_outer_fold(
    data, repeat, fold, ti, vi, directory, grid, inner_count, seeds
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    train, validation = data.iloc[ti].copy(), data.iloc[vi].copy()
    if set(train.StoreID) & set(validation.StoreID):
        raise ValueError("Outer reseller overlap")
    selections, identities, inner_scores = inner_selection(
        train, grid, inner_count, 78, seeds
    )
    selected = choose_configuration(selections)
    predictions, selected_predictions = [], None
    for candidate in grid:
        cutoff = next(
            row["cutoff"] for row in selections if row["candidate"] == candidate["id"]
        )
        scores, fitted, spread, _ = fit_scores(
            train, validation[candidate["features"]], candidate, seeds
        )
        identities.extend(
            [{**item, "stage": "outer", "seed_score_range": spread} for item in fitted]
        )
        saved = validation[KEYS + ["churn", "n_orders_3m"]].copy()
        saved["score"], saved["cutoff"], saved["strategy"] = (
            scores,
            cutoff,
            candidate["id"],
        )
        predictions.append(saved)
        if candidate["id"] == selected["candidate"]:
            selected_predictions = saved.copy()
            selected_predictions["strategy"] = "adaptive_selection"
        if candidate["id"] == "original_baseline":
            fixed = saved.copy()
            fixed["strategy"], fixed["cutoff"] = "original_fixed_006", 0.06
            predictions.append(fixed)
    predictions.append(selected_predictions)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions["repeat"], all_predictions["outer_fold"] = repeat, fold
    all_predictions.to_csv(directory / "predictions.csv", index=False)
    inner_scores.to_csv(directory / "inner_scores.csv", index=False)
    pd.DataFrame(selections).to_csv(directory / "inner_selections.csv", index=False)
    write_json(directory / "selected.json", selected)
    write_json(directory / "model_identities.json", identities)
    write_json(
        directory / "split.json",
        dict(
            repeat=repeat,
            outer_fold=fold,
            train_rows=ti.tolist(),
            validation_rows=vi.tolist(),
            train_store_ids=sorted(train.StoreID.unique().tolist()),
            validation_store_ids=sorted(validation.StoreID.unique().tolist()),
        ),
    )
    write_json(
        directory / "complete.json",
        dict(
            fits=len(identities),
            outputs={p.name: file_hash(p) for p in directory.iterdir() if p.is_file()},
        ),
    )
    return dict(
        repeat=repeat,
        fold=fold,
        selected=selected["candidate"],
        fits=len(identities),
        directory=str(directory),
    )


def summarize(predictions):
    fold_rows, repeat_rows = [], []
    for keys, group in predictions.groupby(["repeat", "outer_fold", "strategy"]):
        for subgroup, subset in (
            ("overall", group),
            ("active", group[group.n_orders_3m.gt(0)]),
            ("inactive", group[group.n_orders_3m.eq(0)]),
        ):
            fold_rows.append(
                dict(
                    repeat=keys[0],
                    outer_fold=keys[1],
                    strategy=keys[2],
                    subgroup=subgroup,
                    **decision_metrics(subset.churn, subset.score, subset.cutoff),
                )
            )
    for keys, group in predictions.groupby(["repeat", "strategy"]):
        for subgroup, subset in (
            ("overall", group),
            ("active", group[group.n_orders_3m.gt(0)]),
            ("inactive", group[group.n_orders_3m.eq(0)]),
        ):
            repeat_rows.append(
                dict(
                    repeat=keys[0],
                    strategy=keys[1],
                    subgroup=subgroup,
                    **decision_metrics(subset.churn, subset.score, subset.cutoff),
                )
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(repeat_rows)


def cluster_bootstrap(predictions, draws=5000, seed=2026):
    """Paired whole-reseller draws shared across snapshots and repetitions."""
    arms = ["original_baseline", "adaptive_selection"]
    stores = sorted(predictions.StoreID.unique())
    repeats = sorted(predictions.repeat.unique())
    store_index = {store: i for i, store in enumerate(stores)}
    totals = np.zeros((len(stores), len(repeats), 2, 4), dtype=int)
    for ri, repeat in enumerate(repeats):
        for ai, arm in enumerate(arms):
            group = predictions[
                predictions.repeat.eq(repeat) & predictions.strategy.eq(arm)
            ]
            y, alert = (
                group.churn.eq(1).to_numpy(),
                group.score.ge(group.cutoff).to_numpy(),
            )
            codes = np.where(y & alert, 0, np.where(~y & alert, 1, np.where(y, 2, 3)))
            indices = group.StoreID.map(store_index).to_numpy()
            np.add.at(totals[:, ri, ai, :], (indices, codes), 1)
    rng = np.random.default_rng(seed)
    deltas = np.zeros((draws, 2))
    for draw in range(draws):
        counts = totals[rng.integers(0, len(stores), len(stores))].sum(axis=0)
        precision = counts[:, :, 0] / np.maximum(1, counts[:, :, 0] + counts[:, :, 1])
        recall = counts[:, :, 0] / np.maximum(1, counts[:, :, 0] + counts[:, :, 2])
        deltas[draw] = [
            (precision[:, 1] - precision[:, 0]).mean(),
            (recall[:, 1] - recall[:, 0]).mean(),
        ]
    return dict(
        draws=draws,
        seed=seed,
        cluster_unit="StoreID; all snapshots and repetitions sampled jointly",
        primary_comparison="adaptive_selection minus original_baseline with training-selected cutoffs",
        precision_delta_95=np.quantile(deltas[:, 0], [0.025, 0.975]).tolist(),
        recall_delta_95=np.quantile(deltas[:, 1], [0.025, 0.975]).tolist(),
        interpretation="Marginal percentile intervals conditional on saved fits/cutoffs; no retraining or reselection uncertainty; repetitions are dependent",
    )


def synthetic_development():
    rng = np.random.default_rng(77)
    rows = []
    for store in range(1, 81):
        for snapshot in SNAPSHOTS:
            row = dict(StoreID=store, snapshot=snapshot, churn=int(rng.random() < 0.25))
            row.update({name: float(rng.random()) for name in FEATURES + EXTRA})
            row["n_orders_3m"] = int(rng.integers(0, 4))
            row["weighted_discount_3m"] = (
                np.nan if row["n_orders_3m"] == 0 else row["weighted_discount_3m"]
            )
            rows.append(row)
    return validate_development(pd.DataFrame(rows), canonical=False)


def run(args):
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; evidence is immutable")
    grid = candidates()
    repeats, outer_count, inner_count = REPEAT_SEEDS, 5, 3
    if args.smoke:
        data = synthetic_development()
        grid, repeats, outer_count, inner_count = [grid[0], grid[-1]], (42,), 2, 2
        inputs = {"synthetic_seed": 77}
    else:
        if args.development is None or args.preparation_manifest is None:
            raise ValueError(
                "Explicit development CSV and preparation manifest required"
            )
        preparation = json.loads(args.preparation_manifest.read_text())
        if preparation.get("october_test_outcomes_used") is not False:
            raise ValueError("Preparation must declare no final-test outcomes used")
        if file_hash(args.development) != preparation["outputs"][args.development.name]:
            raise ValueError("Development CSV checksum differs from preparation")
        # Preserve the historically specified training CSV parser.
        data = validate_development(pd.read_csv(args.development))
        inputs = {
            "development": str(args.development.resolve()),
            "development_sha256": file_hash(args.development),
            "preparation_manifest": str(args.preparation_manifest.resolve()),
            "preparation_manifest_sha256": file_hash(args.preparation_manifest),
        }
    args.output.mkdir(parents=True)
    protocol = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        main_revision=subprocess.check_output(
            ["git", "rev-parse", "main"], text=True
        ).strip(),
        code_sha256=file_hash(__file__),
        inputs=inputs,
        grid=grid,
        model_seeds=MODEL_SEEDS,
        repeat_seeds=repeats,
        outer_folds=outer_count,
        inner_folds=inner_count,
        inner_seed=78,
        rows=len(data),
        churners=int(data.churn.sum()),
        resellers=int(data.StoreID.nunique()),
        snapshots=list(SNAPSHOTS),
        objective="Maximize inner pooled precision with recall strictly >0.90",
        threshold_tie_break="precision, recall, higher cutoff",
        configuration_tie_break="precision, recall, fewer features, lexicographic candidate ID",
        primary_comparison="Adaptive inner selection procedure versus original_baseline under the same cutoff selection rule",
        eligibility="Adaptive outer pooled recall >0.90 in every repetition and higher mean repeat precision than baseline",
        strong_support="Eligibility plus positive lower 95% paired reseller-cluster precision-difference bound",
        bootstrap_draws=5000 if not args.smoke else 100,
        bootstrap_seed=2026,
        test_data_used=False,
        temporal_validation_used=False,
        smoke=args.smoke,
        limitations=[
            "Exploratory development reuse: features and previous split results have been inspected",
            "Grouped generalization to held-out reseller identities; not a future-period claim",
            "Bootstrap conditions on fitted models/cutoffs; repeats dependent",
            "No final-test access, model deployment, or operational configuration change",
        ],
    )
    write_json(args.output / "protocol.json", protocol)
    data.to_csv(args.output / "development_input.csv", index=False)
    tasks = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for repeat in repeats:
            for fold, (ti, vi) in enumerate(grouped_splits(data, outer_count, repeat)):
                directory = args.output / f"repeat_{repeat}" / f"fold_{fold}"
                tasks.append(
                    executor.submit(
                        evaluate_outer_fold,
                        data,
                        repeat,
                        fold,
                        ti,
                        vi,
                        directory,
                        grid,
                        inner_count,
                        MODEL_SEEDS,
                    )
                )
        completed = []
        for task in as_completed(tasks):
            result = task.result()
            completed.append(result)
            print(
                f"Completed {len(completed)}/{len(tasks)} outer folds: repeat {result['repeat']}, fold {result['fold']}, selected {result['selected']}",
                flush=True,
            )
    predictions = pd.concat(
        [read_csv(Path(item["directory"]) / "predictions.csv") for item in completed],
        ignore_index=True,
    )
    predictions = predictions.sort_values(
        ["repeat", "outer_fold", "strategy", "snapshot", "StoreID"]
    ).reset_index(drop=True)
    predictions.to_csv(args.output / "validation_predictions.csv", index=False)
    fold_metrics, repeat_metrics = summarize(predictions)
    fold_metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    repeat_metrics.to_csv(args.output / "repeat_metrics.csv", index=False)
    overall = repeat_metrics[repeat_metrics.subgroup.eq("overall")]
    adaptive = overall[overall.strategy.eq("adaptive_selection")].set_index("repeat")
    baseline = overall[overall.strategy.eq("original_baseline")].set_index("repeat")
    uncertainty = cluster_bootstrap(predictions, draws=protocol["bootstrap_draws"])
    write_json(args.output / "bootstrap.json", uncertainty)
    mean_precision_gain = float((adaptive.precision - baseline.precision).mean())
    eligible = bool(adaptive.recall.gt(0.9).all() and mean_precision_gain > 0)
    strong_support = bool(eligible and uncertainty["precision_delta_95"][0] > 0)
    decision = dict(
        eligible=eligible,
        strong_support=strong_support,
        mean_precision_gain=mean_precision_gain,
        all_repetitions_above_90_recall=bool(adaptive.recall.gt(0.9).all()),
        test_evaluated=False,
        model_frozen=False,
    )
    # Final full-development selection follows the identical inner procedure,
    # only when the predeclared evidence criteria support a replacement.
    if strong_support and not args.smoke:
        selections, identities, inner_scores = inner_selection(
            data, grid, inner_count, 78, MODEL_SEEDS
        )
        selected = choose_configuration(selections)
        selected_candidate = next(
            item for item in grid if item["id"] == selected["candidate"]
        )
        _, fitted, spread, models = fit_scores(
            data,
            data[selected_candidate["features"]].iloc[:1],
            selected_candidate,
            MODEL_SEEDS,
            retain_models=True,
        )
        frozen = dict(
            candidate=selected_candidate,
            cutoff=selected["cutoff"],
            seeds=MODEL_SEEDS,
            training_sha256=inputs["development_sha256"],
            protocol_sha256=file_hash(args.output / "protocol.json"),
            fitted_models=fitted,
            seed_score_range_probe=spread,
            test_evaluated=False,
        )
        joblib.dump(
            dict(models=models, specification=frozen),
            args.output / "research_candidate.joblib",
        )
        write_json(args.output / "frozen_candidate.json", frozen)
        write_json(args.output / "final_training_identities.json", identities + fitted)
        pd.DataFrame(selections).to_csv(
            args.output / "final_inner_selections.csv", index=False
        )
        inner_scores.to_csv(args.output / "final_inner_scores.csv", index=False)
        decision["model_frozen"] = True
    write_json(args.output / "decision.json", decision)
    write_json(args.output / "completed_folds.json", completed)
    write_json(
        args.output / "manifest.json",
        {
            "protocol_sha256": file_hash(args.output / "protocol.json"),
            "outer_fit_count": sum(item["fits"] for item in completed),
            "outputs": {
                str(p.relative_to(args.output)): file_hash(p)
                for p in sorted(args.output.rglob("*"))
                if p.is_file()
            },
        },
    )
    print(
        overall[
            overall.strategy.isin(
                ["original_baseline", "full_baseline", "adaptive_selection"]
            )
        ].to_string(index=False),
        flush=True,
    )
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--preparation-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
