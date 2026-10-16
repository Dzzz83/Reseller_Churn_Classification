"""Controlled component ablations on repeated grouped development validation."""

import argparse
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from comparisons import grouped_precision_recall as study

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ("overdue_ratio", "n_orders_12m", "gap_cv", "seven_leaves", "l2_one")
REUSE = {
    0: "original_baseline",
    7: "cadence_baseline",
    24: "original_regularized",
    31: "cadence_regularized",
}
MASKS = sorted(
    {0, 7, 24, 31, *(1 << i for i in range(5)), *(31 ^ (1 << i) for i in range(5))}
)


def configuration(mask):
    parameters = dict(study.BASE_PARAMETERS)
    if mask & 8:
        parameters["max_leaf_nodes"] = 7
    if mask & 16:
        parameters["l2_regularization"] = 1.0
    additions = [name for i, name in enumerate(FACTORS[:3]) if mask & (1 << i)]
    return dict(
        id=f"f{mask:02d}",
        mask=mask,
        features=study.FEATURES + additions,
        parameters=parameters,
    )


def contrasts():
    result = []
    for i, factor in enumerate(FACTORS):
        bit = 1 << i
        result.append(
            dict(
                id=f"add_{factor}",
                kind="single_addition",
                weights={f"f{bit:02d}": 1.0, "f00": -1.0},
            )
        )
        result.append(
            dict(
                id=f"remove_{factor}",
                kind="benefit_of_restoring_removed_component",
                weights={"f31": 1.0, f"f{31 ^ bit:02d}": -1.0},
            )
        )
    for name, positive, negative in (
        ("cadence_alone", 7, 0),
        ("regularization_alone", 24, 0),
        ("cadence_with_regularization", 31, 24),
        ("regularization_with_cadence", 31, 7),
    ):
        result.append(
            dict(
                id=name,
                kind="bundle",
                weights={f"f{positive:02d}": 1.0, f"f{negative:02d}": -1.0},
            )
        )
    result.extend(
        [
            dict(
                id="cadence_regularization_interaction",
                kind="interaction",
                weights={"f31": 1.0, "f07": -1.0, "f24": -1.0, "f00": 1.0},
            ),
            dict(
                id="leaves_l2_interaction_without_cadence",
                kind="interaction",
                weights={"f24": 1.0, "f08": -1.0, "f16": -1.0, "f00": 1.0},
            ),
            dict(
                id="leaves_l2_interaction_with_cadence",
                kind="interaction",
                weights={"f31": 1.0, "f15": -1.0, "f23": -1.0, "f07": 1.0},
            ),
        ]
    )
    return result


def equivalent(left, right):
    return (
        left["features"] == right["features"]
        and left["parameters"] == right["parameters"]
    )


def validate_reuse_source(source):
    protocol = json.loads((source / "protocol.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if (
        protocol["test_data_used"]
        or protocol["temporal_validation_used"]
        or protocol["smoke"]
    ):
        raise ValueError("Expected the completed grouped development study")
    if protocol["code_sha256"] != study.file_hash(Path(study.__file__)):
        raise ValueError("Shared fitting/selection code differs from prior study")
    if (
        protocol["repeat_seeds"] != list(study.REPEAT_SEEDS)
        or protocol["model_seeds"] != list(study.MODEL_SEEDS)
        or protocol["outer_folds"] != 5
        or protocol["inner_folds"] != 3
        or protocol["inner_seed"] != 78
    ):
        raise ValueError("Prior split/model protocol is incompatible")
    for name, expected in manifest["outputs"].items():
        path = (source / name).resolve()
        if (
            not path.is_relative_to(source.resolve())
            or study.file_hash(path) != expected
        ):
            raise ValueError(f"Prior artifact integrity failed: {name}")
    for mask, name in REUSE.items():
        previous = next(item for item in protocol["grid"] if item["id"] == name)
        if not equivalent(configuration(mask), previous):
            raise ValueError("Reused configuration is not identical")
    development = Path(protocol["inputs"]["development"])
    if study.file_hash(development) != protocol["inputs"]["development_sha256"]:
        raise ValueError("Development input changed")
    return study.validate_development(pd.read_csv(development)), protocol


def evaluate_fold(data, repeat, fold, ti, vi, source_folder, directory, inner_count):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source_folder = Path(source_folder)
    prior_split = json.loads((source_folder / "split.json").read_text())
    if (
        prior_split["train_rows"] != ti.tolist()
        or prior_split["validation_rows"] != vi.tolist()
    ):
        raise ValueError("Reused outer fold differs")
    train, validation = data.iloc[ti].copy(), data.iloc[vi].copy()
    if set(train.StoreID) & set(validation.StoreID):
        raise ValueError("Outer reseller overlap")
    prior_predictions = study.read_csv(source_folder / "predictions.csv")
    prior_inner = study.read_csv(source_folder / "inner_scores.csv")
    prior_choices = study.read_csv(source_folder / "inner_selections.csv")
    prior_identities = json.loads((source_folder / "model_identities.json").read_text())
    reused_predictions, reused_inner, reused_choices, identities = [], [], [], []
    for mask, name in REUSE.items():
        candidate = configuration(mask)
        predicted = prior_predictions[prior_predictions.strategy.eq(name)].copy()
        predicted["strategy"] = candidate["id"]
        inner = prior_inner[prior_inner.candidate.eq(name)].copy()
        inner["candidate"] = candidate["id"]
        chosen = prior_choices[prior_choices.candidate.eq(name)].copy()
        chosen["candidate"] = candidate["id"]
        reused_predictions.append(predicted)
        reused_inner.append(inner)
        reused_choices.extend(chosen.to_dict("records"))
        identities.extend(
            [
                {
                    **item,
                    "candidate": candidate["id"],
                    "source_candidate": name,
                    "reused": True,
                }
                for item in prior_identities
                if item["candidate"] == name
            ]
        )
    grid = [configuration(mask) for mask in MASKS if mask not in REUSE]
    selections, new_identities, inner_scores = study.inner_selection(
        train, grid, inner_count, 78, study.MODEL_SEEDS
    )
    identities.extend([{**item, "reused": False} for item in new_identities])
    predictions = []
    for candidate in grid:
        cutoff = next(
            item["cutoff"]
            for item in selections
            if item["candidate"] == candidate["id"]
        )
        scores, fitted, spread, _ = study.fit_scores(
            train, validation[candidate["features"]], candidate, study.MODEL_SEEDS
        )
        identities.extend(
            [
                {**item, "stage": "outer", "seed_score_range": spread, "reused": False}
                for item in fitted
            ]
        )
        saved = validation[study.KEYS + ["churn", "n_orders_3m"]].copy()
        saved["strategy"], saved["score"], saved["cutoff"] = (
            candidate["id"],
            scores,
            cutoff,
        )
        predictions.append(saved)
    predictions = pd.concat([*reused_predictions, *predictions], ignore_index=True)
    predictions["snapshot"] = pd.to_datetime(
        predictions.snapshot, format="mixed"
    ).dt.strftime("%Y-%m-%d")
    predictions["repeat"], predictions["outer_fold"] = repeat, fold
    predictions.to_csv(directory / "predictions.csv", index=False)
    combined_inner = pd.concat([*reused_inner, inner_scores], ignore_index=True)
    combined_inner["snapshot"] = pd.to_datetime(
        combined_inner.snapshot, format="mixed"
    ).dt.strftime("%Y-%m-%d")
    combined_inner.to_csv(directory / "inner_scores.csv", index=False)
    pd.DataFrame([*reused_choices, *selections]).to_csv(
        directory / "inner_selections.csv", index=False
    )
    study.write_json(directory / "model_identities.json", identities)
    study.write_json(directory / "split.json", prior_split)
    complete = dict(
        new_fits=sum(not item["reused"] for item in identities),
        reused_fits=sum(item["reused"] for item in identities),
        outputs={
            p.name: study.file_hash(p) for p in directory.iterdir() if p.is_file()
        },
    )
    study.write_json(directory / "complete.json", complete)
    return dict(
        repeat=repeat,
        fold=fold,
        directory=str(directory),
        **{key: complete[key] for key in ("new_fits", "reused_fits")},
    )


def counts_tensor(predictions, arms, repeats):
    stores = sorted(predictions.StoreID.unique())
    index = {store: i for i, store in enumerate(stores)}
    totals = np.zeros((len(stores), len(repeats), len(arms), 3), dtype=float)
    for ri, repeat in enumerate(repeats):
        for ai, arm in enumerate(arms):
            part = predictions[
                predictions.repeat.eq(repeat) & predictions.strategy.eq(arm)
            ]
            y = part.churn.eq(1).to_numpy()
            alerted = part.score.ge(part.cutoff).to_numpy()
            codes = np.where(
                y & alerted, 0, np.where(~y & alerted, 1, np.where(y, 2, -1))
            )
            included = codes >= 0
            np.add.at(
                totals[:, ri, ai, :],
                (part.StoreID.map(index).to_numpy()[included], codes[included]),
                1,
            )
    return totals


def contrast_statistics(predictions, definitions, draws=50000, seed=2026):
    arms = sorted(predictions.strategy.unique())
    repeats = sorted(predictions.repeat.unique())
    totals = counts_tensor(predictions, arms, repeats)
    matrix = np.zeros((len(arms), len(definitions)))
    for ci, contrast in enumerate(definitions):
        for arm, weight in contrast["weights"].items():
            matrix[arms.index(arm), ci] = weight
    point = totals.sum(axis=0)
    p = point[:, :, 0] / np.maximum(1, point[:, :, 0] + point[:, :, 1])
    r = point[:, :, 0] / np.maximum(1, point[:, :, 0] + point[:, :, 2])
    point_precision = p.mean(axis=0) @ matrix
    point_recall = r.mean(axis=0) @ matrix
    flat = totals.reshape(len(totals), -1)
    rng = np.random.default_rng(seed)
    precision_draws, recall_draws = [], []
    with threadpool_limits(limits=1):
        for start in range(0, draws, 500):
            count = min(500, draws - start)
            weights = rng.multinomial(
                len(totals), np.full(len(totals), 1 / len(totals)), size=count
            )
            sampled = (weights @ flat).reshape(count, len(repeats), len(arms), 3)
            bp = sampled[:, :, :, 0] / np.maximum(
                1, sampled[:, :, :, 0] + sampled[:, :, :, 1]
            )
            br = sampled[:, :, :, 0] / np.maximum(
                1, sampled[:, :, :, 0] + sampled[:, :, :, 2]
            )
            precision_draws.append(bp.mean(axis=1) @ matrix)
            recall_draws.append(br.mean(axis=1) @ matrix)
    precision_draws = np.concatenate(precision_draws)
    recall_draws = np.concatenate(recall_draws)
    tail = 0.05 / (2 * len(definitions))
    simultaneous = np.quantile(precision_draws, [tail, 1 - tail], axis=0)
    marginal = np.quantile(precision_draws, [0.025, 0.975], axis=0)
    recall_interval = np.quantile(recall_draws, [0.025, 0.975], axis=0)
    rows = []
    for ci, contrast in enumerate(definitions):
        rows.append(
            dict(
                contrast=contrast["id"],
                kind=contrast["kind"],
                precision_delta=point_precision[ci],
                recall_delta=point_recall[ci],
                precision_marginal_low=marginal[0, ci],
                precision_marginal_high=marginal[1, ci],
                precision_family_low=simultaneous[0, ci],
                precision_family_high=simultaneous[1, ci],
                recall_marginal_low=recall_interval[0, ci],
                recall_marginal_high=recall_interval[1, ci],
                supports_positive_precision=bool(simultaneous[0, ci] > 0),
            )
        )
    return pd.DataFrame(rows)


def summarize_configs(predictions):
    fold, repeat = study.summarize(predictions)
    overall = repeat[repeat.subgroup.eq("overall")]
    rows = []
    for arm, group in overall.groupby("strategy"):
        candidate = configuration(int(arm[1:]))
        rows.append(
            dict(
                configuration=arm,
                changes=" + ".join(
                    name
                    for i, name in enumerate(FACTORS)
                    if candidate["mask"] & (1 << i)
                )
                or "baseline",
                recall=group.recall.mean(),
                precision=group.precision.mean(),
                average_precision=group.average_precision.mean(),
                tp=group.tp.mean(),
                fp=group.fp.mean(),
                repeats_above_90=int(group.recall.gt(0.9).sum()),
                minimum_repeat_recall=group.recall.min(),
            )
        )
    return fold, repeat, pd.DataFrame(rows)


def validate_predictions(predictions, data, repeats):
    if set(predictions.snapshot) != set(study.SNAPSHOTS):
        raise ValueError("Unexpected validation snapshot")
    if predictions.duplicated(["repeat", "strategy", *study.KEYS]).any():
        raise ValueError("Repeated validation key")
    expected = set(
        map(
            tuple,
            data[study.KEYS]
            .assign(snapshot=data.snapshot.dt.strftime("%Y-%m-%d"))
            .to_numpy(),
        )
    )
    for repeat in repeats:
        for mask in MASKS:
            group = predictions[
                predictions.repeat.eq(repeat) & predictions.strategy.eq(f"f{mask:02d}")
            ]
            if set(map(tuple, group[study.KEYS].to_numpy())) != expected:
                raise ValueError("Ablation population mismatch")


def run(args):
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    if args.smoke:
        data = study.synthetic_development()
        repeats, outer_count, inner_count = (42,), 2, 2
        prior = {"synthetic": True}
    else:
        data, prior = validate_reuse_source(args.source)
        repeats, outer_count, inner_count = study.REPEAT_SEEDS, 5, 3
    args.output.mkdir(parents=True)
    protocol = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        main_revision=subprocess.check_output(
            ["git", "rev-parse", "main"], text=True
        ).strip(),
        code_sha256=study.file_hash(__file__),
        shared_code_sha256=study.file_hash(study.__file__),
        source_directory=str(args.source.resolve()) if not args.smoke else None,
        source_protocol_sha256=study.file_hash(args.source / "protocol.json")
        if not args.smoke
        else None,
        source_manifest_sha256=study.file_hash(args.source / "manifest.json")
        if not args.smoke
        else None,
        development_sha256=prior.get("inputs", {}).get("development_sha256"),
        factors=FACTORS,
        grid=[configuration(mask) for mask in MASKS],
        contrasts=contrasts(),
        reused_configurations=REUSE,
        repeat_seeds=repeats,
        outer_folds=outer_count,
        inner_folds=inner_count,
        inner_seed=78,
        model_seeds=study.MODEL_SEEDS,
        threshold_rule="Highest inner pooled precision with recall strictly >90%; ties recall then higher cutoff",
        bootstrap_draws=50000 if not args.smoke else 500,
        bootstrap_seed=2026,
        multiplicity="Bonferroni percentile intervals: familywise alpha 0.05 across 17 predeclared precision contrasts; recall 95% marginal exploratory; AP point estimates only",
        primary_endpoint="Mean repeat pooled precision difference under identical inner cutoff selection policy",
        test_data_used=False,
        temporal_validation_used=False,
        model_changed=False,
        smoke=args.smoke,
        runtime=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            sklearn=sklearn.__version__,
            joblib=joblib.__version__,
        ),
        limitations=[
            "Exploratory ablation after inspecting prior grouped development results",
            "Operational effects include training-only cutoff adaptation; AP describes ranking separately",
            "Conditional cluster intervals exclude refitting/reselection uncertainty; dependent repeats",
            "Component contributions are contextual, not causal or additive percentages",
            "No automatic model replacement or final testing",
        ],
    )
    study.write_json(args.output / "protocol.json", protocol)
    data.to_csv(args.output / "development_input.csv", index=False)
    tasks = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for repeat in repeats:
            for fold, (ti, vi) in enumerate(
                study.grouped_splits(data, outer_count, repeat)
            ):
                source_folder = args.source / f"repeat_{repeat}" / f"fold_{fold}"
                if args.smoke:
                    source_folder = (
                        args.output
                        / "synthetic_source"
                        / f"repeat_{repeat}"
                        / f"fold_{fold}"
                    )
                    source_grid = [
                        next(item for item in study.candidates() if item["id"] == name)
                        for name in REUSE.values()
                    ]
                    study.evaluate_outer_fold(
                        data,
                        repeat,
                        fold,
                        ti,
                        vi,
                        source_folder,
                        source_grid,
                        inner_count,
                        study.MODEL_SEEDS,
                    )
                tasks.append(
                    executor.submit(
                        evaluate_fold,
                        data,
                        repeat,
                        fold,
                        ti,
                        vi,
                        source_folder,
                        args.output / f"repeat_{repeat}" / f"fold_{fold}",
                        inner_count,
                    )
                )
        completed = []
        for future in as_completed(tasks):
            item = future.result()
            completed.append(item)
            print(
                f"Completed {len(completed)}/{len(tasks)} ablation folds: repeat {item['repeat']}, fold {item['fold']}",
                flush=True,
            )
    predictions = (
        pd.concat(
            [
                study.read_csv(Path(item["directory"]) / "predictions.csv")
                for item in completed
            ],
            ignore_index=True,
        )
        .sort_values(["repeat", "outer_fold", "strategy", "snapshot", "StoreID"])
        .reset_index(drop=True)
    )
    validate_predictions(predictions, data, repeats)
    predictions.to_csv(args.output / "validation_predictions.csv", index=False)
    fold, repeat, summary = summarize_configs(predictions)
    fold.to_csv(args.output / "fold_metrics.csv", index=False)
    repeat.to_csv(args.output / "repeat_metrics.csv", index=False)
    summary.to_csv(args.output / "configuration_summary.csv", index=False)
    estimates = contrast_statistics(
        predictions, contrasts(), draws=protocol["bootstrap_draws"]
    )
    estimates.to_csv(args.output / "contributions.csv", index=False)
    study.write_json(args.output / "completed_folds.json", completed)
    study.write_json(
        args.output / "manifest.json",
        dict(
            protocol_sha256=study.file_hash(args.output / "protocol.json"),
            new_fits=sum(item["new_fits"] for item in completed),
            reused_fits=sum(item["reused_fits"] for item in completed),
            outputs={
                str(path.relative_to(args.output)): study.file_hash(path)
                for path in args.output.rglob("*")
                if path.is_file()
            },
        ),
    )
    print(summary.to_string(index=False), flush=True)
    print(
        estimates[
            [
                "contrast",
                "precision_delta",
                "recall_delta",
                "precision_family_low",
                "precision_family_high",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/repeated-grouped-precision"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
