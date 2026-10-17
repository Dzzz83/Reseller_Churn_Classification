"""No-refit checks of SKU source shares, train-only vocabularies and decisions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.verify_basket_breadth import independent_cutoff
from reseller_churn.artifacts import input_identity, verify_manifest, write_json

KEYS = ["StoreID", "snapshot"]


def ordered(frame):
    return frame.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)


def verify_profiles(directory):
    manifest = verify_manifest(directory)
    orders = pd.read_csv(
        Path(manifest["inputs"]["orders"]["path"]), parse_dates=["OrderDate"]
    )
    lines = pd.read_csv(Path(manifest["inputs"]["lines"]["path"]))
    saved = read_csv(directory / "case_sku_shares.csv")
    source = lines.merge(
        orders[["SalesOrderID", "StoreID", "OrderDate"]],
        on="SalesOrderID",
        validate="many_to_one",
    )
    checked = 0
    for (store, snapshot, scope), part in saved.groupby(KEYS + ["scope"]):
        history = source[
            (source.StoreID == store)
            & source.OrderDate.ge(snapshot - pd.DateOffset(months=12))
            & source.OrderDate.lt(snapshot)
        ]
        if scope == "latest_order":
            last = (
                history.sort_values(["OrderDate", "SalesOrderID"]).iloc[-1].SalesOrderID
            )
            history = history[history.SalesOrderID.eq(last)]
        expected = history.groupby("ProductID").LineTotal.sum().sort_index()
        actual = part.set_index("ProductID").sort_index()
        np.testing.assert_array_equal(actual.index, expected.index)
        np.testing.assert_allclose(
            actual.share, expected / expected.sum(), rtol=1e-12, atol=1e-12
        )
        checked += 1
    result = dict(status="passed", source_profiles=checked, models_fitted=0)
    write_json(directory / "independent_verification.json", result)
    print(json.dumps(result))


def verify_comparison(directory):
    manifest = verify_manifest(directory)
    source = Path(manifest["inputs"]["cached_source"]["path"]).parent
    profiles_dir = Path(manifest["inputs"]["profiles"]["path"]).parent
    verify_manifest(source)
    verify_manifest(profiles_dir)
    profiles = read_csv(profiles_dir / "case_sku_shares.csv")
    data = read_csv(directory / "development_input.csv")
    pd.testing.assert_frame_equal(data, read_csv(source / "development_input.csv"))
    protocol = json.loads((directory / "protocol.json").read_text())
    candidate = protocol["arms"][1]["id"]
    fits, cuts = 0, 0
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        folder = directory / relative
        verify_manifest(folder)
        inner, outer = (
            read_csv(folder / "inner_scores.csv"),
            read_csv(folder / "predictions.csv"),
        )
        for current, original, column in [
            (inner, read_csv(source / relative / "inner_scores.csv"), "candidate"),
            (outer, read_csv(source / relative / "predictions.csv"), "strategy"),
        ]:
            pd.testing.assert_frame_equal(
                ordered(current[current[column].eq("baseline")]),
                ordered(original[original[column].eq("baseline")]),
            )
        assignment = ordered(inner[inner.candidate.eq("baseline")]).inner_fold
        for arm in ("baseline", candidate):
            inside, outside = (
                ordered(inner[inner.candidate.eq(arm)]),
                ordered(outer[outer.strategy.eq(arm)]),
            )
            pd.testing.assert_frame_equal(
                inside[KEYS + ["churn"]], ordered(train)[KEYS + ["churn"]]
            )
            pd.testing.assert_frame_equal(
                outside[KEYS + ["churn"]], ordered(valid)[KEYS + ["churn"]]
            )
            require(
                inside.inner_fold.eq(assignment).all()
                and inside.groupby("StoreID").inner_fold.nunique().eq(1).all(),
                "Inner memberships differ",
            )
            require(
                outside.cutoff.eq(
                    independent_cutoff(inside.churn.to_numpy(), inside.score.to_numpy())
                ).all(),
                "Inner cutoff differs",
            )
            cuts += 1
        identities = json.loads((folder / "model_identities.json").read_text())
        require(len(identities) == 20, "Unexpected ensemble fit count")
        for identity in identities:
            fitted = train
            if identity["stage"] == "inner":
                held = inner.loc[
                    inner.candidate.eq(candidate)
                    & inner.inner_fold.eq(identity["inner_fold"]),
                    "StoreID",
                ]
                fitted = train[~train.StoreID.isin(held)]
            if "bootstrap_store_ids" in identity:
                stores = np.sort(fitted.StoreID.unique())
                expected = (
                    np.random.default_rng(identity["seed"])
                    .choice(stores, size=len(stores), replace=True)
                    .tolist()
                )
                require(
                    expected == identity["bootstrap_store_ids"],
                    "Bootstrap draw sequence differs",
                )
                fitted = pd.concat(
                    [fitted[fitted.StoreID.eq(store)] for store in expected],
                    ignore_index=True,
                )
            history = profiles.merge(
                fitted[KEYS].drop_duplicates(), on=KEYS, validate="many_to_one"
            )
            annual = history[history.scope.eq("annual") & history.share.gt(0)]
            counts = (
                annual[["ProductID", "StoreID"]]
                .drop_duplicates()
                .groupby("ProductID")
                .size()
            )
            ranking = sorted(
                counts.index, key=lambda product: (-counts[product], product)
            )
            expanded = fitted.copy()
            if "projection" in identity:
                require(
                    set(identity["features"]) == set(protocol["arms"][1]["features"]),
                    "Projection feature set differs",
                )
                projection = identity["projection"]
                vocabulary = sorted(annual.ProductID.astype(int).unique().tolist())
                require(
                    vocabulary == projection["vocabulary"],
                    "Projection vocabulary differs",
                )
                matrices = {}
                for scope in ("annual", "latest_order"):
                    matrix = (
                        history[history.scope.eq(scope)]
                        .pivot(index=KEYS, columns="ProductID", values="share")
                        .fillna(0)
                    )
                    matrices[scope] = matrix.reindex(
                        index=pd.MultiIndex.from_frame(fitted[KEYS]),
                        columns=vocabulary,
                        fill_value=0,
                    ).to_numpy()
                mean, components = (
                    np.array(projection["mean"]),
                    np.array(projection["components"]),
                )
                np.testing.assert_allclose(
                    mean, matrices["annual"].mean(axis=0), rtol=1e-12, atol=1e-12
                )
                np.testing.assert_allclose(
                    components @ components.T, np.eye(5), rtol=1e-12, atol=1e-12
                )
                centered = matrices["annual"] - mean
                covariance = centered.T @ centered / (len(centered) - 1)
                eigenvalues = np.array(projection["explained_variance"])
                np.testing.assert_allclose(
                    covariance @ components.T,
                    components.T * eigenvalues,
                    rtol=1e-9,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    np.array(projection["singular_values"]) ** 2 / (len(centered) - 1),
                    eigenvalues,
                    rtol=1e-12,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    eigenvalues / np.trace(covariance),
                    projection["explained_variance_ratio"],
                    rtol=1e-12,
                    atol=1e-12,
                )
                require(
                    np.all(np.diff(eigenvalues) <= 0), "Projection components unsorted"
                )
                for scope, matrix in matrices.items():
                    names = [f"{scope}_mix_pc{i + 1}" for i in range(5)]
                    expanded[names] = (matrix - mean) @ components.T
                    expanded[f"{scope}_unseen_share"] = np.clip(
                        1 - matrix.sum(axis=1), 0, 1
                    )
            else:
                vocabulary = [
                    int(product) for product in ranking if counts[product] >= 20
                ][:20]
                require(
                    vocabulary == identity["vocabulary"], "Training vocabulary differs"
                )
                expanded = fitted.copy()
                for scope in ("annual", "latest_order"):
                    matrix = (
                        history[history.scope.eq(scope)]
                        .pivot(index=KEYS, columns="ProductID", values="share")
                        .fillna(0)
                    )
                    matrix = matrix.reindex(
                        index=pd.MultiIndex.from_frame(fitted[KEYS]),
                        columns=vocabulary,
                        fill_value=0,
                    )
                    names = [f"{scope}_sku_{product}" for product in vocabulary]
                    expanded[names] = matrix.to_numpy()
                    expanded[f"{scope}_sku_other"] = np.clip(
                        1 - matrix.sum(axis=1).to_numpy(), 0, 1
                    )
            content = pd.util.hash_pandas_object(
                expanded[[*KEYS, "churn", *identity["features"]]], index=False
            ).values.tobytes()
            require(
                hashlib.sha256(content).hexdigest() == identity["training_sha256"],
                "Training content differs",
            )
            fits += 1
    predictions = read_csv(directory / "validation_predictions.csv")
    require(
        len(predictions) == 8250
        and not predictions.duplicated(KEYS + ["repeat", "strategy"]).any(),
        "Incomplete paired population",
    )
    metrics = pd.read_csv(directory / "metrics.csv")
    checked = 0
    for scope, columns in [
        ("repeat_pooled", ["repeat", "strategy"]),
        ("outer_fold", ["repeat", "strategy", "outer_fold"]),
    ]:
        for values, part in predictions.groupby(columns):
            mask = metrics.scope.eq(scope) & metrics.slice.eq("overall")
            for name, value in zip(columns, values):
                mask &= metrics["policy" if name == "strategy" else name].eq(value)
            row = metrics[mask].squeeze()
            a, y = part.score.ge(part.cutoff), part.churn.eq(1)
            require(
                [int(m.sum()) for m in [a & y, a & ~y, ~a & y, ~a & ~y]]
                == [row.tp, row.fp, row.fn, row.tn],
                "Metric counts differ",
            )
            checked += 1
    result = dict(
        status="passed",
        outer_splits=15,
        inner_cutoffs=cuts,
        training_vocabulary_identities=fits,
        overall_metric_rows=checked,
        models_fitted=0,
        october_outcomes_accessed=False,
    )
    reference = read_csv(source / "validation_predictions.csv")
    reference = reference[reference.strategy.eq("cadence_no_count")]
    reference_inputs = {}
    if candidate in {"cadence_product_mix_bagged", "cadence_latent_mix"}:
        unbagged_dir = source.parent / "product-mix-comparison-20261017"
        verify_manifest(unbagged_dir)
        unbagged = read_csv(unbagged_dir / "validation_predictions.csv")
        reference = pd.concat(
            [reference, unbagged[unbagged.strategy.eq("cadence_product_mix")]]
        )
        reference_inputs = input_identity(
            {"unbagged_reference": unbagged_dir / "manifest.json"}
        )

    pd.testing.assert_frame_equal(
        ordered(
            reference[
                reference.repeat.eq(301) & reference.strategy.eq("cadence_no_count")
            ]
        )[KEYS + ["churn"]],
        ordered(
            predictions[predictions.repeat.eq(301) & predictions.strategy.eq(candidate)]
        )[KEYS + ["churn"]],
    )
    comparisons = []
    combined = pd.concat([reference, predictions[predictions.strategy.eq(candidate)]])
    for (repeat, arm), part in combined.groupby(["repeat", "strategy"]):
        for activity, subset in [
            ("overall", part),
            ("recent_purchase", part[part.n_orders_3m.gt(0)]),
            ("no_recent_purchase", part[part.n_orders_3m.eq(0)]),
        ]:
            a, y = subset.score.ge(subset.cutoff), subset.churn.eq(1)
            tp, fp = int((a & y).sum()), int((a & ~y).sum())
            comparisons.append(
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
    pd.DataFrame(comparisons).to_csv(
        directory / "cadence_reference_comparison.csv", index=False
    )
    if candidate == "cadence_latent_mix":
        projection_rows = []
        for path in sorted(directory.glob("repeat_*/fold_*/model_identities.json")):
            rows = json.loads(path.read_text())
            for row in rows:
                projection = row["projection"]
                projection_rows.append(
                    dict(
                        fold=str(path.parent.relative_to(directory)),
                        stage=row["stage"],
                        inner_fold=row.get("inner_fold"),
                        seed=row["seed"],
                        vocabulary_size=len(projection["vocabulary"]),
                        explained_variance_fraction=sum(
                            projection["explained_variance_ratio"]
                        ),
                    )
                )
        pd.DataFrame(projection_rows).to_csv(
            directory / "projection_summary.csv", index=False
        )
    derived = {"secondary_reference": directory / "cadence_reference_comparison.csv"}
    if candidate == "cadence_latent_mix":
        derived["projection_summary"] = directory / "projection_summary.csv"
    result["derived_outputs"] = input_identity(derived)
    result["secondary_reference_inputs"] = reference_inputs
    result["verifier_identity"] = input_identity({"script": Path(__file__)})
    write_json(directory / "independent_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profiles", type=Path)
    mode.add_argument("--comparison", type=Path)
    args = parser.parse_args()
    if args.profiles:
        verify_profiles(args.profiles)
    else:
        verify_comparison(args.comparison)
