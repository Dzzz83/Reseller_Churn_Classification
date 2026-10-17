"""Audit saved reseller-count sensitivity outputs without new fits/cutoffs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from reseller_churn.artifacts import input_identity, verify_manifest, write_json

KEYS = ["StoreID", "snapshot"]


def verify(directory):
    manifest = verify_manifest(directory)
    source = Path(manifest["inputs"]["source"]["path"]).parent
    profile_source = Path(manifest["inputs"]["profiles"]["path"]).parent
    verify_manifest(source)
    verify_manifest(profile_source)
    data = read_csv(source / "development_input.csv")
    profiles = read_csv(profile_source / "case_sku_shares.csv")
    splits = json.loads((source / "protocol.json").read_text())["splits"]
    predictions = read_csv(directory / "predictions.csv")
    metrics = pd.read_csv(directory / "metrics.csv")
    fits = 0
    for split in splits:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        folder = directory / relative
        verify_manifest(folder)
        identities = json.loads((folder / "model_identities.json").read_text())
        require(len(identities) == 10, "Unexpected fit count")
        saved = read_csv(source / relative / "predictions.csv")
        baseline = saved[saved.strategy.eq("baseline")]
        target = int((baseline.score.ge(baseline.cutoff) & baseline.churn.eq(1)).sum())
        full = saved[saved.strategy.eq("cadence_product_mix")]
        for fraction in [0.5, 0.75, 1.0]:
            part = predictions[
                predictions.repeat.eq(split["repeat"])
                & predictions.outer_fold.eq(split["outer_fold"])
                & predictions.fraction.eq(fraction)
            ]
            aligned = part.merge(
                valid[KEYS + ["churn"]],
                on=KEYS,
                validate="one_to_one",
                suffixes=("", "_source"),
            )
            require(
                len(aligned) == len(valid)
                and aligned.churn.eq(aligned.churn_source).all(),
                "Changed natural validation population",
            )
            require(
                part.cutoff.eq(full.cutoff.iloc[0]).all(),
                "Full-training cutoff changed",
            )
            if fraction == 1:
                matched = part.merge(
                    full, on=KEYS, validate="one_to_one", suffixes=("", "_source")
                )
                np.testing.assert_array_equal(matched.score, matched.score_source)
            selected = np.random.default_rng(2026).permutation(
                np.sort(train.StoreID.unique())
            )[: int(train.StoreID.nunique() * fraction)]
            fitted = train[train.StoreID.isin(selected)]
            history = profiles.merge(fitted[KEYS], on=KEYS, validate="many_to_one")
            annual = history[history.scope.eq("annual") & history.share.gt(0)]
            counts = (
                annual[["StoreID", "ProductID"]]
                .drop_duplicates()
                .groupby("ProductID")
                .size()
            )
            vocabulary = [
                int(product)
                for product in sorted(
                    counts.index, key=lambda product: (-counts[product], product)
                )
                if counts[product] >= 20
            ][:20]
            expanded = fitted.copy()
            for scope in ["annual", "latest_order"]:
                matrix = (
                    history[history.scope.eq(scope)]
                    .pivot(index=KEYS, columns="ProductID", values="share")
                    .fillna(0)
                    .reindex(
                        index=pd.MultiIndex.from_frame(fitted[KEYS]),
                        columns=vocabulary,
                        fill_value=0,
                    )
                )
                names = [f"{scope}_sku_{product}" for product in vocabulary]
                expanded[names] = matrix.to_numpy()
                expanded[f"{scope}_sku_other"] = np.clip(
                    1 - matrix.sum(axis=1).to_numpy(), 0, 1
                )
            for identity in [row for row in identities if row["fraction"] == fraction]:
                require(
                    identity["selected_resellers"] == selected.tolist(),
                    "Training selection differs",
                )
                require(
                    identity["vocabulary"] == vocabulary, "Train vocabulary differs"
                )
                content = pd.util.hash_pandas_object(
                    expanded[[*KEYS, "churn", *identity["features"]]], index=False
                ).values.tobytes()
                require(
                    hashlib.sha256(content).hexdigest() == identity["training_sha256"],
                    "Training content differs",
                )
                fits += 1
            record = metrics[
                metrics.repeat.eq(split["repeat"])
                & metrics.outer_fold.eq(split["outer_fold"])
                & metrics.fraction.eq(fraction)
            ].squeeze()
            a, y = part.score.ge(part.cutoff), part.churn.eq(1)
            require(
                [int((a & y).sum()), int((a & ~y).sum())] == [record.tp, record.fp],
                "Operating probe counts differ",
            )
            np.testing.assert_allclose(
                record.average_precision,
                average_precision_score(y, part.score),
                rtol=1e-12,
            )
            options = []
            for score in part.score.unique():
                alert = part.score.ge(score)
                caught = int((alert & y).sum())
                if caught >= target:
                    options.append((int((alert & ~y).sum()), caught))
            if target == 0:
                options.append((0, 0))
            require(
                min(options) == (record.oracle_fp, record.oracle_tp),
                "Oracle curve point differs",
            )
    result = dict(
        status="passed",
        training_identities=fits,
        metric_rows=len(metrics),
        validation_rows=len(predictions),
        models_fitted=0,
        operational_acceptance_assessed=False,
        cutoffs_selected=0,
        verifier=input_identity(dict(script=Path(__file__))),
    )
    write_json(directory / "independent_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
