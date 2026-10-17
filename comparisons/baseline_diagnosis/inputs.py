"""Verify development input identity, fold isolation, and saved cutoff provenance."""

import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

from reseller_churn.artifacts import file_sha256, input_identity
from reseller_churn.config.frozen_model import (
    FEATURES,
    PARAMETERS,
    SEEDS,
    TRAINING_SNAPSHOTS,
)
from reseller_churn.evaluation.metrics import calculate_threshold_metrics

from .definitions import KEYS, POLICIES, REPEATS


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", parse_dates=["snapshot"])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_study_input(source: Path, recorded_path: str) -> Path:
    """Locate moved study inputs without rewriting their historical manifests."""
    original = Path(recorded_path)
    parts = PureWindowsPath(recorded_path).parts
    if "results" in parts:
        relative = parts[parts.index("results") + 1 :]
        require(".." not in relative, "Unsafe recorded study input path")
        relocated = source.parent.joinpath(*relative)
        if relocated.is_file():
            return relocated
    require(original.is_file(), f"Study input unavailable: {recorded_path}")
    return original


def validate_partition(
    features: pd.DataFrame, split: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require a complete row partition with every reseller on only one side."""
    from comparisons.development_checks import (
        validate_development_cases,
        validate_development_split,
    )

    training_rows, validation_rows = split["train_rows"], split["validation_rows"]
    require(
        set(training_rows).isdisjoint(validation_rows)
        and sorted(training_rows + validation_rows) == list(range(len(features))),
        "Outer split does not partition development",
    )
    train, valid = features.iloc[training_rows], features.iloc[validation_rows]
    require(
        not set(train.StoreID) & set(valid.StoreID),
        "Reseller leakage across outer fold",
    )
    require(
        set(train.StoreID) == set(split["train_store_ids"])
        and set(valid.StoreID) == set(split["validation_store_ids"]),
        "Split identity mismatch",
    )
    validate_development_cases(features)
    validate_development_split(train, valid)
    return train, valid


def verify_baseline(
    source: Path, orders_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Audit only the development artifacts consumed by this diagnosis."""
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        name.replace("\\", "/"): value for name, value in manifest["outputs"].items()
    }
    consumed = {"study_manifest": manifest_path}

    def verified(name: str) -> Path:
        path = source / name
        require(
            name in expected and file_sha256(path) == expected[name],
            f"Changed study input: {name}",
        )
        consumed[name] = path
        return path

    protocol_path = source / "protocol.json"
    require(
        file_sha256(protocol_path) == manifest["protocol_sha256"],
        "Changed study protocol",
    )
    consumed["study_protocol"] = protocol_path
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    baseline = next(
        item for item in protocol["grid"] if item["id"] == "original_baseline"
    )
    require(
        baseline["features"] == list(FEATURES) and baseline["parameters"] == PARAMETERS,
        "Saved baseline differs from the frozen ten-feature configuration",
    )
    require(protocol["model_seeds"] == list(SEEDS), "Unexpected model seeds")
    require(
        protocol["repeat_seeds"] == list(REPEATS) and protocol["outer_folds"] == 5,
        "Unexpected grouped validation schedule",
    )
    require(
        protocol["snapshots"] == list(TRAINING_SNAPSHOTS)
        and not protocol["test_data_used"],
        "Only approved development snapshots are allowed",
    )

    preparation = resolve_study_input(
        source, protocol["inputs"]["preparation_manifest"]
    )
    require(
        file_sha256(preparation) == protocol["inputs"]["preparation_manifest_sha256"],
        "Changed preparation provenance",
    )
    consumed["preparation_manifest"] = preparation
    preparation_metadata = json.loads(preparation.read_text(encoding="utf-8"))
    require(
        file_sha256(orders_path) == preparation_metadata["input_hashes"]["orders"],
        "Order source differs from the preparation reference",
    )
    consumed["orders"] = orders_path
    original_features = resolve_study_input(source, protocol["inputs"]["development"])
    require(
        file_sha256(original_features) == protocol["inputs"]["development_sha256"],
        "Changed original development features",
    )
    consumed["original_development"] = original_features
    features = read_csv(verified("development_input.csv"))
    # Reconstruct the study training parser, which used pandas defaults.
    # Saved study CSVs use round-trip parsing to preserve those fitted values.
    original = pd.read_csv(original_features)
    original["snapshot"] = pd.to_datetime(original.snapshot)
    original = original.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)
    require(not features.duplicated(KEYS).any(), "Duplicate development keys")
    require(
        set(features.snapshot.dt.strftime("%Y-%m-%d")) == set(TRAINING_SNAPSHOTS),
        "Unapproved development snapshot",
    )
    require(
        (len(features), int(features.churn.sum()), features.StoreID.nunique())
        == (1375, 212, 422),
        "Unexpected development population",
    )
    pd.testing.assert_frame_equal(
        features[KEYS + ["churn"]], original[KEYS + ["churn"]]
    )
    np.testing.assert_allclose(
        features[list(FEATURES)],
        original[list(FEATURES)],
        rtol=0,
        atol=0,
        equal_nan=True,
    )

    predictions = read_csv(verified("validation_predictions.csv"))
    predictions = predictions[predictions.strategy.isin(POLICIES)].copy()
    require(set(predictions.strategy) == set(POLICIES), "Missing baseline policy")
    require(set(predictions.repeat) == set(REPEATS), "Missing validation repetition")
    require(
        not predictions.duplicated([*KEYS, "repeat", "strategy"]).any(),
        "Duplicate predictions",
    )
    require(
        np.isfinite(predictions[["score", "cutoff"]]).all().all()
        and predictions[["score", "cutoff"]].ge(0).all().all()
        and predictions[["score", "cutoff"]].le(1).all().all(),
        "Invalid saved scores/cutoffs",
    )
    for (_, _), rows in predictions.groupby(["strategy", "repeat"]):
        ordered = rows.sort_values(["snapshot", "StoreID"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            ordered[KEYS + ["churn", "n_orders_3m"]],
            features[KEYS + ["churn", "n_orders_3m"]],
        )

    for repeat in REPEATS:
        primary = predictions[
            (predictions.repeat == repeat)
            & predictions.strategy.eq("original_baseline")
        ]
        fixed = predictions[
            (predictions.repeat == repeat)
            & predictions.strategy.eq("original_fixed_006")
        ]
        alignment = primary.merge(
            fixed, on=KEYS, validate="one_to_one", suffixes=("", "_fixed")
        )
        require(
            np.array_equal(alignment.score, alignment.score_fixed)
            and np.array_equal(alignment.outer_fold, alignment.outer_fold_fixed)
            and alignment.cutoff_fixed.eq(0.06).all(),
            "Baseline policies do not share scores/folds",
        )
        require(set(primary.outer_fold) == set(range(5)), "Missing outer fold")
        for fold in range(5):
            prefix = f"repeat_{repeat}/fold_{fold}"
            split = json.loads(
                verified(f"{prefix}/split.json").read_text(encoding="utf-8")
            )
            train, valid = validate_partition(features, split)
            held = primary[primary.outer_fold.eq(fold)].sort_values(
                ["snapshot", "StoreID"]
            )
            pd.testing.assert_frame_equal(
                held[KEYS].reset_index(drop=True), valid[KEYS].reset_index(drop=True)
            )
            selections = pd.read_csv(
                verified(f"{prefix}/inner_selections.csv"), float_precision="round_trip"
            )
            selection = selections[
                selections.candidate.eq("original_baseline")
            ].squeeze()
            require(
                held.cutoff.eq(selection.cutoff).all(),
                "Cutoff differs from saved inner selection",
            )
            inner = read_csv(verified(f"{prefix}/inner_scores.csv"))
            inner = inner[inner.candidate.eq("original_baseline")]
            pd.testing.assert_frame_equal(
                inner.sort_values(["snapshot", "StoreID"])[
                    KEYS + ["churn"]
                ].reset_index(drop=True),
                train[KEYS + ["churn"]].reset_index(drop=True),
            )
            require(
                inner.groupby("StoreID").inner_fold.nunique().eq(1).all(),
                "Inner reseller leakage",
            )
            metrics = calculate_threshold_metrics(
                inner.churn, inner.score, selection.cutoff
            )
            require(
                metrics["recall"] > 0.9
                and np.isclose(metrics["recall"], selection.recall)
                and np.isclose(metrics["precision"], selection.precision),
                "Saved inner operating metrics differ",
            )
            identities = json.loads(
                verified(f"{prefix}/model_identities.json").read_text(encoding="utf-8")
            )
            outer = [
                item
                for item in identities
                if item["candidate"] == "original_baseline" and item["stage"] == "outer"
            ]
            require(
                sorted(item["seed"] for item in outer) == sorted(SEEDS),
                "Unexpected outer ensemble seeds",
            )
            content = pd.util.hash_pandas_object(
                train[[*KEYS, "churn", *FEATURES]], index=False
            ).values.tobytes()
            require(
                all(
                    item["training_sha256"] == hashlib.sha256(content).hexdigest()
                    for item in outer
                ),
                "Recorded outer training content differs",
            )
    for name in ("repeat_metrics.csv", "fold_metrics.csv"):
        verified(name)
    return (
        features,
        predictions,
        {
            "inputs": input_identity(consumed),
            "outer_folds_verified": 15,
            "baseline_configuration": baseline,
            "model_seeds": list(SEEDS),
        },
    )
