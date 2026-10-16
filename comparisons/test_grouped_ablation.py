"""Synthetic controls for component definitions, reuse, and uncertainty."""

import json

import pandas as pd
import pytest

from comparisons import grouped_ablation as ablation
from comparisons import grouped_precision_recall as study


def test_grid_covers_single_additions_and_complete_candidate_removals():
    assert len(ablation.MASKS) == 14
    for index in range(5):
        assert (1 << index) in ablation.MASKS
        assert (31 ^ (1 << index)) in ablation.MASKS
    assert len(set(ablation.MASKS) - set(ablation.REUSE)) == 10
    for index, name in enumerate(ablation.FACTORS[:3]):
        candidate = ablation.configuration(1 << index)
        assert candidate["features"] == study.FEATURES + [name]
        assert candidate["parameters"] == study.BASE_PARAMETERS
    assert ablation.configuration(8)["parameters"]["l2_regularization"] == 0.0
    assert ablation.configuration(16)["parameters"]["max_leaf_nodes"] == 31


def test_reuse_matches_features_order_and_parameters_exactly():
    prior = {item["id"]: item for item in study.candidates()}
    for mask, name in ablation.REUSE.items():
        assert ablation.equivalent(ablation.configuration(mask), prior[name])
    changed = dict(prior["cadence_regularized"])
    changed["features"] = list(reversed(changed["features"]))
    assert not ablation.equivalent(ablation.configuration(31), changed)


def test_predeclared_contrasts_are_identifiable_and_signed():
    definitions = ablation.contrasts()
    assert len(definitions) == 17
    assert len({item["id"] for item in definitions}) == 17
    for item in definitions:
        assert sum(item["weights"].values()) == 0
        assert set(item["weights"]).issubset(
            {f"f{mask:02d}" for mask in ablation.MASKS}
        )
    interaction = next(
        item
        for item in definitions
        if item["id"] == "cadence_regularization_interaction"
    )
    assert interaction["weights"] == {"f31": 1.0, "f07": -1.0, "f24": -1.0, "f00": 1.0}


def test_cluster_intervals_are_zero_for_identical_arms_and_wider_when_adjusted():
    rows = []
    for repeat in (42, 78):
        for store in range(1, 9):
            for snapshot in study.SNAPSHOTS:
                for arm in ("f00", "f01", "f02"):
                    rows.append(
                        dict(
                            repeat=repeat,
                            StoreID=store,
                            snapshot=snapshot,
                            strategy=arm,
                            churn=store % 2,
                            score=0.7 if store % 2 else 0.2,
                            cutoff=0.4,
                        )
                    )
    data = pd.DataFrame(rows)
    contrasts = [
        dict(id="same", kind="synthetic", weights={"f01": 1.0, "f00": -1.0}),
        dict(id="same2", kind="synthetic", weights={"f02": 1.0, "f00": -1.0}),
    ]
    result = ablation.contrast_statistics(data, contrasts, draws=500)
    assert result.precision_delta.eq(0).all()
    assert result.precision_family_low.eq(0).all()
    assert result.precision_family_high.eq(0).all()
    data.loc[(data.strategy == "f01") & (data.StoreID == 2), "score"] = 0.8
    changed = ablation.contrast_statistics(data, contrasts, draws=500)
    assert (changed.precision_family_low <= changed.precision_marginal_low).all()
    assert (changed.precision_family_high >= changed.precision_marginal_high).all()


def test_reuse_requires_exact_outer_indices_before_any_fit(tmp_path):
    data = study.synthetic_development()
    ti, vi = study.grouped_splits(data, 2, 42)[0]
    source = tmp_path / "source"
    source.mkdir()
    (source / "split.json").write_text(
        json.dumps(dict(train_rows=vi.tolist(), validation_rows=ti.tolist()))
    )
    with pytest.raises(ValueError, match="Reused outer fold differs"):
        ablation.evaluate_fold(data, 42, 0, ti, vi, source, tmp_path / "output", 2)


def test_population_validator_rejects_missing_account():
    data = study.synthetic_development()
    predictions = []
    for mask in ablation.MASKS:
        part = data[study.KEYS + ["churn"]].copy()
        part["snapshot"] = part.snapshot.dt.strftime("%Y-%m-%d")
        part["strategy"], part["repeat"] = f"f{mask:02d}", 42
        predictions.append(part)
    frame = pd.concat(predictions, ignore_index=True)
    ablation.validate_predictions(frame, data, [42])
    with pytest.raises(ValueError, match="population mismatch"):
        ablation.validate_predictions(frame.iloc[1:], data, [42])
