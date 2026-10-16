"""Verify saved SHAP arithmetic and held-out model scores without fitting."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from compare_subtotal_totaldue import ROOT, checked_hash
from explain_churn_shap import FEATURES, contribution_tables, error_categories, verify_reconstruction
from reseller_churn.config.model_settings import MODEL_SEEDS
from verify_subtotal_totaldue_outputs import assert_report_equal


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["models_refitted"] != 25 or manifest["explained_rows"] != 1375 or manifest["october_accessed"]:
        raise ValueError("Unexpected explanation scope")
    for category in ["input_sha256", "code_sha256", "model_sha256"]:
        for name, expected in manifest[category].items():
            checked_hash(ROOT / name, expected)
    for name, expected in manifest["output_sha256"].items():
        checked_hash(output / name, expected)

    def read(name):
        return pd.read_csv(output / name, float_precision="round_trip")

    cases, values, individual, backgrounds = (read(name) for name in [
        "held_out_cases.csv", "ensemble_contributions.csv", "individual_seed_explanations.csv", "background_rows.csv"])
    keys = ["StoreID", "snapshot", "outer_fold"]
    np.testing.assert_array_equal(cases[keys], values[keys])
    shap_columns = [f"shap_{feature}" for feature in FEATURES]
    verify_reconstruction(cases.base_probability, values[shap_columns], cases.churn_probability)
    grouped = individual.groupby(keys)
    if not grouped.seed.apply(lambda seeds: set(seeds) == set(MODEL_SEEDS)).all():
        raise ValueError("Incomplete seed explanations")
    if individual[keys + ["seed"]].duplicated().any():
        raise ValueError("Duplicate seed explanations")
    means = grouped[[*shap_columns, "base_probability", "churn_probability"]].mean().reset_index()
    merged = cases.merge(values, on=keys, validate="one_to_one")
    assert_report_equal(merged, means, keys)
    saved_nested = next(name for name in manifest["input_sha256"] if name.endswith("outer_predictions.csv"))
    expected = pd.read_csv(ROOT / saved_nested, float_precision="round_trip")
    expected = expected[expected.architecture.eq("hist_gradient_boosting") & expected.feature_set.eq("subtotal")]
    assert_report_equal(cases, expected[[*keys, "actual_churn", "threshold", "churn_probability"]], keys)
    np.testing.assert_array_equal(cases.error_category,
        error_categories(cases.actual_churn, cases.churn_probability, cases.threshold.to_numpy()))
    for fold, group in cases.groupby("outer_fold"):
        model_path = ROOT / "models/provisional" / output.name / f"fold_{fold}.joblib"
        bundle = joblib.load(model_path)
        reference = bundle["background"]
        if set(reference.StoreID) & set(group.StoreID):
            raise ValueError("Held-out identity entered reference background")
        saved_background = backgrounds[backgrounds.outer_fold.eq(fold)]
        assert_report_equal(saved_background, reference, ["StoreID", "snapshot"])
        for seed, model in zip(MODEL_SEEDS, bundle["models"]):
            seed_cases = individual[individual.outer_fold.eq(fold) & individual.seed.eq(seed)]
            paired = group.merge(seed_cases, on=keys, validate="one_to_one", suffixes=("", "_seed"))
            probability = model.predict_proba(paired[FEATURES])[:, 1]
            np.testing.assert_allclose(paired.churn_probability_seed, probability, atol=1e-12, rtol=0)
            verify_reconstruction(paired.base_probability_seed, paired[shap_columns], probability)
            np.testing.assert_allclose(paired.base_probability_seed,
                model.predict_proba(reference[FEATURES])[:, 1].mean(), atol=1e-12, rtol=0)
    importance, by_error, by_fold = contribution_tables(cases, values[shap_columns].to_numpy())
    for name, expected_table, table_keys in [
        ("global_importance.csv", importance, ["feature"]),
        ("importance_by_error.csv", by_error, ["error_category", "feature"]),
        ("importance_by_fold.csv", by_fold, ["outer_fold", "feature"]),
    ]:
        assert_report_equal(read(name), expected_table, table_keys)
    examples = read("examples.csv")
    for _, example in examples.iterrows():
        assert ((cases.StoreID.eq(example.StoreID)) & cases.snapshot.eq(example.snapshot)).sum() == 1
    print("Verified hashes, reproduced held-out scores, training-only backgrounds,")
    print("seed and ensemble probability reconstruction, thresholds, errors, and importance reports.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output.resolve())
