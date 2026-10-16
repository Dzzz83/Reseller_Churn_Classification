"""Explain reproduced held-out GB10 SubTotal scores without accessing October."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from time import monotonic

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compare_seven_ten_features import FEATURE_SETS
from compare_subtotal_totaldue import checked_hash, load_pair
from nested_revenue_comparison import nested_splits
from reseller_churn.config.model_settings import BASELINE_GRADIENT_BOOSTING, MODEL_SEEDS
from reseller_churn.config.project_paths import get_run_directory
from reseller_churn.modeling.development import file_sha256, repository_identity, write_json
from reseller_churn.modeling.model_factory import ModelFactory

FEATURES = FEATURE_SETS["ten"]
BACKGROUND_ROWS = 20
MAX_EVALS = 210  # Ten forward/backward permutations when all ten features vary.
REPEAT_SEED = 42


def probability_function(model, columns):
    """Preserve feature names when SHAP passes masked NumPy arrays."""
    def predict(values):
        return model.predict_proba(pd.DataFrame(values, columns=columns))[:, 1]
    return predict


def verify_reconstruction(bases, contributions, probabilities):
    reconstructed = np.asarray(bases) + np.asarray(contributions).sum(axis=1)
    np.testing.assert_allclose(reconstructed, probabilities, atol=1e-8, rtol=0)
    return float(np.max(abs(reconstructed - probabilities)))


def error_categories(target, probabilities, threshold):
    alert = np.asarray(probabilities) >= threshold
    churn = np.asarray(target) == 1
    return np.select([alert & churn, alert & ~churn, ~alert & churn],
                     ["caught_churner", "false_alert", "missed_churner"], default="correct_non_alert")


def pick_examples(cases):
    """Purposeful teaching cases, not a representative performance sample."""
    result = []
    active_false = cases[cases.error_category.eq("false_alert") & cases.n_orders_3m.gt(0)]
    if not active_false.empty:
        row = active_false.sort_values(["churn_probability", "StoreID", "snapshot"], ascending=[False, True, True]).iloc[0]
        result.append({"example": "walkthrough", **row.to_dict()})
    for category in ["caught_churner", "missed_churner", "false_alert"]:
        group = cases[cases.error_category.eq(category)].sort_values(["StoreID", "snapshot"])
        if not group.empty:
            result.append({"example": category, **group.iloc[0].to_dict()})
    return pd.DataFrame(result)


def contribution_tables(cases, contributions):
    keys = cases[["StoreID", "snapshot", "outer_fold", "error_category"]].reset_index(drop=True)
    values = pd.DataFrame(contributions, columns=FEATURES)
    long = pd.concat([keys, values], axis=1).melt(
        id_vars=keys.columns.tolist(), var_name="feature", value_name="shap_probability")
    long["absolute_shap"] = long.shap_probability.abs()
    global_importance = long.groupby("feature").agg(
        mean_absolute_shap=("absolute_shap", "mean"), mean_signed_shap=("shap_probability", "mean")
    ).sort_values("mean_absolute_shap", ascending=False).reset_index()
    by_error = long.groupby(["error_category", "feature"]).absolute_shap.mean().rename("mean_absolute_shap").reset_index()
    by_fold = long.groupby(["outer_fold", "feature"]).absolute_shap.mean().rename("mean_absolute_shap").reset_index()
    return global_importance, by_error, by_fold


def render_plots(output, cases, contributions, bases, examples, importance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    plots = output / "plots"
    plots.mkdir()
    indices = cases.set_index(["StoreID", "snapshot"]).index
    for _, example in examples.iterrows():
        index = indices.get_loc((example.StoreID, example.snapshot))
        explanation = shap.Explanation(values=contributions[index], base_values=bases[index],
            data=cases.iloc[index][FEATURES].to_numpy(dtype=float), feature_names=FEATURES)
        shap.plots.waterfall(explanation, max_display=10, show=False)
        figure = plt.gcf()
        figure.set_size_inches(11, 7.5)
        figure.suptitle(
            f"{example['example']}: reseller {int(example.StoreID)}, {example.snapshot}\n"
            f"Score {example.churn_probability:.3f}; alert cutoff {example.threshold:.2f}; actual churn {int(example.actual_churn)}",
            fontsize=12, y=1.03)
        figure.savefig(plots / f"{example['example']}_waterfall.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
    figure, axes = plt.subplots(figsize=(10, 6))
    ordered = importance.iloc[::-1]
    axes.barh(ordered.feature, ordered.mean_absolute_shap * 100, color="#326c91")
    axes.set_xlabel("Mean absolute contribution (probability percentage points)")
    axes.set_title("GB10 SubTotal: held-out reseller explanations\nFive outer models; training-fold reference backgrounds")
    figure.tight_layout()
    figure.savefig(plots / "global_importance.png", dpi=160)
    plt.close(figure)
    explanation = shap.Explanation(values=contributions, base_values=bases,
        data=cases[FEATURES].to_numpy(dtype=float), feature_names=FEATURES)
    shap.plots.beeswarm(explanation, max_display=10, show=False, plot_size=(10, 6))
    figure = plt.gcf()
    plt.xlabel("SHAP contribution to churn probability")
    figure.savefig(plots / "global_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def run(preparation, nested, run_id):
    import shap

    get_run_directory(run_id)
    output = ROOT / "results/debugging/shap_analysis" / run_id
    model_directory = ROOT / "models/provisional" / run_id
    if output.exists() or model_directory.exists():
        raise FileExistsError("Choose a new run ID; explanation or model directory exists")
    original_protocol = json.loads((nested / "protocol.json").read_text())
    for category in ["input_sha256", "code_sha256"]:
        for name, expected in original_protocol[category].items():
            checked_hash(ROOT / name, expected)
    data = load_pair(preparation)["subtotal"]
    source = nested / "repeat_42"
    completion = json.loads((source / "completion.json").read_text())
    for name in ["outer_predictions.csv", "selected_thresholds.csv", "memberships.csv"]:
        checked_hash(source / name, completion["output_sha256"][name])
    expected = pd.read_csv(source / "outer_predictions.csv", float_precision="round_trip")
    expected = expected[expected.architecture.eq("hist_gradient_boosting") & expected.feature_set.eq("subtotal")]
    output.mkdir(parents=True)
    model_directory.mkdir(parents=True)
    write_json(output / "protocol.json", {
        "model": "GB10 SubTotal", "features": FEATURES, "repeat_seed": REPEAT_SEED,
        "model_seeds": list(MODEL_SEEDS), "planned_refits": 25,
        "method": "PermutationExplainer", "output_scale": "churn probability",
        "background": "20 uniformly sampled training-fold rows, seed 42 + outer fold",
        "max_evals": MAX_EVALS, "aggregation": "mean seed bases and probability contributions",
        "missing_values": "native; no imputation", "october_accessed": False,
        "case_selection": "highest-score recently active false alert for walkthrough; first StoreID/snapshot per error category",
        "limitations": ["Approximate permutation SHAP, small reference backgrounds", "Interventional masking can mix correlated inputs implausibly",
                        "Fold-specific models/backgrounds; summary is not one deployed model", "Explanations do not establish causation, calibration, or improved accuracy"],
    })
    records, mean_values, mean_bases, backgrounds, per_seed, checks = [], [], [], [], [], []
    bundles = {}
    start = monotonic()
    for fold, (train, validation, _) in enumerate(nested_splits(data, REPEAT_SEED), start=1):
        training, cases = data.iloc[train], data.iloc[validation]
        if set(training.StoreID) & set(cases.StoreID):
            raise ValueError("Explanation population was used for training")
        reference = training.sample(n=BACKGROUND_ROWS, random_state=42 + fold)
        background = reference[["StoreID", "snapshot", *FEATURES]].copy()
        background["outer_fold"] = fold
        backgrounds.append(background)
        saved = expected[expected.outer_fold.eq(fold)].sort_values(["snapshot", "StoreID"])
        cases = cases.sort_values(["snapshot", "StoreID"])
        np.testing.assert_array_equal(cases[["StoreID", "snapshot"]], saved[["StoreID", "snapshot"]])
        threshold = float(saved.threshold.iloc[0])
        models, values, bases, scores = [], [], [], []
        for seed in MODEL_SEEDS:
            model = ModelFactory.create_gradient_boosting(BASELINE_GRADIENT_BOOSTING, seed)
            model.fit(training[FEATURES], training.churn, sample_weight=compute_sample_weight("balanced", training.churn))
            probability = probability_function(model, FEATURES)
            explainer = shap.PermutationExplainer(probability,
                shap.maskers.Independent(reference[FEATURES].to_numpy(), max_samples=BACKGROUND_ROWS),
                feature_names=FEATURES, seed=42 + fold)
            explained = explainer(cases[FEATURES].to_numpy(), max_evals=MAX_EVALS, silent=True)
            score = probability(cases[FEATURES].to_numpy())
            error = verify_reconstruction(explained.base_values, explained.values, score)
            np.testing.assert_allclose(explained.base_values, probability(reference[FEATURES].to_numpy()).mean(), atol=1e-10, rtol=0)
            checks.append({"outer_fold": fold, "seed": seed, "max_reconstruction_error": error})
            seed_rows = cases[["StoreID", "snapshot"]].copy()
            seed_rows["outer_fold"], seed_rows["seed"] = fold, seed
            seed_rows["base_probability"], seed_rows["churn_probability"] = explained.base_values, score
            for index, feature in enumerate(FEATURES):
                seed_rows[f"shap_{feature}"] = explained.values[:, index]
            per_seed.append(seed_rows)
            models.append(model)
            values.append(explained.values)
            bases.append(explained.base_values)
            scores.append(score)
        contributions, baseline, score = np.mean(values, axis=0), np.mean(bases, axis=0), np.mean(scores, axis=0)
        verify_reconstruction(baseline, contributions, score)
        np.testing.assert_allclose(score, saved.churn_probability, atol=1e-10, rtol=0)
        row = cases[["StoreID", "snapshot", *FEATURES]].copy()
        row["actual_churn"], row["outer_fold"], row["threshold"] = cases.churn, fold, threshold
        row["churn_probability"], row["base_probability"] = score, baseline
        row["error_category"] = error_categories(cases.churn, score, threshold)
        row["alert"] = score >= threshold
        records.append(row)
        mean_values.append(contributions)
        mean_bases.append(baseline)
        bundles[fold] = {"models": models, "features": FEATURES, "background": reference[["StoreID", "snapshot", *FEATURES]]}
        joblib.dump(bundles[fold], model_directory / f"fold_{fold}.joblib")
        print(f"Fold {fold}: reproduced and explained {len(cases)} held-out rows across five seeds", flush=True)
    cases = pd.concat(records, ignore_index=True)
    contributions, bases = np.concatenate(mean_values), np.concatenate(mean_bases)
    if len(cases) != 1375 or cases[["StoreID", "snapshot"]].duplicated().any():
        raise ValueError("Incomplete held-out explanation coverage")
    examples = pick_examples(cases)
    importance, by_error, by_fold = contribution_tables(cases, contributions)
    # Repeat the walkthrough with twice the evaluation budget and different permutations.
    example = examples[examples.example.eq("walkthrough")].iloc[0]
    index = cases.set_index(["StoreID", "snapshot"]).index.get_loc((example.StoreID, example.snapshot))
    bundle = bundles[int(example.outer_fold)]
    repeated_values = []
    for model in bundle["models"]:
        explainer = shap.PermutationExplainer(probability_function(model, FEATURES),
            shap.maskers.Independent(bundle["background"][FEATURES].to_numpy(), max_samples=BACKGROUND_ROWS), seed=78)
        repeated = explainer(cases.iloc[[index]][FEATURES].to_numpy(), max_evals=420, silent=True)
        repeated_values.append(repeated.values[0])
    stability = pd.DataFrame({"feature": FEATURES, "initial_contribution": contributions[index],
                              "repeated_contribution": np.mean(repeated_values, axis=0)})
    stability["absolute_difference"] = abs(stability.initial_contribution - stability.repeated_contribution)
    cases.to_csv(output / "held_out_cases.csv", index=False)
    values_table = cases[["StoreID", "snapshot", "outer_fold"]].copy()
    for index, feature in enumerate(FEATURES):
        values_table[f"shap_{feature}"] = contributions[:, index]
    for name, frame in {"ensemble_contributions.csv": values_table, "individual_seed_explanations.csv": pd.concat(per_seed, ignore_index=True),
        "background_rows.csv": pd.concat(backgrounds, ignore_index=True), "examples.csv": examples,
        "global_importance.csv": importance, "importance_by_error.csv": by_error,
        "importance_by_fold.csv": by_fold, "walkthrough_sampling_stability.csv": stability,
        "reconstruction_checks.csv": pd.DataFrame(checks)}.items():
        frame.to_csv(output / name, index=False)
    render_plots(output, cases, contributions, bases, examples, importance)
    code = [Path(__file__), ROOT / "debug/nested_revenue_comparison.py", ROOT / "debug/compare_subtotal_totaldue.py",
            ROOT / "reseller_churn/modeling/model_factory.py", ROOT / "reseller_churn/config/model_settings.py"]
    inputs = [preparation / "development_subtotal.csv", preparation / "preparation_manifest.json",
              nested / "protocol.json", source / "outer_predictions.csv", source / "completion.json"]
    write_json(output / "manifest.json", {"models_refitted": 25, "explained_rows": len(cases),
        "october_accessed": False, "elapsed_seconds": monotonic() - start, "repository": repository_identity(),
        "versions": {"shap": shap.__version__, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "max_reconstruction_error": max(item["max_reconstruction_error"] for item in checks),
        "walkthrough_sampling_max_difference": float(stability.absolute_difference.max()),
        "input_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in inputs},
        "code_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in code},
        "model_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in model_directory.iterdir()},
        "output_sha256": {str(p.relative_to(output)): file_sha256(p) for p in output.rglob('*') if p.is_file()},
    })
    print(importance.to_string(index=False))
    print(examples[["example", "StoreID", "snapshot", "churn_probability", "threshold", "actual_churn"]].to_string(index=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--nested", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reseller-shap-mpl-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        run(args.preparation.resolve(), args.nested.resolve(), args.run_id)
