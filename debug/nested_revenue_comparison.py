"""Repeated nested reseller validation of the frozen revenue variants."""

import argparse
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compare_seven_ten_features import FEATURE_SETS, fit_scores
from compare_subtotal_totaldue import CONFIGURATIONS, checked_hash, load_pair
from reseller_churn.config.model_settings import MODEL_SEEDS
from reseller_churn.config.project_paths import get_run_directory
from reseller_churn.evaluation.metrics import calculate_ranking_metrics, calculate_threshold_metrics
from reseller_churn.evaluation.selection import SELECTION_RULE, THRESHOLDS, evaluate_threshold_tradeoffs, select_for_precision_floor
from reseller_churn.modeling.development import file_sha256, repository_identity, write_json

REPEAT_SEEDS = (42, 78, 88)
BOOTSTRAP_SEED = 20261015
BOOTSTRAP_DRAWS = 5000
MINIMUM_GAIN = .01
CASE_KEYS = ["StoreID", "snapshot"]


def split_groups(data, seed):
    """Positional folds; enforce class support, group separation, and coverage."""
    if data[CASE_KEYS].duplicated().any():
        raise ValueError("Duplicate reseller/snapshot keys")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(data, data.churn, groups=data.StoreID))
    held_out = []
    for train, validation in folds:
        left, right = data.iloc[train], data.iloc[validation]
        if set(left.StoreID) & set(right.StoreID):
            raise ValueError("Reseller crosses split boundary")
        if left.churn.nunique() != 2 or right.churn.nunique() != 2:
            raise ValueError("Both split sides must contain both classes")
        held_out.extend(validation)
    if sorted(held_out) != list(range(len(data))):
        raise ValueError("Held-out folds do not cover each row once")
    return folds


def nested_splits(data, seed):
    """Map inner positions to original rows, keeping outer cases excluded."""
    result = []
    for train, validation in split_groups(data, seed):
        inner = [(train[left], train[right]) for left, right in split_groups(data.iloc[train], seed)]
        outer_ids = set(data.iloc[validation].StoreID)
        for left, right in inner:
            if outer_ids & set(data.iloc[np.concatenate([left, right])].StoreID):
                raise ValueError("Outer reseller entered inner selection")
        result.append((train, validation, inner))
    return result


def fit_probabilities(frame, train, validation, architecture, columns):
    """Return mean probabilities and seed records, fitting only supplied rows."""
    training, cases = frame.iloc[train], frame.iloc[validation]
    records, scores = [], []
    for seed in MODEL_SEEDS:
        score = fit_scores(architecture, columns, training, cases, seed)
        item = cases[[*CASE_KEYS, "n_orders_3m"]].copy()
        item["actual_churn"] = cases.churn.to_numpy()
        item["churn_probability"], item["seed"] = score, seed
        records.append(item)
        scores.append(score)
    ensemble = cases[[*CASE_KEYS, "n_orders_3m"]].copy()
    ensemble["actual_churn"] = cases.churn.to_numpy()
    ensemble["churn_probability"] = np.mean(scores, axis=0)
    return ensemble, pd.concat(records, ignore_index=True)


def select_inner(predictions, candidate):
    folds = {fold: {"target": group.actual_churn.to_numpy(), "probabilities": group.churn_probability.to_numpy()}
             for fold, group in predictions.groupby("inner_fold")}
    grid = evaluate_threshold_tradeoffs(candidate, folds)
    return select_for_precision_floor(grid, .30), grid


def operating_rows(cases, threshold, status, scope):
    """Keep ranking scores and status explicit even without an eligible cutoff."""
    rows = []
    for subgroup, group in [
        ("overall", cases), ("purchased_in_3m", cases[cases.n_orders_3m.gt(0)]),
        ("no_purchase_in_3m", cases[cases.n_orders_3m.eq(0)]),
    ]:
        if group.empty:
            continue
        row = {**scope, "subgroup": subgroup, "status": status, "threshold": threshold,
               "rows": len(group), "churners": int(group.actual_churn.sum()),
               **calculate_ranking_metrics(group.actual_churn, group.churn_probability)}
        if status == "selected":
            metrics = calculate_threshold_metrics(group.actual_churn, group.churn_probability, threshold)
            row.update(metrics)
            row["meets_precision_floor"] = metrics["precision"] >= .30
        rows.append(row)
    return rows


def pooled_rows(predictions):
    """Pool decisions from varying outer thresholds only for complete repeats."""
    rows = []
    for (architecture, definition), cases in predictions.groupby(["architecture", "feature_set"]):
        scope = {"architecture": architecture, "feature_set": definition}
        complete = cases.status.eq("selected").all() and cases.outer_fold.nunique() == 5
        if not complete:
            rows.append({**scope, "status": "incomplete_inner_selection", "subgroup": "overall",
                         "threshold": "varies_by_outer_fold", "rows": len(cases)})
            continue
        for subgroup, group in [
            ("overall", cases), ("purchased_in_3m", cases[cases.n_orders_3m.gt(0)]),
            ("no_purchase_in_3m", cases[cases.n_orders_3m.eq(0)]),
        ]:
            if group.empty:
                continue
            # Use saved fold-specific decisions rather than a single pooled cutoff.
            metrics = calculate_threshold_metrics(group.actual_churn, group.alert.to_numpy(dtype=float), .5)
            rows.append({**scope, "subgroup": subgroup, "status": "selected",
                         "threshold": "varies_by_outer_fold", "rows": len(group),
                         "churners": int(group.actual_churn.sum()),
                         "meets_precision_floor": metrics["precision"] >= .30,
                         **metrics, **calculate_ranking_metrics(group.actual_churn, group.churn_probability)})
    return pd.DataFrame(rows)


def bootstrap_differences(predictions, draws=BOOTSTRAP_DRAWS):
    """Pair resampled reseller clusters across variants and all split repeats."""
    stores = np.sort(predictions.StoreID.unique())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(stores), np.full(len(stores), 1 / len(stores)), size=draws)
    rows = []
    for architecture, cases in predictions.groupby("architecture"):
        if not cases.status.eq("selected").all():
            rows.append({"architecture": architecture, "status": "incomplete_inner_selection"})
            continue
        # Equal-weight means of each repeat's pooled metrics, not independent repeats.
        values = {name: {"recall": [], "precision": []} for name in ["subtotal", "totaldue"]}
        for _, repeat in cases.groupby("repeat_seed"):
            left = repeat[repeat.feature_set.eq("subtotal")].sort_values(CASE_KEYS).reset_index(drop=True)
            right = repeat[repeat.feature_set.eq("totaldue")].sort_values(CASE_KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(left[CASE_KEYS + ["actual_churn"]], right[CASE_KEYS + ["actual_churn"]], check_exact=True)
            for definition, group in [("subtotal", left), ("totaldue", right)]:
                counts = pd.DataFrame({"StoreID": group.StoreID,
                    "tp": group.actual_churn.eq(1) & group.alert.eq(1),
                    "fp": group.actual_churn.eq(0) & group.alert.eq(1),
                    "fn": group.actual_churn.eq(1) & group.alert.eq(0)})
                per_store = counts.groupby("StoreID")[["tp", "fp", "fn"]].sum().reindex(stores, fill_value=0)
                sampled = weights @ per_store.to_numpy()
                tp, fp, fn = sampled.T
                values[definition]["recall"].append(np.divide(tp, tp + fn, out=np.zeros(draws), where=(tp + fn) != 0))
                values[definition]["precision"].append(np.divide(tp, tp + fp, out=np.zeros(draws), where=(tp + fp) != 0))
        for metric in ["recall", "precision"]:
            differences = np.mean(values["totaldue"][metric], axis=0) - np.mean(values["subtotal"][metric], axis=0)
            lower, upper = np.quantile(differences, [.025, .975])
            adjusted_lower, adjusted_upper = np.quantile(differences, [.00625, .99375])
            rows.append({"architecture": architecture, "metric": metric, "status": "estimated",
                         "lower_95": lower, "upper_95": upper,
                         "lower_98_75": adjusted_lower, "upper_98_75": adjusted_upper,
                         "bootstrap_draws": draws, "bootstrap_seed": BOOTSTRAP_SEED})
    return pd.DataFrame(rows)


def classify_gain(recall_gain, precision_gain, recall_lower, precision_lower, recall_upper, precision_upper, floors_pass):
    if floors_pass and recall_lower >= 0 and precision_lower >= 0 and max(recall_gain, precision_gain) >= MINIMUM_GAIN:
        return "totaldue_supported_conditionally"
    if floors_pass and recall_upper <= 0 and precision_upper <= 0 and max(-recall_gain, -precision_gain) >= MINIMUM_GAIN:
        return "subtotal_supported_conditionally"
    if recall_gain * precision_gain < 0:
        return "tradeoff_not_dominance"
    return "inconclusive"


def run_repeat(output, datasets, seed):
    start = monotonic()
    baseline = datasets["subtotal"]
    splits = nested_splits(baseline, seed)
    memberships, counts = [], []
    for outer_number, (train, validation, inner) in enumerate(splits, start=1):
        for stage, fold, left, right in [("outer", 0, train, validation), *[
            ("inner", number, a, b) for number, (a, b) in enumerate(inner, start=1)]]:
            for role, indices in [("training", left), ("validation", right)]:
                members = baseline.iloc[indices][[*CASE_KEYS, "churn"]].copy()
                members["stage"], members["outer_fold"], members["inner_fold"], members["role"] = stage, outer_number, fold, role
                memberships.append(members)
                counts.append({"stage": stage, "outer_fold": outer_number, "inner_fold": fold,
                               "role": role, "rows": len(members), "resellers": members.StoreID.nunique(),
                               "churners": int(members.churn.sum())})
    output.mkdir()
    pd.concat(memberships, ignore_index=True).to_csv(output / "memberships.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "fold_counts.csv", index=False)
    inner_scores, seed_scores, outer_scores, grids, selections, metrics = [], [], [], [], [], []
    fit_count = 0
    for outer_number, (train, validation, inner) in enumerate(splits, start=1):
        for architecture, feature_set in CONFIGURATIONS.items():
            for definition, frame in datasets.items():
                scope = {"architecture": architecture, "feature_set": definition, "outer_fold": outer_number, "repeat_seed": seed}
                predictions = []
                for inner_number, (left, right) in enumerate(inner, start=1):
                    mean, individual = fit_probabilities(frame, left, right, architecture, FEATURE_SETS[feature_set])
                    for table in [mean, individual]:
                        for name, value in scope.items():
                            table[name] = value
                        table["inner_fold"], table["stage"] = inner_number, "inner"
                    predictions.append(mean)
                    inner_scores.append(mean)
                    seed_scores.append(individual)
                    fit_count += len(MODEL_SEEDS)
                selection, grid = select_inner(pd.concat(predictions, ignore_index=True), f"{architecture}_{definition}")
                for name, value in scope.items():
                    grid[name] = value
                grids.append(grid)
                selections.append({**selection, **scope})
                mean, individual = fit_probabilities(frame, train, validation, architecture, FEATURE_SETS[feature_set])
                for table in [mean, individual]:
                    for name, value in scope.items():
                        table[name] = value
                    table["inner_fold"], table["stage"] = 0, "outer"
                mean["status"], mean["threshold"] = selection["status"], selection["threshold"]
                mean["alert"] = (mean.churn_probability.ge(selection["threshold"]).astype(int)
                                 if selection["status"] == "selected" else np.nan)
                outer_scores.append(mean)
                seed_scores.append(individual)
                fit_count += len(MODEL_SEEDS)
                metrics.extend(operating_rows(mean, selection["threshold"], selection["status"], scope))
            print(f"Repeat {seed} outer fold {outer_number}: {architecture}; {fit_count}/600 fits", flush=True)
    if fit_count != 600:
        raise ValueError("Unexpected fit count")
    outer = pd.concat(outer_scores, ignore_index=True)
    tables = {"inner_predictions.csv": pd.concat(inner_scores, ignore_index=True),
              "individual_predictions.csv": pd.concat(seed_scores, ignore_index=True),
              "outer_predictions.csv": outer, "inner_threshold_tradeoffs.csv": pd.concat(grids, ignore_index=True),
              "selected_thresholds.csv": pd.DataFrame(selections), "outer_metrics.csv": pd.DataFrame(metrics),
              "pooled_metrics.csv": pooled_rows(outer)}
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    write_json(output / "completion.json", {"repeat_seed": seed, "models_fitted": fit_count,
        "elapsed_seconds": monotonic() - start,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}})
    print(tables["pooled_metrics.csv"].query("subgroup == 'overall'").to_string(index=False), flush=True)


def analyze(output):
    analysis = output / "analysis"
    if analysis.exists():
        raise FileExistsError("Completed analysis exists; preserve it")
    predictions, metrics, selections = [], [], []
    for seed in REPEAT_SEEDS:
        folder = output / f"repeat_{seed}"
        for destination, name in [(predictions, "outer_predictions.csv"), (metrics, "pooled_metrics.csv"),
                                  (selections, "selected_thresholds.csv")]:
            frame = pd.read_csv(folder / name, float_precision="round_trip")
            frame["repeat_seed"] = seed
            destination.append(frame)
    predictions = pd.concat(predictions, ignore_index=True)
    pooled = pd.concat(metrics, ignore_index=True)
    thresholds = pd.concat(selections, ignore_index=True)
    intervals = bootstrap_differences(predictions)
    comparisons = []
    overall = pooled[pooled.subgroup.eq("overall")]
    for architecture, group in overall.groupby("architecture"):
        if not group.status.eq("selected").all():
            comparisons.append({"architecture": architecture, "conclusion": "incomplete_inner_selection"})
            continue
        means = group.groupby("feature_set")[["recall", "precision"]].mean()
        gains = means.loc["totaldue"] - means.loc["subtotal"]
        bounds = intervals[intervals.architecture.eq(architecture)].set_index("metric")
        floors_pass = group.precision.ge(.30).all()
        conclusion = classify_gain(gains.recall, gains.precision, bounds.loc["recall", "lower_98_75"],
            bounds.loc["precision", "lower_98_75"], bounds.loc["recall", "upper_98_75"],
            bounds.loc["precision", "upper_98_75"], floors_pass)
        comparisons.append({"architecture": architecture, "subtotal_recall": means.loc["subtotal", "recall"],
            "totaldue_recall": means.loc["totaldue", "recall"], "subtotal_precision": means.loc["subtotal", "precision"],
            "totaldue_precision": means.loc["totaldue", "precision"], "delta_recall": gains.recall,
            "delta_precision": gains.precision, "pooled_floor_passes_all_repeats": bool(floors_pass), "conclusion": conclusion})
    stability = thresholds.groupby(["architecture", "feature_set"]).threshold.agg(["min", "median", "max"]).reset_index()
    analysis.mkdir()
    for name, table in {"paired_intervals.csv": intervals, "comparison_summary.csv": pd.DataFrame(comparisons),
                        "threshold_stability.csv": stability, "repeat_pooled_metrics.csv": pooled,
                        "all_selected_thresholds.csv": thresholds}.items():
        table.to_csv(analysis / name, index=False)
    write_json(analysis / "manifest.json", {"models_fitted": 1800, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED, "uncertainty": "Conditional on fitted models/cutoffs; repeats are dependent",
        "output_sha256": {p.name: file_sha256(p) for p in analysis.iterdir() if p.is_file()}})
    print(pd.DataFrame(comparisons).to_string(index=False))


def run(preparation, run_id, resume=False, first_repeat_only=False):
    get_run_directory(run_id)
    output = ROOT / "results/debugging/subtotal_vs_totaldue" / run_id
    datasets = load_pair(preparation)
    code_paths = [Path(__file__), *[ROOT / path for path in [
        "debug/compare_seven_ten_features.py", "debug/compare_subtotal_totaldue.py",
        "reseller_churn/config/model_settings.py", "reseller_churn/modeling/model_factory.py",
        "reseller_churn/evaluation/selection.py", "reseller_churn/evaluation/metrics.py"]]]
    inputs = [preparation / name for name in ["preparation_manifest.json", "development_subtotal.csv", "development_totaldue.csv"]]
    versions = {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}
    if resume:
        protocol = json.loads((output / "protocol.json").read_text())
        for category in ["input_sha256", "code_sha256"]:
            for name, expected in protocol[category].items():
                checked_hash(ROOT / name, expected)
        if protocol["versions"] != versions or protocol["repeat_seeds"] != list(REPEAT_SEEDS):
            raise ValueError("Resume conditions differ")
    else:
        if output.exists():
            raise FileExistsError("Run exists; preserve it or resume completed repeats explicitly")
        output.mkdir(parents=True)
        write_json(output / "protocol.json", {"repeat_seeds": list(REPEAT_SEEDS), "outer_folds": 5, "inner_folds": 5,
            "model_seeds": list(MODEL_SEEDS), "planned_fits": 1800, "precision_floor": .30,
            "selection_rule": SELECTION_RULE, "thresholds": list(THRESHOLDS), "minimum_gain": MINIMUM_GAIN,
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
            "final_outcomes_accessed": False, "versions": versions, "repository": repository_identity(),
            "input_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in inputs},
            "code_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in code_paths}})
        text = (ROOT / "docs/debugging.md").read_text().split("## Nested grouped revenue study: protocol — 2026-10-15", 1)[1]
        (output / "protocol.md").write_text("# Nested grouped revenue study\n" + text)
    for seed in REPEAT_SEEDS:
        folder = output / f"repeat_{seed}"
        if folder.exists():
            completion = json.loads((folder / "completion.json").read_text())
            for name, expected in completion["output_sha256"].items():
                checked_hash(folder / name, expected)
            if completion["models_fitted"] != 600:
                raise ValueError("Incomplete repeat fit count")
            continue
        run_repeat(folder, datasets, seed)
        if first_repeat_only:
            print("First-repeat execution check complete; remaining frozen repeats require --resume.")
            return output
    analyze(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--first-repeat-only", action="store_true")
    args = parser.parse_args()
    run(args.preparation.resolve(), args.run_id, args.resume, args.first_repeat_only)
