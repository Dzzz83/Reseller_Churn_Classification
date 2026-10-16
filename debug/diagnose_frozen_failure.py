"""Describe saved prediction failures; never fit models or choose cutoffs.

This is exploratory reuse of an already inspected October cohort. All inputs
are explicit saved artifacts. No outcomes are constructed and no alternative
model, feature set, or decision rule is evaluated on October.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

KEYS = ["StoreID", "snapshot"]
CANDIDATE = "frozen_cadence_candidate"
BASELINE = "original_baseline"
FEATURES = [
    "n_orders_3m",
    "recency_days",
    "n_orders_12m",
    "overdue_ratio",
    "gap_cv",
    "revenue_3m",
    "revenue_12m",
    "share_bikes",
    "weighted_discount_3m",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_csv(directory, name, manifest):
    path = directory / name
    if sha(path) != manifest["outputs"][name]:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return pd.read_csv(path, float_precision="round_trip")


def matched_cases(evaluated, features):
    """Require identical populations, outcomes and saved frozen decisions."""
    if set(evaluated.strategy) != {CANDIDATE, BASELINE}:
        raise ValueError("Expected only the two previously evaluated rules")
    if evaluated.duplicated([*KEYS, "strategy"]).any():
        raise ValueError("Duplicate prediction keys")
    if features.duplicated(KEYS).any() or "churn" in features:
        raise ValueError("Expected unique unlabeled features")
    if not evaluated.churn.isin([0, 1]).all():
        raise ValueError("Outcomes must be binary")
    if not np.array_equal(
        evaluated.predicted_churn, evaluated.score.ge(evaluated.cutoff).astype(int)
    ):
        raise ValueError("Saved decisions differ from saved cutoffs")
    left = evaluated[evaluated.strategy.eq(CANDIDATE)].drop(columns="strategy")
    right = evaluated[evaluated.strategy.eq(BASELINE)].drop(columns="strategy")
    paired = left.merge(
        right,
        on=KEYS,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
        how="outer",
        indicator=True,
    )
    if not paired._merge.eq("both").all():
        raise ValueError("Prediction populations differ")
    for col in ["churn", "n_orders_3m"]:
        if not paired[f"{col}_candidate"].equals(paired[f"{col}_baseline"]):
            raise ValueError(f"Paired {col} differs")
    paired = paired.drop(columns="_merge").merge(
        features[KEYS + FEATURES],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not paired._merge.eq("both").all():
        raise ValueError("Feature and prediction populations differ")
    if not paired.n_orders_3m.equals(paired.n_orders_3m_candidate):
        raise ValueError("Scored activity differs from feature activity")
    paired = paired.drop(columns="_merge")
    paired["churn"] = paired.churn_candidate
    b = paired.predicted_churn_baseline.eq(1)
    c = paired.predicted_churn_candidate.eq(1)
    y = paired.churn.eq(1)
    paired["change"] = np.select(
        [~b & c & ~y, b & ~c & ~y, ~b & c & y, ~c & y, b & c & ~y, ~b & ~c & ~y],
        [
            "added_false_alert",
            "removed_false_alert",
            "newly_caught",
            "still_missed",
            "shared_false_alert",
            "shared_correct_negative",
        ],
        default="shared_caught",
    )
    return paired


def metrics(frame, prediction, score):
    y = frame.churn.eq(1)
    p = frame[prediction].astype(bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    positives, negatives = int(y.sum()), int((~y).sum())
    return dict(
        rows=len(frame),
        churners=positives,
        tp=tp,
        fp=fp,
        recall=tp / positives if positives else np.nan,
        false_positive_rate=fp / negatives if negatives else np.nan,
        precision=tp / (tp + fp) if tp + fp else np.nan,
        prevalence=float(y.mean()),
        average_precision=average_precision_score(y, frame[score])
        if positives
        else np.nan,
    )


def grouped_diagnostics(predictions, development, strategy_column):
    """Keep repeat estimates separate and join only matching development keys."""
    if development.duplicated(KEYS).any():
        raise ValueError("Duplicate development keys")
    rows = []
    for (strategy, repeat), group in predictions.groupby([strategy_column, "repeat"]):
        joined = group.merge(
            development,
            on=KEYS,
            how="outer",
            validate="one_to_one",
            suffixes=("", "_feature"),
            indicator=True,
        )
        if not joined._merge.eq("both").all():
            raise ValueError("Development diagnostic populations differ")
        for column in ["churn", "n_orders_3m"]:
            if not np.array_equal(joined[column], joined[f"{column}_feature"]):
                raise ValueError(f"Development diagnostic {column} differs")
        joined["prediction"] = joined.score.ge(joined.cutoff)
        active = joined.n_orders_3m.gt(0)
        segments = [
            ("activity", "overall", joined),
            ("activity", "active", joined[active]),
            ("activity", "inactive", joined[~active]),
        ]
        for snapshot, part in joined.groupby("snapshot"):
            segments.append(("snapshot", snapshot, part))
            segments.append(("active_snapshot", snapshot, part[part.n_orders_3m.gt(0)]))
        for label, part in joined[active].groupby(
            pd.cut(joined.loc[active, "n_orders_12m"], [0, 1, 2, np.inf]), observed=True
        ):
            segments.append(("active_annual_orders", str(label), part))
        for label, part in joined[active].groupby(
            pd.cut(joined.loc[active, "recency_days"], [-1, 30, 60, 90]), observed=True
        ):
            segments.append(("active_recency", str(label), part))
        for dimension, segment, part in segments:
            rows.append(
                dict(
                    strategy=strategy,
                    repeat=repeat,
                    dimension=dimension,
                    segment=segment,
                    **metrics(part, "prediction", "score"),
                )
            )
        for fold, part in joined.groupby("outer_fold"):
            rows.append(
                dict(
                    strategy=strategy,
                    repeat=repeat,
                    dimension="outer_fold",
                    segment=str(fold),
                    **metrics(part, "prediction", "score"),
                )
            )
    return pd.DataFrame(rows)


def precision_at_prevalence(recall, false_positive_rate, prevalence):
    true_alerts = recall * prevalence
    false_alerts = false_positive_rate * (1 - prevalence)
    denominator = true_alerts + false_alerts
    return true_alerts / denominator if denominator else np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    em = json.loads((args.evaluation / "evaluation_manifest.json").read_text())
    if sha(args.evaluation / "protocol.json") != em["outputs"]["protocol.json"]:
        raise ValueError("Evaluation protocol checksum mismatch")
    protocol = json.loads((args.evaluation / "protocol.json").read_text())
    dm = json.loads((args.development / "manifest.json").read_text())
    evaluated = checked_csv(args.evaluation, "evaluated_cases.csv", em)
    if sha(args.features) != protocol["features_sha256"]:
        raise ValueError("Scoring features changed since evaluation")
    scoring_features = pd.read_csv(args.features, float_precision="round_trip")
    if (
        not pd.to_datetime(scoring_features.last_order)
        .lt(pd.to_datetime(scoring_features.snapshot))
        .all()
    ):
        raise ValueError("Saved latest orders reach or exceed the prediction date")
    cases = matched_cases(evaluated, scoring_features)
    if set(cases.snapshot) != {"2013-10-01"}:
        raise ValueError("Only the already inspected October cohort is supported")
    dev = checked_csv(args.development, "development_input.csv", dm)
    predictions = checked_csv(args.development, "validation_predictions.csv", dm)
    if dev.duplicated(KEYS).any():
        raise ValueError("Duplicate development keys")
    allowed = {"2012-07-01", "2012-10-01", "2013-01-01", "2013-04-01"}
    if set(dev.snapshot) != allowed:
        raise ValueError("Unexpected development snapshots")
    model_path = args.development / "research_candidate.joblib"
    if sha(model_path) != protocol["model_sha256"]:
        raise ValueError("Frozen model identity changed")
    specification = json.loads((args.development / "frozen_candidate.json").read_text())
    if specification != protocol["model_specification"]:
        raise ValueError("Frozen specification changed")
    if (
        not evaluated.loc[evaluated.strategy.eq(CANDIDATE), "cutoff"]
        .eq(specification["cutoff"])
        .all()
    ):
        raise ValueError("Candidate cutoff changed")
    if not evaluated.loc[evaluated.strategy.eq(BASELINE), "cutoff"].eq(0.06).all():
        raise ValueError("Historical baseline cutoff changed")
    am = json.loads((args.ablation / "manifest.json").read_text())
    ablation_predictions = checked_csv(args.ablation, "validation_predictions.csv", am)
    ablation_dev = checked_csv(args.ablation, "development_input.csv", am)
    if not dev.equals(ablation_dev):
        raise ValueError("Ablation development features differ")
    arm = json.loads((args.ablation / "report_manifest.json").read_text())
    if sha(args.ablation / "manifest.json") != arm["source_manifest_sha256"]:
        raise ValueError("Ablation report belongs to a different source run")
    components = checked_csv(args.ablation, "component_effects.csv", arm)
    stability = grouped_diagnostics(
        predictions[
            predictions.strategy.isin(
                [
                    "cadence_regularized",
                    "original_baseline",
                    "original_fixed_006",
                    "adaptive_selection",
                ]
            )
        ],
        dev,
        "strategy",
    )
    ablation_stability = grouped_diagnostics(ablation_predictions, dev, "strategy")
    selected = stability[stability.strategy.eq("cadence_regularized")]
    reference = stability[stability.strategy.eq("original_baseline")]
    effects = selected.merge(
        reference,
        on=["repeat", "dimension", "segment"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    for metric in ["recall", "precision", "false_positive_rate", "average_precision"]:
        effects[f"{metric}_delta"] = (
            effects[f"{metric}_candidate"] - effects[f"{metric}_baseline"]
        )
    rows = []
    for (strategy, repeat), group in predictions.groupby(["strategy", "repeat"]):
        if strategy not in {
            "cadence_regularized",
            "original_baseline",
            "original_fixed_006",
        }:
            continue
        joined = group.merge(
            dev[KEYS + ["churn"]],
            on=KEYS,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not joined._merge.eq("both").all() or not joined.churn_x.equals(
            joined.churn_y
        ):
            raise ValueError("Development prediction population/outcomes differ")
        group = group.copy()
        group["predicted"] = group.score.ge(group.cutoff)
        for subgroup in ["overall", "active", "inactive"]:
            part = (
                group
                if subgroup == "overall"
                else group[
                    group.n_orders_3m.gt(0)
                    if subgroup == "active"
                    else group.n_orders_3m.eq(0)
                ]
            )
            rows.append(
                dict(
                    cohort="grouped_validation",
                    strategy=strategy,
                    repeat=repeat,
                    subgroup=subgroup,
                    **metrics(part, "predicted", "score"),
                )
            )
    for rule in ["candidate", "baseline"]:
        for subgroup in ["overall", "active", "inactive"]:
            part = (
                cases
                if subgroup == "overall"
                else cases[
                    cases.n_orders_3m.gt(0)
                    if subgroup == "active"
                    else cases.n_orders_3m.eq(0)
                ]
            )
            rows.append(
                dict(
                    cohort="October",
                    strategy=rule,
                    repeat=-1,
                    subgroup=subgroup,
                    **metrics(part, f"predicted_churn_{rule}", f"score_{rule}"),
                )
            )
    cohort_metrics = pd.DataFrame(rows)
    standardized = []
    for (strategy, repeat), group in cohort_metrics[
        cohort_metrics.cohort.eq("grouped_validation")
        & cohort_metrics.subgroup.ne("overall")
    ].groupby(["strategy", "repeat"]):
        total_tp, total_fp = 0.0, 0.0
        for row in group.itertuples():
            population = cases[
                cases.n_orders_3m.gt(0)
                if row.subgroup == "active"
                else cases.n_orders_3m.eq(0)
            ]
            positives = int(population.churn.sum())
            negatives = len(population) - positives
            tp, fp = positives * row.recall, negatives * row.false_positive_rate
            total_tp += tp
            total_fp += fp
            standardized.append(
                dict(
                    strategy=strategy,
                    repeat=repeat,
                    subgroup=row.subgroup,
                    expected_tp=tp,
                    expected_fp=fp,
                    precision=precision_at_prevalence(
                        row.recall, row.false_positive_rate, positives / len(population)
                    ),
                )
            )
        standardized.append(
            dict(
                strategy=strategy,
                repeat=repeat,
                subgroup="overall",
                expected_tp=total_tp,
                expected_fp=total_fp,
                precision=total_tp / (total_tp + total_fp),
            )
        )
    profile_rows, segment_rows = [], []
    for name, part in cases.groupby("change"):
        for feature in FEATURES:
            profile_rows.append(
                dict(
                    change=name,
                    rows=len(part),
                    feature=feature,
                    median=part[feature].median(),
                    missing=int(part[feature].isna().sum()),
                )
            )
    # Post-hoc descriptive bands, not learned rules or proposed test cutoffs.
    for cohort, population in [("development", dev), ("October", cases)]:
        active = population[population.n_orders_3m.gt(0)].copy()
        bands = {
            "annual_orders": pd.cut(active.n_orders_12m, [0, 1, 2, np.inf]),
            "recency_days": pd.cut(active.recency_days, [-1, 30, 60, 90]),
        }
        for feature, band in bands.items():
            for label, part in active.groupby(band, observed=True):
                segment_rows.append(
                    dict(
                        cohort=cohort,
                        feature=feature,
                        band=str(label),
                        rows=len(part),
                        churners=int(part.churn.sum()),
                        prevalence=float(part.churn.mean()),
                    )
                )
    feature_coverage = []
    for cohort, population in [("development", dev), ("October", scoring_features)]:
        for feature in specification["candidate"]["features"]:
            values = population[feature]
            if np.isinf(values.to_numpy(dtype=float)).any():
                raise ValueError(f"Infinite feature values: {feature}")
            feature_coverage.append(
                dict(
                    cohort=cohort,
                    feature=feature,
                    rows=len(values),
                    missing=int(values.isna().sum()),
                    minimum=values.min(),
                    median=values.median(),
                    maximum=values.max(),
                )
            )
    args.output.mkdir(parents=True)
    tables = {
        "account_cases.csv": cases,
        "cohort_metrics.csv": cohort_metrics,
        "feature_profiles.csv": pd.DataFrame(profile_rows),
        "active_segments.csv": pd.DataFrame(segment_rows),
        "active_churners.csv": cases[cases.churn.eq(1) & cases.n_orders_3m.gt(0)],
        "development_stability.csv": stability,
        "development_paired_effects.csv": effects,
        "ablation_stability.csv": ablation_stability,
        "existing_component_effects.csv": components,
        "prevalence_standardization.csv": pd.DataFrame(standardized),
        "feature_coverage.csv": pd.DataFrame(feature_coverage),
    }
    for name, table in tables.items():
        table.to_csv(args.output / name, index=False)
    manifest = dict(
        status="post-hoc descriptive diagnosis of already inspected outcomes",
        models_fitted=0,
        thresholds_optimized=False,
        new_prediction_rules_evaluated=0,
        source_manifests={
            str(args.evaluation / "evaluation_manifest.json"): sha(
                args.evaluation / "evaluation_manifest.json"
            ),
            str(args.development / "manifest.json"): sha(
                args.development / "manifest.json"
            ),
            str(args.ablation / "manifest.json"): sha(args.ablation / "manifest.json"),
            str(args.ablation / "report_manifest.json"): sha(
                args.ablation / "report_manifest.json"
            ),
        },
        feature_sha256=sha(args.features),
        model_sha256=sha(model_path),
        frozen_cutoff=specification["cutoff"],
        script_sha256=sha(__file__),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        outputs={name: sha(args.output / name) for name in tables},
        limitations=[
            "No causal attribution to individual model changes",
            "Repeated grouped predictions are dependent; not additional accounts",
            "Original model identity unavailable; baseline is a saved reproduction",
            "Outcomes supplied by earlier reconstruction; no fresh database audit",
        ],
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(cases.groupby("change").size().to_string())
    print(f"Saved descriptive tables to {args.output}; zero fits or cutoff changes.")


if __name__ == "__main__":
    main()
