"""Describe saved inner/outer operating differences and decision variation."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from comparisons.baseline_diagnosis.inputs import read_csv, require
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]


def decision_variation(aligned):
    a = aligned.score_a.ge(aligned.cutoff_a)
    b = aligned.score_b.ge(aligned.cutoff_b)
    ab = aligned.score_a.ge(aligned.cutoff_b)
    ba = aligned.score_b.ge(aligned.cutoff_a)
    return {
        "rows": len(aligned),
        "actual_disagreement": float(a.ne(b).mean()),
        "cutoff_only_disagreement": float((a.ne(ab).mean() + b.ne(ba).mean()) / 2),
        "score_only_disagreement": float((a.ne(ba).mean() + b.ne(ab).mean()) / 2),
        "nonadditive_diagnostic": True,
    }


def run(last, annual, output):
    require(not output.exists(), "Output directory exists")
    for source in (last, annual):
        verify_manifest(source)
    last_predictions = read_csv(last / "validation_predictions.csv")
    annual_predictions = read_csv(annual / "validation_predictions.csv")
    columns = [*KEYS, "repeat", "outer_fold", "churn", "score", "cutoff", "n_orders_3m"]
    sort = ["repeat", "snapshot", "StoreID"]
    a = (
        last_predictions[last_predictions.strategy.eq("baseline")][columns]
        .sort_values(sort)
        .reset_index(drop=True)
    )
    b = (
        annual_predictions[annual_predictions.strategy.eq("baseline")][columns]
        .sort_values(sort)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=0, atol=1e-12)
    arms = {
        "baseline": (last, "baseline"),
        "last_order_lines": (last, "basket_breadth"),
        "annual_products": (annual, "basket_breadth"),
    }
    diagnose_arms(arms, output, ROOT / "comparisons/cutoff_transfer_diagnosis.md")


def diagnose_arms(arms, output, protocol):
    require(not output.exists(), "Output directory exists")
    for source, _ in arms.values():
        verify_manifest(source)
    transfer, variation, distributions, inner_folds = [], [], [], []
    for arm, (source, strategy) in arms.items():
        predictions = read_csv(source / "validation_predictions.csv")
        predictions = predictions[predictions.strategy.eq(strategy)]
        for (repeat, fold), outer in predictions.groupby(["repeat", "outer_fold"]):
            folder = source / f"repeat_{repeat}/fold_{fold}"
            verify_manifest(folder)
            inner = read_csv(folder / "inner_scores.csv")
            inner = inner[inner.candidate.eq(strategy)]
            require(
                inner.groupby("StoreID").inner_fold.nunique().eq(1).all(),
                "Inner grouping changed",
            )
            require(
                set(inner.StoreID).isdisjoint(outer.StoreID),
                "Inner/outer reseller overlap",
            )
            identities = json.loads((folder / "model_identities.json").read_text())
            outer_seed_range = max(
                row["seed_score_range"]
                for row in identities
                if row["candidate"] == strategy and row["stage"] == "outer"
            )
            inner_seed_range = max(
                row["seed_score_range"]
                for row in identities
                if row["candidate"] == strategy and row["stage"] == "inner"
            )
            cutoff = outer.cutoff.iloc[0]
            require(outer.cutoff.eq(cutoff).all(), "Multiple fold cutoffs")
            for population, frame in [
                ("inner_oof", inner),
                ("outer_validation", outer),
            ]:
                for churn, subset in frame.groupby("churn"):
                    distributions.append(
                        dict(
                            arm=arm,
                            repeat=repeat,
                            fold=fold,
                            population=population,
                            churn=int(churn),
                            rows=len(subset),
                            **{
                                f"score_q{q}": float(subset.score.quantile(q / 100))
                                for q in [5, 25, 50, 75, 95]
                            },
                        )
                    )
            for inner_fold, frame in inner.groupby("inner_fold"):
                alerts = frame.score.ge(cutoff)
                tp = int((alerts & frame.churn.eq(1)).sum())
                inner_folds.append(
                    dict(
                        arm=arm,
                        repeat=repeat,
                        fold=fold,
                        inner_fold=int(inner_fold),
                        rows=len(frame),
                        recall=tp / frame.churn.sum(),
                        precision=tp / max(1, alerts.sum()),
                        alert_rate=float(alerts.mean()),
                        cutoff=float(cutoff),
                    )
                )
            inside, outside = inner.score.ge(cutoff), outer.score.ge(cutoff)
            tp_inner = int((inner.churn.eq(1) & inside).sum())
            tp_outer = int((outer.churn.eq(1) & outside).sum())
            transfer.append(
                {
                    "arm": arm,
                    "repeat": repeat,
                    "fold": fold,
                    "cutoff": cutoff,
                    "outer_seed_range": outer_seed_range,
                    "inner_seed_range": inner_seed_range,
                    "inner_rows": len(inner),
                    "outer_rows": len(outer),
                    "inner_recall": tp_inner / inner.churn.sum(),
                    "outer_recall": tp_outer / outer.churn.sum(),
                    "inner_precision": tp_inner / inside.sum(),
                    "outer_precision": tp_outer / outside.sum(),
                    "inner_alert_rate": inside.mean(),
                    "outer_alert_rate": outside.mean(),
                    "outer_fpr": (outside & outer.churn.eq(0)).sum()
                    / outer.churn.eq(0).sum(),
                }
            )
        for repeat_a, repeat_b in combinations(sorted(predictions.repeat.unique()), 2):
            joined = predictions[predictions.repeat.eq(repeat_a)].merge(
                predictions[predictions.repeat.eq(repeat_b)],
                on=[*KEYS, "churn", "n_orders_3m"],
                validate="one_to_one",
                suffixes=("_a", "_b"),
            )
            require(len(joined) == 1375, "Changed repeat population")
            for subgroup, subset in [
                ("overall", joined),
                ("active", joined[joined.n_orders_3m.gt(0)]),
                ("inactive", joined[joined.n_orders_3m.eq(0)]),
            ]:
                variation.append(
                    {
                        "arm": arm,
                        "repeat_a": repeat_a,
                        "repeat_b": repeat_b,
                        "subgroup": subgroup,
                        **decision_variation(subset),
                    }
                )
    out = new_output_directory(output)
    pd.DataFrame(transfer).to_csv(out / "inner_outer_operating.csv", index=False)
    pd.DataFrame(variation).to_csv(out / "decision_variation.csv", index=False)
    pd.DataFrame(distributions).to_csv(
        out / "class_score_distributions.csv", index=False
    )
    pd.DataFrame(inner_folds).to_csv(out / "inner_fold_operating.csv", index=False)
    (out / "protocol.md").write_bytes(protocol.read_bytes())
    write_manifest(
        out,
        {
            "stage": "saved_score_transfer_diagnosis",
            "models_fitted": 0,
            "cutoffs_selected": 0,
            "october_outcomes_accessed": False,
            "inputs": input_identity(
                {
                    **{
                        arm: source / "manifest.json"
                        for arm, (source, _) in arms.items()
                    },
                    "protocol": protocol,
                }
            ),
            "implementation": input_identity({"script": Path(__file__)}),
        },
    )
    print(
        pd.DataFrame(transfer)
        .groupby("arm")[
            ["inner_recall", "outer_recall", "inner_precision", "outer_precision"]
        ]
        .mean()
        .to_string()
    )
    print(
        pd.DataFrame(variation)
        .groupby(["arm", "subgroup"])[
            [
                "actual_disagreement",
                "cutoff_only_disagreement",
                "score_only_disagreement",
            ]
        ]
        .mean()
        .to_string()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--last", type=Path, default=ROOT / "results/basket-breadth-20261016-serial"
    )
    parser.add_argument(
        "--annual", type=Path, default=ROOT / "results/annual-breadth-20261016"
    )
    parser.add_argument("--sku", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sku:
        source = ROOT / "results/cadence-stability-20261016"
        diagnose_arms(
            {
                "baseline": (source, "baseline"),
                "cadence": (source, "cadence_no_count"),
                "sku": (
                    ROOT / "results/product-mix-comparison-20261017",
                    "cadence_product_mix",
                ),
            },
            args.output,
            ROOT / "comparisons/cutoff_transfer_diagnosis.md",
        )
    else:
        run(args.last, args.annual, args.output)


if __name__ == "__main__":
    main()
