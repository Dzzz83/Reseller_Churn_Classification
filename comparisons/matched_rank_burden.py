"""Descriptive tied-score burden at the saved baseline's exact catch count."""

import argparse
from pathlib import Path

import numpy as np
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
SOURCES = {
    "cadence_no_count": "cadence-stability-20261016",
    "cadence_product_mix": "product-mix-comparison-20261017",
    "cadence_product_mix_bagged": "product-bagging-comparison-20261017",
    "cadence_latent_mix": "latent-mix-comparison-20261017",
}


def burden(y, scores, target):
    require(0 <= target <= np.sum(y), "Infeasible catch target")
    if target == 0:
        return dict(oracle_tp=0, oracle_fp=0)
    bands = (
        pd.DataFrame({"y": y, "score": scores})
        .groupby("score")
        .y.agg(["sum", "size"])
        .sort_index(ascending=False)
    )
    tp, fp = bands["sum"].cumsum(), (bands["size"] - bands["sum"]).cumsum()
    idx = np.flatnonzero(tp.to_numpy() >= target)[0]
    return dict(oracle_tp=int(tp.iloc[idx]), oracle_fp=int(fp.iloc[idx]))


def summarize(part, identifier, cutoff):
    baseline = part[part[identifier].eq("baseline")]
    target = int((baseline.churn.eq(1) & baseline.score.ge(cutoff)).sum())
    reference = burden(baseline.churn.to_numpy(), baseline.score.to_numpy(), target)
    rows = []
    for arm, data in part.groupby(identifier):
        require(
            set(map(tuple, data[KEYS].to_numpy()))
            == set(map(tuple, baseline[KEYS].to_numpy())),
            "Different ranking cases",
        )
        oracle = burden(data.churn.to_numpy(), data.score.to_numpy(), target)
        a, y = data.score.ge(data.cutoff), data.churn.eq(1)
        rows.append(
            dict(
                policy=arm,
                rows=len(data),
                churners=int(y.sum()),
                baseline_target_tp=target,
                **oracle,
                actual_tp=int((a & y).sum()),
                actual_fp=int((a & ~y).sum()),
                baseline_oracle_fp=reference["oracle_fp"],
                oracle_fp_reduction=reference["oracle_fp"] - oracle["oracle_fp"],
            )
        )
    return rows


def run(output):
    rows = []
    for candidate, name in SOURCES.items():
        source = ROOT / "results" / name
        verify_manifest(source)
        for folder in sorted(source.glob("repeat_*/fold_*")):
            verify_manifest(folder)
            outer = read_csv(folder / "predictions.csv")
            identifier = "strategy"
            outer = outer[outer[identifier].isin(["baseline", candidate])]
            repeat, fold = int(outer.repeat.iloc[0]), int(outer.outer_fold.iloc[0])
            cut = outer.loc[outer.strategy.eq("baseline"), "cutoff"].iloc[0]
            for item in summarize(outer, identifier, cut):
                rows.append(
                    dict(
                        candidate=candidate,
                        repeat=repeat,
                        outer_fold=fold,
                        stage="outer",
                        inner_fold=None,
                        **item,
                    )
                )
            inner = read_csv(folder / "inner_scores.csv")
            inner = inner[inner.candidate.isin(["baseline", candidate])].rename(
                columns={"candidate": "strategy"}
            )
            selections = (
                pd.read_csv(
                    folder / "inner_selections.csv", float_precision="round_trip"
                )
                .set_index("candidate")
                .cutoff
            )
            inner["cutoff"] = inner.strategy.map(selections)
            for inner_fold, part in inner.groupby("inner_fold"):
                for item in summarize(part, "strategy", cut):
                    rows.append(
                        dict(
                            candidate=candidate,
                            repeat=repeat,
                            outer_fold=fold,
                            stage="inner",
                            inner_fold=int(inner_fold),
                            **item,
                        )
                    )
    result = pd.DataFrame(rows)
    out = new_output_directory(output)
    result.to_csv(out / "fold_burden.csv", index=False)
    summary = (
        result[result.policy.ne("baseline")]
        .groupby(["policy", "stage"])
        .agg(
            comparison_folds=("oracle_fp_reduction", "size"),
            mean_oracle_fp_reduction=("oracle_fp_reduction", "mean"),
            better_folds=("oracle_fp_reduction", lambda x: int(x.gt(0).sum())),
            worse_folds=("oracle_fp_reduction", lambda x: int(x.lt(0).sum())),
            tied_folds=("oracle_fp_reduction", lambda x: int(x.eq(0).sum())),
        )
    )
    summary.to_csv(out / "summary.csv")
    result.groupby(["candidate", "policy", "stage", "repeat"])[
        [
            "baseline_target_tp",
            "oracle_tp",
            "oracle_fp",
            "actual_tp",
            "actual_fp",
            "baseline_oracle_fp",
            "oracle_fp_reduction",
        ]
    ].sum().to_csv(out / "repeat_sums.csv")
    write_manifest(
        out,
        dict(
            stage="exact_catch_descriptive_ranking",
            models_fitted=0,
            operational_cutoffs_selected=0,
            oracle_points="outcome-derived; never deployable",
            october_outcomes_accessed=False,
            inputs=input_identity(
                {
                    **{
                        arm: ROOT / "results" / name / "manifest.json"
                        for arm, name in SOURCES.items()
                    },
                    "protocol": ROOT / "comparisons/matched_rank_burden_protocol.md",
                }
            ),
            implementation=input_identity({"script": Path(__file__)}),
        ),
    )
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
