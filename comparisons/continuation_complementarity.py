"""No-fit error/rank overlap for two fully reported failed development procedures."""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from comparisons.baseline_diagnosis.inputs import read_csv
from comparisons.breadth_reporting import markdown_table
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)


def run(output):
    root = Path(__file__).resolve().parents[1] / "results"
    paths = {
        "calibration": root / "retained-calibration-20261018",
        "experts": root / "activity-experts-20261018-02",
    }
    parts = {}
    for name, path in paths.items():
        verify_manifest(path)
        frame = read_csv(path / "validation_predictions.csv")
        parts[name] = frame[frame.strategy.ne("baseline")]
    keys = ["StoreID", "snapshot", "repeat", "outer_fold", "churn", "n_orders_3m"]
    paired = parts["calibration"].merge(
        parts["experts"],
        on=keys,
        suffixes=("_calibration", "_experts"),
        validate="one_to_one",
    )
    rows = []
    for (repeat, snapshot), part in paired.groupby(["repeat", "snapshot"]):
        a = part.score_calibration.ge(part.cutoff_calibration)
        b = part.score_experts.ge(part.cutoff_experts)
        y = part.churn.eq(1)
        rows.append(
            dict(
                repeat=int(repeat),
                snapshot=str(snapshot.date()),
                calibration_only_catches=int((a & ~b & y).sum()),
                experts_only_catches=int((~a & b & y).sum()),
                both_miss=int((~a & ~b & y).sum()),
                calibration_only_fp=int((a & ~b & ~y).sum()),
                experts_only_fp=int((~a & b & ~y).sum()),
            )
        )
    overlap = pd.DataFrame(rows)
    ranks = []
    for (repeat, fold), part in paired.groupby(["repeat", "outer_fold"]):
        ranks.append(
            dict(
                repeat=int(repeat),
                outer_fold=int(fold),
                spearman_midrank=float(
                    spearmanr(part.score_calibration, part.score_experts).statistic
                ),
            )
        )
    ranks = pd.DataFrame(ranks)
    out = new_output_directory(output)
    overlap.to_csv(out / "overlap.csv", index=False)
    ranks.to_csv(out / "rank_correlations.csv", index=False)
    pooled = overlap.groupby("repeat").sum(numeric_only=True).reset_index()
    write_json(
        out / "diagnosis.json",
        dict(
            mean_within_fold_rho=float(ranks.spearman_midrank.mean()),
            note="Error overlap is descriptive,not a fitted ensemble or an acceptance decision",
        ),
    )
    (out / "report.md").write_text(
        "# Failed-procedure complementarity diagnosis\n\n"
        + markdown_table(pooled)
        + "\n\n"
        + markdown_table(overlap)
        + "\n\nAll15within-fold midrank correlations retained. No combined score,weight,cutoff or winner selected. "
        "Union catches are not an executable accepted method. Repeated observations remain dependent.\n"
    )
    write_manifest(
        out,
        dict(
            inputs=input_identity(
                {
                    **{k: v / "manifest.json" for k, v in paths.items()},
                    "script": Path(__file__),
                }
            ),
            models_fitted=0,
        ),
    )
    print(pooled.to_string(index=False))
    print("Mean within-fold rho:", ranks.spearman_midrank.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
