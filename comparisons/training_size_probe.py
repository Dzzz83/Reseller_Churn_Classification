"""Fixed training-reseller sensitivity probe; no new cutoff or candidate claim."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from comparisons.baseline_diagnosis.inputs import read_csv, validate_partition
from comparisons.matched_rank_burden import burden
from comparisons.product_mix_comparison import BASE_FEATURES, fit_product_mix
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["StoreID", "snapshot"]


def select_training_resellers(train, fraction):
    stores = np.sort(train.StoreID.unique())
    selected = np.random.default_rng(2026).permutation(stores)[
        : int(len(stores) * fraction)
    ]
    return train[train.StoreID.isin(selected)], selected.tolist()


def run(output):
    source = ROOT / "results/product-mix-comparison-20261017"
    profiles_dir = ROOT / "results/product-mix-diagnosis-20261016"
    verify_manifest(source)
    verify_manifest(profiles_dir)
    data = read_csv(source / "development_input.csv")
    profiles = read_csv(profiles_dir / "case_sku_shares.csv")
    protocol = json.loads((source / "protocol.json").read_text())
    out = new_output_directory(output)
    parts, metrics = [], []
    for split in protocol["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        verify_manifest(source / relative)
        saved = read_csv(source / relative / "predictions.csv")
        full = saved[saved.strategy.eq("cadence_product_mix")].copy()
        baseline = saved[saved.strategy.eq("baseline")]
        target = int((baseline.churn.eq(1) & baseline.score.ge(baseline.cutoff)).sum())
        cutoff = float(full.cutoff.iloc[0])
        folder = new_output_directory(out / relative)
        identities = []
        for fraction in [0.5, 0.75, 1.0]:
            if fraction < 1:
                subset, stores = select_training_resellers(train, fraction)
                scores, fitted = fit_product_mix(
                    subset, valid[KEYS + BASE_FEATURES], profiles
                )
                identities.extend(
                    dict(**row, fraction=fraction, selected_resellers=stores)
                    for row in fitted
                )
                rows = valid[KEYS + ["churn", "n_orders_3m"]].copy()
                rows["score"], rows["cutoff"] = scores, cutoff
            else:
                subset = train
                rows = full[KEYS + ["churn", "n_orders_3m", "score", "cutoff"]].copy()
            rows["repeat"], rows["outer_fold"], rows["fraction"] = (
                split["repeat"],
                split["outer_fold"],
                fraction,
            )
            parts.append(rows)
            a, y = rows.score.ge(cutoff), rows.churn.eq(1)
            metrics.append(
                dict(
                    repeat=split["repeat"],
                    outer_fold=split["outer_fold"],
                    fraction=fraction,
                    training_resellers=subset.StoreID.nunique(),
                    training_rows=len(subset),
                    average_precision=average_precision_score(y, rows.score),
                    tp=int((a & y).sum()),
                    fp=int((a & ~y).sum()),
                    reused_full_training_cutoff=cutoff,
                    median_churn_score=rows.loc[y, "score"].median(),
                    median_nonchurn_score=rows.loc[~y, "score"].median(),
                    **burden(y.to_numpy().astype(int), rows.score.to_numpy(), target),
                )
            )
        write_json(folder / "model_identities.json", identities)
        write_json(folder / "split.json", split)
        write_manifest(folder, dict(new_fits=10))
        print(
            f"Completed repeat {split['repeat']} fold {split['outer_fold']}", flush=True
        )
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    table = pd.DataFrame(metrics)
    table.to_csv(out / "metrics.csv", index=False)
    summary = table.groupby(["repeat", "fraction"])[
        [
            "average_precision",
            "oracle_fp",
            "tp",
            "fp",
            "median_churn_score",
            "median_nonchurn_score",
        ]
    ].mean()
    summary.to_csv(out / "repeat_summary.csv")
    checks = []
    for repeat, frame in table.groupby("repeat"):
        mean = (
            frame.groupby("fraction")[["average_precision", "oracle_fp"]]
            .mean()
            .sort_index()
        )
        checks.append(
            dict(
                repeat=int(repeat),
                ap_monotone=bool(mean.average_precision.diff().dropna().ge(0).all()),
                oracle_fp_monotone=bool(mean.oracle_fp.diff().dropna().le(0).all()),
            )
        )
    write_json(
        out / "descriptive_consistency.json",
        dict(checks=checks, operational_acceptance_assessed=False),
    )
    write_manifest(
        out,
        dict(
            stage="training_size_sensitivity",
            fits=150,
            cutoffs_selected=0,
            october_outcomes_accessed=False,
            limitation="full-training cutoff reused across sizes; not nested operational candidate comparison",
            inputs=input_identity(
                dict(
                    source=source / "manifest.json",
                    profiles=profiles_dir / "manifest.json",
                    protocol=ROOT / "comparisons/training_size_probe_protocol.md",
                )
            ),
            implementation=input_identity(
                dict(
                    runner=Path(__file__),
                    fitting=ROOT / "comparisons/product_mix_comparison.py",
                )
            ),
        ),
    )
    print(summary.to_string())
    print(checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
