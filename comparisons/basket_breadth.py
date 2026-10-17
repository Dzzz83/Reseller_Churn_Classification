"""Matched nested grouped comparison of one historical basket-breadth feature."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import (
    read_csv,
    require,
    validate_partition,
    verify_baseline,
)
from comparisons.matched_evaluation import evaluate_predictions
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("baseline", "basket_breadth")


def configurations(feature="last_order_lines"):
    return [
        {
            "id": arm,
            "features": study.FEATURES + ([feature] if arm != "baseline" else []),
            "parameters": dict(study.BASE_PARAMETERS),
        }
        for arm in ARMS
    ]


def evaluate_fold(data, split, output, feature="last_order_lines"):
    """Outer labels are used only after train-only selection and fitting."""
    repeat, fold = split["repeat"], split["outer_fold"]
    train, valid = validate_partition(data, split)
    directory = new_output_directory(Path(output) / f"repeat_{repeat}" / f"fold_{fold}")
    choices, identities, inner = study.inner_selection(
        train, configurations(feature), 3, 78, study.MODEL_SEEDS
    )
    predictions = []
    for configuration in configurations(feature):
        arm = configuration["id"]
        cutoff = next(row["cutoff"] for row in choices if row["candidate"] == arm)
        scores, fitted, spread, _ = study.fit_scores(
            train, valid[configuration["features"]], configuration
        )
        identities.extend(
            [{**row, "stage": "outer", "seed_score_range": spread} for row in fitted]
        )
        saved = valid[[*study.KEYS, "churn", "n_orders_3m"]].copy()
        saved["score"], saved["cutoff"], saved["strategy"] = scores, cutoff, arm
        saved["repeat"], saved["outer_fold"] = repeat, fold
        predictions.append(saved)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(directory / "predictions.csv", index=False)
    inner.to_csv(directory / "inner_scores.csv", index=False)
    pd.DataFrame(choices).to_csv(directory / "inner_selections.csv", index=False)
    write_json(directory / "model_identities.json", identities)
    write_json(directory / "split.json", split)
    write_manifest(directory, {"fits": len(identities)})
    return predictions


def run(args):
    require(not args.output.exists(), "Output exists")
    features, old_predictions, audit = verify_baseline(args.source, args.orders)
    verify_manifest(args.histories)
    history = read_csv(args.histories / "case_histories.csv")
    data = features.merge(
        history[[*study.KEYS, args.feature]], on=study.KEYS, validate="one_to_one"
    )
    require(
        len(data) == 1375 and data[args.feature].gt(0).all(), "Invalid breadth merge"
    )
    splits = [
        json.loads(
            (args.source / f"repeat_{repeat}/fold_{fold}/split.json").read_text()
        )
        for repeat in study.REPEAT_SEEDS
        for fold in range(5)
    ]
    for split in splits:
        validate_partition(data, split)
    out = new_output_directory(args.output)
    protocol = (
        ROOT
        / "comparisons"
        / (
            "annual_breadth_protocol.md"
            if args.feature == "unique_products_12m"
            else "basket_breadth_protocol.md"
        )
    )
    tail = 0.0125 if args.feature == "unique_products_12m" else 0.025
    (out / "protocol.md").write_bytes(protocol.read_bytes())
    data.to_csv(out / "development_input.csv", index=False)
    write_json(
        out / "protocol.json",
        {
            "arms": configurations(args.feature),
            "feature": args.feature,
            "interval_tail_probability": tail,
            "seeds": list(study.MODEL_SEEDS),
            "repeat_seeds": list(study.REPEAT_SEEDS),
            "outer_folds": 5,
            "inner_folds": 3,
            "inner_seed": 78,
            "splits": splits,
            "bootstrap_draws": 5000,
            "bootstrap_seed": 2026,
            "historical_audit": audit,
            "october_outcomes_accessed": False,
        },
    )
    parts = []
    if args.workers == 1:
        for split in splits:
            parts.append(evaluate_fold(data, split, out, args.feature))
            print(
                f"Completed repeat {split['repeat']} fold {split['outer_fold']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {
                pool.submit(evaluate_fold, data, split, out, args.feature): split
                for split in splits
            }
            for future in as_completed(jobs):
                parts.append(future.result())
                split = jobs[future]
                print(
                    f"Completed repeat {split['repeat']} fold {split['outer_fold']}",
                    flush=True,
                )
    predictions = pd.concat(parts, ignore_index=True).sort_values(
        ["strategy", "repeat", "outer_fold", *study.KEYS]
    )
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    profiles = history[[*study.KEYS, "n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    decision = evaluate_predictions(
        features, predictions, profiles, out, "basket_breadth", tail
    )
    baseline = predictions[predictions.strategy.eq("baseline")]
    old = old_predictions[old_predictions.strategy.eq("original_baseline")]
    aligned = baseline.merge(
        old,
        on=[*study.KEYS, "repeat", "outer_fold"],
        suffixes=("_current", "_historical"),
        validate="one_to_one",
    )
    difference = aligned.score_current - aligned.score_historical
    fidelity = {
        "rows": len(aligned),
        "max_absolute_score_difference": float(difference.abs().max()),
        "changed_decisions": int(
            (
                aligned.score_current.ge(aligned.cutoff_current)
                != aligned.score_historical.ge(aligned.cutoff_historical)
            ).sum()
        ),
    }
    write_json(out / "baseline_fidelity.json", fidelity)
    write_manifest(
        out,
        {
            "stage": "basket_breadth",
            "fits": 600,
            "inputs": input_identity(
                {
                    "history_manifest": args.histories / "manifest.json",
                    "protocol": protocol,
                    "study_manifest": args.source / "manifest.json",
                }
            ),
            "implementation": input_identity(
                {
                    "runner": Path(__file__),
                    "shared_study": Path(study.__file__),
                    "paired_analysis": ROOT / "comparisons/matched_evaluation.py",
                }
            ),
            "decision": decision,
            "baseline_fidelity": fidelity,
        },
    )
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/repeated-grouped-precision"
    )
    parser.add_argument(
        "--histories",
        type=Path,
        default=ROOT / "results/persistent-error-investigation-20261016",
    )
    parser.add_argument(
        "--orders",
        type=Path,
        default=ROOT
        / "results/research-provenance-20261016/orders_clean_historical_bytes.csv",
    )
    parser.add_argument(
        "--feature",
        choices=["last_order_lines", "unique_products_12m"],
        default="last_order_lines",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
