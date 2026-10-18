"""Select the veto's own development cutoff, then evaluate unchanged October scores."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.positive_floor_october_evaluation import join_supplied_outcomes
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "positive_floor_veto"
SOURCE = ROOT / "results/positive-floor-october-evaluation-20261018"


def run(output):
    verify_manifest(SOURCE)
    out = new_output_directory(output)
    # Freeze the requested threshold objective before selection or outcome access.
    write_json(
        out / "protocol.json",
        dict(
            authorization="User: select a new threshold in development and evaluate on final test.",
            objective="Maximum pooled development OOF precision with recall strictly above 90%; ties recall, then higher cutoff.",
            baseline_recall_constraint=False,
            baseline_cutoff_lower_bound=False,
            candidate=CANDIDATE,
            model_and_veto_floor_unchanged=True,
            independent_test=False,
            tuning_data="1375 approved development OOF cases only",
            inputs=input_identity(
                {
                    "prediction_preparation": SOURCE / "manifest.json",
                    "script": Path(__file__),
                }
            ),
        ),
    )
    development = read_csv(SOURCE / "development_oof.csv")
    development = development[development.candidate.eq(CANDIDATE)].copy()
    require(
        len(development) == 1375
        and development.churn.sum() == 212
        and development.StoreID.nunique() == 422,
        "Development population differs",
    )
    selected = study.select_threshold(
        development.churn.to_numpy(), development.score.to_numpy()
    )
    records = []
    y = development.churn.to_numpy() == 1
    scores = development.score.to_numpy()
    # Independent exhaustive tied-score enumeration, with integer recall comparison.
    for cut in sorted(set([0.0, *scores])):
        alert = scores >= cut
        tp = int((alert & y).sum())
        fp = int((alert & ~y).sum())
        records.append(
            dict(
                cutoff=float(cut),
                tp=tp,
                fp=fp,
                fn=int(y.sum()) - tp,
                tn=int((~y).sum()) - fp,
                recall=tp / int(y.sum()),
                precision=tp / max(1, tp + fp),
                alerts=tp + fp,
                feasible=10 * tp > 9 * int(y.sum()),
            )
        )
    scan = pd.DataFrame(records)
    feasible = [r for r in records if r["feasible"]]
    best = max(feasible, key=lambda r: (r["precision"], r["recall"], r["cutoff"]))
    require(
        all(best[k] == selected[k] for k in ["cutoff", "precision", "recall"]),
        "Independent cutoff differs",
    )
    scan.to_csv(out / "development_cutoff_scan.csv", index=False)
    write_json(
        out / "selected_threshold.json",
        dict(
            **selected,
            independent_enumeration_passed=True,
            tied_boundaries=len(records),
            development_counts=best,
        ),
    )
    # Save decisions using only the selected development rule before outcome access.
    predictions = read_csv(SOURCE / "unlabeled_predictions.csv")
    predictions = predictions[predictions.strategy.eq(CANDIDATE)].copy()
    previous_cutoff = float(predictions.cutoff.iloc[0])
    old_alert = predictions.predicted_churn.to_numpy().copy()
    predictions["cutoff"] = selected["cutoff"]
    predictions["predicted_churn"] = predictions.score.ge(selected["cutoff"]).astype(
        int
    )
    predictions.to_csv(out / "unlabeled_predictions.csv", index=False)
    write_json(
        out / "prediction_checkpoint.json",
        dict(
            threshold=selected["cutoff"],
            previous_threshold=previous_cutoff,
            changed_test_decisions=int(
                np.sum(old_alert != predictions.predicted_churn.to_numpy())
            ),
            outcomes_read=False,
            models_fitted=0,
        ),
    )
    write_manifest(
        out,
        dict(stage="predictions_checkpointed", outcomes_read=False, models_fitted=0),
    )
    verify_manifest(out)
    outcomes = ROOT / "results/current-model-diagnosis/october_cases.csv"
    from comparisons.evaluate_frozen_october import verified_artifact

    verified_artifact(outcomes.parent / "manifest.json", outcomes)
    labels = pd.read_csv(outcomes, usecols=study.KEYS + ["churn"])
    cases = join_supplied_outcomes(predictions, labels)
    evaluation = new_output_directory(out / "evaluation")
    cases.to_csv(evaluation / "evaluated_cases.csv", index=False)
    rows = []
    for name, part in [
        ("overall", cases),
        ("recent_purchase", cases[cases.n_orders_3m.gt(0)]),
        ("no_recent_purchase", cases[cases.n_orders_3m.eq(0)]),
    ]:
        row = dict(
            subgroup=name, **study.decision_metrics(part.churn, part.score, part.cutoff)
        )
        positive = part.churn.eq(1)
        alert = part.predicted_churn.eq(1)
        require(
            [row[k] for k in ["tp", "fp", "fn", "tn"]]
            == [
                int((positive & alert).sum()),
                int((~positive & alert).sum()),
                int((positive & ~alert).sum()),
                int((~positive & ~alert).sum()),
            ],
            "Independent confusion counts differ",
        )
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(evaluation / "metrics.csv", index=False)
    report = f"""# Candidate-specific development threshold evaluation

The user requested selecting the candidate's own threshold on development data.
The frozen objective is maximum OOF precision with recall strictly above90%, ties
recall then higher cutoff. There is NO baseline-TP constraint or baseline-cutoff
lower bound. Models, veto floor and test scores remain unchanged; no refitting.

Independent exhaustive enumeration of{len(records)}intact tied boundaries agrees.
Selected cutoff:{selected["cutoff"]:.17g}. Development OOF recall:{selected["recall"]:.2%};
precision:{selected["precision"]:.2%}; TP{best["tp"]},FP{best["fp"]},FN{best["fn"]},TN{best["tn"]}.

This independently selected candidate cutoff is EXACTLY THE SAME as the previous
cutoff ({previous_cutoff:.17g}); there are no changed test decisions. A new selection
procedure need not produce a numerically different threshold. The veto already
retains191/212OOF churners; at least191are required to exceed90%recall.

Predictions were saved and verified before supplied October outcomes were joined.

```text
{metrics.to_string(index=False)}
```

October was previously inspected, so this is descriptive reuse, not independent
validation. No test outcomes selected the threshold. Earlier development full
acceptance remains failed. No deployment or model/feature/floor adjustment.
"""
    (evaluation / "report.md").write_text(report)
    write_manifest(
        evaluation,
        dict(
            stage="evaluated",
            models_fitted=0,
            cutoffs_selected_on_test=0,
            inputs=input_identity(
                {
                    "prediction_checkpoint": out / "manifest.json",
                    "supplied_outcomes": outcomes,
                }
            ),
        ),
    )
    print("Selected candidate-only development threshold:", selected)
    print(
        "Changed test decisions:",
        int(np.sum(old_alert != predictions.predicted_churn.to_numpy())),
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
