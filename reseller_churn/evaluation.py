"""Recall, precision and false alerts for grouped development decisions only."""

import pandas as pd


def decision_metrics(rows):
    positive = rows.churn.eq(1)
    alert = rows.predicted_churn.eq(True)
    tp = int((positive & alert).sum())
    fp = int((~positive & alert).sum())
    fn = int((positive & ~alert).sum())
    tn = int((~positive & ~alert).sum())
    return {
        "rows": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "false_alerts": fp,
        "alerts": tp + fp,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
    }


def assessment_metrics(assessment, expected_rows):
    results = []
    for repeat, rows in assessment.groupby("repeat"):
        for snapshot, part in [("overall", rows), *rows.groupby("snapshot")]:
            results.append(
                {
                    "repeat": repeat,
                    "snapshot": str(snapshot),
                    "complete_repeat": len(rows) == expected_rows,
                    **decision_metrics(part),
                }
            )
    return pd.DataFrame(results)
