"""Small report helpers shared by development outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def error_analysis_from_predictions(
    validation_predictions: pd.DataFrame,
    candidate_id: str,
    threshold: float,
) -> pd.DataFrame:
    """Classify errors using retained validation probabilities, without refitting."""
    required_columns = {
        "candidate_id",
        "fold",
        "StoreID",
        "snapshot",
        "actual_churn",
        "churn_probability",
    }
    missing = required_columns - set(validation_predictions.columns)
    if missing:
        raise ValueError(
            f"Validation predictions are missing columns: {sorted(missing)}"
        )

    result = validation_predictions[
        validation_predictions["candidate_id"] == candidate_id
    ].copy()
    if result.empty:
        raise ValueError(f"No saved predictions for candidate {candidate_id}.")

    result["predicted_churn"] = (result["churn_probability"] >= threshold).astype(int)
    actual_churn = result["actual_churn"].astype(int)
    predicted_churn = result["predicted_churn"]
    result["result_type"] = "correct_non_churn"
    result.loc[(actual_churn == 1) & (predicted_churn == 1), "result_type"] = (
        "detected_churn"
    )
    result.loc[(actual_churn == 1) & (predicted_churn == 0), "result_type"] = (
        "missed_churn"
    )
    result.loc[(actual_churn == 0) & (predicted_churn == 1), "result_type"] = (
        "false_alert"
    )
    return result


def plot_precision_recall_tradeoffs(
    tradeoffs: pd.DataFrame,
    output_path: Path,
    minimum_precision: float | None = None,
) -> None:
    """Render a compact recall-versus-precision plot from saved metrics."""
    if tradeoffs.empty:
        return

    figure, axis = plt.subplots(figsize=(9, 6))
    for candidate_id, candidate_rows in tradeoffs.groupby("candidate_id"):
        ordered = candidate_rows.sort_values("threshold")
        axis.plot(
            ordered["mean_precision"],
            ordered["mean_recall"],
            marker=".",
            linewidth=1,
            markersize=2,
            alpha=0.65,
            label=candidate_id,
        )

    if minimum_precision is not None:
        axis.axvline(
            minimum_precision,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Precision floor {minimum_precision:.2f}",
        )
    axis.set(
        xlabel="Mean precision across development folds",
        ylabel="Mean recall across development folds",
        title="Development precision–recall tradeoffs",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(True, alpha=0.25)
    if tradeoffs["candidate_id"].nunique() <= 12:
        axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
