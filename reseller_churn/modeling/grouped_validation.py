"""Reproducible reseller groups and training-only cutoff selection."""

from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def grouped_splits(data: pd.DataFrame, count: int, seed: int):
    """Validate every row once while keeping all snapshots of each reseller together."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("Grouped fold count must be an integer of at least two.")
    if data.StoreID.nunique() < count:
        raise ValueError("Not enough resellers for the requested grouped folds.")
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    splits = list(splitter.split(data, data.churn, data.StoreID))
    coverage = np.zeros(len(data), dtype=int)
    for training, validation in splits:
        if not len(training) or not len(validation):
            raise ValueError("Grouped folds must have nonempty partitions.")
        if set(data.iloc[training].StoreID) & set(data.iloc[validation].StoreID):
            raise ValueError("Reseller overlap across training and validation")
        if data.iloc[training].churn.nunique() != 2:
            raise ValueError("Training fold lacks both classes")
        coverage[validation] += 1
    if not np.all(coverage == 1):
        raise ValueError("Validation folds must cover each row once")
    return splits


@dataclass(frozen=True)
class GroupedValidationPlan:
    """The repeated outer assessment and inner selection settings of one study."""

    outer_folds: int = 5
    inner_folds: int = 3
    repeat_seeds: tuple[int, ...] = (42, 78, 88)
    inner_seed: int = 78

    def __post_init__(self):
        for count in (self.outer_folds, self.inner_folds):
            if isinstance(count, bool) or not isinstance(count, int) or count < 2:
                raise ValueError(
                    "Grouped fold counts must be integers of at least two."
                )
        if not self.repeat_seeds or len(set(self.repeat_seeds)) != len(
            self.repeat_seeds
        ):
            raise ValueError("Repeat seeds must be nonempty and unique.")
        for seed in (*self.repeat_seeds, self.inner_seed):
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("Grouped seeds must be nonnegative integers.")

    def outer_splits(self, data):
        for repeat in self.repeat_seeds:
            for fold, (training, validation) in enumerate(
                grouped_splits(data, self.outer_folds, repeat)
            ):
                yield repeat, fold, training, validation

    def inner_splits(self, data):
        return grouped_splits(data, self.inner_folds, self.inner_seed)

    def to_dict(self):
        return {
            **asdict(self),
            "repeat_seeds": list(self.repeat_seeds),
            "design": "repeated_nested_stratified_reseller_groups",
        }


def cutoff_scan(target, scores):
    """Exact observed-score candidates; decisions include scores equal to the cutoff."""
    y = np.asarray(target)
    score = np.asarray(scores, dtype=float)
    if (
        y.ndim != 1
        or score.ndim != 1
        or len(y) != len(score)
        or not len(y)
        or not np.isin(y, [0, 1]).all()
        or not (y == 1).any()
        or not np.isfinite(score).all()
        or ((score < 0) | (score > 1)).any()
    ):
        raise ValueError("Invalid training OOF scores or labels")
    positive = np.sort(score[y == 1])
    negative = np.sort(score[y == 0])
    cutoffs = np.unique(np.r_[0.0, score])
    tp = len(positive) - np.searchsorted(positive, cutoffs, side="left")
    fp = len(negative) - np.searchsorted(negative, cutoffs, side="left")
    alerts = tp + fp
    return pd.DataFrame(
        {
            "cutoff": cutoffs,
            "tp": tp,
            "fp": fp,
            "fn": len(positive) - tp,
            "tn": len(negative) - fp,
            "alerts": alerts,
            "recall": tp / len(positive),
            "precision": np.divide(tp, alerts, out=np.zeros(len(tp)), where=alerts > 0),
        }
    )


@dataclass(frozen=True)
class ThresholdPolicy:
    """An explicit development objective, separate from a frozen prediction cutoff."""

    objective: str = "recall_at_precision"
    minimum_precision: float = 0.4
    minimum_recall: float = 0.9

    def __post_init__(self):
        if self.objective not in {"recall_at_precision", "precision_at_recall"}:
            raise ValueError("Unknown cutoff objective.")
        for value in (self.minimum_precision, self.minimum_recall):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    "Cutoff policy limits must be finite fractions in [0, 1]."
                )

    def select(self, scan):
        if self.objective == "recall_at_precision":
            floor = Fraction(str(self.minimum_precision))
            feasible = [
                r
                for r in scan.to_dict("records")
                if r["alerts"] > 0 and Fraction(int(r["tp"]), int(r["alerts"])) >= floor
            ]

            def key(r):
                return (r["tp"], Fraction(int(r["tp"]), int(r["alerts"])), r["cutoff"])
        else:
            feasible = scan[scan.recall.gt(self.minimum_recall)].to_dict("records")

            def key(r):
                return (r["precision"], r["recall"], r["cutoff"])

        return max(feasible, key=key) if feasible else None

    def to_dict(self):
        return asdict(self)
