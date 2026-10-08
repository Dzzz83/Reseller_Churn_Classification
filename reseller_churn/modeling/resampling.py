import pandas as pd


def balance_classes_by_random_oversampling(
    features: pd.DataFrame,
    target: pd.Series,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Duplicate minority churn rows until both classes have equal size.

    This function must only be called on a training fold.
    """
    features = features.reset_index(drop=True)
    target = target.reset_index(drop=True)

    combined = features.copy()
    combined["_target"] = target

    non_churn = combined[
        combined["_target"] == 0
    ]

    churn = combined[
        combined["_target"] == 1
    ]

    sampled_churn = churn.sample(
        n=len(non_churn),
        replace=True,
        random_state=random_seed,
    )

    balanced = pd.concat(
        [non_churn, sampled_churn],
        ignore_index=True,
    )

    balanced = balanced.sample(
        frac=1,
        random_state=random_seed,
    ).reset_index(drop=True)

    balanced_target = balanced.pop(
        "_target"
    ).astype(int)

    return balanced, balanced_target
