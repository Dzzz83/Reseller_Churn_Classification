
import pandas as pd
import numpy as np

DATA_PATH = "datasets/processed/reseller_order_age_features.csv"
THRESHOLD = 0.80

TRAINING_FOLDS = {
    "fold_1": ["2012-07-01"],
    "fold_2": ["2012-07-01", "2012-10-01"],
}

EXCLUDE = ["StoreID", "snapshot"]


def analyze_fold(df, name, dates):
    train = df[df["snapshot"].isin(pd.to_datetime(dates))]

    X = train.drop(columns=EXCLUDE)
    X = X.select_dtypes(include="number")

    missing = X.isna().mean() * 100
    corr = X.corr(method="spearman")

    pairs = []

    for i, feature_a in enumerate(corr.columns):
        for feature_b in corr.columns[i + 1:]:
            rho = corr.loc[feature_a, feature_b]

            if pd.notna(rho) and abs(rho) > THRESHOLD:
                pairs.append((feature_a, feature_b, rho))

    print(f"\n=== {name.upper()} ===")
    print(f"Training observations: {len(train)}")
    print(f"Features: {len(X.columns)}")

    print("\nMissing values above 10%:")
    for feature, percentage in missing.items():
        if percentage > 10:
            print(f"  {feature}: {percentage:.2f}%")

    print("\nHighly correlated pairs:")
    for a, b, rho in sorted(
        pairs, key=lambda item: abs(item[2]), reverse=True
    ):
        print(f"  {a} <-> {b}: {rho:.3f}")

    print(f"\nTotal pairs: {len(pairs)}")

    assert len(train) == sum(
        (df["snapshot"] == pd.Timestamp(d)).sum()
        for d in dates
    )
    assert not X.empty
    assert not np.isinf(X.to_numpy(dtype=float)).any()

    print("[PASS] Fold audit completed")

    return set(
        tuple(sorted((a, b))) for a, b, _ in pairs
    )


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot"],
    )

    print("=== Stage 1G.2: Feature Stability Audit ===")

    fold_pairs = {}

    for name, dates in TRAINING_FOLDS.items():
        fold_pairs[name] = analyze_fold(df, name, dates)

    shared = fold_pairs["fold_1"] & fold_pairs["fold_2"]

    print("\n=== Stable Correlated Pairs ===")
    for a, b in sorted(shared):
        print(f"{a} <-> {b}")

    print(f"\nShared pairs: {len(shared)}")
    print("[PASS] Feature stability audit completed")


if __name__ == "__main__":
    main()
