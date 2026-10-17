"""Exploratory historical descriptors and transparent baseline-space matching."""

import numpy as np
import pandas as pd

from reseller_churn.config.frozen_model import FEATURES
from reseller_churn.data.prediction_window import PredictionWindow

from .definitions import KEYS
from .inputs import require

LOG_FEATURES = {"n_orders_3m", "revenue_3m", "revenue_12m"}
CATEGORIES = ["rev_bikes", "rev_components", "rev_clothing", "rev_accessories"]
DESCRIPTORS = [
    "n_orders_12m",
    "observed_span_days",
    "mean_gap",
    "gap_cv",
    "overdue_ratio",
    "last_gap_ratio",
    "revenue_ratio_3m",
    "orders_ratio_3m",
    "last_order_earlier_revenue_ratio",
    "last_order_qty",
    "last_order_lines",
    "last_order_unit_revenue",
    "category_mix_change",
    "product_overlap",
    "unique_products_12m",
    "last_order_products",
    "same_day_last_orders",
]


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else np.nan


def history_descriptors(keys, orders, lines):
    """Use only accepted orders in each [t-12 months, t) window."""
    rows, trajectories = [], []
    for snapshot, cohort in keys.groupby("snapshot"):
        history = PredictionWindow.from_date(snapshot).select_history(orders)
        history = history[history.StoreID.isin(cohort.StoreID)]
        for store, purchases in history.groupby("StoreID"):
            purchases = purchases.sort_values(["OrderDate", "SalesOrderID"])
            last, earlier = purchases.iloc[-1], purchases.iloc[:-1]
            dates = purchases.OrderDate.drop_duplicates().sort_values()
            gaps = dates.diff().dt.days.dropna()
            recent = purchases[
                purchases.OrderDate.ge(snapshot - pd.DateOffset(months=3))
            ]
            previous = purchases[
                purchases.OrderDate.ge(snapshot - pd.DateOffset(months=6))
                & purchases.OrderDate.lt(snapshot - pd.DateOffset(months=3))
            ]
            detail = lines[lines.SalesOrderID.isin(purchases.SalesOrderID)]
            last_products = set(
                detail.loc[detail.SalesOrderID.eq(last.SalesOrderID), "ProductID"]
            )
            old_products = set(
                detail.loc[detail.SalesOrderID.isin(earlier.SalesOrderID), "ProductID"]
            )
            mean_gap = gaps.mean()
            change = np.nan
            if len(earlier):
                old_mix = earlier[CATEGORIES].sum() / earlier.SubTotal.sum()
                last_mix = last[CATEGORIES].astype(float) / last.SubTotal
                change = float((last_mix - old_mix).abs().sum() / 2)
            rows.append(
                {
                    "StoreID": store,
                    "snapshot": snapshot,
                    "n_orders_12m": len(purchases),
                    "observed_span_days": (dates.max() - dates.min()).days,
                    "mean_gap": mean_gap,
                    "gap_cv": ratio(gaps.std(), mean_gap),
                    "overdue_ratio": ratio((snapshot - dates.max()).days, mean_gap),
                    "last_gap_ratio": ratio(gaps.iloc[-1], gaps.iloc[:-1].mean())
                    if len(gaps) >= 2
                    else np.nan,
                    "revenue_ratio_3m": ratio(
                        recent.SubTotal.sum(), previous.SubTotal.sum()
                    ),
                    "orders_ratio_3m": ratio(len(recent), len(previous)),
                    "last_order_earlier_revenue_ratio": ratio(
                        last.SubTotal, earlier.SubTotal.mean()
                    ),
                    "last_order_qty": last.qty,
                    "last_order_lines": last.n_lines,
                    "last_order_unit_revenue": last.SubTotal / last.qty,
                    "category_mix_change": change,
                    "product_overlap": len(last_products & old_products)
                    / len(last_products | old_products)
                    if len(earlier)
                    else np.nan,
                    "unique_products_12m": detail.ProductID.nunique(),
                    "last_order_products": len(last_products),
                    "same_day_last_orders": int(
                        purchases.OrderDate.eq(last.OrderDate).sum()
                    ),
                }
            )
            for purchase in purchases.itertuples(index=False):
                products = sorted(
                    lines.loc[
                        lines.SalesOrderID.eq(purchase.SalesOrderID), "ProductID"
                    ].unique()
                )
                trajectories.append(
                    {
                        "StoreID": store,
                        "snapshot": snapshot,
                        "SalesOrderID": purchase.SalesOrderID,
                        "OrderDate": purchase.OrderDate,
                        "SubTotal": purchase.SubTotal,
                        "qty": purchase.qty,
                        "n_lines": purchase.n_lines,
                        **{c: getattr(purchase, c) for c in CATEGORIES},
                        "products": "|".join(str(p) for p in products),
                    }
                )
    result = pd.DataFrame(rows)
    require(len(result) == len(keys), "Missing eligible historical case")
    return result, pd.DataFrame(trajectories)


def similarity_coordinates(frame):
    coordinates, scales = frame[list(FEATURES)].copy(), []
    for feature in FEATURES:
        values = coordinates[feature]
        if feature in LOG_FEATURES:
            require(values.dropna().ge(0).all(), "Negative log-coordinate input")
            values = np.log1p(values)
        require(not np.isinf(values).any(), "Infinite matching coordinate")
        low, high = values.quantile([0.05, 0.95])
        span = high - low
        if not span > 0:
            span = values.max() - values.min()
        if not span > 0:
            span = 1.0
        coordinates[feature] = values / span
        scales.append(
            {"feature": feature, "log1p": feature in LOG_FEATURES, "scale": float(span)}
        )
    return coordinates, scales


def coordinate_distances(target, controls):
    """Equal feature weights; explicit missingness penalty, no imputation."""
    distances = (controls - target).abs().clip(upper=1)
    both_missing = controls.isna() & target.isna()
    one_missing = controls.isna() ^ target.isna()
    return distances.mask(both_missing, 0).mask(one_missing, 1)


def comparison_groups(cases):
    recent = cases.n_orders_3m.gt(0)
    return {
        "recent_false_alert_vs_caught": (
            cases.churn.eq(0) & cases.alerts.eq(3) & recent,
            cases.churn.eq(1) & cases.alerts.eq(3) & recent,
        ),
        "missed_vs_correct_non_churn": (
            cases.churn.eq(1) & cases.alerts.eq(0) & recent,
            cases.churn.eq(0) & cases.alerts.eq(0) & recent,
        ),
        "inactive_continuing_vs_churn": (
            cases.churn.eq(0) & ~recent,
            cases.churn.eq(1) & cases.alerts.eq(3) & ~recent,
        ),
    }


def match_cases(cases, k_values=(1, 3, 5)):
    """Match each target in baseline space; retain unmatched targets explicitly."""
    matches, scales, coverage = [], [], []
    groups = comparison_groups(cases)
    for snapshot, frame in cases.groupby("snapshot"):
        coordinates, snapshot_scales = similarity_coordinates(frame)
        scales.extend({"snapshot": snapshot, **row} for row in snapshot_scales)
        for name, (target_mask, control_mask) in groups.items():
            targets = frame[target_mask.reindex(frame.index)]
            controls = frame[control_mask.reindex(frame.index)]
            for idx, target in targets.iterrows():
                pool = controls[controls.StoreID.ne(target.StoreID)]
                contributions = coordinate_distances(
                    coordinates.loc[idx], coordinates.loc[pool.index]
                )
                ordered = pd.DataFrame(
                    {"distance": contributions.mean(axis=1), "control": pool.StoreID}
                )
                ordered = ordered.sort_values(["distance", "control"])
                for k in k_values:
                    selected = ordered.head(k)
                    boundary_ties = (
                        0
                        if selected.empty
                        else int(ordered.distance.eq(selected.distance.iloc[-1]).sum())
                    )
                    coverage.append(
                        {
                            "comparison": name,
                            "snapshot": snapshot,
                            "StoreID": target.StoreID,
                            "k": k,
                            "available_controls": len(pool),
                            "matched_controls": len(selected),
                            "boundary_ties": boundary_ties,
                        }
                    )
                    for rank, (control_idx, row) in enumerate(selected.iterrows(), 1):
                        matches.append(
                            {
                                "comparison": name,
                                "snapshot": snapshot,
                                "StoreID": target.StoreID,
                                "control_StoreID": int(row.control),
                                "k": k,
                                "neighbor": rank,
                                "distance": row.distance,
                                **{
                                    f"distance_{f}": contributions.loc[control_idx, f]
                                    for f in FEATURES
                                },
                            }
                        )
    return pd.DataFrame(matches), pd.DataFrame(scales), pd.DataFrame(coverage)


def matched_differences(cases, matches, descriptors=DESCRIPTORS):
    values = cases[[*KEYS, *descriptors]]
    pairs = matches.merge(values, on=KEYS, validate="many_to_one")
    controls = values.rename(
        columns={
            "StoreID": "control_StoreID",
            **{f: f"control_{f}" for f in descriptors},
        }
    )
    pairs = pairs.merge(
        controls, on=["control_StoreID", "snapshot"], validate="many_to_one"
    )
    rows = []
    for identity, group in pairs.groupby(["comparison", "k", *KEYS]):
        for feature in descriptors:
            target = group[feature].iloc[0]
            control = group[f"control_{feature}"].mean()
            rows.append(
                {
                    **dict(zip(["comparison", "k", *KEYS], identity)),
                    "feature": feature,
                    "target": target,
                    "control_mean": control,
                    "difference": target - control,
                    "control_observed": int(group[f"control_{feature}"].notna().sum()),
                    "matched_controls": len(group),
                }
            )
    return pd.DataFrame(rows)


def summarize_differences(differences):
    tables = []
    for scope, columns in [
        ("overall", ["comparison", "k", "feature"]),
        ("snapshot", ["comparison", "k", "feature", "snapshot"]),
    ]:
        for identity, group in differences.groupby(columns):
            defined = group.difference.dropna()
            tables.append(
                {
                    "scope": scope,
                    **dict(zip(columns, identity)),
                    "targets": len(group),
                    "paired_observed": len(defined),
                    "target_missing": int(group.target.isna().sum()),
                    "control_missing": int(group.control_mean.isna().sum()),
                    "partial_controls": int(
                        group.control_observed.lt(group.matched_controls).sum()
                    ),
                    "target_median": group.target.median(),
                    "control_median": group.control_mean.median(),
                    "median_difference": defined.median(),
                    "mean_difference": defined.mean(),
                    "fraction_positive": defined.gt(0).mean()
                    if len(defined)
                    else np.nan,
                    "fraction_negative": defined.lt(0).mean()
                    if len(defined)
                    else np.nan,
                }
            )
    return pd.DataFrame(tables)
