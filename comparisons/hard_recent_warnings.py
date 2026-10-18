"""Descriptive development-only warning signs among missed recent-purchase churners."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.history_comparison import (
    DESCRIPTORS,
    coordinate_distances,
    matched_differences,
    similarity_coordinates,
    summarize_differences,
)
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.development_checks import validate_development_cases
from comparisons.evaluate_frozen_october import verified_artifact
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = study.KEYS
FLAGS = {
    "revenue_decline": ("revenue_ratio_3m", "below"),
    "order_count_decline": ("orders_ratio_3m", "below"),
    "smaller_last_order": ("last_order_earlier_revenue_ratio", "below"),
    "overdue_vs_own_cadence": ("overdue_ratio", "above"),
    "lengthening_last_gap": ("last_gap_ratio", "above"),
}


def add_flags(cases):
    result = cases.copy()
    for name, (feature, direction) in FLAGS.items():
        result[name] = (
            result[feature].lt(1).astype(float)
            if direction == "below"
            else result[feature].gt(1).astype(float)
        )
        result.loc[result[feature].isna(), name] = np.nan
    result["warning_flags_observed"] = result[list(FLAGS)].notna().sum(axis=1)
    result["warning_flags_present"] = result[list(FLAGS)].eq(1).sum(axis=1)
    result["any_warning"] = result.warning_flags_present.gt(0).astype(float)
    result.loc[result.warning_flags_observed.eq(0), "any_warning"] = np.nan
    return result


def flag_tables(cases):
    recent = cases[cases.n_orders_3m.gt(0)]
    groups = {
        "missed_churn": recent[recent.churn.eq(1) & ~recent.alert],
        "caught_churn": recent[recent.churn.eq(1) & recent.alert],
        "continuing_all": recent[recent.churn.eq(0)],
        "continuing_unflagged": recent[recent.churn.eq(0) & ~recent.alert],
        "continuing_false_alert": recent[recent.churn.eq(0) & recent.alert],
    }
    rows = []
    for name, group in groups.items():
        for snapshot, part in [("overall", group), *list(group.groupby("snapshot"))]:
            for flag in [*FLAGS, "any_warning"]:
                values = part[flag]
                rows.append(
                    dict(
                        group=name,
                        snapshot=str(snapshot),
                        flag=flag,
                        rows=len(part),
                        resellers=part.StoreID.nunique(),
                        defined=int(values.notna().sum()),
                        positive=int(values.eq(1).sum()),
                        unknown=int(values.isna().sum()),
                        fraction=float(values.mean())
                        if values.notna().any()
                        else np.nan,
                    )
                )
    return pd.DataFrame(rows)


def matching(cases):
    records = []
    coverage = []
    scales = []
    recent = cases.n_orders_3m.gt(0)
    target_mask = recent & cases.churn.eq(1) & ~cases.alert
    controls = {
        "vs_continuing_all": recent & cases.churn.eq(0),
        "vs_continuing_unflagged": recent & cases.churn.eq(0) & ~cases.alert,
        "vs_caught_churn": recent & cases.churn.eq(1) & cases.alert,
    }
    for snapshot, frame in cases.groupby("snapshot"):
        # Both names have identical numeric meaning. Provide the alias for the
        # helper's historical or operational schema without changing source data.
        matching_input = frame.assign(
            discount_frequency_12m=frame.discounted_line_share_12m
        )
        coordinates, scale = similarity_coordinates(matching_input)
        scales.extend([dict(snapshot=snapshot, **r) for r in scale])
        targets = frame[target_mask.reindex(frame.index)]
        for comparison, mask in controls.items():
            available = frame[mask.reindex(frame.index)]
            for index, target in targets.iterrows():
                pool = available[available.StoreID.ne(target.StoreID)]
                distances = coordinate_distances(
                    coordinates.loc[index], coordinates.loc[pool.index]
                ).mean(axis=1)
                ranked = pd.DataFrame(
                    dict(distance=distances, control_StoreID=pool.StoreID)
                ).sort_values(["distance", "control_StoreID"])
                for k in [1, 3, 5]:
                    selected = ranked.head(k)
                    ties = (
                        int(ranked.distance.eq(selected.distance.iloc[-1]).sum())
                        if len(selected)
                        else 0
                    )
                    coverage.append(
                        dict(
                            comparison=comparison,
                            snapshot=snapshot,
                            StoreID=target.StoreID,
                            k=k,
                            available_controls=len(pool),
                            matched_controls=len(selected),
                            boundary_ties=ties,
                        )
                    )
                    for neighbor, row in enumerate(selected.itertuples(index=False), 1):
                        records.append(
                            dict(
                                comparison=comparison,
                                snapshot=snapshot,
                                StoreID=target.StoreID,
                                k=k,
                                control_StoreID=int(row.control_StoreID),
                                neighbor=neighbor,
                                distance=row.distance,
                            )
                        )
    return pd.DataFrame(records), pd.DataFrame(coverage), pd.DataFrame(scales)


def reconstruct_ratios(keys, orders):
    rows = []
    require(
        not orders.duplicated(KEYS + ["SalesOrderID"]).any(),
        "Duplicate trajectory case/order",
    )
    for key, group in orders.groupby(KEYS):
        snapshot = pd.Timestamp(key[1])
        group = group.sort_values(["OrderDate", "SalesOrderID"])
        require(
            group.OrderDate.ge(snapshot - pd.DateOffset(months=12)).all()
            and group.OrderDate.lt(snapshot).all(),
            "Future/out-of-window order",
        )
        recent = group[group.OrderDate.ge(snapshot - pd.DateOffset(months=3))]
        previous = group[
            group.OrderDate.ge(snapshot - pd.DateOffset(months=6))
            & group.OrderDate.lt(snapshot - pd.DateOffset(months=3))
        ]
        earlier = group.iloc[:-1]
        dates = group.OrderDate.drop_duplicates().sort_values()
        gaps = dates.diff().dt.days.dropna()

        def ratio(a, b):
            return float(a / b) if b > 0 else np.nan

        rows.append(
            dict(
                StoreID=key[0],
                snapshot=snapshot,
                revenue_ratio_3m=ratio(recent.SubTotal.sum(), previous.SubTotal.sum()),
                orders_ratio_3m=ratio(len(recent), len(previous)),
                last_order_earlier_revenue_ratio=ratio(
                    group.SubTotal.iloc[-1], earlier.SubTotal.mean()
                ),
                overdue_ratio=ratio((snapshot - dates.iloc[-1]).days, gaps.mean()),
                last_gap_ratio=ratio(gaps.iloc[-1], gaps.iloc[:-1].mean())
                if len(gaps) >= 2
                else np.nan,
            )
        )
    rebuilt = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(
        keys[KEYS].sort_values(KEYS).reset_index(drop=True),
        rebuilt[KEYS].sort_values(KEYS).reset_index(drop=True),
    )
    return rebuilt


def run(output):
    source = ROOT / "results/all-candidate-precision-october-20261018-04"
    history_source = ROOT / "results/persistent-error-investigation-20261016"
    verify_manifest(history_source)
    for name in [
        "development_input.csv",
        "development_oof.csv",
        "development_selections.csv",
    ]:
        verified_artifact(source / "manifest.json", source / name)
    data = read_csv(source / "development_input.csv")
    validate_development_cases(data)
    predictions = read_csv(source / "development_oof.csv")
    predictions = predictions[predictions.candidate.eq("period_class_weighted")]
    selected = pd.read_csv(
        source / "development_selections.csv", float_precision="round_trip"
    )
    cutoff = float(
        selected[
            selected.candidate.eq("period_class_weighted")
            & selected.minimum_precision_percent.eq(40)
        ].cutoff.iloc[0]
    )
    histories = read_csv(history_source / "case_histories.csv")
    trajectories = read_csv(history_source / "order_trajectories.csv")
    trajectories.OrderDate = pd.to_datetime(trajectories.OrderDate)
    comparison = histories.merge(
        data[KEYS + ["churn"] + study.FEATURES],
        on=KEYS,
        validate="one_to_one",
        suffixes=("_history", "_canonical"),
    )
    for feature in ["churn", *study.FEATURES]:
        np.testing.assert_allclose(
            comparison[feature + "_history"],
            comparison[feature + "_canonical"],
            rtol=1e-12,
            atol=1e-9,
            equal_nan=True,
        )
    cases = (
        data[KEYS + ["churn"] + study.FEATURES]
        .merge(histories[KEYS + DESCRIPTORS], on=KEYS, validate="one_to_one")
        .merge(predictions[KEYS + ["score"]], on=KEYS, validate="one_to_one")
    )
    require(
        len(cases) == 1375
        and cases.churn.sum() == 212
        and cases.StoreID.nunique() == 422,
        "Changed cases",
    )
    cases["alert"] = cases.score.ge(cutoff)
    rebuilt = reconstruct_ratios(cases[KEYS], trajectories)
    checked = cases.merge(
        rebuilt, on=KEYS, suffixes=("_saved", "_rebuilt"), validate="one_to_one"
    )
    for feature, _ in FLAGS.values():
        np.testing.assert_allclose(
            checked[feature + "_saved"],
            checked[feature + "_rebuilt"],
            rtol=1e-12,
            atol=1e-9,
            equal_nan=True,
        )
    cases = add_flags(cases)
    out = new_output_directory(output)
    shutil.copyfile(
        ROOT / "comparisons/hard_recent_warning_protocol.md", out / "protocol.md"
    )
    shutil.copyfile(Path(__file__), out / "as_run_hard_recent_warnings.py")
    write_json(
        out / "input_identity.json",
        input_identity(
            {
                "development_oof": source / "development_oof.csv",
                "development_cases": source / "development_input.csv",
                "development_selection": source / "development_selections.csv",
                "historical_features": history_source / "case_histories.csv",
                "historical_orders": history_source / "order_trajectories.csv",
                "runner": Path(__file__),
                "protocol": ROOT / "comparisons/hard_recent_warning_protocol.md",
            }
        ),
    )
    cases.to_csv(out / "development_cases.csv", index=False)
    rebuilt.to_csv(out / "independent_ratio_reconstruction.csv", index=False)
    raw = flag_tables(cases)
    raw.to_csv(out / "warning_rates.csv", index=False)
    pairs, coverage, scales = matching(cases)
    pairs.to_csv(out / "matches.csv", index=False)
    coverage.to_csv(out / "matching_coverage.csv", index=False)
    scales.to_csv(out / "matching_scales.csv", index=False)
    differences = matched_differences(
        cases, pairs, descriptors=DESCRIPTORS + list(FLAGS) + ["any_warning"]
    )
    differences.to_csv(out / "matched_differences.csv", index=False)
    summary = summarize_differences(differences)
    summary.to_csv(out / "matched_summary.csv", index=False)
    reuse = (
        pairs.groupby(["comparison", "k", "snapshot", "control_StoreID"])
        .size()
        .rename("times_reused")
        .reset_index()
    )
    reuse.to_csv(out / "control_reuse.csv", index=False)
    require(
        cases[cases.churn.eq(1) & cases.n_orders_3m.gt(0) & ~cases.alert].shape[0]
        == 67,
        "Unexpected target count",
    )
    write_json(
        out / "verification.json",
        dict(
            status="passed",
            approved_cases=1375,
            target_cases=67,
            independent_ratios=6875,
            target_resellers=cases.loc[
                cases.churn.eq(1) & cases.n_orders_3m.gt(0) & ~cases.alert, "StoreID"
            ].nunique(),
            matching_comparisons=3,
            neighborhoods=[1, 3, 5],
            coverage_rows=len(coverage),
            all_five_controls_available=bool(
                coverage.matched_controls.eq(coverage.k).all()
            ),
            models_fitted=0,
            october_inputs=False,
            cutoff=cutoff,
        ),
    )
    write_manifest(
        out,
        dict(
            stage="development_only_warning_audit",
            models_fitted=0,
            october_inputs=False,
        ),
    )
    print(
        raw[raw.snapshot.eq("overall")][
            ["group", "flag", "rows", "defined", "positive", "fraction"]
        ].to_string(index=False)
    )
    print("Matched main k3 flag contrasts:")
    print(
        summary[
            summary.scope.eq("overall")
            & summary.k.eq(3)
            & summary.feature.isin([*FLAGS, "any_warning"])
        ][
            [
                "comparison",
                "feature",
                "paired_observed",
                "mean_difference",
                "median_difference",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
