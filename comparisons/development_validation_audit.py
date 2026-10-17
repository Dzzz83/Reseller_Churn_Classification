"""No-fit audit of the latest candidate's development validation limitations.

Only the saved development study is read. No test inputs, predictions, outcomes,
models or test reports are accessed. No thresholds are selected or changed.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from comparisons import grouped_precision_recall as study
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/cadence-unit-revenue-20261017"
CANDIDATE = "cadence_unit_revenue"


def label_maturity(snapshot):
    return pd.Timestamp(snapshot) + pd.DateOffset(months=6)


def run(output):
    verify_manifest(SOURCE)
    data = study.read_csv(SOURCE / "development_input.csv")
    data.snapshot = pd.to_datetime(data.snapshot)
    assert (
        len(data) == 1375 and data.churn.sum() == 212 and data.StoreID.nunique() == 422
    )
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    out = new_output_directory(output)
    write_json(
        out / "protocol.json",
        dict(
            purpose="Development-only explanation audit; no fitting or selection",
            planned_checks=[
                "split group isolation and calendar maturity",
                "inner-to-outer operating transfer",
                "snapshot/activity metric reversals",
                "increment beyond cadence reference",
                "matured temporal development feasibility",
            ],
            test_access=False,
            model_fits=0,
            threshold_selection=False,
            maturity_rule="training snapshot + 6 calendar months <= validation snapshot",
            inputs=input_identity(
                {
                    "development_manifest": SOURCE / "manifest.json",
                    "script": Path(__file__),
                }
            ),
        ),
    )
    temporal, transfer = [], []
    for split in protocol["splits"]:
        train, valid = (
            data.iloc[split["train_rows"]],
            data.iloc[split["validation_rows"]],
        )
        assert not set(train.StoreID) & set(valid.StoreID)
        folder = SOURCE / f"repeat_{split['repeat']}/fold_{split['outer_fold']}"
        verify_manifest(folder)
        inner = study.read_csv(folder / "inner_scores.csv")
        outer = study.read_csv(folder / "predictions.csv")
        for frame in (inner, outer):
            frame.snapshot = pd.to_datetime(frame.snapshot)
        selections = study.read_csv(folder / "inner_selections.csv").set_index(
            "candidate"
        )
        for snapshot, rows in valid.groupby("snapshot"):
            maturity = train.snapshot.map(label_maturity)
            temporal.append(
                dict(
                    repeat=split["repeat"],
                    fold=split["outer_fold"],
                    snapshot=str(snapshot.date()),
                    validation_rows=len(rows),
                    train_rows=len(train),
                    train_rows_from_later_snapshots=int(
                        train.snapshot.gt(snapshot).sum()
                    ),
                    train_labels_unresolved_at_prediction=int(
                        maturity.gt(snapshot).sum()
                    ),
                    train_labels_matured_at_prediction=int(maturity.le(snapshot).sum()),
                    shared_resellers=0,
                )
            )
        for arm in ("baseline", CANDIDATE):
            cutoff = selections.loc[arm, "cutoff"]
            for stage, predictions in [
                ("inner_oof_selected", inner[inner.candidate.eq(arm)]),
                ("outer_heldout", outer[outer.strategy.eq(arm)]),
            ]:
                if "n_orders_3m" not in predictions:
                    predictions = predictions.merge(
                        data[study.KEYS + ["n_orders_3m"]],
                        on=study.KEYS,
                        validate="one_to_one",
                    )
                for activity, rows in [
                    ("overall", predictions),
                    ("recent_purchase", predictions[predictions.n_orders_3m.gt(0)]),
                ]:
                    values = study.decision_metrics(rows.churn, rows.score, cutoff)
                    transfer.append(
                        dict(
                            repeat=split["repeat"],
                            fold=split["outer_fold"],
                            arm=arm,
                            stage=stage,
                            activity=activity,
                            cutoff=cutoff,
                            **values,
                            false_positive_rate=values["fp"]
                            / max(1, values["fp"] + values["tn"]),
                        )
                    )
    pd.DataFrame(temporal).to_csv(out / "calendar_split_audit.csv", index=False)
    transfer = pd.DataFrame(transfer)
    transfer.to_csv(out / "inner_outer_transfer.csv", index=False)
    transfer.groupby(["arm", "stage", "activity"])[
        ["recall", "precision", "false_positive_rate"]
    ].agg(["mean", "min", "max"]).to_csv(out / "transfer_summary.csv")
    metrics = study.read_csv(SOURCE / "metrics.csv")
    snapshot = metrics[
        (metrics.scope == "repeat_pooled") & (metrics.slice == "snapshot_slice")
    ]
    snapshot.groupby(["policy", "group"])[
        ["rows", "churners", "tp", "fp", "recall", "precision", "false_positive_rate"]
    ].mean().to_csv(out / "snapshot_summary.csv")
    snapshot.to_csv(out / "all_snapshot_repeat_metrics.csv", index=False)
    changes = []
    for (repeat, group), frame in snapshot.groupby(["repeat", "group"]):
        frame = frame.set_index("policy")
        old, new = frame.loc["baseline"], frame.loc[CANDIDATE]
        changes.append(
            dict(
                repeat=repeat,
                snapshot=group,
                tp_change=int(new.tp - old.tp),
                fp_reduction=int(old.fp - new.fp),
                recall_change=new.recall - old.recall,
            )
        )
    pd.DataFrame(changes).to_csv(out / "snapshot_paired_changes.csv", index=False)
    # Feasibility only: these counts are not new fitted temporal evaluation results.
    feasibility = []
    for snapshot, valid in data.groupby("snapshot"):
        mature = data[data.snapshot.map(label_maturity).le(snapshot)]
        isolated = mature[~mature.StoreID.isin(valid.StoreID)]
        feasibility.append(
            dict(
                snapshot=str(snapshot.date()),
                validation_rows=len(valid),
                matured_training_rows=len(mature),
                matured_training_resellers=mature.StoreID.nunique(),
                matured_and_reseller_isolated_training_rows=len(isolated),
                matured_and_reseller_isolated_training_resellers=isolated.StoreID.nunique(),
                isolated_training_churners=int(isolated.churn.sum()),
            )
        )
    pd.DataFrame(feasibility).to_csv(out / "temporal_feasibility.csv", index=False)
    profiles = []
    for (snapshot, recent), part in data.groupby(
        [data.snapshot, data.n_orders_3m.gt(0)]
    ):
        profiles.append(
            dict(
                snapshot=str(snapshot.date()),
                recent_purchase=bool(recent),
                rows=len(part),
                churners=int(part.churn.sum()),
                prevalence=part.churn.mean(),
            )
        )
    pd.DataFrame(profiles).to_csv(out / "development_prevalence.csv", index=False)
    reference = study.read_csv(SOURCE / "cadence_reference_comparison.csv")
    reference.to_csv(out / "incremental_cadence_reference.csv", index=False)
    decision = json.loads((SOURCE / "decision.json").read_text())
    write_json(
        out / "verified_findings.json",
        dict(
            development_accepted=decision["accepted"],
            failed_checks=[
                name for name, passed in decision["checks"].items() if not passed
            ],
            independent_resellers=int(data.StoreID.nunique()),
            account_snapshot_rows=len(data),
            outer_folds=len(protocol["splits"]),
            model_fits=0,
            test_access=False,
            all_folds_reseller_isolated=True,
            all_60_fold_snapshot_cells_have_unresolved_training_labels=bool(
                pd.DataFrame(temporal).train_labels_unresolved_at_prediction.gt(0).all()
            ),
            identical_seed_scores=all(
                row["seed_score_range"] == 0
                for split in protocol["splits"]
                for row in json.loads(
                    (
                        SOURCE
                        / f"repeat_{split['repeat']}/fold_{split['outer_fold']}/model_identities.json"
                    ).read_text()
                )
            ),
        ),
    )
    write_manifest(
        out,
        dict(
            stage="development_validation_audit",
            inputs=input_identity({"source": SOURCE / "manifest.json"}),
        ),
    )
    print(pd.DataFrame(feasibility).to_string(index=False))
    print(
        transfer.groupby(["arm", "stage", "activity"])[
            ["recall", "precision", "false_positive_rate"]
        ]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
