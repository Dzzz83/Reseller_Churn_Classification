"""Reconstruct saved October decisions and counts without fitting or tuning."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require
from comparisons.crossfit_percentile import FEATURES
from comparisons.verify_basket_breadth import independent_cutoff
from reseller_churn.artifacts import (
    file_sha256,
    input_identity,
    verify_manifest,
    write_json,
)

KEYS = study.KEYS
CANDIDATE = "positive_floor_veto"


def ordered(frame):
    return frame.sort_values(KEYS).reset_index(drop=True)


def verify(directory, with_outcomes=False):
    verify_manifest(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    for identity in protocol["inputs"].values():
        require(
            file_sha256(Path(identity["path"])) == identity["sha256"],
            "Frozen source changed",
        )
    data = read_csv(directory / "development_input.csv")
    inner = read_csv(directory / "development_oof.csv")
    baseline = ordered(inner[inner.candidate.eq("baseline")])
    component = ordered(inner[inner.candidate.eq("retained_logistic")])
    pd.testing.assert_frame_equal(
        baseline[KEYS + ["churn"]], ordered(data[KEYS + ["churn"]])
    )
    pd.testing.assert_frame_equal(
        baseline[KEYS + ["churn"]], component[KEYS + ["churn"]]
    )
    cut = independent_cutoff(baseline.churn.to_numpy(), baseline.score.to_numpy())
    caught = (baseline.score >= cut) & baseline.churn.eq(1)
    floor = float(min(component.loc[caught, "score"]))
    selection = json.loads((directory / "selection.json").read_text())
    require(
        selection
        == dict(
            reference_cutoff=cut,
            veto_floor=floor,
            protected_inner_cases=int(caught.sum()),
            fallback=cut == 0.0,
        ),
        "Rule differs",
    )
    records = json.loads((directory / "training_identities.json").read_text())["fits"]
    require(len(records) == 40, "Wrong classifier/map count")
    for record in records:
        fit = pd.DataFrame(record["fitting_cases"])
        fit.snapshot = pd.to_datetime(fit.snapshot)
        ref = pd.DataFrame(record["reference_cases"])
        ref.snapshot = pd.to_datetime(ref.snapshot)
        require(
            set(fit.StoreID).isdisjoint(ref.StoreID), "Classifier/reference leakage"
        )
        subset = fit.merge(data, on=KEYS, how="left", validate="one_to_one")
        import hashlib

        content = pd.util.hash_pandas_object(
            subset[KEYS + ["churn"] + FEATURES], index=False
        ).values.tobytes()
        require(
            hashlib.sha256(content).hexdigest() == record["training_sha256"],
            "Training identity differs",
        )
        labels = ref.merge(data[KEYS + ["churn"]], on=KEYS, validate="one_to_one").churn
        np.testing.assert_array_equal(labels, record["reference_labels"])
        if record["stage"] == "inner":
            fold = record["inner_fold"]
            held = baseline[baseline.inner_fold.eq(fold)]
            require(
                set(held.StoreID).isdisjoint(fit.StoreID)
                and set(held.StoreID).isdisjoint(ref.StoreID),
                "Inner held leakage",
            )
            expected = set(
                map(tuple, data[~data.StoreID.isin(held.StoreID)][KEYS].to_numpy())
            )
            require(
                set(map(tuple, pd.concat([fit, ref])[KEYS].to_numpy())) == expected,
                "Inner source population differs",
            )
        raw = np.array(record["raw_held_scores"])
        mapped = expit(
            record["slope"] * logit(np.clip(raw, 1e-12, 1 - 1e-12))
            + record["intercept"]
        )
        np.testing.assert_array_equal(mapped, record["mapped_scores"])
    # Independently reconstruct pooled inner component scores from all ten maps per fold.
    for fold in sorted(baseline.inner_fold.unique()):
        subset = [
            r for r in records if r["stage"] == "inner" and r["inner_fold"] == fold
        ]
        require(len(subset) == 10, "Missing inner pair")
        keys = pd.DataFrame(subset[0]["held_cases"])
        keys.snapshot = pd.to_datetime(keys.snapshot)
        keys["score"] = np.mean([r["mapped_scores"] for r in subset], axis=0)
        np.testing.assert_array_equal(
            ordered(keys).score, ordered(component[component.inner_fold.eq(fold)]).score
        )
    test = ordered(read_csv(directory / "unlabeled_features.csv"))
    bundle = joblib.load(directory / "frozen_models.joblib")
    require(bundle["selection"] == selection, "Bundle rule differs")
    full = [r for r in records if r["stage"] == "full"]
    require(
        len(full) == 10 and len(bundle["component"]) == 10,
        "Full retained-pair count differs",
    )
    mapped = []
    for pair, record in zip(bundle["component"], full):
        model = pair["model"]
        # In-memory joblib hashes change after serialization (synthetically
        # reproduced). The bundle SHA is verified by the preparation manifest;
        # exact component scores below verify its operational identity.
        require(list(model.feature_names_in_) == FEATURES, "Component schema differs")
        params = {**protocol["component_parameters"], "random_state": record["seed"]}
        require(
            all(model.get_params()[k] == v for k, v in params.items()),
            "Component parameters differ",
        )
        raw = model.predict_proba(test[FEATURES])[:, 1]
        np.testing.assert_array_equal(raw, record["raw_held_scores"])
        mapped.append(
            expit(
                pair["slope"] * logit(np.clip(raw, 1e-12, 1 - 1e-12))
                + pair["intercept"]
            )
        )
    scores = np.mean(mapped, axis=0)
    np.testing.assert_allclose(
        scores, read_csv(directory / "component_scores.csv").score, rtol=0, atol=1e-15
    )
    base = np.mean(
        [m.predict_proba(test[study.FEATURES])[:, 1] for m in bundle["baseline"]],
        axis=0,
    )
    veto = base.copy()
    if cut != 0:
        veto[scores < floor] = 0
    predictions = read_csv(directory / "unlabeled_predictions.csv")
    for name, values, threshold in [
        ("baseline", base, cut),
        (CANDIDATE, veto, cut),
        ("historical_0.06", test.score.to_numpy(), 0.06),
    ]:
        rows = ordered(predictions[predictions.strategy.eq(name)])
        pd.testing.assert_frame_equal(rows[KEYS], test[KEYS])
        np.testing.assert_array_equal(rows.score, values)
        require(rows.cutoff.eq(threshold).all(), "Changed cutoff")
        np.testing.assert_array_equal(rows.predicted_churn, values >= threshold)
    metric_rows = 0
    if with_outcomes:
        verify_manifest(directory / "completed_evaluation")
        cases = read_csv(directory / "completed_evaluation/evaluated_cases.csv")
        metrics = pd.read_csv(
            directory / "completed_evaluation/metrics.csv", float_precision="round_trip"
        )
        supplied = (
            Path(protocol["inputs"]["previous_manifest"]["path"]).parent.parent
            / "current-model-diagnosis/october_cases.csv"
        )
        labels = read_csv(supplied)[KEYS + ["churn"]]
        for name, rows in cases.groupby("strategy"):
            expected = predictions[predictions.strategy.eq(name)].merge(
                labels, on=KEYS, validate="one_to_one"
            )
            pd.testing.assert_frame_equal(
                ordered(rows), ordered(expected)[rows.columns]
            )
        for _, row in metrics.iterrows():
            part = cases[cases.strategy.eq(row.strategy)]
            if row.subgroup != "overall":
                part = part[
                    part.n_orders_3m.gt(0).eq(row.subgroup == "recent_purchase")
                ]
            y = part.churn.eq(1)
            alert = part.predicted_churn.eq(1)
            tp = int((y & alert).sum())
            fp = int((~y & alert).sum())
            fn = int((y & ~alert).sum())
            tn = int((~y & ~alert).sum())
            for key, value in dict(
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                recall=tp / (tp + fn),
                precision=tp / (tp + fp),
                alerts=tp + fp,
            ).items():
                require(row[key] == value, "Independent metric differs: " + key)
            require(
                row.average_precision == average_precision_score(y, part.score),
                "AP differs",
            )
            require(row.roc_auc == roc_auc_score(y, part.score), "AUC differs")
            metric_rows += 1
    result = dict(
        status="passed",
        new_fits=0,
        training_identities=40,
        full_component_models=10,
        unlabeled_decisions=len(predictions),
        metric_rows=metric_rows,
        with_outcomes=with_outcomes,
        verifier=input_identity({"script": Path(__file__)}),
    )
    destination = directory / (
        "evaluation_verification.json"
        if with_outcomes
        else "preparation_verification.json"
    )
    require(not destination.exists(), "Preserve audit")
    write_json(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--with-outcomes", action="store_true")
    args = parser.parse_args()
    verify(args.directory, args.with_outcomes)
