"""One frozen training-vocabulary SKU-share trial; no outcome-selected products."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons import grouped_precision_recall as study
from comparisons.baseline_diagnosis.inputs import read_csv, require, validate_partition
from comparisons.matched_evaluation import evaluate_predictions
from reseller_churn.artifacts import (
    input_identity,
    new_output_directory,
    verify_manifest,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = study.KEYS
CANDIDATE = "cadence_product_mix"
BASE_FEATURES = study.FEATURES + ["overdue_ratio", "gap_cv"]
TAIL = 0.05 / 46


def product_vocabulary(train_keys, profiles):
    """Unsupervised support counts distinct training resellers, not snapshots."""
    annual = profiles[profiles.scope.eq("annual")].merge(
        train_keys[KEYS].drop_duplicates(), on=KEYS, validate="many_to_one"
    )
    support = annual[annual.share.gt(0)].groupby("ProductID").StoreID.nunique()
    ranked = (
        support.rename("resellers")
        .reset_index()
        .sort_values(["resellers", "ProductID"], ascending=[False, True])
    )
    selected = ranked[ranked.resellers.ge(20)].head(20)
    return selected.ProductID.astype(int).tolist(), ranked


def add_product_shares(cases, profiles, vocabulary):
    """Row-local histories are fixed; validation never defines the vocabulary."""
    result = cases.copy()
    columns = []
    index = pd.MultiIndex.from_frame(cases[KEYS])
    for scope in ("annual", "latest_order"):
        matrix = (
            profiles[profiles.scope.eq(scope)]
            .pivot(index=KEYS, columns="ProductID", values="share")
            .fillna(0)
            .reindex(index)
        )
        require(matrix.notna().all().all(), "Missing case product profile")
        values = matrix.reindex(columns=vocabulary, fill_value=0)
        names = [f"{scope}_sku_{product}" for product in vocabulary]
        result[names] = values.to_numpy()
        other = f"{scope}_sku_other"
        result[other] = np.clip(1 - values.sum(axis=1).to_numpy(), 0, 1)
        columns.extend([*names, other])
        np.testing.assert_allclose(result[[*names, other]].sum(axis=1), 1, atol=1e-12)
    return result, columns


def fit_product_mix(train, validation_features, profiles, seeds=study.MODEL_SEEDS):
    vocabulary, support = product_vocabulary(train[KEYS], profiles)
    expanded, extra = add_product_shares(train, profiles, vocabulary)
    validation, _ = add_product_shares(validation_features, profiles, vocabulary)
    spec = dict(
        id=CANDIDATE,
        features=BASE_FEATURES + extra,
        parameters={
            **study.BASE_PARAMETERS,
            "max_leaf_nodes": 7,
            "l2_regularization": 1.0,
        },
    )
    scores, identities, spread, _ = study.fit_scores(
        expanded, validation[spec["features"]], spec, seeds=seeds
    )
    for identity in identities:
        identity.update(
            vocabulary=vocabulary,
            support=support.to_dict("records"),
            features=spec["features"],
            seed_score_range=spread,
        )
    return scores, identities


def bootstrap_training_rows(train, seed):
    """Draw reseller clusters; preserve every snapshot and bootstrap multiplicity."""
    stores = np.sort(train.StoreID.unique())
    draws = np.random.default_rng(seed).choice(stores, size=len(stores), replace=True)
    groups = {store: frame for store, frame in train.groupby("StoreID", sort=False)}
    sampled = pd.concat([groups[store] for store in draws], ignore_index=True)
    require(sampled.churn.nunique() == 2, "Bootstrap training lost a class")
    return sampled, draws.tolist()


def fit_bagged_product_mix(train, validation_features, profiles):
    scores, identities = [], []
    for seed in study.MODEL_SEEDS:
        sampled, draws = bootstrap_training_rows(train, seed)
        prediction, fitted = fit_product_mix(
            sampled, validation_features, profiles, seeds=(seed,)
        )
        scores.append(prediction)
        for identity in fitted:
            identity.update(
                candidate="cadence_product_mix_bagged",
                bootstrap_store_ids=draws,
                resampling_seed=seed,
            )
        identities.extend(fitted)
    spread = float(np.ptp(scores, axis=0).max())
    for identity in identities:
        identity["seed_score_range"] = spread
    return np.mean(scores, axis=0), identities


def run(output, bagged=False, latent=False):
    candidate_id = "cadence_product_mix_bagged" if bagged else CANDIDATE
    fitter = fit_bagged_product_mix if bagged else fit_product_mix
    tail = 0.05 / 48 if bagged else TAIL
    if latent:
        from comparisons.latent_product_mix import fit_latent_mix

        candidate_id, fitter, tail = "cadence_latent_mix", fit_latent_mix, 0.05 / 50
    source = ROOT / "results/cadence-stability-20261016"
    sku_source = ROOT / "results/product-mix-diagnosis-20261016"
    verify_manifest(source)
    verify_manifest(sku_source)
    previous = json.loads((source / "protocol.json").read_text())
    data = read_csv(source / "development_input.csv")
    profiles = read_csv(sku_source / "case_sku_shares.csv")
    history = read_csv(
        ROOT / "results/persistent-error-investigation-20261016/case_histories.csv"
    )
    subgroups = history[[*KEYS, "n_orders_12m", "observed_span_days"]].rename(
        columns={"observed_span_days": "observed_history_span_days"}
    )
    out = new_output_directory(output)
    protocol = ROOT / (
        "comparisons/product_bagging_protocol.md"
        if bagged
        else "comparisons/product_mix_protocol.md"
    )
    if latent:
        protocol = ROOT / "comparisons/latent_mix_protocol.md"
    shutil.copyfile(protocol, out / "protocol.md")
    data.to_csv(out / "development_input.csv", index=False)
    spec = dict(
        id=candidate_id,
        features=BASE_FEATURES
        + (
            [
                f"{scope}_mix_pc{i + 1}"
                for scope in ("annual", "latest_order")
                for i in range(5)
            ]
            + ["annual_unseen_share", "latest_order_unseen_share"]
            if latent
            else ["training_selected_SKU_shares"]
        ),
        parameters={
            **study.BASE_PARAMETERS,
            "max_leaf_nodes": 7,
            "l2_regularization": 1.0,
        },
    )
    write_json(
        out / "protocol.json",
        dict(
            description=(
                "Fixed training-only five-component product-mix projection"
                if latent
                else "Fixed reseller-bootstrap cadence/product ensemble"
                if bagged
                else "Training-vocabulary annual/latest SKU shares added to fixed cadence HGB"
            ),
            arms=[previous["arms"][0], spec],
            splits=previous["splits"],
            repeat_seeds=previous["repeat_seeds"],
            interval_tail_probability=tail,
            new_fits=300,
            october_outcomes_accessed=False,
        ),
    )
    parts = []
    for split in previous["splits"]:
        train, valid = validate_partition(data, split)
        relative = Path(f"repeat_{split['repeat']}/fold_{split['outer_fold']}")
        source_fold = source / relative
        verify_manifest(source_fold)
        folder = new_output_directory(out / relative)
        cached_inner = read_csv(source_fold / "inner_scores.csv")
        baseline_inner = cached_inner[cached_inner.candidate.eq("baseline")]
        assignment = baseline_inner.set_index(KEYS).inner_fold.reindex(
            pd.MultiIndex.from_frame(train[KEYS])
        )
        require(assignment.notna().all(), "Incomplete inner assignment")
        inner_parts, identities = [], []
        for fold in sorted(assignment.unique()):
            ti, vi = (
                np.flatnonzero(assignment.to_numpy() != fold),
                np.flatnonzero(assignment.to_numpy() == fold),
            )
            require(
                set(train.iloc[ti].StoreID).isdisjoint(train.iloc[vi].StoreID),
                "Inner reseller leakage",
            )
            scores, fitted = fitter(
                train.iloc[ti], train.iloc[vi][KEYS + BASE_FEATURES], profiles
            )
            rows = train.iloc[vi][KEYS + ["churn"]].copy()
            rows["score"], rows["candidate"], rows["inner_fold"] = (
                scores,
                candidate_id,
                int(fold),
            )
            inner_parts.append(rows)
            identities.extend(
                {**item, "stage": "inner", "inner_fold": int(fold)} for item in fitted
            )
        inner = pd.concat([baseline_inner, *inner_parts], ignore_index=True)
        selections = []
        for arm in ("baseline", candidate_id):
            rows = inner[inner.candidate.eq(arm)]
            selections.append(
                dict(
                    candidate=arm,
                    **study.select_threshold(
                        rows.churn.to_numpy(), rows.score.to_numpy()
                    ),
                )
            )
        cached = read_csv(source_fold / "predictions.csv")
        baseline = cached[cached.strategy.eq("baseline")]
        require(
            baseline.cutoff.eq(selections[0]["cutoff"]).all(), "Cached cutoff differs"
        )
        scores, fitted = fitter(train, valid[KEYS + BASE_FEATURES], profiles)
        identities.extend({**item, "stage": "outer"} for item in fitted)
        rows = valid[KEYS + ["churn", "n_orders_3m"]].copy()
        rows["score"], rows["cutoff"], rows["strategy"] = (
            scores,
            selections[1]["cutoff"],
            candidate_id,
        )
        rows["repeat"], rows["outer_fold"] = split["repeat"], split["outer_fold"]
        outer = pd.concat([baseline, rows], ignore_index=True)
        parts.append(outer)
        outer.to_csv(folder / "predictions.csv", index=False)
        inner.to_csv(folder / "inner_scores.csv", index=False)
        pd.DataFrame(selections).to_csv(folder / "inner_selections.csv", index=False)
        write_json(folder / "model_identities.json", identities)
        shutil.copyfile(source_fold / "split.json", folder / "split.json")
        write_manifest(folder, {"new_fits": len(identities)})
        print(
            f"Completed repeat {split['repeat']} fold {split['outer_fold']}", flush=True
        )
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    decision = evaluate_predictions(
        data, predictions, subgroups, out, candidate_id, tail
    )
    write_json(
        out / "baseline_fidelity.json",
        {"status": "unchanged cached HGB", "changed_baseline_decisions": 0},
    )
    write_manifest(
        out,
        dict(
            stage="historical_product_mix",
            fits=300,
            decision=decision,
            inputs=input_identity(
                {
                    "cached_source": source / "manifest.json",
                    "profiles": sku_source / "manifest.json",
                    "protocol": protocol,
                }
            ),
            implementation=input_identity(
                {
                    "runner": Path(__file__),
                    "fitting": Path(study.__file__),
                    **(
                        {"projection": ROOT / "comparisons/latent_product_mix.py"}
                        if latent
                        else {}
                    ),
                    "assessment": ROOT / "comparisons/matched_evaluation.py",
                }
            ),
        ),
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bagged", action="store_true")
    mode.add_argument("--latent", action="store_true")
    args = parser.parse_args()
    run(args.output, args.bagged, args.latent)
