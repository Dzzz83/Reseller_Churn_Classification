"""Small synthetic prepare/train/predict/evaluate check; not model-quality evidence."""

import sys
import tempfile
from pathlib import Path

# Installed verification must resolve the package outside the source checkout.
if "--installed" not in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from reseller_churn.cli import main


def synthetic_sources(directory: Path) -> dict[str, Path]:
    store_dates = {}
    for store in range(1, 9):
        dates = ["2012-03-01", "2012-06-01", "2012-09-01", "2012-12-01", "2013-03-01"]
        if store > 2:
            dates += ["2013-06-01", "2013-09-01"]
        elif store == 2:
            dates += ["2013-06-01"]
        else:
            dates = dates[:2]
        store_dates[store] = dates
    return write_synthetic_sources(directory, store_dates)


def write_synthetic_sources(directory: Path, store_dates: dict) -> dict[str, Path]:
    orders, headers, lines = [], [], []
    order_id = 0
    for store, dates in store_dates.items():
        for index, ordered in enumerate(dates):
            order_id += 1
            discount = 0.1 if index % 2 else 0.2
            quantity = float(store)
            revenue = quantity * 10 * (1 - discount)
            orders.append(
                {
                    "StoreID": store,
                    "SalesOrderID": order_id,
                    "OrderDate": ordered,
                    "SubTotal": revenue,
                    "qty": quantity,
                    "n_lines": 1,
                    "rev_bikes": revenue,
                    "rev_components": 0.0,
                    "rev_clothing": 0.0,
                    "rev_accessories": 0.0,
                }
            )
            headers.append(
                {
                    "StoreID": store,
                    "SalesOrderID": order_id,
                    "OrderDate": ordered,
                    "SubTotal": revenue,
                }
            )
            lines.append(
                {
                    "SalesOrderID": order_id,
                    "SalesOrderDetailID": order_id,
                    "ProductID": 1,
                    "OrderQty": quantity,
                    "UnitPrice": 10.0,
                    "UnitPriceDiscount": discount,
                    "LineTotal": revenue,
                }
            )
    frames = {
        "orders": pd.DataFrame(orders),
        "headers": pd.DataFrame(headers),
        "lines": pd.DataFrame(lines),
        "stores": pd.DataFrame({"StoreID": sorted(store_dates)}),
    }
    paths = {}
    for name, frame in frames.items():
        paths[name] = directory / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    return paths


def source_manifest(paths, path, start="2011-01-01", end="2013-10-01"):
    from reseller_churn.artifacts import file_sha256, write_json

    write_json(
        path,
        {
            "schema": "source-coverage-v1",
            "attested_by": "synthetic fixture",
            "approved_excluded_order_ids": [],
            "sources": {
                name: {
                    "sha256": file_sha256(p),
                    **(
                        {"coverage_from": start, "coverage_through": end}
                        if name != "stores"
                        else {}
                    ),
                }
                for name, p in paths.items()
            },
        },
    )
    return path


def synthetic_development():
    """Balanced synthetic cells with the same identities in all four periods."""
    import numpy as np

    from reseller_churn.config.frozen_model import FEATURES, TRAINING_SNAPSHOTS

    rows = []
    for snapshot in TRAINING_SNAPSHOTS:
        for store in range(1, 33):
            values = [
                1,
                20 + store,
                10 + store,
                0.7,
                0.1,
                0.2,
                100 + store,
                0.1,
                np.nan if store % 5 == 0 else 0.02,
                0.5,
            ]
            rows.append(
                {
                    "StoreID": store,
                    "snapshot": pd.Timestamp(snapshot),
                    **dict(zip(FEATURES, values)),
                    "churn": store % 2,
                }
            )
    return pd.DataFrame(rows)


def smoke(directory: Path) -> None:
    from reseller_churn.artifacts import verify_manifest
    from reseller_churn.modeling.grouped_validation import (
        GroupedValidationPlan,
        ThresholdPolicy,
    )
    from reseller_churn.modeling.training import develop
    from reseller_churn.prediction import CSV_COLUMNS

    paths = synthetic_sources(directory)
    coverage = source_manifest(paths, directory / "source_manifest.json")
    # Small grouped plan exercises all production responsibilities. Precision zero
    # makes this execution fixture feasible; it makes no quality claim.
    develop(
        synthetic_development(),
        directory / "development",
        plan=GroupedValidationPlan(outer_folds=2, inner_folds=2, repeat_seeds=(42,)),
        policy=ThresholdPolicy(minimum_precision=0),
    )
    inputs = [
        value for name, path in paths.items() for value in [f"--{name}", str(path)]
    ]
    main(
        [
            "export",
            *inputs,
            "--source-manifest",
            str(coverage),
            "--coverage-from",
            "2011-01-01",
            "--coverage-through",
            "2013-10-01",
            "--snapshot",
            "2013-10-01",
            "--model",
            str(directory / "development/models.joblib"),
            "--model-manifest",
            str(directory / "development/model_manifest.json"),
            "--output",
            str(directory / "export"),
        ]
    )
    result = pd.read_csv(
        directory / "export/predictions.csv", float_precision="round_trip"
    )
    assert list(result) == CSV_COLUMNS
    assert result.StoreID.tolist() == list(range(2, 9))
    assert set(result["Predicted churn"]) <= {"Yes", "No"}
    manifest = verify_manifest(directory / "export")
    assert manifest["models_fitted"] == 0 and manifest["outcomes_accessed"] is False
    print("Synthetic grouped training and Power BI export smoke passed.")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(
        prefix="reseller-churn-production-smoke-"
    ) as temporary:
        smoke(Path(temporary))
