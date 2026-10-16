"""Small synthetic prepare/train/predict/evaluate check; not model-quality evidence."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from reseller_churn.cli import main


def synthetic_sources(directory: Path) -> dict[str, Path]:
    orders, headers, lines = [], [], []
    order_id = 0
    for store in range(1, 9):
        dates = ["2012-03-01", "2012-06-01", "2012-09-01", "2012-12-01", "2013-03-01"]
        if store > 2:
            dates += ["2013-06-01", "2013-09-01"]
        elif store == 2:
            dates += ["2013-06-01"]
        else:
            dates = dates[:2]
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
        "stores": pd.DataFrame({"StoreID": range(1, 9)}),
    }
    paths = {}
    for name, frame in frames.items():
        paths[name] = directory / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    return paths


def smoke(directory: Path) -> None:
    paths = synthetic_sources(directory)
    inputs = [
        value for name, path in paths.items() for value in [f"--{name}", str(path)]
    ]
    inputs += ["--coverage-from", "2011-01-01", "--coverage-through", "2013-10-01"]
    main(["prepare", *inputs, "--output", str(directory / "preparation")])
    main(
        [
            "train",
            "--development",
            str(directory / "preparation/development.csv"),
            "--prediction-date",
            "2013-10-01",
            "--output",
            str(directory / "model"),
        ]
    )
    main(
        [
            "features",
            *inputs,
            "--snapshot",
            "2013-10-01",
            "--output",
            str(directory / "features"),
        ]
    )
    main(
        [
            "predict",
            "--model",
            str(directory / "model"),
            "--features",
            str(directory / "features/features.csv"),
            "--output",
            str(directory / "prediction"),
        ]
    )
    predictions = pd.read_csv(directory / "prediction/ranked_predictions.csv")
    # Entirely synthetic outcomes: no historical final-test labels are read.
    outcomes = predictions[["StoreID", "snapshot"]].assign(
        actual_churn=predictions.StoreID.mod(2).astype(int),
    )
    outcomes.to_csv(directory / "outcomes.csv", index=False)
    main(
        [
            "evaluate",
            "--predictions",
            str(directory / "prediction/ranked_predictions.csv"),
            "--outcomes",
            str(directory / "outcomes.csv"),
            "--cutoffs",
            "0.06",
            "0.3",
            "0.55",
            "--outcomes-through",
            "2014-04-01",
            "--cohort-status",
            "inspected",
            "--output",
            str(directory / "evaluation"),
        ]
    )
    metrics = pd.read_csv(directory / "evaluation/cutoff_metrics.csv")
    assert metrics[metrics.subgroup.eq("overall")].shape[0] == 3
    assert metrics[metrics.subgroup.eq("overall")].rows.eq(len(predictions)).all()
    print("Synthetic end-to-end smoke passed.")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="reseller-churn-smoke-") as temporary:
        smoke(Path(temporary))
