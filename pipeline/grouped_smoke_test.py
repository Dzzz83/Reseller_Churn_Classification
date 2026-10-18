"""Synthetic source preparation, nested grouped fitting and outcome-free export."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pipeline.smoke_test import write_synthetic_sources
from reseller_churn.cli import main


def smoke(directory: Path) -> None:
    dates = pd.date_range("2011-05-01", "2013-09-01", freq="MS")
    stop_dates = pd.to_datetime(
        ["2012-06-01", "2012-09-01", "2012-12-01", "2013-03-01", "2013-09-01"]
    )
    store_dates = {
        cohort * 8 + member + 1: [str(date.date()) for date in dates if date <= stop]
        for cohort, stop in enumerate(stop_dates)
        for member in range(8)
    }
    paths = write_synthetic_sources(directory, store_dates)
    inputs = [
        value for name, path in paths.items() for value in [f"--{name}", str(path)]
    ]
    inputs += ["--coverage-from", "2011-01-01", "--coverage-through", "2013-10-01"]
    main(["prepare", *inputs, "--output", str(directory / "preparation")])
    main(
        [
            "develop",
            "--development",
            str(directory / "preparation/development.csv"),
            "--output",
            str(directory / "development"),
            "--models",
            str(directory / "models"),
            "--outer-folds",
            "2",
            "--inner-folds",
            "2",
            "--repeat-seeds",
            "42",
            "--seeds",
            "42",
            "--precision-floors",
            "0.1",
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
    selection = directory / "development/selection/policy_00.json"
    threshold = json.loads(selection.read_text())["threshold"]
    main(
        [
            "export",
            "--model",
            str(selection),
            "--features",
            str(directory / "features/features.csv"),
            "--threshold",
            repr(threshold),
            "--output",
            str(directory / "export"),
        ]
    )
    exported = pd.read_csv(
        directory / "export/predictions.csv", float_precision="round_trip"
    )
    assert len(exported) == 8
    assert exported.predicted_churn.eq(exported.churn_probability.ge(threshold)).all()
    manifest = json.loads((directory / "export/manifest.json").read_text())
    assert manifest["models_fitted"] == 0 and manifest["outcomes_accessed"] is False
    print("Synthetic grouped end-to-end smoke passed.")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(
        prefix="reseller-churn-grouped-smoke-"
    ) as temporary:
        smoke(Path(temporary))
