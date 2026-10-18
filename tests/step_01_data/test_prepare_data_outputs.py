import hashlib
import json

import pandas as pd

import pipeline.legacy_prepare_data as prepare_module


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_data_writes_audit_without_modifying_references(
    tmp_path,
    monkeypatch,
):
    orders = pd.DataFrame(
        {
            "StoreID": [1],
            "SalesOrderID": [10],
            "OrderDate": pd.to_datetime(["2013-01-15"]),
            "SubTotal": [10.0],
            "qty": [1],
            "rev_bikes": [10.0],
            "rev_components": [0.0],
            "rev_clothing": [0.0],
            "rev_accessories": [0.0],
        }
    )
    stores = pd.DataFrame({"StoreID": [1]})
    engineered = pd.DataFrame(
        {
            "StoreID": [1],
            "snapshot": ["2013-01-01"],
            "recency_days": [15.0],
        }
    )
    labeled = engineered.assign(churn=0)
    final_features = pd.DataFrame(
        {
            "StoreID": [1],
            "snapshot": ["2013-10-01"],
            "recency_days": [8.0],
        }
    )

    results_root = tmp_path / "results"
    monkeypatch.setattr(
        prepare_module,
        "get_run_directory",
        lambda run_id: results_root / run_id,
    )
    monkeypatch.setattr(prepare_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(prepare_module, "ORDERS_PATH", tmp_path / "orders.csv")
    monkeypatch.setattr(prepare_module, "STORES_PATH", tmp_path / "stores.csv")
    prepare_module.ORDERS_PATH.write_text("orders fixture\n", encoding="utf-8")
    prepare_module.STORES_PATH.write_text("stores fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        prepare_module.DatasetLoader,
        "load_orders",
        staticmethod(lambda: orders.copy()),
    )
    monkeypatch.setattr(
        prepare_module.DatasetLoader,
        "load_stores",
        staticmethod(lambda: stores.copy()),
    )
    monkeypatch.setattr(
        prepare_module.FeaturePipeline,
        "build",
        lambda self, **kwargs: engineered.copy(),
    )
    monkeypatch.setattr(
        prepare_module,
        "build_ml_datasets",
        lambda **kwargs: (labeled.copy(), final_features.copy()),
    )

    reference_paths = {
        "features": tmp_path / "features.csv",
        "labeled": tmp_path / "labeled.csv",
        "final": tmp_path / "final.csv",
    }
    for name, frame in (
        ("features", engineered),
        ("labeled", labeled),
        ("final", final_features),
    ):
        frame.to_csv(reference_paths[name], index=False)
    monkeypatch.setattr(
        prepare_module,
        "ALL_ENGINEERED_FEATURES_PATH",
        reference_paths["features"],
    )
    monkeypatch.setattr(
        prepare_module,
        "LABELED_SNAPSHOTS_PATH",
        reference_paths["labeled"],
    )
    monkeypatch.setattr(
        prepare_module,
        "FINAL_TEST_FEATURES_PATH",
        reference_paths["final"],
    )
    reference_hashes = {
        name: _sha256(path) for name, path in reference_paths.items()
    }

    labeled_output = prepare_module.prepare_data("test-run")

    assert labeled_output.is_file()
    assert all(
        _sha256(path) == reference_hashes[name]
        for name, path in reference_paths.items()
    )
    audit_path = labeled_output.parent / "source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["order_rows"] == 1
    assert audit["store_rows"] == 1
    assert all(audit["checks"].values())
    assert audit["source_files"]["orders"]["sha256"] == _sha256(
        prepare_module.ORDERS_PATH
    )
