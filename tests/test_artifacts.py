"""Source identity checks protect models after package restructuring."""

import json
import shutil
from pathlib import Path

import pytest

from reseller_churn import artifacts
from reseller_churn.data.preparation import build_development, load_sources
from reseller_churn.model import load_model, train


def test_source_identity_tracks_nested_files_with_unique_relative_paths(
    tmp_path, monkeypatch
):
    package = tmp_path / "package"
    for name in [
        "artifacts.py",
        "__init__.py",
        "config/__init__.py",
        "config/frozen_model.py",
    ]:
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Example source."""\n', encoding="utf-8")
    monkeypatch.setattr(artifacts, "__file__", str(package / "artifacts.py"))

    before = artifacts.source_identity()
    assert before["package_hash_format"] == artifacts.PACKAGE_HASH_FORMAT
    assert set(before["package_sha256"]) == {
        "artifacts.py",
        "__init__.py",
        "config/__init__.py",
        "config/frozen_model.py",
    }
    (package / "config/frozen_model.py").write_text(
        "CHANGED = True\n", encoding="utf-8"
    )
    after = artifacts.source_identity()
    assert before["package_sha256"] != after["package_sha256"]
    assert (
        before["package_sha256"]["__init__.py"]
        == after["package_sha256"]["__init__.py"]
    )


def test_model_loading_rejects_nested_source_changes_and_legacy_identity(
    source_paths, tmp_path, monkeypatch
):
    package = tmp_path / "package"
    shutil.copytree(
        Path(artifacts.__file__).parent,
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    monkeypatch.setattr(artifacts, "__file__", str(package / "artifacts.py"))
    source = load_sources(source_paths, "2011-01-01", "2013-10-01", "2013-10-01")
    development = tmp_path / "development.csv"
    build_development(source).to_csv(development, index=False)
    model_directory = tmp_path / "model"
    train(development, model_directory, "2013-10-01")
    load_model(model_directory)

    nested = package / "config/frozen_model.py"
    original = nested.read_bytes()
    nested.write_bytes(original + b"\n# Source changed after fitting.\n")
    with pytest.raises(ValueError, match="package code has changed"):
        load_model(model_directory)
    nested.write_bytes(original)

    manifest_path = model_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["source"]["package_hash_format"]
    artifacts.write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Legacy model source identity"):
        load_model(model_directory)
