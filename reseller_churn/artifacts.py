"""Immutable run directories and reproducibility metadata."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity() -> dict:
    package_directory = Path(__file__).resolve().parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=package_directory,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "package_sha256": {
            source_file.name: file_sha256(source_file)
            for source_file in sorted(package_directory.glob("*.py"))
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


def input_identity(paths: dict[str, Path]) -> dict:
    return {
        name: {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
        for name, path in paths.items()
    }


def new_output_directory(path: Path) -> Path:
    """Reject reuse rather than overwrite a previous run's evidence."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value: dict) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_manifest(directory: Path, metadata: dict) -> None:
    write_json(
        directory / "manifest.json",
        {
            **metadata,
            "source": source_identity(),
            "outputs": {
                output_file.name: file_sha256(output_file)
                for output_file in sorted(directory.iterdir())
                if output_file.is_file()
            },
        },
    )


def verify_manifest(directory: Path) -> dict:
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    for filename, expected_checksum in manifest["outputs"].items():
        if Path(filename).name != filename:
            raise ValueError(f"Artifact checksum mismatch: {filename}")
        actual_checksum = file_sha256(Path(directory) / filename)
        if actual_checksum != expected_checksum:
            raise ValueError(f"Artifact checksum mismatch: {filename}")
    return manifest
