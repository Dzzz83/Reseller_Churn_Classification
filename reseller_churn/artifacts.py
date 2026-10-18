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

PACKAGE_HASH_FORMAT = "relative-python-paths-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity() -> dict:
    package = Path(__file__).resolve().parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=package,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "package_hash_format": PACKAGE_HASH_FORMAT,
        "package_sha256": {
            p.relative_to(package).as_posix(): file_sha256(p)
            for p in sorted(package.rglob("*.py"))
        },
        "versions": runtime_versions(),
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
                p.name: file_sha256(p)
                for p in sorted(directory.iterdir())
                if p.is_file()
            },
        },
    )


def verify_manifest(directory: Path) -> dict:
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    for name, expected in manifest["outputs"].items():
        if Path(name).name != name or file_sha256(Path(directory) / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return manifest


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
