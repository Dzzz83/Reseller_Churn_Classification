"""Handoff extraction must preserve files and reject tampering or unsafe paths."""

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "handoff/2026-10-16/restore_artifacts.py"
SPEC = importlib.util.spec_from_file_location("handoff_restore", SCRIPT)
RESTORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESTORE)


def bundle(tmp_path, files):
    archive = tmp_path / "example.zip"
    with ZipFile(archive, "w") as zipped:
        for name, value in files.items():
            zipped.writestr(name, value)
    manifest = {
        "archives": {
            archive.name: {
                "sha256": RESTORE.digest(archive.read_bytes()),
                "files": {
                    name: {"sha256": RESTORE.digest(value), "bytes": len(value)}
                    for name, value in files.items()
                },
            }
        }
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_restore_verification_and_idempotence(tmp_path):
    manifest = bundle(tmp_path, {"results/a.txt": b"saved result"})
    destination = tmp_path / "checkout"
    assert (
        RESTORE.restore_bundle(manifest, destination, verify_only=True)[
            "restored_files"
        ]
        == 0
    )
    assert not destination.exists()
    assert RESTORE.restore_bundle(manifest, destination)["restored_files"] == 1
    assert (destination / "results/a.txt").read_bytes() == b"saved result"
    assert RESTORE.restore_bundle(manifest, destination)["already_identical"] == 1


def test_restore_refuses_changed_local_files_before_any_write(tmp_path):
    manifest = bundle(
        tmp_path, {"results/new.txt": b"new", "results/existing.txt": b"original"}
    )
    destination = tmp_path / "checkout"
    (destination / "results").mkdir(parents=True)
    existing = destination / "results/existing.txt"
    existing.write_bytes(b"local changes")
    with pytest.raises(FileExistsError, match="before writing"):
        RESTORE.restore_bundle(manifest, destination)
    assert existing.read_bytes() == b"local changes"
    assert not (destination / "results/new.txt").exists()


def test_restore_rejects_archive_tampering(tmp_path):
    manifest = bundle(tmp_path, {"results/a.txt": b"original"})
    with (tmp_path / "example.zip").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Archive checksum"):
        RESTORE.restore_bundle(manifest, tmp_path / "checkout")


def test_restore_rejects_path_traversal(tmp_path):
    manifest = bundle(tmp_path, {"../escape.txt": b"unsafe"})
    with pytest.raises(ValueError, match="Unsafe member"):
        RESTORE.restore_bundle(manifest, tmp_path / "checkout")
    assert not (tmp_path / "escape.txt").exists()
