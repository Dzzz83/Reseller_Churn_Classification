"""Verify and restore the handoff archives without overwriting local changes."""

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def restore_bundle(
    manifest_path: Path, destination: Path, verify_only: bool = False
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination = destination.resolve()
    planned = []
    unchanged = 0
    with ExitStack() as stack:
        for name, archive in manifest["archives"].items():
            if Path(name).name != name or "\\" in name:
                raise ValueError(f"Unsafe archive filename: {name}")
            archive_path = manifest_path.parent / name
            if digest(archive_path.read_bytes()) != archive["sha256"]:
                raise ValueError(f"Archive checksum mismatch: {name}")
            zipped = stack.enter_context(ZipFile(archive_path))
            if len(zipped.namelist()) != len(archive["files"]) or set(
                zipped.namelist()
            ) != set(archive["files"]):
                raise ValueError(f"Archive contents differ from manifest: {name}")
            for member, expected in archive["files"].items():
                relative = PurePosixPath(member)
                target = destination.joinpath(*relative.parts)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in member
                    or ":" in member
                ):
                    raise ValueError(f"Unsafe member path: {member}")
                if not target.resolve().is_relative_to(destination):
                    raise ValueError(f"Member escapes destination: {member}")
                data = zipped.read(member)
                if len(data) != expected["bytes"] or digest(data) != expected["sha256"]:
                    raise ValueError(f"Member checksum mismatch: {member}")
                if not verify_only and target.exists():
                    if (
                        not target.is_file()
                        or digest(target.read_bytes()) != expected["sha256"]
                    ):
                        raise FileExistsError(
                            f"Local file differs; restore stopped before writing: {target}"
                        )
                    unchanged += 1
                else:
                    planned.append((zipped, member, target))
        # Every archive and existing destination was checked before any write.
        if not verify_only:
            for zipped, member, target in planned:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(zipped.read(member))
    return {
        "verified_files": len(planned) + unchanged,
        "restored_files": 0 if verify_only else len(planned),
        "already_identical": unchanged,
        "verify_only": verify_only,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(__file__).resolve().parent / "artifacts/manifest.json"
    print(
        json.dumps(
            restore_bundle(manifest_path, args.destination, args.verify_only), indent=2
        )
    )


if __name__ == "__main__":
    main()
