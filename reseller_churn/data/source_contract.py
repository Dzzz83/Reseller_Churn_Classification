"""Hash-bound completeness declarations and source provenance for operations."""

import json
from pathlib import Path

from reseller_churn.artifacts import input_identity
from reseller_churn.data.preparation import date, load_sources

SOURCE_NAMES = {"orders", "stores", "headers", "lines"}
DATED_SOURCES = ("orders", "headers", "lines")


def load_verified_sources(
    paths, manifest_path, coverage_from, coverage_through, cutoff
):
    """Verify declarations against the supplied files, then reconcile their rows.

    Completeness must be attested by the source owner. A last transaction date
    cannot establish coverage, even when every supplied row reconciles.
    """
    if set(paths) != SOURCE_NAMES:
        raise ValueError("Provide explicit orders, stores, headers and lines paths.")
    manifest_path = Path(manifest_path)
    before = input_identity({**paths, "source_manifest": manifest_path})
    contract = json.loads(manifest_path.read_text())
    if contract.get("schema") != "source-coverage-v1" or not contract.get(
        "attested_by"
    ):
        raise ValueError(
            "Source coverage requires a schema and completeness attestation."
        )
    if set(contract.get("sources", {})) != SOURCE_NAMES:
        raise ValueError("Source manifest must identify all four inputs.")
    for name in SOURCE_NAMES:
        if before[name]["sha256"] != contract["sources"][name].get("sha256"):
            raise ValueError(f"Source checksum mismatch: {name}")
    start, end = date(coverage_from), date(coverage_through)
    effective_from = max(
        date(contract["sources"][n]["coverage_from"]) for n in DATED_SOURCES
    )
    effective_through = min(
        date(contract["sources"][n]["coverage_through"]) for n in DATED_SOURCES
    )
    if start < effective_from or end > effective_through:
        raise ValueError("Declared coverage exceeds effective per-source coverage.")
    if not effective_from <= start < date(cutoff) <= end:
        raise ValueError(
            "Insufficient effective source coverage for the requested cutoff."
        )
    source = load_sources(
        paths,
        start,
        end,
        cutoff,
        approved_excluded_order_ids=contract.get("approved_excluded_order_ids", []),
    )
    if before != input_identity({**paths, "source_manifest": manifest_path}):
        raise ValueError("Source files changed during loading.")
    provenance = {
        "inputs": before,
        "coverage_declaration": contract,
        "coverage_from": str(start.date()),
        "coverage_through": str(end.date()),
        "effective_coverage_from": str(effective_from.date()),
        "effective_coverage_through": str(effective_through.date()),
        "loaded_through": str(source.loaded_through.date()),
        "excluded_order_ids": list(source.excluded_order_ids),
    }
    return source, provenance
