"""Filesystem artifact helpers for SynAirG CT volume cases."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from synairg_core.volume_manifest import JsonValue

from volume_gen.errors import VolumeImportError


@dataclass(frozen=True)
class VolumeCase:
    """Paths emitted for a canonical SynAirG CT volume case."""

    case_id: str
    case_dir: Path
    ct_path: Path
    metadata_path: Path
    provenance_path: Path


def utc_timestamp() -> str:
    """Return the current UTC time in an ISO-8601 form suitable for JSON metadata."""
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def prepare_case_dir(output_root: Path, case_id: str, *, overwrite: bool) -> Path:
    """Create a per-case output directory, optionally replacing an existing case."""
    case_dir = output_root / case_id
    if case_dir.exists():
        if not overwrite and any(case_dir.iterdir()):
            raise VolumeImportError(f"{case_dir} already exists; pass --overwrite to replace it")
        if overwrite:
            shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    """Write deterministic JSON for sidecar manifests."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: Iterable[Path]) -> list[dict[str, JsonValue]]:
    """Hash a stable list of files for provenance capture."""
    artifacts: list[dict[str, JsonValue]] = []
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        artifacts.append(file_artifact(path))
    return artifacts


def file_artifact(path: Path) -> dict[str, JsonValue]:
    """Build a JSON artifact descriptor for a file."""
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def copy_or_compress_nifti(source: Path, destination: Path) -> None:
    """Normalize a local NIfTI file to the canonical ct.nii.gz filename."""
    suffixes = source.suffixes
    if suffixes[-2:] == [".nii", ".gz"]:
        shutil.copy2(source, destination)
        return
    if source.suffix == ".nii":
        with source.open("rb") as reader, gzip.open(destination, "wb") as writer:
            shutil.copyfileobj(reader, writer)
        return
    raise VolumeImportError(f"{source} must be a .nii or .nii.gz file")


def as_json_list(values: Iterable[JsonValue]) -> JsonValue:
    """Cast a homogenous iterable into the recursive JSON value type."""
    return cast(JsonValue, list(values))
