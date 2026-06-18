"""Filesystem and JSON helpers for mesh generation artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from synairg_core.volume_manifest import JsonValue

from mesh_gen.errors import MeshGenError


@dataclass(frozen=True)
class MeshCasePaths:
    """Canonical per-case paths emitted by the mesh workflow."""

    case_id: str
    case_dir: Path
    airway_mask_path: Path
    obj_path: Path
    ply_path: Path
    usd_path: Path
    centerline_graph_path: Path
    branch_semantics_path: Path
    radius_profile_path: Path
    mesh_registration_path: Path
    quality_report_path: Path
    preview_path: Path
    segmentation_overlay_path: Path
    mesh_review_path: Path
    viewer_path: Path


def build_case_paths(output_root: Path, case_id: str) -> MeshCasePaths:
    """Build the canonical mesh case paths under an output root."""
    case_dir = output_root / case_id
    return MeshCasePaths(
        case_id=case_id,
        case_dir=case_dir,
        airway_mask_path=case_dir / "airway_mask.nii.gz",
        obj_path=case_dir / "airway_mesh.obj",
        ply_path=case_dir / "airway_mesh.ply",
        usd_path=case_dir / "airway.usd",
        centerline_graph_path=case_dir / "centerline_graph.json",
        branch_semantics_path=case_dir / "branch_semantics.json",
        radius_profile_path=case_dir / "radius_profile.parquet",
        mesh_registration_path=case_dir / "mesh_registration.json",
        quality_report_path=case_dir / "quality_report.json",
        preview_path=case_dir / "preview.png",
        segmentation_overlay_path=case_dir / "segmentation_overlay.png",
        mesh_review_path=case_dir / "mesh_review.png",
        viewer_path=case_dir / "viewer.html",
    )


def prepare_case_dir(output_root: Path, case_id: str, *, overwrite: bool) -> MeshCasePaths:
    """Create a per-case output directory, optionally replacing an existing one."""
    paths = build_case_paths(output_root, case_id)
    if paths.case_dir.exists():
        if not overwrite and any(paths.case_dir.iterdir()):
            raise MeshGenError(f"{paths.case_dir} already exists; pass --overwrite to replace it")
        if overwrite:
            shutil.rmtree(paths.case_dir)
    paths.case_dir.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    """Write deterministic JSON with stable key ordering."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_timestamp() -> str:
    """Return the current UTC timestamp in JSON-friendly ISO-8601 form."""
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_artifact(path: Path) -> dict[str, JsonValue]:
    """Build a JSON descriptor for an emitted file."""
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def artifact_manifest(paths: Iterable[Path]) -> dict[str, JsonValue]:
    """Build a deterministic mapping of artifact filename to descriptor."""
    return {path.name: file_artifact(path) for path in sorted(paths)}
