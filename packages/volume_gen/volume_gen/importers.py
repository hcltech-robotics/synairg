"""Import workflows for canonical SynAirG CT volume cases."""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path
from typing import cast

from synairg_core.volume_manifest import (
    JsonValue,
    ManifestValidationError,
    VolumeManifest,
    VolumeSource,
    read_volume_manifest,
)

from volume_gen.artifacts import VolumeCase, file_artifact, hash_files, prepare_case_dir, utc_timestamp, write_json
from volume_gen.normalization import candidate_dicom_files, convert_dicom_folder, normalize_local_nifti


def import_local_nifti(manifest_path: Path, output_root: Path, *, overwrite: bool = False) -> VolumeCase:
    """Import a local NIfTI volume described by a manifest."""
    manifest = read_volume_manifest(manifest_path)
    if manifest.source.source_type != "local_nifti":
        raise ManifestValidationError("import-local requires source.type to be local_nifti")
    source_path = resolve_local_uri(manifest.source.uri, base_dir=manifest_path.parent)
    case_dir = prepare_case_dir(output_root, manifest.case_id, overwrite=overwrite)
    ct_path = case_dir / "ct.nii.gz"
    volume_metadata = normalize_local_nifti(source_path, ct_path)
    return write_case_sidecars(
        manifest=manifest,
        case_dir=case_dir,
        ct_path=ct_path,
        source_artifacts=[source_path],
        volume_metadata=volume_metadata,
        pipeline="volume_gen.import_local_nifti",
    )


def import_dicom_folder(
    input_dir: Path,
    output_root: Path,
    *,
    case_id: str | None = None,
    overwrite: bool = False,
) -> VolumeCase:
    """Import a local DICOM folder as a canonical NIfTI volume case."""
    selected_case_id = case_id or default_case_id(input_dir.name)
    source = VolumeSource(
        source_type="local_dicom",
        uri=str(input_dir),
        metadata={"input_directory": str(input_dir)},
    )
    manifest = VolumeManifest(case_id=selected_case_id, source=source)
    case_dir = prepare_case_dir(output_root, manifest.case_id, overwrite=overwrite)
    ct_path = case_dir / "ct.nii.gz"
    volume_metadata = convert_dicom_folder(input_dir, ct_path)
    return write_case_sidecars(
        manifest=manifest,
        case_dir=case_dir,
        ct_path=ct_path,
        source_artifacts=candidate_dicom_files(input_dir),
        volume_metadata=volume_metadata,
        pipeline="volume_gen.import_dicom_folder",
    )


def write_case_sidecars(
    *,
    manifest: VolumeManifest,
    case_dir: Path,
    ct_path: Path,
    source_artifacts: list[Path],
    volume_metadata: dict[str, JsonValue],
    pipeline: str,
    output_artifacts: list[Path] | None = None,
) -> VolumeCase:
    """Write standard metadata and provenance sidecars for a canonical CT case."""
    metadata_path = case_dir / "volume_metadata.json"
    provenance_path = case_dir / "source_provenance.json"
    created_at = utc_timestamp()
    ct_artifact = file_artifact(ct_path)
    input_artifacts = hash_files(source_artifacts)
    extra_output_artifacts = hash_files(output_artifacts or [])
    artifacts_payload: dict[str, JsonValue] = {"ct": ct_artifact}
    outputs_payload: list[dict[str, JsonValue]] = [ct_artifact]
    if extra_output_artifacts:
        artifacts_payload["extra_outputs"] = cast(JsonValue, extra_output_artifacts)
        outputs_payload = [ct_artifact, *extra_output_artifacts]

    metadata_payload: dict[str, JsonValue] = {
        "schema_version": "1.0",
        "case_id": manifest.case_id,
        "modality": "CT",
        "source_type": manifest.source.source_type,
        "created_at_utc": created_at,
        "artifacts": artifacts_payload,
        "volume": volume_metadata,
        "case_metadata": manifest.metadata,
    }
    provenance_payload: dict[str, JsonValue] = {
        "schema_version": "1.0",
        "case_id": manifest.case_id,
        "generated_at_utc": created_at,
        "pipeline": pipeline,
        "source": manifest.source.to_dict(),
        "inputs": cast(JsonValue, input_artifacts),
        "outputs": cast(JsonValue, outputs_payload),
    }
    write_json(metadata_path, metadata_payload)
    write_json(provenance_path, provenance_payload)
    return VolumeCase(
        case_id=manifest.case_id,
        case_dir=case_dir,
        ct_path=ct_path,
        metadata_path=metadata_path,
        provenance_path=provenance_path,
    )


def resolve_local_uri(uri: str | None, *, base_dir: Path) -> Path:
    """Resolve a manifest URI into a local filesystem path."""
    if uri is None:
        raise ManifestValidationError("source.uri is required for local NIfTI imports")
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        raise ManifestValidationError("local NIfTI source.uri must be a local path or file:// URI")
    raw_path = urllib.parse.unquote(parsed.path if parsed.scheme == "file" else uri)
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path


def default_case_id(name: str) -> str:
    """Create a safe case ID from a folder name."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-_")
    if not normalized:
        return "dicom-case"
    return normalized
