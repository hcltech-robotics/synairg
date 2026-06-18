"""Mesh-to-volume registration sidecar generation and validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from synairg_core.volume_manifest import JsonValue

from mesh_gen.artifacts import MeshCasePaths, sha256_file, utc_timestamp, write_json
from mesh_gen.errors import MeshGenError
from mesh_gen.schemas import AirwayMask, AirwayMesh, CenterlineGraph

SCHEMA_VERSION = "1.0"
AFFINE_TOLERANCE_MM = 1.0e-3
ROUND_TRIP_TOLERANCE_MM = 1.0e-6
FloatMatrix: TypeAlias = NDArray[np.float64]
FloatVector: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class MeshRegistrationInputs:
    """Inputs needed to build a portable mesh-to-volume registration sidecar."""

    case_id: str
    ct_path: Path
    paths: MeshCasePaths
    mask: AirwayMask
    mesh: AirwayMesh
    graph: CenterlineGraph


def write_mesh_registration(inputs: MeshRegistrationInputs) -> dict[str, JsonValue]:
    """Write and validate the mesh registration sidecar for a generated case."""
    payload = build_mesh_registration(inputs)
    write_json(inputs.paths.mesh_registration_path, payload)
    validate_mesh_registration(inputs.paths.mesh_registration_path, base_dir=inputs.paths.case_dir)
    return payload


def build_mesh_registration(inputs: MeshRegistrationInputs) -> dict[str, JsonValue]:
    """Build the deterministic registration payload for one generated mesh case."""
    image: Any = nib.load(str(inputs.ct_path))
    ct_affine = np.asarray(image.affine, dtype=np.float64)
    mask_affine = np.asarray(inputs.mask.affine, dtype=np.float64)
    if ct_affine.shape != (4, 4):
        raise MeshGenError("CT affine must be 4x4 to build mesh registration")
    affine_delta = float(np.max(np.abs(ct_affine - mask_affine)))
    if affine_delta > AFFINE_TOLERANCE_MM:
        raise MeshGenError("CT affine does not match mask affine for mesh registration")

    volume_metadata_path = inputs.ct_path.parent / "volume_metadata.json"
    provenance_path = inputs.ct_path.parent / "source_provenance.json"
    volume_metadata = _read_json_if_exists(volume_metadata_path)
    provenance = _read_json_if_exists(provenance_path)
    dicom_reference = _dicom_reference_from_metadata(volume_metadata)

    voxel_to_volume = ct_affine
    volume_to_voxel = _invert_matrix(voxel_to_volume, "voxel_index_to_volume_world_mm")
    mesh_to_volume = np.eye(4, dtype=np.float64)
    voxel_to_dicom = _dicom_voxel_matrix(dicom_reference)
    volume_to_dicom = None if voxel_to_dicom is None else voxel_to_dicom @ volume_to_voxel
    mesh_to_dicom = None if volume_to_dicom is None else volume_to_dicom @ mesh_to_volume

    transforms = {
        "voxel_index_to_volume_world_mm": _matrix_to_json(voxel_to_volume),
        "volume_world_mm_to_voxel_index": _matrix_to_json(volume_to_voxel),
        "mesh_world_mm_to_volume_world_mm": _matrix_to_json(mesh_to_volume),
        "volume_world_mm_to_dicom_patient_lps_mm": _optional_matrix_to_json(volume_to_dicom),
        "mesh_world_mm_to_dicom_patient_lps_mm": _optional_matrix_to_json(mesh_to_dicom),
        "voxel_index_to_dicom_patient_lps_mm": _optional_matrix_to_json(voxel_to_dicom),
    }
    validation = _validation_payload(
        ct_affine=ct_affine,
        mask_affine=mask_affine,
        mesh=inputs.mesh,
        graph=inputs.graph,
        transforms=transforms,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": inputs.case_id,
        "created_at_utc": utc_timestamp(),
        "source_volume": _source_volume_payload(
            ct_path=inputs.ct_path,
            volume_metadata_path=volume_metadata_path,
            provenance_path=provenance_path,
            volume_metadata=volume_metadata,
            provenance=provenance,
            base_dir=inputs.paths.case_dir,
        ),
        "mesh_artifacts": _mesh_artifacts_payload(inputs.paths),
        "coordinate_frames": _coordinate_frames_payload(dicom_reference is not None),
        "transforms": transforms,
        "dicom_reference": dicom_reference,
        "privacy": _privacy_payload(),
        "validation": validation,
    }


def validate_mesh_registration(path: Path, *, base_dir: Path | None = None) -> dict[str, JsonValue]:
    """Validate a registration sidecar's matrices, hashes, and embedded round-trip checks."""
    payload = _read_required_json(path)
    root = base_dir or path.parent
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")

    transforms = payload.get("transforms")
    if not isinstance(transforms, dict):
        errors.append("transforms must be an object")
    else:
        for name, value in transforms.items():
            if value is None:
                continue
            try:
                matrix = _matrix_from_json(value, str(name))
            except MeshGenError as exc:
                errors.append(str(exc))
                continue
            try:
                _invert_matrix(matrix, str(name))
            except MeshGenError as exc:
                errors.append(str(exc))

    for descriptor in _artifact_descriptors(payload.get("source_volume")):
        _validate_artifact_descriptor(descriptor, root, errors)
    for descriptor in _artifact_descriptors(payload.get("mesh_artifacts")):
        _validate_artifact_descriptor(descriptor, root, errors)

    validation = payload.get("validation")
    if not isinstance(validation, dict):
        errors.append("validation must be an object")
    else:
        checks = validation.get("checks")
        if not isinstance(checks, dict) or not all(isinstance(value, bool) and value for value in checks.values()):
            errors.append("validation checks did not all pass")

    status = "pass" if not errors else "fail"
    result: dict[str, JsonValue] = {"status": status, "errors": cast(JsonValue, list(errors))}
    if errors:
        raise MeshGenError("invalid mesh registration sidecar: " + "; ".join(errors))
    return result


def _source_volume_payload(
    *,
    ct_path: Path,
    volume_metadata_path: Path,
    provenance_path: Path,
    volume_metadata: dict[str, JsonValue] | None,
    provenance: dict[str, JsonValue] | None,
    base_dir: Path,
) -> dict[str, JsonValue]:
    return {
        "source_type": _source_type(volume_metadata, provenance),
        "ct.nii.gz": _file_descriptor(ct_path, base_dir),
        "volume_metadata.json": (
            _file_descriptor(volume_metadata_path, base_dir) if volume_metadata_path.is_file() else None
        ),
        "source_provenance.json": _file_descriptor(provenance_path, base_dir) if provenance_path.is_file() else None,
    }


def _mesh_artifacts_payload(paths: MeshCasePaths) -> dict[str, JsonValue]:
    artifacts = (
        paths.airway_mask_path,
        paths.obj_path,
        paths.ply_path,
        paths.usd_path,
        paths.centerline_graph_path,
        paths.branch_semantics_path,
        paths.radius_profile_path,
        paths.preview_path,
        paths.segmentation_overlay_path,
        paths.mesh_review_path,
        paths.viewer_path,
    )
    return {path.name: _file_descriptor(path, paths.case_dir) for path in artifacts}


def _coordinate_frames_payload(has_dicom_reference: bool) -> dict[str, JsonValue]:
    return {
        "voxel_index": {
            "units": "voxel",
            "axis_order": ["i", "j", "k"],
            "convention": "zero-based NIfTI voxel index coordinates",
            "handedness": "defined_by_affine",
        },
        "volume_world_mm": {
            "units": "mm",
            "axis_order": ["x", "y", "z"],
            "convention": "NIfTI affine world coordinates for the shipped CT volume",
            "handedness": "defined_by_affine",
        },
        "mesh_world_mm": {
            "units": "mm",
            "axis_order": ["x", "y", "z"],
            "convention": "mesh vertices and centerline point_mm values share volume_world_mm",
            "handedness": "defined_by_volume_world_mm",
        },
        "dicom_patient_lps_mm": {
            "units": "mm",
            "axis_order": ["L", "P", "S"],
            "convention": "DICOM patient LPS coordinates",
            "available": has_dicom_reference,
            "handedness": "patient_lps",
        },
    }


def _privacy_payload() -> dict[str, JsonValue]:
    return {
        "patient_identifiers": "omitted",
        "dates": "omitted",
        "private_tags": "omitted",
        "dicom_uid_policy": "sha256_when_available",
        "artifact_paths": "relative_to_mesh_registration_json",
    }


def _validation_payload(
    *,
    ct_affine: FloatMatrix,
    mask_affine: FloatMatrix,
    mesh: AirwayMesh,
    graph: CenterlineGraph,
    transforms: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    affine_delta = float(np.max(np.abs(ct_affine - mask_affine)))
    checks = {
        "ct_mask_affine_consistent": affine_delta <= AFFINE_TOLERANCE_MM,
        "transform_matrices_invertible": _all_transform_matrices_invertible(transforms),
        "mesh_vertices_finite": bool(np.isfinite(mesh.vertices).all()),
        "centerline_points_finite": bool(
            all(np.isfinite(np.asarray(node.point_mm, dtype=np.float64)).all() for node in graph.nodes)
        ),
    }
    round_trip = _round_trip_payload(mesh=mesh, graph=graph, transforms=transforms)
    round_trip_error = _round_trip_max_error(round_trip)
    checks["round_trip_points_within_tolerance"] = round_trip_error <= ROUND_TRIP_TOLERANCE_MM
    return {
        "tolerances": {
            "ct_mask_affine_atol_mm": AFFINE_TOLERANCE_MM,
            "round_trip_atol_mm": ROUND_TRIP_TOLERANCE_MM,
        },
        "ct_mask_affine_max_abs_diff_mm": affine_delta,
        "round_trip": round_trip,
        "checks": {key: bool(value) for key, value in checks.items()},
    }


def _round_trip_payload(
    *,
    mesh: AirwayMesh,
    graph: CenterlineGraph,
    transforms: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    mesh_to_volume = _matrix_from_json(
        transforms["mesh_world_mm_to_volume_world_mm"], "mesh_world_mm_to_volume_world_mm"
    )
    volume_to_voxel = _matrix_from_json(transforms["volume_world_mm_to_voxel_index"], "volume_world_mm_to_voxel_index")
    voxel_to_volume = _matrix_from_json(transforms["voxel_index_to_volume_world_mm"], "voxel_index_to_volume_world_mm")
    samples = [
        ("mesh_vertex", "0", np.asarray(mesh.vertices[0], dtype=np.float64)),
        ("centerline_point", graph.nodes[0].node_id, np.asarray(graph.nodes[0].point_mm, dtype=np.float64)),
    ]
    payload_samples: list[JsonValue] = []
    max_error = 0.0
    for sample_type, sample_id, point in samples:
        volume_world = _transform_point(mesh_to_volume, point)
        voxel_index = _transform_point(volume_to_voxel, volume_world)
        reconstructed = _transform_point(voxel_to_volume, voxel_index)
        error_mm = float(np.linalg.norm(reconstructed - volume_world))
        max_error = max(max_error, error_mm)
        payload_samples.append(
            {
                "type": sample_type,
                "id": sample_id,
                "mesh_world_mm": _vector_to_json(point),
                "volume_voxel_index": _vector_to_json(voxel_index),
                "reconstructed_volume_world_mm": _vector_to_json(reconstructed),
                "error_mm": error_mm,
            }
        )
    return {
        "sample_count": len(payload_samples),
        "max_error_mm": max_error,
        "samples": payload_samples,
    }


def _round_trip_max_error(payload: dict[str, JsonValue]) -> float:
    value = payload.get("max_error_mm")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return float("inf")


def _dicom_reference_from_metadata(volume_metadata: dict[str, JsonValue] | None) -> dict[str, JsonValue] | None:
    if volume_metadata is None:
        return None
    volume = volume_metadata.get("volume")
    if not isinstance(volume, dict):
        return None
    reference = volume.get("dicom_reference")
    if not isinstance(reference, dict):
        return None
    allowed_keys = {
        "coordinate_frame",
        "frame_of_reference_uid_sha256",
        "image_orientation_patient",
        "image_positions_patient",
        "instance_order",
        "modality",
        "pixel_spacing_mm",
        "privacy",
        "series_instance_uid_sha256",
        "slice_spacing_derivation",
        "slice_spacing_mm",
        "source_file_hashes",
        "study_instance_uid_sha256",
        "voxel_index_to_dicom_patient_lps_mm",
    }
    return {key: value for key, value in reference.items() if key in allowed_keys}


def _dicom_voxel_matrix(dicom_reference: dict[str, JsonValue] | None) -> FloatMatrix | None:
    if dicom_reference is None:
        return None
    raw_matrix = dicom_reference.get("voxel_index_to_dicom_patient_lps_mm")
    if raw_matrix is None:
        return None
    return _matrix_from_json(raw_matrix, "dicom_reference.voxel_index_to_dicom_patient_lps_mm")


def _source_type(
    volume_metadata: dict[str, JsonValue] | None,
    provenance: dict[str, JsonValue] | None,
) -> JsonValue:
    if volume_metadata is not None:
        source_type = volume_metadata.get("source_type")
        if isinstance(source_type, str):
            return source_type
    if provenance is not None:
        source = provenance.get("source")
        if isinstance(source, dict):
            source_type = source.get("type")
            if isinstance(source_type, str):
                return source_type
    return "unknown"


def _read_json_if_exists(path: Path) -> dict[str, JsonValue] | None:
    if not path.is_file():
        return None
    return _read_required_json(path)


def _read_required_json(path: Path) -> dict[str, JsonValue]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MeshGenError(f"{path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise MeshGenError(f"{path} must contain a JSON object")
    return cast(dict[str, JsonValue], payload)


def _file_descriptor(path: Path, base_dir: Path) -> dict[str, JsonValue]:
    stat = path.stat()
    return {
        "path": _relative_path(path, base_dir),
        "bytes": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def _relative_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), start=base_dir.resolve())


def _artifact_descriptors(value: object) -> list[dict[str, JsonValue]]:
    descriptors: list[dict[str, JsonValue]] = []
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            descriptors.append(cast(dict[str, JsonValue], value))
        else:
            for item in value.values():
                descriptors.extend(_artifact_descriptors(item))
    elif isinstance(value, list):
        for item in value:
            descriptors.extend(_artifact_descriptors(item))
    return descriptors


def _validate_artifact_descriptor(descriptor: dict[str, JsonValue], base_dir: Path, errors: list[str]) -> None:
    raw_path = descriptor.get("path")
    raw_bytes = descriptor.get("bytes")
    raw_sha256 = descriptor.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(raw_bytes, int) or not isinstance(raw_sha256, str):
        errors.append("artifact descriptor must include path, bytes, and sha256")
        return
    descriptor_path = Path(raw_path)
    if descriptor_path.is_absolute():
        errors.append(f"artifact path {raw_path!r} must be relative")
        return
    resolved_path = (base_dir / descriptor_path).resolve()
    if not resolved_path.is_file():
        errors.append(f"artifact {raw_path!r} is missing")
        return
    if resolved_path.stat().st_size != raw_bytes:
        errors.append(f"artifact {raw_path!r} byte size changed")
    if sha256_file(resolved_path) != raw_sha256:
        errors.append(f"artifact {raw_path!r} sha256 changed")


def _all_transform_matrices_invertible(transforms: dict[str, JsonValue]) -> bool:
    for name, value in transforms.items():
        if value is None:
            continue
        try:
            _invert_matrix(_matrix_from_json(value, name), name)
        except MeshGenError:
            return False
    return True


def _matrix_from_json(value: object, name: str) -> FloatMatrix:
    if not isinstance(value, list) or len(value) != 4:
        raise MeshGenError(f"{name} must be a 4x4 matrix")
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise MeshGenError(f"{name} must be a 4x4 matrix")
        try:
            rows.append([float(item) for item in row])
        except (TypeError, ValueError) as exc:
            raise MeshGenError(f"{name} must contain numeric values") from exc
    matrix: FloatMatrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise MeshGenError(f"{name} must contain finite values")
    return matrix


def _invert_matrix(matrix: FloatMatrix, name: str) -> FloatMatrix:
    try:
        return cast(FloatMatrix, np.linalg.inv(matrix))
    except np.linalg.LinAlgError as exc:
        raise MeshGenError(f"{name} must be invertible") from exc


def _matrix_to_json(matrix: FloatMatrix) -> JsonValue:
    return [[float(matrix[row, column]) for column in range(4)] for row in range(4)]


def _optional_matrix_to_json(matrix: FloatMatrix | None) -> JsonValue:
    return None if matrix is None else _matrix_to_json(matrix)


def _vector_to_json(vector: FloatVector) -> JsonValue:
    return [float(value) for value in vector[:3]]


def _transform_point(matrix: FloatMatrix, point: FloatVector) -> FloatVector:
    homogeneous: FloatVector = np.ones(4, dtype=np.float64)
    homogeneous[:3] = point[:3]
    transformed: FloatVector = matrix @ homogeneous
    return np.asarray(transformed[:3], dtype=np.float64)
