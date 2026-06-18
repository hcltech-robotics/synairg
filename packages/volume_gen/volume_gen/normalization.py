"""Normalization and conversion routines for SynAirG CT volumes."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from synairg_core.volume_manifest import JsonValue

from volume_gen.artifacts import as_json_list, copy_or_compress_nifti, sha256_file
from volume_gen.errors import VolumeImportError


def normalize_local_nifti(source: Path, destination: Path) -> dict[str, JsonValue]:
    """Copy or compress a local NIfTI input to the canonical SynAirG filename."""
    if not source.exists() or not source.is_file():
        raise VolumeImportError(f"{source} is not a readable file")
    copy_or_compress_nifti(source, destination)
    return {
        "canonical_filename": destination.name,
        "normalization": "copy_or_gzip",
    }


def convert_dicom_folder(input_dir: Path, destination: Path) -> dict[str, JsonValue]:
    """Convert a folder of uncompressed CT DICOM slices into a canonical NIfTI file."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise VolumeImportError(f"{input_dir} is not a readable directory")

    pydicom = _import_optional_module(
        "pydicom",
        "DICOM import requires pydicom. Install SynAirG with its runtime dependencies and retry.",
    )
    nibabel = _import_optional_module(
        "nibabel",
        "DICOM import requires nibabel. Install SynAirG with its runtime dependencies and retry.",
    )
    pydicom_api = cast(Any, pydicom)
    nibabel_api = cast(Any, nibabel)
    dcmread = pydicom_api.dcmread

    raw_records: list[tuple[Path, object]] = []
    for path in sorted(candidate_dicom_files(input_dir)):
        try:
            dataset = dcmread(str(path), force=True)
        except Exception:
            continue
        if not hasattr(dataset, "PixelData"):
            continue
        raw_records.append((path, dataset))

    if not raw_records:
        raise VolumeImportError(f"{input_dir} does not contain readable DICOM image slices")

    orientation = _orientation_for(raw_records[0][1])
    records = [
        (_slice_sort_key(dataset, float(index), orientation[2]), path, dataset)
        for index, (path, dataset) in enumerate(raw_records)
    ]
    records.sort(key=lambda record: (record[0], str(record[1])))
    images: list[npt.NDArray[np.float32]] = []
    for _, _, dataset in records:
        pixel_array = cast(Any, dataset).pixel_array
        array = cast(npt.NDArray[np.float32], np.asarray(pixel_array, dtype=np.float32))
        slope = _float_attr(dataset, "RescaleSlope", 1.0)
        intercept = _float_attr(dataset, "RescaleIntercept", 0.0)
        images.append(array * np.float32(slope) + np.float32(intercept))

    shapes = {image.shape for image in images}
    if len(shapes) != 1:
        raise VolumeImportError("DICOM slices have inconsistent pixel dimensions")

    volume = cast(npt.NDArray[np.float32], np.stack(images, axis=-1))
    affine = _affine_for(records)
    nifti_image = nibabel_api.Nifti1Image(volume, affine)
    nibabel_api.save(nifti_image, str(destination))

    first_dataset = records[0][2]
    return {
        "canonical_filename": destination.name,
        "normalization": "dicom_to_nifti",
        "dicom_file_count": len(records),
        "shape": as_json_list(int(value) for value in volume.shape),
        "voxel_spacing": as_json_list(float(np.linalg.norm(affine[:3, index])) for index in range(3)),
        "modality": str(getattr(first_dataset, "Modality", "CT")),
        "dicom_reference": _dicom_reference(records, affine),
    }


def candidate_dicom_files(input_dir: Path) -> list[Path]:
    """Return candidate DICOM files beneath an input directory."""
    return [path for path in input_dir.rglob("*") if path.is_file()]


def _import_optional_module(module_name: str, error_message: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise VolumeImportError(error_message) from exc


def _float_attr(dataset: object, attribute: str, default: float) -> float:
    value = getattr(dataset, attribute, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _slice_sort_key(dataset: object, fallback: float, slice_direction: npt.NDArray[np.float64]) -> float:
    position = _float_sequence(getattr(dataset, "ImagePositionPatient", None), length=3)
    if position is not None:
        return float(np.dot(np.asarray(position, dtype=np.float64), slice_direction))
    return _float_attr(dataset, "InstanceNumber", fallback)


def _affine_for(records: list[tuple[float, Path, object]]) -> npt.NDArray[np.float64]:
    first_dataset = records[0][2]
    row_direction, column_direction, slice_direction = _orientation_for(first_dataset)
    pixel_spacing = getattr(first_dataset, "PixelSpacing", [1.0, 1.0])
    row_spacing = _sequence_float(pixel_spacing, 0, 1.0)
    column_spacing = _sequence_float(pixel_spacing, 1, 1.0)
    slice_spacing, _ = _slice_spacing(records, slice_direction)
    origin = _float_sequence(getattr(first_dataset, "ImagePositionPatient", None), length=3) or (0.0, 0.0, 0.0)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = column_direction * row_spacing
    affine[:3, 1] = row_direction * column_spacing
    affine[:3, 2] = slice_direction * slice_spacing
    affine[:3, 3] = np.asarray(origin, dtype=np.float64)
    return affine


def _sequence_float(values: Any, index: int, default: float) -> float:
    try:
        return float(values[index])
    except (IndexError, TypeError, ValueError):
        return default


def _slice_spacing(
    records: list[tuple[float, Path, object]],
    slice_direction: npt.NDArray[np.float64],
) -> tuple[float, str]:
    first_dataset = records[0][2]
    if len(records) > 1:
        positions = [
            _float_sequence(getattr(record[2], "ImagePositionPatient", None), length=3) for record in records
        ]
        if all(position is not None for position in positions):
            projections = [
                float(np.dot(np.asarray(cast(tuple[float, float, float], position), dtype=np.float64), slice_direction))
                for position in positions
            ]
            deltas = [abs(projections[index + 1] - projections[index]) for index in range(len(projections) - 1)]
            positive_deltas = [delta for delta in deltas if delta > 0]
            if positive_deltas:
                return (float(np.median(np.asarray(positive_deltas, dtype=np.float64))), "image_position_projection")
    return (_float_attr(first_dataset, "SliceThickness", 1.0), "slice_thickness")


def _dicom_reference(
    records: list[tuple[float, Path, object]],
    affine: npt.NDArray[np.float64],
) -> dict[str, JsonValue]:
    first_dataset = records[0][2]
    row_direction, column_direction, slice_direction = _orientation_for(first_dataset)
    pixel_spacing = getattr(first_dataset, "PixelSpacing", [1.0, 1.0])
    row_spacing = _sequence_float(pixel_spacing, 0, 1.0)
    column_spacing = _sequence_float(pixel_spacing, 1, 1.0)
    slice_spacing, slice_spacing_derivation = _slice_spacing(records, slice_direction)
    instance_order: list[dict[str, JsonValue]] = []
    source_file_hashes: list[dict[str, JsonValue]] = []
    for ordinal, (_, path, dataset) in enumerate(records):
        source_sha256 = sha256_file(path)
        source_file_hashes.append({"ordinal": ordinal, "filename": path.name, "sha256": source_sha256})
        instance_order.append(
            {
                "ordinal": ordinal,
                "filename": path.name,
                "source_sha256": source_sha256,
                "sop_instance_uid_sha256": _hash_uid(getattr(dataset, "SOPInstanceUID", None)),
                "instance_number": _optional_int(getattr(dataset, "InstanceNumber", None)),
                "image_position_patient": _json_float_triplet(
                    _float_sequence(getattr(dataset, "ImagePositionPatient", None), length=3)
                ),
                "rescale_slope": _float_attr(dataset, "RescaleSlope", 1.0),
                "rescale_intercept": _float_attr(dataset, "RescaleIntercept", 0.0),
            }
        )
    return {
        "coordinate_frame": "DICOM patient LPS",
        "privacy": {
            "patient_identifiers": "omitted",
            "dates": "omitted",
            "private_tags": "omitted",
            "dicom_uid_policy": "sha256",
        },
        "study_instance_uid_sha256": _hash_uid(getattr(first_dataset, "StudyInstanceUID", None)),
        "series_instance_uid_sha256": _hash_uid(getattr(first_dataset, "SeriesInstanceUID", None)),
        "frame_of_reference_uid_sha256": _hash_uid(getattr(first_dataset, "FrameOfReferenceUID", None)),
        "image_orientation_patient": as_json_list(float(value) for value in (*row_direction, *column_direction)),
        "image_positions_patient": [
            _json_float_triplet(_float_sequence(getattr(record[2], "ImagePositionPatient", None), length=3))
            for record in records
        ],
        "pixel_spacing_mm": [float(row_spacing), float(column_spacing)],
        "slice_spacing_mm": float(slice_spacing),
        "slice_spacing_derivation": slice_spacing_derivation,
        "modality": str(getattr(first_dataset, "Modality", "CT")),
        "voxel_index_to_dicom_patient_lps_mm": [
            [float(affine[row, column]) for column in range(4)] for row in range(4)
        ],
        "instance_order": cast(JsonValue, instance_order),
        "source_file_hashes": cast(JsonValue, source_file_hashes),
    }


def _orientation_for(dataset: object) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    image_orientation = _float_sequence(getattr(dataset, "ImageOrientationPatient", None), length=6)
    if image_orientation is None:
        row_direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        column_direction = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        row_direction = np.asarray(image_orientation[:3], dtype=np.float64)
        column_direction = np.asarray(image_orientation[3:], dtype=np.float64)
    row_direction = _normalized(row_direction, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    column_direction = _normalized(column_direction, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    slice_direction = _normalized(
        np.asarray(np.cross(row_direction, column_direction), dtype=np.float64),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    )
    return row_direction, column_direction, slice_direction


def _normalized(
    vector: npt.NDArray[np.float64],
    default: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        return default
    return np.asarray(vector / norm, dtype=np.float64)


def _float_sequence(value: object, *, length: int) -> tuple[float, ...] | None:
    if isinstance(value, str | bytes):
        return None
    try:
        if len(cast(Any, value)) < length:
            return None
    except TypeError:
        return None
    values: list[float] = []
    for index in range(length):
        try:
            values.append(float(cast(Any, value)[index]))
        except (IndexError, TypeError, ValueError):
            return None
    if any(not np.isfinite(item) for item in values):
        return None
    return tuple(values)


def _json_float_triplet(value: tuple[float, ...] | None) -> JsonValue:
    if value is None or len(value) < 3:
        return None
    return [float(value[0]), float(value[1]), float(value[2])]


def _optional_int(value: object) -> JsonValue:
    try:
        return int(cast(Any, value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _hash_uid(value: object) -> JsonValue:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
