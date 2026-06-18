"""Segmentation backend interfaces and precomputed-mask loader."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import nibabel as nib
import numpy as np
from synairg_core.volume_manifest import JsonValue

from mesh_gen.errors import MeshGenError
from mesh_gen.schemas import AirwayMask


@dataclass(frozen=True)
class SegmentationResult:
    """Binary mask emitted by a segmentation backend."""

    mask: AirwayMask
    backend_name: str
    metadata: dict[str, JsonValue] = field(default_factory=dict)


class SegmentationBackend(Protocol):
    """Protocol for pluggable airway segmentation backends."""

    name: str

    def segment(self, *, case_id: str, ct_path: Path, mask_path: Path | None = None) -> SegmentationResult:
        """Segment a CT volume into a binary airway mask."""


class PrecomputedMaskBackend:
    """Load and validate an already computed airway mask."""

    name = "precomputed-mask"

    def segment(self, *, case_id: str, ct_path: Path, mask_path: Path | None = None) -> SegmentationResult:
        """Read a mask NIfTI and validate it against the CT volume."""
        if mask_path is None:
            raise MeshGenError("precomputed-mask backend requires --mask")
        ct_image: Any = nib.load(str(ct_path))
        mask_image: Any = nib.load(str(mask_path))
        ct_shape = tuple(int(value) for value in ct_image.shape[:3])
        mask_shape = tuple(int(value) for value in mask_image.shape[:3])
        if len(ct_shape) != 3 or len(mask_shape) != 3:
            raise MeshGenError("CT and mask inputs must be 3D NIfTI volumes")
        if ct_shape != mask_shape:
            raise MeshGenError(f"CT shape {ct_shape} does not match mask shape {mask_shape}")
        ct_affine = np.asarray(ct_image.affine, dtype=np.float64)
        mask_affine = np.asarray(mask_image.affine, dtype=np.float64)
        if not np.allclose(ct_affine, mask_affine, atol=1.0e-3):
            raise MeshGenError("CT affine does not match mask affine")

        mask_data = np.asarray(mask_image.get_fdata(dtype=np.float32) > 0, dtype=np.bool_)
        spacing = _zooms(mask_image.header.get_zooms())
        mask = AirwayMask(case_id=case_id, data=mask_data, affine=mask_affine, spacing=spacing, source_path=mask_path)
        return SegmentationResult(
            mask=mask,
            backend_name=self.name,
            metadata={
                "ct_path": str(ct_path),
                "mask_path": str(mask_path),
                "shape": [int(value) for value in mask.data.shape],
                "spacing_mm": [float(value) for value in spacing],
            },
        )


class ExternalCommandSegmentationBackend:
    """Run an external airway segmenter command that writes a binary mask NIfTI."""

    name = "external-command"

    def __init__(self, command: str | None, *, timeout_seconds: float = 3600.0) -> None:
        self.command = command or os.environ.get("SYNAIRG_AIRWAY_SEGMENTER_COMMAND")
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise MeshGenError("segmenter timeout must be positive")

    def segment(self, *, case_id: str, ct_path: Path, mask_path: Path | None = None) -> SegmentationResult:
        """Run the configured command and validate its emitted mask."""
        del mask_path
        if not self.command:
            raise MeshGenError(
                "external-command backend requires --segmenter-command or SYNAIRG_AIRWAY_SEGMENTER_COMMAND"
            )
        with tempfile.TemporaryDirectory(prefix=f"{case_id}-airway-seg-") as temp_root:
            output_mask = Path(temp_root) / "airway_mask.nii.gz"
            argv = _format_command(self.command, case_id=case_id, ct_path=ct_path, output_mask=output_mask)
            completed = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise MeshGenError(f"airway segmenter exited with {completed.returncode}: {detail}")
            if not output_mask.is_file():
                raise MeshGenError(f"airway segmenter did not create {output_mask}")
            loaded = PrecomputedMaskBackend().segment(case_id=case_id, ct_path=ct_path, mask_path=output_mask)
            return SegmentationResult(
                mask=loaded.mask,
                backend_name=self.name,
                metadata={
                    "ct_path": str(ct_path),
                    "command": argv,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                    "shape": [int(value) for value in loaded.mask.data.shape],
                    "spacing_mm": [float(value) for value in loaded.mask.spacing],
                },
            )


class ThresholdSegmentationBackend:
    """Simple threshold backend useful for local experiments and backend tests."""

    name = "ct-threshold"

    def __init__(self, threshold_hu: float) -> None:
        self.threshold_hu = threshold_hu

    def segment(self, *, case_id: str, ct_path: Path, mask_path: Path | None = None) -> SegmentationResult:
        """Create a binary mask from CT voxels less than or equal to a threshold."""
        image: Any = nib.load(str(ct_path))
        data = np.asarray(image.get_fdata(dtype=np.float32) <= self.threshold_hu, dtype=np.bool_)
        affine = np.asarray(image.affine, dtype=np.float64)
        spacing = _zooms(image.header.get_zooms())
        mask = AirwayMask(case_id=case_id, data=data, affine=affine, spacing=spacing, source_path=ct_path)
        return SegmentationResult(
            mask=mask,
            backend_name=self.name,
            metadata={
                "ct_path": str(ct_path),
                "threshold_hu": float(self.threshold_hu),
                "shape": [int(value) for value in mask.data.shape],
                "spacing_mm": [float(value) for value in spacing],
            },
        )


def _zooms(values: tuple[float, ...]) -> tuple[float, float, float]:
    if len(values) < 3:
        raise MeshGenError("NIfTI header must provide three spatial zooms")
    spacing = (float(values[0]), float(values[1]), float(values[2]))
    if any(value <= 0 for value in spacing):
        raise MeshGenError("NIfTI spatial zooms must be positive")
    return spacing


def _format_command(command: str, *, case_id: str, ct_path: Path, output_mask: Path) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise MeshGenError(f"invalid segmenter command: {exc}") from exc
    if not tokens:
        raise MeshGenError("segmenter command must not be empty")
    replacements = {
        "case_id": case_id,
        "ct": str(ct_path),
        "ct_path": str(ct_path),
        "mask": str(output_mask),
        "output": str(output_mask),
        "output_mask": str(output_mask),
    }
    try:
        return [token.format(**replacements) for token in tokens]
    except KeyError as exc:
        raise MeshGenError(f"unknown segmenter command placeholder {{{exc.args[0]}}}") from exc
