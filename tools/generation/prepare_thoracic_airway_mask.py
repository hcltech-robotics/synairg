#!/usr/bin/env python3
"""Prepare a thoracic MAISI source mask for airway-focused CT generation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

BODY_LABEL = 200

ANCHOR_LABELS = (28, 29, 30, 31, 32, 115, 132)
THORACIC_LABELS = (
    6,  # aorta
    11,  # esophagus
    23,  # lung tumor, when present in a chest source mask
    28,
    29,
    30,
    31,
    32,
    *range(38, 58),  # T12-C1 vertebrae plus trachea
    *range(63, 93),  # ribs, scapulae, clavicles
    104,
    105,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    119,
    121,
    122,
    123,
    124,
    125,
    126,
    132,
)
ABDOMINAL_LABELS = (
    1,
    3,
    4,
    5,
    7,
    8,
    9,
    10,
    12,
    13,
    14,
    15,
    17,
    19,
    24,
    25,
    62,
    93,
    94,
    95,
    106,
    107,
    116,
    117,
    118,
)


@dataclass(frozen=True)
class ThoracicMaskPrep:
    """Settings for reducing a full-body MAISI mask to thoracic context."""

    anchor_labels: tuple[int, ...] = ANCHOR_LABELS
    thoracic_labels: tuple[int, ...] = THORACIC_LABELS
    abdominal_labels: tuple[int, ...] = ABDOMINAL_LABELS
    body_label: int = BODY_LABEL
    slice_axis: int = 2
    inferior_margin_slices: int = 0
    superior_margin_slices: int = 8


def prepare_thoracic_airway_mask(
    source_mask_path: Path,
    output_mask_path: Path,
    *,
    prep: ThoracicMaskPrep | None = None,
) -> dict[str, Any]:
    """Write a source mask that keeps thoracic labels and removes abdominal context."""
    prep = prep or ThoracicMaskPrep()
    image = nib.load(str(source_mask_path))
    data = np.asarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D source mask, got shape {data.shape}")
    if prep.slice_axis not in (0, 1, 2):
        raise ValueError("slice_axis must be 0, 1, or 2")
    if prep.inferior_margin_slices < 0 or prep.superior_margin_slices < 0:
        raise ValueError("crop margins must be non-negative")

    labels = np.rint(data).astype(np.uint16, copy=False)
    if not np.allclose(data, labels):
        raise ValueError("source mask must contain integer-like label ids")

    anchors = np.isin(labels, np.asarray(prep.anchor_labels, dtype=np.uint16))
    anchor_coordinates = np.argwhere(anchors)
    if anchor_coordinates.size == 0:
        raise ValueError(f"source mask does not contain thoracic anchor labels {list(prep.anchor_labels)}")

    anchor_min = int(anchor_coordinates[:, prep.slice_axis].min())
    anchor_max = int(anchor_coordinates[:, prep.slice_axis].max())
    crop_min = max(0, anchor_min - prep.inferior_margin_slices)
    crop_max = min(labels.shape[prep.slice_axis] - 1, anchor_max + prep.superior_margin_slices)
    crop = _axis_window(labels.shape, prep.slice_axis, crop_min, crop_max)

    source_body = labels > 0
    thoracic = np.isin(labels, np.asarray(prep.thoracic_labels, dtype=np.uint16))
    output = np.zeros_like(labels, dtype=np.uint16)
    output[source_body & crop] = np.uint16(prep.body_label)
    output[thoracic & crop] = labels[thoracic & crop]

    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    out_image = nib.Nifti1Image(output, image.affine, image.header)
    out_image.header.set_data_dtype(np.uint16)
    nib.save(out_image, str(output_mask_path))

    input_labels, input_counts = np.unique(labels, return_counts=True)
    output_labels, output_counts = np.unique(output, return_counts=True)
    removed_abdominal = {
        str(label): int(np.count_nonzero((labels == label) & (output != label)))
        for label in prep.abdominal_labels
        if np.any(labels == label)
    }
    removed_foreground_labels = sorted(
        int(label)
        for label in input_labels
        if label != 0 and label not in set(int(v) for v in output_labels)
    )
    return {
        "source_mask": str(source_mask_path),
        "output_mask": str(output_mask_path),
        "shape": [int(v) for v in labels.shape],
        "slice_axis": prep.slice_axis,
        "anchor_labels": [int(v) for v in prep.anchor_labels],
        "thoracic_labels": [int(v) for v in prep.thoracic_labels],
        "body_label": prep.body_label,
        "thoracic_crop_min_index": crop_min,
        "thoracic_crop_max_index": crop_max,
        "input_label_counts": _label_counts(input_labels, input_counts),
        "output_label_counts": _label_counts(output_labels, output_counts),
        "removed_foreground_labels": removed_foreground_labels,
        "removed_abdominal_voxels": removed_abdominal,
        "body_voxels": int(np.count_nonzero(output == prep.body_label)),
    }


def main() -> int:
    args = _parse_args()
    config = _load_config(args.config)
    base_dir = args.config.parent if args.config is not None else Path.cwd()
    source_mask = _path_from_config_or_arg(config, "source_mask_path", args.source_mask, base_dir)
    output_mask = _path_from_config_or_arg(config, "output_mask_path", args.output, base_dir)
    prep = ThoracicMaskPrep(
        anchor_labels=tuple(int(v) for v in config.get("anchor_labels", args.anchor_label)),
        thoracic_labels=tuple(int(v) for v in config.get("thoracic_labels", args.thoracic_label)),
        abdominal_labels=tuple(int(v) for v in config.get("abdominal_labels", args.abdominal_label)),
        body_label=int(config.get("body_label", args.body_label)),
        slice_axis=int(config.get("slice_axis", args.slice_axis)),
        inferior_margin_slices=int(config.get("inferior_margin_slices", args.inferior_margin_slices)),
        superior_margin_slices=int(config.get("superior_margin_slices", args.superior_margin_slices)),
    )
    report = prepare_thoracic_airway_mask(source_mask, output_mask, prep=prep)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON config for source/output paths and prep settings.")
    parser.add_argument("--source-mask", type=Path, help="Input MAISI source mask NIfTI.")
    parser.add_argument("--output", type=Path, help="Output thoracic source mask NIfTI.")
    parser.add_argument("--anchor-label", action="append", type=int, default=list(ANCHOR_LABELS))
    parser.add_argument("--thoracic-label", action="append", type=int, default=list(THORACIC_LABELS))
    parser.add_argument("--abdominal-label", action="append", type=int, default=list(ABDOMINAL_LABELS))
    parser.add_argument("--body-label", default=BODY_LABEL, type=int)
    parser.add_argument("--slice-axis", default=2, type=int)
    parser.add_argument("--inferior-margin-slices", default=0, type=int)
    parser.add_argument("--superior-margin-slices", default=8, type=int)
    args = parser.parse_args()
    if args.config is None and (args.source_mask is None or args.output is None):
        parser.error("--source-mask and --output are required unless --config is provided")
    return args


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _path_from_config_or_arg(config: dict[str, Any], key: str, arg: Path | None, base_dir: Path) -> Path:
    value = config.get(key)
    if value is None:
        if arg is None:
            raise ValueError(f"{key} is required")
        path = arg
    else:
        path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _axis_window(shape: tuple[int, int, int], axis: int, min_index: int, max_index: int) -> np.ndarray:
    indices = np.arange(shape[axis])
    one_dimensional = (indices >= min_index) & (indices <= max_index)
    reshape = [1, 1, 1]
    reshape[axis] = shape[axis]
    return np.broadcast_to(one_dimensional.reshape(reshape), shape)


def _label_counts(labels: np.ndarray, counts: np.ndarray) -> dict[str, int]:
    return {str(int(label)): int(count) for label, count in zip(labels, counts, strict=True) if int(label) != 0}


if __name__ == "__main__":
    raise SystemExit(main())
