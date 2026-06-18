#!/usr/bin/env python3
"""Evaluate candidate airway masks against a trusted reference mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from mesh_gen.preview import _fit_nearest_2d, _window_ct, _write_png


def evaluate_airway_masks(
    reference_path: Path,
    candidates: dict[str, Path],
    *,
    ct_path: Path | None = None,
    output_json: Path | None = None,
    output_png: Path | None = None,
    min_dice: float = 0.85,
    allow_affine_mismatch: bool = False,
) -> dict[str, Any]:
    """Evaluate candidate masks and optionally write JSON plus a 2D comparison PNG."""
    reference_image = nib.load(str(reference_path))
    reference = _load_mask(reference_image)
    ct = None
    if ct_path is not None:
        ct_image = nib.load(str(ct_path))
        _assert_compatible(reference_image, ct_image, "reference", "ct")
        ct = np.asarray(ct_image.get_fdata(dtype=np.float32), dtype=np.float32)

    candidate_arrays: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for name, path in candidates.items():
        candidate_image = nib.load(str(path))
        _assert_compatible(
            reference_image,
            candidate_image,
            "reference",
            name,
            allow_affine_mismatch=allow_affine_mismatch,
        )
        candidate = _load_mask(candidate_image)
        candidate_arrays[name] = candidate
        candidate_metrics = _metrics(reference, candidate)
        candidate_metrics["path"] = str(path)
        candidate_metrics["affine_matches_reference"] = bool(
            np.allclose(reference_image.affine, candidate_image.affine, atol=1.0e-5)
        )
        candidate_metrics["passes_min_dice"] = bool(candidate_metrics["dice"] >= min_dice)
        metrics[name] = candidate_metrics

    report = {
        "reference": str(reference_path),
        "ct": str(ct_path) if ct_path is not None else None,
        "shape": list(reference.shape),
        "reference_voxels": int(np.count_nonzero(reference)),
        "min_dice": float(min_dice),
        "candidates": metrics,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        _write_comparison_png(output_png, reference, candidate_arrays, ct=ct)
    return report


def main() -> int:
    args = _parse_args()
    candidates = {name: Path(path) for name, path in args.candidate}
    report = evaluate_airway_masks(
        args.reference,
        candidates,
        ct_path=args.ct,
        output_json=args.output_json,
        output_png=args.output_png,
        min_dice=args.min_dice,
        allow_affine_mismatch=args.allow_affine_mismatch,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Trusted binary airway reference mask.")
    parser.add_argument("--ct", type=Path, help="CT NIfTI used as the PNG background.")
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="Candidate mask name and NIfTI path. Repeat for multiple candidates.",
    )
    parser.add_argument("--output-json", type=Path, help="Write a JSON metric report.")
    parser.add_argument("--output-png", type=Path, help="Write a 2D comparison PNG.")
    parser.add_argument("--min-dice", default=0.85, type=float, help="Dice threshold for passes_min_dice.")
    parser.add_argument(
        "--allow-affine-mismatch",
        action="store_true",
        help="Evaluate by voxel index even if candidate mask affine differs; the JSON report records this.",
    )
    return parser.parse_args()


def _load_mask(image: Any) -> np.ndarray:
    data = np.asarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D mask, got shape {data.shape}")
    return data > 0


def _assert_compatible(
    left: Any,
    right: Any,
    left_name: str,
    right_name: str,
    *,
    allow_affine_mismatch: bool = False,
) -> None:
    if left.shape[:3] != right.shape[:3]:
        raise ValueError(f"{right_name} shape {right.shape[:3]} does not match {left_name} shape {left.shape[:3]}")
    if not allow_affine_mismatch and not np.allclose(left.affine, right.affine, atol=1.0e-5):
        raise ValueError(f"{right_name} affine does not match {left_name} affine")


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference_voxels = int(np.count_nonzero(reference))
    candidate_voxels = int(np.count_nonzero(candidate))
    intersection = int(np.count_nonzero(reference & candidate))
    union = int(np.count_nonzero(reference | candidate))
    false_positive = int(np.count_nonzero(candidate & ~reference))
    false_negative = int(np.count_nonzero(reference & ~candidate))
    denominator = reference_voxels + candidate_voxels
    bbox = _bbox(candidate)
    return {
        "candidate_voxels": candidate_voxels,
        "reference_voxels": reference_voxels,
        "intersection_voxels": intersection,
        "union_voxels": union,
        "false_positive_voxels": false_positive,
        "false_negative_voxels": false_negative,
        "dice": float((2 * intersection) / denominator) if denominator else 1.0,
        "iou": float(intersection / union) if union else 1.0,
        "precision": float(intersection / candidate_voxels) if candidate_voxels else 0.0,
        "recall": float(intersection / reference_voxels) if reference_voxels else 0.0,
        "bbox_min_ijk": bbox[0],
        "bbox_max_ijk": bbox[1],
    }


def _bbox(mask: np.ndarray) -> tuple[list[int] | None, list[int] | None]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None, None
    return coordinates.min(axis=0).astype(int).tolist(), coordinates.max(axis=0).astype(int).tolist()


def _write_comparison_png(
    path: Path,
    reference: np.ndarray,
    candidates: dict[str, np.ndarray],
    *,
    ct: np.ndarray | None,
    tile_size: int = 256,
) -> None:
    views = _view_slices(reference)
    columns = 1 + len(candidates)
    rows = len(views)
    gutter = 10
    header_height = 22
    canvas = np.full(
        (
            header_height + rows * tile_size + (rows - 1) * gutter,
            columns * tile_size + (columns - 1) * gutter,
            3,
        ),
        24,
        dtype=np.uint8,
    )
    for column, title in enumerate(("reference", *candidates.keys())):
        _draw_label(canvas, column * (tile_size + gutter) + 6, 8, title, np.array([238, 241, 244], dtype=np.uint8))
    for row, (_view_name, slicer) in enumerate(views):
        reference_slice = slicer(reference)
        background = slicer(ct) if ct is not None else np.zeros_like(reference_slice, dtype=np.float32)
        panels = [_overlay_tile(background, reference_slice, reference_slice)]
        for candidate in candidates.values():
            panels.append(_overlay_tile(background, reference_slice, slicer(candidate)))
        for column, panel in enumerate(panels):
            y0 = header_height + row * (tile_size + gutter)
            x0 = column * (tile_size + gutter)
            canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = panel
    _write_png(path, canvas)


def _view_slices(reference: np.ndarray):
    z_index = int(np.argmax(reference.sum(axis=(0, 1))))
    y_index = int(np.argmax(reference.sum(axis=(0, 2))))
    x_index = int(np.argmax(reference.sum(axis=(1, 2))))
    return (
        ("axial", lambda array: array[:, :, z_index].T),
        ("coronal", lambda array: array[:, y_index, :].T),
        ("sagittal", lambda array: array[x_index, :, :].T),
    )


def _overlay_tile(background: np.ndarray, reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if background.dtype == np.bool_:
        grayscale = np.asarray(background, dtype=np.uint8) * 255
    else:
        grayscale = _window_ct(np.asarray(background, dtype=np.float32))
    base = _fit_nearest_2d(grayscale, size=256, fill_value=0)
    reference_image = _fit_nearest_2d(np.asarray(reference, dtype=np.uint8), size=256, fill_value=0).astype(bool)
    candidate_image = _fit_nearest_2d(np.asarray(candidate, dtype=np.uint8), size=256, fill_value=0).astype(bool)
    rgb = np.repeat(base[:, :, None], 3, axis=2)
    rgb[reference_image] = ((0.35 * rgb[reference_image]) + (0.65 * np.array([255, 140, 25]))).astype(np.uint8)
    rgb[candidate_image] = ((0.35 * rgb[candidate_image]) + (0.65 * np.array([45, 220, 255]))).astype(np.uint8)
    rgb[reference_image & candidate_image] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def _draw_label(canvas: np.ndarray, x: int, y: int, text: str, color: np.ndarray) -> None:
    cursor = x
    for character in text.lower()[:48]:
        glyph = _FONT.get(character, _FONT[" "])
        for row, values in enumerate(glyph):
            for column, value in enumerate(values):
                if value == "1":
                    yy = y + row
                    xx = cursor + column
                    if 0 <= yy < canvas.shape[0] and 0 <= xx < canvas.shape[1]:
                        canvas[yy, xx] = color
        cursor += 4
        if cursor >= canvas.shape[1] - 3:
            return


_FONT = {
    " ": ("000", "000", "000", "000", "000"),
    "-": ("000", "000", "111", "000", "000"),
    "_": ("000", "000", "000", "000", "111"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "a": ("010", "101", "111", "101", "101"),
    "b": ("110", "101", "110", "101", "110"),
    "c": ("111", "100", "100", "100", "111"),
    "d": ("110", "101", "101", "101", "110"),
    "e": ("111", "100", "111", "100", "111"),
    "f": ("111", "100", "111", "100", "100"),
    "g": ("111", "100", "101", "101", "111"),
    "h": ("101", "101", "111", "101", "101"),
    "i": ("111", "010", "010", "010", "111"),
    "j": ("001", "001", "001", "101", "111"),
    "k": ("101", "101", "110", "101", "101"),
    "l": ("100", "100", "100", "100", "111"),
    "m": ("101", "111", "111", "101", "101"),
    "n": ("101", "111", "111", "111", "101"),
    "o": ("111", "101", "101", "101", "111"),
    "p": ("111", "101", "111", "100", "100"),
    "q": ("111", "101", "101", "111", "001"),
    "r": ("110", "101", "110", "101", "101"),
    "s": ("111", "100", "111", "001", "111"),
    "t": ("111", "010", "010", "010", "010"),
    "u": ("101", "101", "101", "101", "111"),
    "v": ("101", "101", "101", "101", "010"),
    "w": ("101", "101", "111", "111", "101"),
    "x": ("101", "101", "010", "101", "101"),
    "y": ("101", "101", "010", "010", "010"),
    "z": ("111", "001", "010", "100", "111"),
}


if __name__ == "__main__":
    raise SystemExit(main())
