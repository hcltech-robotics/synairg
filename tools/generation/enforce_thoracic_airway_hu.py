#!/usr/bin/env python3
"""Enforce airway-lumen CT attenuation for MAISI thoracic source-mask cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.morphology import dilation


@dataclass(frozen=True)
class AirwayHuCorrection:
    """Parameters for CT intensity correction from a labelled source mask."""

    airway_label: int = 132
    airway_labels: tuple[int, ...] | None = None
    airway_hu: float = -1000.0
    halo_hu: float = -750.0
    outside_body_hu: float = 0.0
    halo_iterations: int = 1


def enforce_thoracic_airway_hu(
    ct_path: Path,
    source_mask_path: Path,
    output_path: Path,
    *,
    correction: AirwayHuCorrection | None = None,
) -> dict[str, object]:
    """Write a corrected CT where the source-mask airway is air-like and background is not."""
    correction = correction or AirwayHuCorrection()
    ct_image = nib.load(str(ct_path))
    mask_image = nib.load(str(source_mask_path))
    ct = np.asarray(ct_image.get_fdata(dtype=np.float32), dtype=np.float32)
    labels = np.asarray(mask_image.dataobj)
    if ct.shape != labels.shape:
        raise ValueError(f"CT shape {ct.shape} does not match source mask shape {labels.shape}")
    if not np.allclose(ct_image.affine, mask_image.affine, atol=1.0e-5):
        raise ValueError("CT affine does not match source mask affine")
    if correction.halo_iterations < 0:
        raise ValueError("halo_iterations must be non-negative")

    airway_labels = correction.airway_labels or (correction.airway_label,)
    airway_labels = tuple(int(label) for label in airway_labels)
    if not airway_labels:
        raise ValueError("at least one airway label is required")

    airway = np.isin(labels, airway_labels)
    if not np.any(airway):
        raise ValueError(f"source mask does not contain any airway labels {list(airway_labels)}")
    body = labels > 0
    corrected = ct.copy()
    corrected[~body] = correction.outside_body_hu

    halo = np.zeros_like(airway, dtype=np.bool_)
    dilated = airway
    for _ in range(correction.halo_iterations):
        dilated = dilation(dilated)
    if correction.halo_iterations > 0:
        halo = dilated & ~airway & body
        corrected[halo] = np.minimum(corrected[halo], correction.halo_hu)
    corrected[airway] = correction.airway_hu

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_image = nib.Nifti1Image(corrected.astype(np.float32), ct_image.affine, ct_image.header)
    out_image.header.set_data_dtype(np.float32)
    nib.save(out_image, str(output_path))

    return {
        "ct": str(ct_path),
        "source_mask": str(source_mask_path),
        "output": str(output_path),
        "shape": list(ct.shape),
        "airway_label": airway_labels[0] if len(airway_labels) == 1 else None,
        "airway_labels": list(airway_labels),
        "airway_voxels": int(np.count_nonzero(airway)),
        "halo_voxels": int(np.count_nonzero(halo)),
        "outside_body_voxels": int(np.count_nonzero(~body)),
        "airway_hu": correction.airway_hu,
        "halo_hu": correction.halo_hu,
        "outside_body_hu": correction.outside_body_hu,
    }


def main() -> int:
    args = _parse_args()
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        args.ct = Path(config["ct_path"])
        args.source_mask = Path(config["source_mask_path"])
        args.output = Path(config["output_path"])
        if "airway_labels" in config:
            args.airway_labels = [int(label) for label in config["airway_labels"]]
        else:
            args.airway_label = int(config.get("airway_label", args.airway_label))
        args.airway_hu = float(config.get("airway_hu", args.airway_hu))
        args.halo_hu = float(config.get("halo_hu", args.halo_hu))
        args.outside_body_hu = float(config.get("outside_body_hu", args.outside_body_hu))
        args.halo_iterations = int(config.get("halo_iterations", args.halo_iterations))
    report = enforce_thoracic_airway_hu(
        args.ct,
        args.source_mask,
        args.output,
        correction=AirwayHuCorrection(
            airway_label=args.airway_label,
            airway_labels=tuple(args.airway_labels) if args.airway_labels is not None else None,
            airway_hu=args.airway_hu,
            halo_hu=args.halo_hu,
            outside_body_hu=args.outside_body_hu,
            halo_iterations=args.halo_iterations,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON config with CT, source mask, output, and correction values.")
    parser.add_argument("--ct", type=Path, help="Generated CT NIfTI to correct.")
    parser.add_argument("--source-mask", type=Path, help="MAISI labelled source mask NIfTI.")
    parser.add_argument("--output", type=Path, help="Corrected CT NIfTI output path.")
    parser.add_argument("--airway-label", default=132, type=int, help="Source-mask integer label for airway lumen.")
    parser.add_argument(
        "--airway-labels",
        nargs="+",
        type=int,
        help="One or more source-mask integer labels to treat as airway lumen.",
    )
    parser.add_argument("--airway-hu", default=-1000.0, type=float, help="HU assigned to the airway label.")
    parser.add_argument(
        "--halo-hu",
        default=-750.0,
        type=float,
        help="Maximum HU assigned to the one-voxel airway halo.",
    )
    parser.add_argument("--outside-body-hu", default=0.0, type=float, help="HU assigned to source-mask background.")
    parser.add_argument("--halo-iterations", default=1, type=int, help="Binary dilation iterations for airway halo.")
    args = parser.parse_args()
    if args.config is None and (args.ct is None or args.source_mask is None or args.output is None):
        parser.error("--ct, --source-mask, and --output are required unless --config is provided")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
