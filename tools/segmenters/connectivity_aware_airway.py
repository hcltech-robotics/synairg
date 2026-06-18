"""Run the Connectivity-Aware pulmonary airway segmenter on a NIfTI CT.

This adapter avoids the upstream MONAI/SimpleITK runtime path and uses nibabel
plus a small torch sliding-window loop so SynAirG can call it as an external
segmentation command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTER_ROOT = REPO_ROOT / "external" / "segmenters" / "Connectivity-Aware-Airway-Segmentaion"
DEFAULT_CHECKPOINT = DEFAULT_SEGMENTER_ROOT / "checkpoints" / "airway_model.pth"


def main() -> int:
    args = _parse_args()
    segmenter_root = args.segmenter_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    if not (segmenter_root / "networks" / "airway_network.py").is_file():
        raise SystemExit(f"Connectivity-Aware segmenter root not found: {segmenter_root}")

    sys.path.insert(0, str(segmenter_root))
    from networks.airway_network import UNet3D, lumTrans

    device = _resolve_device(args.device)
    image = nib.load(str(args.ct))
    ct = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    model_ct = np.transpose(ct, (2, 1, 0))
    normalized = _normalize_ct(lumTrans(model_ct) if args.hu_window else model_ct)

    state = torch.load(str(checkpoint_path), map_location="cpu")
    base_features = _infer_base_features(state)
    model = UNet3D(
        in_channels=1,
        out_channels=2,
        finalsigmoid=1,
        fmaps_degree=base_features,
        fmaps_layer_number=4,
        layer_order="cip",
        GroupNormNumber=4,
        device=[device],
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tensor = torch.from_numpy(normalized[None, None].astype(np.float32)).to(device)
    roi_size = tuple(int(value) for value in args.roi_size)
    probabilities = _sliding_window_predict(
        model,
        tensor,
        roi_size=roi_size,
        overlap=float(args.overlap),
        device=device,
    )
    mask = probabilities[0, 1].cpu().numpy() > float(args.threshold)
    if args.keep_largest_component:
        mask = _keep_largest_component(mask)
    mask = np.transpose(mask, (2, 1, 0))

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    out_image = nib.Nifti1Image(mask.astype(np.uint8), image.affine, image.header)
    out_image.header.set_data_dtype(np.uint8)
    nib.save(out_image, str(output))
    print(
        {
            "ct": str(args.ct),
            "output": str(output),
            "voxels": int(np.count_nonzero(mask)),
            "shape": [int(value) for value in mask.shape],
            "device": str(device),
            "roi_size": [int(value) for value in roi_size],
        }
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Connectivity-Aware airway segmentation NIfTI adapter.")
    parser.add_argument("--ct", required=True, type=Path, help="Input CT NIfTI.")
    parser.add_argument("--output", "--mask", required=True, type=Path, help="Output binary airway mask NIfTI.")
    parser.add_argument("--segmenter-root", type=Path, default=DEFAULT_SEGMENTER_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto.")
    parser.add_argument("--roi-size", nargs=3, type=int, default=(128, 128, 128))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hu-window", action="store_true", help="Apply the upstream lung HU window before normalize.")
    parser.add_argument("--keep-largest-component", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _normalize_ct(image: np.ndarray) -> np.ndarray:
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32)
    return np.asarray((image - minimum) / (maximum - minimum), dtype=np.float32)


def _infer_base_features(state: dict[str, torch.Tensor]) -> int:
    weight = state.get("final_conv.weight")
    if weight is None:
        raise SystemExit("checkpoint is missing final_conv.weight")
    return int(weight.shape[1])


@torch.no_grad()
def _sliding_window_predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    *,
    roi_size: tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> torch.Tensor:
    if not 0.0 <= overlap < 1.0:
        raise SystemExit("--overlap must be in [0.0, 1.0)")
    original_shape = tuple(int(value) for value in tensor.shape[2:])
    padded, crop_slices = _pad_to_roi(tensor, roi_size)
    _, _, depth, height, width = padded.shape
    starts = [
        _axis_starts(size, roi, overlap)
        for size, roi in zip((depth, height, width), roi_size, strict=False)
    ]
    output = torch.zeros((1, 2, depth, height, width), dtype=torch.float32, device=device)
    counts = torch.zeros_like(output)
    total = len(starts[0]) * len(starts[1]) * len(starts[2])
    done = 0
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                patch = padded[:, :, z : z + roi_size[0], y : y + roi_size[1], x : x + roi_size[2]]
                pred = model(patch)
                output[:, :, z : z + roi_size[0], y : y + roi_size[1], x : x + roi_size[2]] += pred
                counts[:, :, z : z + roi_size[0], y : y + roi_size[1], x : x + roi_size[2]] += 1
                done += 1
                if done == 1 or done == total or done % 4 == 0:
                    print(f"window {done}/{total}", file=sys.stderr)
    averaged = output / torch.clamp(counts, min=1.0)
    cropped = averaged[(slice(None), slice(None), *crop_slices)]
    assert tuple(int(value) for value in cropped.shape[2:]) == original_shape
    return cropped


def _pad_to_roi(
    tensor: torch.Tensor,
    roi_size: tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[slice, slice, slice]]:
    shape = tuple(int(value) for value in tensor.shape[2:])
    pad_after = [max(roi - size, 0) for size, roi in zip(shape, roi_size, strict=False)]
    padded = torch.nn.functional.pad(
        tensor,
        (0, pad_after[2], 0, pad_after[1], 0, pad_after[0]),
        mode="constant",
        value=0.0,
    )
    crop_slices = tuple(slice(0, size) for size in shape)
    return padded, crop_slices


def _axis_starts(size: int, roi: int, overlap: float) -> list[int]:
    if size <= roi:
        return [0]
    stride = max(int(round(roi * (1.0 - overlap))), 1)
    starts = list(range(0, size - roi + 1, stride))
    if starts[-1] != size - roi:
        starts.append(size - roi)
    return starts


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import label
    except ImportError:
        return mask
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = label(mask, structure=structure)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


if __name__ == "__main__":
    raise SystemExit(main())
