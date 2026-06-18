"""Temporal training manifest helpers for BronchoGen diffusion datasets."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from synairg_core.volume_manifest import JsonValue, validate_case_id

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CONDITION_SUFFIXES = IMAGE_SUFFIXES | {".exr", ".npy", ".npz"}
SUPPORTED_CONDITIONS = {
    "depth",
    "edge",
    "flow",
    "mask",
    "normal",
    "orifice_mask",
    "pps",
}
SCHEMA_VERSION = "1.0"


class BronchoGenManifestError(ValueError):
    """Raised when a BronchoGen training manifest cannot be built or validated."""


@dataclass(frozen=True)
class BronchoGenManifestConfig:
    """Configuration for building a temporally grouped diffusion training manifest."""

    rgb_root: Path
    case_id: str
    dataset_name: str = "bronchogen"
    caption: str = "bronchoscopic airway view"
    condition_roots: dict[str, Path] = field(default_factory=dict)
    output_path: Path | None = None
    sequence_depth: int = 1
    temporal_context_radius: int = 2
    fps: float | None = None
    require_conditions: bool = True


@dataclass(frozen=True)
class _IndexedFrame:
    frame_id: str
    sequence_id: str
    frame_index: int
    rgb_path: Path
    rgb_relative_path: Path
    condition_paths: dict[str, Path]


def build_bronchogen_manifest(config: BronchoGenManifestConfig) -> dict[str, JsonValue]:
    """Build a manifest for image, ControlNet, or temporal diffusion training."""
    _validate_config(config)
    path_base = _path_base(config)
    sequences = _indexed_sequences(config)
    frame_entries: list[dict[str, JsonValue]] = []
    sequence_entries: list[dict[str, JsonValue]] = []
    condition_names = sorted(config.condition_roots)

    for sequence_id in sorted(sequences):
        frames = sequences[sequence_id]
        frame_ids = [frame.frame_id for frame in frames]
        sequence_entries.append(
            {
                "sequence_id": sequence_id,
                "frame_count": len(frames),
                "frame_ids": cast(JsonValue, frame_ids),
            }
        )
        for position, frame in enumerate(frames):
            neighbors = _temporal_neighbors(frame_ids, position, radius=config.temporal_context_radius)
            timestamp_seconds = None if config.fps is None else float(frame.frame_index / config.fps)
            entry: dict[str, JsonValue] = {
                "frame_id": frame.frame_id,
                "case_id": config.case_id,
                "sequence_id": frame.sequence_id,
                "frame_index": frame.frame_index,
                "rgb_path": _json_path(frame.rgb_path, base_dir=path_base),
                "rgb_relative_path": frame.rgb_relative_path.as_posix(),
                "caption": config.caption,
                "condition_paths": {
                    name: _json_path(path, base_dir=path_base) for name, path in sorted(frame.condition_paths.items())
                },
                "temporal_neighbor_frame_ids": cast(JsonValue, neighbors),
            }
            if timestamp_seconds is not None:
                entry["timestamp_seconds"] = timestamp_seconds
            frame_entries.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": config.dataset_name,
        "case_id": config.case_id,
        "task": "bronchoscopic_conditional_diffusion",
        "path_base": str(path_base),
        "conditioning": {
            "modalities": cast(JsonValue, condition_names),
            "missing_condition_policy": "error" if config.require_conditions else "omit",
            "recommended_control_stack": cast(
                JsonValue,
                [
                    "depth",
                    "normal",
                    "pps",
                    "mask",
                    "edge",
                    "flow",
                ],
            ),
        },
        "temporal": {
            "sequence_count": len(sequence_entries),
            "context_radius": int(config.temporal_context_radius),
            "fps": float(config.fps) if config.fps is not None else None,
            "contract": (
                "Pixel generators must preserve neighboring frame anatomy and illumination continuity. "
                "Use temporal_neighbor_frame_ids for frame-consistency losses or video model windows."
            ),
        },
        "sequences": cast(JsonValue, sequence_entries),
        "frames": cast(JsonValue, frame_entries),
    }


def write_bronchogen_manifest(config: BronchoGenManifestConfig) -> dict[str, JsonValue]:
    """Build and write a BronchoGen manifest to ``config.output_path``."""
    if config.output_path is None:
        raise BronchoGenManifestError("output_path is required when writing a manifest")
    manifest = build_bronchogen_manifest(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_bronchogen_manifest(path: Path, *, check_files: bool = True) -> dict[str, JsonValue]:
    """Validate a written BronchoGen manifest and optionally verify referenced files."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BronchoGenManifestError(f"could not read manifest {path}: {exc}") from exc
    mapping = _require_mapping(payload, "manifest")
    errors: list[str] = []
    if mapping.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    frames = mapping.get("frames")
    if not isinstance(frames, list) or not frames:
        errors.append("frames must be a non-empty list")
    frame_ids: set[str] = set()
    path_base = Path(str(mapping.get("path_base") or path.parent))
    if isinstance(frames, list):
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                errors.append(f"frames[{index}] must be an object")
                continue
            frame_id = frame.get("frame_id")
            if not isinstance(frame_id, str) or not frame_id:
                errors.append(f"frames[{index}].frame_id must be a non-empty string")
            elif frame_id in frame_ids:
                errors.append(f"duplicate frame_id {frame_id!r}")
            else:
                frame_ids.add(frame_id)
            _validate_frame_path(frame, "rgb_path", path_base, check_files, errors, index)
            condition_paths = frame.get("condition_paths")
            if not isinstance(condition_paths, dict):
                errors.append(f"frames[{index}].condition_paths must be an object")
            else:
                for name, value in condition_paths.items():
                    if not isinstance(name, str) or name not in SUPPORTED_CONDITIONS:
                        errors.append(f"frames[{index}] has unsupported condition {name!r}")
                    if not isinstance(value, str):
                        errors.append(f"frames[{index}].condition_paths.{name} must be a string path")
                    elif check_files and not _resolve_path(value, path_base).is_file():
                        errors.append(f"frames[{index}].condition_paths.{name} does not exist: {value}")
            neighbors = frame.get("temporal_neighbor_frame_ids")
            if not isinstance(neighbors, list) or not all(isinstance(item, str) for item in neighbors):
                errors.append(f"frames[{index}].temporal_neighbor_frame_ids must be a list of frame IDs")
    if errors:
        raise BronchoGenManifestError("invalid BronchoGen manifest: " + "; ".join(errors))
    return {"status": "pass", "frame_count": len(frame_ids)}


def _validate_config(config: BronchoGenManifestConfig) -> None:
    validate_case_id(config.case_id)
    if not config.dataset_name:
        raise BronchoGenManifestError("dataset_name must be non-empty")
    if not config.caption:
        raise BronchoGenManifestError("caption must be non-empty")
    if config.sequence_depth < 0:
        raise BronchoGenManifestError("sequence_depth must be non-negative")
    if config.temporal_context_radius < 0:
        raise BronchoGenManifestError("temporal_context_radius must be non-negative")
    if config.fps is not None and config.fps <= 0:
        raise BronchoGenManifestError("fps must be positive when provided")
    if not config.rgb_root.is_dir():
        raise BronchoGenManifestError(f"rgb_root is not a directory: {config.rgb_root}")
    for name, root in config.condition_roots.items():
        if name not in SUPPORTED_CONDITIONS:
            expected = ", ".join(sorted(SUPPORTED_CONDITIONS))
            raise BronchoGenManifestError(f"unsupported condition {name!r}; expected one of: {expected}")
        if not root.is_dir():
            raise BronchoGenManifestError(f"{name} condition root is not a directory: {root}")


def _indexed_sequences(config: BronchoGenManifestConfig) -> dict[str, list[_IndexedFrame]]:
    rgb_paths = _rgb_paths(config.rgb_root)
    if not rgb_paths:
        raise BronchoGenManifestError(f"no RGB images found under {config.rgb_root}")

    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for path in rgb_paths:
        relative_path = path.relative_to(config.rgb_root)
        grouped_paths[_sequence_id(relative_path, depth=config.sequence_depth)].append(path)

    sequences: dict[str, list[_IndexedFrame]] = {}
    for sequence_id in sorted(grouped_paths):
        frames: list[_IndexedFrame] = []
        for frame_index, rgb_path in enumerate(sorted(grouped_paths[sequence_id])):
            relative_path = rgb_path.relative_to(config.rgb_root)
            condition_paths = _condition_paths(config, relative_path)
            frame_id = _frame_id(config.case_id, sequence_id, frame_index)
            frames.append(
                _IndexedFrame(
                    frame_id=frame_id,
                    sequence_id=sequence_id,
                    frame_index=frame_index,
                    rgb_path=rgb_path,
                    rgb_relative_path=relative_path,
                    condition_paths=condition_paths,
                )
            )
        sequences[sequence_id] = frames
    return sequences


def _rgb_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _condition_paths(config: BronchoGenManifestConfig, relative_rgb_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for name, root in sorted(config.condition_roots.items()):
        path = _matching_condition_path(root, relative_rgb_path)
        if path is None:
            missing.append(name)
            continue
        paths[name] = path
    if missing and config.require_conditions:
        raise BronchoGenManifestError(
            f"missing condition files for {relative_rgb_path.as_posix()}: {', '.join(missing)}"
        )
    return paths


def _matching_condition_path(root: Path, relative_rgb_path: Path) -> Path | None:
    exact = root / relative_rgb_path
    if exact.is_file():
        return exact
    base = (root / relative_rgb_path).with_suffix("")
    for suffix in sorted(CONDITION_SUFFIXES):
        candidate = base.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def _sequence_id(relative_path: Path, *, depth: int) -> str:
    if depth == 0 or len(relative_path.parts) == 1:
        return "default"
    parts = relative_path.parts[: min(depth, len(relative_path.parts) - 1)]
    return "/".join(parts)


def _frame_id(case_id: str, sequence_id: str, frame_index: int) -> str:
    safe_sequence = re.sub(r"[^A-Za-z0-9_.-]+", "-", sequence_id).strip("-") or "default"
    return f"{case_id}-{safe_sequence}-{frame_index:06d}"


def _temporal_neighbors(frame_ids: list[str], position: int, *, radius: int) -> list[str]:
    if radius == 0:
        return []
    start = max(0, position - radius)
    stop = min(len(frame_ids), position + radius + 1)
    return [frame_ids[index] for index in range(start, stop) if index != position]


def _path_base(config: BronchoGenManifestConfig) -> Path:
    if config.output_path is not None:
        return config.output_path.parent.resolve()
    return config.rgb_root.parent.resolve()


def _json_path(path: Path, *, base_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base_dir).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_path(value: str, path_base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else path_base / path


def _validate_frame_path(
    frame: Mapping[str, object],
    field_name: str,
    path_base: Path,
    check_files: bool,
    errors: list[str],
    index: int,
) -> None:
    value = frame.get(field_name)
    if not isinstance(value, str) or not value:
        errors.append(f"frames[{index}].{field_name} must be a non-empty string path")
        return
    if check_files and not _resolve_path(value, path_base).is_file():
        errors.append(f"frames[{index}].{field_name} does not exist: {value}")


def _require_mapping(value: object, field_name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise BronchoGenManifestError(f"{field_name} must be an object")
    return cast(dict[str, JsonValue], value)
