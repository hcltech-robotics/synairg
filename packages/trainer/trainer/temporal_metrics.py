"""Lightweight blind temporal-consistency metrics for generated scope videos."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
from synairg_core.volume_manifest import JsonValue


class TemporalMetricError(ValueError):
    """Raised when temporal metrics cannot be computed."""


@dataclass(frozen=True)
class TemporalMetricConfig:
    """Configuration for blind frame-to-frame temporal metrics."""

    video_path: Path
    frame_stride: int = 1
    max_frames: int | None = None


def compute_blind_temporal_metrics(config: TemporalMetricConfig) -> dict[str, JsonValue]:
    """Compute no-reference temporal metrics from a generated video."""
    if config.frame_stride < 1:
        raise TemporalMetricError("frame_stride must be positive")
    if config.max_frames is not None and config.max_frames < 2:
        raise TemporalMetricError("max_frames must be at least 2 when provided")
    if not config.video_path.is_file():
        raise TemporalMetricError(f"video does not exist: {config.video_path}")
    frames = _load_grayscale_frames(config)
    if len(frames) < 2:
        raise TemporalMetricError("at least two frames are required for temporal metrics")

    raw_absdiff: list[float] = []
    flicker_energy: list[float] = []
    global_luminance_delta: list[float] = []
    for previous, current in zip(frames[:-1], frames[1:], strict=True):
        if previous.shape != current.shape:
            raise TemporalMetricError("video frames must have a stable shape")
        delta = current - previous
        global_delta = float(np.median(delta))
        residual = delta - global_delta
        raw_absdiff.append(float(np.mean(np.abs(delta))))
        flicker_energy.append(float(np.mean(np.abs(residual))))
        global_luminance_delta.append(float(np.mean(delta)))

    return {
        "schema_version": "1.0",
        "video_path": str(config.video_path),
        "frame_count": len(frames),
        "frame_stride": int(config.frame_stride),
        "max_frames": int(config.max_frames) if config.max_frames is not None else None,
        "metric_notes": (
            "Blind temporal proxy. Flicker energy subtracts the per-frame median luminance delta; "
            "it does not use ground-truth optical flow or target RGB."
        ),
        "raw_absdiff_mean": _mean(raw_absdiff),
        "raw_absdiff_p95": _percentile(raw_absdiff, 95.0),
        "flicker_energy_mean": _mean(flicker_energy),
        "flicker_energy_p95": _percentile(flicker_energy, 95.0),
        "global_luminance_delta_mean": _mean(global_luminance_delta),
        "global_luminance_delta_abs_mean": _mean([abs(value) for value in global_luminance_delta]),
    }


def write_blind_temporal_metrics(config: TemporalMetricConfig, output_path: Path) -> dict[str, JsonValue]:
    """Compute and write blind temporal metrics as deterministic JSON."""
    metrics = compute_blind_temporal_metrics(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _load_grayscale_frames(config: TemporalMetricConfig) -> list[np.ndarray]:
    try:
        return _load_grayscale_frames_imageio(config)
    except (ImportError, OSError, RuntimeError, ValueError):
        if shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None:
            return _load_grayscale_frames_ffmpeg(config)
        raise TemporalMetricError(
            "imageio could not read the video and no ffmpeg/ffprobe fallback is available"
        ) from None


def _load_grayscale_frames_imageio(config: TemporalMetricConfig) -> list[np.ndarray]:
    try:
        imageio = cast(Any, import_module("imageio.v2"))
    except ImportError as exc:
        raise ImportError("imageio is required to read video files for temporal metrics") from exc

    frames: list[np.ndarray] = []
    reader = imageio.get_reader(config.video_path)
    try:
        for frame_index, frame in enumerate(reader):
            if frame_index % config.frame_stride != 0:
                continue
            frames.append(_to_grayscale(frame))
            if config.max_frames is not None and len(frames) >= config.max_frames:
                break
    finally:
        reader.close()
    return frames


def _load_grayscale_frames_ffmpeg(config: TemporalMetricConfig) -> list[np.ndarray]:
    width, height = _ffprobe_dimensions(config.video_path)
    frame_size = width * height * 3
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(config.video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise TemporalMetricError("ffmpeg did not expose stdout")
    frames: list[np.ndarray] = []
    raw_index = 0
    try:
        while True:
            chunk = process.stdout.read(frame_size)
            if not chunk:
                break
            if len(chunk) != frame_size:
                raise TemporalMetricError("ffmpeg emitted a partial video frame")
            if raw_index % config.frame_stride == 0:
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
                frames.append(_to_grayscale(frame))
                if config.max_frames is not None and len(frames) >= config.max_frames:
                    break
            raw_index += 1
    finally:
        if config.max_frames is not None and len(frames) >= config.max_frames:
            process.kill()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
        process.wait()
    if process.returncode not in (0, -9) and not frames:
        raise TemporalMetricError(f"ffmpeg could not read video: {stderr.strip()}")
    return frames


def _ffprobe_dimensions(path: Path) -> tuple[int, int]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise TemporalMetricError(f"ffprobe could not inspect video: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TemporalMetricError("ffprobe did not return usable video dimensions") from exc
    if width <= 0 or height <= 0:
        raise TemporalMetricError("video dimensions must be positive")
    return width, height


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame[..., :3], dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise TemporalMetricError("video frames must be RGB-like arrays")
    return np.asarray(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2], dtype=np.float32)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0
