from __future__ import annotations

import json

import numpy as np
import pytest
from synairg_core.cli import app
from trainer.temporal_metrics import (
    TemporalMetricConfig,
    TemporalMetricError,
    compute_blind_temporal_metrics,
    write_blind_temporal_metrics,
)
from typer.testing import CliRunner

runner = CliRunner()

imageio = pytest.importorskip("imageio.v2")


def test_blind_temporal_metrics_compensate_global_luminance_delta(tmp_path) -> None:
    video_path = tmp_path / "uniform.mp4"
    writer = imageio.get_writer(video_path, fps=4, macro_block_size=1)
    try:
        for value in (20, 40, 60, 80):
            writer.append_data(np.full((12, 16, 3), value, dtype=np.uint8))
    finally:
        writer.close()

    metrics = compute_blind_temporal_metrics(TemporalMetricConfig(video_path=video_path))

    assert metrics["frame_count"] == 4
    assert metrics["raw_absdiff_mean"] > 0
    assert metrics["flicker_energy_mean"] < 1.0e-6


def test_blind_temporal_metrics_detect_spatial_flicker(tmp_path) -> None:
    video_path = tmp_path / "flicker.mp4"
    writer = imageio.get_writer(video_path, fps=4, macro_block_size=1)
    try:
        for index in range(4):
            frame = np.full((12, 16, 3), 60, dtype=np.uint8)
            frame[:, ::2] = 120 if index % 2 else 20
            writer.append_data(frame)
    finally:
        writer.close()

    metrics = compute_blind_temporal_metrics(TemporalMetricConfig(video_path=video_path))

    assert metrics["flicker_energy_mean"] > 0.05
    assert metrics["flicker_energy_p95"] >= metrics["flicker_energy_mean"]


def test_blind_temporal_metrics_reject_short_videos(tmp_path) -> None:
    video_path = tmp_path / "one-frame.mp4"
    writer = imageio.get_writer(video_path, fps=4, macro_block_size=1)
    try:
        writer.append_data(np.full((12, 16, 3), 20, dtype=np.uint8))
    finally:
        writer.close()

    with pytest.raises(TemporalMetricError, match="at least two frames"):
        compute_blind_temporal_metrics(TemporalMetricConfig(video_path=video_path))


def test_train_cli_evaluates_temporal_consistency(tmp_path) -> None:
    video_path = tmp_path / "uniform.mp4"
    output_path = tmp_path / "metrics.json"
    writer = imageio.get_writer(video_path, fps=4, macro_block_size=1)
    try:
        for value in (20, 40, 60):
            writer.append_data(np.full((12, 16, 3), value, dtype=np.uint8))
    finally:
        writer.close()

    result = runner.invoke(
        app,
        [
            "train",
            "evaluate-temporal-consistency",
            "--video",
            str(video_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["frame_count"] == 3
    assert "flicker_mean=" in result.output


def test_write_blind_temporal_metrics_emits_json(tmp_path) -> None:
    video_path = tmp_path / "uniform.mp4"
    output_path = tmp_path / "metrics.json"
    writer = imageio.get_writer(video_path, fps=4, macro_block_size=1)
    try:
        for value in (20, 40, 60):
            writer.append_data(np.full((12, 16, 3), value, dtype=np.uint8))
    finally:
        writer.close()

    metrics = write_blind_temporal_metrics(TemporalMetricConfig(video_path=video_path), output_path)

    assert output_path.is_file()
    assert metrics["frame_count"] == 3
