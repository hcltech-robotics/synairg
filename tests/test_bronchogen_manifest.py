from __future__ import annotations

import json

import pytest
from synairg_core.cli import app
from trainer.bronchogen_manifest import (
    BronchoGenManifestConfig,
    BronchoGenManifestError,
    build_bronchogen_manifest,
    validate_bronchogen_manifest,
    write_bronchogen_manifest,
)
from typer.testing import CliRunner

runner = CliRunner()


def test_bronchogen_manifest_groups_sequences_and_temporal_neighbors(tmp_path) -> None:
    rgb_root = tmp_path / "rgb"
    depth_root = tmp_path / "depth"
    normal_root = tmp_path / "normal"
    for root in (rgb_root, depth_root, normal_root):
        (root / "patient-a" / "seq-001").mkdir(parents=True)
    for index in range(3):
        filename = f"{index:03d}.png"
        (rgb_root / "patient-a" / "seq-001" / filename).write_bytes(b"rgb")
        (depth_root / "patient-a" / "seq-001" / filename).write_bytes(b"depth")
        (normal_root / "patient-a" / "seq-001" / filename).write_bytes(b"normal")

    manifest = build_bronchogen_manifest(
        BronchoGenManifestConfig(
            rgb_root=rgb_root,
            case_id="bronchogen-case",
            condition_roots={"depth": depth_root, "normal": normal_root},
            output_path=tmp_path / "manifest.json",
            sequence_depth=2,
            temporal_context_radius=1,
            fps=10.0,
        )
    )

    frames = manifest["frames"]
    assert isinstance(frames, list)
    assert len(frames) == 3
    assert frames[0]["sequence_id"] == "patient-a/seq-001"
    assert frames[0]["temporal_neighbor_frame_ids"] == ["bronchogen-case-patient-a-seq-001-000001"]
    assert frames[1]["temporal_neighbor_frame_ids"] == [
        "bronchogen-case-patient-a-seq-001-000000",
        "bronchogen-case-patient-a-seq-001-000002",
    ]
    assert frames[1]["timestamp_seconds"] == 0.1
    assert frames[0]["condition_paths"] == {
        "depth": "depth/patient-a/seq-001/000.png",
        "normal": "normal/patient-a/seq-001/000.png",
    }
    assert manifest["temporal"]["contract"]


def test_bronchogen_manifest_rejects_missing_required_conditions(tmp_path) -> None:
    rgb_root = tmp_path / "rgb"
    depth_root = tmp_path / "depth"
    (rgb_root / "seq").mkdir(parents=True)
    depth_root.mkdir()
    (rgb_root / "seq" / "000.png").write_bytes(b"rgb")

    with pytest.raises(BronchoGenManifestError, match="missing condition files"):
        build_bronchogen_manifest(
            BronchoGenManifestConfig(
                rgb_root=rgb_root,
                case_id="missing-condition-case",
                condition_roots={"depth": depth_root},
            )
        )


def test_bronchogen_manifest_write_and_validate_checks_files(tmp_path) -> None:
    rgb_root = tmp_path / "rgb"
    rgb_root.mkdir()
    (rgb_root / "000.jpg").write_bytes(b"rgb")
    manifest_path = tmp_path / "manifest.json"

    write_bronchogen_manifest(
        BronchoGenManifestConfig(
            rgb_root=rgb_root,
            case_id="validate-case",
            output_path=manifest_path,
            sequence_depth=0,
        )
    )

    result = validate_bronchogen_manifest(manifest_path)

    assert result == {"status": "pass", "frame_count": 1}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["frames"][0]["rgb_path"] == "rgb/000.jpg"


def test_train_cli_builds_bronchogen_manifest(tmp_path) -> None:
    rgb_root = tmp_path / "rgb"
    pps_root = tmp_path / "pps"
    for root in (rgb_root, pps_root):
        (root / "seq-a").mkdir(parents=True)
    (rgb_root / "seq-a" / "000.png").write_bytes(b"rgb")
    (pps_root / "seq-a" / "000.npy").write_bytes(b"pps")
    manifest_path = tmp_path / "bronchogen_manifest.json"

    result = runner.invoke(
        app,
        [
            "train",
            "build-bronchogen-manifest",
            "--rgb-root",
            str(rgb_root),
            "--pps-root",
            str(pps_root),
            "--case-id",
            "cli-case",
            "--output",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["case_id"] == "cli-case"
    assert payload["conditioning"]["modalities"] == ["pps"]
    assert payload["frames"][0]["condition_paths"]["pps"] == "pps/seq-a/000.npy"
