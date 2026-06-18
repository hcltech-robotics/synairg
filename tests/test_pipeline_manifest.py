from __future__ import annotations

import json

import pytest
from synairg_core.cli import app
from synairg_core.pipeline_manifest import build_pipeline_plan, read_pipeline_manifest
from synairg_core.volume_manifest import ManifestValidationError
from typer.testing import CliRunner


def test_pipeline_manifest_builds_mask_to_omniverse_plan(tmp_path) -> None:
    manifest_path = tmp_path / "pipeline.yml"
    manifest_path.write_text(
        """
schema_version: "1.0"
case_id: case-a
output_root: outputs
volume:
  kind: local_ct
  ct_path: inputs/case-a.nii.gz
mask_source:
  kind: monai_segmenter
  command: "python -m monai.bundle run airway --ct {ct} --mask {mask}"
mesh:
  case_id: case-a-render
render:
  condition_modalities: [rgb, depth, normal, pps, mask]
  material_variant: healthy
  final_renderer: omniverse_rtx
rl:
  task: intraluminal-navigation
""".lstrip(),
        encoding="utf-8",
    )

    manifest = read_pipeline_manifest(manifest_path)
    plan = build_pipeline_plan(manifest)

    assert plan["case_id"] == "case-a"
    assert plan["contracts"]["mask_required_for_meshing"] is True
    assert plan["contracts"]["condition_modalities"] == ["rgb", "depth", "normal", "pps", "mask"]
    steps = plan["steps"]
    assert [step["stage"] for step in steps] == ["volume", "mesh", "pose", "video", "pbr", "rtx"]
    mesh_step = steps[1]
    assert "--backend external-command" in mesh_step["shell"]
    assert "--segmenter-command" in mesh_step["shell"]
    video_step = steps[3]
    assert "--export-conditions outputs/videos/case-a-conditions" in video_step["shell"]
    rtx_step = steps[-1]
    assert "tools/render_omniverse_mdl_bundle.py" in rtx_step["shell"]


def test_pipeline_manifest_requires_all_render_conditions(tmp_path) -> None:
    manifest_path = tmp_path / "pipeline.yml"
    manifest_path.write_text(
        """
case_id: case-a
volume:
  kind: local_ct
  ct_path: inputs/case-a.nii.gz
mask_source:
  kind: precomputed
  mask_path: inputs/case-a-mask.nii.gz
render:
  condition_modalities: [rgb, depth, normal, mask]
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="rgb, depth, normal, pps, and mask"):
        read_pipeline_manifest(manifest_path)


def test_pipeline_cli_outputs_plan(tmp_path) -> None:
    manifest_path = tmp_path / "pipeline.json"
    manifest_path.write_text(
        json.dumps(
            {
                "case_id": "case-a",
                "volume": {"kind": "mask_conditioned_maisi", "config": "maisi.json"},
                "mask_source": {"kind": "mask_conditioned_maisi", "mask_path": "airway-mask.nii.gz"},
                "render": {"condition_modalities": ["rgb", "depth", "normal", "pps", "mask"]},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["pipeline", "plan", "--manifest", str(manifest_path)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["steps"][0]["stage"] == "volume"
    assert payload["steps"][1]["stage"] == "mesh"
    assert "airway-mask.nii.gz" in payload["steps"][1]["shell"]
