from __future__ import annotations

import json

import numpy as np
from mesh_gen.schemas import CenterlineEdge, CenterlineGraph, CenterlineNode
from pose_gen.paths import BronchoscopeProfile, generate_scope_paths
from synairg_core.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_scope_path_generation_records_intrinsics_and_mechanical_limits(tmp_path) -> None:
    graph_path = tmp_path / "centerline_graph.json"
    _write_pose_fixture_graph(graph_path)
    output_path = tmp_path / "scope_paths.json"

    payload = generate_scope_paths(
        centerline_graph_path=graph_path,
        output_path=output_path,
        path_count=2,
        frustum_every_frames=5,
        profile=BronchoscopeProfile(),
    )

    assert output_path.is_file()
    assert payload["bronchoscope"]["intrinsics"]["fx_px"] == 400.00000000000006
    assert payload["sampling"]["mechanics"] == "single-plane distal bending plus insertion-tube axial rotation"
    minimum_lumen_diameter = payload["bronchoscope"]["minimum_lumen_diameter_mm"]
    assert len(payload["paths"]) == 2
    for path in payload["paths"]:
        frames = path["frames"]
        assert len(frames) > 10
        assert path["target_node_id"] != "n0004"
        assert path["minimum_lumen_diameter_mm"] > payload["bronchoscope"]["distal_outer_diameter_mm"]
        assert path["minimum_lumen_diameter_mm"] >= minimum_lumen_diameter
        assert frames[0]["show_frustum"] is True
        assert all(frame["frame_index"] % 5 == 0 for frame in frames if frame["show_frustum"])
        assert all(4.0 <= frame["speed_mm_s"] <= 18.0 for frame in frames)
        assert all(-120.0 <= frame["axial_rotation_deg"] <= 120.0 for frame in frames)
        assert all(-130.0 <= frame["bend_deg"] <= 210.0 for frame in frames)
        assert all(frame["centerline_anchor_offset_mm"] <= 0.001 for frame in frames)
        assert all(frame["radius_mm"] * 2.0 >= minimum_lumen_diameter for frame in frames)


def test_scope_path_generation_can_emit_multi_target_procedure(tmp_path) -> None:
    graph_path = tmp_path / "centerline_graph.json"
    _write_pose_fixture_graph(graph_path)
    output_path = tmp_path / "scope_paths.json"

    payload = generate_scope_paths(
        centerline_graph_path=graph_path,
        output_path=output_path,
        path_count=2,
        frustum_every_frames=5,
        profile=BronchoscopeProfile(),
        include_procedure=True,
    )

    procedure = payload["paths"][-1]
    frames = procedure["frames"]
    phases = {frame["motion_phase"] for frame in frames}
    assert procedure["path_id"] == "scope-procedure-01"
    assert procedure["path_type"] == "multi-target-procedure"
    assert len(procedure["procedure_targets"]) == 2
    assert all(target["target_node_id"] != "n0004" for target in procedure["procedure_targets"])
    assert all(
        target["minimum_lumen_diameter_mm"] >= payload["bronchoscope"]["minimum_lumen_diameter_mm"]
        for target in procedure["procedure_targets"]
    )
    assert {"insertion", "retraction", "branch_pause", "scope_rotate", "bend_curl", "final_retraction"} <= phases
    assert "redirect" not in phases
    assert max(abs(frame["axial_rotation_deg"]) for frame in frames) <= 0.5
    assert all(
        frame["speed_mm_s"] == 0.0
        for frame in frames
        if frame["motion_phase"] in {"branch_pause", "scope_rotate", "bend_curl"}
    )
    assert _retraction_frames_look_distal(frames)
    assert max(frame["insertion_depth_mm"] for frame in frames) > frames[-1]["insertion_depth_mm"]
    assert all(frame["centerline_anchor_offset_mm"] <= 0.001 for frame in frames)


def test_pose_run_cli_writes_scope_paths_without_review_render(tmp_path) -> None:
    mesh_case = tmp_path / "mesh-case"
    mesh_case.mkdir()
    _write_pose_fixture_graph(mesh_case / "centerline_graph.json")
    output_root = tmp_path / "poses"

    result = runner.invoke(
        app,
        [
            "pose",
            "run",
            "--mesh-case",
            str(mesh_case),
            "--output",
            str(output_root),
            "--path-count",
            "2",
            "--frustum-every-frames",
            "5",
            "--include-procedure",
            "--no-render-review",
        ],
    )

    assert result.exit_code == 0, result.output
    paths_json = output_root / "mesh-case" / "scope_paths.json"
    payload = json.loads(paths_json.read_text(encoding="utf-8"))
    assert payload["mesh_case_id"] == "pose-fixture"
    assert len(payload["paths"]) == 3
    assert payload["paths"][-1]["path_id"] == "scope-procedure-01"
    assert not (output_root / "mesh-case" / "scope_paths_review.png").exists()


def _retraction_frames_look_distal(frames) -> bool:
    for previous, current in zip(frames[:-1], frames[1:], strict=False):
        if current["motion_phase"] not in {"retraction", "final_retraction"}:
            continue
        if previous["motion_phase"] != current["motion_phase"]:
            continue
        displacement = np.asarray(current["position_mm"], dtype=np.float64) - np.asarray(
            previous["position_mm"],
            dtype=np.float64,
        )
        if float(np.linalg.norm(displacement)) <= 1.0e-6:
            continue
        forward = np.asarray(current["forward"], dtype=np.float64)
        if float(np.dot(forward, displacement)) >= 0.0:
            return False
    return True


def _write_pose_fixture_graph(path) -> None:
    graph = CenterlineGraph(
        case_id="pose-fixture",
        nodes=(
            CenterlineNode(
                node_id="n0000",
                point_mm=(0.0, 0.0, 0.0),
                radius_mm=4.0,
                branch_id="trachea",
                voxel_index=(0.0, 0.0, 0.0),
            ),
            CenterlineNode(
                node_id="n0001",
                point_mm=(0.0, 0.0, -40.0),
                radius_mm=3.5,
                branch_id="trachea",
                voxel_index=(0.0, 0.0, 1.0),
                distance_from_root_mm=40.0,
            ),
            CenterlineNode(
                node_id="n0002",
                point_mm=(25.0, 0.0, -75.0),
                radius_mm=2.4,
                branch_id="right-main",
                voxel_index=(1.0, 0.0, 2.0),
                generation=1,
                distance_from_root_mm=83.0,
            ),
            CenterlineNode(
                node_id="n0003",
                point_mm=(-25.0, 0.0, -70.0),
                radius_mm=2.2,
                branch_id="left-main",
                voxel_index=(-1.0, 0.0, 2.0),
                generation=1,
                distance_from_root_mm=80.0,
            ),
            CenterlineNode(
                node_id="n0004",
                point_mm=(45.0, 10.0, -110.0),
                radius_mm=1.5,
                branch_id="right-upper",
                voxel_index=(2.0, 1.0, 3.0),
                generation=2,
                distance_from_root_mm=124.0,
            ),
        ),
        edges=(
            CenterlineEdge(source="n0000", target="n0001", length_mm=40.0, branch_id="trachea"),
            CenterlineEdge(source="n0001", target="n0002", length_mm=43.0, branch_id="right-main"),
            CenterlineEdge(source="n0001", target="n0003", length_mm=40.0, branch_id="left-main"),
            CenterlineEdge(source="n0002", target="n0004", length_mm=41.0, branch_id="right-upper"),
        ),
        root_id="n0000",
        method="fixture",
    )
    path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
