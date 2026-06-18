from __future__ import annotations

import json
import sys

import nibabel as nib
import numpy as np
import pytest
from mesh_gen.centerline import apply_bronchoscope_accessibility, prune_mask_to_accessible_centerline
from mesh_gen.errors import MeshGenError
from mesh_gen.exports import write_obj, write_ply, write_usd
from mesh_gen.mesh import count_boundary_edges, generate_airway_mesh, keep_largest_mesh_component
from mesh_gen.preview import _scale_grayscale
from mesh_gen.qa import build_quality_report
from mesh_gen.registration import validate_mesh_registration
from mesh_gen.schemas import AirwayMask, CenterlineEdge, CenterlineGraph, CenterlineNode
from mesh_gen.topology import cap_laryngeal_components, connected_components, repair_airway_mask
from synairg_core.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_centerline_graph_schema_round_trips_and_validates_edges(tmp_path) -> None:
    nodes = (
        CenterlineNode(
            node_id="n0000",
            point_mm=(0.0, 0.0, 0.0),
            radius_mm=1.0,
            branch_id="airway-main",
            voxel_index=(0.0, 0.0, 0.0),
        ),
        CenterlineNode(
            node_id="n0001",
            point_mm=(0.0, 0.0, 1.0),
            radius_mm=0.9,
            branch_id="airway-main",
            voxel_index=(0.0, 0.0, 1.0),
        ),
    )
    graph = CenterlineGraph(
        case_id="case-001",
        nodes=nodes,
        edges=(CenterlineEdge(source="n0000", target="n0001", length_mm=1.0, branch_id="airway-main"),),
        root_id="n0000",
        method="fixture",
    )
    path = tmp_path / "centerline_graph.json"
    path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")

    loaded = CenterlineGraph.read_json(path)

    assert loaded == graph
    with pytest.raises(MeshGenError, match="does-not-exist"):
        CenterlineGraph(
            case_id="case-001",
            nodes=nodes,
            edges=(CenterlineEdge(source="n0000", target="does-not-exist", length_mm=1.0, branch_id="airway-main"),),
            root_id="n0000",
            method="fixture",
        )


def test_mesh_export_writes_obj_ply_and_usd(tmp_path) -> None:
    mask = _fixture_mask("mesh-export")

    mesh = generate_airway_mesh(mask)
    obj_path = tmp_path / "airway_mesh.obj"
    ply_path = tmp_path / "airway_mesh.ply"
    usd_path = tmp_path / "airway.usd"
    write_obj(obj_path, mesh)
    write_ply(ply_path, mesh)
    write_usd(usd_path, mesh)

    assert obj_path.read_text(encoding="utf-8").startswith("# SynAirG airway mesh")
    assert ply_path.read_text(encoding="utf-8").startswith("ply\nformat ascii 1.0")
    usd_text = usd_path.read_text(encoding="utf-8")
    assert usd_text.startswith("#usda 1.0")
    assert 'def Mesh "AirwayMesh"' in usd_text
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0


def test_hollow_mesh_uncaps_only_top_airway_end() -> None:
    data = np.zeros((5, 5, 7), dtype=np.bool_)
    for k in range(1, 6):
        data[2, 2, k] = True
        data[1, 2, k] = True
        data[3, 2, k] = True
        data[2, 1, k] = True
        data[2, 3, k] = True
    mask = AirwayMask(case_id="hollow-tube", data=data, affine=np.eye(4), spacing=(1.0, 1.0, 1.0))
    graph = CenterlineGraph(
        case_id="hollow-tube",
        nodes=(
            CenterlineNode(
                node_id="n0000",
                point_mm=(2.0, 2.0, 1.0),
                radius_mm=1.25,
                branch_id="airway-main",
                voxel_index=(2.0, 2.0, 1.0),
            ),
            CenterlineNode(
                node_id="n0001",
                point_mm=(2.0, 2.0, 5.0),
                radius_mm=1.25,
                branch_id="airway-main",
                voxel_index=(2.0, 2.0, 5.0),
                distance_from_root_mm=4.0,
            ),
        ),
        edges=(CenterlineEdge(source="n0000", target="n0001", length_mm=4.0, branch_id="airway-main"),),
        root_id="n0000",
        method="fixture",
    )

    sealed = generate_airway_mesh(mask)
    hollow = generate_airway_mesh(mask, centerline_graph=graph, uncap_top=True)
    boundary_vertices = _boundary_vertex_coordinates(hollow.vertices, hollow.faces)

    assert count_boundary_edges(sealed.faces) == 0
    assert hollow.metadata["hollow"] is True
    assert hollow.metadata["top_uncapped"] is True
    assert hollow.metadata["top_opening_count"] == 1
    assert hollow.metadata["removed_cap_faces"] > 0
    assert hollow.metadata["boundary_edge_count"] > 0
    assert boundary_vertices[:, 2].min() > 3.0
    assert hollow.faces.shape[0] < sealed.faces.shape[0]


def test_quality_report_accepts_one_approved_hollow_opening() -> None:
    data = np.zeros((5, 5, 7), dtype=np.bool_)
    for k in range(1, 6):
        data[2, 2, k] = True
        data[1, 2, k] = True
        data[3, 2, k] = True
        data[2, 1, k] = True
        data[2, 3, k] = True
    mask = AirwayMask(case_id="qa-hollow-opening", data=data, affine=np.eye(4), spacing=(1.0, 1.0, 1.0))
    graph = CenterlineGraph(
        case_id="qa-hollow-opening",
        nodes=(
            CenterlineNode(
                node_id="n0000",
                point_mm=(2.0, 2.0, 1.0),
                radius_mm=1.25,
                branch_id="airway-main",
                voxel_index=(2.0, 2.0, 1.0),
            ),
            CenterlineNode(
                node_id="n0001",
                point_mm=(2.0, 2.0, 5.0),
                radius_mm=1.25,
                branch_id="airway-main",
                voxel_index=(2.0, 2.0, 5.0),
                distance_from_root_mm=4.0,
            ),
        ),
        edges=(CenterlineEdge(source="n0000", target="n0001", length_mm=4.0, branch_id="airway-main"),),
        root_id="n0000",
        method="fixture",
    )
    repaired, topology = repair_airway_mask(mask.data, spacing=mask.spacing)
    mesh = generate_airway_mesh(mask.with_data(repaired), centerline_graph=graph, uncap_top=True)

    report = build_quality_report(
        mask=mask.with_data(repaired),
        mesh=mesh,
        graph=graph,
        topology=topology,
        artifacts=(),
    )

    assert report["status"] == "pass"
    assert report["checks"]["mask_single_connected_component"] is True
    assert report["checks"]["mesh_single_connected_component"] is True
    assert report["checks"]["mesh_has_expected_boundary_openings"] is True
    assert report["mesh"]["quality"]["mesh_component_count"] == 1
    assert report["mesh"]["opening"]["boundary_loop_count"] == 1
    assert report["mesh"]["opening"]["approved_opening_count"] == 1
    assert report["mesh"]["opening"]["accidental_boundary_loop_count"] == 0
    assert report["centerline"]["accessible_node_fraction"] == 1.0
    assert report["centerline"]["terminal_node_count"] == 2


def test_proximal_opening_trim_leaves_one_approved_entry_loop() -> None:
    data = np.zeros((7, 7, 10), dtype=np.bool_)
    for k in range(1, 9):
        data[3, 3, k] = True
        data[2:5, 3, k] = True
        data[3, 2:5, k] = True
    mask = AirwayMask(case_id="proximal-trim-opening", data=data, affine=np.eye(4), spacing=(1.0, 1.0, 1.0))
    graph = CenterlineGraph(
        case_id="proximal-trim-opening",
        nodes=(
            CenterlineNode(
                node_id="n0000",
                point_mm=(3.0, 3.0, 1.0),
                radius_mm=1.25,
                branch_id="airway-main",
                voxel_index=(3.0, 3.0, 1.0),
            ),
            CenterlineNode(
                node_id="n0001",
                point_mm=(3.0, 3.0, 8.0),
                radius_mm=1.25,
                branch_id="airway-main",
                voxel_index=(3.0, 3.0, 8.0),
                distance_from_root_mm=7.0,
            ),
        ),
        edges=(CenterlineEdge(source="n0000", target="n0001", length_mm=7.0, branch_id="airway-main"),),
        root_id="n0000",
        method="fixture",
    )
    repaired, topology = repair_airway_mask(mask.data, spacing=mask.spacing)

    mesh = generate_airway_mesh(
        mask.with_data(repaired),
        centerline_graph=graph,
        uncap_top=True,
        proximal_opening_trim=True,
    )
    report = build_quality_report(
        mask=mask.with_data(repaired),
        mesh=mesh,
        graph=graph,
        topology=topology,
        artifacts=(),
    )

    assert mesh.metadata["proximal_opening_trim"] is True
    assert mesh.metadata["proximal_trim_depth_mm"] > 0
    assert report["checks"]["mesh_has_expected_boundary_openings"] is True
    assert report["mesh"]["opening"]["boundary_loop_count"] == 1
    assert report["mesh"]["opening"]["approved_opening_count"] == 1
    assert report["mesh"]["opening"]["accidental_boundary_loop_count"] == 0


def test_decimation_does_not_create_accidental_wall_holes() -> None:
    mask = _fixture_mask("decimate-closed")

    mesh = generate_airway_mesh(mask, decimate_fraction=0.15)

    assert count_boundary_edges(mesh.faces) == 0
    assert mesh.metadata["decimation_backend"] in {
        "pyvista-decimate-pro-preserve-topology",
        "skipped-no-topology-preserving-backend",
    }


def test_inward_normal_orientation_reverses_vertex_normals() -> None:
    mask = _fixture_mask("normal-orientation")

    outward = generate_airway_mesh(mask, normal_orientation="outward")
    inward = generate_airway_mesh(mask, normal_orientation="inward")
    normal_alignment = np.einsum("ij,ij->i", outward.vertex_normals, inward.vertex_normals)

    assert inward.metadata["normal_orientation"] == "inward"
    assert float(np.median(normal_alignment)) < -0.95


def test_taubin_refinement_records_backend_and_preserves_closed_topology() -> None:
    mask = _fixture_mask("taubin-refined")

    mesh = generate_airway_mesh(mask, smooth_iterations=3, smoothing_method="taubin", taubin_pass_band=0.1)

    assert mesh.metadata["smoothing_method"] == "taubin"
    assert mesh.metadata["smooth_iterations"] == 3
    assert mesh.metadata["smoothing_backend"] in {"pyvista-smooth-taubin", "skipped-no-taubin-backend"}
    assert count_boundary_edges(mesh.faces) == 0


def test_mesh_component_healing_keeps_dominant_surface_shell() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.int64,
    )

    healed_vertices, healed_faces, report = keep_largest_mesh_component(vertices, faces)

    assert healed_vertices.shape == (4, 3)
    assert healed_faces.shape == (4, 3)
    assert report["mesh_component_count_before_healing"] == 2
    assert report["removed_mesh_component_count"] == 1
    assert report["removed_mesh_component_face_count"] == 1


def _boundary_vertex_coordinates(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for edge in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = tuple(sorted(edge))
            counts[key] = counts.get(key, 0) + 1
    boundary_indices = sorted({index for edge, count in counts.items() if count == 1 for index in edge})
    assert boundary_indices
    return vertices[boundary_indices]


def test_mesh_run_cli_emits_canonical_case_outputs(tmp_path) -> None:
    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    _write_fixture_nifti(ct_path, mask_path)
    output_root = tmp_path / "meshes"

    result = runner.invoke(
        app,
        [
            "mesh",
            "run",
            "--ct",
            str(ct_path),
            "--mask",
            str(mask_path),
            "--case-id",
            "fixture-case",
            "--output",
            str(output_root),
            "--min-accessible-diameter-mm",
            "0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    case_dir = output_root / "fixture-case"
    for filename in [
        "airway_mask.nii.gz",
        "airway_mesh.obj",
        "airway_mesh.ply",
        "airway.usd",
        "centerline_graph.json",
        "branch_semantics.json",
        "radius_profile.parquet",
        "mesh_registration.json",
        "quality_report.json",
        "preview.png",
        "segmentation_overlay.png",
        "mesh_review.png",
        "viewer.html",
    ]:
        assert (case_dir / filename).is_file(), filename

    quality = json.loads((case_dir / "quality_report.json").read_text(encoding="utf-8"))
    registration = json.loads((case_dir / "mesh_registration.json").read_text(encoding="utf-8"))
    graph = json.loads((case_dir / "centerline_graph.json").read_text(encoding="utf-8"))
    branches = json.loads((case_dir / "branch_semantics.json").read_text(encoding="utf-8"))
    assert quality["status"] == "pass"
    assert quality["artifacts"]["mesh_registration.json"]["sha256"]
    assert quality["mesh"]["vertex_count"] > 0
    assert registration["schema_version"] == "1.0"
    assert registration["source_volume"]["ct.nii.gz"]["sha256"]
    assert registration["mesh_artifacts"]["airway_mesh.obj"]["path"] == "airway_mesh.obj"
    assert registration["transforms"]["mesh_world_mm_to_volume_world_mm"] == np.eye(4).tolist()
    assert registration["validation"]["round_trip"]["max_error_mm"] <= 1.0e-6
    assert {sample["type"] for sample in registration["validation"]["round_trip"]["samples"]} == {
        "mesh_vertex",
        "centerline_point",
    }
    assert validate_mesh_registration(case_dir / "mesh_registration.json")["status"] == "pass"
    assert graph["nodes"]
    assert branches["branches"][0]["branch_id"] == "airway-main"
    assert (case_dir / "preview.png").read_bytes().startswith(b"\x89PNG")
    assert (case_dir / "segmentation_overlay.png").read_bytes().startswith(b"\x89PNG")
    assert (case_dir / "mesh_review.png").read_bytes().startswith(b"\x89PNG")
    viewer = (case_dir / "viewer.html").read_text(encoding="utf-8")
    assert "<canvas" in viewer
    assert "DATA =" in viewer


def test_mesh_registration_uses_dicom_patient_transform_when_available(tmp_path) -> None:
    volume_dir = tmp_path / "volume-case"
    volume_dir.mkdir()
    ct_path = volume_dir / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    affine = np.asarray(
        [
            [0.0, 1.1, 0.0, 10.0],
            [0.8, 0.0, 0.0, -20.0],
            [0.0, 0.0, 1.5, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _write_fixture_nifti(ct_path, mask_path, affine=affine)
    (volume_dir / "volume_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "case_id": "volume-case",
                "source_type": "local_dicom",
                "volume": {
                    "dicom_reference": {
                        "coordinate_frame": "DICOM patient LPS",
                        "privacy": {"patient_identifiers": "omitted", "dicom_uid_policy": "sha256"},
                        "image_orientation_patient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        "pixel_spacing_mm": [0.8, 1.1],
                        "slice_spacing_mm": 1.5,
                        "slice_spacing_derivation": "image_position_projection",
                        "voxel_index_to_dicom_patient_lps_mm": affine.tolist(),
                        "source_file_hashes": [],
                        "instance_order": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (volume_dir / "source_provenance.json").write_text(
        json.dumps({"schema_version": "1.0", "source": {"type": "local_dicom"}}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "mesh",
            "run",
            "--ct",
            str(ct_path),
            "--mask",
            str(mask_path),
            "--case-id",
            "dicom-mesh-case",
            "--output",
            str(tmp_path / "meshes"),
            "--min-accessible-diameter-mm",
            "0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    registration_path = tmp_path / "meshes" / "dicom-mesh-case" / "mesh_registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    assert registration["source_volume"]["source_type"] == "local_dicom"
    assert registration["source_volume"]["volume_metadata.json"]["sha256"]
    assert registration["dicom_reference"]["privacy"]["patient_identifiers"] == "omitted"
    assert np.allclose(registration["transforms"]["voxel_index_to_dicom_patient_lps_mm"], affine)
    assert np.allclose(registration["transforms"]["volume_world_mm_to_dicom_patient_lps_mm"], np.eye(4), atol=1.0e-6)
    assert "PatientName" not in registration_path.read_text(encoding="utf-8")


def test_mesh_run_cli_rejects_mismatched_ct_and_mask_shapes(tmp_path) -> None:
    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(ct_path))
    nib.save(nib.Nifti1Image(np.ones((5, 4, 4), dtype=np.uint8), np.eye(4)), str(mask_path))

    result = runner.invoke(
        app,
        ["mesh", "run", "--ct", str(ct_path), "--mask", str(mask_path), "--output", str(tmp_path / "meshes")],
    )

    assert result.exit_code == 1
    assert "does not match mask shape" in result.output


def test_mesh_run_cli_rejects_mismatched_ct_and_mask_affines(tmp_path) -> None:
    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(ct_path))
    mask_affine = np.diag([2.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), mask_affine), str(mask_path))

    result = runner.invoke(
        app,
        ["mesh", "run", "--ct", str(ct_path), "--mask", str(mask_path), "--output", str(tmp_path / "meshes")],
    )

    assert result.exit_code == 1
    assert "affine does not match" in result.output


def test_preview_scaler_downsamples_full_slice_instead_of_cropping() -> None:
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)

    scaled = _scale_grayscale(image, size=2)

    assert scaled.tolist() == [[0, 3], [12, 15]]


def test_mesh_run_cli_rejects_unknown_bronchoscope_profile(tmp_path) -> None:
    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    _write_fixture_nifti(ct_path, mask_path)

    result = runner.invoke(
        app,
        [
            "mesh",
            "run",
            "--ct",
            str(ct_path),
            "--mask",
            str(mask_path),
            "--output",
            str(tmp_path / "meshes"),
            "--bronchoscope-profile",
            "not-a-scope",
        ],
    )

    assert result.exit_code == 1
    assert "unknown bronchoscope profile" in result.output


def test_mesh_run_cli_bronchoscope_profile_records_diameter_and_depth(tmp_path) -> None:
    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    _write_fixture_nifti(ct_path, mask_path)
    output_root = tmp_path / "meshes"

    result = runner.invoke(
        app,
        [
            "mesh",
            "run",
            "--ct",
            str(ct_path),
            "--mask",
            str(mask_path),
            "--case-id",
            "profile-case",
            "--output",
            str(output_root),
            "--bronchoscope-profile",
            "ultrathin-mp190f",
            "--keep-inaccessible-diameter",
        ],
    )

    assert result.exit_code == 0, result.output
    quality = json.loads((output_root / "profile-case" / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["segmentation"]["min_accessible_diameter_mm"] == 3.0
    assert quality["segmentation"]["max_accessible_depth_mm"] == 600.0
    assert quality["segmentation"]["bronchoscope_profile"].startswith("ultrathin-mp190f:")


def test_mesh_run_cli_high_detail_refinement_records_mesh_quality(tmp_path) -> None:
    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    _write_fixture_nifti(ct_path, mask_path)
    output_root = tmp_path / "meshes"

    result = runner.invoke(
        app,
        [
            "mesh",
            "run",
            "--ct",
            str(ct_path),
            "--mask",
            str(mask_path),
            "--case-id",
            "refined-case",
            "--output",
            str(output_root),
            "--min-accessible-diameter-mm",
            "0.5",
            "--refinement-preset",
            "high-detail",
        ],
    )

    assert result.exit_code == 0, result.output
    quality = json.loads((output_root / "refined-case" / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["status"] == "pass"
    assert quality["refinement"]["preset"] == "high-detail"
    assert quality["refinement"]["smoothing_method"] == "taubin"
    assert quality["refinement"]["smooth_iterations"] == 50
    assert quality["refinement"]["normal_orientation"] == "inward"
    assert quality["mesh"]["quality"]["degenerate_face_count"] == 0
    assert quality["mesh"]["quality"]["non_manifold_edge_count"] == 0
    assert quality["mesh"]["surface_fidelity"]["checked"] is True


def test_mesh_run_cli_uses_external_command_backend(tmp_path) -> None:
    ct_path = tmp_path / "ct.nii.gz"
    reference_mask_path = tmp_path / "reference_mask.nii.gz"
    _write_fixture_nifti(ct_path, reference_mask_path)
    script_path = tmp_path / "fake_segmenter.py"
    script_path.write_text(
        "\n".join(
            [
                "import argparse",
                "import nibabel as nib",
                "import numpy as np",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--ct')",
                "parser.add_argument('--output')",
                "args = parser.parse_args()",
                "image = nib.load(args.ct)",
                "mask = np.zeros(image.shape[:3], dtype=np.uint8)",
                "mask[2:4, 2:4, 1:5] = 1",
                "out = nib.Nifti1Image(mask, image.affine, image.header)",
                "out.header.set_data_dtype(np.uint8)",
                "nib.save(out, args.output)",
            ]
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "meshes"

    result = runner.invoke(
        app,
        [
            "mesh",
            "run",
            "--ct",
            str(ct_path),
            "--backend",
            "external-command",
            "--segmenter-command",
            f"{sys.executable} {script_path} --ct {{ct}} --output {{mask}}",
            "--case-id",
            "external-case",
            "--output",
            str(output_root),
            "--min-accessible-diameter-mm",
            "0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    quality = json.loads((output_root / "external-case" / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["segmentation"]["backend"] == "external-command"
    assert quality["segmentation"]["metadata"]["command"][0] == sys.executable


def test_accessible_diameter_pruning_keeps_root_reachable_voxels() -> None:
    data = np.zeros((5, 3, 3), dtype=np.bool_)
    data[1:4, 1, 1] = True
    mask = AirwayMask(case_id="diameter-prune", data=data, affine=np.eye(4), spacing=(1.0, 1.0, 1.0))
    graph = CenterlineGraph(
        case_id="diameter-prune",
        nodes=(
            CenterlineNode(
                node_id="n0000",
                point_mm=(1.0, 1.0, 1.0),
                radius_mm=2.0,
                branch_id="airway-main",
                voxel_index=(1.0, 1.0, 1.0),
                accessible=True,
            ),
            CenterlineNode(
                node_id="n0001",
                point_mm=(2.0, 1.0, 1.0),
                radius_mm=1.0,
                branch_id="airway-main",
                voxel_index=(2.0, 1.0, 1.0),
                accessible=False,
            ),
            CenterlineNode(
                node_id="n0002",
                point_mm=(3.0, 1.0, 1.0),
                radius_mm=2.0,
                branch_id="airway-main",
                voxel_index=(3.0, 1.0, 1.0),
                accessible=False,
            ),
        ),
        edges=(
            CenterlineEdge(source="n0000", target="n0001", length_mm=1.0, branch_id="airway-main", accessible=False),
            CenterlineEdge(source="n0001", target="n0002", length_mm=1.0, branch_id="airway-main", accessible=False),
        ),
        root_id="n0000",
        method="fixture",
    )

    filtered, report = prune_mask_to_accessible_centerline(mask, graph)

    assert report["checked"] is True
    assert report["removed_voxels"] == 2
    assert int(np.count_nonzero(filtered)) == 1
    assert filtered[1, 1, 1]
    assert not filtered[2, 1, 1]
    assert not filtered[3, 1, 1]


def test_accessible_depth_pruning_removes_voxels_beyond_working_length() -> None:
    data = np.zeros((5, 3, 3), dtype=np.bool_)
    data[1:4, 1, 1] = True
    mask = AirwayMask(case_id="depth-prune", data=data, affine=np.eye(4), spacing=(1.0, 1.0, 1.0))
    graph = CenterlineGraph(
        case_id="depth-prune",
        nodes=(
            CenterlineNode(
                node_id="n0000",
                point_mm=(1.0, 1.0, 1.0),
                radius_mm=2.0,
                branch_id="airway-main",
                voxel_index=(1.0, 1.0, 1.0),
                distance_from_root_mm=0.0,
            ),
            CenterlineNode(
                node_id="n0001",
                point_mm=(2.0, 1.0, 1.0),
                radius_mm=2.0,
                branch_id="airway-main",
                voxel_index=(2.0, 1.0, 1.0),
                distance_from_root_mm=1.0,
            ),
            CenterlineNode(
                node_id="n0002",
                point_mm=(3.0, 1.0, 1.0),
                radius_mm=2.0,
                branch_id="airway-main",
                voxel_index=(3.0, 1.0, 1.0),
                distance_from_root_mm=2.5,
            ),
        ),
        edges=(
            CenterlineEdge(source="n0000", target="n0001", length_mm=1.0, branch_id="airway-main"),
            CenterlineEdge(source="n0001", target="n0002", length_mm=1.5, branch_id="airway-main"),
        ),
        root_id="n0000",
        method="fixture",
    )
    reachable_graph = apply_bronchoscope_accessibility(
        graph,
        min_accessible_diameter_mm=1.0,
        max_accessible_depth_mm=1.5,
    )

    filtered, report = prune_mask_to_accessible_centerline(mask, reachable_graph)

    assert [node.accessible for node in reachable_graph.nodes] == [True, True, False]
    assert [edge.accessible for edge in reachable_graph.edges] == [True, False]
    assert report["removed_voxels"] == 1
    assert int(np.count_nonzero(filtered)) == 2
    assert filtered[1, 1, 1]
    assert filtered[2, 1, 1]
    assert not filtered[3, 1, 1]


def test_topology_repair_bridges_near_sizeable_airway_fragments() -> None:
    mask = np.zeros((16, 8, 8), dtype=np.bool_)
    mask[2:6, 3, 3] = True
    mask[9:13, 3, 3] = True
    repaired, report = repair_airway_mask(
        mask,
        spacing=(1.0, 1.0, 1.0),
        min_component_voxels=3,
        max_bridge_distance_mm=6.0,
    )

    assert report.component_count == 2
    assert report.kept_component_count == 2
    assert report.bridged_component_count == 1
    assert report.bridge_voxels > 0
    assert report.final_component_count == 1
    assert report.unbridged_component_count == 0
    assert repaired[7, 3, 3]
    assert int(np.count_nonzero(repaired)) > int(np.count_nonzero(mask))
    _, final_component_count = connected_components(repaired)
    assert final_component_count == 1


def test_laryngeal_capping_removes_superior_disconnected_remnant() -> None:
    mask = np.zeros((8, 8, 10), dtype=np.bool_)
    mask[3:5, 3:5, 1:5] = True
    mask[3:5, 3:5, 8:10] = True

    capped, report = cap_laryngeal_components(
        mask,
        spacing=(1.0, 1.0, 1.0),
        superior_axis=2,
    )

    assert report.checked is True
    assert report.removed_component_count == 1
    assert report.removed_voxels == 8
    assert report.cap_index == 4
    assert not np.any(capped[:, :, 8:10])
    assert np.any(capped[:, :, 1:5])


def test_mesh_run_cli_laryngeal_capping_starts_airway_below_broken_larynx(tmp_path) -> None:
    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    ct = np.zeros((8, 8, 10), dtype=np.float32)
    mask = np.zeros((8, 8, 10), dtype=np.uint8)
    mask[3:5, 3:5, 1:5] = 1
    mask[3:5, 3:5, 8:10] = 1
    affine = np.eye(4, dtype=np.float64)
    nib.save(nib.Nifti1Image(ct, affine), str(ct_path))
    nib.save(nib.Nifti1Image(mask, affine), str(mask_path))
    output_root = tmp_path / "meshes"

    result = runner.invoke(
        app,
        [
            "mesh",
            "run",
            "--ct",
            str(ct_path),
            "--mask",
            str(mask_path),
            "--case-id",
            "laryngeal-gap-case",
            "--output",
            str(output_root),
            "--min-accessible-diameter-mm",
            "0.5",
            "--max-bridge-distance-mm",
            "8.0",
        ],
    )

    assert result.exit_code == 0, result.output
    quality = json.loads((output_root / "laryngeal-gap-case" / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["status"] == "pass"
    assert quality["segmentation"]["laryngeal_capping"]["removed_component_count"] == 1
    assert quality["segmentation"]["laryngeal_capping"]["cap_index"] == 4
    assert quality["mesh"]["bounds_max_mm"][2] < 6.0
    assert quality["mesh"]["opening"]["approved_opening_count"] == 1


def _fixture_mask(case_id: str) -> AirwayMask:
    data = np.zeros((5, 5, 5), dtype=np.bool_)
    data[2, 2, 1:4] = True
    data[2, 1:4, 2] = True
    data[1:4, 2, 2] = True
    return AirwayMask(case_id=case_id, data=data, affine=np.eye(4, dtype=np.float64), spacing=(1.0, 1.0, 1.0))


def _write_fixture_nifti(ct_path, mask_path, *, affine=None) -> None:
    ct = np.zeros((6, 6, 6), dtype=np.float32)
    mask = np.zeros((6, 6, 6), dtype=np.uint8)
    mask[2:4, 2:4, 1:5] = 1
    if affine is None:
        affine = np.diag([0.7, 0.7, 1.25, 1.0])
    affine = np.asarray(affine, dtype=np.float64)
    spacing = tuple(float(np.linalg.norm(affine[:3, index])) for index in range(3))
    ct_image = nib.Nifti1Image(ct, affine)
    mask_image = nib.Nifti1Image(mask, affine)
    ct_image.header.set_zooms(spacing)
    mask_image.header.set_zooms(spacing)
    nib.save(ct_image, str(ct_path))
    nib.save(mask_image, str(mask_path))
