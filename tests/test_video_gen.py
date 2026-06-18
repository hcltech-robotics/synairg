from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from video_gen.scope import (
    PBR_MATERIAL_REGION_BLEND_SOFTNESS,
    PBR_MATERIAL_REGION_TINT_STRENGTH,
    PBR_RENDER_ATLAS_ALBEDO_POST_FILL_SIGMA_PX,
    PBR_RENDER_ATLAS_ALBEDO_SIGMA_PX,
    PBR_RENDER_ATLAS_SCALAR_POST_FILL_SIGMA_PX,
    PBR_RENDER_ATLAS_SCALAR_SIGMA_PX,
    PBR_TEXTURE_ATLAS_CHANNELS,
    _apply_scope_diagnostic_mask,
    _apply_scope_view_mask,
    _CenterlineMaterialNodes,
    _compose_diagnostic_frame,
    _condition_frame_filename,
    _condition_sequence_id,
    _ct_plane_slice,
    _CtContext,
    _depth_to_condition_png,
    _depth_to_rgb,
    _face_varying_texture_coordinates,
    _hit_mask_from_depth_buffer,
    _load_pbr_material_map,
    _mask_to_condition_png,
    _metric_depth_from_buffer,
    _mucosal_pbr_material_atlas,
    _mucosal_texture_rgb,
    _pbr_cell_region_ids,
    _pbr_region_actor_specs,
    _pbr_texture_atlas_images,
    _pps_rgb,
    _render_ct_orthogonal_panel,
    _select_path,
    _surface_deposit_mesh,
    _surface_deposit_specs_from_material_map,
    _view_normal_rgb,
    export_pbr_mdl_bundle,
    export_pbr_texture_atlas,
)


def test_scope_view_mask_keeps_center_and_blacks_out_corners() -> None:
    image = np.full((80, 100, 3), 200, dtype=np.uint8)

    masked = _apply_scope_view_mask(image)

    assert masked.shape == image.shape
    assert masked[40, 50].mean() > 150
    assert masked[0, 0].tolist() == [0, 0, 0]
    assert masked[-1, -1].tolist() == [0, 0, 0]


def test_select_path_defaults_to_first_path_and_validates_requested_id() -> None:
    payload = {
        "paths": [
            {"path_id": "scope-path-01", "frames": []},
            {"path_id": "scope-path-02", "frames": []},
        ]
    }

    assert _select_path(payload, None)["path_id"] == "scope-path-01"
    assert _select_path(payload, "scope-path-02")["path_id"] == "scope-path-02"
    with pytest.raises(ValueError, match="was not found"):
        _select_path(payload, "missing")


def test_diagnostic_mask_and_depth_colorizer_preserve_metric_panel_shape() -> None:
    image = np.full((80, 100, 3), 180, dtype=np.uint8)
    masked = _apply_scope_diagnostic_mask(image)

    assert masked.shape == image.shape
    assert masked[40, 50].mean() > 150
    assert masked[0, 0].tolist() == [0, 0, 0]

    raw_depth = np.linspace(-1.0, -95.0, 80 * 100, dtype=np.float32).reshape(80, 100)
    raw_depth[40, 50] = np.nan
    depth = _metric_depth_from_buffer(raw_depth, max_depth_mm=95.0)

    assert np.isfinite(depth).all()
    assert depth[40, 50] == 95.0

    depth_rgb = _depth_to_rgb(depth, max_depth_mm=95.0)

    assert depth_rgb.shape == image.shape
    assert depth_rgb.dtype == np.uint8
    assert depth_rgb[40, 50].mean() > 0
    assert not np.array_equal(depth_rgb[40, 50], depth_rgb[40, 70])


def test_condition_export_helpers_encode_training_maps() -> None:
    depth_mm = np.asarray([[0.0, 47.5, 95.0, 120.0]], dtype=np.float32)

    depth_png = _depth_to_condition_png(depth_mm, max_depth_mm=95.0)

    assert depth_png.dtype == np.uint16
    assert depth_png.tolist() == [[0, 32768, 65535, 65535]]
    assert _mask_to_condition_png(np.asarray([[False, True]], dtype=bool)).tolist() == [[0, 255]]
    assert _condition_sequence_id("scope/procedure 01") == "scope-procedure-01"
    assert _condition_frame_filename(12) == "000012.png"


def test_hit_mask_uses_valid_depth_inside_scope_circle() -> None:
    depth = np.full((20, 20), np.nan, dtype=np.float32)
    depth[10, 10] = -12.0
    depth[0, 0] = -8.0
    depth[10, 12] = 0.0

    mask = _hit_mask_from_depth_buffer(depth)

    assert mask[10, 10]
    assert not mask[0, 0]
    assert not mask[10, 12]


def test_ct_orthogonal_panel_follows_frame_position(tmp_path) -> None:
    grid = np.indices((16, 18, 20)).sum(axis=0).astype(np.float32)
    context = _CtContext(
        volume=grid,
        mesh_to_voxel=np.eye(4, dtype=np.float64),
        spacing_mm=(1.0, 1.0, 1.0),
        source_ct_path=tmp_path / "ct.nii.gz",
        window_hu=(0.0, 54.0),
    )
    frame = {"centerline_position_mm": [8.0, 9.0, 10.0]}

    panel = _render_ct_orthogonal_panel(context, frame, width_px=192, height_px=144)

    assert panel.shape == (144, 192, 3)
    assert panel.dtype == np.uint8
    assert panel.mean() > 0
    assert panel[79, 48].sum() > 0


def test_coronal_and_sagittal_ct_slices_are_superior_side_up(tmp_path) -> None:
    volume = np.broadcast_to(np.arange(6, dtype=np.float32), (4, 5, 6))
    context = _CtContext(
        volume=volume,
        mesh_to_voxel=np.eye(4, dtype=np.float64),
        spacing_mm=(1.0, 1.0, 1.0),
        source_ct_path=tmp_path / "ct.nii.gz",
        window_hu=(0.0, 5.0),
    )
    voxel = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    coronal, coronal_cross, _, _ = _ct_plane_slice(context, voxel, plane="coronal")
    sagittal, sagittal_cross, _, _ = _ct_plane_slice(context, voxel, plane="sagittal")

    assert coronal[0, 0] == 5.0
    assert coronal[-1, 0] == 0.0
    assert sagittal[0, 0] == 5.0
    assert sagittal[-1, 0] == 0.0
    assert coronal_cross[1] == 2.0
    assert sagittal_cross[1] == 2.0


def test_view_normal_rgb_uses_camera_relative_axes() -> None:
    frame = {
        "forward": [0.0, 0.0, 1.0],
        "up": [0.0, 1.0, 0.0],
    }
    normals = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    rgb = _view_normal_rgb(normals, frame)

    assert rgb.shape == (2, 3)
    assert rgb[0, 2] > 250
    assert 120 <= rgb[0, 0] <= 135
    assert rgb[1, 1] > 250


def test_pps_rgb_is_camera_light_dependent() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 60.0],
            [0.0, 20.0, 10.0],
        ],
        dtype=np.float64,
    )
    normals = np.asarray(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    frame = {
        "position_mm": [0.0, 0.0, 0.0],
        "forward": [0.0, 0.0, 1.0],
        "radius_mm": 2.0,
    }

    rgb = _pps_rgb(points, normals, frame, depth_of_field_mm=95.0)

    assert rgb.shape == (3, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0].mean() > rgb[1].mean()
    assert rgb[0].mean() > rgb[2].mean()


def test_mucosal_texture_rgb_is_deterministic_and_varied() -> None:
    points = np.asarray(
        [
            [-8.0, -4.0, 0.0],
            [-4.0, 2.0, 6.0],
            [0.0, 0.0, 12.0],
            [3.0, -2.5, 18.0],
            [7.5, 4.0, 24.0],
        ],
        dtype=np.float64,
    )

    first = _mucosal_texture_rgb(points)
    second = _mucosal_texture_rgb(points.copy())

    assert first.shape == (5, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert len(np.unique(first, axis=0)) > 1
    assert first[:, 0].mean() > first[:, 2].mean()


def test_mucosal_pbr_material_atlas_uses_anatomy_and_variant() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0],
            [3.0, 0.5, 5.0],
            [5.0, 2.0, 8.0],
            [7.0, 3.0, 11.0],
            [9.0, 3.5, 14.0],
        ],
        dtype=np.float64,
    )
    centerline = _CenterlineMaterialNodes(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 4.0],
                [6.0, 2.0, 9.0],
                [9.0, 3.5, 14.0],
            ],
            dtype=np.float64,
        ),
        generation=np.asarray([0, 1, 3, 5], dtype=np.float32),
        distance_from_root_mm=np.asarray([0.0, 15.0, 45.0, 85.0], dtype=np.float32),
        radius_mm=np.asarray([7.5, 5.2, 2.5, 1.2], dtype=np.float32),
    )

    healthy = _mucosal_pbr_material_atlas(points, centerline_nodes=centerline, material_variant="healthy")
    inflamed = _mucosal_pbr_material_atlas(points, centerline_nodes=centerline, material_variant="inflamed")
    smoker = _mucosal_pbr_material_atlas(points, centerline_nodes=centerline, material_variant="smoker")
    edematous = _mucosal_pbr_material_atlas(points, centerline_nodes=centerline, material_variant="edematous")

    assert healthy.albedo_rgb.shape == (6, 3)
    assert healthy.albedo_rgb.dtype == np.uint8
    assert healthy.roughness.shape == (6,)
    assert healthy.specular.shape == (6,)
    assert healthy.mucus.shape == (6,)
    assert healthy.mucus_shininess.shape == (6,)
    assert healthy.erythema.shape == (6,)
    assert healthy.petechiae.shape == (6,)
    assert healthy.stain.shape == (6,)
    assert healthy.bump_height.shape == (6,)
    assert healthy.displacement_height.shape == (6,)
    assert set(np.unique(healthy.region_id)).issuperset({0, 1, 2, 3})
    assert np.all((healthy.roughness >= 0.0) & (healthy.roughness <= 1.0))
    assert np.all((healthy.specular >= 0.0) & (healthy.specular <= 1.0))
    assert np.all((healthy.mucus >= 0.0) & (healthy.mucus <= 1.0))
    assert np.all((healthy.mucus_shininess >= 0.0) & (healthy.mucus_shininess <= 1.0))
    assert np.all((healthy.stain >= 0.0) & (healthy.stain <= 1.0))
    assert np.all((healthy.bump_height >= 0.0) & (healthy.bump_height <= 1.0))
    assert np.all((healthy.displacement_height >= 0.0) & (healthy.displacement_height <= 1.0))
    assert inflamed.albedo_rgb[:, 0].mean() > healthy.albedo_rgb[:, 0].mean()
    assert inflamed.wetness.mean() > healthy.wetness.mean()
    assert inflamed.erythema.mean() > healthy.erythema.mean()
    assert inflamed.petechiae.mean() > healthy.petechiae.mean()
    assert smoker.stain.mean() > healthy.stain.mean()
    assert edematous.mucus.mean() > healthy.mucus.mean()
    assert edematous.mucus_shininess.mean() > healthy.mucus_shininess.mean()
    assert edematous.displacement_height.mean() > healthy.displacement_height.mean()
    assert healthy.summary["region_model"] == "nearest centerline generation/radius/distance"
    assert healthy.summary["region_edge_model"] == "soft normalized anatomy-region weights"
    assert healthy.summary["region_blend_softness"] == PBR_MATERIAL_REGION_BLEND_SOFTNESS
    assert healthy.summary["region_tint_strength"] == PBR_MATERIAL_REGION_TINT_STRENGTH
    assert "mucosal_pbr_bump_height" in healthy.summary["channels"]
    assert "mucosal_pbr_mucus_shininess" in healthy.summary["channels"]
    assert healthy.summary["heterogeneity_model"]


def test_mucosal_pbr_material_atlas_applies_authorable_material_map() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0],
            [3.0, 0.5, 5.0],
            [5.0, 2.0, 8.0],
            [7.0, 3.0, 11.0],
            [9.0, 3.5, 14.0],
        ],
        dtype=np.float64,
    )
    centerline = _CenterlineMaterialNodes(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 4.0],
                [6.0, 2.0, 9.0],
                [9.0, 3.5, 14.0],
            ],
            dtype=np.float64,
        ),
        generation=np.asarray([0, 1, 3, 5], dtype=np.float32),
        distance_from_root_mm=np.asarray([0.0, 15.0, 45.0, 85.0], dtype=np.float32),
        radius_mm=np.asarray([7.5, 5.2, 2.5, 1.2], dtype=np.float32),
    )
    material_map = {
        "schema_version": "1.0",
        "name": "unit-test-secretions",
        "regions": [
            {
                "name": "distal mucus",
                "regions": ["distal_airway"],
                "channels": {
                    "mucus": 0.4,
                    "mucus_shininess": 0.22,
                    "displacement_height": 0.12,
                    "wetness": 0.12,
                    "specular": 0.08,
                    "roughness": -0.06,
                },
                "albedo_delta_rgb": [20, 12, 4],
            }
        ],
        "spots": [
            {
                "name": "focal redness",
                "center_normalized_xyz": [0.0, 0.0, 0.0],
                "radius": 2.0,
                "channels": {"erythema": 0.25, "vascularity": 0.08},
                "albedo_delta_rgb": [18, -8, -8],
            }
        ],
    }

    base = _mucosal_pbr_material_atlas(points, centerline_nodes=centerline, material_variant="healthy")
    mapped = _mucosal_pbr_material_atlas(
        points,
        centerline_nodes=centerline,
        material_variant="healthy",
        material_map=material_map,
        material_map_source="inline-test.json",
    )

    distal = base.region_id == 3
    assert mapped.mucus[distal].mean() > base.mucus[distal].mean()
    assert mapped.mucus_shininess[distal].mean() > base.mucus_shininess[distal].mean()
    assert mapped.displacement_height[distal].mean() > base.displacement_height[distal].mean()
    assert mapped.wetness.mean() > base.wetness.mean()
    assert mapped.erythema.mean() > base.erythema.mean()
    assert mapped.albedo_rgb[:, 0].mean() > base.albedo_rgb[:, 0].mean()
    assert mapped.summary["material_map"]["name"] == "unit-test-secretions"
    assert mapped.summary["material_map"]["source"] == "inline-test.json"
    assert mapped.summary["material_map"]["region_entry_count"] == 1
    assert mapped.summary["material_map"]["spot_count"] == 1


def test_pbr_material_map_supports_soft_weights_and_organic_modulation() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0],
            [2.0, 0.2, 4.0],
            [4.5, 1.2, 7.0],
            [7.0, 2.2, 10.0],
            [9.0, 3.0, 14.0],
        ],
        dtype=np.float64,
    )
    centerline = _CenterlineMaterialNodes(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 4.0],
                [5.0, 1.5, 8.0],
                [9.0, 3.0, 14.0],
            ],
            dtype=np.float64,
        ),
        generation=np.asarray([0, 1, 2, 5], dtype=np.float32),
        distance_from_root_mm=np.asarray([0.0, 12.0, 42.0, 85.0], dtype=np.float32),
        radius_mm=np.asarray([7.0, 4.8, 2.6, 1.1], dtype=np.float32),
    )
    material_map = {
        "schema_version": "1.0",
        "name": "soft-weight-test",
        "regions": [
            {
                "name": "soft main wash",
                "regions": ["main_bronchus"],
                "edge_softness": 0.8,
                "opacity": 0.5,
                "texture_noise_strength": 0.25,
                "texture_noise_scale": 2.0,
                "channels": {"stain": 0.3},
            }
        ],
        "spots": [
            {
                "name": "wide feathered spot",
                "center_normalized_xyz": [0.0, 0.0, 0.0],
                "radius": 0.08,
                "falloff": 4.0,
                "edge_softness": 0.5,
                "opacity": 0.8,
                "texture_noise_strength": 0.2,
                "channels": {"mucus": 0.35, "mucus_shininess": 0.2},
            }
        ],
    }

    base = _mucosal_pbr_material_atlas(points, centerline_nodes=centerline, material_variant="healthy")
    mapped = _mucosal_pbr_material_atlas(
        points,
        centerline_nodes=centerline,
        material_variant="healthy",
        material_map=material_map,
        material_map_source="inline-soft-test.json",
    )

    hard_main_count = int(np.count_nonzero(base.region_id == 1))
    applied = mapped.summary["material_map"]
    assert applied["applied_regions"][0]["selected_point_count"] >= hard_main_count
    assert applied["applied_regions"][0]["peak_weight"] <= 0.5
    assert applied["applied_spots"][0]["selected_point_count"] > 1
    assert mapped.stain.mean() > base.stain.mean()
    assert mapped.mucus_shininess.mean() > base.mucus_shininess.mean()


def test_secretions_subtype_presets_separate_color_relief_and_blood_channels() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    thick_map = _load_pbr_material_map(repo_root / "configs/pbr_material_maps/secretions-thick-yellow.json")
    bloody_map = _load_pbr_material_map(repo_root / "configs/pbr_material_maps/secretions-bloody.json")
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0],
            [3.0, 0.5, 5.0],
            [5.0, 2.0, 8.0],
            [7.0, 3.0, 11.0],
            [9.0, 3.5, 14.0],
            [10.5, 4.0, 18.0],
            [12.0, 4.5, 22.0],
        ],
        dtype=np.float64,
    )
    centerline = _CenterlineMaterialNodes(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 4.0],
                [6.0, 2.0, 9.0],
                [9.0, 3.5, 14.0],
                [12.0, 4.5, 22.0],
            ],
            dtype=np.float64,
        ),
        generation=np.asarray([0, 1, 3, 5, 6], dtype=np.float32),
        distance_from_root_mm=np.asarray([0.0, 15.0, 45.0, 85.0, 115.0], dtype=np.float32),
        radius_mm=np.asarray([7.5, 5.2, 2.5, 1.2, 0.8], dtype=np.float32),
    )

    thin = _mucosal_pbr_material_atlas(
        points,
        centerline_nodes=centerline,
        material_variant="edematous",
        material_map=_load_pbr_material_map(
            repo_root / "configs/pbr_material_maps/secretions-thin-clear.json"
        ),
    )
    slight_yellow = _mucosal_pbr_material_atlas(
        points,
        centerline_nodes=centerline,
        material_variant="edematous",
        material_map=_load_pbr_material_map(
            repo_root / "configs/pbr_material_maps/secretions-slightly-yellow.json"
        ),
    )
    thick = _mucosal_pbr_material_atlas(
        points,
        centerline_nodes=centerline,
        material_variant="edematous",
        material_map=thick_map,
    )
    bloody = _mucosal_pbr_material_atlas(
        points,
        centerline_nodes=centerline,
        material_variant="edematous",
        material_map=bloody_map,
    )
    smoker = _mucosal_pbr_material_atlas(
        points,
        centerline_nodes=centerline,
        material_variant="smoker",
        material_map=_load_pbr_material_map(repo_root / "configs/pbr_material_maps/smoker-tar-debris.json"),
    )

    thin_rgb = thin.albedo_rgb.astype(np.int16)
    thick_rgb = thick.albedo_rgb.astype(np.int16)
    slight_yellow_rgb = slight_yellow.albedo_rgb.astype(np.int16)
    thin_yellow_signal = float(np.mean(thin_rgb[:, 0] + thin_rgb[:, 1] - 2 * thin_rgb[:, 2]))
    thick_yellow_signal = float(np.mean(thick_rgb[:, 0] + thick_rgb[:, 1] - 2 * thick_rgb[:, 2]))
    slight_yellow_signal = float(
        np.mean(slight_yellow_rgb[:, 0] + slight_yellow_rgb[:, 1] - 2 * slight_yellow_rgb[:, 2])
    )

    assert thick.mucus.mean() > thin.mucus.mean()
    assert thick.displacement_height.mean() > thin.displacement_height.mean()
    assert thick_yellow_signal > thin_yellow_signal
    assert slight_yellow_signal > thin_yellow_signal
    assert bloody.erythema.mean() > slight_yellow.erythema.mean()
    assert bloody.petechiae.mean() > thin.petechiae.mean()
    assert {overlay["material"] for overlay in thick_map["surface_overlays"]} == {"mucus_film", "mucus_plug"}
    assert {overlay["material"] for overlay in bloody_map["surface_overlays"]} == {
        "blood_clot",
        "blood_droplet",
        "blood_film",
    }
    assert all(float(overlay["edge_feather"]) > 0.0 for overlay in thick_map["surface_overlays"])
    assert all(float(overlay["edge_feather"]) > 0.0 for overlay in bloody_map["surface_overlays"])
    assert smoker.summary["material_map"]["applied_regions"][1]["peak_weight"] <= 0.72


def test_pbr_texture_atlas_images_pack_editable_material_channels() -> None:
    points = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, -1.0, 3.0],
        ],
        dtype=np.float64,
    )
    atlas = _mucosal_pbr_material_atlas(points, material_variant="healthy")

    payload = _pbr_texture_atlas_images(
        points,
        atlas=atlas,
        centerline_nodes=None,
        width_px=32,
        height_px=16,
    )

    images = payload["images"]
    assert tuple(images) == PBR_TEXTURE_ATLAS_CHANNELS
    assert images["albedo_rgb"].shape == (16, 32, 3)
    assert images["roughness"].shape == (16, 32)
    assert images["mucus_shininess"].shape == (16, 32)
    assert images["bump_height"].shape == (16, 32)
    assert images["displacement_height"].shape == (16, 32)
    assert images["region_id"].shape == (16, 32, 3)
    assert images["occupancy"].dtype == np.uint8
    assert np.count_nonzero(images["occupancy"]) >= 4
    assert payload["point_uv"].shape == (4, 2)
    assert payload["uv_mapping"]["centerline_used"] is False
    render_filter = payload["uv_mapping"]["render_texture_filter"]
    assert render_filter["albedo_sigma_px"] == PBR_RENDER_ATLAS_ALBEDO_SIGMA_PX
    assert render_filter["scalar_sigma_px"] == PBR_RENDER_ATLAS_SCALAR_SIGMA_PX
    assert render_filter["albedo_post_fill_sigma_px"] == PBR_RENDER_ATLAS_ALBEDO_POST_FILL_SIGMA_PX
    assert render_filter["scalar_post_fill_sigma_px"] == PBR_RENDER_ATLAS_SCALAR_POST_FILL_SIGMA_PX


def test_face_varying_texture_coordinates_unwraps_cylindrical_seam() -> None:
    point_uv = np.asarray(
        [
            [0.98, 0.20],
            [0.02, 0.24],
            [0.04, 0.30],
            [0.40, 0.35],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

    st = _face_varying_texture_coordinates(point_uv, faces)

    assert st.shape == (6, 2)
    assert np.all(st[:3, 0] >= 0.98)
    assert np.max(st[:3, 0]) - np.min(st[:3, 0]) < 0.07
    assert np.allclose(st[3:, 0], [0.02, 0.04, 0.40])


def test_export_pbr_texture_atlas_writes_pngs_metadata_and_raw_arrays(tmp_path) -> None:
    import pyvista as pv

    mesh = pv.Plane(i_resolution=2, j_resolution=2)
    mesh_path = tmp_path / "airway_mesh.ply"
    mesh.save(mesh_path)

    result = export_pbr_texture_atlas(
        mesh_path=mesh_path,
        output_root=tmp_path / "atlas-export",
        material_variant="healthy",
        atlas_width_px=32,
        atlas_height_px=16,
    )

    assert result.point_count == mesh.n_points
    assert result.metadata_path.is_file()
    assert result.preview_path.is_file()
    assert result.raw_npz_path.is_file()
    assert tuple(result.channel_paths) == PBR_TEXTURE_ATLAS_CHANNELS
    assert all(path.is_file() for path in result.channel_paths.values())
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["material_profile"] == "pbr"
    assert metadata["atlas_width_px"] == 32
    assert metadata["atlas_height_px"] == 16
    with np.load(result.raw_npz_path) as arrays:
        assert arrays["point_uv"].shape == (mesh.n_points, 2)
        assert arrays["point_mucus_shininess"].shape == (mesh.n_points,)
        assert arrays["point_bump_height"].shape == (mesh.n_points,)
        assert arrays["point_displacement_height"].shape == (mesh.n_points,)
        assert arrays["atlas_occupancy"].shape == (16, 32)


def test_export_pbr_mdl_bundle_writes_usd_mdl_scene_and_normal_map(tmp_path) -> None:
    import pyvista as pv

    mesh = pv.Plane(i_resolution=2, j_resolution=2)
    mesh_path = tmp_path / "airway_mesh.ply"
    mesh.save(mesh_path)

    result = export_pbr_mdl_bundle(
        mesh_path=mesh_path,
        output_root=tmp_path / "mdl-export",
        material_variant="healthy",
        atlas_width_px=32,
        atlas_height_px=16,
    )

    assert result.usd_path.is_file()
    assert result.metadata_path.is_file()
    assert result.render_script_path.is_file()
    assert result.normal_map_path.is_file()
    assert result.mesh_point_count == mesh.n_points
    assert result.mesh_face_count > 0
    usd_text = result.usd_path.read_text(encoding="utf-8")
    assert 'asset info:mdl:sourceAsset = @OmniPBR_ClearCoat.mdl@' in usd_text
    assert 'token info:mdl:sourceAsset:subIdentifier = "OmniPBR_ClearCoat"' in usd_text
    assert "bool inputs:enable_clearcoat = true" in usd_text
    assert "float inputs:reflection_roughness_texture_influence = 1" in usd_text
    assert "synairg_mucus_shininess_texture" not in usd_text
    assert "synairg_displacement_height_texture" not in usd_text
    assert "bool doubleSided = true" in usd_text
    assert 'texCoord2f[] primvars:st' in usd_text
    assert 'uniform token info:id = "UsdPreviewSurface"' in usd_text
    assert "normal_from_bump.png" in usd_text
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["material_profile"] == "pbr-mdl"
    assert metadata["shader_targets"]["omniverse_mdl"]["source_asset"] == "OmniPBR_ClearCoat.mdl"
    assert metadata["shader_targets"]["omniverse_mdl"]["shader_target"] == "omnipbr-clearcoat"


def test_export_pbr_mdl_bundle_can_target_omnisurface_texture_coat(tmp_path) -> None:
    import pyvista as pv

    mesh = pv.Plane(i_resolution=2, j_resolution=2)
    mesh_path = tmp_path / "airway_mesh.ply"
    mesh.save(mesh_path)

    result = export_pbr_mdl_bundle(
        mesh_path=mesh_path,
        output_root=tmp_path / "mdl-export-omnisurface",
        material_variant="edematous",
        atlas_width_px=32,
        atlas_height_px=16,
        mdl_shader_target="omnisurface",
    )

    assert result.usd_path.is_file()
    assert result.coat_roughness_map_path is not None
    assert result.coat_roughness_map_path.is_file()
    usd_text = result.usd_path.read_text(encoding="utf-8")
    assert 'asset info:mdl:sourceAsset = @OmniSurface.mdl@' in usd_text
    assert 'token info:mdl:sourceAsset:subIdentifier = "OmniSurface"' in usd_text
    assert "asset inputs:diffuse_reflection_color_image" in usd_text
    assert "asset inputs:specular_reflection_weight_image" in usd_text
    assert "asset inputs:coat_weight_image" in usd_text
    assert "asset inputs:coat_roughness_image" in usd_text
    assert "coat_roughness_from_mucus.png" in usd_text
    assert "asset inputs:coat_normal_image" in usd_text
    assert "asset inputs:geometry_normal_image" in usd_text
    assert "asset inputs:geometry_displacement_image" in usd_text
    assert "synairg_mucus_shininess_texture" not in usd_text
    assert "synairg_displacement_height_texture" not in usd_text
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["shader_targets"]["omniverse_mdl"]["source_asset"] == "OmniSurface.mdl"
    assert metadata["shader_targets"]["omniverse_mdl"]["shader_target"] == "omnisurface"
    assert "coat_weight_image <- mucus_shininess" in metadata["shader_targets"]["omniverse_mdl"]["connected_maps"]


def test_export_pbr_mdl_bundle_emits_bloody_surface_deposit_geometry(tmp_path) -> None:
    import pyvista as pv

    mesh = pv.Plane(i_resolution=4, j_resolution=4)
    mesh_path = tmp_path / "airway_mesh.ply"
    mesh.save(mesh_path)
    material_map_path = tmp_path / "secretions-bloody.json"
    material_map_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "secretions-bloody",
                "regions": [
                    {
                        "name": "blood wash",
                        "regions": ["all"],
                        "channels": {"mucus": 0.2, "erythema": 0.3, "petechiae": 0.1},
                        "albedo_delta_rgb": [40, -20, -25],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = export_pbr_mdl_bundle(
        mesh_path=mesh_path,
        output_root=tmp_path / "mdl-export-bloody",
        material_variant="edematous",
        material_map_path=material_map_path,
        atlas_width_px=32,
        atlas_height_px=16,
        mdl_shader_target="omnisurface",
    )

    usd_text = result.usd_path.read_text(encoding="utf-8")
    assert 'def Material "SynAirG_Blood_Film"' in usd_text
    assert 'def Material "SynAirG_Blood_Film_Edge"' in usd_text
    assert 'def Material "SynAirG_Blood_Clot"' in usd_text
    assert 'def Material "SynAirG_Blood_Droplet"' in usd_text
    assert 'def Mesh "BloodDeposit_00_ElongatedStreak"' in usd_text
    assert 'def Mesh "BloodDeposit_00_ElongatedStreak_FeatherEdge"' in usd_text
    assert 'rel material:binding = </World/Looks/SynAirG_Blood_Film>' in usd_text
    assert 'rel material:binding = </World/Looks/SynAirG_Blood_Film_Edge>' in usd_text


def test_export_pbr_mdl_bundle_can_disable_surface_deposit_geometry(tmp_path) -> None:
    import pyvista as pv

    mesh = pv.Plane(i_resolution=4, j_resolution=4)
    mesh_path = tmp_path / "airway_mesh.ply"
    mesh.save(mesh_path)
    material_map_path = tmp_path / "secretions-bloody.json"
    material_map_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "secretions-bloody",
                "regions": [
                    {
                        "name": "blood wash",
                        "regions": ["all"],
                        "channels": {"mucus": 0.2, "erythema": 0.3, "petechiae": 0.1},
                        "albedo_delta_rgb": [40, -20, -25],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = export_pbr_mdl_bundle(
        mesh_path=mesh_path,
        output_root=tmp_path / "mdl-export-bloody-atlas-only",
        material_variant="edematous",
        material_map_path=material_map_path,
        atlas_width_px=32,
        atlas_height_px=16,
        mdl_shader_target="omnisurface",
        include_surface_overlays=False,
    )

    usd_text = result.usd_path.read_text(encoding="utf-8")
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["include_surface_overlays"] is False
    assert 'def Material "SynAirG_Airway_PBR"' in usd_text
    assert 'def Material "SynAirG_Blood_Film"' not in usd_text
    assert 'def Mesh "BloodDeposit_00_ElongatedStreak"' not in usd_text


def test_export_pbr_mdl_bundle_emits_authorable_mucus_surface_deposit_geometry(tmp_path) -> None:
    import pyvista as pv

    mesh = pv.Plane(i_resolution=4, j_resolution=4)
    mesh_path = tmp_path / "airway_mesh.ply"
    mesh.save(mesh_path)
    material_map_path = tmp_path / "secretions-thick-yellow.json"
    material_map_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "secretions-thick-yellow",
                "regions": [
                    {
                        "name": "yellow wash",
                        "regions": ["all"],
                        "channels": {"mucus": 0.3, "mucus_shininess": 0.12},
                        "albedo_delta_rgb": [40, 30, -5],
                    }
                ],
                "surface_overlays": [
                    {
                        "name": "mucus plug explicit",
                        "material": "mucus_plug",
                        "shape": "plug",
                        "center_normalized_xyz": [0.0, 0.0, 0.0],
                        "major_mm": 2.4,
                        "minor_mm": 1.2,
                        "height_mm": 0.5,
                        "edge_feather": 0.35,
                        "edge_height_scale": 0.18,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = export_pbr_mdl_bundle(
        mesh_path=mesh_path,
        output_root=tmp_path / "mdl-export-mucus",
        material_variant="edematous",
        material_map_path=material_map_path,
        atlas_width_px=32,
        atlas_height_px=16,
        mdl_shader_target="omnisurface",
    )

    usd_text = result.usd_path.read_text(encoding="utf-8")
    assert 'def Material "SynAirG_Mucus_Plug"' in usd_text
    assert 'def Material "SynAirG_Mucus_Plug_Edge"' in usd_text
    assert 'def Mesh "mucus_plug_explicit"' in usd_text
    assert 'def Mesh "mucus_plug_explicit_FeatherEdge"' in usd_text
    assert 'rel material:binding = </World/Looks/SynAirG_Mucus_Plug>' in usd_text
    assert 'rel material:binding = </World/Looks/SynAirG_Mucus_Plug_Edge>' in usd_text


def test_surface_deposit_physical_morphology_controls_create_lumpy_thickness() -> None:
    specs = _surface_deposit_specs_from_material_map(
        {
            "name": "secretions-bloody-physical",
            "surface_overlays": [
                {
                    "name": "lumpy blood pool",
                    "material": "blood_clot",
                    "shape": "pool",
                    "center_normalized_xyz": [0.0, 0.0, 0.0],
                    "major_mm": 3.0,
                    "minor_mm": 1.3,
                    "height_mm": 0.7,
                    "ring_count": 4,
                    "segment_count": 32,
                    "thickness_noise_strength": 0.42,
                    "rim_height_scale": 0.2,
                    "runoff_strength": 0.12,
                }
            ],
        }
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.thickness_noise_strength == 0.42
    assert spec.rim_height_scale == 0.2
    assert spec.runoff_strength == 0.12

    vertices, faces = _surface_deposit_mesh(
        anchor=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        surface_normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        centerline_nodes=None,
        spec=spec,
        base_offset_mm=0.05,
    )
    first_ring = vertices[1 : 1 + spec.segment_count]
    first_ring_thickness = -(first_ring[:, 2] + 0.05)

    assert faces.shape[0] > spec.segment_count
    assert float(np.std(first_ring_thickness)) > 0.015
    assert float(np.max(first_ring_thickness)) > float(np.min(first_ring_thickness))


def test_surface_deposit_specs_preserve_subtle_thin_film_height() -> None:
    specs = _surface_deposit_specs_from_material_map(
        {
            "name": "secretions-bloody-fresh-sliver-breakup",
            "surface_overlays": [
                {
                    "name": "subtle fresh film",
                    "material": "blood_film_fresh",
                    "shape": "film",
                    "center_normalized_xyz": [0.0, 0.0, 0.0],
                    "major_mm": 0.55,
                    "minor_mm": 0.12,
                    "height_mm": 0.009,
                    "edge_feather": 0.22,
                    "edge_height_scale": 0.018,
                }
            ],
        }
    )

    assert len(specs) == 1
    assert specs[0].height_mm == 0.009
    assert specs[0].edge_height_scale == 0.018


def test_translucent_blood_materials_export_distinct_usd_materials(tmp_path: Path) -> None:
    pytest.importorskip("pyvista")
    import pyvista as pv

    mesh = pv.Sphere(radius=5.0, theta_resolution=12, phi_resolution=8)
    mesh_path = tmp_path / "airway_mesh.ply"
    mesh.save(mesh_path)
    material_map_path = tmp_path / "secretions-bloody-placement-refined.json"
    material_map_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "secretions-bloody-placement-refined",
                "surface_overlays": [
                    {
                        "name": "thin fresh film",
                        "material": "blood_film_thin",
                        "shape": "film",
                        "center_normalized_xyz": [0.0, 0.0, 0.0],
                        "major_mm": 2.8,
                        "minor_mm": 1.1,
                        "height_mm": 0.12,
                        "edge_feather": 0.5,
                        "edge_height_scale": 0.1,
                    },
                    {
                        "name": "soft small clot",
                        "material": "blood_clot_soft",
                        "shape": "pool",
                        "center_normalized_xyz": [0.05, 0.0, 0.0],
                        "major_mm": 1.4,
                        "minor_mm": 0.8,
                        "height_mm": 0.24,
                        "edge_feather": 0.35,
                        "edge_height_scale": 0.12,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = export_pbr_mdl_bundle(
        mesh_path=mesh_path,
        output_root=tmp_path / "mdl-export-blood-translucent",
        material_variant="pale",
        material_map_path=material_map_path,
        atlas_width_px=32,
        atlas_height_px=16,
        mdl_shader_target="omnisurface",
    )

    usd_text = result.usd_path.read_text(encoding="utf-8")
    assert 'def Material "SynAirG_Blood_Film_Thin"' in usd_text
    assert 'def Material "SynAirG_Blood_Film_Thin_Edge"' in usd_text
    assert 'def Material "SynAirG_Blood_Clot_Soft"' in usd_text
    assert 'def Material "SynAirG_Blood_Clot_Soft_Edge"' in usd_text
    assert 'rel material:binding = </World/Looks/SynAirG_Blood_Film_Thin>' in usd_text
    assert 'rel material:binding = </World/Looks/SynAirG_Blood_Clot_Soft>' in usd_text


def test_fresh_blood_materials_export_red_glossy_usd_materials(tmp_path: Path) -> None:
    pytest.importorskip("pyvista")
    import pyvista as pv

    mesh = pv.Sphere(radius=5.0, theta_resolution=12, phi_resolution=8)
    mesh_path = tmp_path / "airway_mesh.ply"
    mesh.save(mesh_path)
    material_map_path = tmp_path / "secretions-bloody-fresh-optics.json"
    material_map_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "secretions-bloody-fresh-optics",
                "surface_overlays": [
                    {
                        "name": "fresh thin film",
                        "material": "blood_film_fresh",
                        "shape": "ribbon",
                        "center_normalized_xyz": [0.0, 0.0, 0.0],
                        "major_mm": 2.8,
                        "minor_mm": 0.4,
                        "height_mm": 0.08,
                        "edge_feather": 0.6,
                        "edge_height_scale": 0.1,
                    },
                    {
                        "name": "fresh soft clot",
                        "material": "blood_clot_fresh",
                        "shape": "pool",
                        "center_normalized_xyz": [0.05, 0.0, 0.0],
                        "major_mm": 1.2,
                        "minor_mm": 0.7,
                        "height_mm": 0.18,
                        "edge_feather": 0.4,
                        "edge_height_scale": 0.12,
                    },
                    {
                        "name": "fresh wet bead",
                        "material": "blood_droplet_fresh",
                        "shape": "droplet",
                        "center_normalized_xyz": [-0.05, 0.0, 0.0],
                        "major_mm": 0.8,
                        "minor_mm": 0.45,
                        "height_mm": 0.16,
                        "edge_feather": 0.35,
                        "edge_height_scale": 0.1,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = export_pbr_mdl_bundle(
        mesh_path=mesh_path,
        output_root=tmp_path / "mdl-export-fresh-blood",
        material_variant="pale",
        material_map_path=material_map_path,
        atlas_width_px=32,
        atlas_height_px=16,
        mdl_shader_target="omnisurface",
    )

    usd_text = result.usd_path.read_text(encoding="utf-8")
    assert 'def Material "SynAirG_Blood_Film_Fresh"' in usd_text
    assert 'def Material "SynAirG_Blood_Film_Fresh_Edge"' in usd_text
    assert 'def Material "SynAirG_Blood_Clot_Fresh"' in usd_text
    assert 'def Material "SynAirG_Blood_Droplet_Fresh"' in usd_text
    assert 'rel material:binding = </World/Looks/SynAirG_Blood_Film_Fresh>' in usd_text
    assert 'rel material:binding = </World/Looks/SynAirG_Blood_Droplet_Fresh>' in usd_text
    assert "color3f inputs:diffuseColor = (0.46, 0.066, 0.042)" in usd_text
    assert "float inputs:clearcoat = 0.96" in usd_text


def test_pbr_region_actor_specs_split_cells_by_material_region() -> None:
    import pyvista as pv

    mesh = pv.Plane(i_resolution=2, j_resolution=1)
    atlas = _mucosal_pbr_material_atlas(np.asarray(mesh.points, dtype=np.float64), material_variant="healthy")
    region_id = np.asarray([0, 0, 3, 3, 3, 3], dtype=np.int16)
    atlas = type(atlas)(
        albedo_rgb=atlas.albedo_rgb,
        roughness=np.asarray([0.72, 0.70, 0.32, 0.34, 0.36, 0.38], dtype=np.float32),
        specular=np.asarray([0.22, 0.24, 0.68, 0.66, 0.64, 0.62], dtype=np.float32),
        wetness=np.asarray([0.20, 0.22, 0.82, 0.80, 0.78, 0.76], dtype=np.float32),
        vascularity=atlas.vascularity,
        mucus=atlas.mucus,
        mucus_shininess=np.asarray([0.10, 0.12, 0.62, 0.64, 0.66, 0.68], dtype=np.float32),
        erythema=atlas.erythema,
        petechiae=atlas.petechiae,
        stain=atlas.stain,
        bump_height=atlas.bump_height,
        displacement_height=atlas.displacement_height,
        region_id=region_id,
        generation=atlas.generation,
        summary=atlas.summary,
    )

    cell_regions = _pbr_cell_region_ids(mesh, region_id)
    specs = _pbr_region_actor_specs(atlas, cell_regions)

    assert cell_regions.tolist() == [0, 3]
    assert [spec.region_id for spec in specs] == [0, 3]
    assert specs[0].roughness > specs[1].roughness
    assert specs[0].specular < specs[1].specular
    assert specs[0].region_name == "tracheal_or_proximal_airway"
    assert specs[1].region_name == "distal_airway"


def test_compose_diagnostic_frame_builds_two_by_two_canvas() -> None:
    panels = [
        np.full((24, 32, 3), value, dtype=np.uint8)
        for value in (40, 80, 120, 160, 200, 220)
    ]

    composite = _compose_diagnostic_frame(
        scope_rgb=panels[0],
        depth_rgb=panels[1],
        normals_rgb=panels[2],
        pps_rgb=panels[3],
        ct_rgb=panels[4],
        locator_rgb=panels[5],
        max_depth_mm=95.0,
    )

    assert composite.shape == (48, 96, 3)
    assert composite[20, 8].mean() >= 35
    assert composite[20, 40].mean() >= 70
    assert composite[40, 8].mean() >= 110
    assert composite[40, 40].mean() >= 150
    assert composite[20, 72].mean() >= 190
    assert composite[40, 72].mean() >= 210
