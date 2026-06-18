"""Bronchoscope point-of-view rendering from airway meshes and camera paths."""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from synairg_core.volume_manifest import JsonValue

MATERIAL_PROFILES = frozenset(("basic", "vascular", "pbr"))
MATERIAL_VARIANTS = frozenset(("healthy", "inflamed", "smoker", "pale", "edematous"))
MDL_SHADER_TARGETS = frozenset(("omnipbr-clearcoat", "omnipbr", "omnisurface"))
LOCAL_PBR_CHANNELS = frozenset(("mucus", "erythema", "petechiae", "stain"))
PBR_PROPERTY_CHANNELS = frozenset(
    ("roughness", "specular", "wetness", "vascularity", "bump_height", "displacement_height", "mucus_shininess")
)
PBR_TEXTURE_ATLAS_CHANNELS = (
    "albedo_rgb",
    "roughness",
    "specular",
    "wetness",
    "vascularity",
    "mucus",
    "mucus_shininess",
    "erythema",
    "petechiae",
    "stain",
    "bump_height",
    "displacement_height",
    "region_id",
    "generation",
    "occupancy",
)
PBR_RENDER_ATLAS_ALBEDO_SIGMA_PX = 5.0
PBR_RENDER_ATLAS_SCALAR_SIGMA_PX = 4.0
PBR_RENDER_ATLAS_ALBEDO_POST_FILL_SIGMA_PX = 12.0
PBR_RENDER_ATLAS_SCALAR_POST_FILL_SIGMA_PX = 12.0
PBR_MATERIAL_REGION_BLEND_SOFTNESS = 2.25
PBR_MATERIAL_REGION_TINT_STRENGTH = 0.58
PBR_NORMAL_MAP_HEIGHT_STRENGTH = 0.32
PBR_NORMAL_MAP_HEIGHT_SMOOTH_SIGMA_PX = 2.5
PBR_MDL_GEOMETRY_NORMAL_ROUGHNESS_STRENGTH = 0.35
PBR_REGION_COLORS = np.asarray(
    (
        (206, 154, 111),
        (203, 91, 82),
        (160, 58, 84),
        (108, 47, 92),
    ),
    dtype=np.uint8,
)


@dataclass(frozen=True)
class ScopeVideoResult:
    """Artifacts emitted by one scope-view render."""

    path_id: str
    output_video_path: Path
    preview_frame_path: Path
    metadata_path: Path
    frame_count: int
    fps: float
    width_px: int
    height_px: int
    panel_width_px: int
    panel_height_px: int
    diagnostic_panels: bool
    condition_output_root: Path | None = None
    condition_manifest_path: Path | None = None
    condition_sequence_id: str | None = None
    condition_modalities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "path_id": self.path_id,
            "output_video_path": str(self.output_video_path),
            "preview_frame_path": str(self.preview_frame_path),
            "metadata_path": str(self.metadata_path),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "panel_width_px": self.panel_width_px,
            "panel_height_px": self.panel_height_px,
            "diagnostic_panels": self.diagnostic_panels,
            "condition_output_root": (
                str(self.condition_output_root) if self.condition_output_root is not None else None
            ),
            "condition_manifest_path": (
                str(self.condition_manifest_path) if self.condition_manifest_path is not None else None
            ),
            "condition_sequence_id": self.condition_sequence_id,
            "condition_modalities": list(self.condition_modalities),
        }


@dataclass(frozen=True)
class PbrTextureAtlasExportResult:
    """Editable material-channel atlas emitted from the PBR mucosa model."""

    output_root: Path
    metadata_path: Path
    preview_path: Path
    raw_npz_path: Path
    channel_paths: dict[str, Path]
    atlas_width_px: int
    atlas_height_px: int
    point_count: int
    material_variant: str
    material_map_path: Path | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "output_root": str(self.output_root),
            "metadata_path": str(self.metadata_path),
            "preview_path": str(self.preview_path),
            "raw_npz_path": str(self.raw_npz_path),
            "channel_paths": {key: str(value) for key, value in self.channel_paths.items()},
            "atlas_width_px": self.atlas_width_px,
            "atlas_height_px": self.atlas_height_px,
            "point_count": self.point_count,
            "material_variant": self.material_variant,
            "material_map_path": str(self.material_map_path) if self.material_map_path is not None else None,
        }


@dataclass(frozen=True)
class PbrMdlBundleExportResult:
    """USD/MDL renderer handoff bundle emitted from the PBR texture atlas."""

    output_root: Path
    usd_path: Path
    metadata_path: Path
    render_script_path: Path
    atlas_result: PbrTextureAtlasExportResult
    normal_map_path: Path
    mesh_point_count: int
    mesh_face_count: int
    material_variant: str
    material_map_path: Path | None = None
    coat_roughness_map_path: Path | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "output_root": str(self.output_root),
            "usd_path": str(self.usd_path),
            "metadata_path": str(self.metadata_path),
            "render_script_path": str(self.render_script_path),
            "atlas": self.atlas_result.to_dict(),
            "normal_map_path": str(self.normal_map_path),
            "mesh_point_count": self.mesh_point_count,
            "mesh_face_count": self.mesh_face_count,
            "material_variant": self.material_variant,
            "material_map_path": str(self.material_map_path) if self.material_map_path is not None else None,
            "coat_roughness_map_path": (
                str(self.coat_roughness_map_path) if self.coat_roughness_map_path is not None else None
            ),
        }


@dataclass(frozen=True)
class _RenderInfo:
    """Internal render accounting for the emitted video layout."""

    frame_count: int
    width_px: int
    height_px: int
    depth_range_mm: tuple[float, float]
    pps_range: tuple[float, float] | None
    source_ct_path: Path | None
    condition_export: _ConditionExportInfo | None
    material_summary: dict[str, JsonValue]


@dataclass(frozen=True)
class _ConditionExportInfo:
    """Condition-frame directories emitted alongside a scope render."""

    output_root: Path
    sequence_id: str
    frame_count: int
    roots: dict[str, Path]

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(sorted(self.roots))


@dataclass(frozen=True)
class _CtContext:
    """Source CT volume and registration needed for following orthogonal slices."""

    volume: np.ndarray
    mesh_to_voxel: np.ndarray
    spacing_mm: tuple[float, float, float]
    source_ct_path: Path
    window_hu: tuple[float, float] = (-1000.0, 500.0)


@dataclass(frozen=True)
class _IsoLocatorContext:
    """Pre-projected isometric locator data for the moving scope marker."""

    base_image: np.ndarray
    route_xy: np.ndarray
    route_depth: np.ndarray


@dataclass(frozen=True)
class _CenterlineMaterialNodes:
    """Centerline features used to derive anatomy-aware material regions."""

    points: np.ndarray
    generation: np.ndarray
    distance_from_root_mm: np.ndarray
    radius_mm: np.ndarray


@dataclass(frozen=True)
class _MucosalMaterialAtlas:
    """Per-point material channels for airway rendering."""

    albedo_rgb: np.ndarray
    roughness: np.ndarray
    specular: np.ndarray
    wetness: np.ndarray
    vascularity: np.ndarray
    mucus: np.ndarray
    mucus_shininess: np.ndarray
    erythema: np.ndarray
    petechiae: np.ndarray
    stain: np.ndarray
    bump_height: np.ndarray
    displacement_height: np.ndarray
    region_id: np.ndarray
    generation: np.ndarray
    summary: dict[str, JsonValue]


@dataclass(frozen=True)
class _PbrRegionActorSpec:
    """Actor-level PBR settings for one extracted material region."""

    region_id: int
    region_name: str
    cell_count: int
    point_count: int
    roughness: float
    specular: float
    specular_power: float
    wetness_mean: float
    vascularity_mean: float
    mucus_shininess_mean: float
    bump_height_mean: float
    displacement_height_mean: float
    ambient: float = 0.22
    diffuse: float = 1.0
    metallic: float = 0.0

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "region_id": self.region_id,
            "region_name": self.region_name,
            "cell_count": self.cell_count,
            "point_count": self.point_count,
            "roughness": self.roughness,
            "specular": self.specular,
            "specular_power": self.specular_power,
            "wetness_mean": self.wetness_mean,
            "vascularity_mean": self.vascularity_mean,
            "mucus_shininess_mean": self.mucus_shininess_mean,
            "bump_height_mean": self.bump_height_mean,
            "displacement_height_mean": self.displacement_height_mean,
            "ambient": self.ambient,
            "diffuse": self.diffuse,
            "metallic": self.metallic,
        }


@dataclass(frozen=True)
class _SurfaceOverlayUsd:
    """USD snippets for optional raised material deposits authored over the airway wall."""

    mesh_lines: tuple[str, ...] = ()
    material_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SurfaceDepositSpec:
    """Authorable wall-adherent fluid geometry emitted alongside the airway mesh."""

    name: str
    material: str
    center_normalized_xyz: tuple[float, float, float]
    major_mm: float
    minor_mm: float
    height_mm: float
    phase: float
    shape: str = "pool"
    segment_count: int = 48
    ring_count: int = 4
    lobe_strength: float = 0.16
    edge_feather: float = 0.0
    edge_height_scale: float = 0.24
    base_offset_mm: float = 0.045
    thickness_noise_strength: float = 0.0
    rim_height_scale: float = 0.0
    runoff_strength: float = 0.0


_SURFACE_DEPOSIT_MATERIALS: dict[str, dict[str, Any]] = {
    "blood_film": {
        "prim": "SynAirG_Blood_Film",
        "diffuse": (0.17, 0.018, 0.012),
        "specular": (0.92, 0.24, 0.17),
        "roughness": 0.065,
        "clearcoat": 0.9,
        "clearcoat_roughness": 0.035,
        "opacity": 0.72,
    },
    "blood_film_edge": {
        "prim": "SynAirG_Blood_Film_Edge",
        "diffuse": (0.12, 0.015, 0.011),
        "specular": (0.52, 0.12, 0.09),
        "roughness": 0.16,
        "clearcoat": 0.5,
        "clearcoat_roughness": 0.09,
        "opacity": 0.34,
    },
    "blood_film_thin": {
        "prim": "SynAirG_Blood_Film_Thin",
        "diffuse": (0.24, 0.042, 0.028),
        "specular": (0.98, 0.34, 0.22),
        "roughness": 0.048,
        "clearcoat": 0.94,
        "clearcoat_roughness": 0.03,
        "opacity": 0.46,
    },
    "blood_film_thin_edge": {
        "prim": "SynAirG_Blood_Film_Thin_Edge",
        "diffuse": (0.18, 0.035, 0.024),
        "specular": (0.74, 0.25, 0.16),
        "roughness": 0.11,
        "clearcoat": 0.62,
        "clearcoat_roughness": 0.055,
        "opacity": 0.18,
    },
    "blood_film_fresh": {
        "prim": "SynAirG_Blood_Film_Fresh",
        "diffuse": (0.46, 0.066, 0.042),
        "specular": (1.0, 0.48, 0.34),
        "roughness": 0.035,
        "clearcoat": 0.96,
        "clearcoat_roughness": 0.024,
        "opacity": 0.38,
    },
    "blood_film_fresh_edge": {
        "prim": "SynAirG_Blood_Film_Fresh_Edge",
        "diffuse": (0.34, 0.072, 0.05),
        "specular": (0.82, 0.32, 0.22),
        "roughness": 0.095,
        "clearcoat": 0.72,
        "clearcoat_roughness": 0.05,
        "opacity": 0.16,
    },
    "blood_clot": {
        "prim": "SynAirG_Blood_Clot",
        "diffuse": (0.12, 0.014, 0.01),
        "specular": (0.58, 0.13, 0.1),
        "roughness": 0.13,
        "clearcoat": 0.68,
        "clearcoat_roughness": 0.06,
        "opacity": 0.86,
    },
    "blood_clot_edge": {
        "prim": "SynAirG_Blood_Clot_Edge",
        "diffuse": (0.09, 0.01, 0.008),
        "specular": (0.32, 0.07, 0.055),
        "roughness": 0.2,
        "clearcoat": 0.38,
        "clearcoat_roughness": 0.14,
        "opacity": 0.42,
    },
    "blood_clot_soft": {
        "prim": "SynAirG_Blood_Clot_Soft",
        "diffuse": (0.105, 0.014, 0.01),
        "specular": (0.44, 0.11, 0.085),
        "roughness": 0.16,
        "clearcoat": 0.52,
        "clearcoat_roughness": 0.08,
        "opacity": 0.64,
    },
    "blood_clot_soft_edge": {
        "prim": "SynAirG_Blood_Clot_Soft_Edge",
        "diffuse": (0.095, 0.018, 0.013),
        "specular": (0.36, 0.1, 0.075),
        "roughness": 0.19,
        "clearcoat": 0.42,
        "clearcoat_roughness": 0.11,
        "opacity": 0.26,
    },
    "blood_clot_fresh": {
        "prim": "SynAirG_Blood_Clot_Fresh",
        "diffuse": (0.22, 0.028, 0.019),
        "specular": (0.68, 0.19, 0.13),
        "roughness": 0.12,
        "clearcoat": 0.7,
        "clearcoat_roughness": 0.052,
        "opacity": 0.56,
    },
    "blood_clot_fresh_edge": {
        "prim": "SynAirG_Blood_Clot_Fresh_Edge",
        "diffuse": (0.18, 0.04, 0.029),
        "specular": (0.46, 0.14, 0.1),
        "roughness": 0.17,
        "clearcoat": 0.5,
        "clearcoat_roughness": 0.09,
        "opacity": 0.24,
    },
    "blood_droplet": {
        "prim": "SynAirG_Blood_Droplet",
        "diffuse": (0.125, 0.009, 0.006),
        "specular": (0.95, 0.32, 0.23),
        "roughness": 0.028,
        "clearcoat": 0.96,
        "clearcoat_roughness": 0.022,
        "opacity": 0.9,
    },
    "blood_droplet_edge": {
        "prim": "SynAirG_Blood_Droplet_Edge",
        "diffuse": (0.105, 0.011, 0.008),
        "specular": (0.48, 0.13, 0.1),
        "roughness": 0.14,
        "clearcoat": 0.46,
        "clearcoat_roughness": 0.08,
        "opacity": 0.28,
    },
    "blood_droplet_fresh": {
        "prim": "SynAirG_Blood_Droplet_Fresh",
        "diffuse": (0.5, 0.048, 0.029),
        "specular": (1.0, 0.46, 0.34),
        "roughness": 0.024,
        "clearcoat": 0.98,
        "clearcoat_roughness": 0.018,
        "opacity": 0.74,
    },
    "blood_droplet_fresh_edge": {
        "prim": "SynAirG_Blood_Droplet_Fresh_Edge",
        "diffuse": (0.34, 0.052, 0.036),
        "specular": (0.7, 0.23, 0.17),
        "roughness": 0.12,
        "clearcoat": 0.56,
        "clearcoat_roughness": 0.07,
        "opacity": 0.22,
    },
    "mucus_film": {
        "prim": "SynAirG_Mucus_Film",
        "diffuse": (0.74, 0.65, 0.42),
        "specular": (0.84, 0.76, 0.55),
        "roughness": 0.2,
        "clearcoat": 0.74,
        "clearcoat_roughness": 0.075,
        "opacity": 0.58,
    },
    "mucus_film_edge": {
        "prim": "SynAirG_Mucus_Film_Edge",
        "diffuse": (0.66, 0.59, 0.4),
        "specular": (0.46, 0.4, 0.3),
        "roughness": 0.28,
        "clearcoat": 0.36,
        "clearcoat_roughness": 0.15,
        "opacity": 0.3,
    },
    "mucus_plug": {
        "prim": "SynAirG_Mucus_Plug",
        "diffuse": (0.69, 0.58, 0.32),
        "specular": (0.6, 0.52, 0.34),
        "roughness": 0.28,
        "clearcoat": 0.48,
        "clearcoat_roughness": 0.12,
        "opacity": 0.86,
    },
    "mucus_plug_edge": {
        "prim": "SynAirG_Mucus_Plug_Edge",
        "diffuse": (0.62, 0.53, 0.31),
        "specular": (0.32, 0.28, 0.2),
        "roughness": 0.34,
        "clearcoat": 0.22,
        "clearcoat_roughness": 0.2,
        "opacity": 0.38,
    },
}


def export_pbr_texture_atlas(
    *,
    mesh_path: Path,
    output_root: Path,
    material_variant: str = "healthy",
    material_map_path: Path | None = None,
    atlas_width_px: int = 512,
    atlas_height_px: int = 256,
    overwrite: bool = False,
) -> PbrTextureAtlasExportResult:
    """Export editable 2D PNG atlases from the anatomy-aware PBR material model."""
    if material_variant not in MATERIAL_VARIANTS:
        raise ValueError(f"material_variant must be one of {sorted(MATERIAL_VARIANTS)}")
    if atlas_width_px < 16 or atlas_height_px < 16:
        raise ValueError("texture atlas dimensions must be at least 16x16")
    if material_map_path is not None and not material_map_path.is_file():
        raise ValueError(f"material map file does not exist: {material_map_path}")
    if output_root.exists():
        if not overwrite and any(output_root.iterdir()):
            raise ValueError(f"texture atlas output already exists: {output_root}; pass overwrite=True")
        if overwrite:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio  # type: ignore[import-not-found]
        import pyvista as pv  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("pyvista and imageio are required to export PBR texture atlases") from exc

    mesh = pv.read(str(mesh_path))
    points = np.asarray(mesh.points, dtype=np.float64)
    centerline_nodes = _load_centerline_material_nodes(mesh_path)
    atlas = _mucosal_pbr_material_atlas(
        points,
        centerline_nodes=centerline_nodes,
        material_variant=material_variant,
        material_map=_load_pbr_material_map(material_map_path),
        material_map_source=str(material_map_path) if material_map_path is not None else None,
    )
    atlas_payload = _pbr_texture_atlas_images(
        points,
        atlas=atlas,
        centerline_nodes=centerline_nodes,
        width_px=atlas_width_px,
        height_px=atlas_height_px,
    )
    channel_paths: dict[str, Path] = {}
    channel_dir = output_root / "channels"
    channel_dir.mkdir(parents=True, exist_ok=True)
    for channel_name in PBR_TEXTURE_ATLAS_CHANNELS:
        path = channel_dir / f"{channel_name}.png"
        imageio.imwrite(path, atlas_payload["images"][channel_name])
        channel_paths[channel_name] = path

    preview_path = output_root / "preview.png"
    imageio.imwrite(preview_path, _compose_pbr_texture_atlas_preview(atlas_payload["images"]))
    raw_npz_path = output_root / "atlas_channels.npz"
    np.savez_compressed(
        raw_npz_path,
        point_uv=atlas_payload["point_uv"],
        point_region_id=atlas.region_id,
        point_generation=atlas.generation,
        point_albedo_rgb=atlas.albedo_rgb,
        point_roughness=atlas.roughness,
        point_specular=atlas.specular,
        point_wetness=atlas.wetness,
        point_vascularity=atlas.vascularity,
        point_mucus=atlas.mucus,
        point_mucus_shininess=atlas.mucus_shininess,
        point_erythema=atlas.erythema,
        point_petechiae=atlas.petechiae,
        point_stain=atlas.stain,
        point_bump_height=atlas.bump_height,
        point_displacement_height=atlas.displacement_height,
        atlas_occupancy=atlas_payload["occupied"].astype(np.uint8),
    )
    metadata_path = output_root / "metadata.json"
    metadata = {
        "schema_version": "1.0",
        "source_mesh_path": str(mesh_path),
        "material_profile": "pbr",
        "material_variant": material_variant,
        "material_map_path": str(material_map_path) if material_map_path is not None else None,
        "atlas_width_px": atlas_width_px,
        "atlas_height_px": atlas_height_px,
        "point_count": int(points.shape[0]),
        "uv_mapping": atlas_payload["uv_mapping"],
        "channels": {key: str(path) for key, path in channel_paths.items()},
        "preview_path": str(preview_path),
        "raw_npz_path": str(raw_npz_path),
        "region_colors_rgb": {
            str(region_id): {
                "name": _pbr_region_name(region_id),
                "rgb": PBR_REGION_COLORS[region_id].astype(int).tolist(),
            }
            for region_id in range(4)
        },
        "material_summary": atlas.summary,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return PbrTextureAtlasExportResult(
        output_root=output_root,
        metadata_path=metadata_path,
        preview_path=preview_path,
        raw_npz_path=raw_npz_path,
        channel_paths=channel_paths,
        atlas_width_px=atlas_width_px,
        atlas_height_px=atlas_height_px,
        point_count=int(points.shape[0]),
        material_variant=material_variant,
        material_map_path=material_map_path,
    )


def export_pbr_mdl_bundle(
    *,
    mesh_path: Path,
    output_root: Path,
    material_variant: str = "healthy",
    material_map_path: Path | None = None,
    atlas_width_px: int = 1024,
    atlas_height_px: int = 512,
    mdl_shader_target: str = "omnipbr-clearcoat",
    include_surface_overlays: bool = True,
    overwrite: bool = False,
) -> PbrMdlBundleExportResult:
    """Export a USD scene with MDL/OmniPBR bindings and PBR texture atlases."""
    if mdl_shader_target not in MDL_SHADER_TARGETS:
        raise ValueError(
            f"unsupported MDL shader target: {mdl_shader_target}; "
            f"expected one of {', '.join(sorted(MDL_SHADER_TARGETS))}"
        )
    if output_root.exists():
        if not overwrite and any(output_root.iterdir()):
            raise ValueError(f"MDL bundle output already exists: {output_root}; pass overwrite=True")
        if overwrite:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio  # type: ignore[import-not-found]
        import pyvista as pv  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("pyvista and imageio are required to export USD/MDL PBR bundles") from exc

    atlas_result = export_pbr_texture_atlas(
        mesh_path=mesh_path,
        output_root=output_root / "textures",
        material_variant=material_variant,
        material_map_path=material_map_path,
        atlas_width_px=atlas_width_px,
        atlas_height_px=atlas_height_px,
        overwrite=True,
    )
    mesh = pv.read(str(mesh_path)).triangulate()
    mesh, normals = _mesh_with_point_normals(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)
    faces = _mesh_triangles(mesh)
    if faces.size == 0:
        raise ValueError(f"mesh has no triangular faces after triangulation: {mesh_path}")

    with np.load(atlas_result.raw_npz_path) as arrays:
        point_uv = np.asarray(arrays["point_uv"], dtype=np.float32)
    if point_uv.shape != (points.shape[0], 2):
        raise ValueError(
            f"atlas point_uv shape {point_uv.shape} does not match mesh point count {points.shape[0]}"
        )

    bump_height = imageio.imread(atlas_result.channel_paths["bump_height"])
    normal_map = _normal_map_from_height_image(
        bump_height,
        strength=PBR_NORMAL_MAP_HEIGHT_STRENGTH,
        pre_smooth_sigma=PBR_NORMAL_MAP_HEIGHT_SMOOTH_SIGMA_PX,
    )
    normal_map_path = atlas_result.output_root / "channels" / "normal_from_bump.png"
    imageio.imwrite(normal_map_path, normal_map)
    mucus_shininess = imageio.imread(atlas_result.channel_paths["mucus_shininess"])
    coat_roughness_map = _coat_roughness_from_mucus_image(mucus_shininess)
    coat_roughness_map_path = atlas_result.output_root / "channels" / "coat_roughness_from_mucus.png"
    imageio.imwrite(coat_roughness_map_path, coat_roughness_map)

    usd_path = output_root / "airway_omnipbr_mdl.usda"
    _write_pbr_mdl_usda(
        usd_path,
        points=points,
        faces=faces,
        normals=normals,
        point_uv=point_uv,
        channel_paths=atlas_result.channel_paths,
        normal_map_path=normal_map_path,
        coat_roughness_map_path=coat_roughness_map_path,
        material_summary=_load_json(atlas_result.metadata_path)["material_summary"],
        mdl_shader_target=mdl_shader_target,
        centerline_nodes=_load_centerline_material_nodes(mesh_path),
        include_surface_overlays=include_surface_overlays,
    )
    render_script_path = output_root / "render_omniverse_rtx.py"
    _write_omniverse_render_script(render_script_path, usd_path.name)

    metadata_path = output_root / "metadata.json"
    metadata: dict[str, JsonValue] = {
        "schema_version": "1.0",
        "source_mesh_path": str(mesh_path),
        "material_profile": "pbr-mdl",
        "material_variant": material_variant,
        "material_map_path": str(material_map_path) if material_map_path is not None else None,
        "include_surface_overlays": include_surface_overlays,
        "usd_path": str(usd_path),
        "render_script_path": str(render_script_path),
        "mesh_point_count": int(points.shape[0]),
        "mesh_face_count": int(faces.shape[0]),
        "texture_root": str(atlas_result.output_root),
        "normal_map_path": str(normal_map_path),
        "normal_map_filter": {
            "height_strength": PBR_NORMAL_MAP_HEIGHT_STRENGTH,
            "height_smoothing_sigma_px": PBR_NORMAL_MAP_HEIGHT_SMOOTH_SIGMA_PX,
            "purpose": "prevent sparse-atlas scalar boundaries from becoming hard lighting edges",
        },
        "coat_roughness_map_path": str(coat_roughness_map_path),
        "shader_targets": {
            "usd_preview_surface": {
                "purpose": "portable debug fallback",
                "connected_maps": [
                    "albedo_rgb",
                    "roughness",
                    "specular",
                    "mucus_shininess",
                    "normal_from_bump",
                    "displacement_height",
                ],
            },
            "omniverse_mdl": {
                "source_asset": _mdl_shader_source_asset(mdl_shader_target),
                "source_asset_subidentifier": _mdl_shader_subidentifier(mdl_shader_target),
                "shader_target": mdl_shader_target,
                "connected_maps": [
                    *(
                        [
                            "diffuse_reflection_color_image <- albedo_rgb",
                            "diffuse_reflection_roughness_image <- roughness",
                            "specular_reflection_weight_image <- specular",
                            "coat_weight_image <- mucus_shininess",
                            "coat_roughness_image <- coat_roughness_from_mucus",
                            "geometry_normal_image <- normal_from_bump",
                            "coat_normal_image <- normal_from_bump",
                            "geometry_displacement_image <- displacement_height",
                        ]
                        if mdl_shader_target == "omnisurface"
                        else [
                            "diffuse_texture <- albedo_rgb",
                            "reflectionroughness_texture <- roughness with texture influence 1.0",
                            "normalmap_texture <- normal_from_bump",
                            "clearcoat_normalmap_texture <- normal_from_bump when target supports clearcoat",
                            "clearcoat scalar response <- mucus_shininess mean when target supports clearcoat",
                        ]
                    ),
                ],
                "unsupported_custom_inputs_removed": [
                    "synairg_mucus_shininess_texture",
                    "synairg_displacement_height_texture",
                ],
            },
        },
        "uv_mapping": {
            "primvar": "st",
            "source": "object_space_cylindrical atlas point_uv",
            "st_t_coordinate": "1.0 - atlas_v so USD samples the upper-left-authored PNG atlas correctly",
        },
        "atlas": atlas_result.to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return PbrMdlBundleExportResult(
        output_root=output_root,
        usd_path=usd_path,
        metadata_path=metadata_path,
        render_script_path=render_script_path,
        atlas_result=atlas_result,
        normal_map_path=normal_map_path,
        mesh_point_count=int(points.shape[0]),
        mesh_face_count=int(faces.shape[0]),
        material_variant=material_variant,
        material_map_path=material_map_path,
        coat_roughness_map_path=coat_roughness_map_path,
    )


def render_scope_video(
    *,
    mesh_path: Path,
    paths_json_path: Path,
    output_video_path: Path,
    preview_frame_path: Path,
    metadata_path: Path,
    path_id: str | None = None,
    fps: float | None = None,
    width_px: int | None = None,
    height_px: int | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    diagnostic_panels: bool = True,
    material_profile: str = "basic",
    material_variant: str = "healthy",
    material_map_path: Path | None = None,
    condition_output_root: Path | None = None,
    condition_manifest_path: Path | None = None,
) -> ScopeVideoResult:
    """Render a bronchoscope-style MP4 from a saved camera path."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when provided")

    payload = json.loads(paths_json_path.read_text(encoding="utf-8"))
    path_payload = _select_path(payload, path_id)
    frames = list(path_payload["frames"])[::frame_stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError("selected scope path has no frames to render")

    bronchoscope = _require_mapping(payload.get("bronchoscope"), "bronchoscope")
    intrinsics = _require_mapping(bronchoscope.get("intrinsics"), "bronchoscope.intrinsics")
    resolved_width = int(width_px or intrinsics["width_px"])
    resolved_height = int(height_px or intrinsics["height_px"])
    resolved_fps = float(fps or bronchoscope.get("frame_rate_fps", 10.0))
    if resolved_width < 16 or resolved_height < 16:
        raise ValueError("video dimensions must be at least 16x16")
    if resolved_fps <= 0:
        raise ValueError("fps must be positive")
    if material_profile not in MATERIAL_PROFILES:
        raise ValueError(f"material_profile must be one of {sorted(MATERIAL_PROFILES)}")
    if material_variant not in MATERIAL_VARIANTS:
        raise ValueError(f"material_variant must be one of {sorted(MATERIAL_VARIANTS)}")
    if material_map_path is not None:
        if material_profile != "pbr":
            raise ValueError("material_map_path requires material_profile='pbr'")
        if not material_map_path.is_file():
            raise ValueError(f"material map file does not exist: {material_map_path}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    preview_frame_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path_id = str(path_payload["path_id"])
    render_info = _render_frames_to_video(
        mesh_path=mesh_path,
        frames=frames,
        bronchoscope=bronchoscope,
        output_video_path=output_video_path,
        preview_frame_path=preview_frame_path,
        fps=resolved_fps,
        width_px=resolved_width,
        height_px=resolved_height,
        diagnostic_panels=diagnostic_panels,
        material_profile=material_profile,
        material_variant=material_variant,
        material_map_path=material_map_path,
        condition_output_root=condition_output_root,
        condition_sequence_id=_condition_sequence_id(selected_path_id),
    )
    condition_export = render_info.condition_export
    result = ScopeVideoResult(
        path_id=selected_path_id,
        output_video_path=output_video_path,
        preview_frame_path=preview_frame_path,
        metadata_path=metadata_path,
        frame_count=render_info.frame_count,
        fps=resolved_fps,
        width_px=render_info.width_px,
        height_px=render_info.height_px,
        panel_width_px=resolved_width,
        panel_height_px=resolved_height,
        diagnostic_panels=diagnostic_panels,
        condition_output_root=condition_export.output_root if condition_export is not None else None,
        condition_manifest_path=condition_manifest_path if condition_export is not None else None,
        condition_sequence_id=condition_export.sequence_id if condition_export is not None else None,
        condition_modalities=condition_export.modalities if condition_export is not None else (),
    )
    metadata = {
        "schema_version": "1.0",
        "source_mesh_path": str(mesh_path),
        "source_paths_json_path": str(paths_json_path),
        "bronchoscope": bronchoscope,
        "sampling": payload.get("sampling", {}),
        "path": {
            "path_id": path_payload["path_id"],
            "path_type": path_payload.get("path_type"),
            "entry_node_id": path_payload.get("entry_node_id"),
            "target_node_id": path_payload.get("target_node_id"),
            "target_branch_id": path_payload.get("target_branch_id"),
            "procedure_targets": path_payload.get("procedure_targets", []),
            "procedure_phases": path_payload.get("procedure_phases", []),
            "source_frame_count": path_payload.get("frame_count"),
            "rendered_frame_count": render_info.frame_count,
            "frame_stride": frame_stride,
            "max_frames": max_frames,
        },
        "diagnostics": {
            "enabled": diagnostic_panels,
            "layout": "3x2" if diagnostic_panels else "scope-only",
            "panels": (
                [
                    "scope_rgb",
                    "metric_depth_mm",
                    "source_ct_orthogonal_slices",
                    "view_relative_normals",
                    "per_pixel_shading_pps",
                    "isometric_scope_position",
                ]
                if diagnostic_panels
                else ["scope_rgb"]
            ),
            "source_ct_path": str(render_info.source_ct_path) if render_info.source_ct_path is not None else None,
            "depth_range_mm": list(render_info.depth_range_mm),
            "depth_no_hit_policy": "fill_with_far_depth_mm",
            "pps_range": list(render_info.pps_range) if render_info.pps_range is not None else None,
            "pps_model": (
                "normalized camera-light shading from surface normal incidence, endoscope beam falloff, "
                "and inverse-square-like distance attenuation"
                if diagnostic_panels
                else None
            ),
            "material_profile": material_profile,
            "material_variant": material_variant,
            "material_map_path": str(material_map_path) if material_map_path is not None else None,
            "material_model": (
                (
                    "anatomy-aware PBR mucosal material atlas plus authorable material map"
                    if material_map_path is not None
                    else "anatomy-aware PBR mucosal material atlas"
                )
                if material_profile == "pbr"
                else "deterministic mesh-attached vascular mucosa point texture"
                if material_profile == "vascular"
                else "legacy scalar mucosal variation colormap"
            ),
            "material_summary": render_info.material_summary,
            "renderer_material_support": (
                "PyVista split-region PBR actors: albedo remains per point, while roughness, specular, "
                "and specular power vary by extracted airway material region."
                if material_profile == "pbr"
                else None
            ),
        },
        "condition_export": (
            {
                "enabled": True,
                "output_root": str(condition_export.output_root),
                "manifest_path": str(condition_manifest_path) if condition_manifest_path is not None else None,
                "sequence_id": condition_export.sequence_id,
                "frame_count": condition_export.frame_count,
                "modalities": list(condition_export.modalities),
                "layout": "{modality}/{sequence_id}/{frame_index:06d}.png",
                "depth_encoding": "uint16 PNG, linear 0..depth_of_field_mm",
                "mask_encoding": "uint8 PNG, 0 background / 255 rendered airway hit",
                "pps_encoding": "uint8 RGB PNG, deterministic camera-light-relative shading",
            }
            if condition_export is not None
            else {"enabled": False}
        ),
        "artifacts": result.to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _render_frames_to_video(
    *,
    mesh_path: Path,
    frames: list[JsonValue],
    bronchoscope: dict[str, JsonValue],
    output_video_path: Path,
    preview_frame_path: Path,
    fps: float,
    width_px: int,
    height_px: int,
    diagnostic_panels: bool,
    material_profile: str,
    material_variant: str,
    material_map_path: Path | None,
    condition_output_root: Path | None,
    condition_sequence_id: str,
) -> _RenderInfo:
    try:
        import imageio.v2 as imageio  # type: ignore[import-not-found]
        import pyvista as pv  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("pyvista and imageio are required to render scope videos") from exc

    mesh = pv.read(str(mesh_path))
    material_summary: dict[str, JsonValue]
    if material_profile == "vascular":
        _add_mucosal_texture(mesh)
        material_scalars = "mucosal_texture_rgb"
        material_rgb = True
        material_cmap = None
        material_clim = None
        material_pbr = False
        material_metallic = None
        material_roughness = None
        material_ambient = 0.42
        material_diffuse = 0.78
        material_specular = 0.22
        material_specular_power = 28
        material_light_intensity = 0.95
        material_summary = {
            "channels": ["mucosal_texture_rgb"],
            "variant": material_variant,
            "region_model": "object-space vascular texture",
        }
    elif material_profile == "pbr":
        atlas = _add_mucosal_pbr_material(
            mesh,
            mesh_path=mesh_path,
            material_variant=material_variant,
            material_map_path=material_map_path,
        )
        material_scalars = "mucosal_pbr_albedo_rgb"
        material_rgb = True
        material_cmap = None
        material_clim = None
        material_pbr = True
        material_metallic = 0.0
        material_roughness = float(np.clip(np.mean(atlas.roughness), 0.18, 0.82))
        material_ambient = 0.22
        material_diffuse = 1.0
        material_specular = float(np.clip(np.mean(atlas.specular), 0.12, 0.7))
        material_specular_power = 36
        material_light_intensity = 1.35
        material_summary = atlas.summary
        pbr_region_specs = _add_pbr_cell_regions(mesh, atlas)
        material_summary["renderer_material_support"] = "split-region PBR actors"
        material_summary["region_actor_count"] = len(pbr_region_specs)
        material_summary["region_actors"] = [spec.to_dict() for spec in pbr_region_specs]
    else:
        _add_mucosal_variation(mesh)
        material_scalars = "mucosal_variation"
        material_rgb = False
        material_cmap = ["#6f302d", "#9e4944", "#bf6259", "#dd8173"]
        material_clim = (0.0, 1.0)
        material_pbr = False
        material_metallic = None
        material_roughness = None
        material_ambient = 0.34
        material_diffuse = 0.84
        material_specular = 0.12
        material_specular_power = 16
        material_light_intensity = 0.95
        material_summary = {
            "channels": ["mucosal_variation"],
            "variant": material_variant,
            "region_model": "legacy object-space scalar variation",
        }
    if material_profile != "pbr":
        pbr_region_specs: list[_PbrRegionActorSpec] = []
    normal_mesh, world_normals = _mesh_with_point_normals(mesh)
    pps_mesh = _mesh_with_pps(mesh)
    depth_range = _depth_range_mm(bronchoscope)
    condition_export = (
        _prepare_condition_export(condition_output_root, condition_sequence_id)
        if condition_output_root is not None
        else None
    )
    needs_control_renders = diagnostic_panels or condition_export is not None
    ct_context = _load_ct_context(mesh_path) if diagnostic_panels else None
    locator_context = _prepare_isometric_locator(
        np.asarray(mesh.points, dtype=np.float64),
        frames,
        width_px=width_px,
        height_px=height_px,
    ) if diagnostic_panels else None

    scope_plotter = pv.Plotter(off_screen=True, window_size=(width_px, height_px))
    scope_plotter.set_background("#120706")
    scope_plotter.add_light(pv.Light(light_type="headlight", intensity=material_light_intensity))
    if material_profile == "pbr" and pbr_region_specs:
        _add_pbr_region_actors(scope_plotter, mesh, pbr_region_specs, scalars=material_scalars)
    else:
        scope_plotter.add_mesh(
            mesh,
            scalars=material_scalars,
            rgb=material_rgb,
            cmap=material_cmap,
            clim=material_clim,
            ambient=material_ambient,
            diffuse=material_diffuse,
            specular=material_specular,
            specular_power=material_specular_power,
            pbr=material_pbr,
            metallic=material_metallic,
            roughness=material_roughness,
            smooth_shading=True,
            split_sharp_edges=material_pbr,
            show_scalar_bar=False,
        )
    normal_plotter: Any | None = None
    pps_plotter: Any | None = None
    if needs_control_renders:
        normal_mesh.point_data["view_normal_rgb"] = np.full((normal_mesh.n_points, 3), 128, dtype=np.uint8)
        normal_plotter = pv.Plotter(off_screen=True, window_size=(width_px, height_px))
        normal_plotter.set_background("#050607")
        normal_plotter.add_mesh(
            normal_mesh,
            scalars="view_normal_rgb",
            rgb=True,
            ambient=1.0,
            diffuse=0.0,
            specular=0.0,
            smooth_shading=True,
            show_scalar_bar=False,
        )
        pps_plotter = pv.Plotter(off_screen=True, window_size=(width_px, height_px))
        pps_plotter.set_background("#050607")
        pps_plotter.add_mesh(
            pps_mesh,
            scalars="pps_rgb",
            rgb=True,
            ambient=1.0,
            diffuse=0.0,
            specular=0.0,
            smooth_shading=True,
            show_scalar_bar=False,
        )
    vertical_fov = _vertical_fov_deg(_require_mapping(bronchoscope["intrinsics"], "intrinsics"))
    for plotter in (scope_plotter, normal_plotter, pps_plotter):
        if plotter is None:
            continue
        plotter.camera.view_angle = vertical_fov
        plotter.camera.clipping_range = (0.35, max(float(bronchoscope.get("depth_of_field_mm", [2.0, 50.0])[1]), 60.0))

    preview_index = min(max(len(frames) // 10, 0), len(frames) - 1)
    preview_frame: np.ndarray | None = None
    writer = imageio.get_writer(
        output_video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        rendered_count = 0
        for frame_index, frame in enumerate(frames):
            frame_mapping = _require_mapping(frame, "frame")
            _place_camera(scope_plotter, frame_mapping)
            image = np.asarray(scope_plotter.screenshot(return_img=True), dtype=np.uint8)
            processed = _apply_scope_view_mask(image)
            if needs_control_renders:
                raw_depth = _raw_depth_from_plotter(scope_plotter)
                depth_mm = _metric_depth_from_buffer(raw_depth, max_depth_mm=depth_range[1])
                hit_mask = _hit_mask_from_depth_buffer(raw_depth)
                depth_panel = _depth_to_rgb(depth_mm, max_depth_mm=depth_range[1])
                assert normal_plotter is not None
                assert pps_plotter is not None
                _update_view_normal_scalars(normal_mesh, _view_normal_rgb(world_normals, frame_mapping))
                _place_camera(normal_plotter, frame_mapping)
                normals_panel = _apply_scope_diagnostic_mask(
                    np.asarray(normal_plotter.screenshot(return_img=True), dtype=np.uint8)
                )
                _update_mesh_rgb_scalars(
                    pps_mesh,
                    "pps_rgb",
                    _pps_rgb(
                        np.asarray(pps_mesh.points, dtype=np.float64),
                        world_normals,
                        frame_mapping,
                        depth_of_field_mm=depth_range[1],
                    ),
                )
                _place_camera(pps_plotter, frame_mapping)
                pps_panel = _apply_scope_diagnostic_mask(
                    np.asarray(pps_plotter.screenshot(return_img=True), dtype=np.uint8)
                )
                if condition_export is not None:
                    _write_condition_frames(
                        imageio=imageio,
                        export=condition_export,
                        frame_index=frame_index,
                        scope_rgb=processed,
                        depth_mm=depth_mm,
                        max_depth_mm=depth_range[1],
                        normals_rgb=normals_panel,
                        pps_rgb=pps_panel,
                        hit_mask=hit_mask,
                    )
                if diagnostic_panels:
                    ct_panel = _render_ct_orthogonal_panel(
                        ct_context,
                        frame_mapping,
                        width_px=width_px,
                        height_px=height_px,
                    )
                    locator_panel = _render_isometric_locator(
                        locator_context,
                        frame_index,
                        width_px=width_px,
                        height_px=height_px,
                    )
                    processed = _compose_diagnostic_frame(
                        scope_rgb=processed,
                        depth_rgb=depth_panel,
                        normals_rgb=normals_panel,
                        pps_rgb=pps_panel,
                        ct_rgb=ct_panel,
                        locator_rgb=locator_panel,
                        max_depth_mm=depth_range[1],
                    )
            if frame_index == preview_index:
                preview_frame = processed
            writer.append_data(processed)
            rendered_count += 1
    finally:
        writer.close()
        scope_plotter.close()
        if normal_plotter is not None:
            normal_plotter.close()
        if pps_plotter is not None:
            pps_plotter.close()

    if preview_frame is None:
        raise ValueError("no scope frames were rendered")
    imageio.imwrite(preview_frame_path, preview_frame)
    return _RenderInfo(
        frame_count=rendered_count,
        width_px=int(preview_frame.shape[1]),
        height_px=int(preview_frame.shape[0]),
        depth_range_mm=depth_range,
        pps_range=(0.0, 1.0) if diagnostic_panels else None,
        source_ct_path=ct_context.source_ct_path if ct_context is not None else None,
        condition_export=(
            _ConditionExportInfo(
                output_root=condition_export.output_root,
                sequence_id=condition_export.sequence_id,
                frame_count=rendered_count,
                roots=condition_export.roots,
            )
            if condition_export is not None
            else None
        ),
        material_summary=material_summary,
    )


def _select_path(payload: dict[str, JsonValue], path_id: str | None) -> dict[str, JsonValue]:
    paths = payload.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("scope paths JSON does not contain any paths")
    for item in paths:
        if not isinstance(item, dict):
            continue
        if path_id is None or item.get("path_id") == path_id:
            return item
    raise ValueError(f"path_id {path_id!r} was not found in scope paths JSON")


def _add_mucosal_variation(mesh: Any) -> None:
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.size == 0:
        mesh.point_data["mucosal_variation"] = np.asarray([], dtype=np.float64)
        return
    normalized = points - points.mean(axis=0, keepdims=True)
    value = (
        0.48
        + 0.24 * np.sin(normalized[:, 0] * 0.18 + normalized[:, 2] * 0.07)
        + 0.18 * np.sin(normalized[:, 1] * 0.23 - normalized[:, 2] * 0.11)
        + 0.10 * np.cos((normalized[:, 0] + normalized[:, 1]) * 0.15)
    )
    mesh.point_data["mucosal_variation"] = np.asarray(np.clip(value, 0.0, 1.0), dtype=np.float64)


def _add_mucosal_texture(mesh: Any) -> None:
    mesh.point_data["mucosal_texture_rgb"] = _mucosal_texture_rgb(np.asarray(mesh.points, dtype=np.float64))


def _add_mucosal_pbr_material(
    mesh: Any,
    *,
    mesh_path: Path,
    material_variant: str,
    material_map_path: Path | None = None,
) -> _MucosalMaterialAtlas:
    atlas = _mucosal_pbr_material_atlas(
        np.asarray(mesh.points, dtype=np.float64),
        centerline_nodes=_load_centerline_material_nodes(mesh_path),
        material_variant=material_variant,
        material_map=_load_pbr_material_map(material_map_path),
        material_map_source=str(material_map_path) if material_map_path is not None else None,
    )
    mesh.point_data["mucosal_pbr_albedo_rgb"] = atlas.albedo_rgb
    mesh.point_data["mucosal_pbr_roughness"] = atlas.roughness
    mesh.point_data["mucosal_pbr_specular"] = atlas.specular
    mesh.point_data["mucosal_pbr_wetness"] = atlas.wetness
    mesh.point_data["mucosal_pbr_vascularity"] = atlas.vascularity
    mesh.point_data["mucosal_pbr_mucus"] = atlas.mucus
    mesh.point_data["mucosal_pbr_mucus_shininess"] = atlas.mucus_shininess
    mesh.point_data["mucosal_pbr_erythema"] = atlas.erythema
    mesh.point_data["mucosal_pbr_petechiae"] = atlas.petechiae
    mesh.point_data["mucosal_pbr_stain"] = atlas.stain
    mesh.point_data["mucosal_pbr_bump_height"] = atlas.bump_height
    mesh.point_data["mucosal_pbr_displacement_height"] = atlas.displacement_height
    mesh.point_data["mucosal_pbr_region_id"] = atlas.region_id
    mesh.point_data["mucosal_pbr_generation"] = atlas.generation
    return atlas


def _mucosal_texture_rgb(points: np.ndarray) -> np.ndarray:
    """Generate a deterministic mesh-attached airway mucosa point texture."""
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    centered = points - points.mean(axis=0, keepdims=True)
    span = np.maximum(np.ptp(centered, axis=0), 1.0)
    normalized = centered / span[np.newaxis, :]
    x = normalized[:, 0]
    y = normalized[:, 1]
    z = normalized[:, 2]
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    broad = (
        0.45
        + 0.23 * np.sin(22.0 * x + 9.0 * z)
        + 0.18 * np.sin(18.0 * y - 7.0 * z + 0.6)
        + 0.12 * np.cos(10.0 * (x + y) + 11.0 * radius)
    )
    fine = (
        0.5
        + 0.5
        * np.sin(
            95.0 * x
            + 71.0 * y
            + 33.0 * z
            + 0.8 * np.sin(8.0 * theta + 17.0 * z)
        )
    )
    folds = 0.5 + 0.5 * np.sin(7.0 * theta + 24.0 * z + 2.5 * np.sin(5.0 * radius))
    vessel_a = np.exp(-((np.sin(4.0 * theta + 17.0 * z) / 0.24) ** 2)) * (0.45 + 0.55 * folds)
    vessel_b = np.exp(-((np.sin(7.0 * theta - 11.0 * z + 1.3) / 0.18) ** 2)) * (0.25 + 0.75 * (1.0 - folds))
    vessel = np.clip(np.maximum(vessel_a, vessel_b) * (0.35 + 0.65 * np.clip(radius * 2.0, 0.0, 1.0)), 0.0, 1.0)
    glint_seeds = (
        np.sin(141.0 * x + 29.0 * y + 53.0 * z)
        + np.cos(61.0 * x - 107.0 * y + 19.0 * z)
    )
    glints = np.clip((glint_seeds - 1.45) / 0.55, 0.0, 1.0) ** 3

    base = np.column_stack(
        (
            145.0 + 46.0 * broad + 16.0 * fine,
            57.0 + 24.0 * broad + 10.0 * fine,
            52.0 + 18.0 * broad + 8.0 * fine,
        )
    )
    base[:, 0] += 18.0 * folds
    base[:, 1] += 7.0 * folds
    base[:, 0] -= 35.0 * vessel
    base[:, 1] -= 31.0 * vessel
    base[:, 2] -= 18.0 * vessel
    base += glints[:, np.newaxis] * np.asarray([60.0, 46.0, 36.0], dtype=np.float64)
    return np.asarray(np.clip(base, 0, 255), dtype=np.uint8)


def _mucosal_pbr_material_atlas(
    points: np.ndarray,
    *,
    centerline_nodes: _CenterlineMaterialNodes | None = None,
    material_variant: str = "healthy",
    material_map: dict[str, JsonValue] | None = None,
    material_map_source: str | None = None,
) -> _MucosalMaterialAtlas:
    """Generate anatomy-aware airway material channels for PBR-style rendering."""
    if material_variant not in MATERIAL_VARIANTS:
        raise ValueError(f"material_variant must be one of {sorted(MATERIAL_VARIANTS)}")
    if points.shape[0] == 0:
        empty_float = np.asarray([], dtype=np.float32)
        empty_int = np.asarray([], dtype=np.int16)
        return _MucosalMaterialAtlas(
            albedo_rgb=np.zeros((0, 3), dtype=np.uint8),
            roughness=empty_float,
            specular=empty_float,
            wetness=empty_float,
            vascularity=empty_float,
            mucus=empty_float,
            mucus_shininess=empty_float,
            erythema=empty_float,
            petechiae=empty_float,
            stain=empty_float,
            bump_height=empty_float,
            displacement_height=empty_float,
            region_id=empty_int,
            generation=empty_int,
            summary=_material_summary(
                material_variant,
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_float,
                empty_int,
                empty_int,
                False,
                None,
            ),
        )

    generation, distance_norm, radius_norm, used_centerline = _material_anatomy_features(points, centerline_nodes)
    region_id = _material_region_ids(generation, radius_norm)
    region_weights = _material_region_blend_weights(
        generation,
        distance_norm,
        radius_norm,
        softness=PBR_MATERIAL_REGION_BLEND_SOFTNESS,
    )
    proximal_w = region_weights[:, 0]
    main_w = region_weights[:, 1]
    segmental_w = region_weights[:, 2]
    distal_w = region_weights[:, 3]
    vascular_rgb = _mucosal_texture_rgb(points).astype(np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    span = np.maximum(np.ptp(centered, axis=0), 1.0)
    normalized = centered / span[np.newaxis, :]
    x = normalized[:, 0]
    y = normalized[:, 1]
    z = normalized[:, 2]
    theta = np.arctan2(y, x)

    lobe_jitter = 0.5 + 0.5 * np.sin(
        5.5 * np.sign(x) + 8.0 * distance_norm + 1.7 * generation + 2.0 * np.sin(4.0 * theta)
    )
    side_tint = np.sign(x)[:, np.newaxis] * np.asarray([5.5, -2.0, 3.5], dtype=np.float32)
    lobe_tint = (lobe_jitter - 0.5)[:, np.newaxis] * np.asarray([15.0, -4.5, 6.5], dtype=np.float32)
    part_tint = PBR_MATERIAL_REGION_TINT_STRENGTH * (
        proximal_w[:, np.newaxis] * np.asarray([18.0, 10.0, 6.0], dtype=np.float32)
        + main_w[:, np.newaxis] * np.asarray([10.0, 2.0, 1.0], dtype=np.float32)
        + segmental_w[:, np.newaxis] * np.asarray([4.0, -2.0, 0.0], dtype=np.float32)
        + distal_w[:, np.newaxis] * np.asarray([-8.0, -5.0, 5.0], dtype=np.float32)
    )

    mottling = 0.5 + 0.5 * np.sin(37.0 * x - 19.0 * y + 11.0 * z + 0.4 * np.sin(9.0 * theta))
    vascularity = np.clip(
        0.24
        + PBR_MATERIAL_REGION_TINT_STRENGTH * (0.22 * segmental_w + 0.30 * distal_w)
        + 0.18 * (1.0 - radius_norm)
        + 0.15 * mottling,
        0.0,
        1.0,
    )
    wetness = np.clip(
        0.36
        + PBR_MATERIAL_REGION_TINT_STRENGTH * (0.16 * main_w + 0.14 * segmental_w)
        + 0.08 * np.sin(13.0 * z + 5.0 * theta)
        + 0.10 * distance_norm,
        0.0,
        1.0,
    )
    cartilage_banding = proximal_w * (0.5 + 0.5 * np.sin(30.0 * z + 5.0 * theta))

    albedo = vascular_rgb + side_tint + part_tint
    albedo[:, 0] += 20.0 * vascularity
    albedo[:, 1] -= 12.0 * vascularity
    albedo[:, 2] -= 8.0 * vascularity
    albedo += cartilage_banding[:, np.newaxis] * np.asarray([22.0, 15.0, 10.0], dtype=np.float32)

    roughness = np.clip(0.58 - 0.22 * wetness + 0.08 * proximal_w + 0.05 * mottling, 0.18, 0.82)
    specular = np.clip(0.18 + 0.38 * wetness + 0.11 * vascularity, 0.08, 0.78)

    albedo, roughness, specular, wetness, vascularity = _apply_material_variant(
        albedo + lobe_tint,
        roughness,
        specular,
        wetness,
        vascularity,
        material_variant=material_variant,
    )
    mucus, erythema, petechiae, stain = _mucosal_local_pbr_signals(
        x=x,
        y=y,
        z=z,
        theta=theta,
        generation=generation,
        distance_norm=distance_norm,
        radius_norm=radius_norm,
        region_weights=region_weights,
        material_variant=material_variant,
    )
    fine_relief = 0.5 + 0.5 * np.sin(64.0 * x - 41.0 * y + 29.0 * z + 0.7 * np.sin(12.0 * theta))
    fold_relief = 0.5 + 0.5 * np.sin(9.0 * theta + 20.0 * z + 2.2 * np.sin(5.0 * distance_norm))
    bump_height = np.clip(
        0.14
        + 0.22 * fold_relief
        + 0.18 * mottling
        + 0.24 * petechiae
        + 0.16 * stain
        + 0.10 * erythema
        + 0.08 * fine_relief
        - 0.12 * mucus,
        0.0,
        1.0,
    )
    displacement_height = np.clip(
        0.10
        + 0.18 * fold_relief
        + 0.20 * cartilage_banding
        + 0.34 * mucus
        + 0.08 * erythema
        + 0.10 * stain,
        0.0,
        1.0,
    )
    mucus_shininess = np.clip(0.10 + 0.62 * mucus + 0.22 * wetness + 0.10 * specular - 0.10 * stain, 0.0, 1.0)
    material_map_summary = None
    if material_map is not None:
        (
            albedo,
            roughness,
            specular,
            wetness,
            vascularity,
            mucus,
            mucus_shininess,
            erythema,
            petechiae,
            stain,
            bump_height,
            displacement_height,
            material_map_summary,
        ) = _apply_pbr_material_map(
            albedo=albedo,
            roughness=roughness,
            specular=specular,
            wetness=wetness,
            vascularity=vascularity,
            mucus=mucus,
            mucus_shininess=mucus_shininess,
            erythema=erythema,
            petechiae=petechiae,
            stain=stain,
            bump_height=bump_height,
            displacement_height=displacement_height,
            normalized_xyz=normalized,
            region_id=region_id,
            generation=generation,
            distance_norm=distance_norm,
            radius_norm=radius_norm,
            material_map=material_map,
            material_map_source=material_map_source,
        )
    albedo = albedo * (1.0 - 0.24 * stain[:, np.newaxis]) + stain[:, np.newaxis] * np.asarray(
        [118.0, 84.0, 60.0],
        dtype=np.float32,
    )
    albedo += mucus[:, np.newaxis] * np.asarray([42.0, 34.0, 18.0], dtype=np.float32)
    albedo += erythema[:, np.newaxis] * np.asarray([58.0, -24.0, -27.0], dtype=np.float32)
    albedo += petechiae[:, np.newaxis] * np.asarray([72.0, -58.0, -48.0], dtype=np.float32)

    mucus_shininess = np.clip(mucus_shininess + 0.18 * mucus + 0.08 * wetness - 0.08 * stain, 0.0, 1.0)
    wetness = np.clip(wetness + 0.30 * mucus + 0.10 * erythema - 0.13 * stain, 0.0, 1.0)
    specular = np.clip(specular + 0.24 * mucus + 0.18 * mucus_shininess + 0.08 * erythema - 0.16 * stain, 0.0, 1.0)
    roughness = np.clip(
        roughness - 0.16 * mucus - 0.10 * mucus_shininess - 0.05 * erythema + 0.07 * petechiae + 0.26 * stain,
        0.0,
        1.0,
    )
    vascularity = np.clip(vascularity + 0.30 * erythema + 0.22 * petechiae - 0.10 * stain, 0.0, 1.0)
    generation_i = np.asarray(np.round(generation), dtype=np.int16)
    region_i = np.asarray(region_id, dtype=np.int16)
    return _MucosalMaterialAtlas(
        albedo_rgb=np.asarray(np.clip(albedo, 0, 255), dtype=np.uint8),
        roughness=np.asarray(np.clip(roughness, 0.0, 1.0), dtype=np.float32),
        specular=np.asarray(np.clip(specular, 0.0, 1.0), dtype=np.float32),
        wetness=np.asarray(np.clip(wetness, 0.0, 1.0), dtype=np.float32),
        vascularity=np.asarray(np.clip(vascularity, 0.0, 1.0), dtype=np.float32),
        mucus=np.asarray(np.clip(mucus, 0.0, 1.0), dtype=np.float32),
        mucus_shininess=np.asarray(np.clip(mucus_shininess, 0.0, 1.0), dtype=np.float32),
        erythema=np.asarray(np.clip(erythema, 0.0, 1.0), dtype=np.float32),
        petechiae=np.asarray(np.clip(petechiae, 0.0, 1.0), dtype=np.float32),
        stain=np.asarray(np.clip(stain, 0.0, 1.0), dtype=np.float32),
        bump_height=np.asarray(np.clip(bump_height, 0.0, 1.0), dtype=np.float32),
        displacement_height=np.asarray(np.clip(displacement_height, 0.0, 1.0), dtype=np.float32),
        region_id=region_i,
        generation=generation_i,
        summary=_material_summary(
            material_variant,
            roughness,
            specular,
            wetness,
            vascularity,
            mucus,
            mucus_shininess,
            erythema,
            petechiae,
            stain,
            bump_height,
            displacement_height,
            region_i,
            generation_i,
            used_centerline,
            material_map_summary,
            region_blend_softness=PBR_MATERIAL_REGION_BLEND_SOFTNESS,
            region_tint_strength=PBR_MATERIAL_REGION_TINT_STRENGTH,
        ),
    )


def _add_pbr_cell_regions(mesh: Any, atlas: _MucosalMaterialAtlas) -> list[_PbrRegionActorSpec]:
    cell_region_id = _pbr_cell_region_ids(mesh, atlas.region_id)
    mesh.cell_data["mucosal_pbr_cell_region_id"] = cell_region_id
    return _pbr_region_actor_specs(atlas, cell_region_id)


def _add_pbr_region_actors(
    plotter: Any,
    mesh: Any,
    specs: list[_PbrRegionActorSpec],
    *,
    scalars: str,
) -> None:
    for spec in specs:
        cell_ids = np.flatnonzero(np.asarray(mesh.cell_data["mucosal_pbr_cell_region_id"]) == spec.region_id)
        if cell_ids.size == 0:
            continue
        region_mesh = mesh.extract_cells(cell_ids)
        plotter.add_mesh(
            region_mesh,
            scalars=scalars,
            rgb=True,
            ambient=spec.ambient,
            diffuse=spec.diffuse,
            specular=spec.specular,
            specular_power=spec.specular_power,
            pbr=True,
            metallic=spec.metallic,
            roughness=spec.roughness,
            smooth_shading=True,
            split_sharp_edges=True,
            show_scalar_bar=False,
            name=f"pbr-region-{spec.region_id}",
        )


def _pbr_cell_region_ids(mesh: Any, point_region_id: np.ndarray) -> np.ndarray:
    point_region_id = np.asarray(point_region_id, dtype=np.int16)
    faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int64)
    cell_regions: list[int] = []
    index = 0
    while index < faces.size:
        point_count = int(faces[index])
        ids = faces[index + 1 : index + 1 + point_count]
        if ids.size:
            values = point_region_id[np.clip(ids, 0, len(point_region_id) - 1)]
            counts = np.bincount(values.astype(np.int64), minlength=4)
            cell_regions.append(int(np.argmax(counts)))
        index += point_count + 1
    if len(cell_regions) == int(getattr(mesh, "n_cells", 0)):
        return np.asarray(cell_regions, dtype=np.int16)
    if point_region_id.size == 0:
        return np.zeros(int(getattr(mesh, "n_cells", 0)), dtype=np.int16)
    counts = np.bincount(point_region_id.astype(np.int64), minlength=4)
    fallback_region = int(np.argmax(counts))
    return np.full(int(getattr(mesh, "n_cells", 0)), fallback_region, dtype=np.int16)


def _pbr_region_actor_specs(
    atlas: _MucosalMaterialAtlas,
    cell_region_id: np.ndarray,
) -> list[_PbrRegionActorSpec]:
    specs: list[_PbrRegionActorSpec] = []
    for region_id in range(4):
        cell_count = int(np.count_nonzero(cell_region_id == region_id))
        point_mask = atlas.region_id == region_id
        point_count = int(np.count_nonzero(point_mask))
        if cell_count == 0 or point_count == 0:
            continue
        roughness_mean = float(np.mean(atlas.roughness[point_mask]))
        specular_mean = float(np.mean(atlas.specular[point_mask]))
        wetness_mean = float(np.mean(atlas.wetness[point_mask]))
        vascularity_mean = float(np.mean(atlas.vascularity[point_mask]))
        mucus_shininess_mean = float(np.mean(atlas.mucus_shininess[point_mask]))
        bump_height_mean = float(np.mean(atlas.bump_height[point_mask]))
        displacement_height_mean = float(np.mean(atlas.displacement_height[point_mask]))
        actor_specular = float(
            np.clip(0.10 + 0.48 * specular_mean + 0.12 * wetness_mean + 0.22 * mucus_shininess_mean, 0.12, 0.9)
        )
        specs.append(
            _PbrRegionActorSpec(
                region_id=region_id,
                region_name=_pbr_region_name(region_id),
                cell_count=cell_count,
                point_count=point_count,
                roughness=round(float(np.clip(roughness_mean, 0.16, 0.86)), 4),
                specular=round(actor_specular, 4),
                specular_power=round(
                    float(20.0 + 38.0 * (1.0 - roughness_mean) + 20.0 * wetness_mean + 24.0 * mucus_shininess_mean),
                    4,
                ),
                wetness_mean=round(wetness_mean, 4),
                vascularity_mean=round(vascularity_mean, 4),
                mucus_shininess_mean=round(mucus_shininess_mean, 4),
                bump_height_mean=round(bump_height_mean, 4),
                displacement_height_mean=round(displacement_height_mean, 4),
                ambient=round(float(0.18 + 0.05 * (region_id == 0) + 0.03 * (region_id == 3)), 4),
            )
        )
    return specs


def _pbr_region_name(region_id: int) -> str:
    names = {
        0: "tracheal_or_proximal_airway",
        1: "main_bronchus",
        2: "lobar_or_segmental_bronchus",
        3: "distal_airway",
    }
    return names.get(region_id, f"region_{region_id}")


def _apply_material_variant(
    albedo: np.ndarray,
    roughness: np.ndarray,
    specular: np.ndarray,
    wetness: np.ndarray,
    vascularity: np.ndarray,
    *,
    material_variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if material_variant == "healthy":
        return albedo, roughness, specular, wetness, vascularity
    if material_variant == "inflamed":
        albedo = albedo + np.asarray([34.0, -10.0, -12.0], dtype=np.float32)
        vascularity = np.clip(vascularity * 1.42 + 0.12, 0.0, 1.0)
        wetness = np.clip(wetness * 1.18 + 0.08, 0.0, 1.0)
        specular = np.clip(specular * 1.12 + 0.06, 0.0, 1.0)
        roughness = np.clip(roughness - 0.08, 0.0, 1.0)
    elif material_variant == "smoker":
        albedo = albedo * np.asarray([0.82, 0.78, 0.70], dtype=np.float32) + np.asarray(
            [24.0, 16.0, 10.0],
            dtype=np.float32,
        )
        vascularity = np.clip(vascularity * 0.85 + 0.06, 0.0, 1.0)
        wetness = np.clip(wetness * 0.86, 0.0, 1.0)
        specular = np.clip(specular * 0.82, 0.0, 1.0)
        roughness = np.clip(roughness + 0.12, 0.0, 1.0)
    elif material_variant == "pale":
        albedo = albedo * np.asarray([1.08, 1.05, 1.02], dtype=np.float32) + np.asarray(
            [16.0, 12.0, 10.0],
            dtype=np.float32,
        )
        vascularity = np.clip(vascularity * 0.62, 0.0, 1.0)
        wetness = np.clip(wetness * 0.9, 0.0, 1.0)
        specular = np.clip(specular * 0.92, 0.0, 1.0)
        roughness = np.clip(roughness + 0.04, 0.0, 1.0)
    elif material_variant == "edematous":
        albedo = albedo * np.asarray([1.0, 0.94, 0.96], dtype=np.float32) + np.asarray(
            [20.0, 2.0, 6.0],
            dtype=np.float32,
        )
        vascularity = np.clip(vascularity * 1.15, 0.0, 1.0)
        wetness = np.clip(wetness * 1.35 + 0.12, 0.0, 1.0)
        specular = np.clip(specular * 1.25 + 0.10, 0.0, 1.0)
        roughness = np.clip(roughness - 0.14, 0.0, 1.0)
    return albedo, roughness, specular, wetness, vascularity


def _mucosal_local_pbr_signals(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    theta: np.ndarray,
    generation: np.ndarray,
    distance_norm: np.ndarray,
    radius_norm: np.ndarray,
    region_weights: np.ndarray,
    material_variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proximal = np.asarray(region_weights[:, 0], dtype=np.float32)
    segmental = np.asarray(region_weights[:, 2], dtype=np.float32)
    distal = np.asarray(region_weights[:, 3], dtype=np.float32)
    folds = 0.5 + 0.5 * np.sin(8.0 * theta + 22.0 * z + 2.2 * np.sin(5.0 * x))
    pool_blobs = _smoothstep(0.5 + 0.5 * np.sin(12.0 * x - 7.0 * y + 15.0 * z), 0.50, 0.95)
    pool_folds = np.exp(-((np.sin(3.3 * theta + 12.0 * z + 1.6 * distance_norm) / 0.34) ** 2))
    mucus_seed = np.clip(
        0.18 * pool_blobs
        + 0.28 * pool_folds * folds
        + 0.20 * distance_norm
        + 0.18 * (1.0 - radius_norm)
        + 0.16 * distal
        + 0.08 * segmental,
        0.0,
        1.0,
    )

    patch_blobs = _smoothstep(
        0.5 + 0.5 * np.sin(5.0 * theta - 14.0 * z + 2.5 * np.sin(6.0 * x)),
        0.46,
        0.92,
    )
    vessel_lines = np.exp(-((np.sin(5.2 * theta + 18.0 * z + 0.3 * generation) / 0.23) ** 2))
    erythema_seed = np.clip(
        0.26 * patch_blobs
        + 0.26 * vessel_lines
        + 0.18 * distance_norm
        + 0.18 * segmental
        + 0.18 * distal,
        0.0,
        1.0,
    )

    speckle_a = _smoothstep(0.5 + 0.5 * np.sin(191.0 * x + 107.0 * y - 139.0 * z), 0.68, 0.96)
    speckle_b = _smoothstep(
        0.5 + 0.5 * np.cos(83.0 * x - 173.0 * y + 71.0 * z + 5.0 * np.sin(theta)),
        0.62,
        0.94,
    )
    petechiae_seed = np.clip(speckle_a * speckle_b * (0.42 + 0.36 * erythema_seed + 0.22 * distal), 0.0, 1.0)

    ash_blobs = _smoothstep(
        0.5 + 0.5 * np.sin(4.8 * x + 3.4 * y + 7.6 * z + 1.2 * np.sin(2.5 * theta)),
        0.34,
        0.96,
    )
    tar_lines = np.exp(-((np.sin(3.6 * theta - 8.5 * z + 0.9 * x) / 0.52) ** 2))
    smoke_wash = 0.5 + 0.5 * np.sin(2.3 * x - 1.9 * y + 3.1 * z + 0.7 * np.sin(1.8 * theta))
    stain_seed = np.clip(
        0.25 * ash_blobs
        + 0.18 * tar_lines
        + 0.16 * smoke_wash * (0.45 + 0.55 * proximal)
        + 0.14 * proximal
        + 0.10 * (1.0 - distance_norm)
        + 0.05 * folds,
        0.0,
        1.0,
    )

    profile = {
        "healthy": (0.32, 0.00, 0.32, 0.00, 0.18, 0.00, 0.08, 0.00),
        "inflamed": (0.58, 0.03, 1.20, 0.10, 1.05, 0.02, 0.08, 0.00),
        "smoker": (0.25, 0.00, 0.44, 0.02, 0.56, 0.01, 0.62, 0.04),
        "pale": (0.20, 0.00, 0.16, 0.00, 0.12, 0.00, 0.04, 0.00),
        "edematous": (1.10, 0.10, 0.64, 0.02, 0.34, 0.00, 0.04, 0.00),
    }[material_variant]
    mucus_gain, mucus_bias, erythema_gain, erythema_bias, petechiae_gain, petechiae_bias, stain_gain, stain_bias = (
        profile
    )
    mucus = np.clip(mucus_seed * mucus_gain + mucus_bias * (0.45 + 0.55 * distance_norm), 0.0, 1.0)
    erythema = np.clip(erythema_seed * erythema_gain + erythema_bias, 0.0, 1.0)
    petechiae = np.clip(petechiae_seed * petechiae_gain + petechiae_bias * erythema_seed, 0.0, 1.0)
    stain = np.clip(stain_seed * stain_gain + stain_bias * (0.55 + 0.45 * proximal), 0.0, 1.0)
    return (
        np.asarray(mucus, dtype=np.float32),
        np.asarray(erythema, dtype=np.float32),
        np.asarray(petechiae, dtype=np.float32),
        np.asarray(stain, dtype=np.float32),
    )


def _load_pbr_material_map(material_map_path: Path | None) -> dict[str, JsonValue] | None:
    if material_map_path is None:
        return None
    payload = json.loads(material_map_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("material map must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version not in (None, "1.0"):
        raise ValueError("material map schema_version must be '1.0'")
    return payload


def _apply_pbr_material_map(
    *,
    albedo: np.ndarray,
    roughness: np.ndarray,
    specular: np.ndarray,
    wetness: np.ndarray,
    vascularity: np.ndarray,
    mucus: np.ndarray,
    mucus_shininess: np.ndarray,
    erythema: np.ndarray,
    petechiae: np.ndarray,
    stain: np.ndarray,
    bump_height: np.ndarray,
    displacement_height: np.ndarray,
    normalized_xyz: np.ndarray,
    region_id: np.ndarray,
    generation: np.ndarray,
    distance_norm: np.ndarray,
    radius_norm: np.ndarray,
    material_map: dict[str, JsonValue],
    material_map_source: str | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, JsonValue],
]:
    fields = {
        "roughness": np.asarray(roughness, dtype=np.float32).copy(),
        "specular": np.asarray(specular, dtype=np.float32).copy(),
        "wetness": np.asarray(wetness, dtype=np.float32).copy(),
        "vascularity": np.asarray(vascularity, dtype=np.float32).copy(),
        "mucus": np.asarray(mucus, dtype=np.float32).copy(),
        "mucus_shininess": np.asarray(mucus_shininess, dtype=np.float32).copy(),
        "erythema": np.asarray(erythema, dtype=np.float32).copy(),
        "petechiae": np.asarray(petechiae, dtype=np.float32).copy(),
        "stain": np.asarray(stain, dtype=np.float32).copy(),
        "bump_height": np.asarray(bump_height, dtype=np.float32).copy(),
        "displacement_height": np.asarray(displacement_height, dtype=np.float32).copy(),
    }
    mapped_albedo = np.asarray(albedo, dtype=np.float32).copy()
    applied_regions = _apply_pbr_material_map_entries(
        entries=material_map.get("regions"),
        normalized_xyz=normalized_xyz,
        region_id=region_id,
        generation=generation,
        distance_norm=distance_norm,
        radius_norm=radius_norm,
        fields=fields,
        albedo=mapped_albedo,
        entry_kind="region",
    )
    applied_spots = _apply_pbr_material_map_entries(
        entries=material_map.get("spots"),
        normalized_xyz=normalized_xyz,
        region_id=region_id,
        generation=generation,
        distance_norm=distance_norm,
        radius_norm=radius_norm,
        fields=fields,
        albedo=mapped_albedo,
        entry_kind="spot",
    )
    summary = {
        "schema_version": str(material_map.get("schema_version", "1.0")),
        "name": str(material_map.get("name", "unnamed-material-map")),
        "description": str(material_map.get("description", "")),
        "source": material_map_source,
        "region_entry_count": len(applied_regions),
        "spot_count": len(applied_spots),
        "applied_regions": applied_regions,
        "applied_spots": applied_spots,
    }
    surface_overlays = material_map.get("surface_overlays")
    if isinstance(surface_overlays, list):
        summary["surface_overlay_count"] = len([entry for entry in surface_overlays if isinstance(entry, dict)])
        summary["surface_overlays"] = surface_overlays
    return (
        mapped_albedo,
        np.clip(fields["roughness"], 0.0, 1.0),
        np.clip(fields["specular"], 0.0, 1.0),
        np.clip(fields["wetness"], 0.0, 1.0),
        np.clip(fields["vascularity"], 0.0, 1.0),
        np.clip(fields["mucus"], 0.0, 1.0),
        np.clip(fields["mucus_shininess"], 0.0, 1.0),
        np.clip(fields["erythema"], 0.0, 1.0),
        np.clip(fields["petechiae"], 0.0, 1.0),
        np.clip(fields["stain"], 0.0, 1.0),
        np.clip(fields["bump_height"], 0.0, 1.0),
        np.clip(fields["displacement_height"], 0.0, 1.0),
        summary,
    )


def _apply_pbr_material_map_entries(
    *,
    entries: JsonValue,
    normalized_xyz: np.ndarray,
    region_id: np.ndarray,
    generation: np.ndarray,
    distance_norm: np.ndarray,
    radius_norm: np.ndarray,
    fields: dict[str, np.ndarray],
    albedo: np.ndarray,
    entry_kind: str,
) -> list[dict[str, JsonValue]]:
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(f"material map {entry_kind}s must be a list")
    applied: list[dict[str, JsonValue]] = []
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"material map {entry_kind} entry {index} must be an object")
        weight = _pbr_material_map_weight(
            raw_entry,
            normalized_xyz,
            region_id,
            generation,
            distance_norm,
            radius_norm,
            entry_kind=entry_kind,
        )
        if not np.any(weight > 1.0e-4):
            continue
        for channel, offset in _material_map_float_mapping(raw_entry.get("channels"), "channels").items():
            fields[channel] = fields[channel] + weight * offset
        for channel, multiplier in _material_map_float_mapping(raw_entry.get("multipliers"), "multipliers").items():
            fields[channel] = fields[channel] * (1.0 + weight * (multiplier - 1.0))
        albedo_delta = _material_map_albedo_delta(raw_entry.get("albedo_delta_rgb"))
        if albedo_delta is not None:
            albedo += weight[:, np.newaxis] * albedo_delta[np.newaxis, :]
        applied.append(
            {
                "name": str(raw_entry.get("name", f"{entry_kind}_{index}")),
                "selected_point_count": int(np.count_nonzero(weight > 1.0e-4)),
                "peak_weight": round(float(np.max(weight)), 4),
                "mean_weight": round(float(np.mean(weight)), 4),
            }
        )
    return applied


def _pbr_material_map_weight(
    entry: dict[str, JsonValue],
    normalized_xyz: np.ndarray,
    region_id: np.ndarray,
    generation: np.ndarray,
    distance_norm: np.ndarray,
    radius_norm: np.ndarray,
    *,
    entry_kind: str,
) -> np.ndarray:
    region_weight = _pbr_material_map_region_weight(entry, region_id, generation, distance_norm, radius_norm)
    if entry_kind == "region":
        weight = region_weight
    else:
        center_value = entry.get("center_normalized_xyz")
        if not isinstance(center_value, list) or len(center_value) != 3:
            raise ValueError("material map spot entries require center_normalized_xyz with three numbers")
        center = np.asarray([_as_float(value, "center_normalized_xyz") for value in center_value], dtype=np.float32)
        radius = max(_as_float(entry.get("radius", 0.2), "radius"), 1.0e-6)
        falloff = max(_as_float(entry.get("falloff", 1.0), "falloff"), 1.0e-6)
        distance2 = np.sum((normalized_xyz.astype(np.float32) - center[np.newaxis, :]) ** 2, axis=1)
        weight = np.exp(-distance2 / ((radius * falloff) * (radius * falloff))).astype(np.float32) * region_weight

    opacity = np.clip(_as_float(entry.get("opacity", 1.0), "opacity"), 0.0, 1.0)
    weight = np.asarray(weight * opacity, dtype=np.float32)
    texture_strength = np.clip(
        _as_float(entry.get("texture_noise_strength", entry.get("noise_strength", 0.0)), "texture_noise_strength"),
        0.0,
        1.0,
    )
    if texture_strength > 0.0:
        modulation = _pbr_material_map_texture_modulation(entry, normalized_xyz)
        weight = weight * (1.0 - texture_strength + texture_strength * modulation)
    return np.asarray(np.clip(weight, 0.0, 1.0), dtype=np.float32)


def _pbr_material_map_region_weight(
    entry: dict[str, JsonValue],
    region_id: np.ndarray,
    generation: np.ndarray,
    distance_norm: np.ndarray,
    radius_norm: np.ndarray,
) -> np.ndarray:
    hard_mask = _pbr_material_map_region_mask(entry, region_id)
    edge_softness = max(
        _as_float(entry.get("edge_softness", entry.get("soft_region_edges", 0.0)), "edge_softness"),
        0.0,
    )
    if edge_softness <= 0.0:
        return hard_mask.astype(np.float32)
    selector = entry.get("regions", entry.get("region_ids", entry.get("region")))
    if selector is None:
        return np.ones(region_id.shape, dtype=np.float32)
    selectors = selector if isinstance(selector, list) else [selector]
    if any(str(value).lower() == "all" for value in selectors):
        return np.ones(region_id.shape, dtype=np.float32)
    weights = np.zeros(region_id.shape, dtype=np.float32)
    for value in selectors:
        weights = np.maximum(
            weights,
            _soft_pbr_region_weight(
                _pbr_region_id_from_selector(value),
                generation,
                distance_norm,
                radius_norm,
                edge_softness,
            ),
        )
    return np.asarray(np.clip(weights, 0.0, 1.0), dtype=np.float32)


def _soft_pbr_region_weight(
    region: int,
    generation: np.ndarray,
    distance_norm: np.ndarray,
    radius_norm: np.ndarray,
    edge_softness: float,
) -> np.ndarray:
    softness = max(float(edge_softness), 1.0e-4)
    gen = np.asarray(generation, dtype=np.float32)
    distance = np.asarray(distance_norm, dtype=np.float32)
    proximal_by_radius = _smoothstep(radius_norm, 0.86 - 0.5 * softness, 0.86 + 0.5 * softness)
    if region == 0:
        proximal_by_generation = 1.0 - _smoothstep(gen, 0.5 - softness, 0.5 + softness)
        proximal_by_distance = 1.0 - _smoothstep(distance, 0.14, 0.34 + 0.08 * softness)
        return np.maximum.reduce((proximal_by_generation, proximal_by_distance, proximal_by_radius)).astype(np.float32)
    if region == 1:
        lower = _smoothstep(gen, 0.5 - softness, 0.5 + softness)
        upper = 1.0 - _smoothstep(gen, 1.5 - softness, 1.5 + softness)
        return np.asarray(lower * upper * (1.0 - 0.65 * proximal_by_radius), dtype=np.float32)
    if region == 2:
        lower = _smoothstep(gen, 1.5 - softness, 1.5 + softness)
        upper = 1.0 - _smoothstep(gen, 3.5 - softness, 3.5 + softness)
        return np.asarray(lower * upper * (1.0 - 0.65 * proximal_by_radius), dtype=np.float32)
    lower = _smoothstep(gen, 3.5 - softness, 3.5 + softness)
    distal_by_distance = _smoothstep(distance, 0.52 - 0.05 * softness, 0.72 + 0.08 * softness)
    return np.asarray(np.maximum(lower, distal_by_distance) * (1.0 - 0.65 * proximal_by_radius), dtype=np.float32)


def _pbr_material_map_texture_modulation(
    entry: dict[str, JsonValue],
    normalized_xyz: np.ndarray,
) -> np.ndarray:
    scale = max(
        _as_float(entry.get("texture_noise_scale", entry.get("noise_scale", 3.0)), "texture_noise_scale"),
        1.0e-6,
    )
    name = str(entry.get("name", "material-map-entry"))
    phase = (sum((index + 1) * ord(char) for index, char in enumerate(name)) % 997) / 997.0 * math.tau
    xyz = normalized_xyz.astype(np.float32)
    value = (
        0.50
        + 0.22 * np.sin(scale * (1.7 * xyz[:, 0] - 0.9 * xyz[:, 1] + 1.1 * xyz[:, 2]) + phase)
        + 0.18 * np.cos(scale * (0.6 * xyz[:, 0] + 1.3 * xyz[:, 1] - 1.5 * xyz[:, 2]) - 0.5 * phase)
        + 0.10 * np.sin(0.55 * scale * (xyz[:, 0] + xyz[:, 1]) + 0.7 * phase)
    )
    return np.asarray(np.clip(value, 0.0, 1.0), dtype=np.float32)


def _pbr_material_map_region_mask(entry: dict[str, JsonValue], region_id: np.ndarray) -> np.ndarray:
    selector = entry.get("regions", entry.get("region_ids", entry.get("region")))
    if selector is None:
        return np.ones(region_id.shape, dtype=bool)
    selectors = selector if isinstance(selector, list) else [selector]
    if any(str(value).lower() == "all" for value in selectors):
        return np.ones(region_id.shape, dtype=bool)
    region_ids = [_pbr_region_id_from_selector(value) for value in selectors]
    return np.isin(region_id.astype(np.int16), np.asarray(region_ids, dtype=np.int16))


def _pbr_region_id_from_selector(value: JsonValue) -> int:
    if isinstance(value, bool):
        raise ValueError("material map region selector must be a region name or id")
    if isinstance(value, int):
        if 0 <= value <= 3:
            return value
        raise ValueError("material map region id must be between 0 and 3")
    if isinstance(value, str):
        if value.isdigit():
            return _pbr_region_id_from_selector(int(value))
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "tracheal": 0,
            "proximal": 0,
            "tracheal_or_proximal_airway": 0,
            "main": 1,
            "main_bronchus": 1,
            "lobar": 2,
            "segmental": 2,
            "lobar_or_segmental_bronchus": 2,
            "distal": 3,
            "distal_airway": 3,
        }
        if normalized in aliases:
            return aliases[normalized]
    raise ValueError(f"unknown material map region selector: {value!r}")


def _material_map_float_mapping(value: JsonValue, field_name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"material map {field_name} must be an object")
    valid_channels = LOCAL_PBR_CHANNELS | PBR_PROPERTY_CHANNELS
    result: dict[str, float] = {}
    for key, raw_number in value.items():
        channel = str(key)
        if channel not in valid_channels:
            raise ValueError(f"unsupported material map channel: {channel}")
        result[channel] = _as_float(raw_number, f"{field_name}.{channel}")
    return result


def _material_map_albedo_delta(value: JsonValue) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("material map albedo_delta_rgb must contain three numbers")
    return np.asarray([_as_float(item, "albedo_delta_rgb") for item in value], dtype=np.float32)


def _as_float(value: JsonValue, name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"material map {name} must be numeric")
    return float(value)


def _material_anatomy_features(
    points: np.ndarray,
    centerline_nodes: _CenterlineMaterialNodes | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    if centerline_nodes is not None and centerline_nodes.points.shape[0] > 0:
        nearest = _nearest_centerline_indices(points, centerline_nodes.points)
        generation = centerline_nodes.generation[nearest].astype(np.float32)
        distance = centerline_nodes.distance_from_root_mm[nearest].astype(np.float32)
        radius = centerline_nodes.radius_mm[nearest].astype(np.float32)
        distance_norm = _normalize01(distance)
        radius_norm = _normalize01(radius)
        return generation, distance_norm, radius_norm, True

    centered = points - points.mean(axis=0, keepdims=True)
    radial = np.linalg.norm(centered[:, :2], axis=1)
    radius_norm = 1.0 - _normalize01(radial)
    distance_norm = _normalize01(centered[:, 2])
    generation = np.clip(np.floor(distance_norm * 6.0), 0.0, 6.0).astype(np.float32)
    return generation, distance_norm.astype(np.float32), radius_norm.astype(np.float32), False


def _material_region_ids(generation: np.ndarray, radius_norm: np.ndarray) -> np.ndarray:
    region_id = np.full(generation.shape, 3, dtype=np.int16)
    region_id[generation <= 0.5] = 0
    region_id[(generation > 0.5) & (generation <= 1.5)] = 1
    region_id[(generation > 1.5) & (generation <= 3.5)] = 2
    region_id[radius_norm > 0.86] = 0
    return region_id


def _material_region_blend_weights(
    generation: np.ndarray,
    distance_norm: np.ndarray,
    radius_norm: np.ndarray,
    *,
    softness: float,
) -> np.ndarray:
    weights = np.column_stack(
        [
            _soft_pbr_region_weight(region, generation, distance_norm, radius_norm, softness)
            for region in range(4)
        ]
    ).astype(np.float32)
    weights = np.maximum(weights, 0.0)
    total = np.sum(weights, axis=1, keepdims=True)
    fallback = np.zeros_like(weights)
    fallback[:, 3] = 1.0
    return np.asarray(np.where(total > 1.0e-6, weights / np.maximum(total, 1.0e-6), fallback), dtype=np.float32)


def _material_summary(
    material_variant: str,
    roughness: np.ndarray,
    specular: np.ndarray,
    wetness: np.ndarray,
    vascularity: np.ndarray,
    mucus: np.ndarray,
    mucus_shininess: np.ndarray,
    erythema: np.ndarray,
    petechiae: np.ndarray,
    stain: np.ndarray,
    bump_height: np.ndarray,
    displacement_height: np.ndarray,
    region_id: np.ndarray,
    generation: np.ndarray,
    used_centerline: bool,
    material_map_summary: dict[str, JsonValue] | None,
    region_blend_softness: float | None = None,
    region_tint_strength: float | None = None,
) -> dict[str, JsonValue]:
    regions: list[dict[str, JsonValue]] = []
    for region_value in range(4):
        count = int(np.count_nonzero(region_id == region_value))
        if count:
            regions.append({"id": region_value, "name": _pbr_region_name(region_value), "point_count": count})
    summary: dict[str, JsonValue] = {
        "variant": material_variant,
        "region_model": (
            "nearest centerline generation/radius/distance"
            if used_centerline
            else "coordinate-derived fallback generation/radius/distance"
        ),
        "region_edge_model": (
            "soft normalized anatomy-region weights"
            if region_blend_softness is not None
            else "hard anatomy-region labels"
        ),
        "region_blend_softness": round(float(region_blend_softness), 4)
        if region_blend_softness is not None
        else None,
        "region_tint_strength": round(float(region_tint_strength), 4)
        if region_tint_strength is not None
        else None,
        "regions": regions,
        "channels": [
            "mucosal_pbr_albedo_rgb",
            "mucosal_pbr_roughness",
            "mucosal_pbr_specular",
            "mucosal_pbr_wetness",
            "mucosal_pbr_vascularity",
            "mucosal_pbr_mucus",
            "mucosal_pbr_mucus_shininess",
            "mucosal_pbr_erythema",
            "mucosal_pbr_petechiae",
            "mucosal_pbr_stain",
            "mucosal_pbr_bump_height",
            "mucosal_pbr_displacement_height",
            "mucosal_pbr_region_id",
            "mucosal_pbr_generation",
        ],
        "heterogeneity_model": (
            "deterministic branch/lobe jitter plus local mucus, erythema, petechiae, stain, "
            "bump/displacement, and mucus-shininess maps"
        ),
        "roughness_mean": _rounded_mean(roughness),
        "specular_mean": _rounded_mean(specular),
        "wetness_mean": _rounded_mean(wetness),
        "vascularity_mean": _rounded_mean(vascularity),
        "mucus_mean": _rounded_mean(mucus),
        "mucus_shininess_mean": _rounded_mean(mucus_shininess),
        "erythema_mean": _rounded_mean(erythema),
        "petechiae_mean": _rounded_mean(petechiae),
        "stain_mean": _rounded_mean(stain),
        "bump_height_mean": _rounded_mean(bump_height),
        "displacement_height_mean": _rounded_mean(displacement_height),
        "generation_min": int(generation.min()) if generation.size else None,
        "generation_max": int(generation.max()) if generation.size else None,
    }
    if material_map_summary is not None:
        summary["material_map"] = material_map_summary
    return summary


def _pbr_texture_atlas_images(
    points: np.ndarray,
    *,
    atlas: _MucosalMaterialAtlas,
    centerline_nodes: _CenterlineMaterialNodes | None,
    width_px: int,
    height_px: int,
) -> dict[str, Any]:
    u, v, uv_mapping = _pbr_texture_atlas_uv(points, centerline_nodes)
    point_uv = np.column_stack((u, v)).astype(np.float32) if u.size else np.zeros((0, 2), dtype=np.float32)
    images: dict[str, np.ndarray] = {}
    albedo, occupied = _rasterize_smoothed_atlas_values(
        atlas.albedo_rgb.astype(np.float32) / 255.0,
        u,
        v,
        width_px=width_px,
        height_px=height_px,
        sigma=PBR_RENDER_ATLAS_ALBEDO_SIGMA_PX,
        post_fill_sigma=PBR_RENDER_ATLAS_ALBEDO_POST_FILL_SIGMA_PX,
    )
    images["albedo_rgb"] = np.asarray(np.clip(albedo * 255.0, 0, 255), dtype=np.uint8)
    scalar_channels = {
        "roughness": atlas.roughness,
        "specular": atlas.specular,
        "wetness": atlas.wetness,
        "vascularity": atlas.vascularity,
        "mucus": atlas.mucus,
        "mucus_shininess": atlas.mucus_shininess,
        "erythema": atlas.erythema,
        "petechiae": atlas.petechiae,
        "stain": atlas.stain,
        "bump_height": atlas.bump_height,
        "displacement_height": atlas.displacement_height,
    }
    for channel_name, values in scalar_channels.items():
        raster, _ = _rasterize_smoothed_atlas_values(
            values,
            u,
            v,
            width_px=width_px,
            height_px=height_px,
            sigma=PBR_RENDER_ATLAS_SCALAR_SIGMA_PX,
            post_fill_sigma=PBR_RENDER_ATLAS_SCALAR_POST_FILL_SIGMA_PX,
        )
        images[channel_name] = np.asarray(np.clip(raster * 255.0, 0, 255), dtype=np.uint8)

    region_rgb = PBR_REGION_COLORS[np.clip(atlas.region_id.astype(np.int16), 0, 3)]
    region_raster, _ = _rasterize_atlas_values(
        region_rgb.astype(np.float32) / 255.0,
        u,
        v,
        width_px=width_px,
        height_px=height_px,
        fill_empty=True,
    )
    images["region_id"] = np.asarray(np.clip(region_raster * 255.0, 0, 255), dtype=np.uint8)
    generation = atlas.generation.astype(np.float32)
    generation_scale = max(float(generation.max()) if generation.size else 0.0, 1.0)
    generation_raster, _ = _rasterize_atlas_values(
        generation / generation_scale,
        u,
        v,
        width_px=width_px,
        height_px=height_px,
        fill_empty=True,
    )
    images["generation"] = np.asarray(np.clip(generation_raster * 255.0, 0, 255), dtype=np.uint8)
    images["occupancy"] = np.where(occupied, 255, 0).astype(np.uint8)
    uv_mapping["occupied_pixel_count"] = int(np.count_nonzero(occupied))
    uv_mapping["occupied_pixel_fraction"] = round(float(np.mean(occupied)) if occupied.size else 0.0, 6)
    uv_mapping["render_texture_filter"] = {
        "fill": "normalized Gaussian splat, nearest fallback only outside smoothed support",
        "albedo_sigma_px": PBR_RENDER_ATLAS_ALBEDO_SIGMA_PX,
        "scalar_sigma_px": PBR_RENDER_ATLAS_SCALAR_SIGMA_PX,
        "albedo_post_fill_sigma_px": PBR_RENDER_ATLAS_ALBEDO_POST_FILL_SIGMA_PX,
        "scalar_post_fill_sigma_px": PBR_RENDER_ATLAS_SCALAR_POST_FILL_SIGMA_PX,
        "purpose": "remove sparse-atlas Voronoi islands and hard material territories before renderer sampling",
    }
    return {
        "images": images,
        "occupied": occupied,
        "point_uv": point_uv,
        "uv_mapping": uv_mapping,
    }


def _pbr_texture_atlas_uv(
    points: np.ndarray,
    centerline_nodes: _CenterlineMaterialNodes | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, JsonValue]]:
    if points.shape[0] == 0:
        return (
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
            {
                "type": "object_space_cylindrical",
                "u": "atan2(centered_y, centered_x)",
                "v": "nearest centerline distance when available, otherwise centered z",
                "centerline_used": False,
            },
        )
    centered = points - points.mean(axis=0, keepdims=True)
    theta = np.arctan2(centered[:, 1], centered[:, 0])
    u = ((theta + math.pi) / (2.0 * math.pi)).astype(np.float32)
    if centerline_nodes is not None and centerline_nodes.points.shape[0] > 0:
        nearest = _nearest_centerline_indices(points, centerline_nodes.points)
        v = _normalize01(centerline_nodes.distance_from_root_mm[nearest])
        centerline_used = True
        v_source = "nearest_centerline_distance_from_root_mm"
    else:
        v = _normalize01(centered[:, 2])
        centerline_used = False
        v_source = "centered_mesh_z"
    return (
        np.clip(u, 0.0, 1.0).astype(np.float32),
        np.clip(v, 0.0, 1.0).astype(np.float32),
        {
            "type": "object_space_cylindrical",
            "u": "atan2(centered_y, centered_x) remapped to 0..1",
            "v": v_source,
            "image_origin": "upper_left",
            "v_axis": "proximal/top to distal/bottom",
            "centerline_used": centerline_used,
        },
    )


def _rasterize_atlas_values(
    values: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    width_px: int,
    height_px: int,
    fill_empty: bool,
) -> tuple[np.ndarray, np.ndarray]:
    sums, counts = _atlas_value_sums_and_counts(
        values,
        u,
        v,
        width_px=width_px,
        height_px=height_px,
    )
    occupied = counts > 0.0
    raster = _normalize_atlas_sums(sums, counts, occupied)
    if fill_empty:
        raster = _fill_empty_atlas_pixels(raster, occupied)
    return raster, occupied


def _rasterize_smoothed_atlas_values(
    values: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    width_px: int,
    height_px: int,
    sigma: float,
    post_fill_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    sums, counts = _atlas_value_sums_and_counts(
        values,
        u,
        v,
        width_px=width_px,
        height_px=height_px,
    )
    occupied = counts > 0.0
    if not bool(np.any(occupied)):
        return _normalize_atlas_sums(sums, counts, occupied), occupied

    smooth_sums = _smooth_render_texture_atlas(sums, sigma=sigma)
    smooth_counts = _smooth_render_texture_atlas(counts, sigma=sigma)
    support = smooth_counts > 1.0e-6
    raster = _normalize_atlas_sums(smooth_sums, smooth_counts, support)
    if not bool(np.all(support)):
        raster = _fill_empty_atlas_pixels(raster, support)
    if post_fill_sigma > 0.0:
        raster = _smooth_render_texture_atlas(raster, sigma=post_fill_sigma)
    return raster, occupied


def _atlas_value_sums_and_counts(
    values: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    width_px: int,
    height_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        output_shape = (height_px, width_px)
        sums = np.zeros(output_shape, dtype=np.float32)
    elif values.ndim == 2:
        output_shape = (height_px, width_px, values.shape[1])
        sums = np.zeros(output_shape, dtype=np.float32)
    else:
        raise ValueError("atlas values must be a 1D or 2D array")
    counts = np.zeros((height_px, width_px), dtype=np.float32)
    if values.shape[0] == 0:
        return sums, counts > 0.0
    x_float = np.clip(u.astype(np.float32) * float(width_px - 1), 0.0, float(width_px - 1))
    y_float = np.clip(v.astype(np.float32) * float(height_px - 1), 0.0, float(height_px - 1))
    x0 = np.floor(x_float).astype(np.int64)
    y0 = np.floor(y_float).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width_px - 1)
    y1 = np.clip(y0 + 1, 0, height_px - 1)
    wx = x_float - x0.astype(np.float32)
    wy = y_float - y0.astype(np.float32)
    _splat_atlas_values(sums, counts, values, x0, y0, (1.0 - wx) * (1.0 - wy))
    _splat_atlas_values(sums, counts, values, x1, y0, wx * (1.0 - wy))
    _splat_atlas_values(sums, counts, values, x0, y1, (1.0 - wx) * wy)
    _splat_atlas_values(sums, counts, values, x1, y1, wx * wy)
    return sums, counts


def _normalize_atlas_sums(sums: np.ndarray, counts: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    raster = np.zeros_like(sums, dtype=np.float32)
    if not bool(np.any(occupied)):
        return raster
    if sums.ndim == 2:
        raster[occupied] = sums[occupied] / counts[occupied]
    else:
        raster[occupied] = sums[occupied] / counts[occupied, np.newaxis]
    return raster


def _splat_atlas_values(
    sums: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> None:
    weights = np.asarray(weights, dtype=np.float32)
    if not bool(np.any(weights > 0.0)):
        return
    np.add.at(counts, (y, x), weights)
    if values.ndim == 1:
        np.add.at(sums, (y, x), values * weights)
    else:
        np.add.at(sums, (y, x), values * weights[:, np.newaxis])


def _fill_empty_atlas_pixels(image: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    if occupied.size == 0 or bool(np.all(occupied)) or not bool(np.any(occupied)):
        return image
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore[import-not-found]

        nearest_y, nearest_x = distance_transform_edt(~occupied, return_distances=False, return_indices=True)
        return image[nearest_y, nearest_x]
    except ImportError:
        return _neighbor_fill_empty_atlas_pixels(image, occupied)


def _smooth_render_texture_atlas(image: np.ndarray, *, sigma: float) -> np.ndarray:
    if image.size == 0 or sigma <= 0.0:
        return image
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore[import-not-found]

        resolved_sigma: float | tuple[float, float, float]
        resolved_sigma = (float(sigma), float(sigma), 0.0) if image.ndim == 3 else float(sigma)
        return np.asarray(gaussian_filter(image, sigma=resolved_sigma, mode="nearest"), dtype=np.float32)
    except ImportError:
        return _box_blur_texture_atlas(image)


def _box_blur_texture_atlas(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1), *(((0, 0),) if image.ndim == 3 else ())), mode="edge")
    result = np.zeros_like(image, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            result += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    return result / 9.0


def _neighbor_fill_empty_atlas_pixels(image: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    filled = image.copy()
    mask = occupied.copy()
    shifts = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for _ in range(max(image.shape[0], image.shape[1])):
        if bool(np.all(mask)):
            break
        previous = filled.copy()
        previous_mask = mask.copy()
        for dy, dx in shifts:
            src_y0 = max(0, -dy)
            src_y1 = image.shape[0] - max(0, dy)
            src_x0 = max(0, -dx)
            src_x1 = image.shape[1] - max(0, dx)
            dst_y0 = max(0, dy)
            dst_y1 = image.shape[0] - max(0, -dy)
            dst_x0 = max(0, dx)
            dst_x1 = image.shape[1] - max(0, -dx)
            candidates = previous_mask[src_y0:src_y1, src_x0:src_x1] & ~mask[dst_y0:dst_y1, dst_x0:dst_x1]
            if not bool(np.any(candidates)):
                continue
            if filled.ndim == 2:
                filled[dst_y0:dst_y1, dst_x0:dst_x1][candidates] = previous[src_y0:src_y1, src_x0:src_x1][
                    candidates
                ]
            else:
                filled[dst_y0:dst_y1, dst_x0:dst_x1][candidates, :] = previous[
                    src_y0:src_y1,
                    src_x0:src_x1,
                ][candidates, :]
            mask[dst_y0:dst_y1, dst_x0:dst_x1][candidates] = True
    return filled


def _compose_pbr_texture_atlas_preview(images: dict[str, np.ndarray]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except ImportError:
        return _compose_pbr_texture_atlas_preview_numpy(images)
    labels = (
        "albedo_rgb",
        "roughness",
        "specular",
        "wetness",
        "vascularity",
        "mucus",
        "mucus_shininess",
        "erythema",
        "petechiae",
        "stain",
        "bump_height",
        "displacement_height",
        "region_id",
        "generation",
        "occupancy",
    )
    tiles = [_atlas_preview_tile(images[label], label, Image, ImageDraw, ImageFont) for label in labels]
    tile_width, tile_height = tiles[0].size
    columns = 4
    rows = int(math.ceil(len(tiles) / columns))
    canvas = Image.new("RGB", (tile_width * columns, tile_height * rows), (0, 0, 0))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    return np.asarray(canvas, dtype=np.uint8)


def _atlas_preview_tile(image: np.ndarray, label: str, image_module: Any, draw_module: Any, font_module: Any) -> Any:
    rgb = _ensure_rgb(image)
    tile = image_module.fromarray(rgb)
    draw = draw_module.Draw(tile, "RGBA")
    font = font_module.load_default()
    text = label.replace("_", " ").title()
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((0, 0, tile.width, bbox[3] - bbox[1] + 10), fill=(0, 0, 0, 155))
    draw.text((5, 4), text, font=font, fill=(250, 246, 235, 235))
    return tile


def _compose_pbr_texture_atlas_preview_numpy(images: dict[str, np.ndarray]) -> np.ndarray:
    labels = tuple(PBR_TEXTURE_ATLAS_CHANNELS)
    tiles = [_ensure_rgb(images[label]) for label in labels]
    tile_height, tile_width = tiles[0].shape[:2]
    columns = 4
    rows = int(math.ceil(len(tiles) / columns))
    canvas = np.zeros((tile_height * rows, tile_width * columns, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y0 = (index // columns) * tile_height
        x0 = (index % columns) * tile_width
        canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
    return canvas


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return np.repeat(image[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3].astype(np.uint8)
    raise ValueError("preview image must be grayscale or RGB")


def _normal_map_from_height_image(
    height_image: np.ndarray,
    *,
    strength: float,
    pre_smooth_sigma: float = 0.0,
) -> np.ndarray:
    height = np.asarray(height_image, dtype=np.float32)
    if height.ndim == 3:
        height = height[:, :, 0]
    if height.size == 0:
        return np.zeros((*height.shape, 3), dtype=np.uint8)
    height = np.clip(height / 255.0, 0.0, 1.0)
    if pre_smooth_sigma > 0.0:
        height = _smooth_render_texture_atlas(height, sigma=pre_smooth_sigma)
    dy, dx = np.gradient(height)
    normal = np.stack(
        (
            -dx * float(strength),
            -dy * float(strength),
            np.ones_like(height, dtype=np.float32),
        ),
        axis=2,
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1.0e-6)
    return np.asarray(np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255), dtype=np.uint8)


def _coat_roughness_from_mucus_image(mucus_shininess_image: np.ndarray) -> np.ndarray:
    shininess = np.asarray(mucus_shininess_image, dtype=np.float32)
    if shininess.ndim == 3:
        shininess = shininess[:, :, 0]
    shininess = np.clip(shininess / 255.0, 0.0, 1.0)
    roughness = np.clip(0.04 + 0.54 * (1.0 - shininess), 0.035, 0.62)
    return np.asarray(np.clip(roughness * 255.0, 0, 255), dtype=np.uint8)


def _mesh_triangles(mesh: Any) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles: list[list[int]] = []
    offset = 0
    while offset < faces.size:
        vertex_count = int(faces[offset])
        if vertex_count < 3:
            offset += vertex_count + 1
            continue
        indices = faces[offset + 1 : offset + 1 + vertex_count]
        if vertex_count == 3:
            triangles.append([int(indices[0]), int(indices[1]), int(indices[2])])
        else:
            root = int(indices[0])
            for index in range(1, vertex_count - 1):
                triangles.append([root, int(indices[index]), int(indices[index + 1])])
        offset += vertex_count + 1
    return np.asarray(triangles, dtype=np.int64)


def _write_pbr_mdl_usda(
    path: Path,
    *,
    points: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    point_uv: np.ndarray,
    channel_paths: dict[str, Path],
    normal_map_path: Path,
    coat_roughness_map_path: Path,
    material_summary: dict[str, JsonValue],
    mdl_shader_target: str,
    centerline_nodes: _CenterlineMaterialNodes | None = None,
    include_surface_overlays: bool = True,
) -> None:
    stage_dir = path.parent
    roughness_mean = _summary_float(material_summary, "roughness_mean", 0.45)
    specular_mean = _summary_float(material_summary, "specular_mean", 0.52)
    mucus_shininess_mean = _summary_float(material_summary, "mucus_shininess_mean", 0.38)
    bump_mean = _summary_float(material_summary, "bump_height_mean", 0.32)
    displacement_mean = _summary_float(material_summary, "displacement_height_mean", 0.22)
    clearcoat_weight = float(np.clip(0.18 + 0.82 * mucus_shininess_mean, 0.0, 1.0))
    clearcoat_roughness = float(np.clip(0.28 - 0.22 * mucus_shininess_mean, 0.035, 0.32))
    clearcoat_transparency = float(np.clip(0.96 + 0.04 * mucus_shininess_mean, 0.92, 1.0))
    clearcoat_flatten = float(np.clip(0.45 + 0.45 * mucus_shininess_mean, 0.35, 0.92))
    clearcoat_bump_strength = float(np.clip(0.06 + 0.32 * bump_mean, 0.03, 0.35))
    bump_strength = float(np.clip(0.06 + 0.62 * bump_mean, 0.03, 0.48))
    displacement_scale = float(np.clip(0.03 + 0.22 * displacement_mean, 0.0, 0.28))
    st = _face_varying_texture_coordinates(point_uv, faces)

    point_values = ", ".join(_usda_vec3(row) for row in points)
    face_counts = ", ".join("3" for _ in faces)
    face_indices = ", ".join(str(int(index)) for face in faces for index in face)
    normal_values = ", ".join(_usda_vec3(row) for row in normals)
    st_values = ", ".join(_usda_vec2(row) for row in st)
    extent_values = ", ".join(_usda_vec3(row) for row in (points.min(axis=0), points.max(axis=0)))
    material_path = "/World/Looks/SynAirG_Airway_PBR"
    mdl_shader_name = _mdl_shader_prim_name(mdl_shader_target)
    mdl_source_asset = _mdl_shader_source_asset(mdl_shader_target)
    mdl_subidentifier = _mdl_shader_subidentifier(mdl_shader_target)
    texture_assets = {
        name: _usda_asset(channel_paths[name], stage_dir)
        for name in (
            "albedo_rgb",
            "roughness",
            "specular",
            "mucus_shininess",
            "displacement_height",
        )
    }
    normal_asset = _usda_asset(normal_map_path, stage_dir)
    coat_roughness_asset = _usda_asset(coat_roughness_map_path, stage_dir)
    mdl_shader_inputs = _mdl_shader_input_lines(
        mdl_shader_target=mdl_shader_target,
        texture_assets=texture_assets,
        normal_asset=normal_asset,
        coat_roughness_asset=coat_roughness_asset,
        roughness_mean=roughness_mean,
        specular_mean=specular_mean,
        clearcoat_weight=clearcoat_weight,
        clearcoat_roughness=clearcoat_roughness,
        clearcoat_transparency=clearcoat_transparency,
        clearcoat_flatten=clearcoat_flatten,
        clearcoat_bump_strength=clearcoat_bump_strength,
        bump_strength=bump_strength,
        displacement_scale=displacement_scale,
    )
    surface_overlay_usda = (
        _surface_overlay_usda(
            points=points,
            normals=normals,
            material_summary=material_summary,
            centerline_nodes=centerline_nodes,
        )
        if include_surface_overlays
        else _SurfaceOverlayUsd()
    )
    lines = [
        "#usda 1.0\n",
        "(\n",
        '    defaultPrim = "World"\n',
        "    metersPerUnit = 0.001\n",
        '    upAxis = "Z"\n',
        ")\n\n",
        'def Xform "World"\n',
        "{\n",
        '    def Xform "Airway"\n',
        "    {\n",
        '        def Mesh "AirwayMesh" (\n',
        '            prepend apiSchemas = ["MaterialBindingAPI"]\n',
        "        )\n",
        "        {\n",
        f"            float3[] extent = [{extent_values}]\n",
        f"            point3f[] points = [{point_values}]\n",
        f"            int[] faceVertexCounts = [{face_counts}]\n",
        f"            int[] faceVertexIndices = [{face_indices}]\n",
        "            bool doubleSided = true\n",
        '            uniform token subdivisionScheme = "none"\n',
        f"            normal3f[] normals = [{normal_values}]\n",
        '            uniform token normals:interpolation = "vertex"\n',
        f"            texCoord2f[] primvars:st = [{st_values}] (\n",
        '                interpolation = "faceVarying"\n',
        "            )\n",
        f"            rel material:binding = <{material_path}>\n",
        "        }\n",
        *surface_overlay_usda.mesh_lines,
        "    }\n\n",
        '    def Scope "Looks"\n',
        "    {\n",
        '        def Material "SynAirG_Airway_PBR"\n',
        "        {\n",
        '            string inputs:frame:stPrimvarName = "st"\n',
        "            token outputs:surface.connect = "
        "</World/Looks/SynAirG_Airway_PBR/PreviewSurface.outputs:surface>\n",
        "            token outputs:displacement.connect = "
        "</World/Looks/SynAirG_Airway_PBR/PreviewSurface.outputs:displacement>\n",
        f"            token outputs:mdl:surface.connect = <{material_path}/{mdl_shader_name}.outputs:out>\n",
        f"            token outputs:mdl:displacement.connect = <{material_path}/{mdl_shader_name}.outputs:out>\n",
        f"            token outputs:mdl:volume.connect = <{material_path}/{mdl_shader_name}.outputs:out>\n\n",
        '            def Shader "PreviewSurface"\n',
        "            {\n",
        '                uniform token info:id = "UsdPreviewSurface"\n',
        "                token outputs:surface\n",
        "                token outputs:displacement\n",
        "                int inputs:useSpecularWorkflow = 1\n",
        "                color3f inputs:diffuseColor.connect = "
        "</World/Looks/SynAirG_Airway_PBR/AlbedoTexture.outputs:rgb>\n",
        "                color3f inputs:specularColor.connect = "
        "</World/Looks/SynAirG_Airway_PBR/SpecularTexture.outputs:rgb>\n",
        "                float inputs:roughness.connect = "
        "</World/Looks/SynAirG_Airway_PBR/RoughnessTexture.outputs:r>\n",
        "                float inputs:clearcoat.connect = "
        "</World/Looks/SynAirG_Airway_PBR/MucusShininessTexture.outputs:r>\n",
        f"                float inputs:clearcoatRoughness = {_usda_float(clearcoat_roughness)}\n",
        "                normal3f inputs:normal.connect = "
        "</World/Looks/SynAirG_Airway_PBR/NormalFromBumpTexture.outputs:rgb>\n",
        "                float inputs:displacement.connect = "
        "</World/Looks/SynAirG_Airway_PBR/DisplacementTexture.outputs:r>\n",
        "                float inputs:metallic = 0\n",
        "                float inputs:opacity = 1\n",
        "            }\n\n",
        _usd_texture_shader(
            "AlbedoTexture",
            texture_assets["albedo_rgb"],
            outputs=("rgb", "r"),
            source_color_space="sRGB",
        ),
        _usd_texture_shader(
            "RoughnessTexture",
            texture_assets["roughness"],
            outputs=("r", "rgb"),
            source_color_space="raw",
        ),
        _usd_texture_shader(
            "SpecularTexture",
            texture_assets["specular"],
            outputs=("rgb", "r"),
            source_color_space="raw",
        ),
        _usd_texture_shader(
            "MucusShininessTexture",
            texture_assets["mucus_shininess"],
            outputs=("r", "rgb"),
            source_color_space="raw",
        ),
        _usd_texture_shader(
            "NormalFromBumpTexture",
            normal_asset,
            outputs=("rgb",),
            source_color_space="raw",
            scale=(2.0, 2.0, 2.0, 1.0),
            bias=(-1.0, -1.0, -1.0, 0.0),
        ),
        _usd_texture_shader(
            "DisplacementTexture",
            texture_assets["displacement_height"],
            outputs=("r", "rgb"),
            source_color_space="raw",
            scale=(displacement_scale, displacement_scale, displacement_scale, 1.0),
        ),
        '            def Shader "PrimvarSt"\n',
        "            {\n",
        '                uniform token info:id = "UsdPrimvarReader_float2"\n',
        "                string inputs:varname.connect = "
        "</World/Looks/SynAirG_Airway_PBR.inputs:frame:stPrimvarName>\n",
        "                float2 outputs:result\n",
        "            }\n\n",
        f'            def Shader "{mdl_shader_name}"\n',
        "            {\n",
        '                uniform token info:implementationSource = "sourceAsset"\n',
        f"                asset info:mdl:sourceAsset = @{mdl_source_asset}@\n",
        f'                token info:mdl:sourceAsset:subIdentifier = "{mdl_subidentifier}"\n',
        "                token outputs:out\n",
        *mdl_shader_inputs,
        "            }\n",
        "        }\n",
        *surface_overlay_usda.material_lines,
        "    }\n\n",
        '    def Xform "EndoscopeRig"\n',
        "    {\n",
        '        def Camera "EndoscopeCamera"\n',
        "        {\n",
        "            float2 clippingRange = (1, 95)\n",
        "            float focalLength = 6\n",
        "            float horizontalAperture = 4.8\n",
        "            float verticalAperture = 3.6\n",
        "        }\n",
        '        def SphereLight "EndoscopeLight"\n',
        "        {\n",
        "            float inputs:intensity = 420\n",
        "            float inputs:radius = 1.1\n",
        "            color3f inputs:color = (1, 0.94, 0.86)\n",
        "        }\n",
        "    }\n",
        "}\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def _face_varying_texture_coordinates(point_uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return face-varying UVs with local unwrap for triangles crossing the cylindrical seam."""
    point_uv = np.asarray(point_uv, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if point_uv.ndim != 2 or point_uv.shape[1] != 2:
        raise ValueError("point_uv must have shape (N, 2)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if faces.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    st_values: list[np.ndarray] = []
    max_index = point_uv.shape[0] - 1
    for face in faces:
        uv = point_uv[np.clip(face, 0, max_index)].astype(np.float32).copy()
        u = uv[:, 0]
        if float(np.max(u) - np.min(u)) > 0.5:
            u[u < 0.5] += 1.0
        st_values.append(np.column_stack((u, 1.0 - uv[:, 1])).astype(np.float32))
    return np.concatenate(st_values, axis=0)


def _surface_overlay_usda(
    *,
    points: np.ndarray,
    normals: np.ndarray,
    material_summary: dict[str, JsonValue],
    centerline_nodes: _CenterlineMaterialNodes | None,
) -> _SurfaceOverlayUsd:
    material_map = material_summary.get("material_map")
    if not isinstance(material_map, dict):
        return _SurfaceOverlayUsd()
    specs = _surface_deposit_specs_from_material_map(material_map)
    if not specs:
        return _SurfaceOverlayUsd()
    deposit_meshes, used_materials = _surface_deposit_mesh_lines(
        points=points,
        normals=normals,
        centerline_nodes=centerline_nodes,
        specs=specs,
    )
    if not deposit_meshes:
        return _SurfaceOverlayUsd()
    material_lines: list[str] = []
    for material in dict.fromkeys(used_materials):
        material_lines.extend(_surface_deposit_material_usda_lines(material))
    return _SurfaceOverlayUsd(
        mesh_lines=tuple(deposit_meshes),
        material_lines=tuple(material_lines),
    )


def _surface_deposit_specs_from_material_map(material_map: dict[str, JsonValue]) -> tuple[_SurfaceDepositSpec, ...]:
    map_name = str(material_map.get("name", "")).lower()
    default_material = "blood_film" if "blood" in map_name or "bloody" in map_name else "mucus_plug"
    raw_overlays = material_map.get("surface_overlays")
    specs: list[_SurfaceDepositSpec] = []
    if isinstance(raw_overlays, list):
        for index, raw_overlay in enumerate(raw_overlays):
            if not isinstance(raw_overlay, dict):
                continue
            center = _surface_deposit_vec3(raw_overlay.get("center_normalized_xyz"))
            if center is None:
                center = _surface_deposit_vec3(raw_overlay.get("center"))
            if center is None:
                continue
            material = str(raw_overlay.get("material", default_material)).lower().replace("-", "_")
            if material not in _SURFACE_DEPOSIT_MATERIALS:
                material = default_material
            shape = str(raw_overlay.get("shape", "pool")).lower().replace("-", "_")
            if shape not in {"film", "streak", "ribbon", "pool", "droplet", "plug"}:
                shape = "pool"
            fallback_name = f"SurfaceDeposit_{index:02d}"
            name = _usd_identifier(str(raw_overlay.get("name", fallback_name)), fallback_name)
            specs.append(
                _SurfaceDepositSpec(
                    name=name,
                    material=material,
                    center_normalized_xyz=center,
                    major_mm=_surface_deposit_float(raw_overlay, "major_mm", 4.0, lower=0.15, upper=24.0),
                    minor_mm=_surface_deposit_float(raw_overlay, "minor_mm", 1.25, lower=0.1, upper=12.0),
                    height_mm=_surface_deposit_float(raw_overlay, "height_mm", 0.28, lower=0.006, upper=3.0),
                    phase=_surface_deposit_float(raw_overlay, "phase", 0.73 * index, lower=-100.0, upper=100.0),
                    shape=shape,
                    segment_count=int(
                        _surface_deposit_float(raw_overlay, "segment_count", 48.0, lower=16.0, upper=96.0)
                    ),
                    ring_count=int(_surface_deposit_float(raw_overlay, "ring_count", 4.0, lower=2.0, upper=8.0)),
                    lobe_strength=_surface_deposit_float(
                        raw_overlay,
                        "lobe_strength",
                        0.16,
                        lower=0.0,
                        upper=0.42,
                    ),
                    edge_feather=_surface_deposit_float(
                        raw_overlay,
                        "edge_feather",
                        0.0,
                        lower=0.0,
                        upper=0.85,
                    ),
                    edge_height_scale=_surface_deposit_float(
                        raw_overlay,
                        "edge_height_scale",
                        0.24,
                        lower=0.01,
                        upper=0.65,
                    ),
                    base_offset_mm=_surface_deposit_float(
                        raw_overlay,
                        "base_offset_mm",
                        0.045,
                        lower=0.004,
                        upper=0.2,
                    ),
                    thickness_noise_strength=_surface_deposit_float(
                        raw_overlay,
                        "thickness_noise_strength",
                        0.0,
                        lower=0.0,
                        upper=0.85,
                    ),
                    rim_height_scale=_surface_deposit_float(
                        raw_overlay,
                        "rim_height_scale",
                        0.0,
                        lower=0.0,
                        upper=0.75,
                    ),
                    runoff_strength=_surface_deposit_float(
                        raw_overlay,
                        "runoff_strength",
                        0.0,
                        lower=0.0,
                        upper=1.0,
                    ),
                )
            )
        if specs:
            return tuple(specs)
    if "bloody" in map_name or "blood" in map_name:
        return _default_blood_surface_deposit_specs()
    if ("thick" in map_name and ("mucus" in map_name or "secretion" in map_name)) or "mucus-plug" in map_name:
        return _default_mucus_surface_deposit_specs()
    return ()


def _surface_deposit_vec3(value: JsonValue) -> tuple[float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    return (vector[0], vector[1], vector[2])


def _surface_deposit_float(
    entry: dict[str, JsonValue],
    key: str,
    default: float,
    *,
    lower: float,
    upper: float,
) -> float:
    value = entry.get(key)
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(np.clip(float(value), lower, upper))
    return default


def _default_blood_surface_deposit_specs() -> tuple[_SurfaceDepositSpec, ...]:
    return (
        _SurfaceDepositSpec(
            name="BloodDeposit_00_ElongatedStreak",
            material="blood_film",
            center_normalized_xyz=(0.083, 0.079, 0.108),
            major_mm=6.8,
            minor_mm=1.8,
            height_mm=0.32,
            phase=0.35,
            shape="streak",
            lobe_strength=0.2,
            edge_feather=0.32,
            edge_height_scale=0.18,
            thickness_noise_strength=0.14,
            runoff_strength=0.18,
        ),
        _SurfaceDepositSpec(
            name="BloodDeposit_01_DarkPool",
            material="blood_clot",
            center_normalized_xyz=(0.072, -0.022, 0.084),
            major_mm=3.5,
            minor_mm=1.55,
            height_mm=0.58,
            phase=1.25,
            shape="pool",
            lobe_strength=0.18,
            edge_feather=0.22,
            edge_height_scale=0.16,
            thickness_noise_strength=0.22,
            rim_height_scale=0.12,
        ),
        _SurfaceDepositSpec(
            name="BloodDeposit_02_WetDroplet",
            material="blood_droplet",
            center_normalized_xyz=(0.071, -0.020, 0.088),
            major_mm=1.55,
            minor_mm=0.95,
            height_mm=0.42,
            phase=2.1,
            shape="droplet",
            lobe_strength=0.07,
            edge_feather=0.12,
            edge_height_scale=0.12,
            thickness_noise_strength=0.08,
            rim_height_scale=0.04,
        ),
    )


def _default_mucus_surface_deposit_specs() -> tuple[_SurfaceDepositSpec, ...]:
    return (
        _SurfaceDepositSpec(
            name="MucusDeposit_00_RaisedPlug",
            material="mucus_plug",
            center_normalized_xyz=(0.15, 0.16, 0.23),
            major_mm=4.6,
            minor_mm=1.9,
            height_mm=0.82,
            phase=0.72,
            shape="plug",
            lobe_strength=0.24,
            edge_feather=0.24,
            edge_height_scale=0.18,
            thickness_noise_strength=0.16,
            rim_height_scale=0.08,
        ),
        _SurfaceDepositSpec(
            name="MucusDeposit_01_ThinTail",
            material="mucus_film",
            center_normalized_xyz=(0.11, 0.11, 0.18),
            major_mm=5.2,
            minor_mm=0.8,
            height_mm=0.18,
            phase=0.95,
            shape="ribbon",
            lobe_strength=0.16,
            edge_feather=0.34,
            edge_height_scale=0.16,
            thickness_noise_strength=0.10,
            runoff_strength=0.16,
        ),
    )


def _surface_deposit_mesh_lines(
    *,
    points: np.ndarray,
    normals: np.ndarray,
    centerline_nodes: _CenterlineMaterialNodes | None,
    specs: tuple[_SurfaceDepositSpec, ...],
) -> tuple[list[str], list[str]]:
    if points.shape[0] == 0 or normals.shape[0] != points.shape[0]:
        return [], []
    centered = points - points.mean(axis=0, keepdims=True)
    span = np.maximum(np.ptp(centered, axis=0), 1.0)
    normalized = centered / span[np.newaxis, :]
    mesh_lines: list[str] = []
    used_materials: list[str] = []
    for spec in specs:
        target = np.asarray(spec.center_normalized_xyz, dtype=np.float64)
        anchor_index = int(np.argmin(np.sum((normalized - target[np.newaxis, :]) ** 2, axis=1)))
        material = spec.material if spec.material in _SURFACE_DEPOSIT_MATERIALS else "blood_film"
        vertices, faces = _surface_deposit_mesh(
            anchor=points[anchor_index],
            surface_normal=normals[anchor_index],
            centerline_nodes=centerline_nodes,
            spec=spec,
            scale_multiplier=1.0,
            height_multiplier=1.0,
            base_offset_mm=spec.base_offset_mm,
        )
        mesh_lines.extend(
            _surface_deposit_mesh_usda_lines(
                spec.name,
                vertices,
                faces,
                _surface_deposit_material_prim(material),
            )
        )
        used_materials.append(material)
        edge_material = _surface_deposit_edge_material(material)
        if spec.edge_feather > 0.0 and edge_material is not None:
            edge_vertices, edge_faces = _surface_deposit_mesh(
                anchor=points[anchor_index],
                surface_normal=normals[anchor_index],
                centerline_nodes=centerline_nodes,
                spec=spec,
                scale_multiplier=1.0 + spec.edge_feather,
                height_multiplier=spec.edge_height_scale,
                base_offset_mm=max(0.006, spec.base_offset_mm * 0.48),
            )
            mesh_lines.extend(
                _surface_deposit_mesh_usda_lines(
                    _usd_identifier(f"{spec.name}_FeatherEdge", f"{spec.name}_FeatherEdge"),
                    edge_vertices,
                    edge_faces,
                    _surface_deposit_material_prim(edge_material),
                )
            )
            used_materials.append(edge_material)
    return mesh_lines, used_materials


def _surface_deposit_mesh(
    *,
    anchor: np.ndarray,
    surface_normal: np.ndarray,
    centerline_nodes: _CenterlineMaterialNodes | None,
    spec: _SurfaceDepositSpec,
    scale_multiplier: float = 1.0,
    height_multiplier: float = 1.0,
    base_offset_mm: float = 0.045,
) -> tuple[np.ndarray, np.ndarray]:
    inward = _surface_inward_direction(anchor, surface_normal, centerline_nodes)
    tangent_a, tangent_b = _surface_tangent_basis(inward)
    tangent_major = math.cos(spec.phase) * tangent_a + math.sin(spec.phase) * tangent_b
    tangent_minor = -math.sin(spec.phase) * tangent_a + math.cos(spec.phase) * tangent_b
    segment_count = max(int(spec.segment_count), 16)
    ring_count = max(int(spec.ring_count), 2)
    scale_multiplier = float(np.clip(scale_multiplier, 0.2, 2.2))
    height_multiplier = float(np.clip(height_multiplier, 0.03, 1.5))
    major_mm = max(float(spec.major_mm) * scale_multiplier, 0.15)
    minor_mm = max(float(spec.minor_mm) * scale_multiplier, 0.1)
    height_mm = max(float(spec.height_mm) * height_multiplier, 0.006)
    base = np.asarray(anchor, dtype=np.float64) + inward * max(float(base_offset_mm), 0.0)
    vertices: list[np.ndarray] = [base + inward * height_mm]
    for ring_index in range(1, ring_count + 1):
        ring_scale = ring_index / ring_count
        height_scale = _surface_deposit_height_profile(ring_scale, spec.shape)
        tail_offset = _surface_deposit_tail_offset(
            ring_scale,
            tangent_major=tangent_major,
            tangent_minor=tangent_minor,
            major_mm=major_mm,
            minor_mm=minor_mm,
            shape=spec.shape,
        ) * (1.0 + spec.runoff_strength)
        for index in range(segment_count):
            angle = math.tau * index / segment_count
            organic = _surface_deposit_outline_factor(
                angle,
                phase=spec.phase,
                shape=spec.shape,
                lobe_strength=spec.lobe_strength,
            )
            offset = (
                tangent_major * major_mm * ring_scale * organic * math.cos(angle)
                + tangent_minor * minor_mm * ring_scale * organic * math.sin(angle)
                + tail_offset
            )
            height_factor = _surface_deposit_height_factor(
                angle,
                ring_scale,
                base_profile=height_scale,
                spec=spec,
            )
            vertices.append(base + offset + inward * height_mm * height_factor)
    faces: list[list[int]] = []
    first_ring = 1
    for index in range(segment_count):
        next_index = (index + 1) % segment_count
        faces.append([0, first_ring + index, first_ring + next_index])
    for ring_index in range(ring_count - 1):
        inner_start = 1 + ring_index * segment_count
        outer_start = inner_start + segment_count
        for index in range(segment_count):
            next_index = (index + 1) % segment_count
            inner = inner_start + index
            inner_next = inner_start + next_index
            outer = outer_start + index
            outer_next = outer_start + next_index
            faces.append([inner, outer, outer_next])
            faces.append([inner, outer_next, inner_next])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _surface_deposit_height_profile(ring_scale: float, shape: str) -> float:
    ring_scale = float(np.clip(ring_scale, 0.0, 1.0))
    if shape == "droplet":
        return max(0.025, 0.92 * (1.0 - ring_scale**2.4) ** 0.55)
    if shape == "plug":
        return max(0.04, 0.82 * (1.0 - ring_scale**1.35) + 0.06 * math.sin(math.pi * ring_scale))
    if shape == "pool":
        return max(0.025, 0.56 * (1.0 - ring_scale**1.15) + 0.05 * math.sin(math.pi * ring_scale))
    if shape in {"streak", "ribbon", "film"}:
        return max(0.012, 0.45 * (1.0 - ring_scale**1.65))
    return max(0.02, 0.5 * (1.0 - ring_scale))


def _surface_deposit_height_factor(
    angle: float,
    ring_scale: float,
    *,
    base_profile: float,
    spec: _SurfaceDepositSpec,
) -> float:
    """Return deterministic local thickness for pooled/streaked surface deposits."""
    ring_scale = float(np.clip(ring_scale, 0.0, 1.0))
    base_profile = max(float(base_profile), 0.0)
    noise_strength = float(np.clip(spec.thickness_noise_strength, 0.0, 0.85))
    if noise_strength > 0.0:
        wave = (
            0.54 * math.sin(3.0 * angle + spec.phase + 1.9 * ring_scale)
            + 0.31 * math.cos(7.0 * angle - 0.35 * spec.phase + 2.4 * ring_scale)
            + 0.15 * math.sin(11.0 * angle + 0.8 * spec.phase)
        )
        base_profile *= max(0.18, 1.0 + noise_strength * wave)

    rim_scale = float(np.clip(spec.rim_height_scale, 0.0, 0.75))
    if rim_scale > 0.0:
        rim = math.exp(-((ring_scale - 0.82) ** 2) / 0.018)
        base_profile += rim_scale * rim

    runoff = float(np.clip(spec.runoff_strength, 0.0, 1.0))
    if runoff > 0.0 and spec.shape in {"streak", "ribbon", "film"}:
        trailing_side = max(0.0, -math.cos(angle - 0.18 * spec.phase))
        base_profile += 0.16 * runoff * trailing_side * (ring_scale**1.4)

    return max(0.004, float(base_profile))


def _surface_deposit_outline_factor(
    angle: float,
    *,
    phase: float,
    shape: str,
    lobe_strength: float,
) -> float:
    lobe_strength = float(np.clip(lobe_strength, 0.0, 0.42))
    organic = (
        1.0
        + lobe_strength * math.sin(2.0 * angle + phase)
        + 0.07 * math.cos(5.0 * angle - 0.6 * phase)
        + 0.035 * math.sin(9.0 * angle + 1.7 * phase)
    )
    if shape in {"streak", "ribbon", "film"}:
        organic *= 1.0 + 0.24 * max(0.0, -math.cos(angle)) + 0.06 * math.sin(angle + phase)
    elif shape == "droplet":
        organic = 1.0 + 0.05 * math.sin(3.0 * angle + phase) + 0.025 * math.cos(7.0 * angle)
    elif shape == "plug":
        organic += 0.12 * math.sin(4.0 * angle - phase) + 0.08 * max(0.0, math.sin(angle - phase))
    return float(np.clip(organic, 0.55, 1.65))


def _surface_deposit_tail_offset(
    ring_scale: float,
    *,
    tangent_major: np.ndarray,
    tangent_minor: np.ndarray,
    major_mm: float,
    minor_mm: float,
    shape: str,
) -> np.ndarray:
    ring_power = float(np.clip(ring_scale, 0.0, 1.0)) ** 1.7
    if shape == "streak":
        return -0.2 * major_mm * ring_power * tangent_major + 0.04 * minor_mm * ring_power * tangent_minor
    if shape in {"ribbon", "film"}:
        return -0.28 * major_mm * ring_power * tangent_major
    if shape == "droplet":
        return -0.045 * major_mm * ring_power * tangent_major
    if shape == "plug":
        return 0.055 * minor_mm * ring_power * tangent_minor
    return np.zeros(3, dtype=np.float64)


def _surface_inward_direction(
    anchor: np.ndarray,
    surface_normal: np.ndarray,
    centerline_nodes: _CenterlineMaterialNodes | None,
) -> np.ndarray:
    normal = _safe_unit(surface_normal, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    if centerline_nodes is not None and centerline_nodes.points.shape[0] > 0:
        nearest = int(np.argmin(np.sum((centerline_nodes.points - anchor[np.newaxis, :]) ** 2, axis=1)))
        inward = centerline_nodes.points[nearest] - anchor
        inward = _safe_unit(inward, -normal)
        return inward
    return -normal


def _surface_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    tangent_a = _safe_unit(np.cross(normal, reference), np.asarray((1.0, 0.0, 0.0), dtype=np.float64))
    tangent_b = _safe_unit(np.cross(normal, tangent_a), np.asarray((0.0, 1.0, 0.0), dtype=np.float64))
    return tangent_a, tangent_b


def _safe_unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        return np.asarray(fallback, dtype=np.float64)
    return vector / norm


def _surface_deposit_mesh_usda_lines(
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    material_prim: str,
) -> list[str]:
    point_values = ", ".join(_usda_vec3(row) for row in vertices)
    face_counts = ", ".join("3" for _ in faces)
    face_indices = ", ".join(str(int(index)) for face in faces for index in face)
    extent_values = ", ".join(_usda_vec3(row) for row in (vertices.min(axis=0), vertices.max(axis=0)))
    return [
        f'        def Mesh "{name}" (\n',
        '            prepend apiSchemas = ["MaterialBindingAPI"]\n',
        "        )\n",
        "        {\n",
        f"            float3[] extent = [{extent_values}]\n",
        f"            point3f[] points = [{point_values}]\n",
        f"            int[] faceVertexCounts = [{face_counts}]\n",
        f"            int[] faceVertexIndices = [{face_indices}]\n",
        "            bool doubleSided = true\n",
        '            uniform token subdivisionScheme = "none"\n',
        f"            rel material:binding = </World/Looks/{material_prim}>\n",
        "        }\n",
    ]


def _surface_deposit_material_prim(material: str) -> str:
    material_payload = _SURFACE_DEPOSIT_MATERIALS.get(material, _SURFACE_DEPOSIT_MATERIALS["blood_film"])
    return str(material_payload["prim"])


def _surface_deposit_edge_material(material: str) -> str | None:
    edge_material = f"{material}_edge"
    if edge_material in _SURFACE_DEPOSIT_MATERIALS:
        return edge_material
    return None


def _surface_deposit_material_usda_lines(material: str) -> list[str]:
    material_payload = _SURFACE_DEPOSIT_MATERIALS.get(material, _SURFACE_DEPOSIT_MATERIALS["blood_film"])
    prim = str(material_payload["prim"])
    diffuse = material_payload["diffuse"]
    specular = material_payload["specular"]
    clearcoat_roughness = _usda_float(float(material_payload["clearcoat_roughness"]))
    return [
        f'        def Material "{prim}"\n',
        "        {\n",
        f"            token outputs:surface.connect = </World/Looks/{prim}/PreviewSurface.outputs:surface>\n",
        '            def Shader "PreviewSurface"\n',
        "            {\n",
        '                uniform token info:id = "UsdPreviewSurface"\n',
        "                token outputs:surface\n",
        f"                color3f inputs:diffuseColor = {_usda_vec3(diffuse)}\n",
        f"                color3f inputs:specularColor = {_usda_vec3(specular)}\n",
        f"                float inputs:roughness = {_usda_float(float(material_payload['roughness']))}\n",
        f"                float inputs:clearcoat = {_usda_float(float(material_payload['clearcoat']))}\n",
        f"                float inputs:clearcoatRoughness = {clearcoat_roughness}\n",
        "                float inputs:metallic = 0\n",
        f"                float inputs:opacity = {_usda_float(float(material_payload['opacity']))}\n",
        "            }\n",
        "        }\n",
    ]


def _mdl_shader_input_lines(
    *,
    mdl_shader_target: str,
    texture_assets: dict[str, str],
    normal_asset: str,
    coat_roughness_asset: str,
    roughness_mean: float,
    specular_mean: float,
    clearcoat_weight: float,
    clearcoat_roughness: float,
    clearcoat_transparency: float,
    clearcoat_flatten: float,
    clearcoat_bump_strength: float,
    bump_strength: float,
    displacement_scale: float,
) -> list[str]:
    if mdl_shader_target == "omnisurface":
        return [
            "                float inputs:diffuse_reflection_weight = 1\n",
            f"                color3f inputs:diffuse_reflection_color = {_usda_vec3((0.68, 0.34, 0.28))}\n",
            f"                asset inputs:diffuse_reflection_color_image = {texture_assets['albedo_rgb']}\n",
            f"                float inputs:diffuse_reflection_roughness = {_usda_float(roughness_mean)}\n",
            f"                asset inputs:diffuse_reflection_roughness_image = {texture_assets['roughness']}\n",
            f"                float inputs:specular_reflection_weight = {_usda_float(specular_mean)}\n",
            f"                asset inputs:specular_reflection_weight_image = {texture_assets['specular']}\n",
            f"                float inputs:specular_reflection_roughness = {_usda_float(roughness_mean)}\n",
            f"                asset inputs:specular_reflection_roughness_image = {texture_assets['roughness']}\n",
            "                float inputs:specular_reflection_ior = 1.38\n",
            f"                float inputs:coat_weight = {_usda_float(clearcoat_weight)}\n",
            f"                asset inputs:coat_weight_image = {texture_assets['mucus_shininess']}\n",
            f"                float inputs:coat_roughness = {_usda_float(clearcoat_roughness)}\n",
            f"                asset inputs:coat_roughness_image = {coat_roughness_asset}\n",
            "                float inputs:coat_ior = 1.34\n",
            "                float inputs:coat_affect_color = 0.12\n",
            "                float inputs:coat_affect_roughness = 0.35\n",
            f"                float inputs:coat_normal_strength = {_usda_float(clearcoat_bump_strength)}\n",
            f"                asset inputs:coat_normal_image = {normal_asset}\n",
            f"                float inputs:geometry_normal_strength = {_usda_float(bump_strength)}\n",
            f"                asset inputs:geometry_normal_image = {normal_asset}\n",
            "                float inputs:geometry_normal_roughness_strength = "
            f"{_usda_float(PBR_MDL_GEOMETRY_NORMAL_ROUGHNESS_STRENGTH)}\n",
            f"                asset inputs:geometry_displacement_image = {texture_assets['displacement_height']}\n",
            f"                float inputs:geometry_displacement_scale = {_usda_float(displacement_scale)}\n",
            "                float inputs:geometry_displacement_scalar_zero_value = 0.5\n",
        ]

    lines = [
        f"                color3f inputs:diffuse_color_constant = {_usda_vec3((0.68, 0.34, 0.28))}\n",
        f"                asset inputs:diffuse_texture = {texture_assets['albedo_rgb']}\n",
        "                float inputs:albedo_brightness = 1\n",
        f"                color3f inputs:diffuse_tint = {_usda_vec3((1.0, 1.0, 1.0))}\n",
        f"                float inputs:reflection_roughness_constant = {_usda_float(roughness_mean)}\n",
        "                float inputs:reflection_roughness_texture_influence = 1\n",
        f"                asset inputs:reflectionroughness_texture = {texture_assets['roughness']}\n",
        f"                float inputs:specular_level = {_usda_float(specular_mean)}\n",
        f"                asset inputs:normalmap_texture = {normal_asset}\n",
        f"                float inputs:bump_factor = {_usda_float(bump_strength)}\n",
        "                float inputs:metallic_constant = 0\n",
    ]
    if mdl_shader_target == "omnipbr-clearcoat":
        lines.extend(
            [
                "                bool inputs:enable_clearcoat = true\n",
                f"                float inputs:clearcoat_weight = {_usda_float(clearcoat_weight)}\n",
                f"                float inputs:clearcoat_reflection_roughness = {_usda_float(clearcoat_roughness)}\n",
                f"                float inputs:clearcoat_transparency = {_usda_float(clearcoat_transparency)}\n",
                f"                float inputs:clearcoat_flatten = {_usda_float(clearcoat_flatten)}\n",
                "                float inputs:clearcoat_ior = 1.34\n",
                f"                float inputs:clearcoat_bump_factor = {_usda_float(clearcoat_bump_strength)}\n",
                f"                asset inputs:clearcoat_normalmap_texture = {normal_asset}\n",
            ]
        )
    return lines


def _usd_texture_shader(
    name: str,
    asset: str,
    *,
    outputs: tuple[str, ...],
    source_color_space: str,
    scale: tuple[float, float, float, float] | None = None,
    bias: tuple[float, float, float, float] | None = None,
) -> str:
    lines = [
        f'            def Shader "{name}"\n',
        "            {\n",
        '                uniform token info:id = "UsdUVTexture"\n',
        f"                asset inputs:file = {asset}\n",
        '                float2 inputs:st.connect = </World/Looks/SynAirG_Airway_PBR/PrimvarSt.outputs:result>\n',
        f'                token inputs:sourceColorSpace = "{source_color_space}"\n',
        '                token inputs:wrapS = "repeat"\n',
        '                token inputs:wrapT = "repeat"\n',
    ]
    if scale is not None:
        lines.append(f"                float4 inputs:scale = {_usda_vec4(scale)}\n")
    if bias is not None:
        lines.append(f"                float4 inputs:bias = {_usda_vec4(bias)}\n")
    for output in outputs:
        if output == "rgb":
            lines.append("                float3 outputs:rgb\n")
        else:
            lines.append(f"                float outputs:{output}\n")
    lines.extend(["            }\n\n"])
    return "".join(lines)


def _write_omniverse_render_script(path: Path, usd_filename: str) -> None:
    script = f'''"""Render the SynAirG USD/MDL airway bundle from an Omniverse Kit Python session.

Example:
    kit --enable omni.kit.viewport.utility --exec "{path.name}" -- --output rtx_preview.png
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import carb.settings
import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="{usd_filename}")
    parser.add_argument("--output", default="rtx_preview.png")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--spp", type=int, default=128)
    args = parser.parse_args()

    bundle_dir = Path(__file__).resolve().parent
    stage_path = bundle_dir / args.stage
    output_path = bundle_dir / args.output

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", int(args.spp))
    settings.set("/rtx/pathtracing/totalSpp", int(args.spp))
    settings.set("/app/renderer/resolution/width", int(args.width))
    settings.set("/app/renderer/resolution/height", int(args.height))

    context = omni.usd.get_context()
    await context.open_stage_async(str(stage_path))
    viewport = get_active_viewport()
    if viewport is not None:
        viewport.camera_path = "/World/EndoscopeRig/EndoscopeCamera"
        await capture_viewport_to_file(viewport, str(output_path))

    await omni.kit.app.get_app().next_update_async()


asyncio.ensure_future(main())
'''
    path.write_text(script, encoding="utf-8")


def _load_json(path: Path) -> dict[str, JsonValue]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _usd_identifier(value: str, fallback: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
    identifier = re.sub(r"_+", "_", identifier).strip("_")
    if not identifier:
        identifier = fallback
    if not re.match(r"^[A-Za-z_]", identifier):
        identifier = f"_{identifier}"
    return identifier


def _summary_float(summary: dict[str, JsonValue], key: str, default: float) -> float:
    value = summary.get(key)
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return default


def _mdl_shader_source_asset(target: str) -> str:
    if target == "omnipbr-clearcoat":
        return "OmniPBR_ClearCoat.mdl"
    if target == "omnipbr":
        return "OmniPBR.mdl"
    if target == "omnisurface":
        return "OmniSurface.mdl"
    raise ValueError(f"unsupported MDL shader target: {target}")


def _mdl_shader_subidentifier(target: str) -> str:
    if target == "omnipbr-clearcoat":
        return "OmniPBR_ClearCoat"
    if target == "omnipbr":
        return "OmniPBR"
    if target == "omnisurface":
        return "OmniSurface"
    raise ValueError(f"unsupported MDL shader target: {target}")


def _mdl_shader_prim_name(target: str) -> str:
    if target == "omnipbr-clearcoat":
        return "ClearCoatPBR"
    if target == "omnipbr":
        return "OmniPBR"
    if target == "omnisurface":
        return "OmniSurfacePBR"
    raise ValueError(f"unsupported MDL shader target: {target}")


def _usda_asset(path: Path, stage_dir: Path) -> str:
    try:
        asset_path = path.resolve().relative_to(stage_dir.resolve())
    except ValueError:
        asset_path = path.resolve()
    return f"@{asset_path.as_posix()}@"


def _usda_float(value: float) -> str:
    return f"{float(value):.9g}"


def _usda_vec2(value: Any) -> str:
    return f"({_usda_float(float(value[0]))}, {_usda_float(float(value[1]))})"


def _usda_vec3(value: Any) -> str:
    return f"({_usda_float(float(value[0]))}, {_usda_float(float(value[1]))}, {_usda_float(float(value[2]))})"


def _usda_vec4(value: Any) -> str:
    return (
        f"({_usda_float(float(value[0]))}, {_usda_float(float(value[1]))}, "
        f"{_usda_float(float(value[2]))}, {_usda_float(float(value[3]))})"
    )


def _rounded_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return round(float(np.mean(values)), 4)


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    minimum = float(np.min(values))
    span = max(float(np.max(values) - minimum), 1.0e-6)
    return np.asarray((values - minimum) / span, dtype=np.float32)


def _smoothstep(values: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    scale = max(float(edge1 - edge0), 1.0e-6)
    t = np.clip((np.asarray(values, dtype=np.float32) - float(edge0)) / scale, 0.0, 1.0)
    return np.asarray(t * t * (3.0 - 2.0 * t), dtype=np.float32)


def _load_centerline_material_nodes(mesh_path: Path) -> _CenterlineMaterialNodes | None:
    graph_path = mesh_path.parent / "centerline_graph.json"
    if not graph_path.is_file():
        return None
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            return None
        points: list[list[float]] = []
        generations: list[float] = []
        distances: list[float] = []
        radii: list[float] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            point = node.get("point_mm")
            if not isinstance(point, list | tuple) or len(point) != 3:
                continue
            points.append([float(point[0]), float(point[1]), float(point[2])])
            generations.append(float(node.get("generation", 0.0)))
            distances.append(float(node.get("distance_from_root_mm", 0.0)))
            radii.append(float(node.get("radius_mm", 1.0)))
        if not points:
            return None
        return _CenterlineMaterialNodes(
            points=np.asarray(points, dtype=np.float64),
            generation=np.asarray(generations, dtype=np.float32),
            distance_from_root_mm=np.asarray(distances, dtype=np.float32),
            radius_mm=np.asarray(radii, dtype=np.float32),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _nearest_centerline_indices(points: np.ndarray, centerline_points: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]

        tree = cKDTree(centerline_points)
        try:
            return np.asarray(tree.query(points, workers=-1)[1], dtype=np.int64)
        except TypeError:
            return np.asarray(tree.query(points)[1], dtype=np.int64)
    except ImportError:
        nearest = np.empty(points.shape[0], dtype=np.int64)
        chunk_size = 512
        for start in range(0, points.shape[0], chunk_size):
            stop = min(start + chunk_size, points.shape[0])
            delta = points[start:stop, np.newaxis, :] - centerline_points[np.newaxis, :, :]
            nearest[start:stop] = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        return nearest


def _mesh_with_point_normals(mesh: Any) -> tuple[Any, np.ndarray]:
    normal_mesh = mesh.compute_normals(
        point_normals=True,
        cell_normals=False,
        consistent_normals=True,
        auto_orient_normals=False,
        inplace=False,
    )
    normals = np.asarray(normal_mesh.point_data.get("Normals"), dtype=np.float64)
    if normals.shape != (normal_mesh.n_points, 3):
        normals = np.zeros((normal_mesh.n_points, 3), dtype=np.float64)
        normals[:, 2] = 1.0
    return normal_mesh, _unit_rows(normals)


def _mesh_with_pps(mesh: Any) -> Any:
    pps_mesh = mesh.copy(deep=True)
    pps_mesh.point_data["pps_rgb"] = np.full((pps_mesh.n_points, 3), 32, dtype=np.uint8)
    return pps_mesh


def _pps_rgb(
    points: np.ndarray,
    world_normals: np.ndarray,
    frame: dict[str, JsonValue],
    *,
    depth_of_field_mm: float,
) -> np.ndarray:
    """Approximate per-pixel shading from an endoscope-local light source."""
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    camera_position = _camera_position(frame)
    forward = _unit(np.asarray(frame["forward"], dtype=np.float64))
    light_to_point = points - camera_position[np.newaxis, :]
    distance = np.linalg.norm(light_to_point, axis=1)
    ray = np.divide(light_to_point, np.maximum(distance[:, np.newaxis], 1.0e-9))
    incidence = np.abs(np.sum(world_normals * ray, axis=1))
    beam = np.clip(np.dot(ray, forward), 0.0, 1.0)
    attenuation_distance = max(min(float(depth_of_field_mm), 95.0) * 0.28, 8.0)
    attenuation = 1.0 / (1.0 + (distance / attenuation_distance) ** 2)
    shading = np.power(incidence, 0.75) * np.power(beam, 1.8) * attenuation
    normalized = np.clip(np.nan_to_num(shading, nan=0.0, posinf=0.35, neginf=0.0) / 0.35, 0.0, 1.0)
    rgb = _sample_colormap(
        normalized,
        np.asarray(
            [
                [4, 7, 9],
                [37, 25, 22],
                [100, 58, 45],
                [190, 119, 82],
                [255, 208, 143],
                [255, 249, 220],
            ],
            dtype=np.float32,
        ),
    )
    return np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)


def _vertical_fov_deg(intrinsics: dict[str, JsonValue]) -> float:
    height = float(intrinsics["height_px"])
    fy = float(intrinsics["fy_px"])
    return math.degrees(2.0 * math.atan(height / (2.0 * fy)))


def _place_camera(plotter: Any, frame: dict[str, JsonValue]) -> None:
    position = np.asarray(frame["position_mm"], dtype=np.float64)
    forward = _unit(np.asarray(frame["forward"], dtype=np.float64))
    up = _unit(np.asarray(frame["up"], dtype=np.float64))
    camera_position = _camera_position(frame)
    focal_point = position + forward * 45.0
    camera = plotter.camera
    camera.SetPosition(*camera_position)
    camera.SetFocalPoint(*focal_point)
    camera.SetViewUp(*up)
    camera.SetClippingRange(1.0, 95.0)
    camera.Modified()
    plotter.render()


def _camera_position(frame: dict[str, JsonValue]) -> np.ndarray:
    position = np.asarray(frame["position_mm"], dtype=np.float64)
    forward = _unit(np.asarray(frame["forward"], dtype=np.float64))
    radius = float(frame.get("radius_mm", 2.0))
    return np.asarray(position - forward * float(np.clip(radius * 0.45, 0.8, 2.0)), dtype=np.float64)


def _metric_depth_from_plotter(plotter: Any, *, max_depth_mm: float) -> np.ndarray:
    depth = _raw_depth_from_plotter(plotter)
    return _metric_depth_from_buffer(depth, max_depth_mm=max_depth_mm)


def _raw_depth_from_plotter(plotter: Any) -> np.ndarray:
    return np.asarray(plotter.get_image_depth(fill_value=np.nan, reset_camera_clipping_range=False), dtype=np.float32)


def _metric_depth_from_buffer(depth: np.ndarray, *, max_depth_mm: float) -> np.ndarray:
    max_depth = max(float(max_depth_mm), 1.0e-6)
    metric_depth = np.where(np.isfinite(depth), np.maximum(-depth, 0.0), max_depth)
    metric_depth = np.asarray(np.clip(metric_depth, 0.0, max_depth), dtype=np.float32)
    if not np.isfinite(metric_depth).all():
        raise ValueError("metric depth buffer contains non-finite values after conversion")
    return metric_depth


def _hit_mask_from_depth_buffer(depth: np.ndarray) -> np.ndarray:
    metric_depth = np.where(np.isfinite(depth), -depth, np.nan)
    visible, _ = _scope_mask(depth.shape[0], depth.shape[1])
    return np.asarray(np.isfinite(metric_depth) & (metric_depth > 0.0) & visible, dtype=bool)


def _prepare_condition_export(root: Path, sequence_id: str) -> _ConditionExportInfo:
    modalities = ("rgb", "depth", "normal", "pps", "mask")
    roots = {modality: root / modality / sequence_id for modality in modalities}
    for modality_root in roots.values():
        if modality_root.exists():
            shutil.rmtree(modality_root)
        modality_root.mkdir(parents=True, exist_ok=True)
    return _ConditionExportInfo(output_root=root, sequence_id=sequence_id, frame_count=0, roots=roots)


def _condition_sequence_id(path_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", path_id).strip("-")
    return safe or "scope-path"


def _condition_frame_filename(frame_index: int) -> str:
    return f"{frame_index:06d}.png"


def _depth_to_condition_png(depth_mm: np.ndarray, *, max_depth_mm: float) -> np.ndarray:
    normalized = np.clip(np.asarray(depth_mm, dtype=np.float32) / max(max_depth_mm, 1.0e-6), 0.0, 1.0)
    return np.asarray(np.round(normalized * 65535.0), dtype=np.uint16)


def _mask_to_condition_png(hit_mask: np.ndarray) -> np.ndarray:
    return np.asarray(hit_mask, dtype=np.uint8) * np.uint8(255)


def _write_condition_frames(
    *,
    imageio: Any,
    export: _ConditionExportInfo,
    frame_index: int,
    scope_rgb: np.ndarray,
    depth_mm: np.ndarray,
    max_depth_mm: float,
    normals_rgb: np.ndarray,
    pps_rgb: np.ndarray,
    hit_mask: np.ndarray,
) -> None:
    filename = _condition_frame_filename(frame_index)
    imageio.imwrite(export.roots["rgb"] / filename, np.asarray(scope_rgb, dtype=np.uint8))
    imageio.imwrite(export.roots["depth"] / filename, _depth_to_condition_png(depth_mm, max_depth_mm=max_depth_mm))
    imageio.imwrite(export.roots["normal"] / filename, np.asarray(normals_rgb, dtype=np.uint8))
    imageio.imwrite(export.roots["pps"] / filename, np.asarray(pps_rgb, dtype=np.uint8))
    imageio.imwrite(export.roots["mask"] / filename, _mask_to_condition_png(hit_mask))


def _apply_scope_view_mask(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image[..., :3], dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    visible, normalized_radius = _scope_mask(height, width)
    vignette = np.clip(1.06 - 0.62 * normalized_radius**2, 0.28, 1.0)
    rgb *= vignette[..., np.newaxis]
    rgb[~visible] = 0.0
    border = (normalized_radius > 0.975) & visible
    rgb[border] *= 0.38
    return np.asarray(np.clip(rgb * 255.0, 0, 255), dtype=np.uint8)


def _apply_scope_diagnostic_mask(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image[..., :3], dtype=np.float32)
    height, width = rgb.shape[:2]
    visible, normalized_radius = _scope_mask(height, width)
    rgb[~visible] = 0.0
    border = (normalized_radius > 0.975) & visible
    rgb[border] *= 0.32
    return np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)


def _scope_mask(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[:height, :width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = min(width, height) * 0.485
    normalized_radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2) / radius
    visible = normalized_radius <= 1.0
    return visible, normalized_radius


def _depth_to_rgb(depth_mm: np.ndarray, *, max_depth_mm: float) -> np.ndarray:
    normalized = np.clip(np.asarray(depth_mm, dtype=np.float32) / max(max_depth_mm, 1.0e-6), 0.0, 1.0)
    rgb = _sample_colormap(
        normalized,
        np.asarray(
            [
                [3, 7, 24],
                [24, 68, 142],
                [31, 143, 184],
                [95, 190, 130],
                [238, 219, 113],
                [255, 248, 216],
            ],
            dtype=np.float32,
        ),
    )
    rgb[~np.isfinite(depth_mm)] = 0.0
    return _apply_scope_diagnostic_mask(np.asarray(rgb, dtype=np.uint8))


def _view_normal_rgb(world_normals: np.ndarray, frame: dict[str, JsonValue]) -> np.ndarray:
    forward = _unit(np.asarray(frame["forward"], dtype=np.float64))
    up = _unit(np.asarray(frame["up"], dtype=np.float64))
    right = _unit(np.cross(forward, up))
    if float(np.linalg.norm(right)) <= 1.0e-6:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    up = _unit(np.cross(right, forward))
    view_normals = np.column_stack(
        (
            np.dot(world_normals, right),
            np.dot(world_normals, up),
            np.dot(world_normals, forward),
        )
    )
    return np.asarray(np.clip((view_normals * 0.5 + 0.5) * 255.0, 0, 255), dtype=np.uint8)


def _update_view_normal_scalars(mesh: Any, values: np.ndarray) -> None:
    _update_mesh_rgb_scalars(mesh, "view_normal_rgb", values)


def _update_mesh_rgb_scalars(mesh: Any, name: str, values: np.ndarray) -> None:
    mesh.point_data[name][:] = values
    point_data = mesh.GetPointData()
    vtk_array = point_data.GetArray(name)
    if vtk_array is not None:
        vtk_array.Modified()
    point_data.Modified()
    mesh.Modified()


def _load_ct_context(mesh_path: Path) -> _CtContext | None:
    registration_path = mesh_path.parent / "mesh_registration.json"
    if not registration_path.is_file():
        return None
    try:
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        source_volume = _require_mapping(registration.get("source_volume"), "source_volume")
        ct_descriptor = _require_mapping(source_volume.get("ct.nii.gz"), "source_volume.ct.nii.gz")
        ct_path = (mesh_path.parent / str(ct_descriptor["path"])).resolve()
        transforms = _require_mapping(registration.get("transforms"), "transforms")
        mesh_to_volume = np.asarray(transforms["mesh_world_mm_to_volume_world_mm"], dtype=np.float64)
        volume_to_voxel = np.asarray(transforms["volume_world_mm_to_voxel_index"], dtype=np.float64)
        if mesh_to_volume.shape != (4, 4) or volume_to_voxel.shape != (4, 4):
            return None
    except (KeyError, TypeError, ValueError, OSError):
        return None
    if not ct_path.is_file():
        return None
    try:
        import nibabel as nib  # type: ignore[import-not-found]
    except ImportError:
        return None

    image = nib.load(str(ct_path))
    volume = np.asarray(image.dataobj, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=-1000.0, posinf=1000.0, neginf=-1000.0)
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    if len(zooms) != 3:
        zooms = (1.0, 1.0, 1.0)
    return _CtContext(
        volume=volume,
        mesh_to_voxel=volume_to_voxel @ mesh_to_volume,
        spacing_mm=(max(zooms[0], 1.0e-6), max(zooms[1], 1.0e-6), max(zooms[2], 1.0e-6)),
        source_ct_path=ct_path,
    )


def _render_ct_orthogonal_panel(
    context: _CtContext | None,
    frame: dict[str, JsonValue],
    *,
    width_px: int,
    height_px: int,
) -> np.ndarray:
    if context is None:
        return _placeholder_panel(width_px, height_px, "Source CT unavailable")
    voxel = _mesh_frame_to_voxel(context, frame)
    panel = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    left_width = width_px // 2
    _paste_ct_plane(panel, context, voxel, plane="axial", box=(0, 0, left_width, height_px))
    _paste_ct_plane(panel, context, voxel, plane="coronal", box=(left_width, 0, width_px - left_width, height_px // 2))
    _paste_ct_plane(
        panel,
        context,
        voxel,
        plane="sagittal",
        box=(left_width, height_px // 2, width_px - left_width, height_px - height_px // 2),
    )
    return panel


def _paste_ct_plane(
    panel: np.ndarray,
    context: _CtContext,
    voxel: np.ndarray,
    *,
    plane: str,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, width, height = box
    image, cross_xy, spacing_xy, index_label = _ct_plane_slice(context, voxel, plane=plane)
    tile = _ct_slice_tile(
        image,
        cross_xy=cross_xy,
        spacing_xy=spacing_xy,
        label=f"{plane.capitalize()} {index_label}",
        width_px=width,
        height_px=height,
        window_hu=context.window_hu,
    )
    panel[y0 : y0 + height, x0 : x0 + width] = tile


def _ct_plane_slice(
    context: _CtContext,
    voxel: np.ndarray,
    *,
    plane: str,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], str]:
    shape = context.volume.shape
    i = float(np.clip(voxel[0], 0.0, shape[0] - 1))
    j = float(np.clip(voxel[1], 0.0, shape[1] - 1))
    k = float(np.clip(voxel[2], 0.0, shape[2] - 1))
    ii = int(round(i))
    jj = int(round(j))
    kk = int(round(k))
    if plane == "axial":
        image = context.volume[:, :, kk].T
        return image, (i, j), (context.spacing_mm[0], context.spacing_mm[1]), f"k={kk}"
    if plane == "coronal":
        image = context.volume[:, jj, ::-1].T
        return image, (i, float(shape[2] - 1) - k), (context.spacing_mm[0], context.spacing_mm[2]), f"j={jj}"
    if plane == "sagittal":
        image = context.volume[ii, :, ::-1].T
        return image, (j, float(shape[2] - 1) - k), (context.spacing_mm[1], context.spacing_mm[2]), f"i={ii}"
    raise ValueError(f"unsupported CT plane: {plane}")


def _ct_slice_tile(
    image: np.ndarray,
    *,
    cross_xy: tuple[float, float],
    spacing_xy: tuple[float, float],
    label: str,
    width_px: int,
    height_px: int,
    window_hu: tuple[float, float],
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except ImportError:
        return _placeholder_panel(width_px, height_px, label)

    tile = Image.new("RGB", (width_px, height_px), (0, 0, 0))
    draw = ImageDraw.Draw(tile, "RGBA")
    font = ImageFont.load_default()
    label_height = 22
    content_height = max(1, height_px - label_height - 8)
    content_width = max(1, width_px - 8)
    crop, crop_cross_xy = _centered_physical_crop(
        image,
        cross_xy=cross_xy,
        spacing_xy=spacing_xy,
        half_extent_mm=72.0,
        fill_value=window_hu[0],
    )
    grayscale = _window_ct_slice(crop, window_hu=window_hu)
    side = max(1, min(content_width, content_height))
    resample = getattr(Image, "Resampling", Image).BILINEAR
    slice_image = Image.fromarray(grayscale).resize((side, side), resample=resample).convert("RGB")
    paste_x = (width_px - side) // 2
    paste_y = label_height + (content_height - side) // 2
    tile.paste(slice_image, (paste_x, paste_y))
    cross_x = paste_x + crop_cross_xy[0] * side / max(float(crop.shape[1]), 1.0)
    cross_y = paste_y + crop_cross_xy[1] * side / max(float(crop.shape[0]), 1.0)
    draw.line((paste_x, cross_y, paste_x + side, cross_y), fill=(81, 221, 232, 190), width=1)
    draw.line((cross_x, paste_y, cross_x, paste_y + side), fill=(81, 221, 232, 190), width=1)
    radius = 3
    draw.ellipse(
        (cross_x - radius, cross_y - radius, cross_x + radius, cross_y + radius),
        outline=(255, 92, 92, 240),
        width=2,
    )
    draw.rectangle((0, 0, width_px, label_height), fill=(0, 0, 0, 175))
    draw.text((8, 6), label, font=font, fill=(245, 248, 250, 232))
    return np.asarray(tile, dtype=np.uint8)


def _centered_physical_crop(
    image: np.ndarray,
    *,
    cross_xy: tuple[float, float],
    spacing_xy: tuple[float, float],
    half_extent_mm: float,
    fill_value: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    half_x = max(1, int(round(half_extent_mm / max(spacing_xy[0], 1.0e-6))))
    half_y = max(1, int(round(half_extent_mm / max(spacing_xy[1], 1.0e-6))))
    center_x = int(round(cross_xy[0]))
    center_y = int(round(cross_xy[1]))
    crop = np.full((half_y * 2 + 1, half_x * 2 + 1), fill_value, dtype=np.float32)
    src_x0 = max(0, center_x - half_x)
    src_x1 = min(image.shape[1], center_x + half_x + 1)
    src_y0 = max(0, center_y - half_y)
    src_y1 = min(image.shape[0], center_y + half_y + 1)
    dst_x0 = src_x0 - (center_x - half_x)
    dst_y0 = src_y0 - (center_y - half_y)
    crop[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = image[
        src_y0:src_y1,
        src_x0:src_x1,
    ]
    return crop, (float(cross_xy[0] - (center_x - half_x)), float(cross_xy[1] - (center_y - half_y)))


def _window_ct_slice(image: np.ndarray, *, window_hu: tuple[float, float]) -> np.ndarray:
    low, high = window_hu
    normalized = np.clip((np.asarray(image, dtype=np.float32) - low) / max(high - low, 1.0e-6), 0.0, 1.0)
    return np.asarray(normalized * 255.0, dtype=np.uint8)


def _mesh_frame_to_voxel(context: _CtContext, frame: dict[str, JsonValue]) -> np.ndarray:
    point = _frame_position_mm(frame)
    voxel = context.mesh_to_voxel @ np.asarray([point[0], point[1], point[2], 1.0], dtype=np.float64)
    return np.asarray(voxel[:3], dtype=np.float64)


def _frame_position_mm(frame: dict[str, JsonValue]) -> np.ndarray:
    point = frame.get("centerline_position_mm", frame.get("position_mm"))
    if not isinstance(point, list | tuple) or len(point) != 3:
        raise ValueError("scope frame must include position_mm or centerline_position_mm")
    return np.asarray(point, dtype=np.float64)


def _prepare_isometric_locator(
    mesh_points: np.ndarray,
    frames: list[JsonValue],
    *,
    width_px: int,
    height_px: int,
) -> _IsoLocatorContext | None:
    route_points = np.asarray(
        [_frame_position_mm(_require_mapping(frame, "frame")) for frame in frames],
        dtype=np.float64,
    )
    if mesh_points.size == 0 or route_points.size == 0:
        return None
    stride = max(1, int(math.ceil(len(mesh_points) / 14000)))
    sampled_mesh = mesh_points[::stride]
    view_dir = _unit(np.asarray([0.82, -1.0, 0.58], dtype=np.float64))
    right = _unit(np.cross(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), view_dir))
    up = _unit(np.cross(view_dir, right))
    all_points = np.vstack((sampled_mesh, route_points))
    raw_xy = np.column_stack((np.dot(all_points, right), np.dot(all_points, up)))
    xy_min = raw_xy.min(axis=0)
    xy_max = raw_xy.max(axis=0)
    span = np.maximum(xy_max - xy_min, 1.0e-6)
    padding = 26.0
    scale = min((width_px - padding * 2.0) / span[0], (height_px - padding * 2.0) / span[1])

    def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        projected = np.column_stack((np.dot(points, right), np.dot(points, up)))
        x = padding + (projected[:, 0] - xy_min[0]) * scale
        y = height_px - padding - (projected[:, 1] - xy_min[1]) * scale
        depth = np.dot(points, view_dir)
        return np.column_stack((x, y)), depth

    mesh_xy, mesh_depth = project(sampled_mesh)
    route_xy, route_depth = project(route_points)
    base_image = _draw_isometric_base(mesh_xy, mesh_depth, route_xy, width_px=width_px, height_px=height_px)
    return _IsoLocatorContext(base_image=base_image, route_xy=route_xy, route_depth=route_depth)


def _draw_isometric_base(
    mesh_xy: np.ndarray,
    mesh_depth: np.ndarray,
    route_xy: np.ndarray,
    *,
    width_px: int,
    height_px: int,
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except ImportError:
        return _placeholder_panel(width_px, height_px, "Isometric scope position")

    image = Image.new("RGB", (width_px, height_px), (5, 7, 9))
    draw = ImageDraw.Draw(image, "RGBA")
    depth_min = float(mesh_depth.min())
    depth_span = max(float(mesh_depth.max() - depth_min), 1.0e-6)
    for index in np.argsort(mesh_depth):
        x, y = mesh_xy[index]
        if x < 0 or y < 0 or x >= width_px or y >= height_px:
            continue
        shade = int(48 + 76 * ((float(mesh_depth[index]) - depth_min) / depth_span))
        draw.point((float(x), float(y)), fill=(shade, shade + 6, shade + 8, 115))
    route = [tuple(point) for point in route_xy]
    if len(route) >= 2:
        draw.line(route, fill=(210, 170, 72, 150), width=2)
    font = ImageFont.load_default()
    draw.rounded_rectangle((10, 10, 174, 32), radius=4, fill=(0, 0, 0, 168))
    draw.text((17, 16), "Isometric scope position", font=font, fill=(245, 248, 250, 232))
    return np.asarray(image, dtype=np.uint8)


def _render_isometric_locator(
    context: _IsoLocatorContext | None,
    frame_index: int,
    *,
    width_px: int,
    height_px: int,
) -> np.ndarray:
    if context is None:
        return _placeholder_panel(width_px, height_px, "Isometric scope position unavailable")
    try:
        from PIL import Image, ImageDraw  # type: ignore[import-not-found]
    except ImportError:
        return context.base_image.copy()

    image = Image.fromarray(context.base_image.copy())
    draw = ImageDraw.Draw(image, "RGBA")
    route_count = len(context.route_xy)
    if route_count:
        index = int(np.clip(frame_index, 0, route_count - 1))
        traversed = [tuple(point) for point in context.route_xy[: index + 1]]
        if len(traversed) >= 2:
            draw.line(traversed, fill=(65, 210, 232, 220), width=3)
        x, y = context.route_xy[index]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(255, 82, 82, 245), outline=(255, 238, 190, 240), width=2)
    return np.asarray(image, dtype=np.uint8)


def _placeholder_panel(width_px: int, height_px: int, label: str) -> np.ndarray:
    panel = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    _draw_panel_label(panel, 0, 0, label)
    return panel


def _compose_diagnostic_frame(
    *,
    scope_rgb: np.ndarray,
    depth_rgb: np.ndarray,
    normals_rgb: np.ndarray,
    pps_rgb: np.ndarray,
    ct_rgb: np.ndarray | None = None,
    locator_rgb: np.ndarray | None = None,
    max_depth_mm: float,
) -> np.ndarray:
    height, width = scope_rgb.shape[:2]
    include_locator_column = ct_rgb is not None and locator_rgb is not None
    column_count = 3 if include_locator_column else 2
    composite = np.zeros((height * 2, width * column_count, 3), dtype=np.uint8)
    panels: tuple[tuple[int, int, np.ndarray, str], ...] = (
        (0, 0, scope_rgb, "Scope RGB"),
        (width, 0, depth_rgb, f"Metric depth 0-{max_depth_mm:.0f} mm"),
        (0, height, normals_rgb, "View normals"),
        (width, height, pps_rgb, "PPS shading"),
    )
    if include_locator_column:
        panels = (
            *panels,
            (width * 2, 0, ct_rgb, ""),
            (width * 2, height, locator_rgb, ""),
        )
    for x0, y0, panel, label in panels:
        composite[y0 : y0 + height, x0 : x0 + width] = panel[..., :3]
        _draw_panel_label(composite, x0, y0, label)
    return composite


def _draw_panel_label(image: np.ndarray, x0: int, y0: int, label: str) -> None:
    if not label:
        return
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except ImportError:
        return

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image, "RGBA")
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    padding_x = 7
    padding_y = 5
    label_width = bbox[2] - bbox[0] + padding_x * 2
    label_height = bbox[3] - bbox[1] + padding_y * 2
    left = x0 + 10
    top = y0 + 10
    draw.rounded_rectangle(
        (left, top, left + label_width, top + label_height),
        radius=4,
        fill=(0, 0, 0, 168),
    )
    draw.text((left + padding_x, top + padding_y - 1), label, font=font, fill=(245, 248, 250, 232))
    image[:] = np.asarray(pil_image, dtype=np.uint8)


def _sample_colormap(values: np.ndarray, colors: np.ndarray) -> np.ndarray:
    safe_values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    scaled = np.clip(safe_values, 0.0, 1.0) * float(len(colors) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.clip(lower + 1, 0, len(colors) - 1)
    alpha = (scaled - lower)[..., np.newaxis]
    return (1.0 - alpha) * colors[lower] + alpha * colors[upper]


def _depth_range_mm(bronchoscope: dict[str, JsonValue]) -> tuple[float, float]:
    depth_of_field = bronchoscope.get("depth_of_field_mm", [2.0, 50.0])
    if isinstance(depth_of_field, list | tuple) and len(depth_of_field) >= 2:
        far = float(depth_of_field[1])
    else:
        far = 50.0
    return (0.0, max(far, 95.0))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(vector / norm, dtype=np.float64)


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, np.maximum(norms, 1.0e-9))


def _require_mapping(value: object, field_name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value
