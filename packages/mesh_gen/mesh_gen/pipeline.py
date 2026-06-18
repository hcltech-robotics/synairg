"""High-level mesh generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from synairg_core.volume_manifest import JsonValue, validate_case_id

from mesh_gen.artifacts import MeshCasePaths, prepare_case_dir, utc_timestamp, write_json
from mesh_gen.centerline import (
    extract_centerline_graph,
    prune_mask_to_accessible_centerline,
    write_centerline_artifacts,
)
from mesh_gen.errors import MeshGenError
from mesh_gen.exports import write_obj, write_ply, write_usd
from mesh_gen.mesh import generate_airway_mesh
from mesh_gen.preview import write_mesh_review_png, write_preview_png, write_segmentation_overlay_png
from mesh_gen.qa import assert_plausible_airway_mask, build_quality_report
from mesh_gen.registration import MeshRegistrationInputs, write_mesh_registration
from mesh_gen.segmentation import (
    ExternalCommandSegmentationBackend,
    PrecomputedMaskBackend,
    SegmentationBackend,
    ThresholdSegmentationBackend,
)
from mesh_gen.topology import cap_laryngeal_components, repair_airway_mask
from mesh_gen.viewer import write_viewer_html


@dataclass(frozen=True)
class MeshRunConfig:
    """Configuration for one mesh generation run."""

    ct_path: Path
    output_root: Path
    case_id: str
    mask_path: Path | None = None
    overwrite: bool = False
    backend: str = "precomputed-mask"
    threshold_hu: float = -500.0
    segmenter_command: str | None = None
    segmenter_timeout_seconds: float = 3600.0
    strict_airway_validation: bool = True
    bronchoscope_profile: str | None = None
    min_accessible_diameter_mm: float = 3.0
    max_accessible_depth_mm: float | None = None
    filter_to_accessible_diameter: bool = True
    laryngeal_capping: bool = True
    min_component_voxels: int = 20
    max_bridge_distance_mm: float = 45.0
    refinement_preset: str = "raw"
    smooth_iterations: int = 0
    smoothing_method: str = "auto"
    taubin_pass_band: float = 0.08
    decimate_fraction: float = 0.0
    normal_orientation: str = "auto"
    hollow_mesh: bool = True
    top_opening_radius_scale: float = 1.25
    proximal_opening_trim: bool = False


@dataclass(frozen=True)
class MeshCase:
    """Result of a mesh generation run."""

    case_id: str
    paths: MeshCasePaths


def run_mesh_case(config: MeshRunConfig) -> MeshCase:
    """Run segmentation, topology repair, meshing, centrelines, exports, and QA."""
    validate_case_id(config.case_id)
    if config.smooth_iterations < 0:
        raise MeshGenError("smooth_iterations must be non-negative")
    if not 0.0 <= config.decimate_fraction < 1.0:
        raise MeshGenError("decimate_fraction must be in [0.0, 1.0)")
    if config.top_opening_radius_scale <= 0:
        raise MeshGenError("top_opening_radius_scale must be positive")
    if config.min_accessible_diameter_mm <= 0:
        raise MeshGenError("min_accessible_diameter_mm must be positive")
    if config.max_accessible_depth_mm is not None and config.max_accessible_depth_mm <= 0:
        raise MeshGenError("max_accessible_depth_mm must be positive")
    if config.min_component_voxels < 1:
        raise MeshGenError("min_component_voxels must be positive")
    if config.max_bridge_distance_mm < 0:
        raise MeshGenError("max_bridge_distance_mm must be non-negative")
    refinement = _resolve_refinement(config)

    paths = prepare_case_dir(config.output_root, config.case_id, overwrite=config.overwrite)
    backend = _segmentation_backend(config)
    result = backend.segment(case_id=config.case_id, ct_path=config.ct_path, mask_path=config.mask_path)
    mask_for_repair = result.mask
    laryngeal_capping: dict[str, JsonValue] = {"checked": False}
    if config.laryngeal_capping:
        capped_data, cap_report = cap_laryngeal_components(
            result.mask.data,
            spacing=result.mask.spacing,
            superior_axis=2,
        )
        mask_for_repair = result.mask.with_data(capped_data)
        laryngeal_capping = cast(dict[str, JsonValue], cap_report.to_dict())
    repaired_data, topology = repair_airway_mask(
        mask_for_repair.data,
        spacing=mask_for_repair.spacing,
        min_component_voxels=config.min_component_voxels,
        max_bridge_distance_mm=config.max_bridge_distance_mm,
    )
    repaired_mask = mask_for_repair.with_data(repaired_data)
    plausibility: dict[str, JsonValue] = {"checked": False}
    if config.strict_airway_validation:
        plausibility = assert_plausible_airway_mask(repaired_mask)
    graph = extract_centerline_graph(
        repaired_mask,
        min_accessible_diameter_mm=config.min_accessible_diameter_mm,
        max_accessible_depth_mm=config.max_accessible_depth_mm,
    )
    accessibility_filter: dict[str, JsonValue] = {"checked": False}
    if config.filter_to_accessible_diameter:
        filtered_data, raw_accessibility_filter = prune_mask_to_accessible_centerline(repaired_mask, graph)
        accessibility_filter = cast(dict[str, JsonValue], raw_accessibility_filter)
        accessibility_filter["min_accessible_diameter_mm"] = float(config.min_accessible_diameter_mm)
        accessibility_filter["max_accessible_depth_mm"] = (
            float(config.max_accessible_depth_mm) if config.max_accessible_depth_mm is not None else None
        )
        if int(cast(int, accessibility_filter["removed_voxels"])) > 0:
            repaired_mask = repaired_mask.with_data(filtered_data)
            if config.strict_airway_validation:
                plausibility = assert_plausible_airway_mask(repaired_mask)
            graph = extract_centerline_graph(
                repaired_mask,
                min_accessible_diameter_mm=config.min_accessible_diameter_mm,
                max_accessible_depth_mm=config.max_accessible_depth_mm,
            )
    repaired_mask.save_nifti(paths.airway_mask_path)

    mesh = generate_airway_mesh(
        repaired_mask,
        smooth_iterations=int(refinement["smooth_iterations"]),
        smoothing_method=str(refinement["smoothing_method"]),
        taubin_pass_band=float(refinement["taubin_pass_band"]),
        decimate_fraction=config.decimate_fraction,
        centerline_graph=graph,
        uncap_top=config.hollow_mesh,
        top_opening_radius_scale=config.top_opening_radius_scale,
        proximal_opening_trim=config.proximal_opening_trim,
        normal_orientation=str(refinement["normal_orientation"]),
    )
    write_obj(paths.obj_path, mesh)
    write_ply(paths.ply_path, mesh)
    write_usd(paths.usd_path, mesh)

    write_centerline_artifacts(
        paths.centerline_graph_path, paths.branch_semantics_path, paths.radius_profile_path, graph
    )
    write_preview_png(paths.preview_path, repaired_mask)
    write_segmentation_overlay_png(paths.segmentation_overlay_path, repaired_mask, config.ct_path)
    write_mesh_review_png(paths.mesh_review_path, mesh, graph)
    write_viewer_html(paths.viewer_path, mesh, graph)
    write_mesh_registration(
        MeshRegistrationInputs(
            case_id=config.case_id,
            ct_path=config.ct_path,
            paths=paths,
            mask=repaired_mask,
            mesh=mesh,
            graph=graph,
        )
    )

    emitted = (
        paths.airway_mask_path,
        paths.obj_path,
        paths.ply_path,
        paths.usd_path,
        paths.centerline_graph_path,
        paths.branch_semantics_path,
        paths.radius_profile_path,
        paths.preview_path,
        paths.segmentation_overlay_path,
        paths.mesh_review_path,
        paths.viewer_path,
        paths.mesh_registration_path,
    )
    report = build_quality_report(mask=repaired_mask, mesh=mesh, graph=graph, topology=topology, artifacts=emitted)
    report["created_at"] = utc_timestamp()
    report["segmentation"] = {
        "backend": result.backend_name,
        "metadata": result.metadata,
        "plausibility": plausibility,
        "bronchoscope_profile": config.bronchoscope_profile,
        "min_accessible_diameter_mm": float(config.min_accessible_diameter_mm),
        "max_accessible_depth_mm": (
            float(config.max_accessible_depth_mm) if config.max_accessible_depth_mm is not None else None
        ),
        "filter_to_accessible_diameter": bool(config.filter_to_accessible_diameter),
        "accessibility_filter": accessibility_filter,
        "laryngeal_capping": laryngeal_capping,
    }
    report["refinement"] = {
        "preset": config.refinement_preset,
        "smoothing_method": mesh.metadata.get("smoothing_method", "none"),
        "smooth_iterations": mesh.metadata.get("smooth_iterations", 0),
        "smoothing_backend": mesh.metadata.get("smoothing_backend", "none"),
        "taubin_pass_band": mesh.metadata.get("taubin_pass_band", float(config.taubin_pass_band)),
        "decimate_fraction": mesh.metadata.get("decimate_fraction", config.decimate_fraction),
        "decimation_backend": mesh.metadata.get("decimation_backend", "none"),
        "normal_orientation": mesh.metadata.get("normal_orientation", "outward"),
    }
    write_json(paths.quality_report_path, report)
    return MeshCase(case_id=config.case_id, paths=paths)


def derive_case_id(path: Path) -> str:
    """Derive a safe case ID from a CT path."""
    name = path.name
    for suffix in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    candidate = name.replace(" ", "-")
    return validate_case_id(candidate)


def _segmentation_backend(config: MeshRunConfig) -> SegmentationBackend:
    if config.backend == "precomputed-mask":
        return PrecomputedMaskBackend()
    if config.backend == "ct-threshold":
        return ThresholdSegmentationBackend(config.threshold_hu)
    if config.backend == "external-command":
        return ExternalCommandSegmentationBackend(
            config.segmenter_command,
            timeout_seconds=config.segmenter_timeout_seconds,
        )
    raise MeshGenError(f"unknown segmentation backend {config.backend!r}")


def _resolve_refinement(config: MeshRunConfig) -> dict[str, int | float | str]:
    if config.refinement_preset not in {"raw", "render-ready", "high-detail"}:
        raise MeshGenError("refinement_preset must be one of: high-detail, raw, render-ready")
    if config.smoothing_method not in {"auto", "laplacian", "taubin", "none"}:
        raise MeshGenError("smoothing_method must be one of: auto, laplacian, none, taubin")
    if config.normal_orientation not in {"auto", "outward", "inward"}:
        raise MeshGenError("normal_orientation must be one of: auto, inward, outward")
    if config.taubin_pass_band <= 0:
        raise MeshGenError("taubin_pass_band must be positive")

    preset_iterations = {
        "raw": 0,
        "render-ready": 30,
        "high-detail": 50,
    }[config.refinement_preset]
    smoothing_method = config.smoothing_method
    if smoothing_method == "auto":
        smoothing_method = "laplacian" if config.refinement_preset == "raw" else "taubin"

    normal_orientation = config.normal_orientation
    if normal_orientation == "auto":
        normal_orientation = "outward" if config.refinement_preset == "raw" else "inward"

    smooth_iterations = config.smooth_iterations if config.smooth_iterations > 0 else preset_iterations
    if smooth_iterations == 0:
        smoothing_method = "laplacian" if smoothing_method == "auto" else smoothing_method
    return {
        "smooth_iterations": int(smooth_iterations),
        "smoothing_method": smoothing_method,
        "taubin_pass_band": float(config.taubin_pass_band),
        "normal_orientation": normal_orientation,
    }
