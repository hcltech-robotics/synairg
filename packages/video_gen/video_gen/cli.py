"""CLI group for SynAirG video workflows."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from synairg_core.volume_manifest import ManifestValidationError, validate_case_id
from trainer.bronchogen_manifest import (
    BronchoGenManifestConfig,
    BronchoGenManifestError,
    validate_bronchogen_manifest,
    write_bronchogen_manifest,
)

from video_gen.scope import (
    MATERIAL_PROFILES,
    MATERIAL_VARIANTS,
    MDL_SHADER_TARGETS,
    export_pbr_mdl_bundle,
    export_pbr_texture_atlas,
    render_scope_video,
)

DEFAULT_OUTPUT_ROOT = Path("datasets/videos")

app = typer.Typer(
    help="Render labelled bronchoscopy frames and videos from meshes and pose trajectories.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def callback() -> None:
    """Show commands for bronchoscopy video rendering workflows."""


@app.command()
def status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable status payload."),
    ] = False,
) -> None:
    """Report whether video workflow commands are available."""
    if json_output:
        typer.echo('{"stage":"video","status":"ready","message":"video_gen commands are installed"}')
        return
    typer.echo("video_gen commands are installed")


@app.command("render-scope")
def render_scope_command(
    mesh_case: Annotated[
        Path,
        typer.Option(
            "--mesh-case",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Canonical mesh case directory containing airway_mesh.ply.",
        ),
    ],
    paths_json: Annotated[
        Path,
        typer.Option(
            "--paths",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="scope_paths.json emitted by pose run.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Root directory for canonical video cases."),
    ] = DEFAULT_OUTPUT_ROOT,
    case_id: Annotated[
        str | None,
        typer.Option("--case-id", help="Case ID to use for the canonical video output directory."),
    ] = None,
    path_id: Annotated[
        str | None,
        typer.Option("--path-id", help="Path ID to render. Defaults to the first saved path."),
    ] = None,
    fps: Annotated[
        float | None,
        typer.Option("--fps", min=1.0, help="Output FPS. Defaults to the bronchoscope path profile FPS."),
    ] = None,
    width_px: Annotated[
        int | None,
        typer.Option("--width", min=16, help="Output width. Defaults to saved intrinsics width."),
    ] = None,
    height_px: Annotated[
        int | None,
        typer.Option("--height", min=16, help="Output height. Defaults to saved intrinsics height."),
    ] = None,
    frame_stride: Annotated[
        int,
        typer.Option("--frame-stride", min=1, help="Render every Nth camera frame."),
    ] = 1,
    max_frames: Annotated[
        int | None,
        typer.Option("--max-frames", min=1, help="Optional cap for quick previews."),
    ] = None,
    diagnostic_panels: Annotated[
        bool,
        typer.Option(
            "--diagnostic-panels/--scope-only",
            help="Render metric depth, normals, and PPS shading panels alongside the scope view.",
        ),
    ] = True,
    material_profile: Annotated[
        str,
        typer.Option(
            "--material-profile",
            help=f"Scope material profile: {', '.join(sorted(MATERIAL_PROFILES))}.",
        ),
    ] = "basic",
    material_variant: Annotated[
        str,
        typer.Option(
            "--material-variant",
            help=f"PBR mucosa material variant: {', '.join(sorted(MATERIAL_VARIANTS))}.",
        ),
    ] = "healthy",
    material_map: Annotated[
        Path | None,
        typer.Option(
            "--material-map",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Optional authorable JSON PBR material map for region and focal channel overrides.",
        ),
    ] = None,
    export_conditions: Annotated[
        Path | None,
        typer.Option(
            "--export-conditions",
            file_okay=False,
            dir_okay=True,
            help="Optional SynAirG condition-frame root with rgb/depth/normal/pps/mask subdirectories.",
        ),
    ] = None,
    condition_manifest: Annotated[
        Path | None,
        typer.Option(
            "--condition-manifest",
            file_okay=True,
            dir_okay=False,
            help="Manifest path for exported condition frames. Defaults to EXPORT_CONDITIONS/bronchogen_manifest.json.",
        ),
    ] = None,
    condition_caption: Annotated[
        str,
        typer.Option("--condition-caption", help="Caption/prompt recorded for exported synthetic RGB frames."),
    ] = "realistic flexible bronchoscopy video inside human airway mucosa",
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing video case directory."),
    ] = False,
) -> None:
    """Render an MP4 showing the airway from the bronchoscope camera path."""
    try:
        resolved_case_id = validate_case_id(case_id or mesh_case.name)
        case_dir = output / resolved_case_id
        if case_dir.exists():
            if not overwrite and any(case_dir.iterdir()):
                _abort(f"video case already exists: {case_dir}; pass --overwrite to replace it")
            if overwrite:
                shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        mesh_path = _mesh_path(mesh_case)
        safe_path_id = (path_id or "first-path").replace("/", "-")
        manifest_path = (
            condition_manifest
            if condition_manifest is not None
            else export_conditions / "bronchogen_manifest.json"
            if export_conditions is not None
            else None
        )
        result = render_scope_video(
            mesh_path=mesh_path,
            paths_json_path=paths_json,
            output_video_path=case_dir / f"{safe_path_id}_scope_view.mp4",
            preview_frame_path=case_dir / f"{safe_path_id}_preview.png",
            metadata_path=case_dir / f"{safe_path_id}_metadata.json",
            path_id=path_id,
            fps=fps,
            width_px=width_px,
            height_px=height_px,
            frame_stride=frame_stride,
            max_frames=max_frames,
            diagnostic_panels=diagnostic_panels,
            material_profile=material_profile,
            material_variant=material_variant,
            material_map_path=material_map,
            condition_output_root=export_conditions,
            condition_manifest_path=manifest_path,
        )
        if export_conditions is not None:
            assert manifest_path is not None
            condition_roots = {
                modality: export_conditions / modality
                for modality in result.condition_modalities
                if modality != "rgb"
            }
            write_bronchogen_manifest(
                BronchoGenManifestConfig(
                    rgb_root=export_conditions / "rgb",
                    case_id=resolved_case_id,
                    dataset_name="synairg-conditions",
                    caption=condition_caption,
                    condition_roots=condition_roots,
                    output_path=manifest_path,
                    sequence_depth=1,
                    temporal_context_radius=2,
                    fps=result.fps,
                    require_conditions=True,
                )
            )
            validate_bronchogen_manifest(manifest_path)
    except (BronchoGenManifestError, ManifestValidationError, RuntimeError, ValueError) as exc:
        _abort(str(exc))
    typer.echo(f"Generated {result.path_id} -> {result.output_video_path}")
    if export_conditions is not None:
        typer.echo(f"Generated SynAirG condition manifest -> {manifest_path}")


@app.command("render-material-variants")
def render_material_variants_command(
    mesh_case: Annotated[
        Path,
        typer.Option(
            "--mesh-case",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Canonical mesh case directory containing airway_mesh.ply.",
        ),
    ],
    paths_json: Annotated[
        Path,
        typer.Option(
            "--paths",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="scope_paths.json emitted by pose run.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Root directory for material comparison cases."),
    ] = Path("runs/texture-material-proof-v1"),
    case_id: Annotated[
        str | None,
        typer.Option("--case-id", help="Case ID for the material comparison output directory."),
    ] = None,
    path_id: Annotated[
        str | None,
        typer.Option("--path-id", help="Path ID to render. Defaults to the first saved path."),
    ] = None,
    variants: Annotated[
        str,
        typer.Option(
            "--variants",
            help="Comma-separated PBR variants to render and compare.",
        ),
    ] = "healthy,inflamed,smoker,pale,edematous",
    fps: Annotated[
        float | None,
        typer.Option("--fps", min=1.0, help="Output FPS. Defaults to the bronchoscope path profile FPS."),
    ] = None,
    width_px: Annotated[
        int | None,
        typer.Option("--width", min=16, help="Per-variant tile width. Defaults to saved intrinsics width."),
    ] = None,
    height_px: Annotated[
        int | None,
        typer.Option("--height", min=16, help="Per-variant tile height. Defaults to saved intrinsics height."),
    ] = None,
    frame_stride: Annotated[
        int,
        typer.Option("--frame-stride", min=1, help="Render every Nth camera frame."),
    ] = 1,
    max_frames: Annotated[
        int | None,
        typer.Option("--max-frames", min=1, help="Optional cap for quick previews."),
    ] = None,
    material_map: Annotated[
        Path | None,
        typer.Option(
            "--material-map",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Optional authorable JSON PBR material map applied to every rendered variant.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing comparison case directory."),
    ] = False,
) -> None:
    """Render PBR material variants and compose a labelled comparison MP4."""
    try:
        variant_names = _parse_material_variants(variants)
        resolved_case_id = validate_case_id(case_id or f"{mesh_case.name}-pbr-material-variants")
        case_dir = output / resolved_case_id
        if case_dir.exists():
            if not overwrite and any(case_dir.iterdir()):
                _abort(f"material variant case already exists: {case_dir}; pass --overwrite to replace it")
            if overwrite:
                shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        mesh_path = _mesh_path(mesh_case)
        safe_path_id = (path_id or "first-path").replace("/", "-")
        variant_records = []
        for variant in variant_names:
            variant_dir = case_dir / "variants" / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            result = render_scope_video(
                mesh_path=mesh_path,
                paths_json_path=paths_json,
                output_video_path=variant_dir / f"{safe_path_id}_scope_view.mp4",
                preview_frame_path=variant_dir / f"{safe_path_id}_preview.png",
                metadata_path=variant_dir / f"{safe_path_id}_metadata.json",
                path_id=path_id,
                fps=fps,
                width_px=width_px,
                height_px=height_px,
                frame_stride=frame_stride,
                max_frames=max_frames,
                diagnostic_panels=False,
                material_profile="pbr",
                material_variant=variant,
                material_map_path=material_map,
            )
            variant_records.append(
                {
                    "variant": variant,
                    "video": result.output_video_path,
                    "preview": result.preview_frame_path,
                    "metadata": result.metadata_path,
                    "frame_count": result.frame_count,
                    "fps": result.fps,
                    "width_px": result.width_px,
                    "height_px": result.height_px,
                }
            )
        grid_video_path = case_dir / f"{safe_path_id}_pbr_variant_grid.mp4"
        grid_preview_path = case_dir / f"{safe_path_id}_pbr_variant_grid_preview.png"
        grid_metadata_path = case_dir / f"{safe_path_id}_pbr_variant_grid_metadata.json"
        composition = _compose_variant_grid_video(
            variant_video_paths=[Path(record["video"]) for record in variant_records],
            labels=variant_names,
            output_video_path=grid_video_path,
            preview_frame_path=grid_preview_path,
            fps=float(variant_records[0]["fps"]),
        )
        metadata = {
            "schema_version": "1.0",
            "case_id": resolved_case_id,
            "path_id": path_id,
            "material_profile": "pbr",
            "material_map": str(material_map) if material_map is not None else None,
            "variants": [
                {
                    **record,
                    "video": str(record["video"]),
                    "preview": str(record["preview"]),
                    "metadata": str(record["metadata"]),
                }
                for record in variant_records
            ],
            "grid": {
                "video": str(grid_video_path),
                "preview": str(grid_preview_path),
                **composition,
            },
        }
        grid_metadata_path.write_text(_json_dumps(metadata), encoding="utf-8")
    except (BronchoGenManifestError, ManifestValidationError, RuntimeError, ValueError) as exc:
        _abort(str(exc))
    typer.echo(f"Generated PBR material variant grid -> {grid_video_path}")


@app.command("render-material-maps")
def render_material_maps_command(
    mesh_case: Annotated[
        Path,
        typer.Option(
            "--mesh-case",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Canonical mesh case directory containing airway_mesh.ply.",
        ),
    ],
    paths_json: Annotated[
        Path,
        typer.Option(
            "--paths",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="scope_paths.json emitted by pose run.",
        ),
    ],
    material_maps: Annotated[
        str,
        typer.Option(
            "--material-maps",
            help="Comma-separated map entries. Use LABEL=PATH, PATH, or LABEL=none for an unmapped baseline.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Root directory for material-map comparison cases."),
    ] = Path("runs/texture-material-proof-v1"),
    case_id: Annotated[
        str | None,
        typer.Option("--case-id", help="Case ID for the material-map comparison output directory."),
    ] = None,
    path_id: Annotated[
        str | None,
        typer.Option("--path-id", help="Path ID to render. Defaults to the first saved path."),
    ] = None,
    material_variant: Annotated[
        str,
        typer.Option(
            "--material-variant",
            help=f"Base PBR mucosa material variant for every map: {', '.join(sorted(MATERIAL_VARIANTS))}.",
        ),
    ] = "healthy",
    fps: Annotated[
        float | None,
        typer.Option("--fps", min=1.0, help="Output FPS. Defaults to the bronchoscope path profile FPS."),
    ] = None,
    width_px: Annotated[
        int | None,
        typer.Option("--width", min=16, help="Per-map tile width. Defaults to saved intrinsics width."),
    ] = None,
    height_px: Annotated[
        int | None,
        typer.Option("--height", min=16, help="Per-map tile height. Defaults to saved intrinsics height."),
    ] = None,
    frame_stride: Annotated[
        int,
        typer.Option("--frame-stride", min=1, help="Render every Nth camera frame."),
    ] = 1,
    max_frames: Annotated[
        int | None,
        typer.Option("--max-frames", min=1, help="Optional cap for quick previews."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing comparison case directory."),
    ] = False,
) -> None:
    """Render authorable PBR material maps and compose a labelled comparison MP4."""
    try:
        if material_variant not in MATERIAL_VARIANTS:
            raise ValueError(f"material_variant must be one of {sorted(MATERIAL_VARIANTS)}")
        map_specs = _parse_material_map_specs(material_maps)
        resolved_case_id = validate_case_id(case_id or f"{mesh_case.name}-pbr-material-maps")
        case_dir = output / resolved_case_id
        if case_dir.exists():
            if not overwrite and any(case_dir.iterdir()):
                _abort(f"material-map case already exists: {case_dir}; pass --overwrite to replace it")
            if overwrite:
                shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        mesh_path = _mesh_path(mesh_case)
        safe_path_id = (path_id or "first-path").replace("/", "-")
        map_records = []
        for label, material_map in map_specs:
            safe_label = _safe_case_component(label)
            map_dir = case_dir / "material_maps" / safe_label
            map_dir.mkdir(parents=True, exist_ok=True)
            result = render_scope_video(
                mesh_path=mesh_path,
                paths_json_path=paths_json,
                output_video_path=map_dir / f"{safe_path_id}_scope_view.mp4",
                preview_frame_path=map_dir / f"{safe_path_id}_preview.png",
                metadata_path=map_dir / f"{safe_path_id}_metadata.json",
                path_id=path_id,
                fps=fps,
                width_px=width_px,
                height_px=height_px,
                frame_stride=frame_stride,
                max_frames=max_frames,
                diagnostic_panels=False,
                material_profile="pbr",
                material_variant=material_variant,
                material_map_path=material_map,
            )
            map_records.append(
                {
                    "label": label,
                    "material_map": material_map,
                    "video": result.output_video_path,
                    "preview": result.preview_frame_path,
                    "metadata": result.metadata_path,
                    "frame_count": result.frame_count,
                    "fps": result.fps,
                    "width_px": result.width_px,
                    "height_px": result.height_px,
                }
            )
        grid_video_path = case_dir / f"{safe_path_id}_pbr_material_map_grid.mp4"
        grid_preview_path = case_dir / f"{safe_path_id}_pbr_material_map_grid_preview.png"
        grid_metadata_path = case_dir / f"{safe_path_id}_pbr_material_map_grid_metadata.json"
        labels = tuple(str(record["label"]) for record in map_records)
        composition = _compose_variant_grid_video(
            variant_video_paths=[Path(record["video"]) for record in map_records],
            labels=labels,
            output_video_path=grid_video_path,
            preview_frame_path=grid_preview_path,
            fps=float(map_records[0]["fps"]),
        )
        metadata = {
            "schema_version": "1.0",
            "case_id": resolved_case_id,
            "path_id": path_id,
            "material_profile": "pbr",
            "material_variant": material_variant,
            "material_maps": [
                {
                    **record,
                    "material_map": str(record["material_map"]) if record["material_map"] is not None else None,
                    "video": str(record["video"]),
                    "preview": str(record["preview"]),
                    "metadata": str(record["metadata"]),
                }
                for record in map_records
            ],
            "grid": {
                "video": str(grid_video_path),
                "preview": str(grid_preview_path),
                **composition,
            },
        }
        grid_metadata_path.write_text(_json_dumps(metadata), encoding="utf-8")
    except (BronchoGenManifestError, ManifestValidationError, RuntimeError, ValueError) as exc:
        _abort(str(exc))
    typer.echo(f"Generated PBR material-map grid -> {grid_video_path}")


@app.command("export-material-atlas")
def export_material_atlas_command(
    mesh_case: Annotated[
        Path,
        typer.Option(
            "--mesh-case",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Canonical mesh case directory containing airway_mesh.ply.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Root directory for editable material atlas exports."),
    ] = Path("runs/texture-material-proof-v1"),
    case_id: Annotated[
        str | None,
        typer.Option("--case-id", help="Case ID for the material atlas output directory."),
    ] = None,
    material_variant: Annotated[
        str,
        typer.Option(
            "--material-variant",
            help=f"PBR mucosa material variant to export: {', '.join(sorted(MATERIAL_VARIANTS))}.",
        ),
    ] = "healthy",
    material_map: Annotated[
        Path | None,
        typer.Option(
            "--material-map",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Optional authorable JSON PBR material map applied before atlas export.",
        ),
    ] = None,
    atlas_width_px: Annotated[
        int,
        typer.Option("--atlas-width", min=16, help="Editable texture atlas width in pixels."),
    ] = 512,
    atlas_height_px: Annotated[
        int,
        typer.Option("--atlas-height", min=16, help="Editable texture atlas height in pixels."),
    ] = 256,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing atlas export directory."),
    ] = False,
) -> None:
    """Export editable PNG material-channel atlases from the SynAirG PBR mucosa model."""
    try:
        resolved_case_id = validate_case_id(case_id or f"{mesh_case.name}-pbr-texture-atlas")
        result = export_pbr_texture_atlas(
            mesh_path=_mesh_path(mesh_case),
            output_root=output / resolved_case_id,
            material_variant=material_variant,
            material_map_path=material_map,
            atlas_width_px=atlas_width_px,
            atlas_height_px=atlas_height_px,
            overwrite=overwrite,
        )
    except (RuntimeError, ValueError) as exc:
        _abort(str(exc))
    typer.echo(f"Generated editable PBR material atlas -> {result.metadata_path}")


@app.command("export-material-mdl")
def export_material_mdl_command(
    mesh_case: Annotated[
        Path,
        typer.Option(
            "--mesh-case",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Canonical mesh case directory containing airway_mesh.ply.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Root directory for USD/MDL material exports."),
    ] = Path("runs/texture-material-proof-v1"),
    case_id: Annotated[
        str | None,
        typer.Option("--case-id", help="Case ID for the USD/MDL output directory."),
    ] = None,
    material_variant: Annotated[
        str,
        typer.Option(
            "--material-variant",
            help=f"PBR mucosa material variant to export: {', '.join(sorted(MATERIAL_VARIANTS))}.",
        ),
    ] = "healthy",
    material_map: Annotated[
        Path | None,
        typer.Option(
            "--material-map",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Optional authorable JSON PBR material map applied before USD/MDL export.",
        ),
    ] = None,
    atlas_width_px: Annotated[
        int,
        typer.Option("--atlas-width", min=16, help="Texture atlas width in pixels."),
    ] = 1024,
    atlas_height_px: Annotated[
        int,
        typer.Option("--atlas-height", min=16, help="Texture atlas height in pixels."),
    ] = 512,
    mdl_shader_target: Annotated[
        str,
        typer.Option(
            "--mdl-shader-target",
            help=f"MDL material target: {', '.join(sorted(MDL_SHADER_TARGETS))}.",
        ),
    ] = "omnipbr-clearcoat",
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing USD/MDL export directory."),
    ] = False,
) -> None:
    """Export an Omniverse-ready USD scene with MDL OmniPBR material bindings."""
    try:
        resolved_case_id = validate_case_id(case_id or f"{mesh_case.name}-pbr-mdl")
        result = export_pbr_mdl_bundle(
            mesh_path=_mesh_path(mesh_case),
            output_root=output / resolved_case_id,
            material_variant=material_variant,
            material_map_path=material_map,
            atlas_width_px=atlas_width_px,
            atlas_height_px=atlas_height_px,
            mdl_shader_target=mdl_shader_target,
            overwrite=overwrite,
        )
    except (RuntimeError, ValueError) as exc:
        _abort(str(exc))
    typer.echo(f"Generated Omniverse USD/MDL PBR bundle -> {result.metadata_path}")


def _mesh_path(mesh_case: Path) -> Path:
    for filename in ("airway_mesh.ply", "airway_mesh.obj"):
        path = mesh_case / filename
        if path.is_file():
            return path
    raise ValueError(f"missing airway mesh in {mesh_case}")


def _parse_material_variants(value: str) -> tuple[str, ...]:
    variants = tuple(item.strip() for item in value.split(",") if item.strip())
    if not variants:
        raise ValueError("at least one material variant is required")
    invalid = [variant for variant in variants if variant not in MATERIAL_VARIANTS]
    if invalid:
        raise ValueError(f"unsupported material variant(s): {', '.join(invalid)}")
    if len(set(variants)) != len(variants):
        raise ValueError("material variants must be unique")
    return variants


def _parse_material_map_specs(value: str) -> tuple[tuple[str, Path | None], ...]:
    specs: list[tuple[str, Path | None]] = []
    for raw_item in (item.strip() for item in value.split(",")):
        if not raw_item:
            continue
        if "=" in raw_item:
            raw_label, raw_path = raw_item.split("=", 1)
            label = raw_label.strip()
            path_text = raw_path.strip()
        else:
            path_text = raw_item
            label = _material_map_label_from_path(Path(path_text))
        if not label:
            raise ValueError("material map labels must not be empty")
        if path_text.lower() in {"none", "baseline", "unmapped"}:
            material_map = None
        else:
            material_map = Path(path_text)
            if not material_map.is_file():
                raise ValueError(f"material map does not exist: {material_map}")
            label = label or _material_map_label_from_path(material_map)
        specs.append((label, material_map))
    if not specs:
        raise ValueError("at least one material map entry is required")
    labels = [label for label, _ in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("material map labels must be unique")
    return tuple(specs)


def _material_map_label_from_path(path: Path) -> str:
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and isinstance(payload.get("name"), str) and payload["name"].strip():
            return payload["name"].strip()
    return path.stem


def _safe_case_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in value.strip().lower())
    return safe.strip("-") or "material-map"


def _compose_variant_grid_video(
    *,
    variant_video_paths: list[Path],
    labels: tuple[str, ...],
    output_video_path: Path,
    preview_frame_path: Path,
    fps: float,
) -> dict[str, int | float]:
    try:
        import imageio.v2 as imageio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("imageio is required to compose material variant grids") from exc
    if len(variant_video_paths) != len(labels):
        raise ValueError("variant_video_paths and labels must have the same length")
    readers = [imageio.get_reader(path) for path in variant_video_paths]
    writer = imageio.get_writer(
        output_video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    preview_frame: np.ndarray | None = None
    frame_count = 0
    try:
        for frames in zip(*(reader.iter_data() for reader in readers), strict=False):
            frame_arrays = [np.asarray(frame[..., :3], dtype=np.uint8) for frame in frames]
            grid = _compose_labeled_grid_frame(frame_arrays, labels)
            if frame_count == 0:
                preview_frame = grid
            writer.append_data(grid)
            frame_count += 1
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    if frame_count == 0 or preview_frame is None:
        raise ValueError("no frames were available to compose the material variant grid")
    imageio.imwrite(preview_frame_path, preview_frame)
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width_px": int(preview_frame.shape[1]),
        "height_px": int(preview_frame.shape[0]),
        "column_count": _grid_shape(len(labels))[0],
        "row_count": _grid_shape(len(labels))[1],
    }


def _compose_labeled_grid_frame(frames: list[np.ndarray], labels: tuple[str, ...]) -> np.ndarray:
    if len(frames) != len(labels):
        raise ValueError("frames and labels must have the same length")
    if not frames:
        raise ValueError("at least one frame is required")
    tile_height, tile_width = frames[0].shape[:2]
    columns, rows = _grid_shape(len(frames))
    canvas = np.zeros((tile_height * rows, tile_width * columns, 3), dtype=np.uint8)
    for index, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        if frame.shape[:2] != (tile_height, tile_width):
            raise ValueError("all frames must have the same dimensions")
        row = index // columns
        column = index % columns
        y0 = row * tile_height
        x0 = column * tile_width
        canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = frame[..., :3]
        _draw_grid_label(canvas, x0, y0, label)
    return canvas


def _grid_shape(count: int) -> tuple[int, int]:
    if count < 1:
        raise ValueError("count must be positive")
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / columns))
    return columns, rows


def _draw_grid_label(image: np.ndarray, x0: int, y0: int, label: str) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except ImportError:
        return
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image, "RGBA")
    font = ImageFont.load_default()
    text = label.replace("_", " ").title()
    bbox = draw.textbbox((0, 0), text, font=font)
    padding_x = 7
    padding_y = 5
    left = x0 + 10
    top = y0 + 10
    draw.rounded_rectangle(
        (left, top, left + bbox[2] - bbox[0] + padding_x * 2, top + bbox[3] - bbox[1] + padding_y * 2),
        radius=4,
        fill=(0, 0, 0, 172),
    )
    draw.text((left + padding_x, top + padding_y - 1), text, font=font, fill=(248, 245, 238, 235))
    image[:] = np.asarray(pil_image, dtype=np.uint8)


def _json_dumps(value: object) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _abort(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(1)
