"""CLI group for SynAirG mesh workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mesh_gen.errors import MeshGenError
from mesh_gen.pipeline import MeshRunConfig, derive_case_id, run_mesh_case

DEFAULT_OUTPUT_ROOT = Path("datasets/meshes")
BRONCHOSCOPE_PROFILES: dict[str, tuple[float, float, str]] = {
    "ultrathin-mp190f": (3.0, 600.0, "Olympus BF-MP190F distal-end outer diameter and working length"),
    "diagnostic-p190": (4.2, 600.0, "Olympus BF-P190 distal-end outer diameter and working length"),
    "therapeutic-1th190": (6.2, 600.0, "Olympus BF-1TH190 distal-end outer diameter and working length"),
}

app = typer.Typer(
    help="Generate airway masks, centrelines, topology reports, and mesh assets.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def callback() -> None:
    """Show commands for airway mesh generation workflows."""


@app.command()
def status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable status payload."),
    ] = False,
) -> None:
    """Report whether mesh workflow commands are available."""
    if json_output:
        typer.echo('{"stage":"mesh","status":"ready","message":"mesh_gen commands are installed"}')
        return
    typer.echo("mesh_gen commands are installed")


@app.command("run")
def run_command(
    ct: Annotated[
        Path,
        typer.Option(
            "--ct",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Local CT NIfTI volume.",
        ),
    ],
    mask: Annotated[
        Path | None,
        typer.Option(
            "--mask",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Precomputed binary airway mask NIfTI.",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Root directory for canonical mesh cases."),
    ] = DEFAULT_OUTPUT_ROOT,
    case_id: Annotated[
        str | None,
        typer.Option("--case-id", help="Case ID to use for the canonical output directory."),
    ] = None,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="Segmentation backend to use: precomputed-mask, external-command, or ct-threshold.",
        ),
    ] = "precomputed-mask",
    threshold_hu: Annotated[
        float,
        typer.Option("--threshold-hu", help="HU threshold when --backend ct-threshold is selected."),
    ] = -500.0,
    segmenter_command: Annotated[
        str | None,
        typer.Option(
            "--segmenter-command",
            help=(
                "External segmenter command for --backend external-command. "
                "Use {ct} for input CT and {mask} for output mask."
            ),
        ),
    ] = None,
    segmenter_timeout_seconds: Annotated[
        float,
        typer.Option("--segmenter-timeout-seconds", min=1.0, help="External segmenter timeout."),
    ] = 3600.0,
    strict_airway_validation: Annotated[
        bool,
        typer.Option(
            "--strict-airway-validation/--allow-non-airway-mask",
            help="Reject implausible large organ/lung masks before meshing.",
        ),
    ] = False,
    bronchoscope_profile: Annotated[
        str | None,
        typer.Option(
            "--bronchoscope-profile",
            help=(
                "Named bronchoscope access profile overriding --min-accessible-diameter-mm and "
                "--max-accessible-depth-mm. "
                "Options: ultrathin-mp190f, diagnostic-p190, therapeutic-1th190."
            ),
        ),
    ] = None,
    min_accessible_diameter_mm: Annotated[
        float,
        typer.Option(
            "--min-accessible-diameter-mm",
            min=0.1,
            help="Minimum airway diameter reachable from the tracheal root for bronchoscopy filtering.",
        ),
    ] = 3.0,
    max_accessible_depth_mm: Annotated[
        float | None,
        typer.Option(
            "--max-accessible-depth-mm",
            min=1.0,
            help="Maximum centerline distance from the tracheal root retained for bronchoscopy filtering.",
        ),
    ] = None,
    filter_to_accessible_diameter: Annotated[
        bool,
        typer.Option(
            "--filter-accessible-diameter/--keep-inaccessible-diameter",
            help="Remove airway voxels nearest to centreline nodes outside the requested bronchoscope reach.",
        ),
    ] = True,
    laryngeal_capping: Annotated[
        bool,
        typer.Option(
            "--laryngeal-capping/--keep-laryngeal-components",
            help="Remove disconnected superior laryngeal remnants before topology bridging.",
        ),
    ] = True,
    min_component_voxels: Annotated[
        int,
        typer.Option(
            "--min-component-voxels",
            min=1,
            help="Minimum disconnected airway component size considered for topology bridging.",
        ),
    ] = 20,
    max_bridge_distance_mm: Annotated[
        float,
        typer.Option(
            "--max-bridge-distance-mm",
            min=0.0,
            help="Maximum physical gap bridged between sizeable airway components.",
        ),
    ] = 45.0,
    refinement_preset: Annotated[
        str,
        typer.Option(
            "--refinement-preset",
            help="Mesh refinement preset: raw, render-ready, or high-detail.",
        ),
    ] = "raw",
    smooth_iterations: Annotated[
        int,
        typer.Option("--smooth-iterations", min=0, help="Smoothing iterations to apply; presets fill this in when 0."),
    ] = 0,
    smoothing_method: Annotated[
        str,
        typer.Option(
            "--smoothing-method",
            help="Smoothing method: auto, laplacian, taubin, or none.",
        ),
    ] = "auto",
    taubin_pass_band: Annotated[
        float,
        typer.Option(
            "--taubin-pass-band",
            min=0.0001,
            help="Taubin/windowed-sinc pass band when Taubin smoothing is selected.",
        ),
    ] = 0.08,
    decimate_fraction: Annotated[
        float,
        typer.Option("--decimate-fraction", min=0.0, max=0.99, help="Fraction of faces to drop after smoothing."),
    ] = 0.0,
    normal_orientation: Annotated[
        str,
        typer.Option(
            "--normal-orientation",
            help="Mesh normal orientation: auto, outward, or inward. Render-ready presets default to inward.",
        ),
    ] = "auto",
    hollow_mesh: Annotated[
        bool,
        typer.Option(
            "--hollow/--sealed",
            help="Uncap the superior/proximal airway top so the surface mesh is traversable from the entry lumen.",
        ),
    ] = True,
    top_opening_radius_scale: Annotated[
        float,
        typer.Option(
            "--top-opening-radius-scale",
            min=0.1,
            help="Radius multiplier used when cutting the superior/proximal top opening.",
        ),
    ] = 1.25,
    proximal_opening_trim: Annotated[
        bool,
        typer.Option(
            "--proximal-opening-trim/--legacy-top-uncap",
            help="Trim the proximal laryngeal end with a cut plane to produce one bronchoscope entry opening.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing case directory."),
    ] = False,
) -> None:
    """Process a local CT plus airway mask into mesh, centreline, QA, and preview artifacts."""
    resolved_case_id = case_id if case_id is not None else derive_case_id(ct)
    resolved_min_accessible_diameter_mm = min_accessible_diameter_mm
    resolved_max_accessible_depth_mm = max_accessible_depth_mm
    bronchoscope_profile_note: str | None = None
    if bronchoscope_profile is not None:
        try:
            (
                resolved_min_accessible_diameter_mm,
                resolved_max_accessible_depth_mm,
                bronchoscope_profile_note,
            ) = BRONCHOSCOPE_PROFILES[bronchoscope_profile]
        except KeyError:
            _abort(
                "unknown bronchoscope profile "
                f"{bronchoscope_profile!r}; expected one of {', '.join(sorted(BRONCHOSCOPE_PROFILES))}"
            )
    try:
        mesh_case = run_mesh_case(
            MeshRunConfig(
                ct_path=ct,
                mask_path=mask,
                output_root=output,
                case_id=resolved_case_id,
                backend=backend,
                threshold_hu=threshold_hu,
                segmenter_command=segmenter_command,
                segmenter_timeout_seconds=segmenter_timeout_seconds,
                strict_airway_validation=strict_airway_validation,
                bronchoscope_profile=(
                    f"{bronchoscope_profile}: {bronchoscope_profile_note}" if bronchoscope_profile_note else None
                ),
                min_accessible_diameter_mm=resolved_min_accessible_diameter_mm,
                max_accessible_depth_mm=resolved_max_accessible_depth_mm,
                filter_to_accessible_diameter=filter_to_accessible_diameter,
                laryngeal_capping=laryngeal_capping,
                min_component_voxels=min_component_voxels,
                max_bridge_distance_mm=max_bridge_distance_mm,
                refinement_preset=refinement_preset,
                smooth_iterations=smooth_iterations,
                smoothing_method=smoothing_method,
                taubin_pass_band=taubin_pass_band,
                decimate_fraction=decimate_fraction,
                normal_orientation=normal_orientation,
                hollow_mesh=hollow_mesh,
                top_opening_radius_scale=top_opening_radius_scale,
                proximal_opening_trim=proximal_opening_trim,
                overwrite=overwrite,
            )
        )
    except MeshGenError as exc:
        _abort(str(exc))
    typer.echo(f"Generated {mesh_case.case_id} -> {mesh_case.paths.case_dir}")


def _abort(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(1)
