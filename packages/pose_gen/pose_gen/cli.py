"""CLI group for SynAirG pose workflows."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer
from synairg_core.volume_manifest import ManifestValidationError, validate_case_id

from pose_gen.paths import generate_scope_paths, write_scope_path_review_png

DEFAULT_OUTPUT_ROOT = Path("datasets/poses")

app = typer.Typer(
    help="Generate bronchoscopy traversal paths, camera poses, and review renders.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def callback() -> None:
    """Show commands for bronchoscopy camera path generation workflows."""


@app.command()
def status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable status payload."),
    ] = False,
) -> None:
    """Report whether pose workflow commands are available."""
    if json_output:
        typer.echo('{"stage":"pose","status":"ready","message":"pose_gen commands are installed"}')
        return
    typer.echo("pose_gen commands are installed")


@app.command("run")
def run_command(
    mesh_case: Annotated[
        Path,
        typer.Option(
            "--mesh-case",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Canonical mesh case directory containing centerline_graph.json and airway_mesh.ply.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Root directory for canonical pose cases."),
    ] = DEFAULT_OUTPUT_ROOT,
    case_id: Annotated[
        str | None,
        typer.Option("--case-id", help="Case ID to use for the canonical pose output directory."),
    ] = None,
    path_count: Annotated[
        int,
        typer.Option("--path-count", min=1, max=6, help="Number of distal bronchoscopy paths to sample."),
    ] = 3,
    frustum_every_frames: Annotated[
        int,
        typer.Option(
            "--frustum-every-frames",
            min=1,
            help="Render camera frustums at this frame interval in the review artifact.",
        ),
    ] = 8,
    render_review: Annotated[
        bool,
        typer.Option(
            "--render-review/--no-render-review",
            help="Render a PNG with airway mesh, scope paths, and sparse camera frustums.",
        ),
    ] = True,
    include_procedure: Annotated[
        bool,
        typer.Option(
            "--include-procedure/--no-procedure",
            help="Append one smooth multi-target procedure path with insertion, retraction, and redirection phases.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing pose case directory."),
    ] = False,
) -> None:
    """Generate bronchoscope camera paths from a mesh case centerline."""
    try:
        resolved_case_id = validate_case_id(case_id or mesh_case.name)
        case_dir = output / resolved_case_id
        if case_dir.exists():
            if not overwrite and any(case_dir.iterdir()):
                _abort(f"pose case already exists: {case_dir}; pass --overwrite to replace it")
            if overwrite:
                shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        centerline_graph_path = mesh_case / "centerline_graph.json"
        if not centerline_graph_path.is_file():
            _abort(f"missing centerline graph: {centerline_graph_path}")

        paths_json_path = case_dir / "scope_paths.json"
        generate_scope_paths(
            centerline_graph_path=centerline_graph_path,
            output_path=paths_json_path,
            path_count=path_count,
            frustum_every_frames=frustum_every_frames,
            include_procedure=include_procedure,
        )
        if render_review:
            mesh_path = _mesh_path(mesh_case)
            write_scope_path_review_png(
                mesh_path=mesh_path,
                paths_json_path=paths_json_path,
                output_path=case_dir / "scope_paths_review.png",
            )
    except (ManifestValidationError, RuntimeError, ValueError) as exc:
        _abort(str(exc))
    typer.echo(f"Generated {resolved_case_id} -> {case_dir}")


def _mesh_path(mesh_case: Path) -> Path:
    for filename in ("airway_mesh.ply", "airway_mesh.obj"):
        path = mesh_case / filename
        if path.is_file():
            return path
    raise ValueError(f"missing airway mesh in {mesh_case}")


def _abort(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(1)
