"""CLI group for SynAirG manifest-driven pipeline planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import typer

from synairg_core.pipeline_manifest import build_pipeline_plan, read_pipeline_manifest
from synairg_core.volume_manifest import ManifestValidationError

app = typer.Typer(
    help="Validate and plan complete SynAirG volume-to-render workflows from JSON or YAML manifests.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def callback() -> None:
    """Show commands for manifest-driven pipeline workflows."""


@app.command()
def status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable status payload."),
    ] = False,
) -> None:
    """Report whether pipeline manifest commands are available."""
    if json_output:
        typer.echo('{"stage":"pipeline","status":"ready","message":"pipeline manifests are supported"}')
        return
    typer.echo("pipeline manifests are supported")


@app.command("validate")
def validate_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="SynAirG pipeline manifest in JSON or YAML form.",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the normalised manifest as JSON."),
    ] = False,
) -> None:
    """Validate a manifest and print the normalised SynAirG contract."""
    try:
        parsed = read_pipeline_manifest(manifest)
    except ManifestValidationError as exc:
        _abort(str(exc))
    if json_output:
        typer.echo(json.dumps(parsed.to_dict(), indent=2, sort_keys=True))
        return
    typer.echo(f"Validated {parsed.case_id}")


@app.command("plan")
def plan_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="SynAirG pipeline manifest in JSON or YAML form.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", file_okay=True, dir_okay=False, help="Optional JSON path for the plan."),
    ] = None,
    shell: Annotated[
        bool,
        typer.Option("--shell", help="Print shell commands instead of JSON."),
    ] = False,
) -> None:
    """Materialise the ordered stage commands for a SynAirG manifest."""
    try:
        parsed = read_pipeline_manifest(manifest)
        plan = build_pipeline_plan(parsed)
    except ManifestValidationError as exc:
        _abort(str(exc))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if shell:
        steps = plan.get("steps")
        if not isinstance(steps, list):
            _abort("pipeline plan did not contain a step list")
        for step in cast(list[object], steps):
            if not isinstance(step, dict):
                continue
            command = step.get("shell")
            if isinstance(command, str):
                typer.echo(command)
        return
    typer.echo(json.dumps(plan, indent=2, sort_keys=True))


def _abort(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(1)
