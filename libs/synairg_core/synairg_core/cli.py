"""Top-level SynAirG command-line application."""

from __future__ import annotations

from typing import Annotated

import typer
from mesh_gen.cli import app as mesh_app
from pose_gen.cli import app as pose_app
from trainer.cli import app as train_app
from video_gen.cli import app as video_app
from volume_gen.cli import app as volume_app

from synairg_core import __version__
from synairg_core.cli_groups import build_stub_group
from synairg_core.pipeline_cli import app as pipeline_app

app = typer.Typer(
    name="synairg",
    help="Synthetic airway generation workflow CLI.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the installed SynAirG version and exit."),
    ] = False,
) -> None:
    """Coordinate SynAirG pipeline stages and developer workflows."""
    if version:
        typer.echo(f"synairg {__version__}")
        raise typer.Exit()


app.add_typer(volume_app, name="volume")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(mesh_app, name="mesh")
app.add_typer(pose_app, name="pose")
app.add_typer(video_app, name="video")
app.add_typer(train_app, name="train")
app.add_typer(
    build_stub_group(
        stage="validate",
        summary="Validation and QA commands are installed; validation backends will arrive in later issues.",
        help_text="Validate SynAirG volumes, meshes, poses, videos, and dataset manifests.",
    ),
    name="validate",
)
app.add_typer(
    build_stub_group(
        stage="publish",
        summary="Dataset publishing commands are installed; publishing backends will arrive in later issues.",
        help_text="Publish SynAirG datasets, manifests, and dataset cards.",
    ),
    name="publish",
)


def main() -> None:
    """Console-script entrypoint."""
    app()
