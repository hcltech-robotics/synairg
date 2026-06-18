"""Helpers for constructing skeletal SynAirG CLI command groups."""

from __future__ import annotations

from typing import Annotated

import typer


def build_stub_group(*, stage: str, summary: str, help_text: str) -> typer.Typer:
    """Create a command group with a lightweight status command."""
    group = typer.Typer(help=help_text, no_args_is_help=True, rich_markup_mode="rich")

    @group.callback()
    def callback() -> None:
        """Show commands for this SynAirG stage."""

    @group.command()
    def status(
        json_output: Annotated[
            bool,
            typer.Option("--json", help="Emit a machine-readable status payload."),
        ] = False,
    ) -> None:
        """Report whether the command group is available."""
        if json_output:
            typer.echo(f'{{"stage":"{stage}","status":"stub","message":"{summary}"}}')
            return
        typer.echo(summary)

    return group
