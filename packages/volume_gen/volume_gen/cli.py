"""CLI group for SynAirG volume workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from synairg_core.volume_manifest import JsonValue, ManifestValidationError

from volume_gen.errors import BackendUnavailableError, VolumeGenError
from volume_gen.importers import import_dicom_folder, import_local_nifti
from volume_gen.maisi import generate_maisi_case
from volume_gen.tcia import list_tcia_series, pull_tcia_case

DEFAULT_OUTPUT_ROOT = Path("datasets/volumes")

app = typer.Typer(
    help="Import, generate, and provenance CT volumes for SynAirG cases.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def callback() -> None:
    """Show commands for CT volume workflows."""


@app.command()
def status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable status payload."),
    ] = False,
) -> None:
    """Report whether the volume workflow commands are available."""
    if json_output:
        typer.echo(
            '{"stage":"volume","status":"ready","message":"volume_gen import and generation commands are installed"}'
        )
        return
    typer.echo("volume_gen import and generation commands are installed")


@app.command("import-local")
def import_local_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="JSON manifest describing a local NIfTI volume.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Root directory for canonical volume cases."),
    ] = DEFAULT_OUTPUT_ROOT,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing case directory."),
    ] = False,
) -> None:
    """Import a local .nii or .nii.gz file as a canonical SynAirG CT case."""
    try:
        volume_case = import_local_nifti(manifest, output, overwrite=overwrite)
    except (ManifestValidationError, VolumeGenError) as exc:
        _abort(str(exc))
    typer.echo(f"Imported {volume_case.case_id} -> {volume_case.case_dir}")


@app.command("import-dicom")
def import_dicom_command(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Folder containing CT DICOM slices.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Root directory for canonical volume cases."),
    ] = DEFAULT_OUTPUT_ROOT,
    case_id: Annotated[
        str | None,
        typer.Option("--case-id", help="Case ID to use for the canonical output directory."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing case directory."),
    ] = False,
) -> None:
    """Convert a local DICOM folder to canonical ct.nii.gz plus sidecars."""
    try:
        volume_case = import_dicom_folder(input_dir, output, case_id=case_id, overwrite=overwrite)
    except (ManifestValidationError, VolumeGenError) as exc:
        _abort(str(exc))
    typer.echo(f"Imported {volume_case.case_id} -> {volume_case.case_dir}")


@app.command("pull-tcia")
def pull_tcia_command(
    collection: Annotated[
        str,
        typer.Option("--collection", help="TCIA/NBIA collection name to pull from."),
    ],
    series_instance_uid: Annotated[
        str | None,
        typer.Option("--series-instance-uid", help="Specific TCIA/NBIA SeriesInstanceUID to download."),
    ] = None,
    patient_id: Annotated[
        str | None,
        typer.Option("--patient-id", help="Optional PatientID filter when auto-selecting a series."),
    ] = None,
    study_instance_uid: Annotated[
        str | None,
        typer.Option("--study-instance-uid", help="Optional StudyInstanceUID filter when auto-selecting a series."),
    ] = None,
    body_part_examined: Annotated[
        str | None,
        typer.Option(
            "--body-part",
            "--body-part-examined",
            help="Optional BodyPartExamined filter, for example CHEST.",
        ),
    ] = None,
    case_id: Annotated[
        str | None,
        typer.Option("--case-id", help="Case ID to use for the canonical output directory."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Root directory for canonical volume cases."),
    ] = DEFAULT_OUTPUT_ROOT,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing case directory."),
    ] = False,
) -> None:
    """Download a TCIA/NBIA CT series to canonical ct.nii.gz plus sidecars."""
    try:
        volume_case = pull_tcia_case(
            collection,
            output,
            series_instance_uid=series_instance_uid,
            patient_id=patient_id,
            study_instance_uid=study_instance_uid,
            body_part_examined=body_part_examined,
            case_id=case_id,
            overwrite=overwrite,
        )
    except (ManifestValidationError, VolumeGenError) as exc:
        _abort(str(exc))
    typer.echo(f"Pulled {volume_case.case_id} -> {volume_case.case_dir}")


@app.command("list-tcia-series")
def list_tcia_series_command(
    collection: Annotated[
        str,
        typer.Option("--collection", help="TCIA/NBIA collection name to query."),
    ],
    patient_id: Annotated[
        str | None,
        typer.Option("--patient-id", help="Optional PatientID filter."),
    ] = None,
    study_instance_uid: Annotated[
        str | None,
        typer.Option("--study-instance-uid", help="Optional StudyInstanceUID filter."),
    ] = None,
    body_part_examined: Annotated[
        str | None,
        typer.Option(
            "--body-part",
            "--body-part-examined",
            help="Optional BodyPartExamined filter, for example CHEST.",
        ),
    ] = None,
    include_non_ct: Annotated[
        bool,
        typer.Option("--include-non-ct", help="Do not restrict listed series to CT modality."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the matching series metadata as JSON."),
    ] = False,
) -> None:
    """Query TCIA/NBIA series metadata before downloading a selected case."""
    try:
        series = list_tcia_series(
            collection,
            patient_id=patient_id,
            study_instance_uid=study_instance_uid,
            body_part_examined=body_part_examined,
            modality=None if include_non_ct else "CT",
        )
    except VolumeGenError as exc:
        _abort(str(exc))
    if json_output:
        typer.echo(json.dumps(series, indent=2, sort_keys=True))
        return
    for item in series:
        typer.echo(
            "\t".join(
                [
                    _metadata_field(item, "SeriesInstanceUID"),
                    _metadata_field(item, "PatientID"),
                    _metadata_field(item, "StudyInstanceUID"),
                    _metadata_field(item, "Modality"),
                    _metadata_field(item, "BodyPartExamined"),
                ]
            )
        )


@app.command("generate-maisi")
def generate_maisi_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="JSON config for a MAISI-compatible generation backend.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Root directory for generated volume cases."),
    ] = DEFAULT_OUTPUT_ROOT,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing case directory."),
    ] = False,
) -> None:
    """Generate a synthetic CT case through the MAISI backend interface."""
    try:
        volume_case = generate_maisi_case(config, output, overwrite=overwrite)
    except (BackendUnavailableError, ManifestValidationError, VolumeGenError) as exc:
        _abort(str(exc))
    typer.echo(f"Generated {volume_case.case_id} -> {volume_case.case_dir}")


def _abort(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(1)


def _metadata_field(item: dict[str, JsonValue], key: str) -> str:
    value = item.get(key)
    if isinstance(value, str):
        return value
    normalized_key = key.lower()
    for item_key, item_value in item.items():
        if item_key.lower() == normalized_key and isinstance(item_value, str):
            return item_value
    return "-"
