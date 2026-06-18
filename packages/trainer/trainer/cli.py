"""CLI group for SynAirG training workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trainer.bronchogen_manifest import (
    BronchoGenManifestConfig,
    BronchoGenManifestError,
    validate_bronchogen_manifest,
    write_bronchogen_manifest,
)
from trainer.public_bronchoscopy_data import (
    BMBronchoLCAuditConfig,
    BMBronchoLCDownloadConfig,
    BMBronchoLCSplitConfig,
    PublicBronchoscopyDataError,
    write_bm_broncholc_audit,
    write_bm_broncholc_download_report,
    write_bm_broncholc_splits,
)
from trainer.temporal_metrics import TemporalMetricConfig, TemporalMetricError, write_blind_temporal_metrics

app = typer.Typer(
    help="Train and evaluate SynAirG depth, segmentation, pose, and generative rendering models.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def callback() -> None:
    """Show commands for SynAirG training workflows."""


@app.command()
def status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable status payload."),
    ] = False,
) -> None:
    """Report whether trainer workflow commands are available."""
    if json_output:
        typer.echo('{"stage":"train","status":"ready","message":"trainer commands are installed"}')
        return
    typer.echo("trainer commands are installed")


@app.command("build-bronchogen-manifest")
def build_bronchogen_manifest_command(
    rgb_root: Annotated[
        Path,
        typer.Option(
            "--rgb-root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Directory containing RGB bronchoscopy frames, grouped by sequence folders.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=True, dir_okay=False, help="Output manifest JSON path."),
    ],
    case_id: Annotated[
        str,
        typer.Option("--case-id", help="Case ID recorded in the manifest."),
    ],
    dataset_name: Annotated[
        str,
        typer.Option("--dataset-name", help="Short dataset name recorded in the manifest."),
    ] = "bronchogen",
    caption: Annotated[
        str,
        typer.Option("--caption", help="Default caption/prompt for RGB frames."),
    ] = "bronchoscopic airway view",
    depth_root: Annotated[
        Path | None,
        typer.Option("--depth-root", file_okay=False, dir_okay=True, readable=True, help="Optional depth maps root."),
    ] = None,
    normal_root: Annotated[
        Path | None,
        typer.Option(
            "--normal-root",
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Optional surface normal maps root.",
        ),
    ] = None,
    pps_root: Annotated[
        Path | None,
        typer.Option(
            "--pps-root",
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Optional per-pixel shading maps root.",
        ),
    ] = None,
    mask_root: Annotated[
        Path | None,
        typer.Option("--mask-root", file_okay=False, dir_okay=True, readable=True, help="Optional airway mask root."),
    ] = None,
    edge_root: Annotated[
        Path | None,
        typer.Option("--edge-root", file_okay=False, dir_okay=True, readable=True, help="Optional edge map root."),
    ] = None,
    flow_root: Annotated[
        Path | None,
        typer.Option(
            "--flow-root",
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Optional optical-flow map root.",
        ),
    ] = None,
    orifice_mask_root: Annotated[
        Path | None,
        typer.Option(
            "--orifice-mask-root",
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Optional bronchial-orifice mask root.",
        ),
    ] = None,
    sequence_depth: Annotated[
        int,
        typer.Option(
            "--sequence-depth",
            min=0,
            help="Number of relative RGB path directories used as the sequence ID.",
        ),
    ] = 1,
    temporal_context_radius: Annotated[
        int,
        typer.Option(
            "--temporal-context-radius",
            min=0,
            help="Number of previous/next frames exposed for temporal consistency losses.",
        ),
    ] = 2,
    fps: Annotated[
        float | None,
        typer.Option("--fps", min=0.0001, help="Optional sequence frame rate for timestamp_seconds."),
    ] = None,
    require_conditions: Annotated[
        bool,
        typer.Option(
            "--require-conditions/--allow-missing-conditions",
            help="Fail when a condition root is missing the matching relative frame.",
        ),
    ] = True,
) -> None:
    """Build a temporally grouped manifest for BronchoGen diffusion training."""
    condition_roots = {
        name: root
        for name, root in {
            "depth": depth_root,
            "normal": normal_root,
            "pps": pps_root,
            "mask": mask_root,
            "edge": edge_root,
            "flow": flow_root,
            "orifice_mask": orifice_mask_root,
        }.items()
        if root is not None
    }
    try:
        manifest = write_bronchogen_manifest(
            BronchoGenManifestConfig(
                rgb_root=rgb_root,
                case_id=case_id,
                dataset_name=dataset_name,
                caption=caption,
                condition_roots=condition_roots,
                output_path=output,
                sequence_depth=sequence_depth,
                temporal_context_radius=temporal_context_radius,
                fps=fps,
                require_conditions=require_conditions,
            )
        )
        validate_bronchogen_manifest(output)
    except BronchoGenManifestError as exc:
        _abort(str(exc))
    frames_value = manifest.get("frames")
    frame_count = len(frames_value) if isinstance(frames_value, list) else 0
    typer.echo(f"Generated BronchoGen manifest with {frame_count} frames -> {output}")


@app.command("evaluate-temporal-consistency")
def evaluate_temporal_consistency_command(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Generated bronchoscopy video to evaluate.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=True, dir_okay=False, help="Output metric JSON path."),
    ],
    frame_stride: Annotated[
        int,
        typer.Option("--frame-stride", min=1, help="Evaluate every Nth frame."),
    ] = 1,
    max_frames: Annotated[
        int | None,
        typer.Option("--max-frames", min=2, help="Optional cap for fast audits."),
    ] = None,
) -> None:
    """Compute blind no-ground-truth temporal consistency metrics for a generated video."""
    try:
        metrics = write_blind_temporal_metrics(
            TemporalMetricConfig(video_path=video, frame_stride=frame_stride, max_frames=max_frames),
            output,
        )
    except TemporalMetricError as exc:
        _abort(str(exc))
    typer.echo(
        "Temporal metrics "
        f"flicker_mean={metrics['flicker_energy_mean']:.6g} "
        f"raw_absdiff_mean={metrics['raw_absdiff_mean']:.6g} -> {output}"
    )


@app.command("download-bm-broncholc")
def download_bm_broncholc_command(
    root: Annotated[
        Path,
        typer.Option(
            "--root",
            file_okay=False,
            dir_okay=True,
            help="Dataset root for BM-BronchoLC raw archives and extracted files.",
        ),
    ] = Path("datasets/real/bm_broncholc"),
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=True,
            dir_okay=False,
            help="Output provenance JSON path.",
        ),
    ] = Path("data/bm_broncholc_download_report.json"),
    extract: Annotated[
        bool,
        typer.Option("--extract/--no-extract", help="Extract verified archives into root/extracted."),
    ] = True,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Redownload archives even when verified files already exist."),
    ] = False,
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout-seconds", min=1.0, help="Network timeout per blocking read operation."),
    ] = 60.0,
) -> None:
    """Download and verify the public BM-BronchoLC Figshare dataset."""
    try:
        report = write_bm_broncholc_download_report(
            BMBronchoLCDownloadConfig(
                root=root,
                extract=extract,
                overwrite=overwrite,
                timeout_seconds=timeout_seconds,
            ),
            output,
        )
    except PublicBronchoscopyDataError as exc:
        _abort(str(exc))
    archives = report.get("archives", [])
    archive_count = len(archives) if isinstance(archives, list) else 0
    typer.echo(f"BM-BronchoLC verified {archive_count} archives -> {output}")


@app.command("audit-bm-broncholc")
def audit_bm_broncholc_command(
    root: Annotated[
        Path,
        typer.Option(
            "--root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="BM-BronchoLC root or extracted directory to audit.",
        ),
    ] = Path("datasets/real/bm_broncholc"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=True, dir_okay=False, help="Output audit JSON path."),
    ] = Path("data/bm_broncholc_audit.json"),
    sample_image_limit: Annotated[
        int,
        typer.Option("--sample-image-limit", min=0, help="Maximum sample dimensions stored per category."),
    ] = 16,
) -> None:
    """Audit BM-BronchoLC patient/video/image structure for no-cheating splits."""
    try:
        report = write_bm_broncholc_audit(
            BMBronchoLCAuditConfig(root=root, sample_image_limit=sample_image_limit),
            output,
        )
    except PublicBronchoscopyDataError as exc:
        _abort(str(exc))
    image_summary = report.get("image_summary", {})
    total_images = image_summary.get("total_images", 0) if isinstance(image_summary, dict) else 0
    typer.echo(f"BM-BronchoLC audit total_images={total_images} -> {output}")


@app.command("split-bm-broncholc")
def split_bm_broncholc_command(
    root: Annotated[
        Path,
        typer.Option(
            "--root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="BM-BronchoLC root or extracted directory to split.",
        ),
    ] = Path("datasets/real/bm_broncholc"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=True, dir_okay=False, help="Output split JSON path."),
    ] = Path("data/bm_broncholc_patient_splits.json"),
    train_fraction: Annotated[
        float,
        typer.Option("--train-fraction", min=0.01, max=0.98, help="Patient fraction assigned to train."),
    ] = 0.70,
    validation_fraction: Annotated[
        float,
        typer.Option("--validation-fraction", min=0.01, max=0.98, help="Patient fraction assigned to validation."),
    ] = 0.15,
    test_fraction: Annotated[
        float,
        typer.Option("--test-fraction", min=0.01, max=0.98, help="Patient fraction assigned to test."),
    ] = 0.15,
    salt: Annotated[
        str,
        typer.Option("--salt", help="Stable salt for deterministic patient assignment."),
    ] = "synairg-bm-broncholc-patient-split-v1",
) -> None:
    """Create deterministic patient-level BM-BronchoLC train/validation/test splits."""
    try:
        report = write_bm_broncholc_splits(
            BMBronchoLCSplitConfig(
                root=root,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                salt=salt,
            ),
            output,
        )
    except PublicBronchoscopyDataError as exc:
        _abort(str(exc))
    leakage_checks = report.get("leakage_checks", {})
    assigned_patient_count = (
        leakage_checks.get("assigned_patient_count", 0) if isinstance(leakage_checks, dict) else 0
    )
    typer.echo(f"BM-BronchoLC split patients={assigned_patient_count} -> {output}")


def _abort(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(1)
