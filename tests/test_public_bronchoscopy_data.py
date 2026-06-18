from __future__ import annotations

import json

from synairg_core.cli import app
from trainer.public_bronchoscopy_data import (
    BMBronchoLCAuditConfig,
    BMBronchoLCSplitConfig,
    audit_bm_broncholc,
    build_bm_broncholc_splits,
)
from typer.testing import CliRunner

runner = CliRunner()

_ONE_BY_ONE_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00"
    b"\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe"
    b"\xdc\xccY\xe7"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_bm_broncholc_audit_counts_patient_video_structure(tmp_path) -> None:
    dataset_root = tmp_path / "bm_broncholc" / "extracted"
    lung_seq = dataset_root / "Lung_cancer" / "imgs" / "patient-a" / "video-1"
    non_lung_seq = dataset_root / "Non_lung_cancer" / "imgs" / "patient-b" / "video-2"
    lung_seq.mkdir(parents=True)
    non_lung_seq.mkdir(parents=True)
    (lung_seq / "000.png").write_bytes(_ONE_BY_ONE_PNG)
    (lung_seq / "001.png").write_bytes(_ONE_BY_ONE_PNG)
    (non_lung_seq / "000.png").write_bytes(_ONE_BY_ONE_PNG)
    (dataset_root / "Lung_cancer" / "annotation.json").write_text("{}", encoding="utf-8")
    (dataset_root / "Non_lung_cancer" / "labels.json").write_text("{}", encoding="utf-8")

    report = audit_bm_broncholc(BMBronchoLCAuditConfig(root=tmp_path / "bm_broncholc", sample_image_limit=1))

    image_summary = report["image_summary"]
    assert isinstance(image_summary, dict)
    assert image_summary["total_images"] == 3
    categories = image_summary["categories"]
    assert isinstance(categories, dict)
    lung = categories["lung_cancer"]
    non_lung = categories["non_lung_cancer"]
    assert isinstance(lung, dict)
    assert isinstance(non_lung, dict)
    assert lung["image_count"] == 2
    assert lung["patient_count"] == 1
    assert lung["video_count"] == 1
    assert lung["metadata_files"] == ["Lung_cancer/annotation.json"]
    assert lung["sample_dimensions"] == [
        {
            "height": 1,
            "path": "Lung_cancer/imgs/patient-a/video-1/000.png",
            "width": 1,
        }
    ]
    assert non_lung["image_count"] == 1
    assert non_lung["metadata_files"] == ["Non_lung_cancer/labels.json"]


def test_train_cli_audits_bm_broncholc_fixture(tmp_path) -> None:
    dataset_root = tmp_path / "bm_broncholc"
    image_dir = dataset_root / "Lung_cancer" / "imgs" / "patient-a" / "video-1"
    image_dir.mkdir(parents=True)
    (image_dir / "000.png").write_bytes(_ONE_BY_ONE_PNG)
    output_path = tmp_path / "audit.json"

    result = runner.invoke(
        app,
        [
            "train",
            "audit-bm-broncholc",
            "--root",
            str(dataset_root),
            "--output",
            str(output_path),
            "--sample-image-limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "total_images=1" in result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["image_summary"]["total_images"] == 1
    assert payload["source"]["license"] == "CC BY 4.0"


def test_bm_broncholc_splits_are_patient_level_and_deterministic(tmp_path) -> None:
    dataset_root = tmp_path / "bm_broncholc"
    for patient_index in range(6):
        image_dir = dataset_root / "Lung_cancer" / "imgs" / f"patient-{patient_index}" / "video-1"
        image_dir.mkdir(parents=True)
        (image_dir / "000.png").write_bytes(_ONE_BY_ONE_PNG)

    config = BMBronchoLCSplitConfig(
        root=dataset_root,
        train_fraction=0.50,
        validation_fraction=0.25,
        test_fraction=0.25,
        salt="unit-test-split",
    )
    first_report = build_bm_broncholc_splits(config)
    second_report = build_bm_broncholc_splits(config)

    assert first_report == second_report
    splits = first_report["splits"]
    assert isinstance(splits, dict)
    leakage_checks = first_report["leakage_checks"]
    assert isinstance(leakage_checks, dict)
    assert leakage_checks["patient_overlap"] is False
    assert leakage_checks["assigned_patient_count"] == 6
    assert splits["train"]["patient_count"] == 2
    assert splits["validation"]["patient_count"] == 2
    assert splits["test"]["patient_count"] == 2
