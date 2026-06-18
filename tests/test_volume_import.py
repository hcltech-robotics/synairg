from __future__ import annotations

import gzip
import json
import sys
import zipfile
from datetime import datetime, timezone
from os import environ
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
from synairg_core.cli import app
from typer.testing import CliRunner
from volume_gen.tcia import list_tcia_series, pull_tcia_case

runner = CliRunner()


def test_import_local_cli_emits_canonical_volume_sidecars(tmp_path) -> None:
    source_path = tmp_path / "source.nii.gz"
    with gzip.open(source_path, "wb") as handle:
        handle.write(b"fixture-nifti")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "case_id": "local-case",
                "source": {"type": "local_nifti", "uri": source_path.name, "metadata": {"origin": "unit-test"}},
                "metadata": {"cohort": "fixture"},
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "volumes"

    result = runner.invoke(
        app,
        ["volume", "import-local", "--manifest", str(manifest_path), "--output", str(output_root)],
    )

    assert result.exit_code == 0, result.output
    case_dir = output_root / "local-case"
    assert (case_dir / "ct.nii.gz").read_bytes() == source_path.read_bytes()
    metadata = json.loads((case_dir / "volume_metadata.json").read_text(encoding="utf-8"))
    provenance = json.loads((case_dir / "source_provenance.json").read_text(encoding="utf-8"))
    assert metadata["case_id"] == "local-case"
    assert metadata["source_type"] == "local_nifti"
    assert metadata["artifacts"]["ct"]["sha256"]
    assert provenance["source"]["metadata"]["origin"] == "unit-test"
    assert provenance["inputs"][0]["sha256"]


def test_import_dicom_cli_converts_folder_to_nifti(tmp_path) -> None:
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    _write_dicom_slice(dicom_dir / "slice-1.dcm", instance_number=1, value=100)
    _write_dicom_slice(dicom_dir / "slice-2.dcm", instance_number=2, value=200)
    output_root = tmp_path / "volumes"

    result = runner.invoke(
        app,
        [
            "volume",
            "import-dicom",
            "--input",
            str(dicom_dir),
            "--case-id",
            "dicom-case",
            "--output",
            str(output_root),
        ],
    )

    assert result.exit_code == 0, result.output
    case_dir = output_root / "dicom-case"
    image = nib.load(str(case_dir / "ct.nii.gz"))
    assert image.shape == (2, 2, 2)
    metadata = json.loads((case_dir / "volume_metadata.json").read_text(encoding="utf-8"))
    provenance = json.loads((case_dir / "source_provenance.json").read_text(encoding="utf-8"))
    assert metadata["volume"]["dicom_file_count"] == 2
    assert metadata["volume"]["shape"] == [2, 2, 2]
    assert provenance["source"]["type"] == "local_dicom"
    assert len(provenance["inputs"]) == 2


def test_import_dicom_cli_preserves_image_plane_geometry(tmp_path) -> None:
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()
    orientation = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]
    _write_dicom_slice(
        dicom_dir / "slice-1.dcm",
        instance_number=1,
        value=100,
        image_orientation=orientation,
        image_position=[10.0, 20.0, 30.0],
        pixel_spacing=[0.8, 1.2],
        slice_thickness=2.5,
    )
    _write_dicom_slice(
        dicom_dir / "slice-2.dcm",
        instance_number=2,
        value=200,
        image_orientation=orientation,
        image_position=[10.0, 20.0, 32.5],
        pixel_spacing=[0.8, 1.2],
        slice_thickness=2.5,
    )
    output_root = tmp_path / "volumes"

    result = runner.invoke(
        app,
        [
            "volume",
            "import-dicom",
            "--input",
            str(dicom_dir),
            "--case-id",
            "oblique-dicom-case",
            "--output",
            str(output_root),
        ],
    )

    assert result.exit_code == 0, result.output
    case_dir = output_root / "oblique-dicom-case"
    image = nib.load(str(case_dir / "ct.nii.gz"))
    expected_affine = np.asarray(
        [
            [-0.8, 0.0, 0.0, 10.0],
            [0.0, 1.2, 0.0, 20.0],
            [0.0, 0.0, 2.5, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    metadata = json.loads((case_dir / "volume_metadata.json").read_text(encoding="utf-8"))
    dicom_reference = metadata["volume"]["dicom_reference"]
    assert np.allclose(image.affine, expected_affine)
    assert np.allclose(dicom_reference["voxel_index_to_dicom_patient_lps_mm"], expected_affine)
    assert dicom_reference["pixel_spacing_mm"] == [0.8, 1.2]
    assert dicom_reference["slice_spacing_mm"] == 2.5
    assert dicom_reference["slice_spacing_derivation"] == "image_position_projection"
    assert dicom_reference["privacy"]["patient_identifiers"] == "omitted"
    assert dicom_reference["study_instance_uid_sha256"].startswith("sha256:")
    assert "PatientName" not in json.dumps(dicom_reference)


def test_pull_tcia_cli_downloads_series_and_emits_canonical_case(tmp_path, monkeypatch) -> None:
    archive_path = _write_tcia_series_archive(tmp_path)
    fake_client = _FakeNbiaClient(archive_path)
    monkeypatch.setattr("volume_gen.tcia.NbiaClient", lambda: fake_client)
    output_root = tmp_path / "volumes"

    result = runner.invoke(
        app,
        [
            "volume",
            "pull-tcia",
            "--collection",
            "LIDC-IDRI",
            "--case-id",
            "tcia-case",
            "--output",
            str(output_root),
        ],
    )

    assert result.exit_code == 0, result.output
    case_dir = output_root / "tcia-case"
    image = nib.load(str(case_dir / "ct.nii.gz"))
    metadata = json.loads((case_dir / "volume_metadata.json").read_text(encoding="utf-8"))
    provenance = json.loads((case_dir / "source_provenance.json").read_text(encoding="utf-8"))
    assert image.shape == (2, 2, 2)
    assert metadata["source_type"] == "tcia"
    assert metadata["volume"]["normalization"] == "tcia_dicom_to_nifti"
    assert metadata["volume"]["series_instance_uid"] == "1.2.826.0.1"
    assert provenance["source"]["type"] == "tcia"
    assert provenance["source"]["metadata"]["collection"] == "LIDC-IDRI"
    assert provenance["source"]["metadata"]["series_metadata"]["Modality"] == "CT"
    assert any("tcia_series.zip" in item["path"] and item["sha256"] for item in provenance["inputs"])


def test_pull_tcia_case_can_select_specific_series(tmp_path) -> None:
    archive_path = _write_tcia_series_archive(tmp_path)
    output_root = tmp_path / "volumes"

    volume_case = pull_tcia_case(
        "LIDC-IDRI",
        output_root,
        series_instance_uid="1.2.826.0.2",
        case_id="selected-tcia-case",
        client=_FakeNbiaClient(archive_path),
    )

    metadata = json.loads(volume_case.metadata_path.read_text(encoding="utf-8"))
    provenance = json.loads(volume_case.provenance_path.read_text(encoding="utf-8"))
    assert volume_case.ct_path.exists()
    assert metadata["volume"]["series_instance_uid"] == "1.2.826.0.2"
    assert provenance["source"]["metadata"]["series_metadata"]["SeriesDescription"] == "Selected CT"


def test_list_tcia_series_filters_ct_body_part() -> None:
    series = list_tcia_series(
        "LIDC-IDRI",
        client=_FakeNbiaClient(Path("unused.zip")),
        body_part_examined="CHEST",
    )

    assert [item["SeriesInstanceUID"] for item in series] == ["1.2.826.0.1", "1.2.826.0.2"]


def test_list_tcia_series_cli_emits_json(tmp_path, monkeypatch) -> None:
    fake_client = _FakeNbiaClient(tmp_path / "unused.zip")
    monkeypatch.setattr("volume_gen.tcia.NbiaClient", lambda: fake_client)

    result = runner.invoke(
        app,
        [
            "volume",
            "list-tcia-series",
            "--collection",
            "LIDC-IDRI",
            "--body-part",
            "CHEST",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [item["SeriesInstanceUID"] for item in payload] == ["1.2.826.0.1", "1.2.826.0.2"]


def test_generate_maisi_cli_fails_cleanly_without_backend(tmp_path) -> None:
    config = tmp_path / "maisi.json"
    config.write_text(json.dumps({"case_id": "maisi-case"}), encoding="utf-8")

    result = runner.invoke(app, ["volume", "generate-maisi", "--config", str(config), "--output", str(tmp_path)])

    assert result.exit_code == 1
    assert "MAISI backend is unavailable" in result.output


def test_generate_maisi_cli_validates_nv_generate_backend_config(tmp_path) -> None:
    config = tmp_path / "maisi.json"
    config.write_text(
        json.dumps(
            {
                "case_id": "maisi-case",
                "backend": {"type": "nv-generate-ct", "command": ["missing-nv-generate-wrapper"]},
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["volume", "generate-maisi", "--config", str(config), "--output", str(tmp_path)])

    assert result.exit_code == 1
    assert "requires backend.infer_config_path" in result.output


def test_generate_maisi_cli_passes_absolute_infer_config_to_nv_backend(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    source_mask = config_dir / "source_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), np.eye(4)), str(source_mask))
    infer_config = config_dir / "infer.json"
    infer_config.write_text(json.dumps({"mask_path": "source_mask.nii.gz"}), encoding="utf-8")
    fake_backend = tmp_path / "fake_nv_backend.py"
    fake_backend.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "from pathlib import Path",
                "import nibabel as nib",
                "import numpy as np",
                "infer_config = Path(sys.argv[1])",
                "output_dir = Path(sys.argv[2])",
                "assert infer_config.is_absolute(), infer_config",
                "assert infer_config.is_file(), infer_config",
                "assert output_dir.is_absolute(), output_dir",
                "output_dir.mkdir(parents=True, exist_ok=True)",
                "image_path = output_dir / 'image.nii.gz'",
                "image = nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.float32), np.eye(4))",
                "nib.save(image, str(image_path))",
                "payload = {'output': {'samples': [{'image_path': str(image_path)}]}}",
                "print(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )
    config = config_dir / "maisi.json"
    config.write_text(
        json.dumps(
            {
                "case_id": "absolute-infer-case",
                "backend": {
                    "type": "nv-generate-ct",
                    "infer_config_path": "infer.json",
                    "command": [sys.executable, str(fake_backend), "{infer_config_path}", "{backend_output_dir}"],
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["volume", "generate-maisi", "--config", str(config), "--output", str(tmp_path)])

    assert result.exit_code == 0, result.output
    case_dir = tmp_path / "absolute-infer-case"
    assert (case_dir / "ct.nii.gz").is_file()
    assert (case_dir / "label.nii.gz").is_file()
    label = np.asarray(nib.load(str(case_dir / "label.nii.gz")).get_fdata())
    assert np.count_nonzero(label) == 27
    metadata = json.loads((case_dir / "volume_metadata.json").read_text(encoding="utf-8"))
    command = metadata["volume"]["command"]
    assert Path(command[2]).is_absolute()
    assert Path(command[2]).is_file()
    assert Path(command[3]).is_absolute()


@pytest.mark.integration
def test_generate_maisi_cli_runs_real_nv_generate_backend_when_configured(tmp_path) -> None:
    config_env = environ.get("SYNAIRG_TEST_MAISI_CONFIG")
    if config_env is None:
        pytest.skip("set SYNAIRG_TEST_MAISI_CONFIG to a real nv-generate-ct backend config JSON")
    config = Path(config_env)
    output_root = tmp_path / "volumes"

    result = runner.invoke(app, ["volume", "generate-maisi", "--config", str(config), "--output", str(output_root)])

    assert result.exit_code == 0, result.output
    case_id = json.loads(config.read_text(encoding="utf-8"))["case_id"]
    case_dir = output_root / case_id
    image = nib.load(str(case_dir / "ct.nii.gz"))
    metadata = json.loads((case_dir / "volume_metadata.json").read_text(encoding="utf-8"))
    provenance = json.loads((case_dir / "source_provenance.json").read_text(encoding="utf-8"))
    assert len(image.shape) == 3
    assert metadata["source_type"] == "maisi"
    assert metadata["volume"]["backend"]
    assert metadata["volume"]["returncode"] == 0
    assert metadata["artifacts"]["ct"]["sha256"]
    assert provenance["source"]["type"] == "maisi"
    assert provenance["source"]["metadata"]["backend"] == metadata["volume"]["backend"]
    assert provenance["outputs"][0]["sha256"]


def _write_dicom_slice(
    path,
    *,
    instance_number: int,
    value: int,
    image_orientation=None,
    image_position=None,
    pixel_spacing=None,
    slice_thickness: float = 1.0,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "SynAirG^Fixture"
    dataset.PatientID = "fixture"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "CT"
    dataset.StudyDate = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.InstanceNumber = instance_number
    dataset.FrameOfReferenceUID = generate_uid()
    dataset.ImageOrientationPatient = image_orientation or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.ImagePositionPatient = image_position or [0.0, 0.0, float(instance_number - 1)]
    dataset.PixelSpacing = pixel_spacing or [1.0, 1.0]
    dataset.SliceThickness = slice_thickness
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleIntercept = 0
    dataset.RescaleSlope = 1
    dataset.PixelData = np.full((2, 2), value, dtype=np.uint16).tobytes()
    dataset.save_as(path, write_like_original=False)


def _write_tcia_series_archive(tmp_path) -> Path:
    dicom_dir = tmp_path / "tcia_dicom"
    dicom_dir.mkdir()
    _write_dicom_slice(dicom_dir / "slice-1.dcm", instance_number=1, value=300)
    _write_dicom_slice(dicom_dir / "slice-2.dcm", instance_number=2, value=400)
    archive_path = tmp_path / "tcia_series.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in sorted(dicom_dir.iterdir()):
            archive.write(path, arcname=f"series/{path.name}")
    return archive_path


class _FakeNbiaClient:
    def __init__(self, archive_path: Path) -> None:
        self.archive_path = archive_path

    def series_url(self, collection: str) -> str:
        return f"https://example.test/getSeries?Collection={collection}"

    def image_url(self, series_instance_uid: str) -> str:
        return f"https://example.test/getImage?SeriesInstanceUID={series_instance_uid}"

    def list_series(self, collection: str):
        return [
            {
                "Collection": collection,
                "PatientID": "patient-1",
                "StudyInstanceUID": "1.2.826.0.study",
                "SeriesInstanceUID": "1.2.826.0.1",
                "SeriesDescription": "Auto CT",
                "Modality": "CT",
                "BodyPartExamined": "CHEST",
                "ImageCount": 2,
            },
            {
                "Collection": collection,
                "PatientID": "patient-2",
                "StudyInstanceUID": "1.2.826.0.study",
                "SeriesInstanceUID": "1.2.826.0.2",
                "SeriesDescription": "Selected CT",
                "Modality": "CT",
                "BodyPartExamined": "CHEST",
                "ImageCount": 2,
            },
            {
                "Collection": collection,
                "PatientID": "patient-3",
                "StudyInstanceUID": "1.2.826.0.study",
                "SeriesInstanceUID": "1.2.826.0.3",
                "SeriesDescription": "Abdomen MR",
                "Modality": "MR",
                "BodyPartExamined": "ABDOMEN",
                "ImageCount": 2,
            },
        ]

    def download_series(self, series_instance_uid: str, destination_zip: Path) -> Path:
        destination_zip.parent.mkdir(parents=True, exist_ok=True)
        destination_zip.write_bytes(self.archive_path.read_bytes())
        return destination_zip
