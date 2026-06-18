from __future__ import annotations

import json

import nibabel as nib
import numpy as np
from generation.enforce_thoracic_airway_hu import AirwayHuCorrection, enforce_thoracic_airway_hu, main
from generation.prepare_thoracic_airway_mask import ThoracicMaskPrep, prepare_thoracic_airway_mask
from generation.run_rflow_ct_from_env import _script_name_for_request, _strip_version_args


def test_enforce_thoracic_airway_hu_uses_source_mask_labels(tmp_path) -> None:
    ct = np.full((5, 5, 5), -1000.0, dtype=np.float32)
    ct[1:4, 1:4, 1:4] = -120.0
    labels = np.zeros((5, 5, 5), dtype=np.uint16)
    labels[1:4, 1:4, 1:4] = 200
    labels[2, 2, 1:3] = 57
    labels[2, 2, 3] = 132

    affine = np.eye(4)
    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    output_path = tmp_path / "corrected.nii.gz"
    nib.save(nib.Nifti1Image(ct, affine), str(ct_path))
    nib.save(nib.Nifti1Image(labels, affine), str(mask_path))

    report = enforce_thoracic_airway_hu(
        ct_path,
        mask_path,
        output_path,
        correction=AirwayHuCorrection(
            airway_labels=(57, 132),
            airway_hu=-980.0,
            halo_hu=-700.0,
            outside_body_hu=25.0,
            halo_iterations=1,
        ),
    )

    corrected = np.asarray(nib.load(str(output_path)).dataobj)
    assert report["airway_voxels"] == 3
    assert report["airway_labels"] == [57, 132]
    assert report["outside_body_voxels"] == int(np.count_nonzero(labels == 0))
    assert np.all(corrected[np.isin(labels, [57, 132])] == -980.0)
    assert corrected[0, 0, 0] == 25.0
    assert corrected[2, 1, 2] == -700.0
    assert corrected[1, 1, 1] == -120.0


def test_enforce_thoracic_airway_hu_cli_accepts_config(tmp_path, monkeypatch) -> None:
    ct = np.zeros((3, 3, 3), dtype=np.float32)
    labels = np.ones((3, 3, 3), dtype=np.uint16) * 200
    labels[1, 1, 1] = 57
    labels[1, 1, 2] = 132
    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    output_path = tmp_path / "corrected.nii.gz"
    config_path = tmp_path / "correction.json"
    nib.save(nib.Nifti1Image(ct, np.eye(4)), str(ct_path))
    nib.save(nib.Nifti1Image(labels, np.eye(4)), str(mask_path))
    config_path.write_text(
        json.dumps(
            {
                "ct_path": str(ct_path),
                "source_mask_path": str(mask_path),
                "output_path": str(output_path),
                "airway_labels": [57, 132],
                "airway_hu": -990.0,
                "halo_iterations": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["enforce_thoracic_airway_hu.py", "--config", str(config_path)])

    assert main() == 0

    corrected = np.asarray(nib.load(str(output_path)).dataobj)
    assert corrected[1, 1, 1] == -990.0
    assert corrected[1, 1, 2] == -990.0


def test_prepare_thoracic_airway_mask_removes_abdominal_context(tmp_path) -> None:
    labels = np.zeros((6, 6, 8), dtype=np.uint16)
    labels[:, :, :] = 200
    labels[:, :, 0:3] = 1
    labels[2:4, 2:4, 3:6] = 28
    labels[3, 3, 4] = 132
    labels[1:3, 1:3, 4] = 115
    labels[0, 0, 5] = 63
    source_path = tmp_path / "source.nii.gz"
    output_path = tmp_path / "thoracic.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.eye(4)), str(source_path))

    report = prepare_thoracic_airway_mask(
        source_path,
        output_path,
        prep=ThoracicMaskPrep(inferior_margin_slices=0, superior_margin_slices=1),
    )

    prepared = np.asarray(nib.load(str(output_path)).dataobj)
    assert report["thoracic_crop_min_index"] == 3
    assert report["thoracic_crop_max_index"] == 6
    assert np.all(prepared[:, :, :3] == 0)
    assert 1 not in np.unique(prepared)
    assert prepared[3, 3, 4] == 132
    assert prepared[2, 2, 3] == 28
    assert prepared[1, 1, 4] == 115
    assert prepared[0, 0, 5] == 63
    assert 200 in np.unique(prepared)


def test_rflow_launcher_dispatches_mask_requests_and_strips_version(tmp_path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({"mask_path": "mask.nii.gz"}), encoding="utf-8")

    assert _script_name_for_request([str(request_path)]) == "run_ct_from_mask.py"
    assert _script_name_for_request(["default"]) == "run_rflow_ct.py"
    assert _strip_version_args([str(request_path), "--version", "rflow-ct", "--yes"]) == [
        str(request_path),
        "--yes",
    ]
