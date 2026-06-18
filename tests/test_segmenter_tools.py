from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
from segmenters.evaluate_airway_mask import evaluate_airway_masks


def test_evaluate_airway_masks_reports_overlap_metrics_and_png(tmp_path) -> None:
    affine = np.eye(4)
    ct = np.zeros((4, 4, 4), dtype=np.float32)
    reference = np.zeros((4, 4, 4), dtype=np.uint8)
    reference[1, 1, 1] = 1
    reference[1, 1, 2] = 1
    candidate = np.zeros((4, 4, 4), dtype=np.uint8)
    candidate[1, 1, 1] = 1
    candidate[2, 2, 2] = 1
    reference_path = tmp_path / "reference.nii.gz"
    candidate_path = tmp_path / "candidate.nii.gz"
    ct_path = tmp_path / "ct.nii.gz"
    json_path = tmp_path / "report.json"
    png_path = tmp_path / "review.png"
    nib.save(nib.Nifti1Image(ct, affine), str(ct_path))
    nib.save(nib.Nifti1Image(reference, affine), str(reference_path))
    nib.save(nib.Nifti1Image(candidate, affine), str(candidate_path))

    report = evaluate_airway_masks(
        reference_path,
        {"candidate": candidate_path},
        ct_path=ct_path,
        output_json=json_path,
        output_png=png_path,
        min_dice=0.7,
    )

    metrics = report["candidates"]["candidate"]
    assert metrics["intersection_voxels"] == 1
    assert metrics["false_positive_voxels"] == 1
    assert metrics["false_negative_voxels"] == 1
    assert metrics["dice"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["passes_min_dice"] is False
    assert json_path.is_file()
    assert png_path.read_bytes().startswith(b"\x89PNG")


def test_evaluate_airway_masks_requires_explicit_affine_mismatch_override(tmp_path) -> None:
    reference = np.ones((3, 3, 3), dtype=np.uint8)
    candidate = np.ones((3, 3, 3), dtype=np.uint8)
    reference_path = tmp_path / "reference.nii.gz"
    candidate_path = tmp_path / "candidate.nii.gz"
    nib.save(nib.Nifti1Image(reference, np.eye(4)), str(reference_path))
    nib.save(nib.Nifti1Image(candidate, np.diag([2.0, 1.0, 1.0, 1.0])), str(candidate_path))

    with pytest.raises(ValueError, match="affine does not match"):
        evaluate_airway_masks(reference_path, {"candidate": candidate_path})

    report = evaluate_airway_masks(
        reference_path,
        {"candidate": candidate_path},
        allow_affine_mismatch=True,
    )

    assert report["candidates"]["candidate"]["affine_matches_reference"] is False
    assert report["candidates"]["candidate"]["dice"] == 1.0
