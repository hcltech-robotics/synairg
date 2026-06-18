from __future__ import annotations

import json

import pytest
from synairg_core.volume_manifest import ManifestValidationError, VolumeManifest, read_volume_manifest


def test_volume_manifest_accepts_local_nifti_source(tmp_path) -> None:
    manifest_path = tmp_path / "volume_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "case_id": "case-001",
                "source": {
                    "type": "local_nifti",
                    "uri": "input.nii.gz",
                    "metadata": {"scanner": "fixture"},
                },
                "metadata": {"split": "train"},
            }
        ),
        encoding="utf-8",
    )

    manifest = read_volume_manifest(manifest_path)

    assert manifest == VolumeManifest.from_mapping(manifest.to_dict())
    assert manifest.case_id == "case-001"
    assert manifest.source.source_type == "local_nifti"
    assert manifest.source.uri == "input.nii.gz"
    assert manifest.metadata["split"] == "train"


def test_volume_manifest_rejects_unsafe_case_ids() -> None:
    with pytest.raises(ManifestValidationError, match="case_id"):
        VolumeManifest.from_mapping(
            {
                "case_id": "../outside",
                "source": {"type": "local_nifti", "uri": "input.nii.gz"},
            }
        )


def test_volume_manifest_rejects_unknown_source_types() -> None:
    with pytest.raises(ManifestValidationError, match="source.type"):
        VolumeManifest.from_mapping(
            {
                "case_id": "case-001",
                "source": {"type": "spreadsheet", "uri": "input.csv"},
            }
        )
