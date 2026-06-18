"""Volume ingestion and generation package for SynAirG."""

from volume_gen.artifacts import VolumeCase
from volume_gen.importers import import_dicom_folder, import_local_nifti
from volume_gen.maisi import (
    CommandMaisiBackend,
    CTGenerator,
    MaisiBackend,
    MaisiBatchSpec,
    MaisiCaseSpec,
    NvGenerateCtBackend,
    generate_maisi_case,
)
from volume_gen.tcia import NbiaClient, list_tcia_series, pull_tcia_case

STAGE_NAME = "volume_gen"

__all__ = [
    "CTGenerator",
    "CommandMaisiBackend",
    "MaisiBackend",
    "MaisiBatchSpec",
    "MaisiCaseSpec",
    "NbiaClient",
    "NvGenerateCtBackend",
    "STAGE_NAME",
    "VolumeCase",
    "generate_maisi_case",
    "import_dicom_folder",
    "import_local_nifti",
    "list_tcia_series",
    "pull_tcia_case",
]
