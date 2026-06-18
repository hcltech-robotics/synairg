"""Training, evaluation, and distillation package for SynAirG."""

from trainer.bronchogen_manifest import (
    BronchoGenManifestConfig,
    BronchoGenManifestError,
    build_bronchogen_manifest,
    validate_bronchogen_manifest,
    write_bronchogen_manifest,
)
from trainer.public_bronchoscopy_data import (
    BMBronchoLCAuditConfig,
    BMBronchoLCDownloadConfig,
    BMBronchoLCSplitConfig,
    PublicBronchoscopyDataError,
    audit_bm_broncholc,
    build_bm_broncholc_splits,
    download_bm_broncholc,
    write_bm_broncholc_audit,
    write_bm_broncholc_download_report,
    write_bm_broncholc_splits,
)
from trainer.temporal_metrics import (
    TemporalMetricConfig,
    TemporalMetricError,
    compute_blind_temporal_metrics,
    write_blind_temporal_metrics,
)

STAGE_NAME = "trainer"

__all__ = [
    "BronchoGenManifestConfig",
    "BronchoGenManifestError",
    "BMBronchoLCAuditConfig",
    "BMBronchoLCDownloadConfig",
    "BMBronchoLCSplitConfig",
    "PublicBronchoscopyDataError",
    "STAGE_NAME",
    "TemporalMetricConfig",
    "TemporalMetricError",
    "audit_bm_broncholc",
    "build_bronchogen_manifest",
    "build_bm_broncholc_splits",
    "compute_blind_temporal_metrics",
    "download_bm_broncholc",
    "validate_bronchogen_manifest",
    "write_bm_broncholc_audit",
    "write_bm_broncholc_download_report",
    "write_bm_broncholc_splits",
    "write_blind_temporal_metrics",
    "write_bronchogen_manifest",
]
