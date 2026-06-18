"""Shared utilities and CLI foundation for SynAirG."""

from importlib.metadata import PackageNotFoundError, version

from synairg_core.volume_manifest import ManifestValidationError, VolumeManifest, VolumeSource

try:
    __version__ = version("synairg")
except PackageNotFoundError:  # pragma: no cover - only during uninstalled source imports
    __version__ = "0.0.0+local"

__all__ = ["ManifestValidationError", "VolumeManifest", "VolumeSource", "__version__"]
