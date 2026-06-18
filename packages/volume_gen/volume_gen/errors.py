"""Exceptions raised by volume_gen workflows."""

from __future__ import annotations


class VolumeGenError(RuntimeError):
    """Base class for volume_gen runtime errors."""


class VolumeImportError(VolumeGenError):
    """Raised when an input volume cannot be imported or converted."""


class BackendUnavailableError(VolumeGenError):
    """Raised when an optional generation backend is not installed or configured."""
