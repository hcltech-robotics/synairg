"""Error types for mesh generation workflows."""

from __future__ import annotations


class MeshGenError(RuntimeError):
    """Raised when a mesh generation step cannot complete."""
