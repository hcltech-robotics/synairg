"""Schemas for airway masks, meshes, centrelines, and branch metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias, cast

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from synairg_core.volume_manifest import JsonValue

from mesh_gen.errors import MeshGenError

BoolArray: TypeAlias = NDArray[np.bool_]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class AirwayMask:
    """A binary airway mask with spatial metadata."""

    case_id: str
    data: BoolArray
    affine: FloatArray
    spacing: tuple[float, float, float]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise MeshGenError("airway mask must be a 3D array")
        if self.affine.shape != (4, 4):
            raise MeshGenError("airway mask affine must be 4x4")
        if any(value <= 0 for value in self.spacing):
            raise MeshGenError("airway mask spacing values must be positive")
        if int(np.count_nonzero(self.data)) == 0:
            raise MeshGenError("airway mask contains no foreground voxels")

    def with_data(self, data: BoolArray) -> AirwayMask:
        """Return this mask metadata with replacement voxel data."""
        return AirwayMask(
            case_id=self.case_id,
            data=np.asarray(data, dtype=np.bool_),
            affine=self.affine,
            spacing=self.spacing,
            source_path=self.source_path,
        )

    def save_nifti(self, path: Path) -> None:
        """Write the binary airway mask as a compressed NIfTI file."""
        image_factory: Any = nib.Nifti1Image
        image: Any = image_factory(self.data.astype(np.uint8), self.affine)
        header: Any = image.header
        header.set_zooms(self.spacing)
        save: Any = nib.save
        save(image, str(path))


@dataclass(frozen=True)
class AirwayMesh:
    """Triangle mesh geometry in physical/world coordinates."""

    vertices: FloatArray
    faces: IntArray
    vertex_normals: FloatArray
    method: str
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise MeshGenError("mesh vertices must have shape (n, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise MeshGenError("mesh faces must have shape (m, 3)")
        if self.vertex_normals.shape != self.vertices.shape:
            raise MeshGenError("mesh vertex normals must match vertices")
        if self.vertices.shape[0] == 0 or self.faces.shape[0] == 0:
            raise MeshGenError("mesh must contain at least one vertex and one face")
        if not np.isfinite(self.vertices).all():
            raise MeshGenError("mesh vertices contain non-finite values")
        if int(self.faces.min()) < 0 or int(self.faces.max()) >= self.vertices.shape[0]:
            raise MeshGenError("mesh faces reference vertices outside the mesh")

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return min/max mesh bounds."""
        minimum = self.vertices.min(axis=0)
        maximum = self.vertices.max(axis=0)
        return (
            (float(minimum[0]), float(minimum[1]), float(minimum[2])),
            (float(maximum[0]), float(maximum[1]), float(maximum[2])),
        )


@dataclass(frozen=True)
class CenterlineNode:
    """A centreline sample point."""

    node_id: str
    point_mm: tuple[float, float, float]
    radius_mm: float
    branch_id: str
    voxel_index: tuple[float, float, float]
    generation: int = 0
    distance_from_root_mm: float = 0.0
    accessible: bool = True

    def __post_init__(self) -> None:
        if not self.node_id:
            raise MeshGenError("centerline node_id must be non-empty")
        if self.radius_mm <= 0:
            raise MeshGenError("centerline node radius_mm must be positive")
        if self.generation < 0:
            raise MeshGenError("centerline node generation must be non-negative")
        if self.distance_from_root_mm < 0:
            raise MeshGenError("centerline node distance_from_root_mm must be non-negative")
        if any(not np.isfinite(value) for value in (*self.point_mm, *self.voxel_index)):
            raise MeshGenError("centerline node coordinates must be finite")

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the node as JSON."""
        return {
            "id": self.node_id,
            "point_mm": [float(value) for value in self.point_mm],
            "radius_mm": float(self.radius_mm),
            "branch_id": self.branch_id,
            "voxel_index": [float(value) for value in self.voxel_index],
            "generation": int(self.generation),
            "distance_from_root_mm": float(self.distance_from_root_mm),
            "accessible": bool(self.accessible),
        }

    @classmethod
    def from_mapping(cls, value: object) -> CenterlineNode:
        """Deserialize and validate a centreline node."""
        mapping = _require_mapping(value, "node")
        return cls(
            node_id=_require_string(mapping.get("id"), "node.id"),
            point_mm=_require_float_triple(mapping.get("point_mm"), "node.point_mm"),
            radius_mm=_require_positive_float(mapping.get("radius_mm"), "node.radius_mm"),
            branch_id=_require_string(mapping.get("branch_id"), "node.branch_id"),
            voxel_index=_require_float_triple(mapping.get("voxel_index"), "node.voxel_index"),
            generation=_require_nonnegative_int(mapping.get("generation", 0), "node.generation"),
            distance_from_root_mm=_require_nonnegative_float(
                mapping.get("distance_from_root_mm", 0.0), "node.distance_from_root_mm"
            ),
            accessible=_require_bool(mapping.get("accessible", True), "node.accessible"),
        )


@dataclass(frozen=True)
class CenterlineEdge:
    """A centreline connection between two sample points."""

    source: str
    target: str
    length_mm: float
    branch_id: str
    generation: int = 0
    accessible: bool = True

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise MeshGenError("centerline edge endpoints must be non-empty")
        if self.source == self.target:
            raise MeshGenError("centerline edge source and target must differ")
        if self.length_mm < 0:
            raise MeshGenError("centerline edge length_mm must be non-negative")
        if self.generation < 0:
            raise MeshGenError("centerline edge generation must be non-negative")

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the edge as JSON."""
        return {
            "source": self.source,
            "target": self.target,
            "length_mm": float(self.length_mm),
            "branch_id": self.branch_id,
            "generation": int(self.generation),
            "accessible": bool(self.accessible),
        }

    @classmethod
    def from_mapping(cls, value: object) -> CenterlineEdge:
        """Deserialize and validate a centreline edge."""
        mapping = _require_mapping(value, "edge")
        return cls(
            source=_require_string(mapping.get("source"), "edge.source"),
            target=_require_string(mapping.get("target"), "edge.target"),
            length_mm=_require_nonnegative_float(mapping.get("length_mm"), "edge.length_mm"),
            branch_id=_require_string(mapping.get("branch_id"), "edge.branch_id"),
            generation=_require_nonnegative_int(mapping.get("generation", 0), "edge.generation"),
            accessible=_require_bool(mapping.get("accessible", True), "edge.accessible"),
        )


@dataclass(frozen=True)
class CenterlineGraph:
    """Validated centreline graph schema."""

    case_id: str
    nodes: tuple[CenterlineNode, ...]
    edges: tuple[CenterlineEdge, ...]
    root_id: str
    method: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.nodes:
            raise MeshGenError("centerline graph must contain at least one node")
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise MeshGenError("centerline graph node IDs must be unique")
        node_ids = set(ids)
        if self.root_id not in node_ids:
            raise MeshGenError("centerline graph root_id must refer to a node")
        for edge in self.edges:
            if edge.source not in node_ids:
                raise MeshGenError(f"centerline edge source {edge.source!r} does not exist")
            if edge.target not in node_ids:
                raise MeshGenError(f"centerline edge target {edge.target!r} does not exist")

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the graph as JSON."""
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "method": self.method,
            "root_id": self.root_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_mapping(cls, value: object) -> CenterlineGraph:
        """Deserialize and validate a centreline graph."""
        mapping = _require_mapping(value, "centerline_graph")
        nodes = tuple(CenterlineNode.from_mapping(item) for item in _require_list(mapping.get("nodes"), "nodes"))
        edges = tuple(CenterlineEdge.from_mapping(item) for item in _require_list(mapping.get("edges"), "edges"))
        return cls(
            case_id=_require_string(mapping.get("case_id"), "case_id"),
            nodes=nodes,
            edges=edges,
            root_id=_require_string(mapping.get("root_id"), "root_id"),
            method=_require_string(mapping.get("method"), "method"),
            schema_version=_require_string(mapping.get("schema_version", "1.0"), "schema_version"),
        )

    @classmethod
    def read_json(cls, path: Path) -> CenterlineGraph:
        """Read a centreline graph JSON file."""
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class BranchSemantic:
    """Semantic metadata for a graph branch."""

    branch_id: str
    label: str
    generation: int
    parent_branch_id: str | None
    node_ids: tuple[str, ...]
    length_mm: float
    mean_radius_mm: float
    min_radius_mm: float
    max_radius_mm: float
    accessible: bool = True

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize branch semantics as JSON."""
        return {
            "branch_id": self.branch_id,
            "label": self.label,
            "generation": self.generation,
            "parent_branch_id": self.parent_branch_id,
            "node_ids": list(self.node_ids),
            "length_mm": float(self.length_mm),
            "mean_radius_mm": float(self.mean_radius_mm),
            "min_radius_mm": float(self.min_radius_mm),
            "max_radius_mm": float(self.max_radius_mm),
            "accessible": bool(self.accessible),
        }


def branch_semantics_payload(case_id: str, branches: tuple[BranchSemantic, ...]) -> dict[str, JsonValue]:
    """Build a branch semantics sidecar payload."""
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "branches": [branch.to_dict() for branch in branches],
    }


def voxel_to_world(
    affine: FloatArray, voxel_index: tuple[float, float, float] | FloatArray
) -> tuple[float, float, float]:
    """Transform a voxel index coordinate into world millimetres."""
    coordinate = np.asarray(voxel_index, dtype=np.float64)
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = coordinate
    world = affine @ homogeneous
    return (float(world[0]), float(world[1]), float(world[2]))


def as_float_tuple(values: NDArray[np.floating]) -> tuple[float, ...]:
    """Cast a numpy vector to a plain float tuple."""
    return tuple(float(value) for value in values)


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MeshGenError(f"{field_name} must be an object")
    return cast(dict[str, object], value)


def _require_list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise MeshGenError(f"{field_name} must be a list")
    return cast(list[object], value)


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MeshGenError(f"{field_name} must be a non-empty string")
    return value


def _require_float_triple(value: object, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise MeshGenError(f"{field_name} must be a three-value coordinate")
    values = tuple(float(item) for item in value)
    if any(not np.isfinite(item) for item in values):
        raise MeshGenError(f"{field_name} must contain finite values")
    return (values[0], values[1], values[2])


def _require_positive_float(value: object, field_name: str) -> float:
    number = _require_number(value, field_name)
    if not np.isfinite(number) or number <= 0:
        raise MeshGenError(f"{field_name} must be positive")
    return number


def _require_nonnegative_float(value: object, field_name: str) -> float:
    number = _require_number(value, field_name)
    if not np.isfinite(number) or number < 0:
        raise MeshGenError(f"{field_name} must be non-negative")
    return number


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MeshGenError(f"{field_name} must be an integer")
    if value < 0:
        raise MeshGenError(f"{field_name} must be non-negative")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise MeshGenError(f"{field_name} must be a boolean")
    return value


def _require_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MeshGenError(f"{field_name} must be a number")
    return float(value)
