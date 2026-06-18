"""Topology repair and connected-component analysis for airway masks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from mesh_gen.errors import MeshGenError
from mesh_gen.schemas import BoolArray, IntArray


@dataclass(frozen=True)
class TopologyReport:
    """Summary of topology cleanup applied to a mask."""

    original_voxels: int
    repaired_voxels: int
    component_count: int
    kept_component_label: int
    removed_component_count: int
    kept_component_count: int = 1
    bridged_component_count: int = 0
    bridge_voxels: int = 0
    min_component_voxels: int = 20
    max_bridge_distance_mm: float = 45.0
    final_component_count: int = 1
    unbridged_component_count: int = 0


@dataclass(frozen=True)
class LaryngealCapReport:
    """Summary of superior disconnected laryngeal-remnant trimming."""

    checked: bool
    original_voxels: int
    retained_voxels: int
    component_count: int
    removed_component_count: int = 0
    removed_voxels: int = 0
    kept_component_label: int | None = None
    superior_axis: int = 2
    cap_index: int | None = None
    gap_mm: float | None = None

    def to_dict(self) -> dict[str, bool | int | float | None]:
        """Serialize the report for quality metadata."""
        return {
            "checked": self.checked,
            "original_voxels": self.original_voxels,
            "retained_voxels": self.retained_voxels,
            "component_count": self.component_count,
            "removed_component_count": self.removed_component_count,
            "removed_voxels": self.removed_voxels,
            "kept_component_label": self.kept_component_label,
            "superior_axis": self.superior_axis,
            "cap_index": self.cap_index,
            "gap_mm": self.gap_mm,
        }


def cap_laryngeal_components(
    mask: BoolArray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    superior_axis: int = 2,
) -> tuple[BoolArray, LaryngealCapReport]:
    """Remove disconnected components wholly superior to the main airway tree.

    This keeps broken laryngeal remnants from being bridged back into the
    tracheobronchial tree as artificial airway geometry.
    """
    if superior_axis not in (0, 1, 2):
        raise MeshGenError("superior_axis must be 0, 1, or 2")
    binary = np.asarray(mask, dtype=np.bool_)
    original_voxels = int(np.count_nonzero(binary))
    if original_voxels == 0:
        raise MeshGenError("cannot cap laryngeal components in an empty airway mask")

    labels, component_count = connected_components(binary)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    kept_label = int(np.argmax(sizes)) if component_count > 0 else None
    if component_count < 2 or kept_label is None:
        return binary, LaryngealCapReport(
            checked=True,
            original_voxels=original_voxels,
            retained_voxels=original_voxels,
            component_count=component_count,
            kept_component_label=kept_label,
            superior_axis=superior_axis,
        )

    kept_coordinates = np.argwhere(labels == kept_label)
    kept_max = int(kept_coordinates[:, superior_axis].max())
    removed_labels: list[int] = []
    removed_minima: list[int] = []
    for component_label in range(1, component_count + 1):
        if component_label == kept_label:
            continue
        coordinates = np.argwhere(labels == component_label)
        if coordinates.size == 0:
            continue
        component_min = int(coordinates[:, superior_axis].min())
        if component_min > kept_max:
            removed_labels.append(component_label)
            removed_minima.append(component_min)

    if not removed_labels:
        return binary, LaryngealCapReport(
            checked=True,
            original_voxels=original_voxels,
            retained_voxels=original_voxels,
            component_count=component_count,
            kept_component_label=kept_label,
            superior_axis=superior_axis,
        )

    removed = np.isin(labels, np.asarray(removed_labels, dtype=np.int64))
    capped = np.asarray(binary & ~removed, dtype=np.bool_)
    retained_voxels = int(np.count_nonzero(capped))
    gap_voxels = max(min(removed_minima) - kept_max - 1, 0)
    return capped, LaryngealCapReport(
        checked=True,
        original_voxels=original_voxels,
        retained_voxels=retained_voxels,
        component_count=component_count,
        removed_component_count=len(removed_labels),
        removed_voxels=original_voxels - retained_voxels,
        kept_component_label=kept_label,
        superior_axis=superior_axis,
        cap_index=kept_max,
        gap_mm=float(gap_voxels * spacing[superior_axis]),
    )


def repair_airway_mask(
    mask: BoolArray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_component_voxels: int = 20,
    max_bridge_distance_mm: float = 45.0,
) -> tuple[BoolArray, TopologyReport]:
    """Prune speckles and bridge plausible distal airway fragments."""
    binary = np.asarray(mask, dtype=np.bool_)
    original_voxels = int(np.count_nonzero(binary))
    if original_voxels == 0:
        raise MeshGenError("cannot repair an empty airway mask")
    if min_component_voxels < 1:
        raise MeshGenError("min_component_voxels must be positive")
    if max_bridge_distance_mm < 0:
        raise MeshGenError("max_bridge_distance_mm must be non-negative")

    labels, component_count = connected_components(binary)
    if component_count == 0:
        raise MeshGenError("connected-component analysis found no foreground components")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    kept_label = int(np.argmax(sizes))
    repaired = np.asarray(labels == kept_label, dtype=np.bool_)
    retained_labels = {kept_label}
    bridged_component_count = 0
    bridge_voxels = 0
    for component_label in np.argsort(sizes)[::-1]:
        label_int = int(component_label)
        if label_int == 0 or label_int == kept_label:
            continue
        if int(sizes[label_int]) < min_component_voxels:
            continue
        component = labels == label_int
        bridged = _bridge_component(
            repaired,
            component,
            spacing=spacing,
            max_distance_mm=max_bridge_distance_mm,
        )
        if bridged is None:
            continue
        bridge, distance_mm = bridged
        if distance_mm > max_bridge_distance_mm:
            continue
        new_bridge_voxels = int(np.count_nonzero(bridge & ~repaired & ~component))
        repaired = np.asarray(repaired | component | bridge, dtype=np.bool_)
        retained_labels.add(label_int)
        bridged_component_count += 1
        bridge_voxels += new_bridge_voxels
    repaired_voxels = int(np.count_nonzero(repaired))
    _, final_component_count = connected_components(repaired)
    unbridged_component_count = sum(
        1
        for component_label in range(1, component_count + 1)
        if component_label not in retained_labels and int(sizes[component_label]) >= min_component_voxels
    )
    report = TopologyReport(
        original_voxels=original_voxels,
        repaired_voxels=repaired_voxels,
        component_count=component_count,
        kept_component_label=kept_label,
        removed_component_count=max(component_count - len(retained_labels), 0),
        kept_component_count=len(retained_labels),
        bridged_component_count=bridged_component_count,
        bridge_voxels=bridge_voxels,
        min_component_voxels=min_component_voxels,
        max_bridge_distance_mm=max_bridge_distance_mm,
        final_component_count=final_component_count,
        unbridged_component_count=unbridged_component_count,
    )
    return repaired, report


def connected_components(mask: BoolArray) -> tuple[IntArray, int]:
    """Label 26-connected foreground components."""
    skimage_label = _optional_callable("skimage.measure", "label")
    if skimage_label is not None:
        labels_any: Any = skimage_label(mask, connectivity=3)
        labels = np.asarray(labels_any, dtype=np.int64)
        return labels, int(labels.max())
    return _connected_components_numpy(mask)


def _connected_components_numpy(mask: BoolArray) -> tuple[IntArray, int]:
    labels = np.zeros(mask.shape, dtype=np.int64)
    component = 0
    for start in np.argwhere(mask):
        index = (int(start[0]), int(start[1]), int(start[2]))
        if labels[index] != 0:
            continue
        component += 1
        labels[index] = component
        queue: deque[tuple[int, int, int]] = deque([index])
        while queue:
            current = queue.popleft()
            for neighbor in _neighbors(current, mask.shape):
                if mask[neighbor] and labels[neighbor] == 0:
                    labels[neighbor] = component
                    queue.append(neighbor)
    return labels, component


def _neighbors(index: tuple[int, int, int], shape: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    i, j, k = index
    candidates = tuple(
        (i + di, j + dj, k + dk)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        for dk in (-1, 0, 1)
        if not (di == 0 and dj == 0 and dk == 0)
    )
    return tuple(
        candidate
        for candidate in candidates
        if 0 <= candidate[0] < shape[0] and 0 <= candidate[1] < shape[1] and 0 <= candidate[2] < shape[2]
    )


def _bridge_component(
    repaired: BoolArray,
    component: BoolArray,
    *,
    spacing: tuple[float, float, float],
    max_distance_mm: float,
) -> tuple[BoolArray, float] | None:
    distance_transform = _optional_callable("scipy.ndimage", "distance_transform_edt")
    if distance_transform is None:
        return None
    distances_any: Any
    indices_any: Any
    distances_any, indices_any = distance_transform(
        ~repaired,
        sampling=spacing,
        return_indices=True,
    )
    distances = np.asarray(distances_any, dtype=np.float64)
    indices = np.asarray(indices_any, dtype=np.int64)
    component_coordinates = np.argwhere(component)
    if component_coordinates.size == 0:
        return None
    component_distances = distances[component]
    nearest_offset = int(np.argmin(component_distances))
    distance_mm = float(component_distances[nearest_offset])
    if distance_mm > max_distance_mm:
        return None
    start = component_coordinates[nearest_offset]
    end = indices[(slice(None), int(start[0]), int(start[1]), int(start[2]))]
    bridge = np.zeros(repaired.shape, dtype=np.bool_)
    for point in _line_points(start, end):
        bridge[point] = True
    return bridge, distance_mm


def _line_points(start: np.ndarray, end: np.ndarray) -> list[tuple[int, int, int]]:
    delta = end.astype(np.float64) - start.astype(np.float64)
    steps = max(int(np.max(np.abs(delta))), 1)
    points: list[tuple[int, int, int]] = []
    for value in np.linspace(0.0, 1.0, steps + 1):
        point = np.rint(start.astype(np.float64) + delta * value).astype(np.int64)
        points.append((int(point[0]), int(point[1]), int(point[2])))
    return points


def _optional_callable(module_name: str, attribute_name: str) -> Any | None:
    try:
        module = import_module(module_name)
    except ImportError:
        return None
    value: Any = getattr(module, attribute_name, None)
    return value if callable(value) else None
