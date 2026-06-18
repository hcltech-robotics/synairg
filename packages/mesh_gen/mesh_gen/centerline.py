"""Centreline extraction and branch graph generation."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from heapq import heappop, heappush
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

from mesh_gen.artifacts import write_json
from mesh_gen.errors import MeshGenError
from mesh_gen.schemas import (
    AirwayMask,
    BoolArray,
    BranchSemantic,
    CenterlineEdge,
    CenterlineGraph,
    CenterlineNode,
    branch_semantics_payload,
    voxel_to_world,
)


def extract_centerline_graph(
    mask: AirwayMask,
    *,
    min_accessible_diameter_mm: float = 3.0,
    max_accessible_depth_mm: float | None = None,
) -> CenterlineGraph:
    """Extract a centreline graph, preferring specialized backends when installed."""
    method = _preferred_centerline_method()
    if method == "skimage-skeleton":
        graph = _extract_skimage_skeleton_graph(mask)
        if graph is not None:
            return apply_bronchoscope_accessibility(
                graph,
                min_accessible_diameter_mm=min_accessible_diameter_mm,
                max_accessible_depth_mm=max_accessible_depth_mm,
            )
    return apply_bronchoscope_accessibility(
        _extract_centroid_slice_graph(mask),
        min_accessible_diameter_mm=min_accessible_diameter_mm,
        max_accessible_depth_mm=max_accessible_depth_mm,
    )


def build_branch_semantics(graph: CenterlineGraph) -> tuple[BranchSemantic, ...]:
    """Create branch-level metadata from a centreline graph."""
    node_by_id = {node.node_id: node for node in graph.nodes}
    lengths_by_branch: dict[str, float] = defaultdict(float)
    parent_by_branch: dict[str, str | None] = {}
    for edge in graph.edges:
        lengths_by_branch[edge.branch_id] += edge.length_mm
        source_branch = node_by_id[edge.source].branch_id
        if source_branch != edge.branch_id and edge.branch_id not in parent_by_branch:
            parent_by_branch[edge.branch_id] = source_branch
    nodes_by_branch: dict[str, list[CenterlineNode]] = defaultdict(list)
    for node in graph.nodes:
        nodes_by_branch[node.branch_id].append(node)

    branches: list[BranchSemantic] = []
    for branch_id in sorted(nodes_by_branch):
        nodes = nodes_by_branch[branch_id]
        length = lengths_by_branch.get(branch_id, 0.0)
        radii = [node.radius_mm for node in nodes]
        mean_radius = float(np.mean(radii))
        min_radius = float(np.min(radii))
        max_radius = float(np.max(radii))
        generation = int(min(node.generation for node in nodes))
        label = "trachea" if branch_id == "airway-main" else branch_id
        branches.append(
            BranchSemantic(
                branch_id=branch_id,
                label=label,
                generation=generation,
                parent_branch_id=parent_by_branch.get(branch_id),
                node_ids=tuple(node.node_id for node in nodes if node.node_id in node_by_id),
                length_mm=length,
                mean_radius_mm=mean_radius,
                min_radius_mm=min_radius,
                max_radius_mm=max_radius,
                accessible=all(node.accessible for node in nodes),
            )
        )
    return tuple(branches)


def write_centerline_artifacts(
    graph_path: Path,
    branch_path: Path,
    radius_profile_path: Path,
    graph: CenterlineGraph,
) -> None:
    """Write centreline graph, branch semantics, and radius profile artifacts."""
    write_json(graph_path, graph.to_dict())
    branches = build_branch_semantics(graph)
    write_json(branch_path, branch_semantics_payload(graph.case_id, branches))
    write_radius_profile(radius_profile_path, graph)


def prune_mask_to_accessible_centerline(
    mask: AirwayMask,
    graph: CenterlineGraph,
) -> tuple[BoolArray, dict[str, int | bool]]:
    """Keep mask voxels nearest to root-reachable centreline nodes."""
    accessible_nodes = [node for node in graph.nodes if node.accessible]
    if len(accessible_nodes) == len(graph.nodes):
        voxel_count = int(np.count_nonzero(mask.data))
        return np.asarray(mask.data, dtype=np.bool_), {
            "checked": True,
            "original_voxels": voxel_count,
            "retained_voxels": voxel_count,
            "removed_voxels": 0,
            "accessible_node_count": len(accessible_nodes),
            "total_node_count": len(graph.nodes),
        }
    if not accessible_nodes:
        raise MeshGenError("bronchoscope diameter filter removed every centerline node")

    coordinates = np.argwhere(mask.data)
    node_coordinates = np.asarray([node.voxel_index for node in graph.nodes], dtype=np.float64)
    accessible_by_index = np.asarray([node.accessible for node in graph.nodes], dtype=np.bool_)
    nearest = _nearest_node_indices(coordinates.astype(np.float64), node_coordinates)
    keep_coordinates = coordinates[accessible_by_index[nearest]]
    filtered = np.zeros(mask.data.shape, dtype=np.bool_)
    filtered[keep_coordinates[:, 0], keep_coordinates[:, 1], keep_coordinates[:, 2]] = True
    retained = int(np.count_nonzero(filtered))
    if retained == 0:
        raise MeshGenError("bronchoscope diameter filter produced an empty airway mask")
    original = int(np.count_nonzero(mask.data))
    return filtered, {
        "checked": True,
        "original_voxels": original,
        "retained_voxels": retained,
        "removed_voxels": original - retained,
        "accessible_node_count": len(accessible_nodes),
        "total_node_count": len(graph.nodes),
    }


def write_radius_profile(path: Path, graph: CenterlineGraph) -> None:
    """Write per-node radius samples as Parquet, falling back to JSON if pyarrow is unavailable."""
    columns: dict[str, Any] = {
        "case_id": [graph.case_id for _ in graph.nodes],
        "node_id": [node.node_id for node in graph.nodes],
        "branch_id": [node.branch_id for node in graph.nodes],
        "distance_mm": [node.distance_from_root_mm for node in graph.nodes],
        "radius_mm": [node.radius_mm for node in graph.nodes],
        "generation": [node.generation for node in graph.nodes],
        "accessible": [node.accessible for node in graph.nodes],
        "x_mm": [node.point_mm[0] for node in graph.nodes],
        "y_mm": [node.point_mm[1] for node in graph.nodes],
        "z_mm": [node.point_mm[2] for node in graph.nodes],
    }
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table: Any = pa.table(columns)
        write_table: Any = pq.write_table
        write_table(table, path)
    except ImportError:
        path.write_text(json.dumps(columns, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _preferred_centerline_method() -> str:
    if find_spec("vmtk") is not None:
        return "vmtk"
    skeletonize = _optional_callable("skimage.morphology", "skeletonize")
    if skeletonize is not None:
        return "skimage-skeleton"
    return "centroid-slice"


def _extract_skimage_skeleton_graph(mask: AirwayMask) -> CenterlineGraph | None:
    skeletonize = _optional_callable("skimage.morphology", "skeletonize")
    if skeletonize is None:
        return None
    skeleton_any: Any = skeletonize(mask.data)
    skeleton = np.asarray(skeleton_any, dtype=np.bool_)
    if int(np.count_nonzero(skeleton)) < 2:
        return None
    return _graph_from_skeleton(mask, skeleton)


def _extract_centroid_slice_graph(mask: AirwayMask) -> CenterlineGraph:
    coordinates = np.argwhere(mask.data)
    if coordinates.size == 0:
        raise MeshGenError("cannot extract centreline from an empty airway mask")
    extents = coordinates.max(axis=0) - coordinates.min(axis=0)
    axis = int(np.argmax(extents))
    axis_values = range(int(coordinates[:, axis].min()), int(coordinates[:, axis].max()) + 1)
    samples: list[np.ndarray] = []
    radii: list[float] = []
    plane_spacing = [mask.spacing[index] for index in range(3) if index != axis]
    voxel_area = float(plane_spacing[0] * plane_spacing[1])
    for axis_value in axis_values:
        slice_coordinates = coordinates[coordinates[:, axis] == axis_value]
        if slice_coordinates.size == 0:
            continue
        centroid = np.asarray(slice_coordinates.mean(axis=0), dtype=np.float64)
        radius = math.sqrt(max(float(slice_coordinates.shape[0]) * voxel_area, 1.0e-9) / math.pi)
        samples.append(centroid)
        radii.append(radius)
    return _graph_from_ordered_samples(mask, samples, radii, "centroid-slice")


def _graph_from_coordinates(mask: AirwayMask, coordinates: np.ndarray, method: str) -> CenterlineGraph:
    extents = coordinates.max(axis=0) - coordinates.min(axis=0)
    axis = int(np.argmax(extents))
    order = np.argsort(coordinates[:, axis])
    ordered = [np.asarray(coordinates[index], dtype=np.float64) for index in order]
    radius = max(min(mask.spacing) * 0.5, 1.0e-3)
    radii = [radius for _ in ordered]
    return _graph_from_ordered_samples(mask, ordered, radii, method)


def _graph_from_skeleton(mask: AirwayMask, skeleton: BoolArray) -> CenterlineGraph | None:
    coordinates = np.argwhere(skeleton)
    if coordinates.shape[0] < 2:
        return None

    radii_by_voxel = _distance_transform(mask)
    coord_tuples = [(int(row[0]), int(row[1]), int(row[2])) for row in coordinates]
    index_by_coord = {coord: index for index, coord in enumerate(coord_tuples)}
    adjacency = _skeleton_adjacency(coord_tuples, index_by_coord)
    root_index = _root_skeleton_index(coord_tuples, radii_by_voxel)
    parent, ordered, distance_by_index = _skeleton_tree(root_index, coord_tuples, adjacency, mask.affine)
    if len(ordered) < 2:
        return None

    children: dict[int, list[int]] = defaultdict(list)
    for child_index, parent_index in enumerate(parent):
        if parent_index != -1:
            children[parent_index].append(child_index)

    branch_by_index: dict[int, str] = {root_index: "airway-main"}
    generation_by_index: dict[int, int] = {root_index: 0}
    branch_counter = 0
    for index in ordered:
        child_indices = children.get(index, [])
        if not child_indices:
            continue
        continuing_child = max(child_indices, key=lambda child: _subtree_size(child, children))
        for child_index in child_indices:
            if index == root_index or (len(child_indices) == 1 and child_index == continuing_child):
                branch_by_index[child_index] = branch_by_index[index]
                generation_by_index[child_index] = generation_by_index[index]
            else:
                branch_counter += 1
                branch_by_index[child_index] = f"airway-b{branch_counter:03d}"
                generation_by_index[child_index] = generation_by_index[index] + 1

    nodes: list[CenterlineNode] = []
    node_id_by_index: dict[int, str] = {}
    for node_number, index in enumerate(ordered):
        coord = coord_tuples[index]
        node_id = f"n{node_number:04d}"
        node_id_by_index[index] = node_id
        radius = max(float(radii_by_voxel[coord]), min(mask.spacing) * 0.5)
        nodes.append(
            CenterlineNode(
                node_id=node_id,
                point_mm=voxel_to_world(mask.affine, (float(coord[0]), float(coord[1]), float(coord[2]))),
                radius_mm=radius,
                branch_id=branch_by_index[index],
                voxel_index=(float(coord[0]), float(coord[1]), float(coord[2])),
                generation=generation_by_index[index],
                distance_from_root_mm=float(distance_by_index[index]),
                accessible=True,
            )
        )

    edges: list[CenterlineEdge] = []
    for index in ordered:
        parent_index = parent[index]
        if parent_index == -1:
            continue
        source_id = node_id_by_index[parent_index]
        target_id = node_id_by_index[index]
        source_point = voxel_to_world(mask.affine, _float_coord(coord_tuples[parent_index]))
        target_point = voxel_to_world(mask.affine, _float_coord(coord_tuples[index]))
        edges.append(
            CenterlineEdge(
                source=source_id,
                target=target_id,
                length_mm=_distance(source_point, target_point),
                branch_id=branch_by_index[index],
                generation=generation_by_index[index],
                accessible=True,
            )
        )

    return CenterlineGraph(
        case_id=mask.case_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        root_id=node_id_by_index[root_index],
        method="skimage-skeleton-graph",
    )


def _distance_transform(mask: AirwayMask) -> np.ndarray:
    distance_transform = _optional_callable("scipy.ndimage", "distance_transform_edt")
    if distance_transform is None:
        return np.full(mask.data.shape, min(mask.spacing) * 0.5, dtype=np.float64)
    distances_any: Any = distance_transform(mask.data, sampling=mask.spacing)
    return np.asarray(distances_any, dtype=np.float64)


def _skeleton_adjacency(
    coord_tuples: list[tuple[int, int, int]],
    index_by_coord: dict[tuple[int, int, int], int],
) -> list[list[int]]:
    offsets = [
        (di, dj, dk)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        for dk in (-1, 0, 1)
        if not (di == 0 and dj == 0 and dk == 0)
    ]
    adjacency: list[list[int]] = [[] for _ in coord_tuples]
    for index, coord in enumerate(coord_tuples):
        for offset in offsets:
            neighbor = (coord[0] + offset[0], coord[1] + offset[1], coord[2] + offset[2])
            neighbor_index = index_by_coord.get(neighbor)
            if neighbor_index is not None:
                adjacency[index].append(neighbor_index)
    return adjacency


def _root_skeleton_index(coord_tuples: list[tuple[int, int, int]], radii_by_voxel: np.ndarray) -> int:
    return max(range(len(coord_tuples)), key=lambda index: float(radii_by_voxel[coord_tuples[index]]))


def _skeleton_tree(
    root_index: int,
    coord_tuples: list[tuple[int, int, int]],
    adjacency: list[list[int]],
    affine: np.ndarray,
) -> tuple[list[int], list[int], list[float]]:
    parent = [-1 for _ in coord_tuples]
    distances = [math.inf for _ in coord_tuples]
    distances[root_index] = 0.0
    queue: list[tuple[float, int]] = [(0.0, root_index)]
    visited: set[int] = set()
    ordered: list[int] = []
    while queue:
        distance, index = heappop(queue)
        if index in visited:
            continue
        visited.add(index)
        ordered.append(index)
        for neighbor in adjacency[index]:
            source = voxel_to_world(affine, _float_coord(coord_tuples[index]))
            target = voxel_to_world(affine, _float_coord(coord_tuples[neighbor]))
            candidate = distance + _distance(source, target)
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                parent[neighbor] = index
                heappush(queue, (candidate, neighbor))
    finite_distances = [0.0 if not math.isfinite(value) else value for value in distances]
    return parent, ordered, finite_distances


def _subtree_size(index: int, children: dict[int, list[int]]) -> int:
    total = 1
    for child in children.get(index, []):
        total += _subtree_size(child, children)
    return total


def _float_coord(coord: tuple[int, int, int]) -> tuple[float, float, float]:
    return (float(coord[0]), float(coord[1]), float(coord[2]))


def _skeleton_diameter_path(coordinates: np.ndarray) -> list[np.ndarray]:
    coord_tuples = [(int(row[0]), int(row[1]), int(row[2])) for row in coordinates]
    index_by_coord = {coord: index for index, coord in enumerate(coord_tuples)}
    adjacency: list[list[int]] = [[] for _ in coord_tuples]
    offsets = [
        (di, dj, dk)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        for dk in (-1, 0, 1)
        if not (di == 0 and dj == 0 and dk == 0)
    ]
    for index, coord in enumerate(coord_tuples):
        for offset in offsets:
            neighbor = (coord[0] + offset[0], coord[1] + offset[1], coord[2] + offset[2])
            neighbor_index = index_by_coord.get(neighbor)
            if neighbor_index is not None:
                adjacency[index].append(neighbor_index)
    endpoint = next((index for index, neighbors in enumerate(adjacency) if len(neighbors) == 1), 0)
    first, _, _ = _farthest_skeleton_node(endpoint, adjacency)
    second, parents, distances = _farthest_skeleton_node(first, adjacency)
    if distances[second] < 1:
        return [np.asarray(coordinates[endpoint], dtype=np.float64)]
    path_indices: list[int] = []
    current = second
    while current != -1:
        path_indices.append(current)
        if current == first:
            break
        current = parents[current]
    path_indices.reverse()
    return [np.asarray(coordinates[index], dtype=np.float64) for index in path_indices]


def _farthest_skeleton_node(start: int, adjacency: list[list[int]]) -> tuple[int, list[int], list[int]]:
    parents = [-1 for _ in adjacency]
    distances = [-1 for _ in adjacency]
    distances[start] = 0
    queue: deque[int] = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if distances[neighbor] != -1:
                continue
            distances[neighbor] = distances[current] + 1
            parents[neighbor] = current
            queue.append(neighbor)
    farthest = max(range(len(distances)), key=lambda index: distances[index])
    return farthest, parents, distances


def _radius_samples(mask: AirwayMask, coordinates: list[np.ndarray]) -> list[float]:
    distance_transform = _optional_callable("scipy.ndimage", "distance_transform_edt")
    if distance_transform is not None:
        distances_any: Any = distance_transform(mask.data, sampling=mask.spacing)
        distances = np.asarray(distances_any, dtype=np.float64)
        return [
            max(float(distances[int(coord[0]), int(coord[1]), int(coord[2])]), min(mask.spacing) * 0.5)
            for coord in coordinates
        ]
    return [min(mask.spacing) * 0.5 for _ in coordinates]


def _graph_from_ordered_samples(
    mask: AirwayMask,
    samples: list[np.ndarray],
    radii: list[float],
    method: str,
) -> CenterlineGraph:
    nodes: list[CenterlineNode] = []
    edges: list[CenterlineEdge] = []
    previous_point: tuple[float, float, float] | None = None
    previous_id: str | None = None
    distance_from_root = 0.0
    for index, sample in enumerate(samples):
        node_id = f"n{index:04d}"
        point = voxel_to_world(mask.affine, (float(sample[0]), float(sample[1]), float(sample[2])))
        radius = radii[index]
        edge_length = 0.0
        if previous_point is not None:
            edge_length = _distance(previous_point, point)
            distance_from_root += edge_length
        nodes.append(
            CenterlineNode(
                node_id=node_id,
                point_mm=point,
                radius_mm=radius,
                branch_id="airway-main",
                voxel_index=(float(sample[0]), float(sample[1]), float(sample[2])),
                distance_from_root_mm=distance_from_root,
            )
        )
        if previous_id is not None:
            edges.append(
                CenterlineEdge(source=previous_id, target=node_id, length_mm=edge_length, branch_id="airway-main")
            )
        previous_id = node_id
        previous_point = point
    return CenterlineGraph(
        case_id=mask.case_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        root_id=nodes[0].node_id,
        method=method,
    )


def apply_bronchoscope_accessibility(
    graph: CenterlineGraph,
    *,
    min_accessible_diameter_mm: float,
    max_accessible_depth_mm: float | None = None,
) -> CenterlineGraph:
    """Mark nodes reachable by a bronchoscope diameter and optional working-length limit."""
    locally_accessible = {
        node.node_id: (
            (node.radius_mm * 2.0) >= min_accessible_diameter_mm
            and (max_accessible_depth_mm is None or node.distance_from_root_mm <= max_accessible_depth_mm)
        )
        for node in graph.nodes
    }
    children_by_id: dict[str, list[str]] = defaultdict(list)
    edge_by_pair = {(edge.source, edge.target): edge for edge in graph.edges}
    for edge in graph.edges:
        children_by_id[edge.source].append(edge.target)

    reachable: set[str] = set()
    if locally_accessible.get(graph.root_id, False):
        queue: deque[str] = deque([graph.root_id])
        while queue:
            node_id = queue.popleft()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            for child_id in children_by_id.get(node_id, []):
                if locally_accessible.get(child_id, False):
                    queue.append(child_id)

    node_by_id = {}
    for node in graph.nodes:
        node_by_id[node.node_id] = CenterlineNode(
            node_id=node.node_id,
            point_mm=node.point_mm,
            radius_mm=node.radius_mm,
            branch_id=node.branch_id,
            voxel_index=node.voxel_index,
            generation=node.generation,
            distance_from_root_mm=node.distance_from_root_mm,
            accessible=node.node_id in reachable,
        )

    edges = tuple(
        CenterlineEdge(
            source=edge.source,
            target=edge.target,
            length_mm=edge.length_mm,
            branch_id=edge.branch_id,
            generation=edge.generation,
            accessible=(edge.source, edge.target) in edge_by_pair
            and node_by_id[edge.source].accessible
            and node_by_id[edge.target].accessible,
        )
        for edge in graph.edges
    )
    return CenterlineGraph(
        case_id=graph.case_id,
        nodes=tuple(node_by_id[node.node_id] for node in graph.nodes),
        edges=edges,
        root_id=graph.root_id,
        method=graph.method,
        schema_version=graph.schema_version,
    )


def _nearest_node_indices(coordinates: np.ndarray, node_coordinates: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(node_coordinates)
        _, indices = tree.query(coordinates, k=1)
        return np.asarray(indices, dtype=np.int64)
    except ImportError:
        nearest = np.empty(coordinates.shape[0], dtype=np.int64)
        chunk_size = 8192
        for start in range(0, coordinates.shape[0], chunk_size):
            stop = min(start + chunk_size, coordinates.shape[0])
            delta = coordinates[start:stop, None, :] - node_coordinates[None, :, :]
            distances = np.sum(delta * delta, axis=2)
            nearest[start:stop] = np.argmin(distances, axis=1)
        return nearest


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    return float(np.linalg.norm(delta))


def _optional_callable(module_name: str, attribute_name: str) -> Any | None:
    try:
        module = import_module(module_name)
    except ImportError:
        return None
    value: Any = getattr(module, attribute_name, None)
    return value if callable(value) else None
