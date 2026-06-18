"""Case-level geometry QA for mesh generation outputs."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import cast

import numpy as np
from synairg_core.volume_manifest import JsonValue

from mesh_gen.artifacts import artifact_manifest
from mesh_gen.errors import MeshGenError
from mesh_gen.schemas import AirwayMask, AirwayMesh, CenterlineGraph, CenterlineNode
from mesh_gen.topology import TopologyReport, connected_components

MIN_REAL_VOLUME_VOXELS = 1_000_000
DEFAULT_MAX_FOREGROUND_FRACTION = 0.03
DEFAULT_MAX_RADIUS_P95_MM = 15.0


def _json_bool(value: JsonValue, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float | str):
        return bool(value)
    return default


def _json_int(value: JsonValue, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def build_quality_report(
    *,
    mask: AirwayMask,
    mesh: AirwayMesh,
    graph: CenterlineGraph,
    topology: TopologyReport,
    artifacts: tuple[Path, ...],
) -> dict[str, JsonValue]:
    """Build a deterministic QA report for a generated geometry case."""
    bounds_min, bounds_max = mesh.bounds
    node_distances = [float(node.distance_from_root_mm) for node in graph.nodes]
    accessible_node_distances = [float(node.distance_from_root_mm) for node in graph.nodes if node.accessible]
    quality = _mesh_quality(mesh)
    fidelity = _surface_fidelity(mask, mesh)
    opening = _opening_quality(mesh, graph)
    centerline_quality = _centerline_quality(graph)
    _, mask_component_count = connected_components(mask.data)
    degenerate_face_count = cast(int, quality["degenerate_face_count"])
    non_manifold_edge_count = cast(int, quality["non_manifold_edge_count"])
    mesh_component_count = cast(int, quality["mesh_component_count"])
    expected_boundary_openings = cast(bool, opening["has_expected_boundary_openings"])
    root_reaches_all_nodes = cast(bool, centerline_quality["root_reaches_all_nodes"])
    accessible_route_non_empty = cast(bool, centerline_quality["accessible_route_non_empty"])
    checks = {
        "mask_non_empty": topology.repaired_voxels > 0,
        "mask_single_connected_component": mask_component_count == 1,
        "mesh_non_empty": mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0,
        "mesh_vertices_finite": bool(np.isfinite(mesh.vertices).all()),
        "mesh_single_connected_component": mesh_component_count == 1,
        "mesh_has_no_degenerate_faces": degenerate_face_count == 0,
        "mesh_has_no_non_manifold_edges": non_manifold_edge_count == 0,
        "mesh_has_expected_boundary_openings": expected_boundary_openings,
        "centerline_non_empty": len(graph.nodes) > 0,
        "centerline_root_reaches_all_nodes": root_reaches_all_nodes,
        "centerline_accessible_route_non_empty": accessible_route_non_empty,
        "all_artifacts_exist": all(path.exists() for path in artifacts),
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": "1.0",
        "case_id": mask.case_id,
        "status": status,
        "checks": {key: bool(value) for key, value in checks.items()},
        "mask": {
            "shape": [int(value) for value in mask.data.shape],
            "spacing_mm": [float(value) for value in mask.spacing],
            "original_voxels": topology.original_voxels,
            "repaired_voxels": topology.repaired_voxels,
            "component_count": topology.component_count,
            "removed_component_count": topology.removed_component_count,
            "kept_component_count": topology.kept_component_count,
            "bridged_component_count": topology.bridged_component_count,
            "bridge_voxels": topology.bridge_voxels,
            "min_component_voxels": topology.min_component_voxels,
            "max_bridge_distance_mm": topology.max_bridge_distance_mm,
            "final_component_count": topology.final_component_count,
            "unbridged_component_count": topology.unbridged_component_count,
        },
        "mesh": {
            "method": mesh.method,
            "vertex_count": int(mesh.vertices.shape[0]),
            "face_count": int(mesh.faces.shape[0]),
            "bounds_min_mm": [float(value) for value in bounds_min],
            "bounds_max_mm": [float(value) for value in bounds_max],
            "hollow": _json_bool(mesh.metadata.get("hollow", False)),
            "top_uncapped": _json_bool(mesh.metadata.get("top_uncapped", False)),
            "top_opening_count": _json_int(mesh.metadata.get("top_opening_count", 0)),
            "removed_cap_faces": _json_int(mesh.metadata.get("removed_cap_faces", 0)),
            "boundary_edge_count": _json_int(mesh.metadata.get("boundary_edge_count", 0)),
            "quality": quality,
            "opening": opening,
            "surface_fidelity": fidelity,
            "metadata": mesh.metadata,
        },
        "centerline": {
            "method": graph.method,
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "root_id": graph.root_id,
            "accessible_node_count": sum(1 for node in graph.nodes if node.accessible),
            "accessible_edge_count": sum(1 for edge in graph.edges if edge.accessible),
            "max_distance_from_root_mm": max(node_distances, default=0.0),
            "max_accessible_distance_from_root_mm": max(accessible_node_distances, default=0.0),
            **centerline_quality,
        },
        "artifacts": artifact_manifest(artifacts),
    }


def assert_plausible_airway_mask(
    mask: AirwayMask,
    *,
    max_foreground_fraction: float = DEFAULT_MAX_FOREGROUND_FRACTION,
    max_radius_p95_mm: float = DEFAULT_MAX_RADIUS_P95_MM,
) -> dict[str, JsonValue]:
    """Reject obvious lung/lobe/organ masks before they become airway meshes."""
    total_voxels = int(np.prod(mask.data.shape))
    foreground_voxels = int(np.count_nonzero(mask.data))
    foreground_fraction = foreground_voxels / max(total_voxels, 1)
    metrics: dict[str, JsonValue] = {
        "checked": total_voxels >= MIN_REAL_VOLUME_VOXELS,
        "foreground_fraction": float(foreground_fraction),
        "max_foreground_fraction": float(max_foreground_fraction),
    }
    if total_voxels < MIN_REAL_VOLUME_VOXELS:
        return metrics

    failures: list[str] = []
    if foreground_fraction > max_foreground_fraction:
        failures.append(
            f"foreground fraction {foreground_fraction:.4f} exceeds airway limit {max_foreground_fraction:.4f}"
        )

    radius_p95 = _radius_p95_mm(mask)
    if radius_p95 is not None:
        metrics["radius_p95_mm"] = float(radius_p95)
        metrics["max_radius_p95_mm"] = float(max_radius_p95_mm)
        if radius_p95 > max_radius_p95_mm:
            failures.append(f"mask radius p95 {radius_p95:.2f} mm exceeds airway limit {max_radius_p95_mm:.2f} mm")

    if failures:
        raise MeshGenError("segmentation does not look like an airway mask: " + "; ".join(failures))
    return metrics


def _mesh_quality(mesh: AirwayMesh) -> dict[str, JsonValue]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_vertices = vertices[faces]
    edge_ab = face_vertices[:, 1] - face_vertices[:, 0]
    edge_bc = face_vertices[:, 2] - face_vertices[:, 1]
    edge_ca = face_vertices[:, 0] - face_vertices[:, 2]
    face_areas = np.linalg.norm(np.cross(edge_ab, -edge_ca), axis=1) * 0.5
    edge_lengths = np.concatenate(
        (
            np.linalg.norm(edge_ab, axis=1),
            np.linalg.norm(edge_bc, axis=1),
            np.linalg.norm(edge_ca, axis=1),
        )
    )
    used_vertices = np.unique(faces.ravel())
    edge_counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for edge in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = _edge_key(edge[0], edge[1])
            edge_counts[key] = edge_counts.get(key, 0) + 1

    return {
        "face_area_min_mm2": _finite_stat(face_areas, "min"),
        "face_area_p50_mm2": _finite_percentile(face_areas, 50.0),
        "face_area_p95_mm2": _finite_percentile(face_areas, 95.0),
        "edge_length_p50_mm": _finite_percentile(edge_lengths, 50.0),
        "edge_length_p95_mm": _finite_percentile(edge_lengths, 95.0),
        "degenerate_face_count": int(np.count_nonzero(face_areas <= 1.0e-12)),
        "isolated_vertex_count": int(vertices.shape[0] - used_vertices.size),
        "boundary_edge_count": int(sum(1 for count in edge_counts.values() if count == 1)),
        "non_manifold_edge_count": int(sum(1 for count in edge_counts.values() if count > 2)),
        "mesh_component_count": _mesh_component_count(faces),
    }


def _mesh_component_count(faces: np.ndarray) -> int:
    adjacency: dict[int, set[int]] = defaultdict(set)
    used_vertices: set[int] = set()
    for face in faces:
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        used_vertices.update((a, b, c))
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    remaining = set(used_vertices)
    component_count = 0
    while remaining:
        component_count += 1
        queue: deque[int] = deque([remaining.pop()])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return component_count


def _opening_quality(mesh: AirwayMesh, graph: CenterlineGraph) -> dict[str, JsonValue]:
    loops = _boundary_loops(mesh.faces)
    expected = _expected_top_opening(graph)
    loop_reports: list[JsonValue] = []
    approved_count = 0
    for loop in loops:
        coordinates = mesh.vertices[np.asarray(sorted(loop), dtype=np.int64)]
        centroid = coordinates.mean(axis=0)
        mean_radius = float(np.mean(np.linalg.norm(coordinates - centroid[np.newaxis, :], axis=1)))
        approved = False
        distance_to_expected: float | None = None
        if expected is not None:
            center, radius = expected
            distance_to_expected = float(np.linalg.norm(centroid - center))
            approved = distance_to_expected <= max(radius * 2.5, 2.5) and mean_radius <= max(radius * 3.0, 3.0)
        if approved:
            approved_count += 1
        loop_reports.append(
            {
                "vertex_count": len(loop),
                "centroid_mm": [float(value) for value in centroid],
                "mean_radius_mm": mean_radius,
                "distance_to_expected_opening_mm": distance_to_expected,
                "approved": approved,
            }
        )

    boundary_loop_count = len(loops)
    accidental_count = boundary_loop_count - approved_count
    hollow = _json_bool(mesh.metadata.get("hollow", False))
    if hollow:
        has_expected_boundary_openings = (
            _json_bool(mesh.metadata.get("top_uncapped", False))
            and _json_int(mesh.metadata.get("top_opening_count", 0)) == 1
            and approved_count >= 1
            and accidental_count == 0
        )
    else:
        has_expected_boundary_openings = boundary_loop_count == 0
    return {
        "boundary_loop_count": boundary_loop_count,
        "approved_opening_count": approved_count,
        "accidental_boundary_loop_count": accidental_count,
        "has_expected_boundary_openings": has_expected_boundary_openings,
        "loops": loop_reports,
    }


def _boundary_loops(faces: np.ndarray) -> list[set[int]]:
    edge_counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for edge in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = _edge_key(edge[0], edge[1])
            edge_counts[key] = edge_counts.get(key, 0) + 1
    adjacency: dict[int, set[int]] = defaultdict(set)
    for (source, target), count in edge_counts.items():
        if count != 1:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    remaining = set(adjacency)
    loops: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        queue: deque[int] = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        loops.append(component)
    return loops


def _edge_key(source: int, target: int) -> tuple[int, int]:
    return (source, target) if source <= target else (target, source)


def _expected_top_opening(graph: CenterlineGraph, *, superior_axis: int = 2) -> tuple[np.ndarray, float] | None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    leaves = [node for node in graph.nodes if len(adjacency.get(node.node_id, set())) == 1]
    candidates = [node for node in leaves if node.accessible] or leaves
    if not candidates:
        return None
    node = max(candidates, key=lambda item: (item.point_mm[superior_axis], item.radius_mm))
    return np.asarray(node.point_mm, dtype=np.float64), float(node.radius_mm)


def _centerline_quality(graph: CenterlineGraph) -> dict[str, JsonValue]:
    node_by_id = {node.node_id: node for node in graph.nodes}
    adjacency: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
        children[edge.source].add(edge.target)
    reachable = _reachable_centerline_nodes(graph.root_id, adjacency)
    accessible_nodes = [node for node in graph.nodes if node.accessible]
    terminal_nodes = [node for node in graph.nodes if len(adjacency.get(node.node_id, set())) <= 1]
    branch_ids = {node.branch_id for node in graph.nodes}
    radii = np.asarray([node.radius_mm for node in graph.nodes], dtype=np.float64)
    abrupt_radius_change_count = _abrupt_radius_change_count(graph, node_by_id)
    accessible_terminal_count = sum(1 for node in terminal_nodes if node.accessible)
    return {
        "root_reachable_node_count": len(reachable),
        "root_reaches_all_nodes": len(reachable) == len(graph.nodes),
        "accessible_route_non_empty": bool(accessible_nodes and node_by_id[graph.root_id].accessible),
        "accessible_node_fraction": len(accessible_nodes) / max(len(graph.nodes), 1),
        "accessible_terminal_node_count": accessible_terminal_count,
        "terminal_node_count": len(terminal_nodes),
        "branch_count": len(branch_ids),
        "max_generation": max((node.generation for node in graph.nodes), default=0),
        "total_centerline_length_mm": float(sum(edge.length_mm for edge in graph.edges)),
        "radius_p05_mm": _finite_percentile(radii, 5.0),
        "radius_p50_mm": _finite_percentile(radii, 50.0),
        "radius_p95_mm": _finite_percentile(radii, 95.0),
        "abrupt_radius_change_count": abrupt_radius_change_count,
    }


def _reachable_centerline_nodes(root_id: str, adjacency: dict[str, set[str]]) -> set[str]:
    reachable = {root_id}
    queue: deque[str] = deque([root_id])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, set()):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    return reachable


def _abrupt_radius_change_count(graph: CenterlineGraph, node_by_id: dict[str, CenterlineNode]) -> int:
    abrupt = 0
    for edge in graph.edges:
        source_radius = node_by_id[edge.source].radius_mm
        target_radius = node_by_id[edge.target].radius_mm
        smaller = max(min(source_radius, target_radius), 1.0e-6)
        larger = max(source_radius, target_radius)
        if larger / smaller > 2.5:
            abrupt += 1
    return abrupt


def _surface_fidelity(mask: AirwayMask, mesh: AirwayMesh) -> dict[str, JsonValue]:
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt  # type: ignore[import-untyped]
    except ImportError:
        return {"checked": False}
    surface = np.asarray(mask.data & ~binary_erosion(mask.data), dtype=np.bool_)
    if int(np.count_nonzero(surface)) == 0:
        return {"checked": False}
    distances = np.asarray(distance_transform_edt(~surface, sampling=mask.spacing), dtype=np.float64)
    inverse_affine = np.linalg.inv(mask.affine)
    homogeneous = np.ones((mesh.vertices.shape[0], 4), dtype=np.float64)
    homogeneous[:, :3] = mesh.vertices
    voxel_vertices = homogeneous @ inverse_affine.T
    indices = np.rint(voxel_vertices[:, :3]).astype(np.int64)
    for axis, size in enumerate(mask.data.shape):
        indices[:, axis] = np.clip(indices[:, axis], 0, size - 1)
    samples = distances[indices[:, 0], indices[:, 1], indices[:, 2]]
    return {
        "checked": True,
        "vertex_to_mask_surface_p50_mm": _finite_percentile(samples, 50.0),
        "vertex_to_mask_surface_p95_mm": _finite_percentile(samples, 95.0),
        "vertex_to_mask_surface_max_mm": _finite_stat(samples, "max"),
    }


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, percentile))


def _finite_stat(values: np.ndarray, name: str) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return 0.0
    if name == "min":
        return float(finite.min())
    if name == "max":
        return float(finite.max())
    raise MeshGenError(f"unknown finite stat {name!r}")


def _radius_p95_mm(mask: AirwayMask) -> float | None:
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return None
    distances = np.asarray(distance_transform_edt(mask.data, sampling=mask.spacing), dtype=np.float64)
    foreground_distances = distances[mask.data]
    if foreground_distances.size == 0:
        return None
    return float(np.percentile(foreground_distances, 95.0))
