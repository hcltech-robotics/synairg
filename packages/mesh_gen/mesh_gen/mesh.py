"""Airway mesh generation, smoothing, repair, normals, and decimation."""

from __future__ import annotations

from collections import defaultdict
from importlib import import_module
from typing import Any

import numpy as np

from mesh_gen.errors import MeshGenError
from mesh_gen.schemas import AirwayMask, AirwayMesh, CenterlineGraph, FloatArray, IntArray


def generate_airway_mesh(
    mask: AirwayMask,
    *,
    smooth_iterations: int = 0,
    smoothing_method: str = "laplacian",
    taubin_pass_band: float = 0.08,
    decimate_fraction: float = 0.0,
    centerline_graph: CenterlineGraph | None = None,
    uncap_top: bool = False,
    top_opening_radius_scale: float = 1.25,
    proximal_opening_trim: bool = False,
    normal_orientation: str = "outward",
) -> AirwayMesh:
    """Generate a triangle mesh from a binary airway mask."""
    _validate_refinement_inputs(
        smooth_iterations=smooth_iterations,
        smoothing_method=smoothing_method,
        taubin_pass_band=taubin_pass_band,
        normal_orientation=normal_orientation,
    )
    vertices, faces, method = _marching_cubes(mask)
    vertices, faces = repair_mesh(vertices, faces)
    smoothing_backend = "none"
    actual_smoothing_method = "none"
    if smooth_iterations > 0:
        actual_smoothing_method = smoothing_method
        if smoothing_method == "laplacian":
            vertices = smooth_mesh(vertices, faces, iterations=smooth_iterations)
            smoothing_backend = "laplacian"
        elif smoothing_method == "taubin":
            vertices, smoothing_backend = smooth_mesh_taubin(
                vertices,
                faces,
                iterations=smooth_iterations,
                pass_band=taubin_pass_band,
            )
        vertices, faces = repair_mesh(vertices, faces)
    decimation_backend = "none"
    if decimate_fraction > 0:
        vertices, faces, decimation_backend = _decimate_mesh_with_backend(
            vertices,
            faces,
            decimate_fraction=decimate_fraction,
        )
    opening_report: dict[str, int | float | bool] = {
        "hollow": False,
        "top_uncapped": False,
        "top_opening_count": 0,
        "removed_cap_faces": 0,
        "top_opening_radius_scale": float(top_opening_radius_scale),
    }
    if uncap_top:
        if centerline_graph is None:
            raise MeshGenError("uncap_top requires a centerline graph")
        faces, opening_report = uncap_top_from_centerline(
            vertices,
            faces,
            centerline_graph,
            top_opening_radius_scale=top_opening_radius_scale,
            proximal_opening_trim=proximal_opening_trim,
            axial_margin_mm=min(mask.spacing),
        )
        vertices, faces = repair_mesh(vertices, faces)
    vertices, faces, component_healing_report = keep_largest_mesh_component(vertices, faces)
    if normal_orientation == "inward":
        faces = flip_face_winding(faces)
    normals = compute_vertex_normals(vertices, faces)
    boundary_edge_count = count_boundary_edges(faces)
    return AirwayMesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        method=method,
        metadata={
            "smoothing_method": actual_smoothing_method,
            "smooth_iterations": smooth_iterations,
            "smoothing_backend": smoothing_backend,
            "taubin_pass_band": float(taubin_pass_band),
            "decimate_fraction": decimate_fraction,
            "decimation_backend": decimation_backend,
            "normal_orientation": normal_orientation,
            "boundary_edge_count": boundary_edge_count,
            **component_healing_report,
            **opening_report,
        },
    )


def repair_mesh(vertices: FloatArray, faces: IntArray) -> tuple[FloatArray, IntArray]:
    """Remove degenerate and duplicate faces, then reindex vertices."""
    if vertices.size == 0 or faces.size == 0:
        raise MeshGenError("cannot repair an empty mesh")
    valid_faces: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for row in faces:
        face = (int(row[0]), int(row[1]), int(row[2]))
        if len(set(face)) != 3:
            continue
        if _triangle_area(vertices, face) <= 1.0e-12:
            continue
        ordered = sorted(face)
        key = (ordered[0], ordered[1], ordered[2])
        if key in seen:
            continue
        seen.add(key)
        valid_faces.append(face)
    if not valid_faces:
        raise MeshGenError("mesh repair removed all faces")
    used = sorted({index for face in valid_faces for index in face})
    remap = {old: new for new, old in enumerate(used)}
    repaired_vertices = np.asarray([vertices[index] for index in used], dtype=np.float64)
    repaired_faces = np.asarray([[remap[index] for index in face] for face in valid_faces], dtype=np.int64)
    return repaired_vertices, repaired_faces


def keep_largest_mesh_component(vertices: FloatArray, faces: IntArray) -> tuple[FloatArray, IntArray, dict[str, int]]:
    """Remove disconnected surface shell fragments and keep the dominant component."""
    components = _face_components(faces)
    if len(components) <= 1:
        return vertices, faces, {
            "mesh_component_count_before_healing": len(components),
            "removed_mesh_component_count": 0,
            "removed_mesh_component_face_count": 0,
        }

    largest = max(components, key=len)
    keep_face_indices = np.asarray(sorted(largest), dtype=np.int64)
    kept_faces = np.asarray(faces[keep_face_indices], dtype=np.int64)
    repaired_vertices, repaired_faces = repair_mesh(vertices, kept_faces)
    removed_face_count = int(faces.shape[0] - kept_faces.shape[0])
    return repaired_vertices, repaired_faces, {
        "mesh_component_count_before_healing": len(components),
        "removed_mesh_component_count": len(components) - 1,
        "removed_mesh_component_face_count": removed_face_count,
    }


def _face_components(faces: IntArray) -> list[set[int]]:
    vertex_to_faces: dict[int, list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            vertex_to_faces[int(vertex_index)].append(face_index)

    remaining = set(range(int(faces.shape[0])))
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for vertex_index in faces[current]:
                for neighbor in vertex_to_faces[int(vertex_index)]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
        components.append(component)
    return components


def _triangle_area(vertices: FloatArray, face: tuple[int, int, int]) -> float:
    a, b, c = face
    edge_ab = vertices[b] - vertices[a]
    edge_ac = vertices[c] - vertices[a]
    return float(np.linalg.norm(np.cross(edge_ab, edge_ac)) * 0.5)


def smooth_mesh(vertices: FloatArray, faces: IntArray, *, iterations: int, alpha: float = 0.35) -> FloatArray:
    """Apply lightweight Laplacian smoothing."""
    if iterations <= 0:
        return vertices
    adjacency: dict[int, set[int]] = defaultdict(set)
    for face in faces:
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    smoothed = vertices.copy()
    for _ in range(iterations):
        updated = smoothed.copy()
        for vertex_index, neighbors in adjacency.items():
            if not neighbors:
                continue
            neighbor_mean = smoothed[list(neighbors)].mean(axis=0)
            updated[vertex_index] = (1.0 - alpha) * smoothed[vertex_index] + alpha * neighbor_mean
        smoothed = updated
    return np.asarray(smoothed, dtype=np.float64)


def smooth_mesh_taubin(
    vertices: FloatArray,
    faces: IntArray,
    *,
    iterations: int,
    pass_band: float,
) -> tuple[FloatArray, str]:
    """Apply shrinkage-resistant Taubin/windowed-sinc smoothing when VTK is available."""
    if iterations <= 0:
        return vertices, "none"
    if pass_band <= 0:
        raise MeshGenError("taubin_pass_band must be positive")
    pyvista_smoothed = _pyvista_smooth_taubin(vertices, faces, iterations=iterations, pass_band=pass_band)
    if pyvista_smoothed is not None:
        return pyvista_smoothed, "pyvista-smooth-taubin"
    return vertices, "skipped-no-taubin-backend"


def decimate_mesh(vertices: FloatArray, faces: IntArray, *, decimate_fraction: float) -> tuple[FloatArray, IntArray]:
    """Reduce mesh faces while preserving topology when a mesh backend is available."""
    decimated_vertices, decimated_faces, _ = _decimate_mesh_with_backend(
        vertices,
        faces,
        decimate_fraction=decimate_fraction,
    )
    return decimated_vertices, decimated_faces


def _decimate_mesh_with_backend(
    vertices: FloatArray,
    faces: IntArray,
    *,
    decimate_fraction: float,
) -> tuple[FloatArray, IntArray, str]:
    """Reduce mesh faces using a topology-preserving backend, otherwise keep the mesh intact."""
    if not 0.0 <= decimate_fraction < 1.0:
        raise MeshGenError("decimate_fraction must be in [0.0, 1.0)")
    if decimate_fraction == 0.0:
        return vertices, faces, "none"
    pyvista_decimated = _pyvista_decimate_mesh(vertices, faces, decimate_fraction=decimate_fraction)
    if pyvista_decimated is not None:
        decimated_vertices, decimated_faces = pyvista_decimated
        return decimated_vertices, decimated_faces, "pyvista-decimate-pro-preserve-topology"
    return vertices, faces, "skipped-no-topology-preserving-backend"


def _pyvista_decimate_mesh(
    vertices: FloatArray,
    faces: IntArray,
    *,
    decimate_fraction: float,
) -> tuple[FloatArray, IntArray] | None:
    try:
        import pyvista as pv
    except ImportError:
        return None
    try:
        face_cells = np.column_stack((np.full(faces.shape[0], 3, dtype=np.int64), faces)).ravel()
        surface = pv.PolyData(np.asarray(vertices, dtype=np.float64), face_cells)
        decimated = surface.decimate_pro(
            reduction=float(decimate_fraction),
            preserve_topology=True,
            boundary_vertex_deletion=False,
        )
        raw_faces = np.asarray(decimated.faces, dtype=np.int64).reshape((-1, 4))
        if raw_faces.size == 0 or not np.all(raw_faces[:, 0] == 3):
            return None
        decimated_vertices = np.asarray(decimated.points, dtype=np.float64)
        decimated_faces = np.asarray(raw_faces[:, 1:4], dtype=np.int64)
        if decimated_vertices.size == 0 or decimated_faces.size == 0:
            return None
        return repair_mesh(decimated_vertices, decimated_faces)
    except Exception:
        return None


def _pyvista_smooth_taubin(
    vertices: FloatArray,
    faces: IntArray,
    *,
    iterations: int,
    pass_band: float,
) -> FloatArray | None:
    try:
        import pyvista as pv
    except ImportError:
        return None
    try:
        face_cells = np.column_stack((np.full(faces.shape[0], 3, dtype=np.int64), faces)).ravel()
        surface = pv.PolyData(np.asarray(vertices, dtype=np.float64), face_cells)
        smoothed = surface.smooth_taubin(
            n_iter=int(iterations),
            pass_band=float(pass_band),
            boundary_smoothing=True,
            feature_smoothing=False,
            non_manifold_smoothing=False,
        )
        smoothed_vertices = np.asarray(smoothed.points, dtype=np.float64)
        if smoothed_vertices.shape != vertices.shape or not np.isfinite(smoothed_vertices).all():
            return None
        return smoothed_vertices
    except Exception:
        return None


def uncap_top_from_centerline(
    vertices: FloatArray,
    faces: IntArray,
    graph: CenterlineGraph,
    *,
    top_opening_radius_scale: float = 1.25,
    proximal_opening_trim: bool = False,
    axial_margin_mm: float = 1.0,
    superior_axis: int = 2,
) -> tuple[IntArray, dict[str, int | float | bool]]:
    """Open the superior/proximal airway for bronchoscope entry."""
    if top_opening_radius_scale <= 0:
        raise MeshGenError("top_opening_radius_scale must be positive")
    opening = _top_opening(graph, superior_axis=superior_axis)
    if opening is None:
        return faces, {
            "hollow": False,
            "top_uncapped": False,
            "top_opening_count": 0,
            "removed_cap_faces": 0,
            "top_opening_radius_scale": float(top_opening_radius_scale),
            "proximal_opening_trim": bool(proximal_opening_trim),
            "proximal_trim_depth_mm": 0.0,
        }

    face_centers = vertices[faces].mean(axis=1)
    remove = np.zeros(faces.shape[0], dtype=np.bool_)
    center, direction, radius = opening
    delta = face_centers - center[np.newaxis, :]
    axial = delta @ direction
    proximal_trim_depth_mm = 0.0
    if proximal_opening_trim:
        proximal_trim_depth_mm = max(float(axial_margin_mm), radius * 0.6)
        remove |= axial >= -proximal_trim_depth_mm
    else:
        radial_vectors = delta - axial[:, np.newaxis] * direction[np.newaxis, :]
        radial = np.linalg.norm(radial_vectors, axis=1)
        backward_margin = max(float(axial_margin_mm), radius * 0.4)
        forward_margin = max(float(axial_margin_mm) * 1.5, radius * 1.5)
        remove |= (
            (axial >= -backward_margin)
            & (axial <= forward_margin)
            & (radial <= radius * top_opening_radius_scale)
        )

    kept_faces = np.asarray(faces[~remove], dtype=np.int64)
    if kept_faces.size == 0:
        raise MeshGenError("top opening removed every mesh face")
    removed = int(np.count_nonzero(remove))
    return kept_faces, {
        "hollow": removed > 0,
        "top_uncapped": removed > 0,
        "top_opening_count": 1 if removed > 0 else 0,
        "removed_cap_faces": removed,
        "top_opening_radius_scale": float(top_opening_radius_scale),
        "proximal_opening_trim": bool(proximal_opening_trim),
        "proximal_trim_depth_mm": float(proximal_trim_depth_mm),
    }


def count_boundary_edges(faces: IntArray) -> int:
    """Count edges belonging to exactly one face."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for face in faces:
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        for edge in ((a, b), (b, c), (c, a)):
            counts[_edge_key(edge[0], edge[1])] += 1
    return sum(1 for count in counts.values() if count == 1)


def _edge_key(source: int, target: int) -> tuple[int, int]:
    return (source, target) if source <= target else (target, source)


def flip_face_winding(faces: IntArray) -> IntArray:
    """Return faces with opposite winding for inward-facing normals."""
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise MeshGenError("mesh faces must have shape (m, 3)")
    return np.asarray(faces[:, (0, 2, 1)], dtype=np.int64)


def compute_vertex_normals(vertices: FloatArray, faces: IntArray) -> FloatArray:
    """Compute area-weighted vertex normals."""
    normals = np.zeros(vertices.shape, dtype=np.float64)
    for face in faces:
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        edge_ab = vertices[b] - vertices[a]
        edge_ac = vertices[c] - vertices[a]
        normal = np.cross(edge_ab, edge_ac)
        norm = float(np.linalg.norm(normal))
        if norm == 0:
            continue
        normal = normal / norm
        normals[a] += normal
        normals[b] += normal
        normals[c] += normal
    lengths = np.linalg.norm(normals, axis=1)
    zero = lengths == 0
    lengths[zero] = 1.0
    normals = normals / lengths[:, np.newaxis]
    normals[zero] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.asarray(normals, dtype=np.float64)


def _validate_refinement_inputs(
    *,
    smooth_iterations: int,
    smoothing_method: str,
    taubin_pass_band: float,
    normal_orientation: str,
) -> None:
    if smooth_iterations < 0:
        raise MeshGenError("smooth_iterations must be non-negative")
    if smoothing_method not in {"laplacian", "taubin", "none"}:
        raise MeshGenError("smoothing_method must be one of: laplacian, none, taubin")
    if smooth_iterations > 0 and smoothing_method == "none":
        raise MeshGenError("smoothing_method 'none' requires smooth_iterations=0")
    if taubin_pass_band <= 0:
        raise MeshGenError("taubin_pass_band must be positive")
    if normal_orientation not in {"outward", "inward"}:
        raise MeshGenError("normal_orientation must be one of: inward, outward")


def _top_opening(
    graph: CenterlineGraph,
    *,
    superior_axis: int,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    if superior_axis not in (0, 1, 2):
        raise MeshGenError("superior_axis must be 0, 1, or 2")
    node_by_id = {node.node_id: node for node in graph.nodes}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)

    leaves = [
        node
        for node in graph.nodes
        if len(adjacency.get(node.node_id, set())) == 1
    ]
    accessible_leaves = [node for node in leaves if node.accessible]
    candidates = accessible_leaves if accessible_leaves else leaves
    if not candidates:
        return None
    node = max(candidates, key=lambda item: (item.point_mm[superior_axis], item.radius_mm))
    neighbors = sorted(adjacency.get(node.node_id, set()))
    if len(neighbors) != 1:
        return None
    neighbor = node_by_id[neighbors[0]]
    center = np.asarray(node.point_mm, dtype=np.float64)
    neighbor_point = np.asarray(neighbor.point_mm, dtype=np.float64)
    direction = center - neighbor_point
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        return None
    return center, direction / norm, float(node.radius_mm)


def _marching_cubes(mask: AirwayMask) -> tuple[FloatArray, IntArray, str]:
    padded = np.pad(mask.data.astype(np.float32), pad_width=1, mode="constant", constant_values=0)
    marching_cubes = _optional_callable("skimage.measure", "marching_cubes")
    if marching_cubes is not None:
        vertices_any: Any
        faces_any: Any
        vertices_any, faces_any, _, _ = marching_cubes(padded, level=0.5, spacing=(1.0, 1.0, 1.0))
        voxel_vertices = np.asarray(vertices_any, dtype=np.float64) - 1.0
        vertices = _apply_affine(mask.affine, voxel_vertices)
        faces = np.asarray(faces_any, dtype=np.int64)
        return vertices, faces, "skimage-marching-cubes"
    vertices, faces = _voxel_surface_mesh(mask)
    return vertices, faces, "voxel-surface-fallback"


def _voxel_surface_mesh(mask: AirwayMask) -> tuple[FloatArray, IntArray]:
    vertex_indices: dict[tuple[int, int, int], int] = {}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    data = mask.data
    directions = (
        ((-1, 0, 0), ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
        ((1, 0, 0), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
        ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
        ((0, 1, 0), ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1))),
        ((0, 0, -1), ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1))),
        ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
    )
    for i, j, k in np.argwhere(data):
        voxel = (int(i), int(j), int(k))
        for direction, corners in directions:
            neighbor = (voxel[0] + direction[0], voxel[1] + direction[1], voxel[2] + direction[2])
            if _inside(neighbor, data.shape) and data[neighbor]:
                continue
            quad: list[int] = []
            for corner in corners:
                key = (2 * voxel[0] + corner[0], 2 * voxel[1] + corner[1], 2 * voxel[2] + corner[2])
                if key not in vertex_indices:
                    vertex_indices[key] = len(vertices)
                    vertices.append((key[0] / 2.0, key[1] / 2.0, key[2] / 2.0))
                quad.append(vertex_indices[key])
            faces.append((quad[0], quad[1], quad[2]))
            faces.append((quad[0], quad[2], quad[3]))
    if not vertices or not faces:
        raise MeshGenError("voxel surface fallback produced an empty mesh")
    return _apply_affine(mask.affine, np.asarray(vertices, dtype=np.float64)), np.asarray(faces, dtype=np.int64)


def _inside(index: tuple[int, int, int], shape: tuple[int, int, int]) -> bool:
    return 0 <= index[0] < shape[0] and 0 <= index[1] < shape[1] and 0 <= index[2] < shape[2]


def _apply_affine(affine: FloatArray, coordinates: FloatArray) -> FloatArray:
    homogeneous = np.ones((coordinates.shape[0], 4), dtype=np.float64)
    homogeneous[:, :3] = coordinates
    transformed = homogeneous @ affine.T
    return np.asarray(transformed[:, :3], dtype=np.float64)


def _optional_callable(module_name: str, attribute_name: str) -> Any | None:
    try:
        module = import_module(module_name)
    except ImportError:
        return None
    value: Any = getattr(module, attribute_name, None)
    return value if callable(value) else None
