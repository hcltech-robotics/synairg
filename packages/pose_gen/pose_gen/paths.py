"""Bronchoscope camera path generation from airway centerlines."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mesh_gen.schemas import CenterlineGraph, CenterlineNode
from synairg_core.volume_manifest import JsonValue


@dataclass(frozen=True)
class BronchoscopeProfile:
    """Optical and mechanical profile used to sample camera poses."""

    name: str = "olympus-bf-mp190f"
    field_of_view_deg: float = 90.0
    field_of_view_axis: str = "diagonal"
    image_width_px: int = 640
    image_height_px: int = 480
    depth_of_field_mm: tuple[float, float] = (2.0, 50.0)
    distal_outer_diameter_mm: float = 3.0
    lumen_clearance_mm: float = 0.2
    working_length_mm: float = 600.0
    max_up_bend_deg: float = 210.0
    max_down_bend_deg: float = 130.0
    max_axial_rotation_deg: float = 120.0
    nominal_speed_mm_s: float = 12.0
    min_speed_mm_s: float = 4.0
    max_speed_mm_s: float = 18.0
    frame_rate_fps: float = 10.0

    @property
    def intrinsics(self) -> dict[str, JsonValue]:
        diagonal = math.hypot(self.image_width_px, self.image_height_px)
        focal = diagonal / (2.0 * math.tan(math.radians(self.field_of_view_deg) / 2.0))
        return {
            "model": "pinhole",
            "width_px": self.image_width_px,
            "height_px": self.image_height_px,
            "fx_px": focal,
            "fy_px": focal,
            "cx_px": self.image_width_px / 2.0,
            "cy_px": self.image_height_px / 2.0,
            "field_of_view_deg": self.field_of_view_deg,
            "field_of_view_axis": self.field_of_view_axis,
        }

    @property
    def minimum_lumen_diameter_mm(self) -> float:
        return self.distal_outer_diameter_mm + self.lumen_clearance_mm

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "field_of_view_deg": self.field_of_view_deg,
            "field_of_view_axis": self.field_of_view_axis,
            "depth_of_field_mm": list(self.depth_of_field_mm),
            "distal_outer_diameter_mm": self.distal_outer_diameter_mm,
            "lumen_clearance_mm": self.lumen_clearance_mm,
            "minimum_lumen_diameter_mm": self.minimum_lumen_diameter_mm,
            "working_length_mm": self.working_length_mm,
            "max_up_bend_deg": self.max_up_bend_deg,
            "max_down_bend_deg": self.max_down_bend_deg,
            "max_axial_rotation_deg": self.max_axial_rotation_deg,
            "nominal_speed_mm_s": self.nominal_speed_mm_s,
            "min_speed_mm_s": self.min_speed_mm_s,
            "max_speed_mm_s": self.max_speed_mm_s,
            "frame_rate_fps": self.frame_rate_fps,
            "intrinsics": self.intrinsics,
        }


@dataclass(frozen=True)
class TrajectoryState:
    """One centerline-anchored pose sample before camera frame assembly."""

    position_mm: np.ndarray
    centerline_position_mm: np.ndarray
    radius_mm: float
    insertion_depth_mm: float
    speed_mm_s: float
    motion_phase: str
    route_index: int
    forward_hint: np.ndarray


def generate_scope_paths(
    *,
    centerline_graph_path: Path,
    output_path: Path,
    path_count: int = 3,
    frustum_every_frames: int = 8,
    profile: BronchoscopeProfile | None = None,
    include_procedure: bool = False,
) -> dict[str, JsonValue]:
    """Generate bronchoscope camera paths from a centerline graph and write JSON."""
    if path_count < 1:
        raise ValueError("path_count must be positive")
    if frustum_every_frames < 1:
        raise ValueError("frustum_every_frames must be positive")
    scope = profile or BronchoscopeProfile()
    graph = CenterlineGraph.read_json(centerline_graph_path)
    routes = _select_routes(graph, path_count=path_count, profile=scope)
    paths = [
        _sample_route(graph, route, index=index, profile=scope, frustum_every_frames=frustum_every_frames)
        for index, route in enumerate(routes, start=1)
    ]
    if include_procedure and len(routes) >= 2:
        paths.append(
            _sample_multi_target_procedure(
                graph,
                routes,
                profile=scope,
                frustum_every_frames=frustum_every_frames,
            )
        )
    payload: dict[str, JsonValue] = {
        "schema_version": "1.0",
        "source_centerline_graph": str(centerline_graph_path),
        "mesh_case_id": graph.case_id,
        "bronchoscope": scope.to_dict(),
        "sampling": {
            "frustum_every_frames": frustum_every_frames,
            "mechanics": "single-plane distal bending plus insertion-tube axial rotation",
            "entry_selection": "superior accessible terminal centerline node",
            "diameter_constraint": (
                "camera paths are restricted to the connected centerline subgraph where local lumen diameter "
                "exceeds bronchoscope distal outer diameter plus clearance"
            ),
            "minimum_lumen_diameter_mm": scope.minimum_lumen_diameter_mm,
            "trajectory_model": (
                "BronchoPose-inspired arc-length traversal with continuous neighboring pose variation, "
                "centerline anchoring, retraction, and multi-target branch navigation"
            ),
        },
        "paths": paths,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_scope_path_review_png(
    *,
    mesh_path: Path,
    paths_json_path: Path,
    output_path: Path,
    image_size: tuple[int, int] = (1800, 1300),
) -> None:
    """Render airway mesh, camera path lines, and sparse frustums to a PNG."""
    try:
        import pyvista as pv  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("pyvista is required to render scope path reviews") from exc

    payload = json.loads(paths_json_path.read_text(encoding="utf-8"))
    mesh = pv.read(str(mesh_path))
    plotter = pv.Plotter(off_screen=True, window_size=image_size)
    plotter.set_background("#071012", top="#10191c")
    plotter.add_mesh(mesh, color="#c8f4ee", opacity=0.18, smooth_shading=True)
    colors = ("#ffb84d", "#56d6ff", "#e974ff", "#74f29b", "#ffffff")
    for path_index, path_payload in enumerate(payload["paths"]):
        color = "#ffffff" if path_payload.get("path_type") == "multi-target-procedure" else colors[path_index % 4]
        frames = path_payload["frames"]
        points = np.asarray([frame["position_mm"] for frame in frames], dtype=np.float64)
        if points.shape[0] < 2:
            continue
        plotter.add_mesh(_polyline(points, pv), color=color, line_width=6 if color == "#ffffff" else 4)
        frustum_frames = [frame for frame in frames if frame.get("show_frustum")]
        for frame in frustum_frames:
            plotter.add_mesh(
                _frustum_polydata(frame, payload["bronchoscope"]["intrinsics"], pv),
                color=color,
                line_width=2,
            )

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1.0)
    plotter.add_light(
        pv.Light(
            position=tuple(center + np.array([extent, -1.5 * extent, 1.3 * extent])),
            focal_point=tuple(center),
            intensity=0.75,
        )
    )
    plotter.camera_position = [
        tuple(center + np.array([0.9 * extent, -1.9 * extent, 1.15 * extent])),
        tuple(center),
        (0, 0, 1),
    ]
    plotter.camera.zoom(1.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path), transparent_background=False)
    plotter.close()


def _sample_route(
    graph: CenterlineGraph,
    route: list[str],
    *,
    index: int,
    profile: BronchoscopeProfile,
    frustum_every_frames: int,
) -> dict[str, JsonValue]:
    nodes = {node.node_id: node for node in graph.nodes}
    depth_by_node = _route_depth_map(route, nodes)
    states = _states_for_node_sequence(
        route,
        nodes,
        depth_by_node,
        profile=profile,
        phase="insertion",
        route_index=index,
    )
    frames = _states_to_frames(states, profile=profile, frustum_every_frames=frustum_every_frames)
    return _path_payload(
        path_id=f"scope-path-{index:02d}",
        path_type="single-target-insertion",
        route=route,
        nodes=nodes,
        profile=profile,
        states=states,
        frames=frames,
    )


def _sample_multi_target_procedure(
    graph: CenterlineGraph,
    routes: list[list[str]],
    *,
    profile: BronchoscopeProfile,
    frustum_every_frames: int,
) -> dict[str, JsonValue]:
    nodes = {node.node_id: node for node in graph.nodes}
    states: list[TrajectoryState] = []
    phases: list[dict[str, JsonValue]] = []
    current_route = routes[0]
    depth_by_node = _route_depth_map(current_route, nodes)
    insertion = _states_for_node_sequence(
        current_route,
        nodes,
        depth_by_node,
        profile=profile,
        phase="insertion",
        route_index=1,
    )
    states.extend(insertion)
    phases.append(_phase_summary("insertion", current_route, nodes, 1, insertion))

    for route_index, next_route in enumerate(routes[1:], start=2):
        common = _common_prefix_length(current_route, next_route)
        current_depth_by_node = _route_depth_map(current_route, nodes)
        next_depth_by_node = _route_depth_map(next_route, nodes)

        retract_nodes = list(reversed(current_route[common - 1 :]))
        if len(retract_nodes) >= 2:
            retraction = _states_for_node_sequence(
                retract_nodes,
                nodes,
                current_depth_by_node,
                profile=profile,
                phase="retraction",
                route_index=route_index - 1,
                forward_sign=-1.0,
            )
            states.extend(retraction)
            phases.append(_phase_summary("retraction", retract_nodes, nodes, route_index - 1, retraction))

        insert_nodes = next_route[common - 1 :]
        if len(insert_nodes) >= 2:
            insertion = _states_for_node_sequence(
                insert_nodes,
                nodes,
                next_depth_by_node,
                profile=profile,
                phase="insertion",
                route_index=route_index,
            )
            branch_states = _branch_navigation_states(
                states[-1],
                insertion[0].forward_hint,
                profile,
                route_index=route_index,
            )
            states.extend(branch_states)
            phases.extend(
                _stationary_phase_summary(phase, insert_nodes[0], nodes, route_index, branch_states)
                for phase in ("branch_pause", "scope_rotate", "bend_curl")
            )
            states.extend(insertion)
            phases.append(_phase_summary("insertion", insert_nodes, nodes, route_index, insertion))
        current_route = next_route

    final_depth_by_node = _route_depth_map(current_route, nodes)
    final_retract_nodes = list(reversed(current_route))
    final_retraction = _states_for_node_sequence(
        final_retract_nodes,
        nodes,
        final_depth_by_node,
        profile=profile,
        phase="final_retraction",
        route_index=len(routes),
        forward_sign=-1.0,
    )
    states.extend(final_retraction)
    phases.append(_phase_summary("final_retraction", final_retract_nodes, nodes, len(routes), final_retraction))

    frames = _states_to_frames(states, profile=profile, frustum_every_frames=frustum_every_frames)
    payload = _path_payload(
        path_id="scope-procedure-01",
        path_type="multi-target-procedure",
        route=current_route,
        nodes=nodes,
        profile=profile,
        states=states,
        frames=frames,
    )
    payload["procedure_targets"] = [
        {
            "route_index": index,
            "target_node_id": route[-1],
            "target_branch_id": nodes[route[-1]].branch_id,
            "target_point_mm": _round_vector(np.asarray(nodes[route[-1]].point_mm, dtype=np.float64)),
            "minimum_lumen_diameter_mm": round(_route_minimum_lumen_diameter(route, nodes), 3),
            "minimum_scope_clearance_mm": round(
                _route_minimum_lumen_diameter(route, nodes) - profile.distal_outer_diameter_mm,
                3,
            ),
        }
        for index, route in enumerate(routes, start=1)
    ]
    payload["procedure_phases"] = phases
    return payload


def _path_payload(
    *,
    path_id: str,
    path_type: str,
    route: list[str],
    nodes: dict[str, CenterlineNode],
    profile: BronchoscopeProfile,
    states: list[TrajectoryState],
    frames: list[dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    duration = frames[-1]["time_s"] if frames else 0.0
    path_length = _state_path_length(states)
    return {
        "path_id": path_id,
        "path_type": path_type,
        "entry_node_id": route[0],
        "entry_branch_id": nodes[route[0]].branch_id,
        "target_node_id": route[-1],
        "target_branch_id": nodes[route[-1]].branch_id,
        "node_count": len(route),
        "minimum_lumen_diameter_mm": round(_route_minimum_lumen_diameter(route, nodes), 3),
        "minimum_scope_clearance_mm": round(
            _route_minimum_lumen_diameter(route, nodes) - profile.distal_outer_diameter_mm,
            3,
        ),
        "length_mm": round(path_length, 3),
        "duration_s": round(float(duration), 3),
        "frame_count": len(frames),
        "mean_speed_mm_s": round(path_length / max(float(duration), 1.0e-6), 3),
        "frames": frames,
    }


def _select_routes(graph: CenterlineGraph, *, path_count: int, profile: BronchoscopeProfile) -> list[list[str]]:
    nodes = {node.node_id: node for node in graph.nodes}
    adjacency = _adjacency(graph)
    base_entry_id = _entry_node_id(graph, adjacency)
    passable_node_ids = {
        node.node_id
        for node in graph.nodes
        if _is_scope_passable(node, profile=profile)
    }
    entry_id = _nearest_passable_entry_id(base_entry_id, adjacency, passable_node_ids)
    if entry_id is None:
        raise ValueError(
            "no centerline node has lumen diameter greater than the bronchoscope distal outer diameter "
            f"plus clearance ({profile.minimum_lumen_diameter_mm:.2f} mm)"
        )
    entry_node = nodes[entry_id]
    parent = _parent_map(graph, entry_id=entry_id, adjacency=adjacency, allowed_node_ids=passable_node_ids)
    parent_ids = set(parent.values())
    route_candidates: list[tuple[CenterlineNode, list[str], float, str]] = []
    reachable_passable_ids = {entry_id, *parent}
    for node_id in reachable_passable_ids:
        node = nodes[node_id]
        if node.node_id == entry_id or node.node_id in parent_ids:
            continue
        route = _route_from_parent_map(parent, entry_id=entry_id, target_id=node.node_id)
        if len(route) < 2 or not all(node_id in nodes for node_id in route):
            continue
        if not all(route_node_id in passable_node_ids for route_node_id in route):
            continue
        length_mm = _route_length(route, nodes)
        if length_mm <= profile.working_length_mm:
            route_candidates.append((node, route, length_mm, _target_sector(entry_node, node)))
    route_candidates.sort(key=lambda item: item[2], reverse=True)
    if not route_candidates:
        return []

    selected: list[tuple[CenterlineNode, list[str], float, str]] = [route_candidates[0]]
    while len(selected) < path_count:
        remaining = [candidate for candidate in route_candidates if candidate not in selected]
        if not remaining:
            break
        selected_sectors = {candidate[3] for candidate in selected}
        best = max(
            remaining,
            key=lambda candidate: _route_diversity_score(candidate, selected, selected_sectors),
        )
        selected.append(best)
    return [route for _, route, _, _ in selected]


def _is_scope_passable(node: CenterlineNode, *, profile: BronchoscopeProfile) -> bool:
    return node.accessible and node.radius_mm * 2.0 > profile.minimum_lumen_diameter_mm


def _nearest_passable_entry_id(
    base_entry_id: str,
    adjacency: dict[str, list[str]],
    passable_node_ids: set[str],
) -> str | None:
    queue: deque[str] = deque([base_entry_id])
    seen = {base_entry_id}
    while queue:
        current = queue.popleft()
        if current in passable_node_ids:
            return current
        for neighbor in adjacency[current]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return None


def _route_diversity_score(
    candidate: tuple[CenterlineNode, list[str], float, str],
    selected: list[tuple[CenterlineNode, list[str], float, str]],
    selected_sectors: set[str],
) -> float:
    node, _, length_mm, sector = candidate
    endpoint = np.asarray(node.point_mm, dtype=np.float64)
    min_target_distance = min(
        float(np.linalg.norm(endpoint - np.asarray(selected_node.point_mm, dtype=np.float64)))
        for selected_node, _, _, _ in selected
    )
    sector_bonus = 80.0 if sector not in selected_sectors else 0.0
    return min_target_distance * 1.7 + length_mm * 0.25 + sector_bonus


def _target_sector(entry_node: CenterlineNode, target_node: CenterlineNode) -> str:
    delta = np.asarray(target_node.point_mm, dtype=np.float64) - np.asarray(entry_node.point_mm, dtype=np.float64)
    lateral = "right" if delta[0] >= 0 else "left"
    vertical = "upper" if delta[1] >= 0 else "lower"
    return f"{lateral}-{vertical}"


def _adjacency(graph: CenterlineGraph) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        adjacency[edge.source].append(edge.target)
        adjacency[edge.target].append(edge.source)
    return adjacency


def _entry_node_id(graph: CenterlineGraph, adjacency: dict[str, list[str]], *, superior_axis: int = 2) -> str:
    leaves = [
        node
        for node in graph.nodes
        if node.accessible and len(adjacency.get(node.node_id, [])) <= 1
    ]
    candidates = leaves or [node for node in graph.nodes if node.accessible] or list(graph.nodes)
    entry = max(candidates, key=lambda node: (node.point_mm[superior_axis], node.radius_mm))
    return entry.node_id


def _parent_map(
    graph: CenterlineGraph,
    *,
    entry_id: str,
    adjacency: dict[str, list[str]],
    allowed_node_ids: set[str] | None = None,
) -> dict[str, str]:
    parent: dict[str, str] = {}
    queue: deque[str] = deque([entry_id])
    seen = {entry_id}
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in seen:
                continue
            if allowed_node_ids is not None and neighbor not in allowed_node_ids:
                continue
            seen.add(neighbor)
            parent[neighbor] = current
            queue.append(neighbor)
    return parent


def _route_minimum_lumen_diameter(route: list[str], nodes: dict[str, CenterlineNode]) -> float:
    return min(nodes[node_id].radius_mm * 2.0 for node_id in route)


def _route_from_parent_map(parent: dict[str, str], *, entry_id: str, target_id: str) -> list[str]:
    route_ids = [target_id]
    current = target_id
    while current != entry_id:
        try:
            current = parent[current]
        except KeyError:
            return []
        route_ids.append(current)
    route_ids.reverse()
    return route_ids


def _route_depth_map(route: list[str], nodes: dict[str, CenterlineNode]) -> dict[str, float]:
    depths = {route[0]: 0.0}
    current_depth = 0.0
    for first, second in zip(route[:-1], route[1:], strict=True):
        current_depth += _distance(nodes[first].point_mm, nodes[second].point_mm)
        depths[second] = current_depth
    return depths


def _route_length(route: list[str], nodes: dict[str, CenterlineNode]) -> float:
    return sum(
        _distance(nodes[first].point_mm, nodes[second].point_mm)
        for first, second in zip(route[:-1], route[1:], strict=True)
    )


def _common_prefix_length(first: list[str], second: list[str]) -> int:
    count = 0
    for first_id, second_id in zip(first, second, strict=False):
        if first_id != second_id:
            break
        count += 1
    return max(count, 1)


def _states_for_node_sequence(
    route: list[str],
    nodes: dict[str, CenterlineNode],
    depth_by_node: dict[str, float],
    *,
    profile: BronchoscopeProfile,
    phase: str,
    route_index: int,
    forward_sign: float = 1.0,
) -> list[TrajectoryState]:
    raw_points = np.asarray([nodes[node_id].point_mm for node_id in route], dtype=np.float64)
    raw_radii = np.asarray([nodes[node_id].radius_mm for node_id in route], dtype=np.float64)
    raw_depths = np.asarray([depth_by_node[node_id] for node_id in route], dtype=np.float64)
    points, radii, depths = _smoothed_centerline(raw_points, raw_radii, raw_depths)
    distances = _arc_lengths(points)
    length_mm = float(distances[-1])
    states: list[TrajectoryState] = []
    distance_mm = 0.0
    frame_index = 0
    while distance_mm <= length_mm:
        state = _state_at_distance(
            points,
            radii,
            depths,
            distances,
            distance_mm,
            length_mm=length_mm,
            frame_index=frame_index,
            profile=profile,
            phase=phase,
            route_index=route_index,
            forward_sign=forward_sign,
        )
        states.append(state)
        distance_mm += max(state.speed_mm_s / profile.frame_rate_fps, 0.35)
        frame_index += 1
    if length_mm > 0 and float(np.linalg.norm(states[-1].centerline_position_mm - points[-1])) > 0.5:
        states.append(
            _state_at_distance(
                points,
                radii,
                depths,
                distances,
                length_mm,
                length_mm=length_mm,
                frame_index=frame_index,
                profile=profile,
                phase=phase,
                route_index=route_index,
                forward_sign=forward_sign,
            )
        )
    return states


def _state_at_distance(
    points: np.ndarray,
    radii: np.ndarray,
    depths: np.ndarray,
    distances: np.ndarray,
    distance_mm: float,
    *,
    length_mm: float,
    frame_index: int,
    profile: BronchoscopeProfile,
    phase: str,
    route_index: int,
    forward_sign: float = 1.0,
) -> TrajectoryState:
    center, radius, depth, tangent = _sample_smooth_path(points, radii, depths, distances, distance_mm)
    speed = _procedure_speed(distance_mm, length_mm, radius, phase, frame_index, profile)
    return TrajectoryState(
        position_mm=center,
        centerline_position_mm=center,
        radius_mm=radius,
        insertion_depth_mm=depth,
        speed_mm_s=speed,
        motion_phase=phase,
        route_index=route_index,
        forward_hint=_unit(tangent * forward_sign),
    )


def _smoothed_centerline(
    raw_points: np.ndarray,
    raw_radii: np.ndarray,
    raw_depths: np.ndarray,
    *,
    step_mm: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if raw_points.shape[0] < 2:
        return raw_points, raw_radii, raw_depths
    raw_distances = _arc_lengths(raw_points)
    length_mm = float(raw_distances[-1])
    sample_distances = np.arange(0.0, length_mm, step_mm, dtype=np.float64)
    if sample_distances.size == 0 or sample_distances[-1] < length_mm:
        sample_distances = np.append(sample_distances, length_mm)
    points = []
    radii = []
    depths = []
    for distance_mm in sample_distances:
        point, radius, depth, _ = _sample_smooth_path(raw_points, raw_radii, raw_depths, raw_distances, distance_mm)
        points.append(point)
        radii.append(radius)
        depths.append(depth)
    smoothed_points = np.asarray(points, dtype=np.float64)
    if smoothed_points.shape[0] >= 7:
        for _ in range(2):
            smoothed_points = _moving_average_points(smoothed_points, window=7)
        smoothed_points[0] = raw_points[0]
        smoothed_points[-1] = raw_points[-1]
    return smoothed_points, np.asarray(radii, dtype=np.float64), np.asarray(depths, dtype=np.float64)


def _moving_average_points(points: np.ndarray, *, window: int) -> np.ndarray:
    radius = window // 2
    smoothed = points.copy()
    for index in range(1, points.shape[0] - 1):
        lower = max(0, index - radius)
        upper = min(points.shape[0], index + radius + 1)
        smoothed[index] = points[lower:upper].mean(axis=0)
    return smoothed


def _sample_smooth_path(
    points: np.ndarray,
    radii: np.ndarray,
    depths: np.ndarray,
    distances: np.ndarray,
    distance_mm: float,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    distance_mm = float(np.clip(distance_mm, 0.0, distances[-1]))
    index = int(np.searchsorted(distances, distance_mm, side="right") - 1)
    index = min(max(index, 0), len(points) - 2)
    span = max(float(distances[index + 1] - distances[index]), 1.0e-6)
    alpha = (distance_mm - float(distances[index])) / span
    point = (1.0 - alpha) * points[index] + alpha * points[index + 1]
    radius = float((1.0 - alpha) * radii[index] + alpha * radii[index + 1])
    depth = float((1.0 - alpha) * depths[index] + alpha * depths[index + 1])
    lookahead = min(distances[-1], distance_mm + 8.0)
    lookbehind = max(0.0, distance_mm - 4.0)
    if lookahead <= lookbehind + 1.0e-6:
        tangent = _unit(points[index + 1] - points[index])
    else:
        ahead = _sample_position_only(points, distances, lookahead)
        behind = _sample_position_only(points, distances, lookbehind)
        tangent = _unit(ahead - behind)
    return point, radius, depth, tangent


def _sample_position_only(points: np.ndarray, distances: np.ndarray, distance_mm: float) -> np.ndarray:
    distance_mm = float(np.clip(distance_mm, 0.0, distances[-1]))
    index = int(np.searchsorted(distances, distance_mm, side="right") - 1)
    index = min(max(index, 0), len(points) - 2)
    span = max(float(distances[index + 1] - distances[index]), 1.0e-6)
    alpha = (distance_mm - float(distances[index])) / span
    return (1.0 - alpha) * points[index] + alpha * points[index + 1]


def _procedure_speed(
    distance_mm: float,
    length_mm: float,
    radius_mm: float,
    phase: str,
    frame_index: int,
    profile: BronchoscopeProfile,
) -> float:
    if phase == "redirect":
        return 0.0
    ramp = min(
        1.0,
        max(0.0, distance_mm / 28.0),
        max(0.0, (length_mm - distance_mm) / 28.0),
    )
    approach_multiplier = 0.42 + 0.58 * _smoothstep(ramp)
    phase_multiplier = 0.78 if "retraction" in phase else 1.0
    radius_slowdown = float(np.clip(radius_mm / 5.0, 0.48, 1.0))
    operator_variation = 0.97 + 0.03 * math.sin(frame_index * 0.045)
    speed = profile.nominal_speed_mm_s * phase_multiplier * approach_multiplier * radius_slowdown * operator_variation
    return float(np.clip(speed, profile.min_speed_mm_s, profile.max_speed_mm_s))


def _branch_navigation_states(
    anchor: TrajectoryState,
    target_forward: np.ndarray,
    profile: BronchoscopeProfile,
    *,
    route_index: int,
) -> list[TrajectoryState]:
    states: list[TrajectoryState] = []
    states.extend(_hold_states(anchor, profile, "branch_pause", route_index, duration_s=0.6))
    states.extend(_hold_states(states[-1], profile, "scope_rotate", route_index, duration_s=0.8))
    states.extend(
        _turn_states(
            states[-1],
            target_forward,
            profile,
            "bend_curl",
            route_index,
            duration_s=1.1,
        )
    )
    return states


def _hold_states(
    anchor: TrajectoryState,
    profile: BronchoscopeProfile,
    phase: str,
    route_index: int,
    *,
    duration_s: float,
) -> list[TrajectoryState]:
    count = max(1, int(round(duration_s * profile.frame_rate_fps)))
    return [
        TrajectoryState(
            position_mm=anchor.position_mm,
            centerline_position_mm=anchor.centerline_position_mm,
            radius_mm=anchor.radius_mm,
            insertion_depth_mm=anchor.insertion_depth_mm,
            speed_mm_s=0.0,
            motion_phase=phase,
            route_index=route_index,
            forward_hint=anchor.forward_hint,
        )
        for _ in range(count)
    ]


def _turn_states(
    anchor: TrajectoryState,
    target_forward: np.ndarray,
    profile: BronchoscopeProfile,
    phase: str,
    route_index: int,
    *,
    duration_s: float = 0.8,
) -> list[TrajectoryState]:
    count = max(2, int(round(duration_s * profile.frame_rate_fps)))
    states: list[TrajectoryState] = []
    start_forward = anchor.forward_hint
    for index in range(1, count + 1):
        alpha = _smoothstep(index / count)
        forward = _unit((1.0 - alpha) * start_forward + alpha * target_forward)
        states.append(
            TrajectoryState(
                position_mm=anchor.position_mm,
                centerline_position_mm=anchor.centerline_position_mm,
                radius_mm=anchor.radius_mm,
                insertion_depth_mm=anchor.insertion_depth_mm,
                speed_mm_s=0.0,
                motion_phase=phase,
                route_index=route_index,
                forward_hint=forward,
            )
        )
    return states


def _states_to_frames(
    states: list[TrajectoryState],
    *,
    profile: BronchoscopeProfile,
    frustum_every_frames: int,
) -> list[dict[str, JsonValue]]:
    frames: list[dict[str, JsonValue]] = []
    previous_forward: np.ndarray | None = None
    previous_up: np.ndarray | None = None
    axial_rotation_deg = 0.0
    bend_deg_smoothed = 0.0
    dt = 1.0 / profile.frame_rate_fps
    for frame_index, state in enumerate(states):
        desired_forward = _unit(state.forward_hint)
        if previous_forward is None:
            forward = desired_forward
        else:
            forward = _rotate_vector_towards(previous_forward, desired_forward, math.radians(7.0))
        up_seed = _transport_up(forward, previous_up)
        target_roll = _target_axial_rotation_deg(state)
        max_roll_step = _max_roll_rate_deg_s(state.motion_phase) * dt
        axial_rotation_deg += float(np.clip(target_roll - axial_rotation_deg, -max_roll_step, max_roll_step))
        axial_rotation_deg = float(
            np.clip(
                axial_rotation_deg,
                -min(profile.max_axial_rotation_deg, 0.5),
                min(profile.max_axial_rotation_deg, 0.5),
            )
        )
        up = _rotate_around_axis(up_seed, forward, math.radians(axial_rotation_deg))
        right = _unit(np.cross(forward, up))
        up = _unit(np.cross(right, forward))

        future_index = min(len(states) - 1, frame_index + 6)
        future_forward = _unit(states[future_index].forward_hint)
        raw_bend_deg = math.degrees(math.acos(float(np.clip(np.dot(forward, future_forward), -1.0, 1.0))))
        signed_bend = raw_bend_deg if float(np.dot(np.cross(forward, future_forward), right)) >= 0 else -raw_bend_deg
        signed_bend = float(np.clip(signed_bend, -profile.max_down_bend_deg, profile.max_up_bend_deg))
        bend_deg_smoothed = 0.82 * bend_deg_smoothed + 0.18 * signed_bend
        anchor_offset = float(np.linalg.norm(state.position_mm - state.centerline_position_mm))
        frames.append(
            {
                "frame_index": frame_index,
                "time_s": round(frame_index * dt, 4),
                "position_mm": _round_vector(state.position_mm),
                "centerline_position_mm": _round_vector(state.centerline_position_mm),
                "centerline_anchor_offset_mm": round(anchor_offset, 4),
                "forward": _round_vector(forward),
                "up": _round_vector(up),
                "right": _round_vector(right),
                "quaternion_wxyz": _round_vector(_rotation_quaternion(right, up, forward), count=4),
                "insertion_depth_mm": round(state.insertion_depth_mm, 3),
                "speed_mm_s": round(state.speed_mm_s, 3),
                "bend_deg": round(bend_deg_smoothed, 3),
                "axial_rotation_deg": round(axial_rotation_deg, 3),
                "radius_mm": round(float(state.radius_mm), 3),
                "motion_phase": state.motion_phase,
                "route_index": state.route_index,
                "show_frustum": frame_index % frustum_every_frames == 0,
            }
        )
        previous_forward = forward
        previous_up = up
    return frames


def _phase_summary(
    phase: str,
    route: list[str],
    nodes: dict[str, CenterlineNode],
    route_index: int,
    states: list[TrajectoryState],
) -> dict[str, JsonValue]:
    return {
        "phase": phase,
        "route_index": route_index,
        "start_node_id": route[0],
        "end_node_id": route[-1],
        "start_branch_id": nodes[route[0]].branch_id,
        "end_branch_id": nodes[route[-1]].branch_id,
        "frame_count": len(states),
        "distance_mm": round(_state_path_length(states), 3),
    }


def _stationary_phase_summary(
    phase: str,
    node_id: str,
    nodes: dict[str, CenterlineNode],
    route_index: int,
    states: list[TrajectoryState],
) -> dict[str, JsonValue]:
    phase_states = [state for state in states if state.motion_phase == phase]
    return {
        "phase": phase,
        "route_index": route_index,
        "start_node_id": node_id,
        "end_node_id": node_id,
        "start_branch_id": nodes[node_id].branch_id,
        "end_branch_id": nodes[node_id].branch_id,
        "frame_count": len(phase_states),
        "distance_mm": 0.0,
    }


def _target_axial_rotation_deg(state: TrajectoryState) -> float:
    if state.motion_phase in {"scope_rotate", "bend_curl"}:
        direction = 1.0 if state.route_index % 2 else -1.0
        return direction * 0.5
    return 0.0


def _max_roll_rate_deg_s(phase: str) -> float:
    if phase == "scope_rotate":
        return 1.2
    if phase == "bend_curl":
        return 0.8
    return 0.6


def _state_path_length(states: list[TrajectoryState]) -> float:
    if len(states) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(second.position_mm - first.position_mm))
        for first, second in zip(states[:-1], states[1:], strict=True)
    )


def _arc_lengths(points: np.ndarray) -> np.ndarray:
    lengths = [0.0]
    for first, second in zip(points[:-1], points[1:], strict=True):
        lengths.append(lengths[-1] + float(np.linalg.norm(second - first)))
    return np.asarray(lengths, dtype=np.float64)


def _transport_up(tangent: np.ndarray, previous_up: np.ndarray | None) -> np.ndarray:
    if previous_up is None:
        candidate = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(candidate, tangent))) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        candidate = previous_up
    candidate = candidate - tangent * float(np.dot(candidate, tangent))
    return _unit(candidate)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(vector / norm, dtype=np.float64)


def _rotate_vector_towards(current: np.ndarray, target: np.ndarray, max_angle_rad: float) -> np.ndarray:
    current = _unit(current)
    target = _unit(target)
    dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
    angle = math.acos(dot)
    if angle <= max_angle_rad or angle <= 1.0e-9:
        return target
    axis = np.cross(current, target)
    if float(np.linalg.norm(axis)) <= 1.0e-9:
        return target
    return _rotate_around_axis(current, axis, max_angle_rad)


def _rotate_around_axis(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _unit(axis)
    return _unit(
        vector * math.cos(angle_rad)
        + np.cross(axis, vector) * math.sin(angle_rad)
        + axis * float(np.dot(axis, vector)) * (1.0 - math.cos(angle_rad))
    )


def _rotation_quaternion(right: np.ndarray, up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    matrix = np.column_stack((right, up, forward))
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    index = int(np.argmax(np.diag(matrix)))
    if index == 0:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    if index == 1:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ],
            dtype=np.float64,
        )
    scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return np.array(
        [
            (matrix[1, 0] - matrix[0, 1]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
        ],
        dtype=np.float64,
    )


def _polyline(points: np.ndarray, pv) -> object:
    cells = []
    for index in range(points.shape[0] - 1):
        cells.extend([2, index, index + 1])
    return pv.PolyData(points, lines=np.asarray(cells, dtype=np.int64))


def _frustum_polydata(frame: dict[str, JsonValue], intrinsics: dict[str, JsonValue], pv) -> object:
    position = np.asarray(frame["position_mm"], dtype=np.float64)
    forward = np.asarray(frame["forward"], dtype=np.float64)
    up = np.asarray(frame["up"], dtype=np.float64)
    right = np.asarray(frame["right"], dtype=np.float64)
    depth = 8.0
    width = float(intrinsics["width_px"])
    height = float(intrinsics["height_px"])
    fx = float(intrinsics["fx_px"])
    fy = float(intrinsics["fy_px"])
    half_w = depth * width / (2.0 * fx)
    half_h = depth * height / (2.0 * fy)
    center = position + forward * depth
    corners = np.asarray(
        [
            center - right * half_w - up * half_h,
            center + right * half_w - up * half_h,
            center + right * half_w + up * half_h,
            center - right * half_w + up * half_h,
        ],
        dtype=np.float64,
    )
    points = np.vstack((position, corners))
    lines = np.asarray(
        [
            2,
            0,
            1,
            2,
            0,
            2,
            2,
            0,
            3,
            2,
            0,
            4,
            2,
            1,
            2,
            2,
            2,
            3,
            2,
            3,
            4,
            2,
            4,
            1,
        ],
        dtype=np.int64,
    )
    return pv.PolyData(points, lines=lines)


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)))


def _round_vector(vector: np.ndarray, *, count: int = 3) -> list[float]:
    return [round(float(value), 6) for value in vector[:count]]


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)
