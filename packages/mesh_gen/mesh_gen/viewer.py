"""Self-contained WebGL viewer for airway meshes."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mesh_gen.schemas import AirwayMesh, CenterlineGraph


def write_viewer_html(path: Path, mesh: AirwayMesh, graph: CenterlineGraph) -> None:
    """Write an interactive HTML viewer with embedded mesh and centreline data."""
    center, scale = _normalization(mesh.vertices)
    vertices = _normalize_points(mesh.vertices, center=center, scale=scale)
    node_positions = np.asarray([node.point_mm for node in graph.nodes], dtype=np.float64)
    nodes = _normalize_points(node_positions, center=center, scale=scale) if len(graph.nodes) else []
    node_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
    line_vertices: list[list[float]] = []
    accessible_line_vertices: list[list[float]] = []
    for edge in graph.edges:
        source = nodes[node_index[edge.source]]
        target = nodes[node_index[edge.target]]
        line_vertices.extend((source, target))
        if edge.accessible:
            accessible_line_vertices.extend((source, target))
    route_points_mm, route_depth_mm, route_min_radius_mm = _accessible_route(graph)
    route = _normalize_points(np.asarray(route_points_mm, dtype=np.float64), center=center, scale=scale) if route_points_mm else []

    accessible_edges = sum(1 for edge in graph.edges if edge.accessible)
    payload = {
        "vertices": vertices,
        "normals": np.asarray(mesh.vertex_normals, dtype=np.float64).round(6).tolist(),
        "faces": np.asarray(mesh.faces, dtype=np.int64).ravel().tolist(),
        "lines": line_vertices,
        "accessibleLines": accessible_line_vertices,
        "route": route,
        "stats": {
            "vertices": int(mesh.vertices.shape[0]),
            "faces": int(mesh.faces.shape[0]),
            "centerlineNodes": len(graph.nodes),
            "centerlineEdges": len(graph.edges),
            "accessibleEdges": accessible_edges,
            "routeNodes": len(route),
            "routeDepthMm": round(route_depth_mm, 1),
            "routeMinRadiusMm": round(route_min_radius_mm, 1),
            "maxGeneration": max((node.generation for node in graph.nodes if node.accessible), default=0),
            "coveragePct": round(accessible_edges / max(len(graph.edges), 1) * 100),
        },
    }
    path.write_text(_html(json.dumps(payload, separators=(",", ":"))), encoding="utf-8")


def _normalization(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = (minimum + maximum) / 2.0
    scale = float(np.max(maximum - minimum))
    return center, max(scale, 1.0e-6)


def _normalize_points(points: np.ndarray, *, center: np.ndarray, scale: float) -> list[list[float]]:
    normalized = (np.asarray(points, dtype=np.float64) - center) / scale * 2.0
    return normalized.round(6).tolist()


def _accessible_route(graph: CenterlineGraph) -> tuple[list[tuple[float, float, float]], float, float]:
    nodes = {node.node_id: node for node in graph.nodes}
    if graph.root_id not in nodes:
        return [], 0.0, 0.0

    adjacency: dict[str, list[str]] = {node.node_id: [] for node in graph.nodes}
    for edge in graph.edges:
        if not edge.accessible:
            continue
        source = nodes.get(edge.source)
        target = nodes.get(edge.target)
        if source is None or target is None or not source.accessible or not target.accessible:
            continue
        adjacency[edge.source].append(edge.target)
        adjacency[edge.target].append(edge.source)

    parent: dict[str, str | None] = {graph.root_id: None}
    stack = [graph.root_id]
    while stack:
        current = stack.pop()
        for neighbor in adjacency.get(current, []):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            stack.append(neighbor)

    reachable = [node_id for node_id in parent if nodes[node_id].accessible]
    if not reachable:
        root = nodes[graph.root_id]
        return [root.point_mm], root.distance_from_root_mm, root.radius_mm

    target_id = max(reachable, key=lambda node_id: nodes[node_id].distance_from_root_mm)
    route_ids: list[str] = []
    current: str | None = target_id
    while current is not None:
        route_ids.append(current)
        current = parent[current]
    route_ids.reverse()
    return (
        [nodes[node_id].point_mm for node_id in route_ids],
        nodes[target_id].distance_from_root_mm,
        min(nodes[node_id].radius_mm for node_id in route_ids),
    )


def _html(payload_json: str) -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SynAirG Airway Viewer</title>
<style>
:root {
  color-scheme: dark;
  --bg: #050708;
  --line-soft: rgba(160, 190, 184, 0.22);
  --line-hard: rgba(206, 231, 225, 0.42);
  --text: #e7f1ef;
  --dim: #78918c;
  --mesh: #7edfd1;
  --tree: #d9a73e;
  --reach: #65f0a8;
  --warn: #f1c45d;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background:
    linear-gradient(90deg, rgba(126,223,209,0.014) 1px, transparent 1px),
    linear-gradient(0deg, rgba(126,223,209,0.014) 1px, transparent 1px),
    var(--bg);
  background-size: 72px 72px, 72px 72px, 100% 100%;
  color: var(--text);
  font-family: "Bahnschrift", "Aptos", "Segoe UI", "Helvetica Neue", sans-serif;
  font-variant-numeric: tabular-nums;
}
body::before {
  content: "";
  position: fixed;
  inset: 52px 88px 52px 104px;
  border: 1px solid rgba(142, 220, 210, 0.10);
  pointer-events: none;
}
body::after {
  content: "";
  position: fixed;
  inset: 0;
  background: linear-gradient(180deg, rgba(255,255,255,0.035), transparent 12%, transparent 86%, rgba(0,0,0,0.28));
  pointer-events: none;
}
canvas {
  display: block;
  width: 100vw;
  height: 100vh;
  cursor: grab;
}
canvas:active { cursor: grabbing; }
.topbar {
  position: fixed;
  inset: 0 0 auto 0;
  height: 52px;
  display: grid;
  grid-template-columns: 104px 1fr auto;
  align-items: center;
  border-bottom: 1px solid var(--line-soft);
  background: linear-gradient(180deg, rgba(9, 15, 17, 0.92), rgba(9, 15, 17, 0.68));
  backdrop-filter: blur(16px);
  z-index: 4;
}
.brand {
  display: grid;
  place-items: center;
  height: 100%;
  border-right: 1px solid var(--line-soft);
  color: var(--mesh);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.18em;
}
.mode {
  display: flex;
  align-items: center;
  gap: 12px;
  padding-left: 20px;
  min-width: 0;
}
.mode strong {
  font-size: 12px;
  letter-spacing: 0.22em;
  font-weight: 700;
}
.mode span {
  color: var(--dim);
  font-size: 10px;
  letter-spacing: 0.14em;
}
.status-strip {
  display: flex;
  align-items: center;
  height: 100%;
  border-left: 1px solid var(--line-soft);
}
.status {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 100%;
  min-width: 72px;
  justify-content: center;
  border-right: 1px solid var(--line-soft);
  color: var(--dim);
  font-size: 10px;
  letter-spacing: 0.16em;
}
.status::before {
  content: "";
  width: 7px;
  height: 7px;
  border: 1px solid currentColor;
  background: var(--reach);
  box-shadow: 0 0 12px rgba(101, 240, 168, 0.62);
}
.no-gl .view-status { color: var(--warn); }
.no-gl .view-status::before { background: transparent; box-shadow: none; }
.no-gl .tool,
.no-gl .readout,
.no-gl .metric {
  opacity: 0.46;
}
.no-gl .view-status,
.no-gl .fallback {
  opacity: 1;
}
.metric-rail {
  position: fixed;
  inset: 52px auto 52px 0;
  width: 104px;
  border-right: 1px solid var(--line-soft);
  background: linear-gradient(90deg, rgba(8, 13, 15, 0.84), rgba(8, 13, 15, 0.48));
  backdrop-filter: blur(16px);
  z-index: 3;
}
.metric {
  display: grid;
  align-content: center;
  height: 68px;
  padding: 0 14px;
  border-bottom: 1px solid rgba(160, 190, 184, 0.14);
}
.metric span {
  color: var(--dim);
  font-size: 10px;
  letter-spacing: 0.18em;
}
.metric strong {
  margin-top: 5px;
  color: var(--text);
  font-size: 17px;
  font-weight: 650;
  letter-spacing: 0.02em;
}
.tool-rail {
  position: fixed;
  inset: 72px 20px auto auto;
  display: grid;
  gap: 8px;
  z-index: 5;
}
button.tool {
  position: relative;
  width: 52px;
  height: 52px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(183, 219, 212, 0.26);
  border-radius: 4px;
  color: var(--dim);
  background: rgba(6, 10, 12, 0.72);
  backdrop-filter: blur(14px);
  padding: 0;
}
button.tool svg {
  width: 24px;
  height: 24px;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.8;
  stroke-linecap: square;
  stroke-linejoin: round;
}
button.tool[aria-pressed="true"] {
  color: var(--tool-color, var(--mesh));
  border-color: var(--tool-color, var(--mesh));
  box-shadow: inset 3px 0 0 var(--tool-color, var(--mesh)), 0 0 22px rgba(126, 223, 209, 0.08);
}
button.tool:focus-visible {
  outline: 2px solid var(--mesh);
  outline-offset: 2px;
}
button.tool::after {
  content: attr(data-tip);
  position: absolute;
  right: 52px;
  top: 50%;
  transform: translateY(-50%);
  opacity: 0;
  pointer-events: none;
  white-space: nowrap;
  padding: 7px 9px;
  border: 1px solid var(--line-soft);
  background: rgba(6, 10, 12, 0.9);
  color: var(--text);
  font-size: 11px;
  letter-spacing: 0.06em;
  transition: opacity 140ms ease;
}
button.tool:hover::after { opacity: 1; }
.reticle {
  position: fixed;
  left: calc(50% + 24px);
  top: calc(50% + 2px);
  width: 188px;
  height: 188px;
  transform: translate(-50%, -50%);
  pointer-events: none;
  z-index: 2;
}
.reticle::before,
.reticle::after {
  content: "";
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
}
.reticle::before {
  width: 100%;
  height: 100%;
  border: 1px solid rgba(126, 223, 209, 0.24);
}
.reticle::after {
  width: 9px;
  height: 9px;
  border: 1px solid rgba(231, 241, 239, 0.44);
}
.crosshair {
  position: fixed;
  left: calc(50% + 24px);
  top: calc(50% + 2px);
  width: 260px;
  height: 260px;
  transform: translate(-50%, -50%);
  pointer-events: none;
  z-index: 1;
  background:
    linear-gradient(90deg, transparent calc(50% - 1px), rgba(126,223,209,0.12) calc(50% - 1px), rgba(126,223,209,0.12) calc(50% + 1px), transparent calc(50% + 1px)),
    linear-gradient(0deg, transparent calc(50% - 1px), rgba(126,223,209,0.12) calc(50% - 1px), rgba(126,223,209,0.12) calc(50% + 1px), transparent calc(50% + 1px));
}
.tip-lock {
  position: fixed;
  left: 50%;
  top: 50%;
  width: 66px;
  height: 66px;
  transform: translate(-50%, -50%);
  pointer-events: none;
  z-index: 3;
  opacity: 0;
}
.tip-lock::before,
.tip-lock::after {
  content: "";
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
}
.tip-lock::before {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  border: 1px solid rgba(231, 241, 239, 0.78);
  box-shadow:
    0 -18px 0 -17px rgba(231, 241, 239, 0.82),
    0 18px 0 -17px rgba(231, 241, 239, 0.82),
    18px 0 0 -17px rgba(231, 241, 239, 0.82),
    -18px 0 0 -17px rgba(231, 241, 239, 0.82);
}
.tip-lock::after {
  width: 8px;
  height: 8px;
  border: 1px solid rgba(101, 240, 168, 0.9);
  background: rgba(6, 10, 12, 0.42);
}
.tip-vector {
  position: fixed;
  left: 50%;
  top: 50%;
  width: 1px;
  height: 2px;
  transform-origin: 0 50%;
  pointer-events: none;
  z-index: 3;
  opacity: 0;
  background: linear-gradient(90deg, rgba(231, 241, 239, 0.96), rgba(101, 240, 168, 0.56));
  box-shadow: 0 0 14px rgba(101, 240, 168, 0.34);
}
.branch-cue {
  position: fixed;
  left: 50%;
  top: 50%;
  width: 34px;
  height: 34px;
  transform: translate(-50%, -50%) rotate(45deg);
  pointer-events: none;
  z-index: 3;
  opacity: 0;
  border-top: 1px solid rgba(101, 240, 168, 0.86);
  border-right: 1px solid rgba(101, 240, 168, 0.86);
  box-shadow: 8px -8px 18px rgba(101, 240, 168, 0.10);
}
.branch-cue::after {
  content: "";
  position: absolute;
  right: -1px;
  top: -1px;
  width: 8px;
  height: 8px;
  border-top: 1px solid rgba(231, 241, 239, 0.78);
  border-right: 1px solid rgba(231, 241, 239, 0.78);
}
.axis {
  position: fixed;
  left: 126px;
  bottom: 72px;
  width: 96px;
  height: 96px;
  border-left: 1px solid var(--line-soft);
  border-bottom: 1px solid var(--line-soft);
  color: var(--dim);
  font-size: 10px;
  letter-spacing: 0.12em;
  pointer-events: none;
  z-index: 4;
}
.axis span {
  position: absolute;
  color: var(--text);
}
.axis .s { left: 44px; top: -18px; }
.axis .i { left: 44px; bottom: -19px; }
.axis .l { left: -18px; top: 42px; }
.axis .r { right: -18px; top: 42px; }
.bottom-readout {
  position: fixed;
  inset: auto 0 0 104px;
  height: 52px;
  display: grid;
  grid-template-columns: repeat(5, minmax(90px, 1fr));
  border-top: 1px solid var(--line-soft);
  background: linear-gradient(0deg, rgba(8, 13, 15, 0.88), rgba(8, 13, 15, 0.48));
  backdrop-filter: blur(16px);
  z-index: 4;
}
.corner-cell {
  position: fixed;
  inset: auto auto 0 0;
  width: 104px;
  height: 52px;
  display: grid;
  place-items: center;
  border-top: 1px solid var(--line-soft);
  border-right: 1px solid var(--line-soft);
  background: linear-gradient(0deg, rgba(8, 13, 15, 0.88), rgba(8, 13, 15, 0.48));
  color: var(--dim);
  font-size: 10px;
  letter-spacing: 0.16em;
  z-index: 4;
}
.readout {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  border-right: 1px solid rgba(160, 190, 184, 0.14);
  color: var(--dim);
  font-size: 11px;
  letter-spacing: 0.12em;
}
.readout svg {
  width: 21px;
  height: 21px;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: square;
  stroke-linejoin: round;
}
.readout strong {
  color: var(--text);
  min-width: 42px;
  text-align: left;
  font-size: 12px;
  font-weight: 650;
}
.readout.mesh { color: var(--mesh); }
.readout.tree { color: var(--tree); }
.readout.reach { color: var(--reach); }
.readout.warn { color: var(--warn); }
.readout.is-off {
  color: var(--dim);
  opacity: 0.58;
}
.readout.is-on svg {
  filter: drop-shadow(0 0 8px rgba(101, 240, 168, 0.18));
}
.fallback {
  display: none;
  position: fixed;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  border: 1px solid var(--line-hard);
  padding: 14px 18px;
  background: rgba(6, 10, 12, 0.82);
  color: var(--warn);
  font-size: 11px;
  letter-spacing: 0.16em;
  z-index: 6;
}
.no-gl .fallback { display: block; }
@media (max-width: 760px) {
  .topbar { grid-template-columns: 84px 1fr; }
  .status-strip { display: none; }
  .metric-rail { width: 84px; }
  .metric { padding: 0 10px; height: 68px; }
  .metric strong { font-size: 14px; }
  .tool-rail { right: 12px; top: 64px; }
  .bottom-readout { left: 84px; grid-template-columns: repeat(2, minmax(0, 1fr)); height: 64px; }
  .corner-cell { width: 84px; height: 64px; }
  .readout { font-size: 9px; }
  .axis { display: none; }
  body::before { inset: 52px 64px 64px 84px; }
}
</style>
</head>
<body>
<canvas id="view"></canvas>
<div class="crosshair"></div>
<div class="reticle"></div>
<div class="branch-cue"></div>
<div class="tip-vector"></div>
<div class="tip-lock"></div>
<div id="fallback" class="fallback">GL UNAVAILABLE</div>
<header class="topbar">
  <div class="brand">SAG</div>
  <div class="mode"><strong>AIRWAY MAP</strong><span>3D NAV</span></div>
  <div class="status-strip">
    <div class="status">REG</div>
    <div class="status">PLAN</div>
    <div class="status view-status">VIEW</div>
  </div>
</header>
<aside class="metric-rail" aria-label="Navigation metrics">
  <div class="metric"><span>DPT</span><strong id="dcount"></strong></div>
  <div class="metric"><span>COV</span><strong id="ccount"></strong></div>
  <div class="metric"><span>GEN</span><strong id="gcount"></strong></div>
  <div class="metric"><span>RTE</span><strong id="rcount"></strong></div>
  <div class="metric"><span>RAD</span><strong id="mcount"></strong></div>
</aside>
<nav class="tool-rail" aria-label="Viewer tools">
  <button class="tool" data-toggle="mesh" data-tip="surface" aria-label="Toggle surface" aria-pressed="true" style="--tool-color: var(--mesh)">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 9.5 12 5l7 4.5v5L12 19l-7-4.5z"/><path d="M5 9.5 12 14l7-4.5"/><path d="M12 14v5"/></svg>
  </button>
  <button class="tool" data-toggle="tree" data-tip="tree" aria-label="Toggle airway tree" aria-pressed="false" style="--tool-color: var(--tree)">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v6"/><path d="M12 10 6 16"/><path d="M12 10l6 6"/><path d="M6 16v4"/><path d="M18 16v4"/><path d="M4 20h4"/><path d="M16 20h4"/></svg>
  </button>
  <button class="tool" data-toggle="route" data-tip="route" aria-label="Toggle route" aria-pressed="true" style="--tool-color: var(--reach)">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18c5-1 5-6 10-7"/><path d="M14 11h5"/><path d="m17 8 3 3-3 3"/><path d="M4 18l4 2"/></svg>
  </button>
  <button class="tool" id="tipBack" data-tip="retract" aria-label="Retract virtual tip" aria-pressed="false" style="--tool-color: var(--text)">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 5 9 12l7 7"/><path d="M18 12H9"/></svg>
  </button>
  <button class="tool" id="tipForward" data-tip="advance" aria-label="Advance virtual tip" aria-pressed="false" style="--tool-color: var(--text)">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 7 7-7 7"/><path d="M6 12h9"/></svg>
  </button>
  <button class="tool" id="resetView" data-tip="reset" aria-label="Reset view" aria-pressed="false" style="--tool-color: var(--text)">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 7h8a5 5 0 1 1-4.5 7.2"/><path d="M6 7h5"/><path d="M6 7v5"/></svg>
  </button>
</nav>
<div class="axis" aria-hidden="true"><span class="s">S</span><span class="i">I</span><span class="l">L</span><span class="r">R</span></div>
<div class="corner-cell" aria-hidden="true">LPS</div>
<footer class="bottom-readout" aria-label="Viewer state">
  <div class="readout mesh" id="surfReadout" aria-label="Surface layer"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 9.5 12 5l7 4.5v5L12 19l-7-4.5z"/><path d="M5 9.5 12 14l7-4.5"/><path d="M12 14v5"/></svg><strong id="surfState">ON</strong></div>
  <div class="readout tree" id="treeReadout" aria-label="Airway tree"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v6"/><path d="M12 10 6 16"/><path d="M12 10l6 6"/><path d="M6 16v4"/><path d="M18 16v4"/></svg><strong id="treeState">--</strong></div>
  <div class="readout reach" id="routeReadout" aria-label="Route lock"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18c4-1 5-5 8-7s4-2 7-6"/><path d="M15.5 5.5h5v5"/><path d="M17.2 15.2a3.5 3.5 0 1 0 0 .1"/></svg><strong id="routeState">LOCK</strong></div>
  <div class="readout warn" id="tipReadout" aria-label="Tip progress"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v4"/><path d="M12 17v4"/><path d="M3 12h4"/><path d="M17 12h4"/><path d="M9 12a3 3 0 1 0 6 0 3 3 0 0 0-6 0"/></svg><strong id="tipState">--</strong></div>
  <div class="readout warn" id="distanceReadout" aria-label="Target distance"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16"/><path d="M12 8v4l3 3"/><path d="M12 2v3"/><path d="M12 19v3"/></svg><strong id="distanceState">--</strong></div>
</footer>
<script>
const DATA = __PAYLOAD__;
const canvas = document.getElementById('view');
const targetLock = {
  reticle: document.querySelector('.reticle'),
  crosshair: document.querySelector('.crosshair'),
};
const tipCue = {
  lock: document.querySelector('.tip-lock'),
  vector: document.querySelector('.tip-vector'),
  branch: document.querySelector('.branch-cue'),
};
document.getElementById('dcount').textContent = Math.round(DATA.stats.routeDepthMm) + 'mm';
document.getElementById('ccount').textContent = DATA.stats.coveragePct + '%';
document.getElementById('gcount').textContent = DATA.stats.maxGeneration;
document.getElementById('rcount').textContent = DATA.stats.routeNodes;
document.getElementById('mcount').textContent = DATA.stats.routeMinRadiusMm + 'mm';
const gl = canvas.getContext('webgl', { antialias: true, alpha: true });
if (!gl) {
  document.body.classList.add('no-gl');
  document.getElementById('surfState').textContent = '--';
  document.getElementById('treeState').textContent = '--';
  document.getElementById('routeState').textContent = '--';
  document.getElementById('tipState').textContent = '--';
  document.getElementById('distanceState').textContent = 'FAIL';
  throw new Error('WebGL is unavailable');
}

const meshProgram = program(`
attribute vec3 position;
attribute vec3 normal;
uniform mat4 mvp;
uniform mat3 normalMatrix;
uniform mat4 modelView;
varying vec3 vNormal;
varying vec3 vView;
void main() {
  vNormal = normalize(normalMatrix * normal);
  vec4 viewPos = modelView * vec4(position, 1.0);
  vView = viewPos.xyz;
  gl_Position = mvp * vec4(position, 1.0);
}`, `
precision mediump float;
varying vec3 vNormal;
varying vec3 vView;
uniform vec3 color;
void main() {
  vec3 n = normalize(vNormal);
  vec3 eye = normalize(-vView);
  vec3 lightA = normalize(vec3(0.35, 0.58, 0.72));
  vec3 lightB = normalize(vec3(-0.66, -0.18, 0.44));
  float diffuse = max(dot(n, lightA), 0.0) * 0.74 + max(dot(n, lightB), 0.0) * 0.22;
  float rim = pow(1.0 - max(dot(n, eye), 0.0), 2.3);
  float spec = pow(max(dot(reflect(-lightA, n), eye), 0.0), 18.0) * 0.14;
  vec3 shaded = color * (0.30 + diffuse) + vec3(0.58, 0.95, 0.91) * rim * 0.30 + vec3(spec);
  gl_FragColor = vec4(shaded, 0.60);
}`);
const lineProgram = program(`
attribute vec3 position;
uniform mat4 mvp;
void main() {
  gl_Position = mvp * vec4(position, 1.0);
}`, `
precision mediump float;
uniform vec3 color;
void main() {
  gl_FragColor = vec4(color, 1.0);
}`);
const markerProgram = program(`
attribute vec3 position;
uniform mat4 mvp;
uniform float size;
void main() {
  gl_Position = mvp * vec4(position, 1.0);
  gl_PointSize = size;
}`, `
precision mediump float;
uniform vec3 color;
uniform float alpha;
void main() {
  vec2 p = gl_PointCoord - vec2(0.5);
  float d = length(p);
  if (d > 0.48) discard;
  float ring = smoothstep(0.34, 0.37, d);
  vec3 c = mix(color, vec3(0.02, 0.04, 0.04), ring * 0.45);
  gl_FragColor = vec4(c, alpha);
}`);

const indexType = DATA.stats.vertices > 65535 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
if (indexType === gl.UNSIGNED_INT) gl.getExtension('OES_element_index_uint');
let tipProgress = initialTipProgress();
let markers = markerPoints();
let activeRouteLines = activeRouteSegments();
let headingLines = headingSegment();
let guidanceHaloPoints = guidanceHaloCorridorPoints();
let guidancePoints = guidanceCorridorPoints();
const routeTickPoints = routeTicks();
const targetPoint = DATA.route.length ? DATA.route[DATA.route.length - 1] : null;
syncTipState();
const routeLines = polylineSegments(DATA.route);
const buffers = {
  vertices: arrayBuffer(new Float32Array(DATA.vertices.flat())),
  normals: arrayBuffer(new Float32Array(DATA.normals.flat())),
  faces: indexBuffer(indexType === gl.UNSIGNED_INT ? new Uint32Array(DATA.faces) : new Uint16Array(DATA.faces)),
  lines: arrayBuffer(new Float32Array(DATA.lines.flat())),
  accessible: arrayBuffer(new Float32Array(DATA.accessibleLines.flat())),
  route: arrayBuffer(new Float32Array(routeLines.flat())),
  activeRoute: arrayBuffer(new Float32Array(activeRouteLines.flat())),
  heading: arrayBuffer(new Float32Array(headingLines.flat())),
  guidanceHalo: arrayBuffer(new Float32Array(guidanceHaloPoints.flat())),
  guidance: arrayBuffer(new Float32Array(guidancePoints.flat())),
  ticks: arrayBuffer(new Float32Array(routeTickPoints.flat())),
  markers: arrayBuffer(new Float32Array(markers.flat())),
};

const controls = { mesh: true, tree: false, route: true };
let rotX = -0.34, rotY = 0.84, zoom = 3.05;
let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener('pointerdown', event => { dragging = true; lastX = event.clientX; lastY = event.clientY; canvas.setPointerCapture(event.pointerId); });
canvas.addEventListener('pointerup', event => { dragging = false; canvas.releasePointerCapture(event.pointerId); });
canvas.addEventListener('pointermove', event => {
  if (!dragging) return;
  const dx = event.clientX - lastX, dy = event.clientY - lastY;
  lastX = event.clientX; lastY = event.clientY;
  rotY += dx * 0.008; rotX += dy * 0.008;
  rotX = Math.max(-1.5, Math.min(1.5, rotX));
  draw();
});
canvas.addEventListener('dblclick', resetView);
canvas.addEventListener('wheel', event => {
  event.preventDefault();
  zoom = Math.max(1.4, Math.min(9.0, zoom + event.deltaY * 0.004));
  draw();
}, { passive: false });
document.querySelectorAll('[data-toggle]').forEach(button => {
  button.addEventListener('click', () => {
    const key = button.dataset.toggle;
    controls[key] = !controls[key];
    button.setAttribute('aria-pressed', String(controls[key]));
    syncState();
    draw();
  });
});
document.getElementById('tipBack').addEventListener('click', () => stepTip(-0.08));
document.getElementById('tipForward').addEventListener('click', () => stepTip(0.08));
document.getElementById('resetView').addEventListener('click', resetView);
window.addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') stepTip(-0.04);
  if (event.key === 'ArrowRight' || event.key === 'ArrowUp') stepTip(0.04);
});
window.addEventListener('resize', draw);
syncState();
draw();

function syncState() {
  document.getElementById('surfState').textContent = controls.mesh ? 'ON' : '--';
  document.getElementById('treeState').textContent = controls.tree ? 'ON' : '--';
  document.getElementById('routeState').textContent = controls.route ? 'LOCK' : '--';
  document.getElementById('surfReadout').classList.toggle('is-off', !controls.mesh);
  document.getElementById('treeReadout').classList.toggle('is-off', !controls.tree);
  document.getElementById('routeReadout').classList.toggle('is-off', !controls.route);
  document.getElementById('surfReadout').classList.toggle('is-on', controls.mesh);
  document.getElementById('treeReadout').classList.toggle('is-on', controls.tree);
  document.getElementById('routeReadout').classList.toggle('is-on', controls.route);
}
function resetView() {
  rotX = -0.34;
  rotY = 0.84;
  zoom = 3.05;
  draw();
}
function initialTipProgress() {
  const params = new URLSearchParams(window.location.search);
  if (!params.has('tip')) return 0.62;
  const requested = Number(params.get('tip'));
  if (!Number.isFinite(requested)) return 0.62;
  return Math.max(0, Math.min(1, requested));
}
function stepTip(delta) {
  if (!DATA.route.length) return;
  tipProgress = Math.max(0, Math.min(1, tipProgress + delta));
  markers = markerPoints();
  activeRouteLines = activeRouteSegments();
  headingLines = headingSegment();
  guidanceHaloPoints = guidanceHaloCorridorPoints();
  guidancePoints = guidanceCorridorPoints();
  updateArrayBuffer(buffers.markers, new Float32Array(markers.flat()));
  updateArrayBuffer(buffers.activeRoute, new Float32Array(activeRouteLines.flat()));
  updateArrayBuffer(buffers.heading, new Float32Array(headingLines.flat()));
  updateArrayBuffer(buffers.guidanceHalo, new Float32Array(guidanceHaloPoints.flat()));
  updateArrayBuffer(buffers.guidance, new Float32Array(guidancePoints.flat()));
  syncTipState();
  draw();
}
function syncTipState() {
  document.getElementById('tipState').textContent = markers.length > 2 ? Math.round(tipProgress * 100) + '%' : '--';
  document.getElementById('distanceState').textContent = DATA.route.length ? Math.max(0, Math.round((1 - tipProgress) * DATA.stats.routeDepthMm)) + 'mm' : '--';
}
function markerPoints() {
  const source = DATA.route.length ? DATA.route : (DATA.accessibleLines.length ? DATA.accessibleLines : DATA.lines);
  if (!source.length) return [];
  return [source[0], pointAtProgress(source, tipProgress), source[source.length - 1]];
}
function activeRouteSegments() {
  const source = DATA.route;
  if (source.length < 2) return [];
  const scaled = tipProgress * (source.length - 1);
  const index = Math.max(0, Math.min(source.length - 2, Math.floor(scaled)));
  const points = source.slice(0, index + 1);
  points.push(pointAtProgress(source, tipProgress));
  return polylineSegments(points);
}
function routeTicks() {
  if (DATA.route.length < 2) return [];
  return [0.2, 0.4, 0.6, 0.8].map(progress => pointAtProgress(DATA.route, progress));
}
function guidanceHaloCorridorPoints() {
  if (DATA.route.length < 2 || tipProgress <= 0) return [];
  const count = Math.max(12, Math.min(48, Math.ceil(tipProgress * 54)));
  const points = [];
  for (let index = 1; index <= count; index++) {
    points.push(pointAtProgress(DATA.route, tipProgress * index / count));
  }
  return points;
}
function guidanceCorridorPoints() {
  if (DATA.route.length < 2 || tipProgress <= 0) return [];
  const count = Math.max(3, Math.min(16, Math.ceil(tipProgress * 18)));
  const points = [];
  for (let index = 1; index <= count; index++) {
    points.push(pointAtProgress(DATA.route, tipProgress * index / count));
  }
  return points;
}
function headingSegment() {
  if (DATA.route.length < 2) return [];
  const tip = pointAtProgress(DATA.route, tipProgress);
  const ahead = pointAtProgress(DATA.route, Math.min(1, tipProgress + 0.08));
  return [tip, ahead];
}
function pointAtProgress(points, progress) {
  if (points.length <= 1) return points[0] || [0, 0, 0];
  const scaled = Math.max(0, Math.min(points.length - 1, progress * (points.length - 1)));
  const index = Math.min(points.length - 2, Math.floor(scaled));
  const t = scaled - index;
  const a = points[index], b = points[index + 1];
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}
function polylineSegments(points) {
  const segments = [];
  for (let index = 1; index < points.length; index++) segments.push(points[index - 1], points[index]);
  return segments;
}
function viewMatrix() {
  return multiply(translate(0.18, -0.02, -zoom), multiply(rotateX(rotX), rotateY(rotY)));
}
function draw() {
  resize();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0.0, 0.0, 0.0, 0.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  const aspect = canvas.width / Math.max(canvas.height, 1);
  const mv = viewMatrix();
  const mvp = multiply(perspective(41 * Math.PI / 180, aspect, 0.05, 100), mv);
  const normalMatrix = normalFromMat4(mv);
  if (controls.mesh) {
    gl.useProgram(meshProgram);
    attrib(meshProgram, 'position', buffers.vertices, 3);
    attrib(meshProgram, 'normal', buffers.normals, 3);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.faces);
    gl.uniformMatrix4fv(gl.getUniformLocation(meshProgram, 'mvp'), false, mvp);
    gl.uniformMatrix4fv(gl.getUniformLocation(meshProgram, 'modelView'), false, mv);
    gl.uniformMatrix3fv(gl.getUniformLocation(meshProgram, 'normalMatrix'), false, normalMatrix);
    gl.uniform3f(gl.getUniformLocation(meshProgram, 'color'), 0.42, 0.86, 0.80);
    gl.drawElements(gl.TRIANGLES, DATA.faces.length, indexType, 0);
  }
  gl.disable(gl.DEPTH_TEST);
  if (controls.tree) drawLines(buffers.accessible, DATA.accessibleLines.length, [0.72, 0.59, 0.34]);
  if (controls.route) {
    drawLines(buffers.route, routeLines.length, [0.18, 0.42, 0.37]);
    drawLines(buffers.activeRoute, activeRouteLines.length, [0.62, 1.0, 0.74]);
    drawLines(buffers.heading, headingLines.length, [0.96, 1.0, 0.94]);
  }
  drawGuidanceHalo(mvp);
  drawRouteTicks(mvp);
  drawGuidanceCorridor(mvp);
  drawMarkers(mvp);
  positionTargetLock(mvp);
  positionTipCue(mvp);
}
function drawLines(buffer, count, color) {
  if (!count) return;
  gl.useProgram(lineProgram);
  attrib(lineProgram, 'position', buffer, 3);
  const aspect = canvas.width / Math.max(canvas.height, 1);
  const mvp = multiply(perspective(41 * Math.PI / 180, aspect, 0.05, 100), viewMatrix());
  gl.uniformMatrix4fv(gl.getUniformLocation(lineProgram, 'mvp'), false, mvp);
  gl.uniform3f(gl.getUniformLocation(lineProgram, 'color'), color[0], color[1], color[2]);
  gl.lineWidth(2);
  gl.drawArrays(gl.LINES, 0, count);
}
function drawMarkers(mvp) {
  if (!markers.length) return;
  gl.useProgram(markerProgram);
  attrib(markerProgram, 'position', buffers.markers, 3);
  gl.uniformMatrix4fv(gl.getUniformLocation(markerProgram, 'mvp'), false, mvp);
  gl.uniform1f(gl.getUniformLocation(markerProgram, 'size'), Math.max(11, Math.min(18, canvas.width / 88)));
  gl.uniform1f(gl.getUniformLocation(markerProgram, 'alpha'), 1.0);
  gl.uniform3f(gl.getUniformLocation(markerProgram, 'color'), 0.95, 0.76, 0.34);
  gl.drawArrays(gl.POINTS, 0, 1);
  if (markers.length > 1) {
    gl.uniform1f(gl.getUniformLocation(markerProgram, 'size'), Math.max(14, Math.min(22, canvas.width / 66)));
    gl.uniform1f(gl.getUniformLocation(markerProgram, 'alpha'), 1.0);
    gl.uniform3f(gl.getUniformLocation(markerProgram, 'color'), 0.88, 0.96, 0.92);
    gl.drawArrays(gl.POINTS, 1, 1);
  }
  if (markers.length > 2) {
    gl.uniform1f(gl.getUniformLocation(markerProgram, 'size'), Math.max(12, Math.min(20, canvas.width / 78)));
    gl.uniform1f(gl.getUniformLocation(markerProgram, 'alpha'), 1.0);
    gl.uniform3f(gl.getUniformLocation(markerProgram, 'color'), 0.40, 0.94, 0.66);
    gl.drawArrays(gl.POINTS, 2, 1);
  }
}
function drawRouteTicks(mvp) {
  if (!controls.route || !routeTickPoints.length) return;
  gl.useProgram(markerProgram);
  attrib(markerProgram, 'position', buffers.ticks, 3);
  gl.uniformMatrix4fv(gl.getUniformLocation(markerProgram, 'mvp'), false, mvp);
  gl.uniform1f(gl.getUniformLocation(markerProgram, 'size'), Math.max(6, Math.min(10, canvas.width / 138)));
  gl.uniform1f(gl.getUniformLocation(markerProgram, 'alpha'), 1.0);
  gl.uniform3f(gl.getUniformLocation(markerProgram, 'color'), 0.38, 0.82, 0.60);
  gl.drawArrays(gl.POINTS, 0, routeTickPoints.length);
}
function drawGuidanceHalo(mvp) {
  if (!controls.route || !guidanceHaloPoints.length) return;
  gl.useProgram(markerProgram);
  attrib(markerProgram, 'position', buffers.guidanceHalo, 3);
  gl.uniformMatrix4fv(gl.getUniformLocation(markerProgram, 'mvp'), false, mvp);
  gl.uniform1f(gl.getUniformLocation(markerProgram, 'size'), Math.max(12, Math.min(18, canvas.width / 82)));
  gl.uniform1f(gl.getUniformLocation(markerProgram, 'alpha'), 0.32);
  gl.uniform3f(gl.getUniformLocation(markerProgram, 'color'), 0.48, 1.0, 0.66);
  gl.drawArrays(gl.POINTS, 0, guidanceHaloPoints.length);
}
function drawGuidanceCorridor(mvp) {
  if (!controls.route || !guidancePoints.length) return;
  gl.useProgram(markerProgram);
  attrib(markerProgram, 'position', buffers.guidance, 3);
  gl.uniformMatrix4fv(gl.getUniformLocation(markerProgram, 'mvp'), false, mvp);
  gl.uniform1f(gl.getUniformLocation(markerProgram, 'size'), Math.max(5, Math.min(9, canvas.width / 150)));
  gl.uniform1f(gl.getUniformLocation(markerProgram, 'alpha'), 1.0);
  gl.uniform3f(gl.getUniformLocation(markerProgram, 'color'), 0.68, 1.0, 0.76);
  gl.drawArrays(gl.POINTS, 0, guidancePoints.length);
}
function positionTargetLock(mvp) {
  if (!targetPoint) {
    targetLock.reticle.style.opacity = '0';
    targetLock.crosshair.style.opacity = '0';
    return;
  }
  const projected = projectPoint(targetPoint, mvp);
  if (!projected) {
    targetLock.reticle.style.opacity = '0';
    targetLock.crosshair.style.opacity = '0';
    return;
  }
  for (const element of [targetLock.reticle, targetLock.crosshair]) {
    element.style.opacity = '1';
    element.style.left = projected.x + 'px';
    element.style.top = projected.y + 'px';
  }
}
function positionTipCue(mvp) {
  if (DATA.route.length < 2) {
    tipCue.lock.style.opacity = '0';
    tipCue.vector.style.opacity = '0';
    tipCue.branch.style.opacity = '0';
    return;
  }
  const tip = projectPoint(pointAtProgress(DATA.route, tipProgress), mvp);
  const ahead = projectPoint(pointAtProgress(DATA.route, Math.min(1, tipProgress + 0.12)), mvp);
  const branch = projectPoint(pointAtProgress(DATA.route, Math.min(1, tipProgress + 0.18)), mvp);
  if (!tip || !ahead) {
    tipCue.lock.style.opacity = '0';
    tipCue.vector.style.opacity = '0';
    tipCue.branch.style.opacity = '0';
    return;
  }
  tipCue.lock.style.opacity = '1';
  tipCue.lock.style.left = tip.x + 'px';
  tipCue.lock.style.top = tip.y + 'px';
  const dx = ahead.x - tip.x;
  const dy = ahead.y - tip.y;
  const length = Math.hypot(dx, dy);
  if (length < 8) {
    tipCue.vector.style.opacity = '0';
    return;
  }
  tipCue.vector.style.opacity = '1';
  tipCue.vector.style.left = tip.x + 'px';
  tipCue.vector.style.top = tip.y + 'px';
  tipCue.vector.style.width = Math.min(86, length) + 'px';
  tipCue.vector.style.transform = 'rotate(' + Math.atan2(dy, dx) + 'rad)';
  if (!branch || tipProgress > 0.94) {
    tipCue.branch.style.opacity = '0';
    return;
  }
  tipCue.branch.style.opacity = '1';
  tipCue.branch.style.left = branch.x + 'px';
  tipCue.branch.style.top = branch.y + 'px';
  tipCue.branch.style.transform = 'translate(-50%, -50%) rotate(' + (Math.atan2(dy, dx) + Math.PI / 4) + 'rad)';
}
function projectPoint(point, matrix) {
  const x = point[0], y = point[1], z = point[2];
  const clipX = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
  const clipY = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
  const clipW = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
  if (clipW <= 0.01) return null;
  const ndcX = clipX / clipW;
  const ndcY = clipY / clipW;
  if (Math.abs(ndcX) > 1.25 || Math.abs(ndcY) > 1.25) return null;
  return {
    x: (ndcX * 0.5 + 0.5) * canvas.clientWidth,
    y: (1 - (ndcY * 0.5 + 0.5)) * canvas.clientHeight,
  };
}
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
}
function program(vsSource, fsSource) {
  const p = gl.createProgram();
  gl.attachShader(p, shader(gl.VERTEX_SHADER, vsSource));
  gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fsSource));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
function shader(type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function arrayBuffer(data) { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); return b; }
function updateArrayBuffer(buffer, data) { gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW); }
function indexBuffer(data) { const b = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW); return b; }
function attrib(p, name, buffer, size) { const loc = gl.getAttribLocation(p, name); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0); }
function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2), nf = 1 / (near - far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,(2*far*near)*nf,0]);
}
function translate(x, y, z) { return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]); }
function rotateX(a) { const c=Math.cos(a),s=Math.sin(a); return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]); }
function rotateY(a) { const c=Math.cos(a),s=Math.sin(a); return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]); }
function multiply(a, b) {
  const out = new Float32Array(16);
  for (let c=0; c<4; c++) for (let r=0; r<4; r++) out[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
  return out;
}
function normalFromMat4(m) { return new Float32Array([m[0],m[1],m[2], m[4],m[5],m[6], m[8],m[9],m[10]]); }
</script>
</body>
</html>
""".replace("__PAYLOAD__", payload_json)
