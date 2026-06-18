"""Small PNG preview writer for airway masks."""

from __future__ import annotations

import binascii
import struct
import zlib
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from mesh_gen.schemas import AirwayMask, AirwayMesh, CenterlineGraph


def write_preview_png(path: Path, mask: AirwayMask, *, size: int = 256) -> None:
    """Write a max-intensity projection PNG preview for a mask."""
    projection = np.asarray(mask.data.max(axis=2), dtype=np.uint8)
    image = _scale_projection(projection, size=size)
    rgb = np.full((image.shape[0], image.shape[1], 3), 248, dtype=np.uint8)
    rgb[image > 0] = np.array([0, 132, 145], dtype=np.uint8)
    _write_png(path, rgb)


def write_segmentation_overlay_png(
    path: Path,
    mask: AirwayMask,
    ct_path: Path,
    *,
    columns: int = 3,
    tile_size: int = 192,
) -> None:
    """Write axial CT slices with mask overlay for 2D segmentation QA."""
    image: Any = nib.load(str(ct_path))
    ct = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if ct.shape[:3] != mask.data.shape:
        raise ValueError("CT and mask shapes must match for segmentation overlay")
    slices = _overlay_slice_indices(mask.data, count=columns * 2)
    tiles = [_overlay_tile(ct[:, :, index], mask.data[:, :, index], size=tile_size) for index in slices]
    rows = int(np.ceil(len(tiles) / columns))
    canvas = np.full((rows * tile_size, columns * tile_size, 3), 248, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns
        y0 = row * tile_size
        x0 = column * tile_size
        canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile
    _write_png(path, canvas)


def write_mesh_review_png(
    path: Path,
    mesh: AirwayMesh,
    graph: CenterlineGraph,
    *,
    panel_size: int = 360,
) -> None:
    """Write a headless 3-view mesh and centreline review PNG."""
    views = (
        ("axial x-y", (0, 1), 2),
        ("coronal x-z", (0, 2), 1),
        ("sagittal y-z", (1, 2), 0),
    )
    panels = [
        _mesh_review_panel(mesh, graph, title, axes, depth_axis, size=panel_size)
        for title, axes, depth_axis in views
    ]
    gutter = 18
    label_height = 34
    height = panel_size + label_height
    width = panel_size * len(panels) + gutter * (len(panels) - 1)
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    for index, panel in enumerate(panels):
        x0 = index * (panel_size + gutter)
        canvas[:, x0 : x0 + panel.shape[1], :] = panel
    _write_png(path, canvas)


def _mesh_review_panel(
    mesh: AirwayMesh,
    graph: CenterlineGraph,
    title: str,
    axes: tuple[int, int],
    depth_axis: int,
    *,
    size: int,
) -> np.ndarray:
    label_height = 34
    panel = np.full((size + label_height, size, 3), 20, dtype=np.uint8)
    drawing = panel[label_height:, :, :]
    vertices = mesh.vertices
    if vertices.size == 0:
        return panel
    projected = vertices[:, list(axes)]
    pixel_vertices = _project_points(projected, size=size, margin=18)
    face_centers = pixel_vertices[mesh.faces].mean(axis=1)
    face_depths = vertices[mesh.faces].mean(axis=1)[:, depth_axis]
    _draw_mesh_points(drawing, face_centers, face_depths)

    node_points = np.asarray([node.point_mm for node in graph.nodes], dtype=np.float64)
    node_pixels = _project_points(node_points[:, list(axes)], size=size, margin=18, bounds=projected)
    node_by_id = {node.node_id: index for index, node in enumerate(graph.nodes)}
    for edge in graph.edges:
        source = node_by_id.get(edge.source)
        target = node_by_id.get(edge.target)
        if source is None or target is None:
            continue
        color = (
            np.array([246, 111, 37], dtype=np.uint8)
            if edge.accessible
            else np.array([120, 124, 132], dtype=np.uint8)
        )
        _draw_line(drawing, node_pixels[source], node_pixels[target], color)
    for node, pixel in zip(graph.nodes, node_pixels, strict=False):
        color = (
            np.array([255, 195, 0], dtype=np.uint8)
            if node.accessible
            else np.array([160, 164, 172], dtype=np.uint8)
        )
        _draw_disc(drawing, pixel, 2, color)

    _draw_text(panel, 10, 10, title, np.array([236, 240, 244], dtype=np.uint8))
    return panel


def _project_points(
    points: np.ndarray,
    *,
    size: int,
    margin: int,
    bounds: np.ndarray | None = None,
) -> np.ndarray:
    source = points if bounds is None else bounds
    minimum = source.min(axis=0)
    maximum = source.max(axis=0)
    extent = np.maximum(maximum - minimum, 1.0e-6)
    scale = (size - 2 * margin) / float(np.max(extent))
    centered = points - ((minimum + maximum) / 2.0)
    pixels = centered * scale + (size / 2.0)
    pixels[:, 1] = size - pixels[:, 1]
    return pixels


def _draw_mesh_points(canvas: np.ndarray, points: np.ndarray, depths: np.ndarray) -> None:
    if points.size == 0:
        return
    order = np.argsort(depths)
    depth_min = float(depths.min())
    depth_span = max(float(depths.max() - depth_min), 1.0e-6)
    for index in order:
        x, y = points[index]
        shade = int(72 + 112 * ((float(depths[index]) - depth_min) / depth_span))
        color = np.array([shade, min(224, shade + 44), 214], dtype=np.uint8)
        _draw_disc(canvas, (x, y), 1, color)


def _draw_line(
    canvas: np.ndarray,
    first: np.ndarray | tuple[float, float],
    second: np.ndarray | tuple[float, float],
    color: np.ndarray,
) -> None:
    x0, y0 = (int(round(float(first[0]))), int(round(float(first[1]))))
    x1, y1 = (int(round(float(second[0]))), int(round(float(second[1]))))
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set_pixel(canvas, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_disc(canvas: np.ndarray, point: np.ndarray | tuple[float, float], radius: int, color: np.ndarray) -> None:
    cx, cy = (int(round(float(point[0]))), int(round(float(point[1]))))
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius:
                _set_pixel(canvas, x, y, color)


def _set_pixel(canvas: np.ndarray, x: int, y: int, color: np.ndarray) -> None:
    if 0 <= y < canvas.shape[0] and 0 <= x < canvas.shape[1]:
        canvas[y, x] = color


def _draw_text(canvas: np.ndarray, x: int, y: int, text: str, color: np.ndarray) -> None:
    cursor = x
    for character in text.lower():
        glyph = _TINY_FONT.get(character, _TINY_FONT[" "])
        for row_index, row in enumerate(glyph):
            for column_index, value in enumerate(row):
                if value == "1":
                    _set_pixel(canvas, cursor + column_index, y + row_index, color)
        cursor += 4
        if cursor >= canvas.shape[1] - 3:
            break


_TINY_FONT = {
    " ": ("000", "000", "000", "000", "000"),
    "-": ("000", "000", "111", "000", "000"),
    "a": ("010", "101", "111", "101", "101"),
    "c": ("111", "100", "100", "100", "111"),
    "g": ("111", "100", "101", "101", "111"),
    "i": ("111", "010", "010", "010", "111"),
    "l": ("100", "100", "100", "100", "111"),
    "n": ("101", "111", "111", "111", "101"),
    "o": ("111", "101", "101", "101", "111"),
    "r": ("110", "101", "110", "101", "101"),
    "s": ("111", "100", "111", "001", "111"),
    "t": ("111", "010", "010", "010", "010"),
    "x": ("101", "101", "010", "101", "101"),
    "y": ("101", "101", "010", "010", "010"),
    "z": ("111", "001", "010", "100", "111"),
}


def _scale_projection(projection: np.ndarray, *, size: int) -> np.ndarray:
    return _fit_nearest_2d(projection, size=size, fill_value=0)


def _overlay_slice_indices(mask: np.ndarray, *, count: int) -> list[int]:
    z_values = np.flatnonzero(mask.any(axis=(0, 1)))
    if z_values.size == 0:
        return [mask.shape[2] // 2]
    if z_values.size <= count:
        return [int(value) for value in z_values]
    sample_positions = np.linspace(0, z_values.size - 1, count)
    return [int(z_values[int(round(position))]) for position in sample_positions]


def _overlay_tile(ct_slice: np.ndarray, mask_slice: np.ndarray, *, size: int) -> np.ndarray:
    grayscale = _window_ct(ct_slice)
    image = _scale_grayscale(grayscale, size=size)
    mask_image = _scale_projection(np.asarray(mask_slice, dtype=np.uint8), size=size).astype(bool)
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    overlay_color = np.array([0, 182, 170], dtype=np.uint8)
    rgb[mask_image] = ((0.45 * rgb[mask_image]) + (0.55 * overlay_color)).astype(np.uint8)
    return rgb


def _window_ct(values: np.ndarray, *, center: float = -600.0, width: float = 1500.0) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(values, low, high)
    scaled = (clipped - low) / max(high - low, 1.0e-6)
    return np.asarray(np.round(scaled * 255.0), dtype=np.uint8)


def _scale_grayscale(image: np.ndarray, *, size: int) -> np.ndarray:
    return _fit_nearest_2d(image, size=size, fill_value=248)


def _fit_nearest_2d(image: np.ndarray, *, size: int, fill_value: int) -> np.ndarray:
    height, width = image.shape
    if height == 0 or width == 0:
        return np.full((size, size), fill_value, dtype=np.uint8)
    scale = min(size / float(height), size / float(width))
    target_height = max(1, min(size, int(round(height * scale))))
    target_width = max(1, min(size, int(round(width * scale))))
    y_indices = np.rint(np.linspace(0, height - 1, target_height)).astype(np.int64)
    x_indices = np.rint(np.linspace(0, width - 1, target_width)).astype(np.int64)
    resized = np.asarray(image[np.ix_(y_indices, x_indices)], dtype=np.uint8)
    canvas = np.full((size, size), fill_value, dtype=np.uint8)
    y_offset = (size - target_height) // 2
    x_offset = (size - target_width) // 2
    canvas[y_offset : y_offset + target_height, x_offset : x_offset + target_width] = resized
    return canvas


def _write_png(path: Path, rgb: np.ndarray) -> None:
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("PNG writer expects an RGB image")
    raw_rows = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    chunks = [
        _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        _chunk(b"IDAT", zlib.compress(raw_rows)),
        _chunk(b"IEND", b""),
    ]
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"".join(chunks))


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(chunk_type)
    checksum = binascii.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)
