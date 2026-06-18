"""OBJ, PLY, and USD mesh exporters."""

from __future__ import annotations

from pathlib import Path

from mesh_gen.schemas import AirwayMesh


def write_obj(path: Path, mesh: AirwayMesh) -> None:
    """Write a Wavefront OBJ mesh."""
    lines = ["# SynAirG airway mesh\n"]
    for vertex in mesh.vertices:
        lines.append(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
    for normal in mesh.vertex_normals:
        lines.append(f"vn {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n")
    for face in mesh.faces:
        a, b, c = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
        lines.append(f"f {a}//{a} {b}//{b} {c}//{c}\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_ply(path: Path, mesh: AirwayMesh) -> None:
    """Write an ASCII PLY mesh."""
    lines = [
        "ply\n",
        "format ascii 1.0\n",
        "comment SynAirG airway mesh\n",
        f"element vertex {mesh.vertices.shape[0]}\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "property float nx\n",
        "property float ny\n",
        "property float nz\n",
        f"element face {mesh.faces.shape[0]}\n",
        "property list uchar int vertex_indices\n",
        "end_header\n",
    ]
    for vertex, normal in zip(mesh.vertices, mesh.vertex_normals, strict=True):
        lines.append(
            f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n"
        )
    for face in mesh.faces:
        lines.append(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_usd(path: Path, mesh: AirwayMesh) -> None:
    """Write an ASCII USD mesh payload to a .usd path."""
    point_values = ", ".join(_point(float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in mesh.vertices)
    face_counts = ", ".join("3" for _ in mesh.faces)
    face_indices = ", ".join(str(int(index)) for face in mesh.faces for index in face)
    normal_values = ", ".join(
        _point(float(normal[0]), float(normal[1]), float(normal[2])) for normal in mesh.vertex_normals
    )
    lines = [
        "#usda 1.0\n",
        "(\n",
        '    defaultPrim = "Airway"\n',
        "    metersPerUnit = 0.001\n",
        '    upAxis = "Z"\n',
        ")\n\n",
        'def Xform "Airway"\n',
        "{\n",
        '    def Mesh "AirwayMesh"\n',
        "    {\n",
        f"        point3f[] points = [{point_values}]\n",
        f"        int[] faceVertexCounts = [{face_counts}]\n",
        f"        int[] faceVertexIndices = [{face_indices}]\n",
        '        uniform token subdivisionScheme = "none"\n',
        f"        normal3f[] normals = [{normal_values}]\n",
        '        uniform token normals:interpolation = "vertex"\n',
        "    }\n",
        "}\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def _point(x: float, y: float, z: float) -> str:
    return f"({x:.9g}, {y:.9g}, {z:.9g})"
