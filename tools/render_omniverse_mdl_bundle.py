"""Headless Isaac Sim/Omniverse RGB capture for SynAirG USD/MDL bundles."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

_TRANSIENT_PRIM_NAME_PREFIXES = (
    "SynAirGDomeLight",
    "SynAirGCameraProofLight",
    "SynAirGDistalLED",
    "SynAirGFillLight",
    "SynAirGProofCamera",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True, help="Path to airway_omnipbr_mdl.usda.")
    parser.add_argument("--output", type=Path, required=True, help="PNG output path.")
    parser.add_argument("--metadata", type=Path, required=True, help="JSON metadata output path.")
    parser.add_argument("--width", type=int, default=768, help="Render width.")
    parser.add_argument("--height", type=int, default=512, help="Render height.")
    parser.add_argument("--stage-open-timeout", type=float, default=180.0, help="Seconds to wait for USD open.")
    parser.add_argument(
        "--experience",
        type=Path,
        default=None,
        help="Optional Kit experience file. Use tools/synairg.mdl_render.kit to avoid Isaac robotics deps.",
    )
    parser.add_argument("--warmup-frames", type=int, default=48, help="Frames to step before capture.")
    parser.add_argument(
        "--renderer",
        default="RealTimePathTracing",
        choices=("RayTracedLighting", "RealTimePathTracing"),
        help="Isaac Sim renderer setting.",
    )
    parser.add_argument(
        "--camera-mode",
        default="external",
        choices=("external", "scope-frame"),
        help="Use the exterior proof camera or a frame from scope_paths.json.",
    )
    parser.add_argument(
        "--scope-paths",
        type=Path,
        default=None,
        help="scope_paths.json for --camera-mode scope-frame.",
    )
    parser.add_argument("--scope-path-id", default=None, help="Optional path_id in scope_paths.json.")
    parser.add_argument(
        "--scope-frame-index",
        type=int,
        default=100,
        help="Frame index within the selected scope path.",
    )
    parser.add_argument(
        "--scope-lookahead-mm",
        type=float,
        default=45.0,
        help="Forward look-ahead distance when the scope frame has no explicit look_at_mm.",
    )
    parser.add_argument("--focal-length", type=float, default=None, help="Camera focal length in stage units.")
    parser.add_argument("--horizontal-aperture", type=float, default=None, help="Camera horizontal aperture.")
    parser.add_argument("--clipping-near", type=float, default=None, help="Near clipping plane.")
    parser.add_argument("--clipping-far", type=float, default=None, help="Far clipping plane.")
    parser.add_argument(
        "--camera-light-radius",
        type=float,
        default=None,
        help="Scale/radius for the light at the camera.",
    )
    parser.add_argument("--rt-subframes", type=int, default=32, help="RTX subframes for the capture step.")
    parser.add_argument(
        "--sequence-output-dir",
        type=Path,
        default=None,
        help="Optional directory for rendering a multi-frame scope trajectory sequence.",
    )
    parser.add_argument(
        "--sequence-start-index",
        type=int,
        default=None,
        help="First scope frame index for --sequence-output-dir. Defaults to --scope-frame-index.",
    )
    parser.add_argument(
        "--sequence-frame-count",
        type=int,
        default=1,
        help="Number of scope frames to render when --sequence-output-dir is set.",
    )
    parser.add_argument(
        "--sequence-frame-stride",
        type=int,
        default=1,
        help="Frame-index stride for --sequence-output-dir.",
    )
    parser.add_argument(
        "--sequence-prefix",
        default="frame",
        help="Filename prefix for sequence frame PNGs.",
    )
    parser.add_argument(
        "--sequence-contact-columns",
        type=int,
        default=4,
        help="Number of columns in the sequence contact sheet written to --output.",
    )
    parser.add_argument(
        "--sequence-contact-sheet-max-frames",
        type=int,
        default=64,
        help="Maximum number of sampled sequence frames included in the contact sheet.",
    )
    parser.add_argument("--dome-intensity", type=float, default=900.0, help="Proof dome light intensity.")
    parser.add_argument(
        "--camera-light-intensity",
        type=float,
        default=180000.0,
        help="Proof light intensity at the camera position.",
    )
    parser.add_argument(
        "--single-camera-light",
        action="store_true",
        help="Use the legacy single camera sphere light instead of dual distal LEDs.",
    )
    parser.add_argument(
        "--scope-led-separation-mm",
        type=float,
        default=2.4,
        help="Center-to-center separation of the two distal scope LEDs in stage millimeters.",
    )
    parser.add_argument(
        "--scope-led-radius-mm",
        type=float,
        default=0.35,
        help="Radius/scale of each distal scope LED sphere in stage millimeters.",
    )
    parser.add_argument(
        "--scope-led-up-offset-mm",
        type=float,
        default=-0.42,
        help="Offset of both LEDs along the camera up vector in stage millimeters.",
    )
    parser.add_argument(
        "--scope-led-forward-offset-mm",
        type=float,
        default=0.85,
        help="Offset of both LEDs along the camera forward vector in stage millimeters.",
    )
    parser.add_argument(
        "--fill-light-intensity",
        type=float,
        default=3500.0,
        help="Proof distant fill light intensity.",
    )
    parser.add_argument(
        "--tone-map",
        choices=("raw", "endoscopic"),
        default="raw",
        help="Optional camera-response tone mapping applied before PNG/metadata output.",
    )
    parser.add_argument(
        "--tone-exposure",
        type=float,
        default=1.45,
        help="Exposure multiplier for --tone-map endoscopic.",
    )
    parser.add_argument(
        "--tone-gamma",
        type=float,
        default=0.60,
        help="Gamma exponent for --tone-map endoscopic; values below 1 lift dark endoscopic frames.",
    )
    parser.add_argument(
        "--tone-saturation",
        type=float,
        default=0.90,
        help="Saturation multiplier for --tone-map endoscopic.",
    )
    parser.add_argument(
        "--tone-black-level",
        type=float,
        default=0.0,
        help="Normalized black level subtracted before --tone-map endoscopic exposure/gamma.",
    )
    parser.add_argument(
        "--tone-white-balance",
        default="1.0,2.1,2.3",
        help="Comma-separated RGB white-balance gains for --tone-map endoscopic.",
    )
    parser.add_argument(
        "--scope-aperture-mask",
        action="store_true",
        help="Apply a circular endoscope aperture/vignette after tone mapping.",
    )
    parser.add_argument(
        "--camera-auto-exposure",
        action="store_true",
        help="Apply constrained per-frame camera auto-exposure after tone mapping.",
    )
    parser.add_argument(
        "--auto-exposure-percentile",
        type=float,
        default=60.0,
        help="Visible-aperture luminance percentile used by --camera-auto-exposure.",
    )
    parser.add_argument(
        "--auto-exposure-target",
        type=float,
        default=0.34,
        help="Normalized luminance target for --camera-auto-exposure.",
    )
    parser.add_argument(
        "--auto-exposure-min-gain",
        type=float,
        default=0.72,
        help="Minimum gain for --camera-auto-exposure.",
    )
    parser.add_argument(
        "--auto-exposure-max-gain",
        type=float,
        default=2.65,
        help="Maximum gain for --camera-auto-exposure.",
    )
    parser.add_argument(
        "--auto-exposure-knee",
        type=float,
        default=0.82,
        help="Soft highlight knee for --camera-auto-exposure.",
    )
    parser.add_argument(
        "--auto-exposure-knee-softness",
        type=float,
        default=0.38,
        help="Soft highlight rolloff width for --camera-auto-exposure.",
    )
    return parser.parse_args()


def _stage_points(stage: Any) -> np.ndarray:
    from pxr import UsdGeom

    points: list[np.ndarray] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        mesh_points = mesh.GetPointsAttr().Get()
        if mesh_points is not None and len(mesh_points) > 0:
            points.append(np.asarray(mesh_points, dtype=np.float64))
    if not points:
        raise RuntimeError("stage contains no mesh points")
    return np.concatenate(points, axis=0)


def _camera_pose(points: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    center = points.mean(axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = maximum - minimum
    radius = max(float(np.linalg.norm(span)), 1.0)
    position = center + np.asarray((0.62 * radius, -1.25 * radius, 0.38 * radius), dtype=np.float64)
    return tuple(float(value) for value in position), tuple(float(value) for value in center)


def _as_vector3(value: Any, *, name: str) -> np.ndarray:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"scope frame must include a 3-vector {name}")
    vector = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"scope frame {name} contains non-finite values")
    return vector


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-9:
        raise ValueError("scope frame contains a zero-length orientation vector")
    return np.asarray(vector / norm, dtype=np.float64)


def _load_scope_path(
    paths_path: Path,
    *,
    path_id: str | None,
) -> dict[str, Any]:
    payload = json.loads(paths_path.read_text(encoding="utf-8"))
    paths = payload.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"scope paths JSON does not contain any paths: {paths_path}")
    selected_path: dict[str, Any] | None = None
    for item in paths:
        if not isinstance(item, dict):
            continue
        if path_id is None or item.get("path_id") == path_id:
            selected_path = item
            break
    if selected_path is None:
        raise ValueError(f"path_id {path_id!r} was not found in {paths_path}")
    return selected_path


def _scope_frame_at(selected_path: dict[str, Any], frame_index: int) -> dict[str, Any]:
    frames = selected_path.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"selected scope path has no frames: {selected_path.get('path_id')!r}")
    resolved_index = frame_index if frame_index >= 0 else len(frames) + frame_index
    if resolved_index < 0 or resolved_index >= len(frames):
        raise ValueError(
            f"scope frame index {frame_index} is outside selected path frame range 0..{len(frames) - 1}"
        )
    frame = frames[resolved_index]
    if not isinstance(frame, dict):
        raise ValueError(f"scope frame {resolved_index} is not a JSON object")
    return frame


def _load_scope_frame(
    paths_path: Path,
    *,
    path_id: str | None,
    frame_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_path = _load_scope_path(paths_path, path_id=path_id)
    frame = _scope_frame_at(selected_path, frame_index)
    return selected_path, frame


def _scope_camera_pose(
    frame: dict[str, Any],
    *,
    lookahead_mm: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    position = _as_vector3(frame.get("position_mm", frame.get("centerline_position_mm")), name="position_mm")
    forward = _unit(_as_vector3(frame.get("forward"), name="forward"))
    up = _unit(_as_vector3(frame.get("up", [0.0, 0.0, 1.0]), name="up"))
    radius_mm = float(frame.get("radius_mm", 2.0))
    camera_offset_mm = float(np.clip(radius_mm * 0.45, 0.8, 2.0))
    camera_position = position - forward * camera_offset_mm
    if "look_at_mm" in frame:
        camera_look_at = _as_vector3(frame["look_at_mm"], name="look_at_mm")
    else:
        camera_look_at = position + forward * max(float(lookahead_mm), 1.0)
    return (
        tuple(float(value) for value in camera_position),
        tuple(float(value) for value in camera_look_at),
        tuple(float(value) for value in up),
    )


def _write_png(path: Path, rgba: np.ndarray) -> None:
    try:
        from PIL import Image

        image = np.asarray(rgba, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise RuntimeError(f"unexpected RGB capture shape: {image.shape}")
        Image.fromarray(image[:, :, :3]).save(path)
    except ImportError as exc:
        raise RuntimeError("Pillow is required in the Isaac Sim Python environment to write PNG captures") from exc


def _parse_rgb_triplet(
    value: str | tuple[float, float, float] | list[float],
    *,
    name: str,
) -> tuple[float, float, float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = list(value)
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly three RGB values")
    triplet = tuple(float(part) for part in parts)
    if not all(math.isfinite(part) and part >= 0.0 for part in triplet):
        raise ValueError(f"{name} values must be finite non-negative numbers")
    return triplet


def _apply_endoscopic_tone_response(
    image: np.ndarray,
    *,
    exposure: float = 1.45,
    gamma: float = 0.60,
    white_balance: tuple[float, float, float] = (1.0, 2.1, 2.3),
    saturation: float = 0.90,
    black_level: float = 0.0,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"unexpected RGB capture shape: {image.shape}")
    if not math.isfinite(float(exposure)) or float(exposure) < 0.0:
        raise ValueError("tone exposure must be a finite non-negative value")
    if not math.isfinite(float(gamma)) or float(gamma) <= 0.0:
        raise ValueError("tone gamma must be a finite positive value")
    if not math.isfinite(float(saturation)) or float(saturation) < 0.0:
        raise ValueError("tone saturation must be a finite non-negative value")
    if not math.isfinite(float(black_level)) or not 0.0 <= float(black_level) < 1.0:
        raise ValueError("tone black level must be in [0, 1)")

    original = np.asarray(image)
    rgb = original[:, :, :3].astype(np.float32) / 255.0
    rgb = np.clip((rgb - float(black_level)) / max(1.0 - float(black_level), 1.0e-6), 0.0, 1.0)
    rgb = np.clip(rgb * float(exposure) * np.asarray(white_balance, dtype=np.float32), 0.0, 1.0)
    rgb = np.power(rgb, float(gamma))
    luminance = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )[:, :, np.newaxis]
    rgb = np.clip(luminance + float(saturation) * (rgb - luminance), 0.0, 1.0)
    toned_rgb = np.rint(rgb * 255.0).astype(np.uint8)
    if original.shape[2] < 4:
        return toned_rgb
    return np.concatenate((toned_rgb, original[:, :, 3:4].astype(np.uint8)), axis=2)


def _apply_camera_response(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    tone_map = str(getattr(args, "tone_map", "raw"))
    if tone_map == "raw":
        toned = np.asarray(image)
    elif tone_map == "endoscopic":
        toned = _apply_endoscopic_tone_response(
            image,
            exposure=float(getattr(args, "tone_exposure", 1.45)),
            gamma=float(getattr(args, "tone_gamma", 0.60)),
            saturation=float(getattr(args, "tone_saturation", 0.90)),
            black_level=float(getattr(args, "tone_black_level", 0.0)),
            white_balance=_parse_rgb_triplet(
                getattr(args, "tone_white_balance", "1.0,2.1,2.3"),
                name="tone white balance",
            ),
        )
    else:
        raise ValueError(f"unsupported tone map: {tone_map}")

    visible_mask = None
    if bool(getattr(args, "scope_aperture_mask", False)):
        visible_mask, _ = _scope_aperture_mask(toned.shape[0], toned.shape[1])
        toned = _apply_scope_aperture_mask(toned)

    if bool(getattr(args, "camera_auto_exposure", False)):
        toned = _apply_camera_auto_exposure(
            toned,
            percentile=float(getattr(args, "auto_exposure_percentile", 60.0)),
            target=float(getattr(args, "auto_exposure_target", 0.34)),
            min_gain=float(getattr(args, "auto_exposure_min_gain", 0.72)),
            max_gain=float(getattr(args, "auto_exposure_max_gain", 2.65)),
            knee=float(getattr(args, "auto_exposure_knee", 0.82)),
            knee_softness=float(getattr(args, "auto_exposure_knee_softness", 0.38)),
            visible_mask=visible_mask,
        )
    return toned


def _apply_camera_auto_exposure(
    image: np.ndarray,
    *,
    percentile: float,
    target: float,
    min_gain: float,
    max_gain: float,
    knee: float,
    knee_softness: float,
    visible_mask: np.ndarray | None = None,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"unexpected RGB capture shape: {image.shape}")
    if not math.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("auto-exposure percentile must be in [0, 100]")
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("auto-exposure target must be positive")
    if not math.isfinite(min_gain) or not math.isfinite(max_gain) or min_gain <= 0.0 or max_gain < min_gain:
        raise ValueError("auto-exposure gain bounds must be finite and satisfy 0 < min <= max")
    if not math.isfinite(knee) or not 0.0 < knee < 1.0:
        raise ValueError("auto-exposure knee must be in (0, 1)")
    if not math.isfinite(knee_softness) or knee_softness <= 0.0:
        raise ValueError("auto-exposure knee softness must be positive")

    original = np.asarray(image)
    rgb = original[:, :, :3].astype(np.float32) / 255.0
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    if visible_mask is not None and visible_mask.shape == luminance.shape and np.any(visible_mask):
        sample = luminance[visible_mask]
    else:
        sample = luminance.reshape(-1)
    reference = float(np.percentile(sample, percentile)) if sample.size else float(target)
    gain = float(np.clip(float(target) / max(reference, 1.0e-3), min_gain, max_gain))
    exposed = rgb * gain
    rolled = np.where(
        exposed <= knee,
        exposed,
        knee + (1.0 - np.exp(-(exposed - knee) / knee_softness)) * (1.0 - knee),
    )
    exposed_rgb = np.rint(np.clip(rolled, 0.0, 1.0) * 255.0).astype(np.uint8)
    if original.shape[2] < 4:
        return exposed_rgb
    return np.concatenate((exposed_rgb, original[:, :, 3:4].astype(np.uint8)), axis=2)


def _apply_scope_aperture_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"unexpected RGB capture shape: {image.shape}")
    original = np.asarray(image)
    rgb = original[:, :, :3].astype(np.float32) / 255.0
    visible, normalized_radius = _scope_aperture_mask(rgb.shape[0], rgb.shape[1])
    vignette = np.clip(1.06 - 0.62 * normalized_radius**2, 0.28, 1.0)
    rgb *= vignette[:, :, np.newaxis]
    rgb[~visible] = 0.0
    border = (normalized_radius > 0.975) & visible
    rgb[border] *= 0.38
    masked_rgb = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    if original.shape[2] < 4:
        return masked_rgb
    return np.concatenate((masked_rgb, original[:, :, 3:4].astype(np.uint8)), axis=2)


def _scope_aperture_mask(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[:height, :width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = min(width, height) * 0.485
    normalized_radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2) / radius
    visible = normalized_radius <= 1.0
    return visible, normalized_radius


def _write_contact_sheet(
    path: Path,
    frames: list[tuple[np.ndarray, str]],
    *,
    columns: int,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont

        if not frames:
            raise RuntimeError("cannot write a contact sheet with no frames")
        columns = max(int(columns), 1)
        thumb_w = 320
        thumb_h = max(int(round(thumb_w * frames[0][0].shape[0] / frames[0][0].shape[1])), 1)
        label_h = 30
        margin = 12
        rows = int(math.ceil(len(frames) / columns))
        sheet = Image.new(
            "RGB",
            (columns * thumb_w + (columns + 1) * margin, rows * (thumb_h + label_h) + (rows + 1) * margin),
            (18, 18, 18),
        )
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        for index, (rgba, label) in enumerate(frames):
            row = index // columns
            col = index % columns
            x = margin + col * (thumb_w + margin)
            y = margin + row * (thumb_h + label_h + margin)
            image = Image.fromarray(np.asarray(rgba[:, :, :3], dtype=np.uint8), mode="RGB")
            sheet.paste(image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y))
            draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill=(28, 28, 28))
            draw.text((x + 8, y + thumb_h + 6), label, fill=(236, 236, 236), font=font)
        sheet.save(path)
    except ImportError as exc:
        raise RuntimeError("Pillow is required in the Isaac Sim Python environment to write contact sheets") from exc


def _sequence_contact_sheet_indices(*, frame_count: int, max_frames: int) -> set[int]:
    frame_count = max(int(frame_count), 0)
    max_frames = max(int(max_frames), 0)
    if frame_count == 0 or max_frames == 0:
        return set()
    if frame_count <= max_frames:
        return set(range(frame_count))
    return {int(round(value)) for value in np.linspace(0, frame_count - 1, max_frames)}


def _destroy_rep_item(item: Any) -> None:
    destroy = getattr(item, "destroy", None)
    if callable(destroy):
        destroy()


def _remove_transient_stage_prims() -> None:
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        paths = [
            prim.GetPath()
            for prim in stage.Traverse()
            if any(prim.GetName().startswith(prefix) for prefix in _TRANSIENT_PRIM_NAME_PREFIXES)
        ]
        for path in sorted(paths, key=lambda value: len(str(value)), reverse=True):
            stage.RemovePrim(path)
    except Exception:
        return


async def _capture_rgb(
    *,
    camera_position: tuple[float, float, float],
    camera_look_at: tuple[float, float, float],
    camera_up: tuple[float, float, float],
    resolution: tuple[int, int],
    camera_light_radius: float,
    dome_intensity: float,
    camera_light_intensity: float,
    fill_light_intensity: float,
    use_dual_scope_leds: bool,
    scope_led_separation_mm: float,
    scope_led_radius_mm: float,
    scope_led_up_offset_mm: float,
    scope_led_forward_offset_mm: float,
    focal_length: float,
    horizontal_aperture: float,
    clipping_range: tuple[float, float],
    rt_subframes: int = 32,
) -> np.ndarray:
    import omni.replicator.core as rep

    _remove_transient_stage_prims()
    dome_light = rep.create.light(
        light_type="dome",
        intensity=float(dome_intensity),
        color=(1.0, 0.96, 0.9),
        name="SynAirGDomeLight",
    )
    camera_lights = _create_scope_lights(
        rep,
        camera_position=camera_position,
        camera_look_at=camera_look_at,
        camera_up=camera_up,
        camera_light_radius=camera_light_radius,
        camera_light_intensity=camera_light_intensity,
        use_dual_scope_leds=use_dual_scope_leds,
        scope_led_separation_mm=scope_led_separation_mm,
        scope_led_radius_mm=scope_led_radius_mm,
        scope_led_up_offset_mm=scope_led_up_offset_mm,
        scope_led_forward_offset_mm=scope_led_forward_offset_mm,
    )
    fill_light = rep.create.light(
        light_type="distant",
        rotation=(52.0, 0.0, 35.0),
        intensity=float(fill_light_intensity),
        color=(1.0, 0.96, 0.9),
        name="SynAirGFillLight",
    )
    camera = rep.create.camera(
        position=camera_position,
        look_at=camera_look_at,
        look_at_up_axis=camera_up,
        clipping_range=clipping_range,
        focal_length=float(focal_length),
        horizontal_aperture=float(horizontal_aperture),
        name="SynAirGProofCamera",
    )
    render_product = rep.create.render_product(camera, resolution, name="SynAirGProofRenderProduct")
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    try:
        await rep.orchestrator.step_async(delta_time=0.0, rt_subframes=rt_subframes)
        return np.asarray(annotator.get_data(do_array_copy=True))
    finally:
        annotator.detach()
        render_product.destroy()
        _destroy_rep_item(camera)
        _destroy_rep_item(fill_light)
        for camera_light in camera_lights:
            _destroy_rep_item(camera_light)
        _destroy_rep_item(dome_light)
        _remove_transient_stage_prims()


def _create_scope_lights(
    rep: Any,
    *,
    camera_position: tuple[float, float, float],
    camera_look_at: tuple[float, float, float],
    camera_up: tuple[float, float, float],
    camera_light_radius: float,
    camera_light_intensity: float,
    use_dual_scope_leds: bool,
    scope_led_separation_mm: float,
    scope_led_radius_mm: float,
    scope_led_up_offset_mm: float,
    scope_led_forward_offset_mm: float,
) -> list[Any]:
    if not use_dual_scope_leds:
        return [
            rep.create.light(
                light_type="sphere",
                position=camera_position,
                scale=(max(float(camera_light_radius), 0.001),) * 3,
                look_at=camera_look_at,
                intensity=float(camera_light_intensity),
                color=(1.0, 0.93, 0.84),
                name="SynAirGCameraProofLight",
            )
        ]

    led_positions = _scope_led_positions(
        camera_position=camera_position,
        camera_look_at=camera_look_at,
        camera_up=camera_up,
        separation_mm=scope_led_separation_mm,
        up_offset_mm=scope_led_up_offset_mm,
        forward_offset_mm=scope_led_forward_offset_mm,
    )
    led_radius = max(float(scope_led_radius_mm), 0.001)
    per_led_intensity = float(camera_light_intensity) * 0.5
    return [
        rep.create.light(
            light_type="sphere",
            position=position,
            scale=(led_radius, led_radius, led_radius),
            look_at=camera_look_at,
            intensity=per_led_intensity,
            color=(1.0, 0.93, 0.84),
            name=f"SynAirGDistalLED{index + 1}",
        )
        for index, position in enumerate(led_positions)
    ]


def _scope_led_positions(
    *,
    camera_position: tuple[float, float, float],
    camera_look_at: tuple[float, float, float],
    camera_up: tuple[float, float, float],
    separation_mm: float,
    up_offset_mm: float,
    forward_offset_mm: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    position = np.asarray(camera_position, dtype=np.float64)
    look_at = np.asarray(camera_look_at, dtype=np.float64)
    forward = _unit(look_at - position)
    requested_up = _unit(np.asarray(camera_up, dtype=np.float64))
    right = _unit(np.cross(forward, requested_up))
    up = _unit(np.cross(right, forward))
    half_separation = max(float(separation_mm), 0.0) * 0.5
    center = position + forward * float(forward_offset_mm) + up * float(up_offset_mm)
    left = center - right * half_separation
    right_pos = center + right * half_separation
    return tuple(float(value) for value in left), tuple(float(value) for value in right_pos)


async def _open_stage_async(context: Any, usd_path: Path, timeout_seconds: float) -> tuple[bool, Any]:
    from pxr import Usd

    resolved = str(usd_path.resolve())
    if not Usd.Stage.IsSupportedFile(resolved):
        raise RuntimeError(f"unsupported USD file: {resolved}")
    context.disable_save_to_recent_files()
    try:
        return await asyncio.wait_for(context.open_stage_async(resolved), timeout=timeout_seconds)
    finally:
        context.enable_save_to_recent_files()


def _capture_metadata(
    *,
    args: argparse.Namespace,
    rgb: np.ndarray,
    output_path: Path,
    camera_position: tuple[float, float, float],
    camera_look_at: tuple[float, float, float],
    camera_up: tuple[float, float, float],
    camera_light_radius: float,
    focal_length: float,
    horizontal_aperture: float,
    clipping_range: tuple[float, float],
    mesh_point_count: int,
    start: float,
    scope_path_payload: dict[str, Any] | None = None,
    scope_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _capture_image_metrics(rgb)
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "usd_path": str(args.usd),
        "output_path": str(output_path),
        "renderer": args.renderer,
        "resolution": [int(args.width), int(args.height)],
        "warmup_frames": int(args.warmup_frames),
        "camera_mode": args.camera_mode,
        "camera_light_radius": float(camera_light_radius),
        "dome_intensity": float(args.dome_intensity),
        "camera_light_intensity": float(args.camera_light_intensity),
        "fill_light_intensity": float(args.fill_light_intensity),
        "light_model": (
            "dual_distal_scope_leds"
            if not getattr(args, "single_camera_light", False)
            else "single_camera_sphere"
        ),
        "scope_led_separation_mm": float(getattr(args, "scope_led_separation_mm", 2.4)),
        "scope_led_radius_mm": float(getattr(args, "scope_led_radius_mm", 0.35)),
        "scope_led_up_offset_mm": float(getattr(args, "scope_led_up_offset_mm", -0.42)),
        "scope_led_forward_offset_mm": float(getattr(args, "scope_led_forward_offset_mm", 0.85)),
        "tone_map": str(getattr(args, "tone_map", "raw")),
        "tone_exposure": float(getattr(args, "tone_exposure", 1.45)),
        "tone_gamma": float(getattr(args, "tone_gamma", 0.60)),
        "tone_saturation": float(getattr(args, "tone_saturation", 0.90)),
        "tone_black_level": float(getattr(args, "tone_black_level", 0.0)),
        "tone_white_balance": list(
            _parse_rgb_triplet(getattr(args, "tone_white_balance", "1.0,2.1,2.3"), name="tone white balance")
        ),
        "scope_aperture_mask": bool(getattr(args, "scope_aperture_mask", False)),
        "camera_auto_exposure": bool(getattr(args, "camera_auto_exposure", False)),
        "auto_exposure_percentile": float(getattr(args, "auto_exposure_percentile", 60.0)),
        "auto_exposure_target": float(getattr(args, "auto_exposure_target", 0.34)),
        "auto_exposure_min_gain": float(getattr(args, "auto_exposure_min_gain", 0.72)),
        "auto_exposure_max_gain": float(getattr(args, "auto_exposure_max_gain", 2.65)),
        "auto_exposure_knee": float(getattr(args, "auto_exposure_knee", 0.82)),
        "auto_exposure_knee_softness": float(getattr(args, "auto_exposure_knee_softness", 0.38)),
        "camera_position": list(camera_position),
        "camera_look_at": list(camera_look_at),
        "camera_up": list(camera_up),
        "focal_length": float(focal_length),
        "horizontal_aperture": float(horizontal_aperture),
        "clipping_range": list(clipping_range),
        "rt_subframes": max(int(args.rt_subframes), 1),
        "mesh_point_count": int(mesh_point_count),
        "rgb_shape": list(rgb.shape),
        **metrics,
        "elapsed_seconds": round(float(time.time() - start), 3),
    }
    if scope_path_payload is not None and scope_frame is not None:
        metadata["scope_paths"] = str(args.scope_paths)
        metadata["scope_path_id"] = scope_path_payload.get("path_id")
        metadata["scope_frame_index"] = scope_frame.get("frame_index", args.scope_frame_index)
        metadata["scope_frame_radius_mm"] = scope_frame.get("radius_mm")
        metadata["scope_frame_insertion_depth_mm"] = scope_frame.get("insertion_depth_mm")
        metadata["scope_frame_motion_phase"] = scope_frame.get("motion_phase")
    if (
        not math.isfinite(float(metadata["rgb_mean"]))
        or float(metadata["capture_coverage_pixel_fraction"]) <= 0.01
    ):
        metadata["status"] = "suspect_capture"
    else:
        metadata["status"] = "captured"
    return metadata


def _capture_image_metrics(image: np.ndarray) -> dict[str, float | int | str | None]:
    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"unexpected RGB capture shape: {image.shape}")
    rgb3 = image[:, :, :3].astype(np.float32)
    nonblack_fraction = float(np.mean(np.max(rgb3, axis=2) > 3.0))
    near_white_fraction = float(np.mean(np.min(rgb3, axis=2) > 250.0))
    metrics: dict[str, float | int | str | None] = {
        "rgb_mean": round(float(np.mean(rgb3)), 4),
        "rgb_std": round(float(np.std(rgb3)), 4),
        "rgb_min": int(np.min(rgb3)),
        "rgb_max": int(np.max(rgb3)),
        "nonblack_pixel_fraction": round(nonblack_fraction, 6),
        "near_white_pixel_fraction": round(near_white_fraction, 6),
        "alpha_coverage_pixel_fraction": None,
        "visible_dark_pixel_fraction": None,
        "capture_coverage_pixel_fraction": round(nonblack_fraction, 6),
        "capture_coverage_source": "rgb_nonblack",
    }
    if image.shape[2] >= 4:
        alpha = image[:, :, 3].astype(np.float32)
        visible = alpha > 0.0
        alpha_coverage = float(np.mean(visible))
        visible_dark = float(np.mean(visible & (np.max(rgb3, axis=2) <= 3.0)))
        metrics.update(
            {
                "alpha_coverage_pixel_fraction": round(alpha_coverage, 6),
                "visible_dark_pixel_fraction": round(visible_dark, 6),
                "capture_coverage_pixel_fraction": round(max(nonblack_fraction, alpha_coverage), 6),
                "capture_coverage_source": "alpha_or_rgb_nonblack",
            }
        )
    return metrics


def main() -> None:
    args = _parse_args()
    if not args.usd.is_file():
        raise SystemExit(f"USD file does not exist: {args.usd}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    launch_config = {
        "headless": True,
        "width": int(args.width),
        "height": int(args.height),
        "renderer": args.renderer,
        "sync_loads": True,
    }
    start = time.time()
    print("[synairg-render] starting SimulationApp", flush=True)
    experience = str(args.experience.resolve()) if args.experience else ""
    from isaacsim import SimulationApp

    app = SimulationApp(launch_config, experience=experience)
    try:
        import carb.settings
        import omni.usd

        settings = carb.settings.get_settings()
        settings.set("/rtx/materialDb/syncLoads", True)
        settings.set("/rtx/hydra/materialSyncLoads", True)
        settings.set("/rtx/pathtracing/spp", 64)
        settings.set("/rtx/pathtracing/totalSpp", 64)
        settings.set("/persistent/app/viewport/displayOptions", 0)

        context = omni.usd.get_context()
        print(f"[synairg-render] opening stage {args.usd}", flush=True)
        stage_open_result, stage_open_error = app.run_coroutine(
            _open_stage_async(context, args.usd, float(args.stage_open_timeout))
        )
        print(
            f"[synairg-render] open_stage_async result={stage_open_result} error={stage_open_error}",
            flush=True,
        )
        if stage_open_result is False:
            raise RuntimeError(f"failed to request stage open: {args.usd}: {stage_open_error}")
        print("[synairg-render] waiting for stage load", flush=True)
        for _ in range(480):
            app.update()
            if context.get_stage() is not None and context.get_stage_loading_status()[2] <= 0:
                break
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("stage did not become available")
        while context.get_stage_loading_status()[2] > 0:
            app.update()

        print("[synairg-render] computing camera pose", flush=True)
        points = _stage_points(stage)
        scope_path_payload: dict[str, Any] | None = None
        scope_frame: dict[str, Any] | None = None
        if args.camera_mode == "scope-frame":
            if args.scope_paths is None:
                raise RuntimeError("--scope-paths is required when --camera-mode scope-frame")
            scope_path_payload, scope_frame = _load_scope_frame(
                args.scope_paths,
                path_id=args.scope_path_id,
                frame_index=int(args.scope_frame_index),
            )
            camera_position, camera_look_at, camera_up = _scope_camera_pose(
                scope_frame,
                lookahead_mm=float(args.scope_lookahead_mm),
            )
            focal_length = float(args.focal_length if args.focal_length is not None else 6.0)
            horizontal_aperture = float(
                args.horizontal_aperture if args.horizontal_aperture is not None else 4.8
            )
            clipping_range = (
                float(args.clipping_near if args.clipping_near is not None else 1.0),
                float(args.clipping_far if args.clipping_far is not None else 95.0),
            )
            camera_light_radius = float(
                args.camera_light_radius if args.camera_light_radius is not None else 1.1
            )
        else:
            camera_position, camera_look_at = _camera_pose(points)
            camera_up = (0.0, 0.0, 1.0)
            focal_length = float(args.focal_length if args.focal_length is not None else 16.0)
            horizontal_aperture = float(
                args.horizontal_aperture if args.horizontal_aperture is not None else 24.0
            )
            clipping_range = (
                float(args.clipping_near if args.clipping_near is not None else 1.0),
                float(args.clipping_far if args.clipping_far is not None else 100000.0),
            )
            scene_span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
            camera_light_radius = float(
                args.camera_light_radius if args.camera_light_radius is not None else max(scene_span * 0.04, 1.0)
            )
        print(
            f"[synairg-render] warmup frames={max(args.warmup_frames, 0)} points={points.shape[0]}",
            flush=True,
        )
        for _ in range(max(args.warmup_frames, 0)):
            app.update()

        if args.sequence_output_dir is not None:
            if args.camera_mode != "scope-frame":
                raise RuntimeError("--sequence-output-dir requires --camera-mode scope-frame")
            if scope_path_payload is None:
                raise RuntimeError("--scope-paths is required when rendering a sequence")
            if int(args.sequence_frame_count) < 1:
                raise RuntimeError("--sequence-frame-count must be positive")
            if int(args.sequence_frame_stride) < 1:
                raise RuntimeError("--sequence-frame-stride must be positive")
            sequence_start = (
                int(args.sequence_start_index)
                if args.sequence_start_index is not None
                else int(args.scope_frame_index)
            )
            args.sequence_output_dir.mkdir(parents=True, exist_ok=True)
            sequence_frames: list[dict[str, Any]] = []
            contact_frames: list[tuple[np.ndarray, str]] = []
            contact_sheet_indices = _sequence_contact_sheet_indices(
                frame_count=int(args.sequence_frame_count),
                max_frames=int(args.sequence_contact_sheet_max_frames),
            )
            print(
                "[synairg-render] capturing sequence "
                f"start={sequence_start} count={args.sequence_frame_count} stride={args.sequence_frame_stride}",
                flush=True,
            )
            for sequence_index in range(int(args.sequence_frame_count)):
                frame_index = sequence_start + sequence_index * int(args.sequence_frame_stride)
                sequence_frame = _scope_frame_at(scope_path_payload, frame_index)
                frame_id = int(sequence_frame.get("frame_index", frame_index))
                camera_position, camera_look_at, camera_up = _scope_camera_pose(
                    sequence_frame,
                    lookahead_mm=float(args.scope_lookahead_mm),
                )
                frame_output_path = (
                    args.sequence_output_dir
                    / f"{args.sequence_prefix}_{sequence_index:04d}_scope-frame-{frame_id:04d}.png"
                )
                print(
                    f"[synairg-render] capturing sequence_index={sequence_index} scope_frame={frame_id}",
                    flush=True,
                )
                rgb = app.run_coroutine(
                    _capture_rgb(
                        camera_position=camera_position,
                        camera_look_at=camera_look_at,
                        camera_up=camera_up,
                        resolution=(int(args.width), int(args.height)),
                        camera_light_radius=camera_light_radius,
                        dome_intensity=float(args.dome_intensity),
                        camera_light_intensity=float(args.camera_light_intensity),
                        fill_light_intensity=float(args.fill_light_intensity),
                        use_dual_scope_leds=not bool(args.single_camera_light),
                        scope_led_separation_mm=float(args.scope_led_separation_mm),
                        scope_led_radius_mm=float(args.scope_led_radius_mm),
                        scope_led_up_offset_mm=float(args.scope_led_up_offset_mm),
                        scope_led_forward_offset_mm=float(args.scope_led_forward_offset_mm),
                        focal_length=focal_length,
                        horizontal_aperture=horizontal_aperture,
                        clipping_range=clipping_range,
                        rt_subframes=max(int(args.rt_subframes), 1),
                    )
                )
                rgb = _apply_camera_response(rgb, args)
                _write_png(frame_output_path, rgb)
                frame_metadata = _capture_metadata(
                    args=args,
                    rgb=rgb,
                    output_path=frame_output_path,
                    camera_position=camera_position,
                    camera_look_at=camera_look_at,
                    camera_up=camera_up,
                    camera_light_radius=camera_light_radius,
                    focal_length=focal_length,
                    horizontal_aperture=horizontal_aperture,
                    clipping_range=clipping_range,
                    mesh_point_count=int(points.shape[0]),
                    start=start,
                    scope_path_payload=scope_path_payload,
                    scope_frame=sequence_frame,
                )
                frame_metadata["sequence_index"] = sequence_index
                frame_metadata_path = frame_output_path.with_suffix(".json")
                frame_metadata["metadata_path"] = str(frame_metadata_path)
                frame_metadata_path.write_text(
                    json.dumps(frame_metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                sequence_frames.append(frame_metadata)
                if sequence_index in contact_sheet_indices:
                    contact_frames.append((rgb, f"frame {frame_id}"))

            print("[synairg-render] writing sequence contact sheet and metadata", flush=True)
            _write_contact_sheet(
                args.output,
                contact_frames,
                columns=max(int(args.sequence_contact_columns), 1),
            )
            nonblack_values = [float(frame["nonblack_pixel_fraction"]) for frame in sequence_frames]
            capture_coverage_values = [
                float(frame["capture_coverage_pixel_fraction"]) for frame in sequence_frames
            ]
            near_white_values = [float(frame["near_white_pixel_fraction"]) for frame in sequence_frames]
            alpha_coverage_values = [
                float(frame["alpha_coverage_pixel_fraction"])
                for frame in sequence_frames
                if frame["alpha_coverage_pixel_fraction"] is not None
            ]
            visible_dark_values = [
                float(frame["visible_dark_pixel_fraction"])
                for frame in sequence_frames
                if frame["visible_dark_pixel_fraction"] is not None
            ]
            coverage_source_counts: dict[str, int] = {}
            for frame in sequence_frames:
                coverage_source = str(frame["capture_coverage_source"])
                coverage_source_counts[coverage_source] = coverage_source_counts.get(coverage_source, 0) + 1
            sequence_status = (
                "captured"
                if all(frame["status"] == "captured" for frame in sequence_frames)
                else "suspect_capture"
            )
            summary = {
                "schema_version": "1.0",
                "status": sequence_status,
                "usd_path": str(args.usd),
                "output_path": str(args.output),
                "sequence_output_dir": str(args.sequence_output_dir),
                "renderer": args.renderer,
                "resolution": [int(args.width), int(args.height)],
                "camera_mode": args.camera_mode,
                "scope_paths": str(args.scope_paths),
                "scope_path_id": scope_path_payload.get("path_id"),
                "sequence_start_index": sequence_start,
                "sequence_frame_count": len(sequence_frames),
                "sequence_frame_stride": int(args.sequence_frame_stride),
                "scope_frame_indices": [frame["scope_frame_index"] for frame in sequence_frames],
                "sequence_contact_sheet_frame_count": len(contact_frames),
                "sequence_contact_sheet_max_frames": int(args.sequence_contact_sheet_max_frames),
                "camera_light_radius": float(camera_light_radius),
                "dome_intensity": float(args.dome_intensity),
                "camera_light_intensity": float(args.camera_light_intensity),
                "fill_light_intensity": float(args.fill_light_intensity),
                "light_model": "dual_distal_scope_leds" if not args.single_camera_light else "single_camera_sphere",
                "scope_led_separation_mm": float(args.scope_led_separation_mm),
                "scope_led_radius_mm": float(args.scope_led_radius_mm),
                "scope_led_up_offset_mm": float(args.scope_led_up_offset_mm),
                "scope_led_forward_offset_mm": float(args.scope_led_forward_offset_mm),
                "tone_map": str(args.tone_map),
                "tone_exposure": float(args.tone_exposure),
                "tone_gamma": float(args.tone_gamma),
                "tone_saturation": float(args.tone_saturation),
                "tone_black_level": float(args.tone_black_level),
                "tone_white_balance": list(
                    _parse_rgb_triplet(args.tone_white_balance, name="tone white balance")
                ),
                "scope_aperture_mask": bool(args.scope_aperture_mask),
                "camera_auto_exposure": bool(args.camera_auto_exposure),
                "auto_exposure_percentile": float(args.auto_exposure_percentile),
                "auto_exposure_target": float(args.auto_exposure_target),
                "auto_exposure_min_gain": float(args.auto_exposure_min_gain),
                "auto_exposure_max_gain": float(args.auto_exposure_max_gain),
                "auto_exposure_knee": float(args.auto_exposure_knee),
                "auto_exposure_knee_softness": float(args.auto_exposure_knee_softness),
                "focal_length": float(focal_length),
                "horizontal_aperture": float(horizontal_aperture),
                "clipping_range": list(clipping_range),
                "rt_subframes": max(int(args.rt_subframes), 1),
                "mesh_point_count": int(points.shape[0]),
                "nonblack_pixel_fraction_min": round(float(min(nonblack_values)), 6),
                "nonblack_pixel_fraction_mean": round(float(np.mean(nonblack_values)), 6),
                "capture_coverage_pixel_fraction_min": round(float(min(capture_coverage_values)), 6),
                "capture_coverage_pixel_fraction_mean": round(float(np.mean(capture_coverage_values)), 6),
                "capture_coverage_source_counts": coverage_source_counts,
                "near_white_pixel_fraction_max": round(float(max(near_white_values)), 6),
                "alpha_coverage_pixel_fraction_min": (
                    round(float(min(alpha_coverage_values)), 6) if alpha_coverage_values else None
                ),
                "alpha_coverage_pixel_fraction_mean": (
                    round(float(np.mean(alpha_coverage_values)), 6) if alpha_coverage_values else None
                ),
                "visible_dark_pixel_fraction_max": (
                    round(float(max(visible_dark_values)), 6) if visible_dark_values else None
                ),
                "visible_dark_pixel_fraction_mean": (
                    round(float(np.mean(visible_dark_values)), 6) if visible_dark_values else None
                ),
                "elapsed_seconds": round(float(time.time() - start), 3),
                "frames": sequence_frames,
            }
            args.metadata.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2, sort_keys=True))
            return

        print("[synairg-render] capturing rgb", flush=True)
        rgb = app.run_coroutine(
            _capture_rgb(
                camera_position=camera_position,
                camera_look_at=camera_look_at,
                camera_up=camera_up,
                resolution=(int(args.width), int(args.height)),
                camera_light_radius=camera_light_radius,
                dome_intensity=float(args.dome_intensity),
                camera_light_intensity=float(args.camera_light_intensity),
                fill_light_intensity=float(args.fill_light_intensity),
                use_dual_scope_leds=not bool(args.single_camera_light),
                scope_led_separation_mm=float(args.scope_led_separation_mm),
                scope_led_radius_mm=float(args.scope_led_radius_mm),
                scope_led_up_offset_mm=float(args.scope_led_up_offset_mm),
                scope_led_forward_offset_mm=float(args.scope_led_forward_offset_mm),
                focal_length=focal_length,
                horizontal_aperture=horizontal_aperture,
                clipping_range=clipping_range,
                rt_subframes=max(int(args.rt_subframes), 1),
            )
        )
        rgb = _apply_camera_response(rgb, args)
        print("[synairg-render] writing png and metadata", flush=True)
        _write_png(args.output, rgb)
        metadata = _capture_metadata(
            args=args,
            rgb=rgb,
            output_path=args.output,
            camera_position=camera_position,
            camera_look_at=camera_look_at,
            camera_up=camera_up,
            camera_light_radius=camera_light_radius,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            clipping_range=clipping_range,
            mesh_point_count=int(points.shape[0]),
            start=start,
            scope_path_payload=scope_path_payload,
            scope_frame=scope_frame,
        )
        args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2, sort_keys=True))
    finally:
        app.close()


if __name__ == "__main__":
    main()
