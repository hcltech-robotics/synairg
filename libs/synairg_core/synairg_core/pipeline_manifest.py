"""Manifest-driven SynAirG pipeline planning."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from synairg_core.volume_manifest import JsonValue, ManifestValidationError, validate_case_id

MASK_SOURCE_KINDS = {
    "maisi_label",
    "mask_conditioned_maisi",
    "monai_segmenter",
    "external_segmenter",
    "precomputed",
    "ct_threshold",
}
VOLUME_SOURCE_KINDS = {"local_ct", "local_dicom", "maisi", "mask_conditioned_maisi"}
PBR_RENDERERS = {"omniverse_rtx", "learnable_renderer", "software_preview"}


@dataclass(frozen=True)
class PipelineStep:
    """One executable stage command derived from a SynAirG manifest."""

    stage: str
    command: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "stage": self.stage,
            "description": self.description,
            "command": list(self.command),
            "shell": shlex.join(self.command),
        }


@dataclass(frozen=True)
class PipelineManifest:
    """Validated manifest for a reusable SynAirG case generation run."""

    case_id: str
    volume: dict[str, JsonValue]
    mask_source: dict[str, JsonValue]
    mesh: dict[str, JsonValue] = field(default_factory=dict)
    pose: dict[str, JsonValue] = field(default_factory=dict)
    render: dict[str, JsonValue] = field(default_factory=dict)
    rl: dict[str, JsonValue] = field(default_factory=dict)
    output_root: Path = Path("datasets")
    schema_version: str = "1.0"

    @classmethod
    def from_mapping(cls, value: object, *, base_dir: Path | None = None) -> PipelineManifest:
        mapping = _require_mapping(value, "manifest")
        case_id = validate_case_id(_require_string(mapping.get("case_id"), "case_id"))
        output_root = Path(_require_string(mapping.get("output_root", "datasets"), "output_root"))
        volume = _json_mapping(mapping.get("volume"), "volume")
        mask_source = _json_mapping(mapping.get("mask_source"), "mask_source")
        mesh = _json_mapping(mapping.get("mesh", {}), "mesh")
        pose = _json_mapping(mapping.get("pose", {}), "pose")
        render = _json_mapping(mapping.get("render", {}), "render")
        rl = _json_mapping(mapping.get("rl", {}), "rl")
        schema_version = _require_string(mapping.get("schema_version", "1.0"), "schema_version")
        manifest = cls(
            case_id=case_id,
            volume=volume,
            mask_source=mask_source,
            mesh=mesh,
            pose=pose,
            render=render,
            rl=rl,
            output_root=output_root,
            schema_version=schema_version,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        """Validate the manifest contract without touching external systems."""
        volume_kind = _kind(self.volume, "volume")
        if volume_kind not in VOLUME_SOURCE_KINDS:
            raise ManifestValidationError(
                f"volume.kind must be one of: {', '.join(sorted(VOLUME_SOURCE_KINDS))}"
            )
        mask_kind = _kind(self.mask_source, "mask_source")
        if mask_kind not in MASK_SOURCE_KINDS:
            raise ManifestValidationError(
                f"mask_source.kind must be one of: {', '.join(sorted(MASK_SOURCE_KINDS))}"
            )
        if volume_kind == "local_ct":
            _require_any_path(self.volume, ("ct_path", "volume_path"), "volume")
        elif volume_kind == "local_dicom":
            _require_any_path(self.volume, ("dicom_dir", "input_dir"), "volume")
        else:
            _require_any_path(self.volume, ("config", "config_path"), "volume")

        if mask_kind in {"precomputed", "mask_conditioned_maisi"}:
            _require_any_path(self.mask_source, ("mask_path", "path"), "mask_source")
        if mask_kind == "maisi_label" and not any(key in self.mask_source for key in ("mask_path", "label_path")):
            raise ManifestValidationError("mask_source must contain mask_path or label_path for maisi_label")
        if mask_kind in {"monai_segmenter", "external_segmenter"}:
            _require_string(self.mask_source.get("command"), "mask_source.command")
        if mask_kind == "ct_threshold":
            _number(self.mask_source.get("threshold_hu", -500.0), "mask_source.threshold_hu")

        renderer = str(self.render.get("final_renderer", "omniverse_rtx"))
        if renderer not in PBR_RENDERERS:
            raise ManifestValidationError(f"render.final_renderer must be one of: {', '.join(sorted(PBR_RENDERERS))}")
        condition_modalities = _string_list(
            self.render.get("condition_modalities", ["rgb", "depth", "normal", "pps", "mask"]),
            "render.condition_modalities",
        )
        missing = {"rgb", "depth", "normal", "pps", "mask"} - set(condition_modalities)
        if missing:
            raise ManifestValidationError(
                "render.condition_modalities must include rgb, depth, normal, pps, and mask"
            )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "output_root": str(self.output_root),
            "volume": self.volume,
            "mask_source": self.mask_source,
            "mesh": self.mesh,
            "pose": self.pose,
            "render": self.render,
            "rl": self.rl,
        }


def read_pipeline_manifest(path: Path) -> PipelineManifest:
    """Read a JSON or YAML SynAirG pipeline manifest."""
    return PipelineManifest.from_mapping(_load_mapping(path), base_dir=path.parent)


def build_pipeline_plan(manifest: PipelineManifest) -> dict[str, JsonValue]:
    """Build an executable command plan for the validated manifest."""
    steps = _pipeline_steps(manifest)
    return {
        "schema_version": "1.0",
        "case_id": manifest.case_id,
        "output_root": str(manifest.output_root),
        "steps": [step.to_dict() for step in steps],
        "rl_conditioning": manifest.rl,
        "contracts": {
            "mask_required_for_meshing": True,
            "condition_modalities": ["rgb", "depth", "normal", "pps", "mask"],
            "preferred_final_renderer": str(manifest.render.get("final_renderer", "omniverse_rtx")),
        },
    }


def _pipeline_steps(manifest: PipelineManifest) -> tuple[PipelineStep, ...]:
    case_id = manifest.case_id
    output_root = manifest.output_root
    volume_root = output_root / "volumes"
    mesh_root = output_root / "meshes"
    pose_root = output_root / "poses"
    video_root = output_root / "videos"
    render_root = output_root / "renders"
    mesh_case_id = _string(manifest.mesh.get("case_id"), f"{case_id}-mesh")
    mesh_case = mesh_root / mesh_case_id
    pose_case = pose_root / mesh_case_id
    conditions_root = video_root / f"{case_id}-conditions"
    pbr_case_id = _string(manifest.render.get("pbr_case_id"), f"{case_id}-pbr")
    pbr_root = render_root / pbr_case_id

    steps: list[PipelineStep] = []
    volume_kind = _kind(manifest.volume, "volume")
    ct_path = _planned_ct_path(manifest)
    if volume_kind == "local_ct":
        steps.append(
            PipelineStep(
                stage="volume",
                command=("synairg", "volume", "status"),
                description="Use the supplied CT volume as the source scan.",
            )
        )
    elif volume_kind == "local_dicom":
        steps.append(
            PipelineStep(
                stage="volume",
                command=(
                    "synairg",
                    "volume",
                    "import-dicom",
                    "--input",
                    str(_path_value(manifest.volume, ("dicom_dir", "input_dir"))),
                    "--case-id",
                    case_id,
                    "--output",
                    str(volume_root),
                    "--overwrite",
                ),
                description="Import a DICOM CT series into canonical NIfTI form.",
            )
        )
    else:
        steps.append(
            PipelineStep(
                stage="volume",
                command=(
                    "synairg",
                    "volume",
                    "generate-maisi",
                    "--config",
                    str(_path_value(manifest.volume, ("config", "config_path"))),
                    "--output",
                    str(volume_root),
                    "--overwrite",
                ),
                description="Generate a physiologically plausible thoracic CT volume through the MAISI backend.",
            )
        )

    mask_kind = _kind(manifest.mask_source, "mask_source")
    mask_path = _planned_mask_path(manifest)
    mesh_command = [
        "synairg",
        "mesh",
        "run",
        "--ct",
        str(ct_path),
        "--case-id",
        mesh_case_id,
        "--output",
        str(mesh_root),
        "--backend",
        _mesh_backend(mask_kind),
        "--refinement-preset",
        _string(manifest.mesh.get("refinement_preset"), "high-detail"),
        "--normal-orientation",
        _string(manifest.mesh.get("normal_orientation"), "inward"),
        "--overwrite",
    ]
    if mask_path is not None:
        mesh_command.extend(["--mask", str(mask_path)])
    if mask_kind in {"monai_segmenter", "external_segmenter"}:
        mesh_command.extend(
            [
                "--segmenter-command",
                _require_string(manifest.mask_source.get("command"), "mask_source.command"),
            ]
        )
    if mask_kind == "ct_threshold":
        threshold_hu = _number(manifest.mask_source.get("threshold_hu", -500.0), "mask_source.threshold_hu")
        mesh_command.extend(["--threshold-hu", str(threshold_hu)])
    steps.append(
        PipelineStep(
            stage="mesh",
            command=tuple(mesh_command),
            description="Segment or reuse the airway mask, mesh it, and write topology/QA artefacts.",
        )
    )

    steps.append(
        PipelineStep(
            stage="pose",
            command=(
                "synairg",
                "pose",
                "run",
                "--mesh-case",
                str(mesh_case),
                "--path-count",
                str(_int(manifest.pose.get("path_count"), 3)),
                "--output",
                str(pose_root),
                "--include-procedure",
                "--overwrite",
            ),
            description="Sample bronchoscope trajectories over the validated centreline graph.",
        )
    )

    material_map = manifest.render.get("material_map")
    video_command = [
        "synairg",
        "video",
        "render-scope",
        "--mesh-case",
        str(mesh_case),
        "--paths",
        str(pose_case / "scope_paths.json"),
        "--material-profile",
        "pbr",
        "--material-variant",
        _string(manifest.render.get("material_variant"), "healthy"),
        "--export-conditions",
        str(conditions_root),
        "--output",
        str(video_root),
        "--overwrite",
    ]
    if material_map is not None:
        video_command.extend(["--material-map", str(material_map)])
    steps.append(
        PipelineStep(
            stage="video",
            command=tuple(video_command),
            description="Render scope video plus aligned RGB, depth, normal, PPS, and mask condition frames.",
        )
    )

    mdl_command = [
        "synairg",
        "video",
        "export-material-mdl",
        "--mesh-case",
        str(mesh_case),
        "--output",
        str(render_root),
        "--case-id",
        pbr_case_id,
        "--material-variant",
        _string(manifest.render.get("material_variant"), "healthy"),
        "--mdl-shader-target",
        _string(manifest.render.get("mdl_shader_target"), "omnipbr-clearcoat"),
        "--overwrite",
    ]
    if material_map is not None:
        mdl_command.extend(["--material-map", str(material_map)])
    steps.append(
        PipelineStep(
            stage="pbr",
            command=tuple(mdl_command),
            description="Export an Omniverse-ready USD/MDL bundle with authorable mucosal material maps.",
        )
    )

    if str(manifest.render.get("final_renderer", "omniverse_rtx")) == "omniverse_rtx":
        steps.append(
            PipelineStep(
                stage="rtx",
                command=(
                    "python",
                    "tools/render_omniverse_mdl_bundle.py",
                    "--usd",
                    str(pbr_root / "airway_omnipbr_mdl.usda"),
                    "--output",
                    str(pbr_root / "rtx_preview.png"),
                    "--metadata",
                    str(pbr_root / "rtx_preview.json"),
                    "--experience",
                    "tools/synairg.mdl_render.kit",
                    "--renderer",
                    "RealTimePathTracing",
                    "--tone-map",
                    "endoscopic",
                ),
                description="Render the USD/MDL bundle with Omniverse RTX path tracing.",
            )
        )
    return tuple(steps)


def _load_mapping(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
            raise ManifestValidationError("PyYAML is required for YAML pipeline manifests") from exc
        loaded = yaml.safe_load(text)
    else:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(f"{path} is not valid JSON: {exc.msg}") from exc
    return _require_mapping(loaded, "manifest")


def _planned_ct_path(manifest: PipelineManifest) -> Path:
    if _kind(manifest.volume, "volume") == "local_ct":
        return _path_value(manifest.volume, ("ct_path", "volume_path"))
    return manifest.output_root / "volumes" / manifest.case_id / "ct.nii.gz"


def _planned_mask_path(manifest: PipelineManifest) -> Path | None:
    mask_kind = _kind(manifest.mask_source, "mask_source")
    if mask_kind in {"precomputed", "mask_conditioned_maisi"}:
        return _path_value(manifest.mask_source, ("mask_path", "path"))
    if mask_kind == "maisi_label":
        if "mask_path" in manifest.mask_source:
            return _path_value(manifest.mask_source, ("mask_path",))
        return manifest.output_root / "volumes" / manifest.case_id / "label.nii.gz"
    return None


def _mesh_backend(mask_kind: str) -> str:
    if mask_kind in {"monai_segmenter", "external_segmenter"}:
        return "external-command"
    if mask_kind == "ct_threshold":
        return "ct-threshold"
    return "precomputed-mask"


def _kind(mapping: dict[str, JsonValue], field_name: str) -> str:
    return _require_string(mapping.get("kind", mapping.get("type")), f"{field_name}.kind")


def _path_value(mapping: dict[str, JsonValue], keys: tuple[str, ...]) -> Path:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value)
    raise ManifestValidationError(f"expected one of {', '.join(keys)}")


def _require_any_path(mapping: dict[str, JsonValue], keys: tuple[str, ...], field_name: str) -> None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return
    raise ManifestValidationError(f"{field_name} must contain one of: {', '.join(keys)}")


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{field_name} must be an object")
    return cast(dict[str, object], value)


def _json_mapping(value: object, field_name: str) -> dict[str, JsonValue]:
    mapping = _require_mapping(value, field_name)
    return {str(key): _json_value(item, f"{field_name}.{key}") for key, item in mapping.items()}


def _json_value(value: object, field_name: str) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{field_name}[]") for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item, f"{field_name}.{key}") for key, item in value.items()}
    raise ManifestValidationError(f"{field_name} must be JSON-serialisable")


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field_name} must be a non-empty string")
    return value


def _string(value: JsonValue | None, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError("expected a non-empty string")
    return value


def _string_list(value: JsonValue | None, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ManifestValidationError(f"{field_name} must be a list")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ManifestValidationError(f"{field_name} entries must be non-empty strings")
        result.append(item)
    return result


def _int(value: JsonValue | None, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int):
        raise ManifestValidationError("expected an integer")
    return value


def _number(value: object, field_name: str) -> float:
    if not isinstance(value, int | float):
        raise ManifestValidationError(f"{field_name} must be numeric")
    return float(value)
