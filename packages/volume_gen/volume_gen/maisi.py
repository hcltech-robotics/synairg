"""MAISI backend abstraction for synthetic CT generation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from synairg_core.volume_manifest import (
    JsonValue,
    ManifestValidationError,
    VolumeManifest,
    VolumeSource,
    validate_case_id,
)

from volume_gen.artifacts import VolumeCase, prepare_case_dir, write_json
from volume_gen.errors import BackendUnavailableError, VolumeImportError
from volume_gen.importers import write_case_sidecars


@dataclass(frozen=True)
class MaisiCaseSpec:
    """Configuration for one MAISI-compatible CT generation request."""

    case_id: str
    infer_config_path: Path
    random_seed: int = 0
    version: str = "rflow-ct"
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class MaisiBatchSpec:
    """Configuration for a batch of MAISI-compatible CT generation requests."""

    cases: tuple[MaisiCaseSpec, ...]
    metadata: dict[str, JsonValue] = field(default_factory=dict)


class CTGenerator(Protocol):
    """Base protocol for configured CT generation backends."""

    name: str

    def generate(self, config: dict[str, JsonValue], output_path: Path) -> dict[str, JsonValue]:
        """Generate a CT NIfTI volume at output_path and return backend metadata."""


class MaisiBackend(CTGenerator, Protocol):
    """Backend interface for MAISI-compatible CT volume generation."""


class UnavailableMaisiBackend:
    """Default backend used when MAISI dependencies are not installed."""

    name = "unavailable-maisi"

    def generate(self, config: dict[str, JsonValue], output_path: Path) -> dict[str, JsonValue]:
        """Fail with an actionable message instead of importing unavailable MAISI code."""
        raise BackendUnavailableError(
            "MAISI backend is unavailable. Install and configure a MAISI-compatible backend, then pass it through "
            "volume_gen.maisi.generate_maisi_case, or configure a command backend in the MAISI config JSON."
        )


class CommandMaisiBackend:
    """MAISI-compatible backend adapter that executes a configured command."""

    def __init__(self, config_path: Path, backend_config: dict[str, JsonValue]) -> None:
        self.config_path = config_path
        self.name = _optional_string(backend_config.get("name"), "command-maisi")
        self.command = _command_from_backend_config(backend_config, "MAISI command backend")
        self.timeout_seconds = _timeout_from_backend_config(backend_config, default=3600)
        self.env = _env_from_backend_config(backend_config)

    def generate(self, config: dict[str, JsonValue], output_path: Path) -> dict[str, JsonValue]:
        """Run the configured command and expect it to create output_path."""
        output_path = output_path.resolve()
        command = [
            _expand_command_part(part, config_path=self.config_path, output_path=output_path, config=config)
            for part in self.command
        ]
        completed = _run_backend_command(
            command,
            cwd=self.config_path.parent,
            timeout_seconds=self.timeout_seconds,
            env=self.env,
            backend_name="MAISI command backend",
        )

        if completed.returncode != 0:
            detail = _trim_output(completed.stderr) or _trim_output(completed.stdout) or "no command output"
            raise VolumeImportError(f"MAISI command backend exited with {completed.returncode}: {detail}")

        metadata: dict[str, JsonValue] = {
            "command": cast(JsonValue, command),
            "returncode": completed.returncode,
        }
        stdout = _trim_output(completed.stdout)
        stderr = _trim_output(completed.stderr)
        if stdout:
            metadata["stdout"] = stdout
        if stderr:
            metadata["stderr"] = stderr
        return metadata


class NvGenerateCtBackend:
    """Adapter for the real NV-Generate-CTMR/MAISI wrapper output contract."""

    def __init__(self, config_path: Path, backend_config: dict[str, JsonValue]) -> None:
        self.config_path = config_path
        self.name = _optional_string(backend_config.get("name"), "nv-generate-ct")
        self.command = _command_from_backend_config(backend_config, "NV-Generate CT backend")
        self.timeout_seconds = _timeout_from_backend_config(backend_config, default=7200)
        self.env = _env_from_backend_config(backend_config)
        self.version = _optional_string(backend_config.get("version"), "rflow-ct")
        self.random_seed = _optional_int(backend_config.get("random_seed"), 0, "backend.random_seed")
        raw_infer_config = backend_config.get("infer_config_path", backend_config.get("infer_config"))
        if not isinstance(raw_infer_config, str) or not raw_infer_config.strip():
            raise ManifestValidationError(
                "NV-Generate CT backend requires backend.infer_config_path or backend.infer_config"
            )
        self.infer_config_path = _resolve_config_relative_path(raw_infer_config, base_dir=config_path.parent)

    def generate(self, config: dict[str, JsonValue], output_path: Path) -> dict[str, JsonValue]:
        """Run NV-Generate-CTMR and copy its first generated image to output_path."""
        output_path = output_path.resolve()
        backend_output_dir = (output_path.parent / "maisi_backend_output").resolve()
        backend_output_dir.mkdir(parents=True, exist_ok=True)
        extra_replacements = {
            "backend_output_dir": str(backend_output_dir),
            "infer_config_path": str(self.infer_config_path),
            "random_seed": str(self.random_seed),
            "version": self.version,
        }
        command = [
            _expand_command_part(
                part,
                config_path=self.config_path,
                output_path=output_path,
                config=config,
                extra_replacements=extra_replacements,
            )
            for part in self.command
        ]
        env = {
            key: _expand_command_part(
                value,
                config_path=self.config_path,
                output_path=output_path,
                config=config,
                extra_replacements=extra_replacements,
            )
            for key, value in self.env.items()
        }
        completed = _run_backend_command(
            command,
            cwd=self.config_path.parent,
            timeout_seconds=self.timeout_seconds,
            env=env,
            backend_name="NV-Generate CT backend",
        )

        if completed.returncode != 0:
            detail = _trim_output(completed.stderr) or _trim_output(completed.stdout) or "no command output"
            raise VolumeImportError(f"NV-Generate CT backend exited with {completed.returncode}: {detail}")

        result_payload = _json_payload_from_stdout(completed.stdout)
        image_path = _first_sample_path(result_payload, "image_path") or _find_generated_image(backend_output_dir)
        if image_path is None or not image_path.is_file():
            raise VolumeImportError("NV-Generate CT backend completed but did not report a generated image NIfTI")
        shutil.copy2(image_path, output_path)

        label_path = _first_sample_path(result_payload, "label_path") or _mask_path_from_request(self.infer_config_path)
        copied_label_path: Path | None = None
        if label_path is not None and label_path.is_file():
            copied_label_path = output_path.parent / "label.nii.gz"
            shutil.copy2(label_path, copied_label_path)

        result_path = output_path.parent / "maisi_backend_result.json"
        write_json(result_path, result_payload)
        summary_path = _summary_path(result_payload)
        copied_summary_path: Path | None = None
        if summary_path is not None and summary_path.is_file():
            copied_summary_path = output_path.parent / "maisi_summary.html"
            shutil.copy2(summary_path, copied_summary_path)

        metadata: dict[str, JsonValue] = {
            "command": cast(JsonValue, command),
            "returncode": completed.returncode,
            "backend_output_dir": str(backend_output_dir),
            "infer_config_path": str(self.infer_config_path),
            "version": self.version,
            "random_seed": self.random_seed,
            "generated_image_path": str(image_path),
            "backend_result_path": str(result_path),
        }
        if copied_label_path is not None:
            metadata["label_path"] = str(copied_label_path)
            metadata["generated_label_path"] = str(label_path)
        if copied_summary_path is not None:
            metadata["summary_html"] = str(copied_summary_path)
        stderr = _trim_output(completed.stderr)
        if stderr:
            metadata["stderr"] = stderr
        return metadata


def generate_maisi_case(
    config_path: Path,
    output_root: Path,
    *,
    backend: MaisiBackend | None = None,
    overwrite: bool = False,
) -> VolumeCase:
    """Generate a SynAirG volume case through a MAISI backend interface."""
    config = _read_config(config_path)
    raw_case_id = config.get("case_id")
    if not isinstance(raw_case_id, str):
        raise ManifestValidationError("MAISI config must include a non-empty string case_id")
    case_id = validate_case_id(raw_case_id)

    selected_backend = backend or _backend_from_config(config, config_path)
    case_dir = prepare_case_dir(output_root, case_id, overwrite=overwrite)
    ct_path = case_dir / "ct.nii.gz"
    backend_metadata = selected_backend.generate(config, ct_path)
    if not ct_path.exists():
        raise VolumeImportError(f"MAISI backend {selected_backend.name} did not create {ct_path}")

    source_artifacts = [config_path, *_existing_metadata_paths(backend_metadata, ("infer_config_path",))]
    output_artifacts = [
        path
        for path in (
            case_dir / "label.nii.gz",
            case_dir / "maisi_backend_result.json",
            case_dir / "maisi_summary.html",
        )
        if path.is_file()
    ]
    manifest = VolumeManifest(
        case_id=case_id,
        source=VolumeSource(
            source_type="maisi",
            uri=str(config_path),
            metadata={"backend": selected_backend.name, "config": config},
        ),
        metadata={"generator": "maisi"},
    )
    return write_case_sidecars(
        manifest=manifest,
        case_dir=case_dir,
        ct_path=ct_path,
        source_artifacts=source_artifacts,
        volume_metadata={"normalization": "maisi_backend_output", "backend": selected_backend.name, **backend_metadata},
        pipeline="volume_gen.generate_maisi",
        output_artifacts=output_artifacts,
    )


def _read_config(path: Path) -> dict[str, JsonValue]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"{path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(loaded, dict):
        raise ManifestValidationError("MAISI config must be a JSON object")
    return {str(key): _json_value(value) for key, value in loaded.items()}


def _backend_from_config(config: dict[str, JsonValue], config_path: Path) -> MaisiBackend:
    raw_backend = config.get("backend")
    if raw_backend is None:
        return UnavailableMaisiBackend()
    if not isinstance(raw_backend, dict):
        raise ManifestValidationError("MAISI config backend must be a JSON object")
    backend_config = {str(key): _json_value(value) for key, value in raw_backend.items()}
    backend_type = _optional_string(backend_config.get("type"), "command")
    if backend_type in {"nv-generate-ct", "nv-generate-ctmr", "nv_generate_ct"}:
        return NvGenerateCtBackend(config_path, backend_config)
    if backend_type != "command":
        raise BackendUnavailableError(f"Unsupported MAISI backend type: {backend_type}")
    return CommandMaisiBackend(config_path, backend_config)


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise ManifestValidationError("MAISI config must be JSON-serializable")


def _optional_string(value: JsonValue | None, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError("MAISI backend string fields must be non-empty strings")
    return value


def _optional_int(value: JsonValue | None, default: int, field_name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestValidationError(f"{field_name} must be an integer")
    return value


def _command_from_backend_config(backend_config: dict[str, JsonValue], backend_name: str) -> list[str]:
    raw_command = backend_config.get("command")
    if not isinstance(raw_command, list) or not raw_command or not all(isinstance(part, str) for part in raw_command):
        raise ManifestValidationError(f"{backend_name} requires backend.command as a non-empty string list")
    return [str(part) for part in raw_command]


def _timeout_from_backend_config(backend_config: dict[str, JsonValue], *, default: float) -> float:
    raw_timeout = backend_config.get("timeout_seconds", default)
    if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, int | float) or raw_timeout <= 0:
        raise ManifestValidationError("MAISI backend timeout_seconds must be a positive number")
    return float(raw_timeout)


def _env_from_backend_config(backend_config: dict[str, JsonValue]) -> dict[str, str]:
    raw_env = backend_config.get("env", {})
    if raw_env is None:
        return {}
    if not isinstance(raw_env, dict):
        raise ManifestValidationError("MAISI backend env must be an object of string values")
    env: dict[str, str] = {}
    for key, value in raw_env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ManifestValidationError("MAISI backend env must be an object of string values")
        env[key] = value
    return env


def _expand_command_part(
    value: str,
    *,
    config_path: Path,
    output_path: Path,
    config: dict[str, JsonValue],
    extra_replacements: dict[str, str] | None = None,
) -> str:
    raw_case_id = config.get("case_id")
    case_id = raw_case_id if isinstance(raw_case_id, str) else ""
    replacements = {
        "config_path": str(config_path),
        "output_path": str(output_path),
        "case_id": case_id,
        **(extra_replacements or {}),
    }
    expanded = value
    for key, replacement in replacements.items():
        expanded = expanded.replace(f"{{{key}}}", replacement)
    return expanded


def _run_backend_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    env: dict[str, str],
    backend_name: str,
) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env.update(env)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=run_env,
        )
    except FileNotFoundError as exc:
        raise BackendUnavailableError(f"{backend_name} executable was not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VolumeImportError(f"{backend_name} timed out after {timeout_seconds:g} seconds") from exc


def _resolve_config_relative_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _json_payload_from_stdout(stdout: str) -> dict[str, JsonValue]:
    stripped = stdout.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise VolumeImportError("NV-Generate CT backend did not emit a JSON result payload")
    try:
        loaded = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VolumeImportError(f"NV-Generate CT backend emitted invalid JSON: {exc.msg}") from exc
    if not isinstance(loaded, dict):
        raise VolumeImportError("NV-Generate CT backend JSON result payload must be an object")
    return {str(key): _json_value(value) for key, value in loaded.items()}


def _first_sample_path(payload: dict[str, JsonValue], key: str) -> Path | None:
    output = payload.get("output")
    if not isinstance(output, dict):
        return None
    samples = output.get("samples")
    if not isinstance(samples, list):
        return None
    for sample in samples:
        if isinstance(sample, dict):
            path_value = sample.get(key)
            if isinstance(path_value, str) and path_value.strip():
                return Path(path_value)
    return None


def _mask_path_from_request(path: Path) -> Path | None:
    try:
        request = _read_config(path)
    except ManifestValidationError:
        return None
    raw_mask_path = request.get("mask_path")
    if not isinstance(raw_mask_path, str) or not raw_mask_path.strip():
        return None
    return _resolve_config_relative_path(raw_mask_path, base_dir=path.parent)


def _summary_path(payload: dict[str, JsonValue]) -> Path | None:
    output = payload.get("output")
    if not isinstance(output, dict):
        return None
    path_value = output.get("summary_html")
    if isinstance(path_value, str) and path_value.strip():
        return Path(path_value)
    return None


def _find_generated_image(output_dir: Path) -> Path | None:
    for pattern in ("*_image.nii.gz", "*image.nii.gz", "*.nii.gz"):
        for path in sorted(output_dir.glob(pattern)):
            if path.name.endswith("_label.nii.gz") or path.name == "label.nii.gz":
                continue
            return path
    return None


def _existing_metadata_paths(metadata: dict[str, JsonValue], keys: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str):
            path = Path(value)
            if path.is_file():
                paths.append(path)
    return paths


def _trim_output(value: str, *, limit: int = 4000) -> str:
    stripped = value.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[-limit:]
