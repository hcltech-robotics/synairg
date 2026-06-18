"""Schemas and validation helpers for SynAirG CT volume manifests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias, cast

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SOURCE_TYPES = {"local_nifti", "local_dicom", "tcia", "maisi"}


class ManifestValidationError(ValueError):
    """Raised when a volume manifest does not match the SynAirG schema."""


@dataclass(frozen=True)
class VolumeSource:
    """Source descriptor for a CT volume."""

    source_type: str
    uri: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: object) -> VolumeSource:
        """Build a validated source descriptor from a JSON object."""
        mapping = _require_mapping(value, "source")
        raw_source_type = mapping.get("type", mapping.get("source_type"))
        source_type = _require_string(raw_source_type, "source.type")
        if source_type not in _SOURCE_TYPES:
            expected = ", ".join(sorted(_SOURCE_TYPES))
            raise ManifestValidationError(f"source.type must be one of: {expected}")

        raw_uri = mapping.get("uri", mapping.get("path"))
        uri = None if raw_uri is None else _require_string(raw_uri, "source.uri")
        raw_metadata = mapping.get("metadata", {})
        metadata = _require_json_mapping(raw_metadata, "source.metadata")

        return cls(source_type=source_type, uri=uri, metadata=metadata)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the source descriptor to a JSON-compatible mapping."""
        payload: dict[str, JsonValue] = {
            "type": self.source_type,
            "metadata": self.metadata,
        }
        if self.uri is not None:
            payload["uri"] = self.uri
        return payload


@dataclass(frozen=True)
class VolumeManifest:
    """Input manifest for local or generated CT volume cases."""

    case_id: str
    source: VolumeSource
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    schema_version: str = "1.0"

    @classmethod
    def from_mapping(cls, value: object) -> VolumeManifest:
        """Build a validated manifest from a JSON object."""
        mapping = _require_mapping(value, "manifest")
        case_id = _require_string(mapping.get("case_id"), "case_id")
        if _CASE_ID_PATTERN.fullmatch(case_id) is None:
            raise ManifestValidationError(
                "case_id must start with an alphanumeric character and contain only letters, numbers, dots, "
                "underscores, or hyphens"
            )

        raw_source = mapping.get("source")
        if raw_source is None:
            raw_source = {
                "type": mapping.get("source_type", "local_nifti"),
                "uri": mapping.get("input_path", mapping.get("path")),
            }

        raw_schema_version = mapping.get("schema_version", "1.0")
        schema_version = _require_string(raw_schema_version, "schema_version")
        metadata = _require_json_mapping(mapping.get("metadata", {}), "metadata")

        return cls(
            case_id=case_id,
            source=VolumeSource.from_mapping(raw_source),
            metadata=metadata,
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the manifest to a JSON-compatible mapping."""
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "source": self.source.to_dict(),
            "metadata": self.metadata,
        }


def read_volume_manifest(path: Path) -> VolumeManifest:
    """Read and validate a JSON volume manifest from disk."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"{path} is not valid JSON: {exc.msg}") from exc
    return VolumeManifest.from_mapping(loaded)


def validate_case_id(case_id: str) -> str:
    """Validate a case identifier outside of manifest parsing."""
    if _CASE_ID_PATTERN.fullmatch(case_id) is None:
        raise ManifestValidationError(
            "case_id must start with an alphanumeric character and contain only letters, numbers, dots, "
            "underscores, or hyphens"
        )
    return case_id


def _require_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{field_name} must be an object")
    return cast(dict[str, object], value)


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_json_mapping(value: object, field_name: str) -> dict[str, JsonValue]:
    mapping = _require_mapping(value, field_name)
    validated: dict[str, JsonValue] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise ManifestValidationError(f"{field_name} keys must be strings")
        validated[key] = _validate_json_value(item, f"{field_name}.{key}")
    return validated


def _validate_json_value(value: object, field_name: str) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_validate_json_value(item, f"{field_name}[]") for item in value]
    if isinstance(value, dict):
        validated: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ManifestValidationError(f"{field_name} keys must be strings")
            validated[key] = _validate_json_value(item, f"{field_name}.{key}")
        return validated
    raise ManifestValidationError(f"{field_name} must be JSON-serializable")
