"""TCIA/NBIA pull workflows for canonical SynAirG CT volume cases."""

from __future__ import annotations

import json
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from synairg_core.volume_manifest import JsonValue, VolumeManifest, VolumeSource, validate_case_id

from volume_gen.artifacts import VolumeCase, prepare_case_dir, utc_timestamp, write_json
from volume_gen.errors import VolumeImportError
from volume_gen.importers import write_case_sidecars
from volume_gen.normalization import candidate_dicom_files, convert_dicom_folder


class TciaClient(Protocol):
    """Client operations required for a TCIA/NBIA series pull."""

    def series_url(self, collection: str) -> str:
        """Build the metadata query URL for a TCIA collection."""

    def image_url(self, series_instance_uid: str) -> str:
        """Build the image download URL for an NBIA series."""

    def list_series(self, collection: str) -> list[dict[str, JsonValue]]:
        """Return series metadata for a TCIA collection."""

    def download_series(self, series_instance_uid: str, destination_zip: Path) -> Path:
        """Download a series archive to destination_zip."""


@dataclass(frozen=True)
class TciaSeriesSelection:
    """A selected TCIA/NBIA series and the metadata used to select it."""

    collection: str
    series_instance_uid: str
    metadata: dict[str, JsonValue]


@dataclass(frozen=True)
class NbiaClient:
    """Small NBIA query and download client for public TCIA collections."""

    base_url: str = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"

    def series_url(self, collection: str) -> str:
        """Build the metadata query URL for a TCIA collection."""
        query = urllib.parse.urlencode({"Collection": collection, "format": "json"})
        return f"{self.base_url.rstrip('/')}/getSeries?{query}"

    def image_url(self, series_instance_uid: str) -> str:
        """Build the image download URL for a TCIA/NBIA series."""
        query = urllib.parse.urlencode({"SeriesInstanceUID": series_instance_uid})
        return f"{self.base_url.rstrip('/')}/getImage?{query}"

    def list_series(self, collection: str) -> list[dict[str, JsonValue]]:
        """Fetch series metadata for a collection from NBIA."""
        request = urllib.request.Request(self.series_url(collection), headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            raise VolumeImportError(f"Failed to fetch TCIA/NBIA series metadata for {collection}: {exc}") from exc
        if not isinstance(payload, list):
            raise VolumeImportError("NBIA returned an unexpected response; expected a JSON list")
        return [_json_object(item) for item in payload]

    def download_series(self, series_instance_uid: str, destination_zip: Path) -> Path:
        """Download a TCIA/NBIA series archive to destination_zip."""
        destination_zip.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(self.image_url(series_instance_uid), headers={"Accept": "application/zip"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response, destination_zip.open("wb") as writer:
                shutil.copyfileobj(response, writer)
        except OSError as exc:
            raise VolumeImportError(f"Failed to download TCIA/NBIA series {series_instance_uid}: {exc}") from exc
        if destination_zip.stat().st_size == 0:
            raise VolumeImportError(f"TCIA/NBIA series {series_instance_uid} downloaded as an empty archive")
        return destination_zip


def pull_tcia_case(
    collection: str,
    output_root: Path,
    *,
    series_instance_uid: str | None = None,
    patient_id: str | None = None,
    study_instance_uid: str | None = None,
    body_part_examined: str | None = None,
    case_id: str | None = None,
    client: TciaClient | None = None,
    overwrite: bool = False,
) -> VolumeCase:
    """Download a TCIA/NBIA CT series and emit a canonical SynAirG volume case."""
    selected_client = client or NbiaClient()
    selection = select_tcia_series(
        collection,
        client=selected_client,
        series_instance_uid=series_instance_uid,
        patient_id=patient_id,
        study_instance_uid=study_instance_uid,
        body_part_examined=body_part_examined,
    )
    selected_case_id = validate_case_id(case_id) if case_id is not None else default_tcia_case_id(
        collection, selection.series_instance_uid
    )
    case_dir = prepare_case_dir(output_root, selected_case_id, overwrite=overwrite)
    source_dir = case_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    series_metadata_path = source_dir / "series_metadata.json"
    source_metadata = _source_metadata(
        collection=collection,
        selection=selection,
        client=selected_client,
        patient_id=patient_id,
        study_instance_uid=study_instance_uid,
        body_part_examined=body_part_examined,
    )
    write_json(series_metadata_path, source_metadata)

    archive_path = source_dir / "tcia_series.zip"
    selected_client.download_series(selection.series_instance_uid, archive_path)
    dicom_dir = source_dir / "dicom"
    extracted_files = _extract_zip(archive_path, dicom_dir)
    if not extracted_files:
        raise VolumeImportError(f"TCIA/NBIA series {selection.series_instance_uid} did not contain files")

    ct_path = case_dir / "ct.nii.gz"
    volume_metadata = convert_dicom_folder(dicom_dir, ct_path)
    volume_metadata.update(
        {
            "normalization": "tcia_dicom_to_nifti",
            "tcia_collection": collection,
            "series_instance_uid": selection.series_instance_uid,
        }
    )
    manifest = VolumeManifest(
        case_id=selected_case_id,
        source=VolumeSource(
            source_type="tcia",
            uri=selected_client.image_url(selection.series_instance_uid),
            metadata=source_metadata,
        ),
        metadata={"collection": collection},
    )
    source_artifacts = [series_metadata_path, archive_path, *candidate_dicom_files(dicom_dir)]
    return write_case_sidecars(
        manifest=manifest,
        case_dir=case_dir,
        ct_path=ct_path,
        source_artifacts=source_artifacts,
        volume_metadata=volume_metadata,
        pipeline="volume_gen.pull_tcia",
    )


def select_tcia_series(
    collection: str,
    *,
    client: TciaClient | None = None,
    series_instance_uid: str | None = None,
    patient_id: str | None = None,
    study_instance_uid: str | None = None,
    body_part_examined: str | None = None,
) -> TciaSeriesSelection:
    """Select a single CT series from a TCIA/NBIA collection."""
    selected_client = client or NbiaClient()
    try:
        series = selected_client.list_series(collection)
    except VolumeImportError:
        if series_instance_uid is None:
            raise
        return TciaSeriesSelection(
            collection=collection,
            series_instance_uid=series_instance_uid,
            metadata={
                "Collection": collection,
                "SeriesInstanceUID": series_instance_uid,
                "metadata_lookup": "unavailable",
            },
        )

    candidates = [
        item
        for item in series
        if _matches_optional_filter(item, "SeriesInstanceUID", series_instance_uid)
        and _matches_optional_filter(item, "PatientID", patient_id)
        and _matches_optional_filter(item, "StudyInstanceUID", study_instance_uid)
        and _matches_optional_body_part(item, body_part_examined)
    ]
    if not candidates:
        raise VolumeImportError(_no_series_message(collection, series_instance_uid, patient_id, study_instance_uid))

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            _metadata_string(item, "PatientID") or "",
            _metadata_string(item, "StudyInstanceUID") or "",
            _metadata_string(item, "SeriesInstanceUID") or "",
        ),
    )
    selected = (
        ordered_candidates[0]
        if series_instance_uid is not None
        else _first_ct_series(collection, ordered_candidates)
    )
    selected_uid = _metadata_string(selected, "SeriesInstanceUID")
    if selected_uid is None:
        raise VolumeImportError("Selected TCIA/NBIA series metadata did not include SeriesInstanceUID")
    selected_modality = _metadata_string(selected, "Modality")
    if selected_modality is not None and selected_modality.upper() != "CT":
        raise VolumeImportError(f"Selected TCIA/NBIA series {selected_uid} has modality {selected_modality}, not CT")
    return TciaSeriesSelection(collection=collection, series_instance_uid=selected_uid, metadata=selected)


def list_tcia_series(
    collection: str,
    *,
    client: TciaClient | None = None,
    patient_id: str | None = None,
    study_instance_uid: str | None = None,
    body_part_examined: str | None = None,
    modality: str | None = "CT",
) -> list[dict[str, JsonValue]]:
    """List TCIA/NBIA series metadata filtered for SynAirG volume workflows."""
    selected_client = client or NbiaClient()
    series = selected_client.list_series(collection)
    candidates = [
        item
        for item in series
        if _matches_optional_filter(item, "PatientID", patient_id)
        and _matches_optional_filter(item, "StudyInstanceUID", study_instance_uid)
        and _matches_optional_body_part(item, body_part_examined)
        and _matches_optional_modality(item, modality)
    ]
    return sorted(
        candidates,
        key=lambda item: (
            _metadata_string(item, "PatientID") or "",
            _metadata_string(item, "StudyInstanceUID") or "",
            _metadata_string(item, "SeriesInstanceUID") or "",
        ),
    )


def write_tcia_stub(collection: str, output: Path, *, client: NbiaClient | None = None) -> Path:
    """Write legacy TCIA integration planning metadata."""
    selected_client = client or NbiaClient()
    output.mkdir(parents=True, exist_ok=True)
    stub_path = output / "tcia_request.json"
    payload: dict[str, JsonValue] = {
        "schema_version": "1.0",
        "collection": collection,
        "status": "integration_stub",
        "generated_at_utc": utc_timestamp(),
        "metadata_url": selected_client.series_url(collection),
        "message": (
            "TCIA/NBIA collection metadata lookup is available through volume_gen.tcia.NbiaClient; "
            "series download and canonical NIfTI emission require an explicit series selection workflow."
        ),
    }
    write_json(stub_path, payload)
    return stub_path


def _json_object(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise VolumeImportError("NBIA returned a non-object item in its series list")
    normalized: dict[str, JsonValue] = {}
    for key, item in value.items():
        if isinstance(key, str):
            normalized[key] = _json_value(item)
    return normalized


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _first_ct_series(collection: str, candidates: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    for item in candidates:
        modality = _metadata_string(item, "Modality")
        if modality is None or modality.upper() == "CT":
            return item
    raise VolumeImportError(f"No CT series were found in TCIA/NBIA collection {collection}")


def _matches_optional_filter(metadata: dict[str, JsonValue], key: str, expected: str | None) -> bool:
    if expected is None:
        return True
    return _metadata_string(metadata, key) == expected


def _matches_optional_body_part(metadata: dict[str, JsonValue], expected: str | None) -> bool:
    if expected is None:
        return True
    for key in ("BodyPartExamined", "BodyPart", "BodyPartDescription"):
        value = _metadata_string(metadata, key)
        if value is not None and value.upper() == expected.upper():
            return True
    return False


def _matches_optional_modality(metadata: dict[str, JsonValue], expected: str | None) -> bool:
    if expected is None:
        return True
    value = _metadata_string(metadata, "Modality")
    return value is None or value.upper() == expected.upper()


def _metadata_string(metadata: dict[str, JsonValue], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str):
        return value
    normalized_key = key.lower()
    for item_key, item_value in metadata.items():
        if item_key.lower() == normalized_key and isinstance(item_value, str):
            return item_value
    return None


def _no_series_message(
    collection: str,
    series_instance_uid: str | None,
    patient_id: str | None,
    study_instance_uid: str | None,
) -> str:
    filters = {
        "SeriesInstanceUID": series_instance_uid,
        "PatientID": patient_id,
        "StudyInstanceUID": study_instance_uid,
    }
    active_filters = [f"{name}={value}" for name, value in filters.items() if value is not None]
    if not active_filters:
        return f"No series were found in TCIA/NBIA collection {collection}"
    return f"No TCIA/NBIA series matched collection {collection} with {', '.join(active_filters)}"


def _source_metadata(
    *,
    collection: str,
    selection: TciaSeriesSelection,
    client: TciaClient,
    patient_id: str | None,
    study_instance_uid: str | None,
    body_part_examined: str | None,
) -> dict[str, JsonValue]:
    metadata: dict[str, JsonValue] = {
        "collection": collection,
        "series_instance_uid": selection.series_instance_uid,
        "series_metadata": selection.metadata,
        "metadata_url": client.series_url(collection),
        "download_url": client.image_url(selection.series_instance_uid),
    }
    if patient_id is not None:
        metadata["patient_id"] = patient_id
    if study_instance_uid is not None:
        metadata["study_instance_uid"] = study_instance_uid
    if body_part_examined is not None:
        metadata["body_part_examined"] = body_part_examined
    return metadata


def _extract_zip(source_zip: Path, destination_dir: Path) -> list[Path]:
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination_dir.resolve()
    extracted_files: list[Path] = []
    try:
        with zipfile.ZipFile(source_zip) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                target = destination_dir / member.filename
                try:
                    target.resolve().relative_to(resolved_destination)
                except ValueError as exc:
                    raise VolumeImportError(f"TCIA/NBIA archive contains unsafe path {member.filename}") from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as reader, target.open("wb") as writer:
                    shutil.copyfileobj(reader, writer)
                extracted_files.append(target)
    except zipfile.BadZipFile as exc:
        raise VolumeImportError(f"{source_zip} is not a readable TCIA/NBIA zip archive") from exc
    return extracted_files


def default_tcia_case_id(collection: str, series_instance_uid: str) -> str:
    """Create a safe default case ID for a TCIA/NBIA series."""
    collection_part = _safe_case_part(collection) or "collection"
    series_part = _safe_case_part(series_instance_uid) or "series"
    return validate_case_id(f"tcia-{collection_part}-{series_part}")


def _safe_case_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-_")
