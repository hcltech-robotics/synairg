"""Public bronchoscopy dataset download and audit helpers."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
from urllib.request import urlopen

from synairg_core.volume_manifest import JsonValue

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SCHEMA_VERSION = "1.0"


class PublicBronchoscopyDataError(ValueError):
    """Raised when a public bronchoscopy dataset cannot be downloaded or audited."""


@dataclass(frozen=True)
class PublicDatasetFile:
    """Provenance and checksum metadata for one public dataset artifact."""

    name: str
    size_bytes: int
    md5: str
    download_url: str

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize file metadata to a JSON-compatible mapping."""
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "md5": self.md5,
            "download_url": self.download_url,
        }


@dataclass(frozen=True)
class BMBronchoLCDownloadConfig:
    """Configuration for downloading the BM-BronchoLC Figshare artifacts."""

    root: Path
    extract: bool = True
    overwrite: bool = False
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class BMBronchoLCAuditConfig:
    """Configuration for auditing an extracted BM-BronchoLC dataset."""

    root: Path
    sample_image_limit: int = 16


@dataclass(frozen=True)
class BMBronchoLCSplitConfig:
    """Configuration for deterministic patient-level BM-BronchoLC splits."""

    root: Path
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    salt: str = "synairg-bm-broncholc-patient-split-v1"


@dataclass
class _CategoryAudit:
    image_count: int = 0
    patients: set[str] = field(default_factory=set)
    videos: set[str] = field(default_factory=set)
    metadata_files: list[str] = field(default_factory=list)
    sample_dimensions: list[dict[str, JsonValue]] = field(default_factory=list)

    def to_dict(self) -> dict[str, JsonValue]:
        metadata_files: list[JsonValue] = [path for path in sorted(self.metadata_files)]
        sample_dimensions: list[JsonValue] = [sample for sample in self.sample_dimensions]
        return {
            "image_count": self.image_count,
            "patient_count": len(self.patients),
            "video_count": len(self.videos),
            "metadata_files": metadata_files,
            "sample_dimensions": sample_dimensions,
        }


@dataclass(frozen=True)
class _PatientRecord:
    category: str
    patient_id: str
    image_count: int

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "category": self.category,
            "patient_id": self.patient_id,
            "image_count": self.image_count,
        }


BM_BRONCHOLC_ARTICLE_ID = "24243670"
BM_BRONCHOLC_DOI = "10.6084/m9.figshare.24243670.v3"
BM_BRONCHOLC_FIGSHARE_URL = "https://figshare.com/articles/dataset/BM-BronchoLC/24243670"
BM_BRONCHOLC_LICENSE = "CC BY 4.0"
BM_BRONCHOLC_CITATION = "Vu, Van Giap (2023). BM-BronchoLC. figshare. Dataset."
BM_BRONCHOLC_FILES = (
    PublicDatasetFile(
        name="Lung_cancer.zip",
        size_bytes=304_836_076,
        md5="b5be15defd71e5a136ee854e5f3f7b74",
        download_url="https://ndownloader.figshare.com/files/43034527",
    ),
    PublicDatasetFile(
        name="Non_lung_cancer.zip",
        size_bytes=259_951_860,
        md5="25b9ca4b1ff568a95a235f2168b2a693",
        download_url="https://ndownloader.figshare.com/files/43034623",
    ),
)
_METADATA_FILENAMES = {"annotation.json", "labels.json", "objects.json"}


def download_bm_broncholc(config: BMBronchoLCDownloadConfig) -> dict[str, JsonValue]:
    """Download, verify, and optionally extract BM-BronchoLC artifacts."""
    if config.timeout_seconds <= 0:
        raise PublicBronchoscopyDataError("timeout_seconds must be positive")
    raw_dir = config.root / "raw"
    extract_dir = config.root / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if config.extract:
        extract_dir.mkdir(parents=True, exist_ok=True)

    records: list[JsonValue] = []
    for spec in BM_BRONCHOLC_FILES:
        archive_path = raw_dir / spec.name
        action = "present"
        if config.overwrite or not archive_path.exists():
            _download_file(spec.download_url, archive_path, timeout_seconds=config.timeout_seconds)
            action = "downloaded"
        md5 = _md5_file(archive_path)
        size = archive_path.stat().st_size
        if size != spec.size_bytes:
            raise PublicBronchoscopyDataError(
                f"{archive_path} size mismatch: expected {spec.size_bytes}, observed {size}"
            )
        if md5 != spec.md5:
            raise PublicBronchoscopyDataError(f"{archive_path} md5 mismatch: expected {spec.md5}, observed {md5}")
        record: dict[str, JsonValue] = {
            "name": spec.name,
            "path": str(archive_path),
            "action": action,
            "size_bytes": size,
            "md5": md5,
            "verified": True,
        }
        if config.extract:
            _safe_extract_zip(archive_path, extract_dir)
            record["extracted_to"] = str(extract_dir)
        records.append(record)

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "BM-BronchoLC",
        "root": str(config.root),
        "source": _bm_broncholc_source(),
        "archives": records,
    }


def audit_bm_broncholc(config: BMBronchoLCAuditConfig) -> dict[str, JsonValue]:
    """Audit the extracted BM-BronchoLC file structure without consuming labels as training targets."""
    if config.sample_image_limit < 0:
        raise PublicBronchoscopyDataError("sample_image_limit must be non-negative")
    if not config.root.exists():
        raise PublicBronchoscopyDataError(f"dataset root does not exist: {config.root}")
    scan_root = _scan_root(config.root)
    if not scan_root.is_dir():
        raise PublicBronchoscopyDataError(f"dataset scan root is not a directory: {scan_root}")

    categories: dict[str, _CategoryAudit] = {
        "lung_cancer": _CategoryAudit(),
        "non_lung_cancer": _CategoryAudit(),
        "uncategorized": _CategoryAudit(),
    }
    extension_counts: Counter[str] = Counter()

    image_paths = sorted(
        path for path in scan_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for path in image_paths:
        rel_path = path.relative_to(scan_root)
        category, category_index = _category_from_parts(rel_path.parts)
        category_key = category if category is not None else "uncategorized"
        audit = categories.setdefault(category_key, _CategoryAudit())
        audit.image_count += 1
        extension_counts[path.suffix.lower()] += 1
        patient_id, video_id = _patient_video_from_parts(rel_path.parts, category_index)
        if patient_id is not None:
            audit.patients.add(patient_id)
        if patient_id is not None and video_id is not None:
            audit.videos.add(f"{patient_id}/{video_id}")
        if len(audit.sample_dimensions) < config.sample_image_limit:
            dimensions = _read_image_size(path)
            sample: dict[str, JsonValue] = {"path": rel_path.as_posix()}
            if dimensions is not None:
                width, height = dimensions
                sample["width"] = width
                sample["height"] = height
            audit.sample_dimensions.append(sample)

    for path in sorted(scan_root.rglob("*.json")):
        if path.name not in _METADATA_FILENAMES:
            continue
        rel_path = path.relative_to(scan_root)
        category, _category_index = _category_from_parts(rel_path.parts)
        category_key = category if category is not None else "uncategorized"
        categories.setdefault(category_key, _CategoryAudit()).metadata_files.append(rel_path.as_posix())

    archive_records = _archive_records(config.root)
    category_payload: dict[str, JsonValue] = {}
    for name, audit in sorted(categories.items()):
        if audit.image_count or audit.metadata_files or name != "uncategorized":
            category_payload[name] = audit.to_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "BM-BronchoLC",
        "root": str(config.root),
        "scan_root": str(scan_root),
        "source": _bm_broncholc_source(),
        "archives": archive_records,
        "image_summary": {
            "total_images": len(image_paths),
            "extension_counts": dict(sorted(extension_counts.items())),
            "categories": category_payload,
        },
        "research_use": {
            "recommended_role": (
                "real clinical appearance corpus for LoRA/style adaptation and held-out distributional realism metrics"
            ),
            "limitations": [
                "sampled still frames at one frame per second, not dense video",
                "no metric depth, normals, optical flow, or camera pose",
                "labels are diagnostic annotations and must not become paired generation targets",
            ],
            "no_cheating_note": (
                "Use patient/video-level splits. Held-out images may be used for distributional metrics only, "
                "not per-input RGB supervision or input-specific prompt/control tuning."
            ),
        },
    }


def write_bm_broncholc_audit(config: BMBronchoLCAuditConfig, output_path: Path) -> dict[str, JsonValue]:
    """Write a deterministic BM-BronchoLC audit report."""
    report = audit_bm_broncholc(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_bm_broncholc_splits(config: BMBronchoLCSplitConfig) -> dict[str, JsonValue]:
    """Create deterministic patient-level splits for BM-BronchoLC."""
    _validate_split_config(config)
    records = _collect_patient_records(config.root)
    if not records:
        raise PublicBronchoscopyDataError(f"no patient-level images found under {config.root}")

    assignments: dict[str, list[_PatientRecord]] = {"train": [], "validation": [], "test": []}
    for category in sorted({record.category for record in records}):
        category_records = [record for record in records if record.category == category]
        category_records.sort(key=lambda record: _split_key(config.salt, record))
        train_count, validation_count, test_count = _split_counts(
            len(category_records),
            config.train_fraction,
            config.validation_fraction,
            config.test_fraction,
        )
        train_end = train_count
        validation_end = train_end + validation_count
        assignments["train"].extend(category_records[:train_end])
        assignments["validation"].extend(category_records[train_end:validation_end])
        assignments["test"].extend(category_records[validation_end : validation_end + test_count])

    split_payload: dict[str, JsonValue] = {}
    for split_name, split_records in assignments.items():
        split_payload[split_name] = _split_summary(split_records)

    leakage_checks = _leakage_checks(assignments)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "BM-BronchoLC",
        "root": str(config.root),
        "source": _bm_broncholc_source(),
        "split_unit": "category/patient_id",
        "split_policy": {
            "method": "deterministic_sha256_by_category_and_patient",
            "salt": config.salt,
            "train_fraction": config.train_fraction,
            "validation_fraction": config.validation_fraction,
            "test_fraction": config.test_fraction,
        },
        "splits": split_payload,
        "leakage_checks": leakage_checks,
        "no_cheating_note": (
            "Use train for LoRA/style fitting, validation for architecture-level stopping only, and test for final "
            "distributional realism metrics. Do not tune prompts, seeds, control weights, or per-input settings "
            "on test."
        ),
    }


def write_bm_broncholc_splits(config: BMBronchoLCSplitConfig, output_path: Path) -> dict[str, JsonValue]:
    """Write deterministic BM-BronchoLC patient-level splits."""
    report = build_bm_broncholc_splits(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def write_bm_broncholc_download_report(
    config: BMBronchoLCDownloadConfig,
    output_path: Path,
) -> dict[str, JsonValue]:
    """Download BM-BronchoLC and write a deterministic provenance report."""
    report = download_bm_broncholc(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _bm_broncholc_source() -> dict[str, JsonValue]:
    expected_files: list[JsonValue] = [spec.to_dict() for spec in BM_BRONCHOLC_FILES]
    return {
        "figshare_article_id": BM_BRONCHOLC_ARTICLE_ID,
        "doi": BM_BRONCHOLC_DOI,
        "url": BM_BRONCHOLC_FIGSHARE_URL,
        "license": BM_BRONCHOLC_LICENSE,
        "citation": BM_BRONCHOLC_CITATION,
        "expected_files": expected_files,
    }


def _download_file(url: str, destination: Path, timeout_seconds: float) -> None:
    temporary_path = destination.with_name(f"{destination.name}.tmp")
    try:
        with urlopen(url, timeout=timeout_seconds) as response, temporary_path.open("wb") as handle:
            _copy_stream(response, handle)
        temporary_path.replace(destination)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise PublicBronchoscopyDataError(f"could not download {url}: {exc}") from exc


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            return
        destination.write(chunk)


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    base = destination.resolve()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(base):
                    raise PublicBronchoscopyDataError(f"unsafe zip member path: {member.filename}")
            archive.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise PublicBronchoscopyDataError(f"{archive_path} is not a valid zip archive") from exc


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _scan_root(root: Path) -> Path:
    extracted = root / "extracted"
    return extracted if extracted.is_dir() else root


def _archive_records(root: Path) -> list[JsonValue]:
    records: list[JsonValue] = []
    for spec in BM_BRONCHOLC_FILES:
        candidates = [root / "raw" / spec.name, root / spec.name]
        existing = next((path for path in candidates if path.is_file()), None)
        record: dict[str, JsonValue] = {
            "name": spec.name,
            "expected_size_bytes": spec.size_bytes,
            "expected_md5": spec.md5,
            "present": existing is not None,
        }
        if existing is not None:
            record["path"] = str(existing)
            record["size_bytes"] = existing.stat().st_size
            record["md5"] = _md5_file(existing)
            record["verified"] = record["size_bytes"] == spec.size_bytes and record["md5"] == spec.md5
        records.append(record)
    return records


def _collect_patient_records(root: Path) -> list[_PatientRecord]:
    scan_root = _scan_root(root)
    image_counts: Counter[tuple[str, str]] = Counter()
    image_paths = sorted(
        path for path in scan_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for path in image_paths:
        rel_path = path.relative_to(scan_root)
        category, category_index = _category_from_parts(rel_path.parts)
        patient_id, _video_id = _patient_video_from_parts(rel_path.parts, category_index)
        if category is None or patient_id is None:
            continue
        image_counts[(category, patient_id)] += 1
    return [
        _PatientRecord(category=category, patient_id=patient_id, image_count=image_count)
        for (category, patient_id), image_count in sorted(image_counts.items())
    ]


def _validate_split_config(config: BMBronchoLCSplitConfig) -> None:
    fractions = [config.train_fraction, config.validation_fraction, config.test_fraction]
    if any(fraction <= 0.0 for fraction in fractions):
        raise PublicBronchoscopyDataError("split fractions must be positive")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise PublicBronchoscopyDataError("split fractions must sum to 1.0")
    if not config.salt.strip():
        raise PublicBronchoscopyDataError("split salt must be non-empty")
    if not config.root.exists():
        raise PublicBronchoscopyDataError(f"dataset root does not exist: {config.root}")


def _split_counts(
    total: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    if total < 3:
        return total, 0, 0
    validation_count = max(1, round(total * validation_fraction))
    test_count = max(1, round(total * test_fraction))
    train_count = total - validation_count - test_count
    if train_count < 1:
        train_count = 1
        overflow = train_count + validation_count + test_count - total
        if test_count >= validation_count and test_count > 1:
            test_count -= overflow
        elif validation_count > 1:
            validation_count -= overflow
    return train_count, validation_count, test_count


def _split_key(salt: str, record: _PatientRecord) -> str:
    payload = f"{salt}|{record.category}|{record.patient_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _split_summary(records: list[_PatientRecord]) -> dict[str, JsonValue]:
    patient_entries: list[JsonValue] = [
        record.to_dict() for record in sorted(records, key=lambda item: item.patient_id)
    ]
    category_counts: Counter[str] = Counter(record.category for record in records)
    category_image_counts: Counter[str] = Counter()
    for record in records:
        category_image_counts[record.category] += record.image_count
    return {
        "patient_count": len(records),
        "image_count": sum(record.image_count for record in records),
        "category_patient_counts": dict(sorted(category_counts.items())),
        "category_image_counts": dict(sorted(category_image_counts.items())),
        "patients": patient_entries,
    }


def _leakage_checks(assignments: dict[str, list[_PatientRecord]]) -> dict[str, JsonValue]:
    seen: dict[str, str] = {}
    overlaps: list[JsonValue] = []
    for split_name, records in assignments.items():
        for record in records:
            key = f"{record.category}/{record.patient_id}"
            previous_split = seen.get(key)
            if previous_split is not None:
                overlaps.append({"patient": key, "splits": [previous_split, split_name]})
            seen[key] = split_name
    return {
        "patient_overlap": bool(overlaps),
        "overlaps": overlaps,
        "assigned_patient_count": len(seen),
        "assigned_image_count": sum(record.image_count for records in assignments.values() for record in records),
    }


def _category_from_parts(parts: tuple[str, ...]) -> tuple[str | None, int | None]:
    for index, part in enumerate(parts):
        normalized = part.lower().replace("-", "_")
        if normalized == "lung_cancer":
            return "lung_cancer", index
        if normalized in {"non_lung_cancer", "nonlungcancer"}:
            return "non_lung_cancer", index
    return None, None


def _patient_video_from_parts(parts: tuple[str, ...], category_index: int | None) -> tuple[str | None, str | None]:
    start = category_index + 1 if category_index is not None else 0
    remaining = list(parts[start:])
    if remaining and remaining[0].lower() in {"imgs", "images"}:
        remaining = remaining[1:]
    if len(remaining) >= 3:
        return remaining[0], remaining[1]
    if len(remaining) >= 2:
        return remaining[0], "."
    return None, None


def _read_image_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            width = int.from_bytes(header[16:20], "big")
            height = int.from_bytes(header[20:24], "big")
            return width, height
        if header.startswith(b"\xff\xd8"):
            handle.seek(0)
            return _read_jpeg_size(handle)
    return None


def _read_jpeg_size(handle: BinaryIO) -> tuple[int, int] | None:
    if handle.read(2) != b"\xff\xd8":
        return None
    while True:
        marker_start = handle.read(1)
        if not marker_start:
            return None
        if marker_start != b"\xff":
            continue
        marker = handle.read(1)
        while marker == b"\xff":
            marker = handle.read(1)
        if not marker:
            return None
        marker_value = marker[0]
        if marker_value in {0xD8, 0xD9}:
            continue
        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            return None
        length = int.from_bytes(length_bytes, "big")
        if length < 2:
            return None
        if marker_value in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            payload = handle.read(5)
            if len(payload) != 5:
                return None
            height = int.from_bytes(payload[1:3], "big")
            width = int.from_bytes(payload[3:5], "big")
            return width, height
        handle.seek(length - 2, 1)
