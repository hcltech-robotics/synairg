"""Launch the installed NV-Generate-CT RFlow skill wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    import os
    import shutil

    skill_root_value = os.environ.get("SYNAIRG_NV_GENERATE_RFLOW_SKILL_ROOT")
    home = Path.home()
    script_name = _script_name_for_request(sys.argv[1:])
    candidates = []
    if skill_root_value:
        candidates.append(Path(skill_root_value).expanduser().resolve() / "scripts" / script_name)
    candidates.extend(
        [
            home / ".agents" / "skills" / "nv-generate-ct-rflow" / "scripts" / script_name,
            home / ".codex" / "skills" / "nv-generate-ct-rflow" / "scripts" / script_name,
        ]
    )
    script = next((path for path in candidates if path.is_file()), None)
    if script is None:
        checked = ", ".join(str(path) for path in candidates)
        print(f"could not find nv-generate-ct-rflow scripts/{script_name}; checked {checked}", file=sys.stderr)
        return 2
    forwarded_args = sys.argv[1:]
    if script_name == "run_ct_from_mask.py":
        forwarded_args = _strip_version_args(forwarded_args)
    command = [sys.executable, str(script), *forwarded_args]
    completed = _run_with_optional_chest_only_candidates(
        command,
        os.environ,
        json=json,
        shutil=shutil,
        tempfile=tempfile,
    )
    return int(completed.returncode)


def _script_name_for_request(args: list[str]) -> str:
    if not args or args[0] == "default":
        return "run_rflow_ct.py"
    request_path = Path(args[0]).expanduser()
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "run_rflow_ct.py"
    if isinstance(request, dict) and "mask_path" in request:
        return "run_ct_from_mask.py"
    return "run_rflow_ct.py"


def _strip_version_args(args: list[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--version":
            skip_next = index + 1 < len(args)
            continue
        if arg.startswith("--version="):
            continue
        stripped.append(arg)
    return stripped


def _run_with_optional_chest_only_candidates(
    command: list[str],
    environ: dict[str, str],
    *,
    json,
    shutil,
    tempfile,
) -> subprocess.CompletedProcess:
    """Run the upstream wrapper, optionally restricting candidate masks to chest-only cases."""
    if environ.get("SYNAIRG_NV_GENERATE_CHEST_ONLY_CANDIDATES") != "1":
        return subprocess.run(command, check=False)
    upstream_root = Path(environ.get("NV_GENERATE_ROOT", "")).expanduser()
    database_paths = [
        path
        for path in (
            upstream_root / "datasets" / "candidate_masks_flexible_size_and_spacing_4000.json",
            upstream_root / "temp_work_dir" / "datasets" / "candidate_masks_flexible_size_and_spacing_4000.json",
        )
        if path.is_file()
    ]
    if not database_paths:
        checked = ", ".join(
            str(path)
            for path in (
                upstream_root / "datasets" / "candidate_masks_flexible_size_and_spacing_4000.json",
                upstream_root / "temp_work_dir" / "datasets" / "candidate_masks_flexible_size_and_spacing_4000.json",
            )
        )
        print(f"candidate database not found for chest-only filtering; checked {checked}", file=sys.stderr)
        return subprocess.CompletedProcess(command, 2)

    backup_dir = Path(tempfile.mkdtemp(prefix="synairg-nvgen-db-"))
    backups: list[tuple[Path, Path]] = []
    for index, database_path in enumerate(database_paths):
        backup_path = backup_dir / f"{index}-{database_path.name}"
        shutil.copy2(database_path, backup_path)
        backups.append((database_path, backup_path))
    try:
        counts: list[str] = []
        for database_path, _backup_path in backups:
            database = json.loads(database_path.read_text(encoding="utf-8"))
            chest_only = [
                item
                for item in database
                if item.get("top_region_index") == [0, 1, 0, 0]
                and item.get("bottom_region_index") == [0, 1, 0, 0]
            ]
            if not chest_only:
                print(f"candidate database has no chest-only entries: {database_path}", file=sys.stderr)
                return subprocess.CompletedProcess(command, 2)
            database_path.write_text(json.dumps(chest_only), encoding="utf-8")
            counts.append(f"{database_path}: {len(chest_only)}")
        print(
            f"[synairg] restricted NV-Generate candidate databases to chest-only entries: {'; '.join(counts)}",
            file=sys.stderr,
        )
        return subprocess.run(command, check=False)
    finally:
        for database_path, backup_path in backups:
            shutil.copy2(backup_path, database_path)
        shutil.rmtree(backup_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
