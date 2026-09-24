#!/usr/bin/env python3
"""
1000lungs paper statistics and figures, streaming edition.

Computes the Section V characterisation numbers and renders the two figures
for the SurgTwin 4-pager without materialising the full dataset. Total
transfer for a full-population run is roughly 180 MB against a corpus whose
naive footprint is ~200 GB, because we only ever touch:

  * meta/episodes.jsonl, meta/info.json          (tiny, fetched once)
  * metadata/cases/<id>/case_description.json    (~16 KB x N cases)
  * data/chunk-*/episode_*.parquet               (column-projected reads of
                                                  action + timestamp only)
  * single video frames via ffmpeg HTTP range    (a few MB total, no full
                                                  MP4 downloads)

It deliberately never touches ct.nii.gz, composite.mp4 or the mesh geometry.

Modes:
  * streaming (default): reads via https://huggingface.co/.../resolve/main
  * local: pass --local-root pointing at a full or partial local copy
    (e.g. the original synairg run tree on the Spark); anything found
    locally is read from disk, anything missing falls back to streaming.

Outputs (into --outdir):
  * stats.json            every computed number, for reproducibility
  * table_values.tex      \\newcommand macros the LaTeX pulls in directly
  * fig_characterisation.pdf   four-panel population characterisation
  * fig_montage.png       cases x modalities frame grid (depth/normal/PPS)

Usage on the Spark, full population:
  python spark_stats.py --outdir figures

Fast smoke test:
  python spark_stats.py --cases 60 --parquet-sample 16 --outdir figures

Author: generated for Chris von Csefalvay, 15 Aug 2026.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("pip install requests")

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    sys.exit("pip install pyarrow")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DEFAULT = "chrisvoncsefalvay/1000lungs"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

# A restrained, colourblind-safe pair. Teal for populations, amber for accents.
C_MAIN = "#2A7F8E"
C_ACCENT = "#D08C2E"
C_GRID = "#D9D9D9"

MODALITIES = [
    ("observation.images.endoscope.depth", "Depth"),
    ("observation.images.endoscope.normal", "Normals"),
    ("observation.images.endoscope.pps", "Per-pixel shading"),
]


# --------------------------------------------------------------------------
# Fetch layer: local-first, then streaming, with a tiny on-disk cache.
# --------------------------------------------------------------------------
class Fetcher:
    def __init__(self, repo: str, local_root: str | None, cache_dir: Path,
                 timeout: float = 60.0, retries: int = 3):
        self.repo = repo
        self.local_root = Path(local_root) if local_root else None
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "synairg-paper-stats/1.0"
        tok = os.environ.get("HF_TOKEN")
        if tok:
            self.session.headers["Authorization"] = f"Bearer {tok}"
        self.bytes_fetched = 0

    def url(self, path: str) -> str:
        return RESOLVE.format(repo=self.repo, path=path)

    def get_bytes(self, path: str, use_cache: bool = True) -> bytes:
        if self.local_root:
            p = self.local_root / path
            if p.exists():
                return p.read_bytes()
        cpath = self.cache_dir / path.replace("/", "__")
        if use_cache and cpath.exists():
            return cpath.read_bytes()
        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(self.url(path), timeout=self.timeout)
                r.raise_for_status()
                data = r.content
                self.bytes_fetched += len(data)
                if use_cache:
                    cpath.write_bytes(data)
                return data
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"failed to fetch {path}: {last}")

    def get_json(self, path: str):
        return json.loads(self.get_bytes(path))

    def get_jsonl(self, path: str):
        return [json.loads(l) for l in
                self.get_bytes(path).decode().splitlines() if l.strip()]


# --------------------------------------------------------------------------
# Quaternion helpers for angular-rate statistics.
# --------------------------------------------------------------------------
def quat_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


# --------------------------------------------------------------------------
# Stage 1: metadata population (episodes.jsonl + case_description.json).
# --------------------------------------------------------------------------
def collect_case_stats(fetch: Fetcher, episodes: list[dict], n_cases: int,
                       workers: int, rng: random.Random) -> list[dict]:
    picked = episodes if n_cases >= len(episodes) else \
        rng.sample(episodes, n_cases)

    def one(ep: dict) -> dict | None:
        mdir = ep.get("metadata_dir") or f"metadata/cases/{ep['case_id']}"
        try:
            d = fetch.get_json(f"{mdir}/case_description.json")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! case {ep['episode_index']}: {exc}", file=sys.stderr)
            return None
        q = d.get("quality", {})
        cl = q.get("mesh", {}).get("centerline", {})
        mesh = q.get("mesh", {}).get("mesh", {})
        checks = q.get("mesh", {}).get("checks", {})
        aw = q.get("airway", {})
        pathq = q.get("path", {})
        ori = d.get("artifacts", {}).get("ct_orientation", {})
        spacing = ori.get("spacing_mm") or [0.747, 0.747, 1.2218]
        voxvol = float(np.prod(spacing))
        return {
            "episode_index": ep["episode_index"],
            "frame_count": ep.get("length") or ep.get("frame_count"),
            "branch_count": cl.get("branch_count"),
            "max_generation": cl.get("max_generation"),
            "terminal_nodes": cl.get("terminal_node_count"),
            "centerline_mm": cl.get("total_centerline_length_mm"),
            "max_root_dist_mm": cl.get("max_distance_from_root_mm"),
            "radius_p50_mm": cl.get("radius_p50_mm"),
            "radius_p05_mm": cl.get("radius_p05_mm"),
            "accessible_fraction": cl.get("accessible_node_fraction"),
            "airway_voxels": aw.get("airway_voxels"),
            "airway_ml": (aw.get("airway_voxels") or 0) * voxvol / 1000.0,
            "faces": mesh.get("face_count"),
            "checks_all": bool(checks) and all(checks.values()),
            "path_mm": pathq.get("length_mm"),
            "path_s": pathq.get("duration_s"),
            "min_lumen_mm": pathq.get("minimum_lumen_diameter_mm"),
            "phases": len(pathq.get("procedure_phases", []) or []),
        }

    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, ep) for ep in picked]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                out.append(r)
            if i % 100 == 0 or i == len(futs):
                print(f"  case descriptions: {i}/{len(futs)}")
    return out


# --------------------------------------------------------------------------
# Stage 1b: anatomy fingerprinting, distinct-subset selection, audit.
# --------------------------------------------------------------------------
def fingerprint(case: dict) -> tuple:
    """Anatomy+trajectory fingerprint. Two episodes sharing this tuple are,
    for all practical purposes, the same rendered world."""
    return (case["branch_count"], round(case["centerline_mm"] or 0, 1),
            case["airway_voxels"], round(case["path_mm"] or 0, 1),
            case["frame_count"])


def select_distinct_first(cases: list[dict], n_first: int) -> list[dict]:
    """Keep the first occurrence of each fingerprint among episodes with
    index < n_first, in episode order."""
    seen: set = set()
    out: list[dict] = []
    for c in sorted(cases, key=lambda c: c["episode_index"]):
        if c["episode_index"] >= n_first:
            continue
        fp = fingerprint(c)
        if fp not in seen:
            seen.add(fp)
            out.append(c)
    return out


def make_audit_figure(all_cases: list[dict], outdir: Path) -> dict:
    """Replica multiplicity across the full corpus: the mode-collapse
    detector. One bar per unique world, height = number of episodes
    sharing its fingerprint."""
    from collections import Counter
    mult = Counter(fingerprint(c) for c in all_cases)
    heights = sorted(mult.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(3.45, 1.55))
    ax.bar(range(1, len(heights) + 1), heights, color=C_MAIN, width=0.85)
    ax.axhline(1, color=C_ACCENT, lw=1.0, ls="--")
    ax.text(len(heights), 1.6, "unique", fontsize=7, color=C_ACCENT,
            ha="right")
    ax.set_xlabel(f"Unique worlds (n={len(heights)}), sorted", fontsize=8)
    ax.set_ylabel("Episodes sharing\nfingerprint", fontsize=8)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.96, bottom=0.28)
    out = outdir / "fig_audit.pdf"
    fig.savefig(out)
    plt.close(fig)
    dup = sum(v - 1 for v in heights if v > 1)
    return {"unique_worlds": len(heights), "duplicate_episodes": dup,
            "max_multiplicity": max(heights), "figure": str(out)}



def collect_kinematics(fetch: Fetcher, info: dict, episodes: list[dict],
                       n_sample: int, workers: int,
                       rng: random.Random) -> list[dict]:
    tmpl = info.get("data_path",
                    "data/chunk-{episode_chunk:03d}/"
                    "episode_{episode_index:06d}.parquet")
    picked = episodes if n_sample >= len(episodes) else \
        rng.sample(episodes, n_sample)
    fps = 20.0
    for feat in info.get("features", {}).values():
        v = feat.get("info", {}).get("video.fps")
        if v:
            fps = float(v)

    def one(ep: dict) -> dict | None:
        path = tmpl.format(episode_chunk=ep.get("data/chunk_index", 0),
                           episode_index=ep["episode_index"])
        try:
            data = fetch.get_bytes(path, use_cache=False)
            tbl = pq.read_table(io.BytesIO(data), columns=["action"])
        except Exception as exc:  # noqa: BLE001
            print(f"  ! parquet {ep['episode_index']}: {exc}",
                  file=sys.stderr)
            return None
        act = np.stack(tbl.column("action").to_numpy(zero_copy_only=False))
        pos, quat, depth = act[:, 0:3], act[:, 3:7], act[:, 7]
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        speed_mm_s = step * fps * 1000.0
        ang = quat_geodesic_deg(quat[:-1], quat[1:]) * fps
        return {
            "episode_index": ep["episode_index"],
            "path_len_mm": float(step.sum() * 1000.0),
            "speed_med_mm_s": float(np.median(speed_mm_s)),
            "speed_p95_mm_s": float(np.percentile(speed_mm_s, 95)),
            "ang_med_deg_s": float(np.median(ang)),
            "ang_p95_deg_s": float(np.percentile(ang, 95)),
            "insertion_max_mm": float(depth.max() * 1000.0),
            "retraction_frac": float((np.diff(depth) < -1e-6).mean()),
        }

    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, ep) for ep in picked]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                out.append(r)
            if i % 20 == 0 or i == len(futs):
                print(f"  parquet kinematics: {i}/{len(futs)}")
    return out


# --------------------------------------------------------------------------
# Stage 3: montage frames via ffmpeg HTTP range requests (no full MP4s).
# --------------------------------------------------------------------------
def grab_frame(fetch: Fetcher, info: dict, ep: dict, key: str,
               frac: float, out_png: Path) -> bool:
    tmpl = info.get("video_path",
                    "videos/chunk-{episode_chunk:03d}/{video_key}/"
                    "episode_{episode_index:06d}.mp4")
    rel = tmpl.format(
        episode_chunk=ep.get(f"videos/{key}/chunk_index", 0),
        video_key=key, episode_index=ep["episode_index"])
    fps = 20.0
    t = max(0.0, frac * (ep.get("length", 1200) / fps))
    if fetch.local_root and (fetch.local_root / rel).exists():
        src = str(fetch.local_root / rel)
    else:
        src = fetch.url(rel)
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", src,
           "-frames:v", "1", "-q:v", "2", "-y", str(out_png)]
    try:
        subprocess.run(cmd, check=True, timeout=120)
        return out_png.exists()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! frame {ep['episode_index']}/{key}: {exc}",
              file=sys.stderr)
        return False


def make_montage(fetch: Fetcher, info: dict, episodes: list[dict],
                 ep_indices: list[int], frac: float, outdir: Path,
                 n_rows: int = 3) -> Path:
    """ep_indices is an ordered candidate queue; rows that 404 on the hub
    are skipped and the next candidate is tried until n_rows are filled."""
    by_idx = {e["episode_index"]: e for e in episodes}
    queue = [by_idx[i] for i in ep_indices if i in by_idx]
    tmp = outdir / "_frames"
    tmp.mkdir(exist_ok=True)
    grid: list[list[Path | None]] = []
    rows: list[dict] = []
    used: set = set()
    for ep in queue:
        if len(rows) >= n_rows:
            break
        if ep["episode_index"] in used:
            continue
        used.add(ep["episode_index"])
        row: list[Path | None] = []
        for key, _label in MODALITIES:
            png = tmp / f"e{ep['episode_index']:06d}_{key.split('.')[-1]}.png"
            ok = png.exists() or grab_frame(fetch, info, ep, key, frac, png)
            row.append(png if ok else None)
        if all(r is not None for r in row):
            grid.append(row)
            rows.append(ep)
        else:
            print(f"  montage: skipping episode {ep['episode_index']} "
                  f"(assets unavailable)", file=sys.stderr)

    n_r, n_c = len(grid), len(MODALITIES)
    fig, axes = plt.subplots(n_r, n_c, figsize=(7.16, 1.82 * n_r))
    axes = np.atleast_2d(axes)
    for r, ep in enumerate(rows):
        for c, (key, label) in enumerate(MODALITIES):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if grid[r][c] is not None:
                ax.imshow(plt.imread(grid[r][c]))
            if r == 0:
                ax.set_title(label, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"Case {ep['episode_index']:04d}", fontsize=8)
    fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.04, right=0.995,
                        top=0.93, bottom=0.01)
    out = outdir / "fig_montage.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Stage 4: characterisation figure and macro emission.
# --------------------------------------------------------------------------
def _panel(ax, data, xlabel, colour=C_MAIN, bins=32, med_fmt="{:.0f}"):
    data = np.asarray([d for d in data if d is not None and np.isfinite(d)])
    bins = min(bins, max(6, len(data) // 3))
    ax.hist(data, bins=bins, color=colour, edgecolor="white",
            linewidth=0.3)
    med = float(np.median(data))
    ax.axvline(med, color=C_ACCENT, lw=1.2)
    ax.text(0.97, 0.92, "med " + med_fmt.format(med),
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=C_ACCENT)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", color=C_GRID, lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return data


def make_characterisation(cases: list[dict], kin: list[dict],
                          outdir: Path) -> dict:
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.75))
    stats: dict = {}

    d = _panel(axes[0], [c["branch_count"] for c in cases],
               "Branches per tree")
    stats["branch"] = dict(median=float(np.median(d)),
                           p05=float(np.percentile(d, 5)),
                           p95=float(np.percentile(d, 95)))
    d = _panel(axes[1], [c["centerline_mm"] for c in cases],
               "Centreline length (mm)")
    stats["centerline_mm"] = dict(median=float(np.median(d)),
                                  p05=float(np.percentile(d, 5)),
                                  p95=float(np.percentile(d, 95)))
    d = _panel(axes[2], [c["path_mm"] for c in cases],
               "Procedure path (mm)")
    stats["path_mm"] = dict(median=float(np.median(d)),
                            p05=float(np.percentile(d, 5)),
                            p95=float(np.percentile(d, 95)))
    d = _panel(axes[3], [k["speed_med_mm_s"] for k in kin],
               "Median tip speed (mm/s)", med_fmt="{:.1f}")
    stats["speed_mm_s"] = dict(median=float(np.median(d)),
                               p05=float(np.percentile(d, 5)),
                               p95=float(np.percentile(d, 95)))
    for ax, lab in zip(axes, "abcd"):
        ax.set_title(f"({lab})", fontsize=8, loc="left")
    fig.subplots_adjust(left=0.045, right=0.995, top=0.86, bottom=0.26,
                        wspace=0.28)
    out = outdir / "fig_characterisation.pdf"
    fig.savefig(out)
    plt.close(fig)
    stats["figure"] = str(out)
    return stats


def emit(outdir: Path, payload: dict, macros: dict) -> None:
    (outdir / "stats.json").write_text(json.dumps(payload, indent=1))
    lines = ["% auto-generated by spark_stats.py, do not edit by hand"]
    for k, v in macros.items():
        lines.append(f"\\newcommand{{\\DS{k}}}{{{v}}}")
    (outdir / "table_values.tex").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--local-root", default=None,
                    help="optional local dataset root; missing files "
                         "fall back to streaming")
    ap.add_argument("--cases", type=int, default=100000,
                    help="number of case_description.json to sample "
                         "(default: all)")
    ap.add_argument("--parquet-sample", type=int, default=128,
                    help="episodes to sample for kinematics")
    ap.add_argument("--montage-episodes", default="auto",
                    help="'auto' or comma-separated episode indices")
    ap.add_argument("--montage-frac", type=float, default=0.35,
                    help="fraction through each episode for the frame grab")
    ap.add_argument("--distinct-first", type=int, default=0,
                    help="restrict analysis to genuinely distinct worlds "
                         "among the first N episodes (0 = off)")
    ap.add_argument("--audit-full", action="store_true",
                    help="also fingerprint the entire corpus and emit the "
                         "replica multiplicity audit figure")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--cache-dir", default=".stats_cache")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("WARNING: ffmpeg not found; montage will be skipped",
              file=sys.stderr)

    rng = random.Random(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fetch = Fetcher(args.repo, args.local_root, Path(args.cache_dir))

    print("[1/5] meta")
    info = fetch.get_json("meta/info.json")
    episodes = fetch.get_jsonl("meta/episodes.jsonl")
    manifest = fetch.get_json("metadata/bulk_generation_manifest.json")
    gates = (manifest.get("generation", {}) or {}).get("quality_gates", {})
    failures = manifest.get("failures", [])
    fps = 20.0
    lengths = np.array([e["length"] for e in episodes], float)

    print("[2/5] case descriptions")
    cases = collect_case_stats(fetch, episodes, args.cases, args.workers,
                               rng)

    audit = None
    if args.audit_full:
        audit = make_audit_figure(cases, outdir)
        print(f"  audit: {audit['unique_worlds']} unique worlds, "
              f"{audit['duplicate_episodes']} duplicate episodes")

    if args.distinct_first:
        cases = select_distinct_first(cases, args.distinct_first)
        keep = {c["episode_index"] for c in cases}
        episodes_an = [e for e in episodes if e["episode_index"] in keep]
        print(f"  distinct-first {args.distinct_first}: "
              f"{len(cases)} unique worlds retained")
    else:
        episodes_an = episodes

    print("[3/5] parquet kinematics")
    kin = collect_kinematics(fetch, info, episodes_an,
                             min(args.parquet_sample, len(episodes_an)),
                             args.workers, rng)

    print("[4/5] montage")
    by_idx = {c["episode_index"]: c for c in cases}
    if args.montage_episodes == "auto":
        ranked = sorted(by_idx, key=lambda i: by_idx[i]["centerline_mm"]
                        or 0)
        base = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
        fallback = []
        for b in base:
            j = ranked.index(b)
            for off in (1, -1, 2, -2):
                fallback.append(ranked[min(max(j + off, 0),
                                           len(ranked) - 1)])
        picks = list(dict.fromkeys(base + fallback))  # ordered, unique
    else:
        picks = [int(x) for x in args.montage_episodes.split(",")]
    montage = None
    if shutil.which("ffmpeg"):
        montage = make_montage(fetch, info, episodes_an, picks,
                               args.montage_frac, outdir)

    print("[5/5] characterisation figure + stats")
    charst = make_characterisation(cases, kin, outdir)

    def med(key, arr=cases, scale=1.0):
        v = [c[key] for c in arr if c.get(key) is not None]
        return float(np.median(v)) * scale if v else float("nan")

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": args.repo,
        "episodes": len(episodes),
        "cases_sampled": len(cases),
        "parquet_sampled": len(kin),
        "fps": fps,
        "frames_total": int(lengths.sum()),
        "hours_total": float(lengths.sum() / fps / 3600.0),
        "duration_s": dict(median=float(np.median(lengths) / fps),
                           p05=float(np.percentile(lengths, 5) / fps),
                           p95=float(np.percentile(lengths, 95) / fps)),
        "quality_gates": gates,
        "failures_logged": len(failures),
        "checks_all_pass_rate": float(np.mean([c["checks_all"]
                                               for c in cases])),
        "medians": {
            "branch_count": med("branch_count"),
            "max_generation": med("max_generation"),
            "terminal_nodes": med("terminal_nodes"),
            "centerline_mm": med("centerline_mm"),
            "airway_ml": med("airway_ml"),
            "path_mm": med("path_mm"),
            "min_lumen_mm": med("min_lumen_mm"),
            "radius_p50_mm": med("radius_p50_mm"),
            "phases": med("phases"),
            "speed_med_mm_s": med("speed_med_mm_s", kin),
            "ang_med_deg_s": med("ang_med_deg_s", kin),
            "insertion_max_mm": med("insertion_max_mm", kin),
            "retraction_frac": med("retraction_frac", kin),
        },
        "characterisation": charst,
        "audit": audit,
        "distinct_first": args.distinct_first,
        "montage_episodes": picks,
        "montage": str(montage) if montage else None,
        "bytes_fetched_mb": round(fetch.bytes_fetched / 1e6, 1),
    }
    m = payload["medians"]
    macros = {
        "episodes": len(episodes),
        "framestotal": f"{int(lengths.sum()):,}".replace(",", "\\,"),
        "hours": f"{payload['hours_total']:.1f}",
        "fps": int(fps),
        "durmed": f"{payload['duration_s']['median']:.0f}",
        "branchmed": f"{m['branch_count']:.0f}",
        "genmed": f"{m['max_generation']:.0f}",
        "termmed": f"{m['terminal_nodes']:.0f}",
        "centerlinemed": f"{m['centerline_mm']/10:.0f}",  # cm
        "airwaymlmed": f"{m['airway_ml']:.0f}",
        "pathmed": f"{m['path_mm']:.0f}",
        "lumenmed": f"{m['min_lumen_mm']:.1f}",
        "speedmed": f"{m['speed_med_mm_s']:.0f}",
        "angmed": f"{m['ang_med_deg_s']:.0f}",
        "insertmed": f"{m['insertion_max_mm']:.0f}",
        "passrate": f"{100*payload['checks_all_pass_rate']:.1f}",
        "casessampled": len(cases),
        "parquetsampled": len(kin),
    }
    if args.distinct_first:
        macros["distinctN"] = len(cases)
        macros["distinctfrom"] = args.distinct_first
        macros["branchlo"] = f"{min(c['branch_count'] for c in cases):.0f}"
        macros["branchhi"] = f"{max(c['branch_count'] for c in cases):.0f}"
        macros["genlo"] = f"{min(c['max_generation'] for c in cases):.0f}"
        macros["genhi"] = f"{max(c['max_generation'] for c in cases):.0f}"
        macros["clmin"] = f"{min(c['centerline_mm'] for c in cases)/1000:.1f}"
        macros["clmax"] = f"{max(c['centerline_mm'] for c in cases)/1000:.1f}"
    if audit:
        macros["uniqueworlds"] = audit["unique_worlds"]
        macros["dupepisodes"] = audit["duplicate_episodes"]
        macros["maxmult"] = audit["max_multiplicity"]
    emit(outdir, payload, macros)

    print("\n==== summary ====")
    for k, v in macros.items():
        print(f"  \\DS{k:<16} {v}")
    print(f"  transferred      {payload['bytes_fetched_mb']} MB")
    print(f"  outputs in       {outdir.resolve()}")


if __name__ == "__main__":
    main()
