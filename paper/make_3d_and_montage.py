#!/usr/bin/env python3
"""
3D world renders + four-column montage for the SurgTwin paper.

For each requested episode this fetches only airway_mask.nii.gz (~0.3 MB)
and the episode parquet (~0.2 MB), reconstructs the luminal surface at full
resolution, decimates it, and renders an anterior 3D view with the procedure
trajectory threaded through, coloured by time. It then recomposes
figures/fig_montage.png as rows x [depth | normals | PPS | 3D world].

Companion to spark_stats.py; shares its streaming Fetcher. Run after it so
the modality frame grabs are already cached.

Usage:
  python make_3d_and_montage.py --episodes 38,3,27 --outdir figures
"""
from __future__ import annotations

import argparse
import gzip
import io
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import nibabel as nib
import pyarrow.parquet as pq
from skimage import measure

import spark_stats as ss

MESH_COLOUR = "#aebac1"
TRAJ_CMAP = "plasma"


def load_mask(fetch: ss.Fetcher, mdir: str):
    raw = fetch.get_bytes(f"{mdir}/airway_mask.nii.gz")
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(raw)
        path = f.name
    img = nib.load(path)
    mask = np.asarray(img.dataobj).astype(np.uint8)
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n > 1:  # keep the largest component, mirroring pipeline repair
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        mask = (lab == (int(np.argmax(sizes)) + 1)).astype(np.uint8)
    return mask, img.affine


def surface_world(mask: np.ndarray, affine: np.ndarray,
                  target_faces: int = 60000):
    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), 0.5)
    if len(faces) > target_faces:
        try:
            import fast_simplification as fs
            verts, faces = fs.simplify(verts.astype(np.float32),
                                       faces.astype(np.int64),
                                       target_count=target_faces)
        except Exception as exc:  # noqa: BLE001
            print(f"  decimation unavailable ({exc}); "
                  f"keeping {len(faces)} faces")
    homo = np.c_[verts, np.ones(len(verts))]
    world = (affine @ homo.T).T[:, :3]
    return world, np.asarray(faces)


def episode_trajectory_mm(fetch: ss.Fetcher, info: dict, ep: dict):
    tmpl = info.get("data_path",
                    "data/chunk-{episode_chunk:03d}/"
                    "episode_{episode_index:06d}.parquet")
    path = tmpl.format(episode_chunk=ep.get("data/chunk_index", 0),
                       episode_index=ep["episode_index"])
    tbl = pq.read_table(io.BytesIO(fetch.get_bytes(path, use_cache=False)),
                        columns=["action"])
    act = np.stack(tbl.column("action").to_numpy(zero_copy_only=False))
    return act[:, 0:3] * 1000.0  # m -> mm


def render_world(fetch: ss.Fetcher, info: dict, ep: dict,
                 out_png: Path) -> Path:
    mdir = ep.get("metadata_dir") or f"metadata/cases/{ep['case_id']}"
    mask, affine = load_mask(fetch, mdir)
    world, faces = surface_world(mask, affine)
    traj = episode_trajectory_mm(fetch, info, ep)

    lo, hi = world.min(0), world.max(0)
    pad = 0.15 * (hi - lo)
    in_frame = np.all((traj > lo - pad) & (traj < hi + pad))

    fig = plt.figure(figsize=(3.2, 2.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(world[:, 0], world[:, 1], world[:, 2],
                    triangles=faces, color=MESH_COLOUR, alpha=0.16,
                    linewidth=0, antialiased=False, shade=True)
    if in_frame:
        seg = np.stack([traj[:-1], traj[1:]], axis=1)
        t = np.linspace(0, 1, len(seg))
        lc = Line3DCollection(seg, cmap=TRAJ_CMAP, linewidths=1.9)
        lc.set_zorder(10)
        lc.set_array(t)
        ax.add_collection3d(lc)
        ax.scatter(*traj[0], color="#1a9641", s=20, depthshade=False, zorder=11)
    else:
        print(f"  ! episode {ep['episode_index']}: trajectory outside "
              f"mesh frame; drawing mesh only")
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=10, azim=-88)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1.06, bottom=-0.06)
    fig.savefig(out_png, dpi=210)
    plt.close(fig)
    print(f"  episode {ep['episode_index']}: {len(faces)} faces, "
          f"traj {'ok' if in_frame else 'skipped'}")
    return out_png


def compose(outdir: Path, ep_indices: list[int]) -> Path:
    cols = [("depth", "Depth"), ("normal", "Normals"),
            ("pps", "Per-pixel shading"), ("world3d", "3D world + path")]
    n_r, n_c = len(ep_indices), len(cols)
    fig, axes = plt.subplots(n_r, n_c, figsize=(7.16, 1.36 * n_r))
    axes = np.atleast_2d(axes)
    for r, idx in enumerate(ep_indices):
        for c, (suffix, label) in enumerate(cols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            p = outdir / "_frames" / f"e{idx:06d}_{suffix}.png"
            if p.exists():
                ax.imshow(plt.imread(p))
            if r == 0:
                ax.set_title(label, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"Case {idx:04d}", fontsize=7)
    fig.subplots_adjust(wspace=0.02, hspace=0.03, left=0.035, right=0.995,
                        top=0.90, bottom=0.01)
    out = outdir / "fig_montage.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=ss.REPO_DEFAULT)
    ap.add_argument("--local-root", default=None)
    ap.add_argument("--episodes", default="38,3,27")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--cache-dir", default=".stats_cache")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "_frames").mkdir(parents=True, exist_ok=True)
    fetch = ss.Fetcher(args.repo, args.local_root, Path(args.cache_dir))
    info = fetch.get_json("meta/info.json")
    episodes = fetch.get_jsonl("meta/episodes.jsonl")
    by_idx = {e["episode_index"]: e for e in episodes}
    idxs = [int(x) for x in args.episodes.split(",")]
    for i in idxs:
        render_world(fetch, info, by_idx[i],
                     outdir / "_frames" / f"e{i:06d}_world3d.png")
    out = compose(outdir, idxs)
    print("montage:", out, f"| transferred {fetch.bytes_fetched/1e6:.1f} MB")


if __name__ == "__main__":
    main()
