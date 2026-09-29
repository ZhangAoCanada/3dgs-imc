#!/usr/bin/env python3
"""Create a bonsai BEV figure with BEV-aligned Ours context.

The context row is not an image-space crop. It rasterizes Ours (GS-LSS) Gaussian
centers/colors into the same deterministic top-down projection and crop used by
bev_grid_overlay.py, so readers can compare the orientation directly.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.ndimage import gaussian_filter

from bev_context_figure import (
    GA,
    SEP,
    VM,
    BINS,
    INTERP,
    NICE,
    ORDER,
    RESIZE_BICUBIC,
    export_bev_panels,
    fnt,
    frac,
    labeled,
    load_bev_data,
    load_manifest,
    manifest_entry,
)

DIAG = Path("/data1/zhangao/outputs/diag/mipnerf360")
BEST_COVERAGE = Path("/data1/zhangao/outputs/best_coverage_gap")
SH_C0 = 0.28209479177387814


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def iteration_num(path):
    match = re.search(r"iteration_(\d+)", str(path))
    return int(match.group(1)) if match else -1


def parse_best_manifest(cell_dir):
    info = {}
    path = cell_dir / "MANIFEST.txt"
    if not path.exists():
        return info
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()
    return info


def point_cloud_candidates(scene, angle, manifest):
    candidates = []

    # Claude memory says this is the authoritative visualization checkpoint collection.
    best_dir = BEST_COVERAGE / f"{scene}_angle_{float(angle):.1f}"
    best_ply = best_dir / "point_cloud.ply"
    if best_ply.exists():
        candidates.append(best_ply)
    best_info = parse_best_manifest(best_dir)
    source_dir = best_info.get("source_dir")
    checkpoint = best_info.get("checkpoint", "").replace("iter_", "")
    if source_dir and checkpoint:
        candidates.append(Path(source_dir) / "point_cloud" / f"iteration_{checkpoint}" / "point_cloud.ply")

    # Fallbacks for cases outside the curated best collection.
    entry = manifest_entry(scene, angle, f"{scene}_{angle}", manifest)
    dirs = []
    manifest_dir = entry.get("ours", {}).get("dir")
    if manifest_dir:
        dirs.append(Path(manifest_dir))
    dirs.extend(sorted(DIAG.glob(f"{scene}_angle_{float(angle):.1f}*")))

    seen_dirs = set()
    for d in dirs:
        if d in seen_dirs:
            continue
        seen_dirs.add(d)
        if not d.exists():
            continue
        pcs = sorted(d.glob("point_cloud/iteration_*/point_cloud.ply"), key=iteration_num, reverse=True)
        candidates.extend(pcs)

    seen_paths = set()
    unique = []
    for path in candidates:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique.append(path)
    return unique


def resolve_point_cloud(scene, angle, manifest):
    candidates = point_cloud_candidates(scene, angle, manifest)
    for path in candidates:
        if path.exists():
            return path, candidates
    tried = "\n  ".join(str(p) for p in candidates) or "<none>"
    raise FileNotFoundError(f"No Ours point_cloud.ply for {scene} {angle} deg. Tried:\n  {tried}")


def gaussian_rgb(vertex):
    names = vertex.data.dtype.names
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        rgb = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
        rgb = np.clip(rgb * SH_C0 + 0.5, 0.0, 1.0)
    return rgb


def rasterize_ours_topdown(scene, angle, ref_crop, ext, center, axes, manifest):
    pc_path, candidates = resolve_point_cloud(scene, angle, manifest)
    ply = PlyData.read(pc_path)
    vertex = ply["vertex"]
    xyz = np.stack([vertex[k] for k in ("x", "y", "z")], axis=1).astype(np.float32)
    xy = (xyz - center) @ axes.T

    h, w = ref_crop.shape
    x0, x1, y0, y1 = ext
    cols = np.floor((xy[:, 0] - x0) / (x1 - x0) * w).astype(np.int64)
    rows = np.floor((xy[:, 1] - y0) / (y1 - y0) * h).astype(np.int64)
    valid = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    if not np.any(valid):
        raise RuntimeError(f"No Ours Gaussians project into BEV crop for {scene} {angle} deg from {pc_path}")

    cols = cols[valid]
    rows = rows[valid]
    flat = rows * w + cols
    rgb = gaussian_rgb(vertex)[valid]
    if "opacity" in vertex.data.dtype.names:
        weights = sigmoid(vertex["opacity"])[valid].astype(np.float32)
    else:
        weights = np.ones(len(flat), dtype=np.float32)

    # Avoid a few ultra-transparent splats making noisy isolated pixels.
    weights = np.clip(weights, 0.0, 1.0)
    size = h * w
    den = np.bincount(flat, weights=weights, minlength=size).reshape(h, w).astype(np.float32)
    sums = []
    for channel in range(3):
        sums.append(np.bincount(flat, weights=weights * rgb[:, channel], minlength=size).reshape(h, w).astype(np.float32))

    sigma = 1.15
    den_s = gaussian_filter(den, sigma=sigma)
    color_s = np.stack([gaussian_filter(s, sigma=sigma) for s in sums], axis=-1)
    color = np.zeros((h, w, 3), dtype=np.float32)
    mask = den_s > 1e-8
    color[mask] = color_s[mask] / den_s[mask, None]

    nonzero = den_s[den_s > 0]
    scale = np.percentile(nonzero, 99.2) if nonzero.size else 1.0
    alpha = np.clip(np.sqrt(den_s / (scale + 1e-8)), 0.0, 1.0)[..., None]

    # White background keeps this explanatory strip readable in print.
    bg = np.ones_like(color) * 0.965
    img = bg * (1.0 - alpha) + color * alpha

    # Add the reference SfM footprint as a subtle cool-gray guide; it is the same footprint used under the BEV heatmaps.
    footprint = np.clip(gaussian_filter(ref_crop.astype(np.float32), sigma=0.45), 0.0, 1.0)[..., None]
    guide = np.array([0.50, 0.53, 0.58], dtype=np.float32)
    img = img * (1.0 - footprint * 0.10) + guide * (footprint * 0.10)
    return np.clip(img, 0.0, 1.0), pc_path, candidates, int(valid.sum()), int(len(xyz))


def labeled_topdown_image(rgb, angle, valid_count, total_count):
    pil = Image.fromarray((rgb[::-1] * 255).astype(np.uint8)).resize((360, 360), RESIZE_BICUBIC)
    pct = 100.0 * valid_count / max(total_count, 1)
    return labeled(pil, f"Ours BEV view @{angle} deg", sub=f"{pct:.0f}% in crop", accent=True)


def export_bev_context(scene, context_rgb, provenance):
    out_dir = SEP / f"{scene}_bevcontext"
    out_dir.mkdir(parents=True, exist_ok=True)
    readme = [
        f"BEV-aligned Ours context exports for BEV_overlay_{scene}_bevcontext.png",
        "",
        "Each PNG is an Ours (GS-LSS) Gaussian color raster in the exact BEV projection/crop.",
        "It is a top-down Gaussian splat map for orientation, not a held-out perspective-camera render.",
        "",
    ]
    for angle, rgb in context_rgb.items():
        pc_path, candidates, valid_count, total_count = provenance[angle]
        raw = Image.fromarray((rgb[::-1] * 255).astype(np.uint8)).resize((720, 720), RESIZE_BICUBIC)
        labeled_img = labeled_topdown_image(rgb, angle, valid_count, total_count)
        raw.save(out_dir / f"ours_bev_rgb_{angle}.png")
        labeled_img.save(out_dir / f"ours_bev_rgb_labeled_{angle}.png")
        readme.extend([
            f"{angle} deg",
            f"  selected point cloud: {pc_path}",
            f"  projected gaussians in crop: {valid_count} / {total_count}",
            "  candidates:",
        ])
        readme.extend([f"    {p}" for p in candidates])
        readme.append("")
    (out_dir / "README.txt").write_text("\n".join(readme) + "\n")
    return out_dir


def draw_composite(scene, context_rgb, angles, ref_crop, grids, vmax, ext):
    nrows = len(ORDER) + 1
    fig, axes = plt.subplots(
        nrows,
        len(angles),
        figsize=(2.35 * len(angles), 2.35 * len(ORDER) + 2.45),
        gridspec_kw={"height_ratios": [1.02] + [1.0] * len(ORDER)},
    )
    if len(angles) == 1:
        axes = axes[:, None]

    for col, angle in enumerate(angles):
        ax = axes[0, col]
        ax.imshow(context_rgb[angle], origin="lower", extent=ext, aspect="equal", interpolation=INTERP)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{angle} deg gap", fontsize=12)
        if col == 0:
            ax.set_ylabel("Ours\nBEV view", fontsize=11, fontweight="bold", color="crimson")

    for row, method in enumerate(ORDER, start=1):
        for col, angle in enumerate(angles):
            ax = axes[row, col]
            grid = grids[(method, angle)]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(
                np.clip(ref_crop, 0, 1),
                origin="lower",
                extent=ext,
                cmap="Greys",
                vmin=0,
                vmax=1.2,
                alpha=0.30,
                aspect="equal",
                interpolation=INTERP,
            )
            if grid is not None:
                shown = np.sqrt(np.clip(grid.astype(float), 0, None) / vmax)
                ax.imshow(
                    np.clip(shown, 0, 1),
                    origin="lower",
                    extent=ext,
                    cmap="inferno",
                    vmin=0,
                    vmax=1,
                    alpha=0.92,
                    aspect="equal",
                    interpolation=INTERP,
                )
                fr = frac(scene, method, angle)
                if fr is not None:
                    ax.text(
                        0.96,
                        0.04,
                        f"{fr * 100:.0f}%",
                        color=("#39ff14" if method == "Ours" else "w"),
                        ha="right",
                        va="bottom",
                        transform=ax.transAxes,
                        fontsize=11,
                        fontweight="bold",
                    )
            else:
                ax.text(0.5, 0.5, "n/a", color="gray", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            if col == 0:
                ax.set_ylabel(
                    NICE[method],
                    fontsize=11,
                    fontweight=("bold" if method == "Ours" else "normal"),
                    color=("crimson" if method == "Ours" else "black"),
                )

    fig.suptitle(
        f"Free-space floater density (top-down BEV) - {scene}    (% = share of model Gaussians in free space)",
        fontsize=13,
        y=0.996,
    )
    fig.text(
        0.5,
        0.018,
        "Top row: Ours (GS-LSS) Gaussian color map in the same BEV projection/crop; heatmaps show free-space floater density.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.035, 0.9, 0.985])

    cax = fig.add_axes([0.915, 0.16, 0.014, 0.60])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="inferno"), cax=cax)
    cb.set_ticks([0.03, 0.97])
    cb.set_ticklabels(["low\n(clean)", "high\n(dense\nfloaters)"])
    cb.ax.tick_params(labelsize=9)
    cb.set_label("free-space floater density -> brighter = more floaters", fontsize=10)

    out = GA / f"BEV_overlay_{scene}_bevcontext.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    return out


def compute_reference_projection(scene):
    from bev_context_figure import reference_points
    from main_crop import main_box_proj, proj_from_ref, px_box

    ref = reference_points(scene)
    center, axes = proj_from_ref(ref)
    r2 = (ref - center) @ axes.T
    pad = 0.35
    mn = r2.min(0) - pad * (r2.max(0) - r2.min(0))
    mx = r2.max(0) + pad * (r2.max(0) - r2.min(0))
    cx, cy, half = main_box_proj(ref, center, axes)
    # Keep defaults exactly synced with bev_context_figure/load_bev_data.
    from bev_context_figure import MASS, MARGIN
    cx, cy, half = main_box_proj(ref, center, axes, mass_frac=MASS, margin=MARGIN)
    i0, i1, j0, j1 = px_box(mn, mx, BINS, cx, cy, half)
    ext = [cx - half, cx + half, cy - half, cy + half]
    return center, axes, (i0, i1, j0, j1), ext


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "bonsai"
    if scene != "bonsai":
        raise SystemExit("bev_aligned_context_figure.py is intentionally configured for bonsai only.")

    manifest = load_manifest()
    angles, ref_crop, grids, vmax, ext = load_bev_data(scene)
    center, axes, _, proj_ext = compute_reference_projection(scene)
    if not np.allclose(np.asarray(ext), np.asarray(proj_ext)):
        raise RuntimeError("Internal BEV crop mismatch between load_bev_data and context projection.")

    context_rgb = {}
    provenance = {}
    for angle in angles:
        rgb, pc_path, candidates, valid_count, total_count = rasterize_ours_topdown(
            scene, angle, ref_crop, ext, center, axes, manifest
        )
        context_rgb[angle] = rgb
        provenance[angle] = (pc_path, candidates, valid_count, total_count)

    context_dir = export_bev_context(scene, context_rgb, provenance)
    bev_dir = export_bev_panels(scene, angles, ref_crop, grids, vmax)
    out = draw_composite(scene, context_rgb, angles, ref_crop, grids, vmax, ext)
    print(f"wrote {out}")
    print(f"BEV-aligned context exports -> {context_dir}")
    print(f"BEV panel exports -> {bev_dir}")


if __name__ == "__main__":
    main()
