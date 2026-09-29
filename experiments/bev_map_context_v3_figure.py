#!/usr/bin/env python3
"""BEV map-context v3: regular layout, true split source, readable camera glyphs.

Fixes v2 problems:
  - no extra reference column; 5-column grid stays aligned with BEV panels
  - no drawing all cameras; representative frustums + arcs prevent clutter
  - gap wedge width is the declared split angle exactly
  - context map uses the same BEV crop as the heatmaps, so layout stays stable
"""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge
from PIL import Image, ImageDraw

from bev_aligned_context_figure import rasterize_ours_topdown
from bev_context_figure import (
    GA,
    SEP,
    INTERP,
    NICE,
    ORDER,
    RESIZE_BICUBIC,
    export_bev_panels,
    frac,
    load_bev_data,
    load_manifest,
    reference_points,
)
from bev_map_context_v2_figure import BLUE, BLUE_FILL, ORANGE, ORANGE_FILL, INK, draw_camera_icon, make_rgb_reference
from bev_map_context_v2_figure import dataset_camera_geometry
from main_crop import proj_from_ref


def clean_rgb(rgb):
    out = np.asarray(rgb, dtype=np.float32)
    out = np.clip((out - 0.50) * 1.18 + 0.53, 0.0, 1.0)
    return out


def angle_diff(a, b):
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - b))))


def representative_angles(angles, max_count, center=None, forbidden_half=None, margin_deg=4.0):
    angles = np.asarray(angles, dtype=float)
    if center is not None and forbidden_half is not None:
        margin = math.radians(margin_deg)
        keep = angle_diff(angles, center) > (forbidden_half + margin)
        if keep.sum() >= max(3, min(max_count, len(angles) // 4)):
            angles = angles[keep]
    if len(angles) <= max_count:
        return angles
    order = np.argsort(np.unwrap(angles))
    idx = np.linspace(0, len(order) - 1, max_count, dtype=int)
    return angles[order[idx]]


def draw_split_map(ax, rgb, ref_crop, ext, geom, angle, panel_label=True):
    cx = (ext[0] + ext[1]) / 2.0
    cy = (ext[2] + ext[3]) / 2.0
    half = (ext[1] - ext[0]) / 2.0
    center = np.array([cx, cy], dtype=float)

    ax.imshow(clean_rgb(rgb), origin="lower", extent=ext, aspect="equal", interpolation=INTERP, zorder=1)
    ax.imshow(np.clip(ref_crop, 0, 1), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.2,
              alpha=0.10, aspect="equal", interpolation=INTERP, zorder=2)

    # Clean schematic ring: arcs show the split; glyphs are representative cameras, not every camera.
    ring_r = 0.82 * half
    ring_w = 0.035 * half
    ax.add_patch(Wedge((cx, cy), ring_r, 0, 360, width=ring_w, facecolor=BLUE,
                       edgecolor="none", alpha=0.22, zorder=3))
    t1 = math.degrees(geom["gap_dir"] - geom["gap_half"])
    t2 = math.degrees(geom["gap_dir"] + geom["gap_half"])
    ax.add_patch(Wedge((cx, cy), ring_r + 0.010 * half, t1, t2, width=ring_w * 2.0,
                       facecolor=ORANGE, edgecolor=ORANGE, linewidth=0.7, alpha=0.35, zorder=4))
    ax.add_patch(Wedge((cx, cy), 0.98 * half, t1, t2, facecolor=ORANGE, edgecolor="none",
                       alpha=0.08, zorder=0))

    train_angles = geom["angles"][~geom["is_gap"]]
    gap_angles = geom["angles"][geom["is_gap"]]
    train_rep = representative_angles(train_angles, 13, geom["gap_dir"], geom["gap_half"])
    gap_rep = representative_angles(gap_angles, 7)
    icon_size = 0.055 * half
    for a in train_rep:
        p = center + 0.73 * half * np.array([math.cos(a), math.sin(a)])
        draw_camera_icon(ax, p, a + math.pi, BLUE, BLUE_FILL, icon_size, alpha=0.95, lw=1.05, z=8)
    for a in gap_rep:
        p = center + 0.89 * half * np.array([math.cos(a), math.sin(a)])
        draw_camera_icon(ax, p, a + math.pi, ORANGE, ORANGE_FILL, icon_size * 1.05, alpha=1.0, lw=1.25, z=13)

    gd = geom["gap_dir"]
    ax.annotate("", xy=(cx + 0.58 * half * math.cos(gd), cy + 0.58 * half * math.sin(gd)), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.0, shrinkA=0, shrinkB=0), zorder=18)
    ax.add_patch(Circle((cx, cy), radius=0.016 * half, facecolor=INK, edgecolor="white", linewidth=0.5, zorder=19))

    if panel_label:
        ax.text(0.045, 0.94, f"{angle} deg gap", transform=ax.transAxes, ha="left", va="top",
                fontsize=10.2, fontweight="bold", color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.0), zorder=25)
        ax.text(0.045, 0.055, f"train {geom['n_train']} | gap {geom['n_gap']}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8.6, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=1.8), zorder=25)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_edgecolor("#383838")


def draw_bev_panel(ax, scene, method, angle, ref_crop, grid, vmax, ext):
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(np.clip(ref_crop, 0, 1), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.2,
              alpha=0.30, aspect="equal", interpolation=INTERP)
    if grid is not None:
        shown = np.sqrt(np.clip(grid.astype(float), 0, None) / vmax)
        ax.imshow(np.clip(shown, 0, 1), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1,
                  alpha=0.92, aspect="equal", interpolation=INTERP)
        fr = frac(scene, method, angle)
        if fr is not None:
            ax.text(0.955, 0.045, f"{fr * 100:.0f}%", color=("#39ff14" if method == "Ours" else "w"),
                    ha="right", va="bottom", transform=ax.transAxes, fontsize=10.5, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "n/a", color="gray", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])


def export_assets(scene, angles, context_rgb, ref_crop, ext, geoms, provenance, rgb_panel, rgb_crop, rgb_boxed, rgb_src, rgb_candidates, box):
    out_dir = SEP / f"{scene}_mapcontext_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_panel.save(out_dir / "rgb_reference_panel.png")
    rgb_crop.save(out_dir / "rgb_reference_crop.png")
    rgb_boxed.save(out_dir / "rgb_reference_full_boxed.png")
    for angle in angles:
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        draw_split_map(ax, context_rgb[angle], ref_crop, ext, geoms[angle], angle)
        fig.tight_layout(pad=0.03)
        fig.savefig(out_dir / f"map_context_{angle}.png", dpi=180)
        plt.close(fig)
        Image.fromarray((context_rgb[angle][::-1] * 255).astype(np.uint8)).resize((720, 720), RESIZE_BICUBIC).save(
            out_dir / f"ours_bev_background_{angle}.png")
    readme = [
        f"Map-context v3 exports for BEV_overlay_{scene}_mapcontext_v3.png",
        "",
        "This version keeps a regular 5-column layout. Camera glyphs are representative, not exhaustive, to avoid clutter.",
        "Train/gap membership and counts come from DATA/mipnerf360/<scene>/test_selection_angle_<angle>.0.txt.",
        "The orange arc/sector width is the declared gap angle exactly.",
        "The map extent is the same BEV crop used by the heatmap panels, so the top context row aligns with the body.",
        "",
        f"RGB reference source: {rgb_src}",
        f"RGB crop box xyxy: {list(box)}",
        "RGB candidates:",
    ]
    readme.extend([f"  {p}" for p in rgb_candidates])
    readme.append("")
    for angle in angles:
        pc_path, candidates, valid_count, total_count = provenance[angle]
        g = geoms[angle]
        readme.extend([
            f"{angle} deg",
            f"  declared_gap_deg: {g['declared_gap_deg']:.1f}",
            f"  Ours BEV background point cloud: {pc_path}",
            f"  projected gaussians in crop: {valid_count} / {total_count}",
            f"  cameras_json: {g['cameras_json']}",
            f"  split: {g['split']}",
            f"  train/gap cameras: {g['n_train']} / {g['n_gap']}",
            f"  gap_dir_rad: {g['gap_dir']:.6f}",
            f"  gap_half_rad: {g['gap_half']:.6f}",
            "  point-cloud candidates:",
        ])
        readme.extend([f"    {p}" for p in candidates])
        readme.append("")
    (out_dir / "README.txt").write_text("\n".join(readme) + "\n")
    return out_dir


def draw_composite(scene, angles, context_rgb, ref_crop, grids, vmax, ext, geoms):
    nrows = len(ORDER) + 1
    fig, axes = plt.subplots(nrows, len(angles), figsize=(2.38 * len(angles), 2.28 * len(ORDER) + 2.55),
                             gridspec_kw={"height_ratios": [1.15] + [1.0] * len(ORDER)})
    if len(angles) == 1:
        axes = axes[:, None]
    for col, angle in enumerate(angles):
        ax = axes[0, col]
        draw_split_map(ax, context_rgb[angle], ref_crop, ext, geoms[angle], angle)
        ax.set_title(f"{angle} deg", fontsize=11.6, fontweight="bold", pad=5)
        if col == 0:
            ax.set_ylabel("BEV context\n(Ours + split)", fontsize=10.7, fontweight="bold", color=INK)
    for row, method in enumerate(ORDER, start=1):
        for col, angle in enumerate(angles):
            ax = axes[row, col]
            draw_bev_panel(ax, scene, method, angle, ref_crop, grids[(method, angle)], vmax, ext)
            if col == 0:
                ax.set_ylabel(NICE[method], fontsize=10.8,
                              fontweight=("bold" if method == "Ours" else "normal"),
                              color=("crimson" if method == "Ours" else INK))
    fig.suptitle("Free-space floater density with dataset-defined camera gaps - bonsai", fontsize=13.1, y=0.997)
    fig.text(0.5, 0.018,
             "Top row: representative camera frustums from the true split; orange arc shows the declared held-out sector. Body: brighter heat = denser free-space floaters.",
             ha="center", va="bottom", fontsize=9.8, color=INK)
    plt.tight_layout(rect=[0, 0.035, 0.90, 0.982])
    cax = fig.add_axes([0.917, 0.155, 0.014, 0.60])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="inferno"), cax=cax)
    cb.set_ticks([0.03, 0.97]); cb.set_ticklabels(["low\n(clean)", "high\n(dense\nfloaters)"])
    cb.ax.tick_params(labelsize=9)
    cb.set_label("free-space floater density -> brighter = more floaters", fontsize=10)
    out = GA / f"BEV_overlay_{scene}_mapcontext_v3.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    return out


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "bonsai"
    if scene != "bonsai":
        raise SystemExit("bev_map_context_v3_figure.py is intentionally configured for bonsai only.")
    manifest = load_manifest()
    angles, ref_crop, grids, vmax, ext = load_bev_data(scene)
    ref = reference_points(scene)
    center, axes = proj_from_ref(ref)
    context_rgb = {}; provenance = {}; geoms = {}
    for angle in angles:
        rgb, pc_path, candidates, valid_count, total_count = rasterize_ours_topdown(
            scene, angle, ref_crop, ext, center, axes, manifest)
        context_rgb[angle] = rgb
        provenance[angle] = (pc_path, candidates, valid_count, total_count)
        geoms[angle] = dataset_camera_geometry(scene, angle, center, axes, ext)
    rgb_panel, rgb_crop, rgb_boxed, rgb_src, rgb_candidates, box = make_rgb_reference(scene)
    out_dir = export_assets(scene, angles, context_rgb, ref_crop, ext, geoms, provenance,
                            rgb_panel, rgb_crop, rgb_boxed, rgb_src, rgb_candidates, box)
    bev_dir = export_bev_panels(scene, angles, ref_crop, grids, vmax)
    out = draw_composite(scene, angles, context_rgb, ref_crop, grids, vmax, ext, geoms)
    print(f"wrote {out}")
    print(f"map context v3 exports -> {out_dir}")
    print(f"BEV panel exports -> {bev_dir}")


if __name__ == "__main__":
    main()
