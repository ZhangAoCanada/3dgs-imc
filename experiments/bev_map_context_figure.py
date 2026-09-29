#!/usr/bin/env python3
"""Create a polished bonsai BEV figure with map-style context.

This version combines the BEV-aligned Ours Gaussian color map with camera-shaped
train/gap glyphs, then places the same BEV floater-density grid below it.
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge
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
    labeled,
    load_bev_data,
    load_manifest,
    reference_points,
)
from bev_orientation_context_figure import camera_geometry, rgb_reference
from main_crop import proj_from_ref

BLUE = "#1f67d2"
BLUE_FILL = "#d8e8ff"
ORANGE = "#ef7d1a"
ORANGE_FILL = "#ffe0bd"
INK = "#202124"
PANEL_EDGE = "#3d3d3d"


def desaturate(rgb, amount=0.46, lift=0.10):
    gray = np.sum(rgb * np.array([0.299, 0.587, 0.114], dtype=np.float32), axis=-1, keepdims=True)
    out = gray * amount + rgb * (1.0 - amount)
    out = out * (1.0 - lift) + lift
    return np.clip(out, 0.0, 1.0)


def sample_angles(angles, max_count):
    angles = np.asarray(angles, dtype=float)
    if len(angles) <= max_count:
        return angles
    order = np.argsort(np.unwrap(angles))
    picked = np.linspace(0, len(order) - 1, max_count, dtype=int)
    return angles[order[picked]]


def poly_points(center, forward, right, pts):
    c = np.asarray(center, dtype=float)
    f = np.asarray(forward, dtype=float)
    r = np.asarray(right, dtype=float)
    return np.asarray([c + x * f + y * r for x, y in pts])


def draw_camera_icon(ax, x, y, theta, color, fill, size, alpha=1.0, lw=1.35, zorder=8):
    """Draw a small top-down camera/frustum glyph. theta points from camera toward scene center."""
    f = np.array([math.cos(theta), math.sin(theta)])
    r = np.array([-math.sin(theta), math.cos(theta)])
    center = np.array([x, y])

    body = poly_points(center - f * size * 0.13, f, r, [
        (-0.24 * size, -0.30 * size),
        (-0.24 * size, 0.30 * size),
        (0.22 * size, 0.30 * size),
        (0.22 * size, -0.30 * size),
    ])
    lens = poly_points(center + f * size * 0.20, f, r, [
        (-0.08 * size, -0.18 * size),
        (-0.08 * size, 0.18 * size),
        (0.18 * size, 0.18 * size),
        (0.18 * size, -0.18 * size),
    ])
    frustum = poly_points(center + f * size * 0.30, f, r, [
        (0.02 * size, -0.14 * size),
        (0.95 * size, -0.58 * size),
        (0.95 * size, 0.58 * size),
        (0.02 * size, 0.14 * size),
    ])

    ax.add_patch(Polygon(frustum, closed=True, facecolor=fill, edgecolor=color, linewidth=lw,
                         alpha=0.28 * alpha, joinstyle="round", zorder=zorder))
    ax.add_patch(Polygon(body, closed=True, facecolor="white", edgecolor=color, linewidth=lw,
                         alpha=0.96 * alpha, joinstyle="round", zorder=zorder + 1))
    ax.add_patch(Polygon(lens, closed=True, facecolor=fill, edgecolor=color, linewidth=lw * 0.8,
                         alpha=0.96 * alpha, joinstyle="round", zorder=zorder + 2))
    ax.add_patch(Circle(center - f * size * 0.14, radius=size * 0.055, facecolor=color,
                        edgecolor="none", alpha=0.90 * alpha, zorder=zorder + 3))


def draw_legend_icon(ax, x, y, color, fill, text):
    draw_camera_icon(ax, x, y, 0.0, color, fill, 0.055, alpha=1.0, lw=1.2, zorder=20)
    ax.text(x + 0.085, y, text, transform=ax.transAxes, ha="left", va="center",
            fontsize=8.8, color=INK, zorder=30)


def draw_map_context(ax, rgb, ref_crop, ext, geom, angle, show_angle=True):
    cx = (ext[0] + ext[1]) / 2.0
    cy = (ext[2] + ext[3]) / 2.0
    half = (ext[1] - ext[0]) / 2.0
    radius = 0.94 * half
    ring = 0.83 * half
    icon_size = 0.067 * half

    ax.imshow(desaturate(rgb), origin="lower", extent=ext, aspect="equal", interpolation=INTERP, zorder=1)
    ax.imshow(np.clip(ref_crop, 0, 1), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.25,
              alpha=0.16, aspect="equal", interpolation=INTERP, zorder=2)

    theta1 = math.degrees(geom["gap_dir"] - geom["gap_half"])
    theta2 = math.degrees(geom["gap_dir"] + geom["gap_half"])
    ax.add_patch(Wedge((cx, cy), radius, theta1, theta2, facecolor=ORANGE, alpha=0.16,
                       edgecolor=ORANGE, linewidth=1.15, zorder=3))

    train = sample_angles(geom["train_angles"], 22)
    gap = sample_angles(geom["test_angles"], 11)
    for a in train:
        x, y = cx + ring * math.cos(a), cy + ring * math.sin(a)
        draw_camera_icon(ax, x, y, a + math.pi, BLUE, BLUE_FILL, icon_size, alpha=0.90, lw=1.05, zorder=6)
    for a in gap:
        x, y = cx + (ring * 0.99) * math.cos(a), cy + (ring * 0.99) * math.sin(a)
        draw_camera_icon(ax, x, y, a + math.pi, ORANGE, ORANGE_FILL, icon_size * 1.08, alpha=1.0, lw=1.35, zorder=12)

    gd = geom["gap_dir"]
    ax.annotate("", xy=(cx + 0.68 * half * math.cos(gd), cy + 0.68 * half * math.sin(gd)), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.0, shrinkA=0, shrinkB=0), zorder=16)
    ax.add_patch(Circle((cx, cy), radius=0.018 * half, facecolor=INK, edgecolor="white", linewidth=0.6, zorder=17))

    if show_angle:
        ax.text(0.045, 0.94, f"{angle} deg gap", transform=ax.transAxes, ha="left", va="top",
                fontsize=10.4, fontweight="bold", color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.2), zorder=30)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_edgecolor(PANEL_EDGE)


def make_rgb_reference(scene):
    img, src, candidates = rgb_reference(scene)
    box = (234, 78, 442, 286)
    boxed = img.copy()
    draw = ImageDraw.Draw(boxed)
    draw.rectangle(box, outline=(235, 46, 36), width=max(3, img.size[0] // 220))
    crop = img.crop(box).resize((720, 720), RESIZE_BICUBIC)
    full = boxed.copy()
    full.thumbnail((720, 500), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
    canvas = Image.new("RGB", (720, 720), (247, 247, 247))
    canvas.paste(full, ((720 - full.width) // 2, 24))
    crop_small = img.crop(box).resize((214, 214), RESIZE_BICUBIC)
    canvas.paste(crop_small, (480, 480))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((480, 480, 693, 693), outline=(235, 46, 36), width=4)
    draw.text((24, 638), "RGB reference: potted bonsai on tabletop", fill=INK, font=fnt_safe(26))
    draw.text((24, 674), "Top maps are aligned BEV views", fill=(80, 80, 80), font=fnt_safe(22))
    return canvas, crop, boxed, src, candidates, box


def fnt_safe(size):
    from bev_context_figure import fnt
    return fnt(size)


def draw_reference_panel(ax, rgb_panel):
    ax.imshow(rgb_panel)
    ax.set_xlim(0, rgb_panel.size[0])
    ax.set_ylim(rgb_panel.size[1], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Scene Reference", fontsize=11.5, fontweight="bold", pad=5)
    ax.add_patch(Rectangle((16, 16), 310, 118, facecolor="white", edgecolor="none", alpha=0.86, zorder=20))
    draw_camera_icon(ax, 54, 56, 0.0, BLUE, BLUE_FILL, 25.0, alpha=1.0, lw=1.15, zorder=30)
    ax.text(95, 56, "training views", ha="left", va="center", fontsize=10.2, color=INK, zorder=31)
    draw_camera_icon(ax, 54, 104, 0.0, ORANGE, ORANGE_FILL, 25.0, alpha=1.0, lw=1.25, zorder=30)
    ax.text(95, 104, "held-out gap", ha="left", va="center", fontsize=10.2, color=INK, zorder=31)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_map_png(path, rgb, ref_crop, ext, geom, angle):
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    draw_map_context(ax, rgb, ref_crop, ext, geom, angle)
    fig.tight_layout(pad=0.03)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_map_assets(scene, angles, context_rgb, ref_crop, ext, geoms, provenance, rgb_panel, rgb_crop, rgb_boxed, rgb_src, rgb_candidates, box):
    out_dir = SEP / f"{scene}_mapcontext"
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_panel.save(out_dir / "rgb_reference_panel.png")
    rgb_crop.save(out_dir / "rgb_reference_crop.png")
    rgb_boxed.save(out_dir / "rgb_reference_full_boxed.png")
    for angle in angles:
        save_map_png(out_dir / f"map_context_{angle}.png", context_rgb[angle], ref_crop, ext, geoms[angle], angle)
        raw = Image.fromarray((context_rgb[angle][::-1] * 255).astype(np.uint8)).resize((720, 720), RESIZE_BICUBIC)
        raw.save(out_dir / f"ours_bev_background_{angle}.png")

    readme = [
        f"Map-context exports for BEV_overlay_{scene}_mapcontext.png",
        "",
        "Each map_context_<angle>.png combines:",
        "  - desaturated Ours (GS-LSS) top-down Gaussian color map from the best_coverage_gap PLY",
        "  - faint SfM footprint in the exact BEV crop",
        "  - blue camera-frustum glyphs for representative training views",
        "  - orange camera-frustum glyphs and sector for held-out gap views.",
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
            f"  Ours BEV background point cloud: {pc_path}",
            f"  projected gaussians in crop: {valid_count} / {total_count}",
            f"  camera source_dir: {g['source_dir']}",
            f"  cameras_json: {g['cameras_json']}",
            f"  split: {g['split']}",
            f"  train/test cameras: {g['n_train']} / {g['n_test']}",
            f"  gap_dir_rad: {g['gap_dir']:.6f}",
            f"  gap_half_rad: {g['gap_half']:.6f}",
            "  point-cloud candidates:",
        ])
        readme.extend([f"    {p}" for p in candidates])
        readme.append("")
    (out_dir / "README.txt").write_text("\n".join(readme) + "\n")
    return out_dir


def draw_bev_panel(ax, scene, method, angle, ref_crop, grid, vmax, ext):
    ax.set_xticks([])
    ax.set_yticks([])
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
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])


def draw_composite(scene, angles, context_rgb, ref_crop, grids, vmax, ext, geoms, rgb_panel):
    nrows = len(ORDER) + 1
    ncols = len(angles) + 1
    fig = plt.figure(figsize=(2.30 + 2.42 * len(angles), 2.24 * len(ORDER) + 2.72))
    gs = GridSpec(nrows, ncols, figure=fig, width_ratios=[0.96] + [1.0] * len(angles),
                  height_ratios=[1.17] + [1.0] * len(ORDER), wspace=0.055, hspace=0.075)

    ax_ref = fig.add_subplot(gs[0, 0])
    draw_reference_panel(ax_ref, rgb_panel)

    for col, angle in enumerate(angles, start=1):
        ax = fig.add_subplot(gs[0, col])
        draw_map_context(ax, context_rgb[angle], ref_crop, ext, geoms[angle], angle, show_angle=False)
        ax.set_title(f"{angle} deg gap", fontsize=11.6, fontweight="bold", pad=5)

    for row, method in enumerate(ORDER, start=1):
        label_ax = fig.add_subplot(gs[row, 0])
        label_ax.axis("off")
        label_ax.text(0.98, 0.5, NICE[method], ha="right", va="center", fontsize=11.0,
                      fontweight=("bold" if method == "Ours" else "normal"),
                      color=("crimson" if method == "Ours" else INK))
        for col, angle in enumerate(angles, start=1):
            ax = fig.add_subplot(gs[row, col])
            draw_bev_panel(ax, scene, method, angle, ref_crop, grids[(method, angle)], vmax, ext)

    fig.suptitle("Free-space floater density in the same top-down BEV map - bonsai", fontsize=13.2, y=0.996)
    fig.text(0.53, 0.018,
             "Top: BEV context maps with Ours geometry and camera frustums. Body: brighter heat means denser free-space floaters.",
             ha="center", va="bottom", fontsize=10.1, color=INK)
    fig.subplots_adjust(left=0.035, right=0.902, top=0.966, bottom=0.044)

    cax = fig.add_axes([0.921, 0.155, 0.014, 0.60])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="inferno"), cax=cax)
    cb.set_ticks([0.03, 0.97])
    cb.set_ticklabels(["low\n(clean)", "high\n(dense\nfloaters)"])
    cb.ax.tick_params(labelsize=9)
    cb.set_label("free-space floater density -> brighter = more floaters", fontsize=10)

    out = GA / f"BEV_overlay_{scene}_mapcontext.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    return out


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "bonsai"
    if scene != "bonsai":
        raise SystemExit("bev_map_context_figure.py is intentionally configured for bonsai only.")

    manifest = load_manifest()
    angles, ref_crop, grids, vmax, ext = load_bev_data(scene)
    ref = reference_points(scene)
    center, axes = proj_from_ref(ref)
    display_center = ((ext[0] + ext[1]) / 2.0, (ext[2] + ext[3]) / 2.0)

    context_rgb = {}
    provenance = {}
    for angle in angles:
        rgb, pc_path, candidates, valid_count, total_count = rasterize_ours_topdown(
            scene, angle, ref_crop, ext, center, axes, manifest
        )
        context_rgb[angle] = rgb
        provenance[angle] = (pc_path, candidates, valid_count, total_count)

    geoms = {angle: camera_geometry(scene, angle, center, axes, display_center) for angle in angles}
    rgb_panel, rgb_crop, rgb_boxed, rgb_src, rgb_candidates, box = make_rgb_reference(scene)
    out_dir = export_map_assets(scene, angles, context_rgb, ref_crop, ext, geoms, provenance,
                                rgb_panel, rgb_crop, rgb_boxed, rgb_src, rgb_candidates, box)
    bev_dir = export_bev_panels(scene, angles, ref_crop, grids, vmax)
    out = draw_composite(scene, angles, context_rgb, ref_crop, grids, vmax, ext, geoms, rgb_panel)
    print(f"wrote {out}")
    print(f"map context exports -> {out_dir}")
    print(f"BEV panel exports -> {bev_dir}")


if __name__ == "__main__":
    main()
