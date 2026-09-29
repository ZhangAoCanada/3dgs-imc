#!/usr/bin/env python3
"""BEV map-context v2: true dataset camera split, camera glyphs, exact gap size.

The top context maps use a larger map extent that contains the projected camera
positions. The BEV crop used by the heatmap grid is drawn as a black rectangle.
The body heatmaps keep the original BEV crop/normalization unchanged.
"""

import json
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
    VM,
    INTERP,
    NICE,
    ORDER,
    RESIZE_BICUBIC,
    export_bev_panels,
    fnt,
    frac,
    load_bev_data,
    load_manifest,
    reference_points,
)
from bev_orientation_context_figure import best_source_dir, rgb_reference, split_names
from main_crop import proj_from_ref

BLUE = "#1764d8"
BLUE_FILL = "#dceaff"
ORANGE = "#ee7b16"
ORANGE_FILL = "#ffe0b8"
INK = "#1e1f22"
EDGE = "#333333"


def circular_mean(a):
    a = np.asarray(a, dtype=float)
    return float(math.atan2(np.sin(a).mean(), np.cos(a).mean()))


def angular_subset(points, angles, max_count):
    points = np.asarray(points, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if len(points) <= max_count:
        return points, angles
    order = np.argsort(np.unwrap(angles))
    pick = np.linspace(0, len(order) - 1, max_count, dtype=int)
    idx = order[pick]
    return points[idx], angles[idx]


def dataset_camera_geometry(scene, angle, center, axes, crop_ext):
    src, manifest = best_source_dir(scene, angle)
    cam_path = src / "cameras.json"
    test_names, split_path = split_names(scene, angle)
    cams = json.loads(cam_path.read_text())
    names = [Path(c["img_name"]).stem for c in cams]
    pos = np.asarray([c["position"] for c in cams], dtype=np.float32)
    xy = (pos - center) @ axes.T
    is_gap = np.asarray([n in test_names for n in names], dtype=bool)
    if not np.any(is_gap):
        raise RuntimeError(f"No test cameras from {split_path} matched {cam_path}")
    if np.any(is_gap & ~is_gap):
        raise AssertionError("impossible split state")

    crop_center = np.array([(crop_ext[0] + crop_ext[1]) / 2.0, (crop_ext[2] + crop_ext[3]) / 2.0])
    vec = xy - crop_center
    cam_ang = np.arctan2(vec[:, 1], vec[:, 0])
    gap_center = circular_mean(cam_ang[is_gap])
    gap_half = math.radians(float(angle) / 2.0)

    # Larger context extent: include true camera positions and the BEV crop, with enough padding for glyphs.
    mn = np.minimum(xy.min(axis=0), np.array([crop_ext[0], crop_ext[2]], dtype=float))
    mx = np.maximum(xy.max(axis=0), np.array([crop_ext[1], crop_ext[3]], dtype=float))
    span = mx - mn
    pad = 0.12 * max(float(span[0]), float(span[1]))
    ctx_ext = [float(mn[0] - pad), float(mx[0] + pad), float(mn[1] - pad), float(mx[1] + pad)]
    # Square it, so the wedge is geometrically honest in display coordinates.
    cx = (ctx_ext[0] + ctx_ext[1]) / 2.0
    cy = (ctx_ext[2] + ctx_ext[3]) / 2.0
    half = max(ctx_ext[1] - ctx_ext[0], ctx_ext[3] - ctx_ext[2]) / 2.0
    ctx_ext = [cx - half, cx + half, cy - half, cy + half]

    return {
        "xy": xy,
        "is_gap": is_gap,
        "angles": cam_ang,
        "gap_dir": gap_center,
        "gap_half": gap_half,
        "declared_gap_deg": float(angle),
        "crop_center": crop_center,
        "ctx_ext": ctx_ext,
        "source_dir": src,
        "manifest": manifest,
        "cameras_json": cam_path,
        "split": split_path,
        "n_train": int((~is_gap).sum()),
        "n_gap": int(is_gap.sum()),
    }


def desaturate(rgb, amount=0.12, lift=0.02):
    gray = np.sum(rgb * np.array([0.299, 0.587, 0.114], dtype=np.float32), axis=-1, keepdims=True)
    out = gray * amount + rgb * (1.0 - amount)
    return np.clip(out * (1.0 - lift) + lift, 0.0, 1.0)


def poly(center, forward, right, pts):
    c = np.asarray(center, dtype=float)
    f = np.asarray(forward, dtype=float)
    r = np.asarray(right, dtype=float)
    return np.asarray([c + x * f + y * r for x, y in pts])


def draw_camera_icon(ax, xy, theta, color, fill, size, alpha=1.0, lw=1.15, z=8):
    f = np.array([math.cos(theta), math.sin(theta)])
    r = np.array([-math.sin(theta), math.cos(theta)])
    c = np.asarray(xy, dtype=float)
    frustum = poly(c + f * size * 0.22, f, r, [
        (0.00 * size, -0.12 * size), (0.78 * size, -0.46 * size),
        (0.78 * size, 0.46 * size), (0.00 * size, 0.12 * size),
    ])
    body = poly(c - f * size * 0.07, f, r, [
        (-0.22 * size, -0.24 * size), (-0.22 * size, 0.24 * size),
        (0.18 * size, 0.24 * size), (0.18 * size, -0.24 * size),
    ])
    ax.add_patch(Polygon(frustum, closed=True, facecolor=fill, edgecolor=color, linewidth=lw,
                         alpha=0.30 * alpha, joinstyle="round", zorder=z))
    ax.add_patch(Polygon(body, closed=True, facecolor="white", edgecolor=color, linewidth=lw,
                         alpha=0.97 * alpha, joinstyle="round", zorder=z + 1))
    ax.add_patch(Circle(c - f * size * 0.06, radius=size * 0.045, facecolor=color, edgecolor="none",
                        alpha=0.95 * alpha, zorder=z + 2))


def make_rgb_reference(scene):
    img, src, candidates = rgb_reference(scene)
    box = (234, 78, 442, 286)
    boxed = img.copy()
    draw = ImageDraw.Draw(boxed)
    draw.rectangle(box, outline=(232, 52, 45), width=max(3, img.size[0] // 220))
    crop = img.crop(box).resize((720, 720), RESIZE_BICUBIC)
    full = boxed.copy()
    full.thumbnail((720, 500), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
    canvas = Image.new("RGB", (720, 720), (248, 248, 248))
    canvas.paste(full, ((720 - full.width) // 2, 28))
    crop_small = img.crop(box).resize((220, 220), RESIZE_BICUBIC)
    canvas.paste(crop_small, (472, 472))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((472, 472, 691, 691), outline=(232, 52, 45), width=4)
    draw.text((24, 626), "Scene reference", fill=INK, font=fnt(31))
    draw.text((24, 666), "potted bonsai on tabletop", fill=(72, 72, 72), font=fnt(23))
    return canvas, crop, boxed, src, candidates, box


def draw_map(ax, rgb, ref_crop, crop_ext, geom, angle, title=True):
    ctx_ext = geom["ctx_ext"]
    ax.set_facecolor("#f6f6f3")
    ax.imshow(desaturate(rgb), origin="lower", extent=crop_ext, aspect="equal", interpolation=INTERP, zorder=2)
    ax.imshow(np.clip(ref_crop, 0, 1), origin="lower", extent=crop_ext, cmap="Greys", vmin=0, vmax=1.25,
              alpha=0.13, aspect="equal", interpolation=INTERP, zorder=3)

    cc = geom["crop_center"]
    ctx_half = (ctx_ext[1] - ctx_ext[0]) / 2.0
    wedge_r = ctx_half * 1.22
    t1 = math.degrees(geom["gap_dir"] - geom["gap_half"])
    t2 = math.degrees(geom["gap_dir"] + geom["gap_half"])
    ax.add_patch(Wedge(tuple(cc), wedge_r, t1, t2, facecolor=ORANGE, alpha=0.13,
                       edgecolor=ORANGE, linewidth=1.05, zorder=1))

    # Draw the exact crop used by the heatmaps below.
    ax.add_patch(Rectangle((crop_ext[0], crop_ext[2]), crop_ext[1] - crop_ext[0], crop_ext[3] - crop_ext[2],
                           facecolor="none", edgecolor=INK, linewidth=1.15, linestyle="-", zorder=5))
    ax.text(crop_ext[0], crop_ext[3], " BEV crop", ha="left", va="bottom", fontsize=8.6, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=1.5), zorder=20)

    train_xy = geom["xy"][~geom["is_gap"]]
    gap_xy = geom["xy"][geom["is_gap"]]
    train_ang = geom["angles"][~geom["is_gap"]]
    gap_ang = geom["angles"][geom["is_gap"]]
    train_xy, train_ang = angular_subset(train_xy, train_ang, 26)
    gap_xy, gap_ang = angular_subset(gap_xy, gap_ang, min(16, max(8, len(gap_xy))))
    icon_size = 0.030 * (ctx_ext[1] - ctx_ext[0])
    for p, a in zip(train_xy, train_ang):
        draw_camera_icon(ax, p, a + math.pi, BLUE, BLUE_FILL, icon_size, alpha=0.92, lw=1.0, z=8)
    for p, a in zip(gap_xy, gap_ang):
        draw_camera_icon(ax, p, a + math.pi, ORANGE, ORANGE_FILL, icon_size * 1.06, alpha=1.0, lw=1.25, z=14)

    ax.annotate("", xy=(cc[0] + 0.42 * ctx_half * math.cos(geom["gap_dir"]),
                         cc[1] + 0.42 * ctx_half * math.sin(geom["gap_dir"])),
                xytext=tuple(cc), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2), zorder=18)
    ax.add_patch(Circle(tuple(cc), radius=0.0065 * (ctx_ext[1] - ctx_ext[0]), facecolor=INK,
                        edgecolor="white", linewidth=0.5, zorder=19))
    if title:
        ax.text(0.045, 0.94, f"{angle} deg held-out sector", transform=ax.transAxes, ha="left", va="top",
                fontsize=10.0, fontweight="bold", color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.0), zorder=30)
    ax.set_xlim(ctx_ext[0], ctx_ext[1]); ax.set_ylim(ctx_ext[2], ctx_ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.75); s.set_edgecolor(EDGE)


def draw_reference_panel(ax, img):
    ax.imshow(img)
    ax.set_xlim(0, img.size[0]); ax.set_ylim(img.size[1], 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Scene Reference", fontsize=11.5, fontweight="bold", pad=5)
    ax.add_patch(Rectangle((16, 16), 322, 118, facecolor="white", edgecolor="none", alpha=0.88, zorder=20))
    draw_camera_icon(ax, np.array([54, 56]), 0.0, BLUE, BLUE_FILL, 25.0, alpha=1.0, lw=1.15, z=30)
    ax.text(98, 56, "training cameras", ha="left", va="center", fontsize=10.0, color=INK, zorder=31)
    draw_camera_icon(ax, np.array([54, 104]), 0.0, ORANGE, ORANGE_FILL, 25.0, alpha=1.0, lw=1.25, z=30)
    ax.text(98, 104, "held-out gap cameras", ha="left", va="center", fontsize=10.0, color=INK, zorder=31)
    for s in ax.spines.values():
        s.set_visible(False)


def draw_bev_panel(ax, scene, method, angle, ref_crop, grid, vmax, crop_ext):
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(np.clip(ref_crop, 0, 1), origin="lower", extent=crop_ext, cmap="Greys", vmin=0, vmax=1.2,
              alpha=0.30, aspect="equal", interpolation=INTERP)
    if grid is not None:
        shown = np.sqrt(np.clip(grid.astype(float), 0, None) / vmax)
        ax.imshow(np.clip(shown, 0, 1), origin="lower", extent=crop_ext, cmap="inferno", vmin=0, vmax=1,
                  alpha=0.92, aspect="equal", interpolation=INTERP)
        fr = frac(scene, method, angle)
        if fr is not None:
            ax.text(0.955, 0.045, f"{fr * 100:.0f}%", color=("#39ff14" if method == "Ours" else "w"),
                    ha="right", va="bottom", transform=ax.transAxes, fontsize=10.5, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "n/a", color="gray", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlim(crop_ext[0], crop_ext[1]); ax.set_ylim(crop_ext[2], crop_ext[3])


def save_assets(scene, angles, context_rgb, ref_crop, crop_ext, geoms, provenance, rgb_panel, rgb_crop, rgb_boxed, rgb_src, rgb_candidates, box):
    out_dir = SEP / f"{scene}_mapcontext_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_panel.save(out_dir / "rgb_reference_panel.png")
    rgb_crop.save(out_dir / "rgb_reference_crop.png")
    rgb_boxed.save(out_dir / "rgb_reference_full_boxed.png")
    for angle in angles:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        draw_map(ax, context_rgb[angle], ref_crop, crop_ext, geoms[angle], angle)
        fig.tight_layout(pad=0.03)
        fig.savefig(out_dir / f"map_context_{angle}.png", dpi=180)
        plt.close(fig)
        Image.fromarray((context_rgb[angle][::-1] * 255).astype(np.uint8)).resize((720, 720), RESIZE_BICUBIC).save(
            out_dir / f"ours_bev_background_{angle}.png")
    readme = [
        f"Map-context v2 exports for BEV_overlay_{scene}_mapcontext_v2.png",
        "",
        "Camera glyphs are drawn from the true dataset split files:",
        "  DATA/mipnerf360/<scene>/test_selection_angle_<angle>.0.txt",
        "Camera positions come from cameras.json for a surviving same-angle run; names are matched to dataset split stems.",
        "Orange sector width is the declared held-out gap angle, not inferred from marker spread.",
        "The black rectangle is the exact BEV crop used by the heatmap panels below.",
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
            f"  context_extent_xyxy: {g['ctx_ext']}",
            f"  gap_dir_rad: {g['gap_dir']:.6f}",
            f"  gap_half_rad: {g['gap_half']:.6f}",
            "  point-cloud candidates:",
        ])
        readme.extend([f"    {p}" for p in candidates])
        readme.append("")
    (out_dir / "README.txt").write_text("\n".join(readme) + "\n")
    return out_dir


def draw_composite(scene, angles, context_rgb, ref_crop, grids, vmax, crop_ext, geoms, rgb_panel):
    nrows = len(ORDER) + 1
    ncols = len(angles) + 1
    fig = plt.figure(figsize=(2.42 + 2.52 * len(angles), 2.24 * len(ORDER) + 2.85))
    gs = GridSpec(nrows, ncols, figure=fig, width_ratios=[1.02] + [1.0] * len(angles),
                  height_ratios=[1.25] + [1.0] * len(ORDER), wspace=0.055, hspace=0.075)
    ax_ref = fig.add_subplot(gs[0, 0]); draw_reference_panel(ax_ref, rgb_panel)
    for col, angle in enumerate(angles, start=1):
        ax = fig.add_subplot(gs[0, col])
        draw_map(ax, context_rgb[angle], ref_crop, crop_ext, geoms[angle], angle, title=False)
        ax.set_title(f"{angle} deg gap", fontsize=11.6, fontweight="bold", pad=5)
    for row, method in enumerate(ORDER, start=1):
        label_ax = fig.add_subplot(gs[row, 0]); label_ax.axis("off")
        label_ax.text(0.98, 0.5, NICE[method], ha="right", va="center", fontsize=11.0,
                      fontweight=("bold" if method == "Ours" else "normal"),
                      color=("crimson" if method == "Ours" else INK))
        for col, angle in enumerate(angles, start=1):
            ax = fig.add_subplot(gs[row, col])
            draw_bev_panel(ax, scene, method, angle, ref_crop, grids[(method, angle)], vmax, crop_ext)
    fig.suptitle("Free-space floater density with dataset-defined camera gaps - bonsai", fontsize=13.2, y=0.996)
    fig.text(0.53, 0.018,
             "Top: true dataset camera split and declared gap sector; black box is the BEV crop used by all heatmaps below.",
             ha="center", va="bottom", fontsize=10.1, color=INK)
    fig.subplots_adjust(left=0.035, right=0.902, top=0.966, bottom=0.044)
    cax = fig.add_axes([0.921, 0.155, 0.014, 0.60])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="inferno"), cax=cax)
    cb.set_ticks([0.03, 0.97]); cb.set_ticklabels(["low\n(clean)", "high\n(dense\nfloaters)"])
    cb.ax.tick_params(labelsize=9)
    cb.set_label("free-space floater density -> brighter = more floaters", fontsize=10)
    out = GA / f"BEV_overlay_{scene}_mapcontext_v2.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    return out


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "bonsai"
    if scene != "bonsai":
        raise SystemExit("bev_map_context_v2_figure.py is intentionally configured for bonsai only.")
    manifest = load_manifest()
    angles, ref_crop, grids, vmax, crop_ext = load_bev_data(scene)
    ref = reference_points(scene)
    center, axes = proj_from_ref(ref)
    context_rgb = {}; provenance = {}
    for angle in angles:
        rgb, pc_path, candidates, valid_count, total_count = rasterize_ours_topdown(
            scene, angle, ref_crop, crop_ext, center, axes, manifest)
        context_rgb[angle] = rgb
        provenance[angle] = (pc_path, candidates, valid_count, total_count)
    geoms = {angle: dataset_camera_geometry(scene, angle, center, axes, crop_ext) for angle in angles}
    rgb_panel, rgb_crop, rgb_boxed, rgb_src, rgb_candidates, box = make_rgb_reference(scene)
    out_dir = save_assets(scene, angles, context_rgb, ref_crop, crop_ext, geoms, provenance,
                          rgb_panel, rgb_crop, rgb_boxed, rgb_src, rgb_candidates, box)
    bev_dir = export_bev_panels(scene, angles, ref_crop, grids, vmax)
    out = draw_composite(scene, angles, context_rgb, ref_crop, grids, vmax, crop_ext, geoms, rgb_panel)
    print(f"wrote {out}")
    print(f"map context v2 exports -> {out_dir}")
    print(f"BEV panel exports -> {bev_dir}")


if __name__ == "__main__":
    main()
