#!/usr/bin/env python3
"""Paper-style bonsai BEV figure using the preferred v3 circular schematic.

This is the cleaned-up version of `bev_map_context_v3_figure.py`:
  - preserves the compact v3 grid style and original method order
  - uses representative train/test camera glyphs on a circular schematic
  - labels the split as train/test instead of gap/held-out
  - writes ordered separate exports for paper assembly
"""

import math
import os
import re
import sys
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Wedge
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

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
    reference_points,
)
from bev_map_context_v2_figure import INK, make_rgb_reference
import bev_map_context_v4_figure as pose_source
from main_crop import proj_from_ref

_COLMAP_SPEC = importlib.util.spec_from_file_location("local_colmap_loader", ROOT / "scene" / "colmap_loader.py")
_COLMAP_LOADER = importlib.util.module_from_spec(_COLMAP_SPEC)
_COLMAP_SPEC.loader.exec_module(_COLMAP_LOADER)
read_points3D_binary = _COLMAP_LOADER.read_points3D_binary
read_points3D_text = _COLMAP_LOADER.read_points3D_text

SCENE = "bonsai"
ANGLES = [20, 40, 60, 80, 100]
SCHEMATIC_GAP_DIR = math.pi
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SHOW_CONTEXT_LEGEND = os.environ.get("BEV_CONTEXT_LEGEND", "0").lower() in {"1", "true", "yes", "on"}
EXPORT_SUFFIX = "_withlegend" if SHOW_CONTEXT_LEGEND else ""
TRAIN_COLOR = "#c0392b"
TRAIN_FRUSTUM = "#e74c3c"
TEST_COLOR = "#2980b9"
TEST_FRUSTUM = "#3498db"
SECTOR_COLOR = "#e67e22"
SFM_CONTEXT_RES = 720


def font(size, bold=False):
    path = FONT_BOLD if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def slug(text):
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")
    return text or "panel"


def text_center(draw, box, text, ft, fill=INK):
    x0, y0, x1, y1 = box
    bb = draw.textbbox((0, 0), text, font=ft)
    draw.text((x0 + (x1 - x0 - (bb[2] - bb[0])) / 2, y0 + (y1 - y0 - (bb[3] - bb[1])) / 2), text, font=ft, fill=fill)


def oriented_poly(center, forward, right, pts):
    c = np.asarray(center, dtype=float)
    f = np.asarray(forward, dtype=float)
    r = np.asarray(right, dtype=float)
    return np.asarray([c + x * f + y * r for x, y in pts])


def draw_camera_icon(ax, xy, theta, color, fill, size, alpha=1.0, lw=1.0, z=8):
    """Draw the pic1 camera: solid body + trapezoid lens + translucent triangular frustum."""
    f = np.array([math.cos(theta), math.sin(theta)])
    r = np.array([-math.sin(theta), math.cos(theta)])
    c = np.asarray(xy, dtype=float)

    frustum = oriented_poly(c, f, r, [
        (0.42 * size, 0.00 * size),
        (2.05 * size, -0.80 * size),
        (2.05 * size, 0.80 * size),
    ])
    lens = oriented_poly(c, f, r, [
        (0.17 * size, -0.27 * size),
        (0.17 * size, 0.27 * size),
        (0.42 * size, 0.40 * size),
        (0.42 * size, -0.40 * size),
    ])
    body = oriented_poly(c, f, r, [
        (-0.23 * size, -0.48 * size),
        (-0.23 * size, 0.48 * size),
        (0.17 * size, 0.48 * size),
        (0.17 * size, -0.48 * size),
    ])
    ax.add_patch(Polygon(frustum, closed=True, facecolor=fill, edgecolor="none",
                         alpha=0.15 * alpha, zorder=z))
    ax.add_patch(Polygon(lens, closed=True, facecolor=color, edgecolor=color,
                         linewidth=0.10 * lw, alpha=0.98 * alpha, joinstyle="miter", zorder=z + 2))
    ax.add_patch(Polygon(body, closed=True, facecolor=color, edgecolor=color,
                         linewidth=0.10 * lw, alpha=0.98 * alpha, joinstyle="miter", zorder=z + 3))


def save_canvas(canvas, path):
    canvas.save(path)
    if path.suffix.lower() != ".pdf":
        canvas.save(path.with_suffix(".pdf"), "PDF", resolution=300.0)


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


def fixed_schematic_camera_angles(gap_dir):
    """A fixed, uniformly spaced camera track used in every column.

    Large pic1-style camera glyphs read best as a clean schematic ring; uniform
    spacing keeps the paper figure legible and avoids local overlaps.
    """
    offsets_deg = np.array(
        [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180],
        dtype=float,
    )
    return gap_dir + np.deg2rad(offsets_deg)


def schematic_camera_angles(gap_dir, gap_half):
    """Use fixed camera positions, coloring by whether each lies in the sector."""
    all_angles = fixed_schematic_camera_angles(gap_dir)
    in_test = angle_diff(all_angles, gap_dir) <= (gap_half + 1e-9)
    return all_angles, in_test


def clean_rgb(rgb):
    """Make the top-down context map a little more readable in print."""
    arr = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    pil = ImageEnhance.Contrast(pil).enhance(1.18)
    pil = ImageEnhance.Sharpness(pil).enhance(1.18)
    pil = ImageEnhance.Brightness(pil).enhance(1.04)
    out = np.asarray(pil, dtype=np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def load_sfm_points(scene):
    sparse_dir = pose_source.DATA / scene / "sparse/0"
    bin_path = sparse_dir / "points3D.bin"
    txt_path = sparse_dir / "points3D.txt"
    if bin_path.exists():
        xyz, rgb, errors = read_points3D_binary(str(bin_path))
        return xyz.astype(np.float32), rgb.astype(np.float32), errors.reshape(-1).astype(np.float32), bin_path
    if txt_path.exists():
        xyz, rgb, errors = read_points3D_text(str(txt_path))
        return xyz.astype(np.float32), rgb.astype(np.float32), errors.reshape(-1).astype(np.float32), txt_path
    raise FileNotFoundError(f"No COLMAP points3D file found under {sparse_dir}")


def rasterize_sfm_topdown(scene, ext, center, axes, res=SFM_CONTEXT_RES):
    xyz, rgb, errors, src = load_sfm_points(scene)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]
    errors = errors[finite]

    # Drop only extreme reconstruction-error outliers; the crop itself handles spatial outliers.
    if errors.size:
        keep = errors <= np.percentile(errors, 99.5)
        xyz = xyz[keep]
        rgb = rgb[keep]
        errors = errors[keep]

    xy = (xyz - center) @ axes.T
    x0, x1, y0, y1 = ext
    cols = np.floor((xy[:, 0] - x0) / (x1 - x0) * res).astype(np.int64)
    rows = np.floor((xy[:, 1] - y0) / (y1 - y0) * res).astype(np.int64)
    valid = (cols >= 0) & (cols < res) & (rows >= 0) & (rows < res)
    if not np.any(valid):
        raise RuntimeError(f"No SfM points project into BEV crop for {scene} from {src}")

    cols = cols[valid]
    rows = rows[valid]
    flat = rows * res + cols
    rgb01 = np.clip(rgb[valid] / 255.0, 0.0, 1.0)
    if errors.size:
        w_raw = 1.0 / (1.0 + np.clip(errors[valid], 0.0, None))
        weights = np.clip(w_raw / max(np.percentile(w_raw, 95), 1e-6), 0.12, 1.0).astype(np.float32)
    else:
        weights = np.ones(len(flat), dtype=np.float32)

    size = res * res
    den = np.bincount(flat, weights=weights, minlength=size).reshape(res, res).astype(np.float32)
    sums = [
        np.bincount(flat, weights=weights * rgb01[:, c], minlength=size).reshape(res, res).astype(np.float32)
        for c in range(3)
    ]

    sigma = 1.35
    den_s = gaussian_filter(den, sigma=sigma)
    color_s = np.stack([gaussian_filter(s, sigma=sigma) for s in sums], axis=-1)
    color = np.zeros((res, res, 3), dtype=np.float32)
    mask = den_s > 1e-8
    color[mask] = color_s[mask] / den_s[mask, None]

    nonzero = den_s[den_s > 0]
    scale = np.percentile(nonzero, 99.5) if nonzero.size else 1.0
    alpha = np.clip((den_s / (scale + 1e-8)) ** 0.42, 0.0, 1.0)[..., None]

    bg = np.ones((res, res, 3), dtype=np.float32) * 0.972
    neutral = np.array([0.28, 0.32, 0.33], dtype=np.float32)
    muted_color = 0.42 * color + 0.58 * neutral
    img = bg * (1.0 - 0.92 * alpha) + muted_color * (0.92 * alpha)

    # Add a crisp high-density core so the sparse SfM footprint remains legible after downscaling.
    den_core = gaussian_filter(den, sigma=0.45)
    core_nz = den_core[den_core > 0]
    core_scale = np.percentile(core_nz, 99.6) if core_nz.size else 1.0
    core = np.clip((den_core / (core_scale + 1e-8)) ** 0.55, 0.0, 1.0)[..., None]
    img = img * (1.0 - 0.18 * core) + neutral * (0.18 * core)
    return np.clip(img, 0.0, 1.0), src, int(valid.sum()), int(len(xyz))


def draw_context_legend(ax, ext):
    x0, x1, y0, y1 = ext
    w = x1 - x0
    h = y1 - y0
    bh = 0.135 * h
    text_x = x1 - 0.018 * w
    icon_x = text_x - 0.095 * w
    bx = icon_x - 0.035 * w
    bw = x1 - 0.004 * w - bx
    by = y1 - bh - 0.020 * h
    ax.add_patch(FancyBboxPatch(
        (bx, by), bw, bh,
        boxstyle=f"round,pad={0.012 * w},rounding_size={0.016 * w}",
        facecolor="white", edgecolor="none", alpha=0.72, zorder=25,
    ))
    y_train = by + 0.091 * h
    y_test = by + 0.040 * h
    icon_size = 0.026 * w
    draw_camera_icon(ax, np.array([icon_x, y_train]), 0.0, TRAIN_COLOR, TRAIN_FRUSTUM,
                     icon_size, alpha=1.0, lw=0.92, z=26)
    draw_camera_icon(ax, np.array([icon_x, y_test]), 0.0, TEST_COLOR, TEST_FRUSTUM,
                     icon_size, alpha=1.0, lw=0.98, z=26)
    ax.text(text_x, y_train, "train", ha="right", va="center", fontsize=7.6, color=INK, zorder=28)
    ax.text(text_x, y_test, "test", ha="right", va="center", fontsize=7.6, color=INK, zorder=28)


def draw_test_sector_marker(ax, cx, cy, half, gap_half, z=4):
    t1 = math.degrees(SCHEMATIC_GAP_DIR - gap_half)
    t2 = math.degrees(SCHEMATIC_GAP_DIR + gap_half)
    outer_r = 0.965 * half
    band_w = 0.055 * half
    ax.add_patch(Wedge((cx, cy), outer_r, t1, t2, width=band_w,
                       facecolor=SECTOR_COLOR, edgecolor="none", alpha=0.28, zorder=z))
    ax.add_patch(Arc((cx, cy), 2 * outer_r, 2 * outer_r, theta1=t1, theta2=t2,
                     edgecolor=SECTOR_COLOR, linewidth=1.25, alpha=0.78,
                     linestyle=(0, (2.4, 2.2)), zorder=z + 1))
    for a in (SCHEMATIC_GAP_DIR - gap_half, SCHEMATIC_GAP_DIR + gap_half):
        inner_r = outer_r - band_w
        xs = [cx + inner_r * math.cos(a), cx + outer_r * math.cos(a)]
        ys = [cy + inner_r * math.sin(a), cy + outer_r * math.sin(a)]
        ax.plot(xs, ys, color=SECTOR_COLOR, linewidth=0.90, alpha=0.46,
                linestyle=(0, (2.6, 2.4)), zorder=z)


def draw_split_map(ax, rgb, ref_crop, ext, geom, angle, panel_label=False,
                   camera_scale=1.0, camera_lw_scale=1.0):
    cx = (ext[0] + ext[1]) / 2.0
    cy = (ext[2] + ext[3]) / 2.0
    half = (ext[1] - ext[0]) / 2.0
    center = np.array([cx, cy], dtype=float)

    # The SfM point-cloud context fills the panel, matching the v3 figure layout.
    ax.imshow(clean_rgb(rgb), origin="lower", extent=ext, aspect="equal", interpolation=INTERP, zorder=1)
    ax.imshow(np.clip(ref_crop, 0, 1), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.2,
              alpha=0.025, aspect="equal", interpolation=INTERP, zorder=2)

    draw_test_sector_marker(ax, cx, cy, half, geom["gap_half"], z=4)

    all_cam_angles, is_test_cam = schematic_camera_angles(SCHEMATIC_GAP_DIR, geom["gap_half"])
    icon_size = 0.066 * half * camera_scale
    cam_radius = 0.79 * half

    for a in all_cam_angles[~is_test_cam]:
        p = center + cam_radius * np.array([math.cos(a), math.sin(a)])
        draw_camera_icon(ax, p, a + math.pi, TRAIN_COLOR, TRAIN_FRUSTUM, icon_size,
                         alpha=0.92, lw=0.95 * camera_lw_scale, z=8)
    for a in all_cam_angles[is_test_cam]:
        p = center + cam_radius * np.array([math.cos(a), math.sin(a)])
        draw_camera_icon(ax, p, a + math.pi, TEST_COLOR, TEST_FRUSTUM, icon_size * 1.06,
                         alpha=1.0, lw=1.18 * camera_lw_scale, z=13)

    if panel_label:
        ax.text(0.045, 0.94, f"{angle} deg test", transform=ax.transAxes, ha="left", va="top",
                fontsize=10.2, fontweight="bold", color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=2.0), zorder=25)
        ax.text(0.045, 0.055, f"train {geom['n_train']} | test {geom['n_gap']}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8.6, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=1.8), zorder=25)
    if SHOW_CONTEXT_LEGEND:
        draw_context_legend(ax, ext)

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_edgecolor("#383838")


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


def export_camera_legend(path):
    fig, ax = plt.subplots(figsize=(5.4, 1.45))
    ax.set_xlim(0.0, 5.4)
    ax.set_ylim(0.0, 1.45)
    ax.axis("off")

    draw_camera_icon(ax, np.array([0.42, 0.92]), 0.0, TRAIN_COLOR, TRAIN_FRUSTUM, 0.17, alpha=1.0, lw=1.15, z=8)
    ax.text(0.72, 0.92, "train camera", ha="left", va="center", fontsize=11.0, color=INK)

    draw_camera_icon(ax, np.array([0.42, 0.46]), 0.0, TEST_COLOR, TEST_FRUSTUM, 0.17, alpha=1.0, lw=1.20, z=8)
    ax.text(0.72, 0.46, "test camera", ha="left", va="center", fontsize=11.0, color=INK)

    cx, cy = 3.15, 0.70
    ax.add_patch(Wedge((cx, cy), 0.48, 145, 215, width=0.075,
                       facecolor=SECTOR_COLOR, edgecolor="none", alpha=0.28, zorder=1))
    ax.add_patch(Arc((cx, cy), 0.96, 0.96, theta1=145, theta2=215,
                     edgecolor=SECTOR_COLOR, linewidth=1.25, alpha=0.78,
                     linestyle=(0, (2.4, 2.2)), zorder=2))
    for a in (math.radians(145), math.radians(215)):
        ax.plot([cx + 0.405 * math.cos(a), cx + 0.48 * math.cos(a)],
                [cy + 0.405 * math.sin(a), cy + 0.48 * math.sin(a)],
                color=SECTOR_COLOR, linewidth=0.9, alpha=0.46,
                linestyle=(0, (2.6, 2.4)), zorder=2)
    ax.text(3.72, 0.70, "test sector", ha="left", va="center", fontsize=11.0, color=INK)

    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def export_ordered_assets(scene, angles, context_rgb, ref_crop, ext, geoms, provenance,
                          rgb_panel, rgb_crop, rgb_boxed, rgb_src, rgb_candidates, box):
    out_dir = SEP / f"{scene}_mapcontext_v9_schematic{EXPORT_SUFFIX}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "panels_all").mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*_ours_bev_rgb.png"):
        stale.unlink()
    rgb_panel.save(out_dir / "00_rgb_reference_panel.png")
    rgb_crop.save(out_dir / "00_rgb_reference_crop.png")
    rgb_boxed.save(out_dir / "00_rgb_reference_full_boxed.png")
    export_camera_legend(out_dir / "00_camera_legend.png")

    context_paths = {}
    for col, angle in enumerate(angles, start=1):
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        draw_split_map(ax, context_rgb[angle], ref_crop, ext, geoms[angle], angle,
                       camera_scale=2.65, camera_lw_scale=1.30)
        fig.tight_layout(pad=0.03)
        path = out_dir / f"{col:02d}_angle{angle:03d}_train_test_context.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        Image.fromarray((context_rgb[angle][::-1] * 255).astype(np.uint8)).resize((720, 720), RESIZE_BICUBIC).save(
            out_dir / f"{col:02d}_angle{angle:03d}_sfm_topdown.png")
        context_paths[angle] = path

    panel_sources = {}
    for row, method in enumerate(ORDER, start=1):
        for col, angle in enumerate(angles, start=1):
            panel_path = SEP / scene / f"{str(NICE[method]).replace('/', '-')}_{angle}.png"
            if not panel_path.exists():
                continue
            img = Image.open(panel_path).convert("RGB")
            if img.height > img.width:
                img = img.crop((0, img.height - img.width, img.width, img.height))
            img = img.resize((720, 720), RESIZE_BICUBIC)
            panel_sources[(method, angle)] = panel_path
            img.save(out_dir / "panels_all" / f"row{row:02d}_{slug(NICE[method])}_col{col:02d}_angle{angle:03d}.png")

    compose_context_row(context_paths, out_dir / "00_context_row.png")

    readme = [
        f"Map-context v9 schematic exports for BEV_overlay_{scene}_mapcontext_v9_schematic{EXPORT_SUFFIX}.png",
        "",
        "This version follows the preferred v3 circular schematic style.",
        "Camera glyphs are representative train/test viewpoints, not exhaustive source-camera positions.",
        "Camera glyph geometry follows the pic1 style: solid body, trapezoid lens, and translucent triangular frustum.",
        "Camera colors also follow pic1: train cameras are red and test cameras are blue; the orange sector boundary marks the test region.",
        "For visual clarity with large pic1-style glyphs, each column reuses the same fixed, uniformly spaced camera positions on one radius.",
        "All test glyphs are placed inside the declared test sector and all train glyphs outside it.",
        "The schematic sector is centered consistently at the left; sector width and train/test counts come from data.",
        ("Per-panel legends are shown inside each context panel."
         if SHOW_CONTEXT_LEGEND
         else "Per-panel context labels are intentionally removed; use 00_camera_legend.png for color/sector meanings."),
        "Train/test counts and sector width still come from the real dataset split files.",
        "The context map is a top-down rasterization of COLMAP SfM points in the same BEV crop.",
        "",
        f"script: {ROOT / 'experiments' / 'bev_map_context_v9_schematic_figure.py'}",
        f"angle order: {angles}",
        f"method order: {[NICE[m] for m in ORDER]}",
        f"camera legend: {out_dir / '00_camera_legend.png'}",
        f"RGB reference source for crop bookkeeping: {rgb_src}",
        f"RGB crop box xyxy for bookkeeping: {list(box)}",
        "RGB bookkeeping candidates:",
    ]
    readme.extend([f"  {p}" for p in rgb_candidates])
    readme.append("")
    for col, angle in enumerate(angles, start=1):
        pc_path, candidates, valid_count, total_count = provenance[angle]
        g = geoms[angle]
        readme.extend([
            f"{col:02d} angle {angle:03d}",
            f"  context panel: {context_paths[angle]}",
            f"  declared_test_deg: {float(angle):.1f}",
            f"  SfM point cloud: {pc_path}",
            f"  projected SfM points in crop: {valid_count} / {total_count}",
            f"  camera pose source: {g['images_bin']}",
            f"  split: {g['split']}",
            f"  train/test cameras: {g['n_train']} / {g['n_gap']}",
            f"  gap_dir_rad: {g['gap_dir']:.6f}",
            f"  gap_half_rad: {g['gap_half']:.6f}",
        ])
        readme.extend([f"    candidate: {p}" for p in candidates])
        readme.append("")
    readme.append("Body panel sources")
    for row, method in enumerate(ORDER, start=1):
        for col, angle in enumerate(angles, start=1):
            src = panel_sources.get((method, angle))
            if src is not None:
                readme.append(f"  row{row:02d} {NICE[method]} col{col:02d} angle{angle:03d}: {src}")
    (out_dir / "README.txt").write_text("\n".join(readme) + "\n")

    return out_dir


def compose_context_row(context_paths, out_path, cell=330, gap=8):
    left = 4
    top = 42
    width = left * 2 + len(ANGLES) * cell + (len(ANGLES) - 1) * gap
    height = top + cell + 4
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for col, angle in enumerate(ANGLES):
        x = left + col * (cell + gap)
        text_center(draw, (x, 8, x + cell, 36), f"{angle} deg", font(15, True))
        img = Image.open(context_paths[angle]).convert("RGB").resize((cell, cell), RESIZE_BICUBIC)
        canvas.paste(img, (x, top))
    save_canvas(canvas, out_path)


def draw_composite(scene, angles, context_rgb, ref_crop, grids, vmax, ext, geoms):
    nrows = len(ORDER) + 1
    fig, axes = plt.subplots(nrows, len(angles), figsize=(2.38 * len(angles), 2.28 * len(ORDER) + 2.55),
                             gridspec_kw={"height_ratios": [1.15] + [1.0] * len(ORDER)})
    if len(angles) == 1:
        axes = axes[:, None]
    for col, angle in enumerate(angles):
        ax = axes[0, col]
        draw_split_map(ax, context_rgb[angle], ref_crop, ext, geoms[angle], angle,
                       camera_scale=4.85, camera_lw_scale=1.45)
        ax.set_title(f"{angle} deg", fontsize=11.6, fontweight="bold", pad=5)
        if col == 0:
            ax.set_ylabel("BEV context\n(SfM + split)", fontsize=10.7, fontweight="bold", color=INK)
    for row, method in enumerate(ORDER, start=1):
        for col, angle in enumerate(angles):
            ax = axes[row, col]
            draw_bev_panel(ax, scene, method, angle, ref_crop, grids[(method, angle)], vmax, ext)
            if col == 0:
                ax.set_ylabel(NICE[method], fontsize=10.8,
                              fontweight=("bold" if method == "Ours" else "normal"),
                              color=("crimson" if method == "Ours" else INK))
    fig.suptitle("Free-space floater density with dataset-defined train/test camera split - bonsai",
                 fontsize=13.1, y=0.997)
    fig.text(0.5, 0.018,
             "Top row: representative train/test cameras over a top-down SfM point-cloud context; orange boundary marks the test sector. Body: brighter heat = denser free-space floaters.",
             ha="center", va="bottom", fontsize=9.8, color=INK)
    plt.tight_layout(rect=[0, 0.035, 0.90, 0.982])
    cax = fig.add_axes([0.917, 0.155, 0.014, 0.60])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="inferno"), cax=cax)
    cb.set_ticks([0.03, 0.97])
    cb.set_ticklabels(["low\n(clean)", "high\n(dense\nfloaters)"])
    cb.ax.tick_params(labelsize=9)
    cb.set_label("free-space floater density -> brighter = more floaters", fontsize=10)
    out = GA / f"BEV_overlay_{scene}_mapcontext_v9_schematic{EXPORT_SUFFIX}.png"
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def load_rgb_reference_assets(scene):
    try:
        return make_rgb_reference(scene)
    except FileNotFoundError:
        ref_dir = SEP / f"{scene}_mapcontext_v3"
        panel = ref_dir / "rgb_reference_panel.png"
        crop = ref_dir / "rgb_reference_crop.png"
        boxed = ref_dir / "rgb_reference_full_boxed.png"
        if not panel.exists():
            raise
        rgb_panel = Image.open(panel).convert("RGB")
        rgb_crop = Image.open(crop).convert("RGB") if crop.exists() else rgb_panel.copy()
        rgb_boxed = Image.open(boxed).convert("RGB") if boxed.exists() else rgb_panel.copy()
        return rgb_panel, rgb_crop, rgb_boxed, panel, [panel, crop, boxed], ()


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else SCENE
    if scene != "bonsai":
        raise SystemExit("bev_map_context_v9_schematic_figure.py is intentionally configured for bonsai only.")
    print(f"v9 schematic BEV generator context_legend={SHOW_CONTEXT_LEGEND}", flush=True)
    angles, ref_crop, grids, vmax, ext = load_bev_data(scene)
    angles = [a for a in ANGLES if a in angles]
    ref = reference_points(scene)
    center, axes = proj_from_ref(ref)
    sfm_rgb, sfm_path, valid_count, total_count = rasterize_sfm_topdown(scene, ext, center, axes)
    context_rgb = {}
    provenance = {}
    geoms = {}
    for angle in angles:
        context_rgb[angle] = sfm_rgb
        provenance[angle] = (sfm_path, [sfm_path], valid_count, total_count)
    geoms_all = pose_source.build_geoms(scene, center, axes, ext)
    geoms = {angle: geoms_all[angle] for angle in angles}

    rgb_panel, rgb_crop, rgb_boxed, rgb_src, rgb_candidates, box = load_rgb_reference_assets(scene)
    out_dir = export_ordered_assets(scene, angles, context_rgb, ref_crop, ext, geoms, provenance,
                                    rgb_panel, rgb_crop, rgb_boxed, rgb_src, rgb_candidates, box)
    bev_dir = export_bev_panels(scene, angles, ref_crop, grids, vmax)
    out = draw_composite(scene, angles, context_rgb, ref_crop, grids, vmax, ext, geoms)
    print(f"wrote {out}", flush=True)
    print(f"wrote {out.with_suffix('.pdf')}", flush=True)
    print(f"map context v9 exports -> {out_dir}", flush=True)
    print(f"BEV panel exports -> {bev_dir}", flush=True)


if __name__ == "__main__":
    main()
