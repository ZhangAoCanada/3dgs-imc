#!/usr/bin/env python3
"""Paper-style bonsai BEV figures with circular train/test camera layouts.

This version is intended as the paper-facing continuation of the BEV context
work. It keeps the same BEV density panels and true dataset split metadata, but
uses a compact circular camera-layout schematic for the context row:

  - blue cameras = train views
  - orange cameras = test views
  - orange sector = test angle
  - central inset = Ours BEV color map for scene orientation
"""

import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Wedge
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import bev_map_context_v4_figure as base

SCENE = "bonsai"
ANGLES = [20, 40, 60, 80, 100]
METHOD_ORDER_ALL = ["3DGS", "MCMC", "MH-3DGS", "Hier", "ScaffoldGS", "PGSR", "MVGS", "CDGS", "Ours"]
METHOD_ORDER_COMPACT = ["3DGS", "MCMC", "MH-3DGS", "Ours"]
NICE = base.NICE

GA = base.GA
SEP = base.SEP
DATA = base.DATA

BLUE = "#1f6ed4"
BLUE_LIGHT = "#d9e9ff"
ORANGE = "#e87516"
ORANGE_LIGHT = "#fff0dd"
INK = "#202124"
MUTED = "#5f6368"

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_BICUBIC = Image.BICUBIC


def font(size, bold=False):
    path = base.FONT if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
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


def save_canvas(canvas, path):
    canvas.save(path)
    if path.suffix.lower() != ".pdf":
        canvas.save(path.with_suffix(".pdf"), "PDF", resolution=300.0)


def make_colorbar(height, width=24):
    cmap = plt.get_cmap("inferno")
    vals = np.linspace(1, 0, height, dtype=np.float32)[:, None]
    rgb = (cmap(vals)[..., :3] * 255).astype(np.uint8).reshape(height, 1, 3)
    return Image.fromarray(np.repeat(rgb, width, axis=1), "RGB")


def annotate_pct(img, scene, method, angle):
    frac = base.frac(scene, method, angle)
    if frac is None:
        return img
    out = img.copy()
    draw = ImageDraw.Draw(out)
    txt = f"{frac * 100:.0f}%"
    ft = font(22, True)
    fill = (57, 255, 20) if method == "Ours" else (255, 255, 255)
    bb = draw.textbbox((0, 0), txt, font=ft)
    x = out.width - (bb[2] - bb[0]) - 10
    y = out.height - (bb[3] - bb[1]) - 10
    draw.text((x + 1, y + 1), txt, font=ft, fill=(0, 0, 0))
    draw.text((x, y), txt, font=ft, fill=fill)
    return out


def load_square_panel(scene, method, angle, size):
    img, path = base.load_panel(scene, method, angle)
    img = img.resize((size, size), RESAMPLE_BICUBIC)
    return annotate_pct(img, scene, method, angle), path


def angle_to_xy(theta, radius):
    return radius * np.cos(theta), radius * np.sin(theta)


def draw_circle_context(ax, bg, geom, angle, legend=False):
    cam_angles = geom["angles"]
    is_test = geom["is_gap"]
    train_angles = cam_angles[~is_test]
    test_angles = cam_angles[is_test]
    gap_dir = geom["gap_dir"]
    gap_half = geom["gap_half"]

    ax.set_facecolor("#fbfaf7")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Camera track and test-sector cue.
    ax.add_patch(Circle((0, 0), 0.94, facecolor="none", edgecolor="#9aa0a6", lw=0.8, alpha=0.72, zorder=1))
    ax.add_patch(Wedge((0, 0), 0.985, math.degrees(gap_dir - gap_half), math.degrees(gap_dir + gap_half),
                       facecolor=ORANGE_LIGHT, edgecolor=ORANGE, lw=1.0, alpha=0.88, zorder=0))
    ax.add_patch(Wedge((0, 0), 0.965, math.degrees(gap_dir - gap_half), math.degrees(gap_dir + gap_half),
                       width=0.105, facecolor=ORANGE, edgecolor="none", alpha=0.20, zorder=2))

    # Central Ours BEV inset for orientation.
    inset = [-0.46, 0.46, -0.46, 0.46]
    ax.imshow(np.asarray(bg) / 255.0, origin="upper", extent=inset, aspect="equal", interpolation="bicubic", zorder=3)
    ax.add_patch(Rectangle((inset[0], inset[2]), inset[1] - inset[0], inset[3] - inset[2],
                           facecolor="none", edgecolor=INK, lw=1.05, zorder=4))
    ax.add_patch(Circle((0, 0), radius=0.018, facecolor=INK, edgecolor="white", lw=0.45, zorder=5))

    # Camera positions are shown on a common circular schematic, preserving the
    # true angular train/test membership while avoiding geographic clutter.
    x_train, y_train = angle_to_xy(train_angles, 0.82)
    x_test, y_test = angle_to_xy(test_angles, 0.82)
    ax.scatter(x_train, y_train, s=8, marker="o", facecolors="white", edgecolors=BLUE, linewidths=0.48, alpha=0.42, zorder=6)
    ax.scatter(x_test, y_test, s=17, marker="o", facecolors=ORANGE_LIGHT, edgecolors=ORANGE, linewidths=0.78, alpha=0.98, zorder=7)

    # A short radial arrow makes the test-sector direction explicit.
    ax.annotate("", xy=angle_to_xy(gap_dir, 0.70), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.7, shrinkA=2, shrinkB=0), zorder=8)

    ax.text(0.050, 0.940, f"{angle} deg test", transform=ax.transAxes, ha="left", va="top",
            fontsize=10.2, fontweight="bold", color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.0), zorder=20)
    ax.text(0.050, 0.060, f"train {geom['n_train']}  |  test {geom['n_gap']}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.4, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.8), zorder=20)

    if legend:
        ax.scatter([-0.91], [0.73], s=20, marker="o", facecolors="white", edgecolors=BLUE, linewidths=0.8, zorder=25)
        ax.text(-0.84, 0.73, "train cameras", ha="left", va="center", fontsize=8.0, color=INK, zorder=25)
        ax.scatter([-0.91], [0.61], s=24, marker="o", facecolors=ORANGE_LIGHT, edgecolors=ORANGE, linewidths=0.8, zorder=25)
        ax.text(-0.84, 0.61, "test cameras", ha="left", va="center", fontsize=8.0, color=INK, zorder=25)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_edgecolor("#3a3a3a")


def render_context_png(path, bg, geom, angle, legend=False, dpi=240):
    fig, ax = plt.subplots(figsize=(3.55, 3.55), dpi=dpi)
    draw_circle_context(ax, bg, geom, angle, legend=legend)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_ordered_assets(scene, outdir, geoms, bgs, bg_sources, crop_ext, ref_path):
    outdir.mkdir(parents=True, exist_ok=True)
    context_paths = {}
    for col, angle in enumerate(ANGLES, start=1):
        path = outdir / f"{col:02d}_angle{angle:03d}_circle_context.png"
        render_context_png(path, bgs[angle], geoms[angle], angle, legend=(col == 1))
        bgs[angle].save(outdir / f"{col:02d}_angle{angle:03d}_ours_bev_inset.png")
        context_paths[angle] = path

    panel_root = outdir / "panels_all"
    panel_root.mkdir(parents=True, exist_ok=True)
    panel_sources = {}
    for row, method in enumerate(METHOD_ORDER_ALL, start=1):
        for col, angle in enumerate(ANGLES, start=1):
            img, src = load_square_panel(scene, method, angle, 720)
            panel_sources[(method, angle)] = src
            img.save(panel_root / f"row{row:02d}_{slug(NICE[method])}_col{col:02d}_angle{angle:03d}.png")

    lines = [
        "BEV bonsai circular-context v8 exports",
        "",
        f"script: {ROOT / 'experiments' / 'bev_map_context_v8_circle_figure.py'}",
        f"scene: {scene}",
        f"angle order: {ANGLES}",
        f"all-method order: {[NICE[m] for m in METHOD_ORDER_ALL]}",
        f"compact-method order: {[NICE[m] for m in METHOD_ORDER_COMPACT]}",
        "",
        "Data provenance",
        f"  camera pose source: {DATA / scene / 'sparse/0/images.bin'}",
        f"  BEV crop reference: {ref_path}",
        f"  BEV metric crop extent: {[round(float(x), 6) for x in crop_ext]}",
        "",
        "Circular context panels",
    ]
    for col, angle in enumerate(ANGLES, start=1):
        geom = geoms[angle]
        lines.extend(
            [
                f"  {col:02d} angle {angle:03d}: {context_paths[angle]}",
                f"    train/test cameras: {geom['n_train']} / {geom['n_gap']}",
                f"    split source: {geom['split']}",
                f"    Ours BEV inset source: {bg_sources[angle]}",
                f"    gap_dir_rad_from_true_test_cameras: {geom['gap_dir']:.8f}",
                f"    gap_half_rad_declared: {geom['gap_half']:.8f}",
            ]
        )
    lines.append("")
    lines.append("Body panel sources")
    for row, method in enumerate(METHOD_ORDER_ALL, start=1):
        for col, angle in enumerate(ANGLES, start=1):
            lines.append(f"  row{row:02d} {NICE[method]} col{col:02d} angle{angle:03d}: {panel_sources[(method, angle)]}")
    (outdir / "README.txt").write_text("\n".join(lines) + "\n")
    return context_paths


def compose_context_row(context_paths, out_path, cell=340, gap=8):
    left = 4
    top = 42
    width = left * 2 + len(ANGLES) * cell + (len(ANGLES) - 1) * gap
    height = top + cell + 4
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for col, angle in enumerate(ANGLES):
        x = left + col * (cell + gap)
        text_center(draw, (x, 8, x + cell, 36), f"{angle} deg test sector", font(15, True))
        img = Image.open(context_paths[angle]).convert("RGB").resize((cell, cell), RESAMPLE_LANCZOS)
        canvas.paste(img, (x, top))
    save_canvas(canvas, out_path)


def compose_grid(scene, context_paths, methods, out_path, title, cell=322, compact=False):
    gap = 8
    left = 150 if compact else 158
    right = 112
    top = 76
    footer = 46
    rows = 1 + len(methods)
    cols = len(ANGLES)
    width = left + cols * cell + (cols - 1) * gap + right
    height = top + rows * cell + (rows - 1) * gap + footer
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((left, 14), title, font=font(19, True), fill=INK)
    for col, angle in enumerate(ANGLES):
        x = left + col * (cell + gap)
        text_center(draw, (x, 42, x + cell, 68), f"{angle} deg", font(16, True))

    y = top
    draw.text((18, y + cell // 2 - 21), "Train/test\ncamera split", font=font(13, True), fill=INK, spacing=4)
    for col, angle in enumerate(ANGLES):
        x = left + col * (cell + gap)
        img = Image.open(context_paths[angle]).convert("RGB").resize((cell, cell), RESAMPLE_LANCZOS)
        canvas.paste(img, (x, y))

    for row, method in enumerate(methods, start=1):
        y = top + row * (cell + gap)
        label = NICE[method]
        ft = font(14 if len(label) > 10 else 15, method == "Ours")
        fill = (178, 38, 52) if method == "Ours" else INK
        bb = draw.multiline_textbbox((0, 0), label, font=ft)
        draw.text((left - 14 - (bb[2] - bb[0]), y + cell // 2 - (bb[3] - bb[1]) // 2), label, font=ft, fill=fill)
        for col, angle in enumerate(ANGLES):
            x = left + col * (cell + gap)
            img, _ = load_square_panel(scene, method, angle, cell)
            canvas.paste(img, (x, y))

    body_y = top + cell + gap
    body_h = len(methods) * cell + (len(methods) - 1) * gap
    cbx = left + cols * cell + (cols - 1) * gap + 30
    canvas.paste(make_colorbar(body_h, 22), (cbx, body_y))
    draw.text((cbx + 31, body_y), "more\nfloaters", font=font(11), fill=INK, spacing=1)
    draw.text((cbx + 31, body_y + body_h - 34), "fewer", font=font(11), fill=INK)
    draw.text((left, height - 30), "Shared scale. Percent labels denote the model share classified as free-space floaters.", font=font(12), fill=MUTED)
    save_canvas(canvas, out_path)


def main():
    scene = SCENE
    print("v8 circular train/test BEV generator", flush=True)
    center, axes, crop_ext, ref_path = base.crop_extent(scene)
    geoms = base.build_geoms(scene, center, axes, crop_ext)
    print("cameras", geoms[20]["n_all"], "angle order", ANGLES, flush=True)

    bgs = {}
    bg_sources = {}
    for angle in ANGLES:
        bgs[angle], bg_sources[angle] = base.load_bg(scene, angle)

    outdir = SEP / f"{scene}_mapcontext_v8_circle"
    context_paths = save_ordered_assets(scene, outdir, geoms, bgs, bg_sources, crop_ext, ref_path)
    compose_context_row(context_paths, outdir / "00_context_row.png")

    all_out = GA / f"BEV_overlay_{scene}_mapcontext_v8_circle.png"
    compact_out = GA / f"BEV_overlay_{scene}_mapcontext_v8_circle_compact.png"
    compose_grid(scene, context_paths, METHOD_ORDER_ALL, all_out, "Bonsai free-space floater density", cell=322, compact=False)
    compose_grid(scene, context_paths, METHOD_ORDER_COMPACT, compact_out, "Bonsai free-space floater density", cell=342, compact=True)
    print("wrote", all_out, flush=True)
    print("wrote", compact_out, flush=True)
    print("separate", outdir, flush=True)


if __name__ == "__main__":
    main()
