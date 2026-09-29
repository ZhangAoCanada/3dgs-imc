#!/usr/bin/env python3
"""BEV map-context v6: publication-oriented bonsai layout.

This keeps the v5 data choices:
  - true COLMAP source cameras from DATA/mipnerf360/bonsai/sparse/0/images.bin
  - held-out membership from the dataset split files
  - body panels loaded from the existing BEV panel exports

Compared with v5, the camera-context cells carry less duplicate text in the
composite and the README records the body-panel sources used in the final PNG.
"""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import bev_map_context_v5_figure as v5

base = v5.base

VM = base.VM
GA = base.GA
SEP = base.SEP
ANGLES = base.ANGLES
ORDER = base.ORDER
NICE = base.NICE
BLUE = base.BLUE
BLUE_FILL = base.BLUE_FILL
ORANGE = base.ORANGE
ORANGE_FILL = base.ORANGE_FILL
INK = base.INK
RESAMPLE = v5.RESAMPLE


def draw_context(ax, bg, crop_ext, geom, angle, legend=False):
    ctx = geom["ctx_ext"]
    cc = geom["crop_center"]
    span = ctx[1] - ctx[0]
    ax.set_facecolor("#fbfaf7")

    corners = np.array(
        [[ctx[0], ctx[2]], [ctx[0], ctx[3]], [ctx[1], ctx[2]], [ctx[1], ctx[3]]],
        dtype=float,
    )
    radius = float(np.linalg.norm(corners - cc[None, :], axis=1).max() * 1.08)
    t1 = math.degrees(geom["gap_dir"] - geom["gap_half"])
    t2 = math.degrees(geom["gap_dir"] + geom["gap_half"])
    ax.add_patch(
        Wedge(
            tuple(cc),
            radius,
            t1,
            t2,
            facecolor=ORANGE,
            edgecolor=ORANGE,
            lw=1.1,
            alpha=0.16,
            zorder=1,
        )
    )

    # Visual orientation inset. It is intentionally not drawn at metric crop
    # scale, otherwise the camera ring becomes unreadable in the top row.
    inset_half = 0.215 * span
    inset = [cc[0] - inset_half, cc[0] + inset_half, cc[1] - inset_half, cc[1] + inset_half]
    ax.imshow(
        np.asarray(bg) / 255.0,
        origin="upper",
        extent=inset,
        aspect="equal",
        interpolation="bicubic",
        zorder=2,
    )
    ax.plot(
        [inset[0], inset[1], inset[1], inset[0], inset[0]],
        [inset[2], inset[2], inset[3], inset[3], inset[2]],
        color=INK,
        lw=1.05,
        zorder=5,
    )

    xy = geom["xy"]
    theta = geom["theta"]
    mask = geom["is_gap"]
    train_size = 0.0120 * span
    gap_size = 0.0148 * span
    for p, t in zip(xy[~mask], theta[~mask]):
        v5.draw_cam(ax, p, t, BLUE, BLUE_FILL, train_size, alpha=0.62, lw=0.42, z=9)
    for p, t in zip(xy[mask], theta[mask]):
        v5.draw_cam(ax, p, t, ORANGE, ORANGE_FILL, gap_size, alpha=1.0, lw=0.66, z=16)
    ax.add_patch(Circle(tuple(cc), radius=0.0045 * span, facecolor=INK, edgecolor="white", lw=0.5, zorder=25))

    ax.text(
        0.045,
        0.945,
        f"gap {geom['n_gap']} / {geom['n_all']} cameras",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.8,
        fontweight="bold",
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=2.2),
        zorder=30,
    )
    if legend:
        x0 = ctx[0] + 0.060 * span
        y0 = ctx[3] - 0.205 * span
        v5.draw_cam(ax, [x0, y0], 0, BLUE, BLUE_FILL, 0.040 * span, 1.0, 0.72, 35)
        v5.draw_cam(ax, [x0, y0 - 0.085 * span], 0, ORANGE, ORANGE_FILL, 0.040 * span, 1.0, 0.78, 35)
        ax.text(x0 + 0.064 * span, y0, "train", ha="left", va="center", fontsize=8.9, color=INK, zorder=36)
        ax.text(
            x0 + 0.064 * span,
            y0 - 0.085 * span,
            "held-out",
            ha="left",
            va="center",
            fontsize=8.9,
            color=INK,
            zorder=36,
        )

    ax.set_xlim(ctx[0], ctx[1])
    ax.set_ylim(ctx[2], ctx[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#3a3a3a")


def save_contexts(scene, crop_ext, geoms, bgs, bg_sources, ref_path):
    outdir = SEP / f"{scene}_mapcontext_v6"
    outdir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Map-context v6 exports for BEV_overlay_{scene}_mapcontext_v6.png",
        "",
        "Publication-oriented continuation of v5: same camera/data sources, less duplicate text in the composite.",
        "All cameras are plotted from the true COLMAP image records.",
        "Held-out membership comes directly from the dataset split files.",
        "The Ours BEV RGB map is a visual orientation inset centered at the scene, not a metric-scale background.",
        "",
        f"script: {ROOT / 'experiments' / 'bev_map_context_v6_figure.py'}",
        f"camera pose source: {base.DATA / scene / 'sparse/0/images.bin'}",
        f"BEV crop reference: {ref_path}",
        f"BEV metric crop extent retained for body panels: {[round(float(x), 6) for x in crop_ext]}",
        "",
    ]
    for angle in ANGLES:
        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=220)
        draw_context(ax, bgs[angle], crop_ext, geoms[angle], angle, legend=(angle == ANGLES[0]))
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(outdir / f"camera_context_{angle}.png", dpi=220)
        plt.close(fig)
        bgs[angle].save(outdir / f"ours_bev_inset_source_{angle}.png")
        geom = geoms[angle]
        lines.extend(
            [
                f"{angle} deg",
                f"  all cameras plotted: {geom['n_all']}",
                f"  train/gap cameras: {geom['n_train']} / {geom['n_gap']}",
                f"  split source: {geom['split']}",
                f"  Ours BEV inset source: {bg_sources[angle]}",
                f"  readable camera context extent: {[round(float(x), 6) for x in geom['ctx_ext']]}",
                f"  gap_dir_rad_from_true_gap_cameras: {geom['gap_dir']:.8f}",
                f"  gap_half_rad_declared: {geom['gap_half']:.8f}",
                "",
            ]
        )
    (outdir / "README.txt").write_text("\n".join(lines) + "\n")
    return outdir


def compose(scene, context_dir, readme_path):
    cell = 360
    gap = 8
    left = 164
    right = 122
    top = 78
    footer = 58
    rows = 1 + len(ORDER)
    cols = len(ANGLES)
    width = left + cols * cell + (cols - 1) * gap + right
    height = top + rows * cell + (rows - 1) * gap + footer
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (left, 18),
        "Free-space floater density with source-defined camera gaps - bonsai",
        font=v5.font(22, True),
        fill=INK,
    )
    for col, angle in enumerate(ANGLES):
        x = left + col * (cell + gap)
        v5.text_center(draw, (x, 42, x + cell, 72), f"{angle} deg gap", v5.font(18, True), INK)

    y0 = top
    draw.text((18, y0 + cell // 2 - 20), "Source\ncamera split", font=v5.font(16, True), fill=INK, spacing=4)
    for col, angle in enumerate(ANGLES):
        img = Image.open(context_dir / f"camera_context_{angle}.png").convert("RGB").resize((cell, cell), RESAMPLE)
        canvas.paste(img, (left + col * (cell + gap), y0))

    panel_sources = {}
    for row, method in enumerate(ORDER, start=1):
        y = top + row * (cell + gap)
        label = NICE[method]
        fill = (180, 30, 45) if method == "Ours" else INK
        font = v5.font(17, method == "Ours")
        bbox = draw.multiline_textbbox((0, 0), label, font=font)
        draw.text((left - 18 - (bbox[2] - bbox[0]), y + cell // 2 - (bbox[3] - bbox[1]) // 2), label, font=font, fill=fill)
        for col, angle in enumerate(ANGLES):
            img, path = base.load_panel(scene, method, angle)
            panel_sources[(method, angle)] = path
            img = img.resize((cell, cell), RESAMPLE)
            img = v5.annotate_pct(img, scene, method, angle)
            canvas.paste(img, (left + col * (cell + gap), y))

    body_y = top + cell + gap
    body_h = len(ORDER) * cell + (len(ORDER) - 1) * gap
    cb = v5.make_colorbar(body_h, 24)
    cbx = left + cols * cell + (cols - 1) * gap + 32
    canvas.paste(cb, (cbx, body_y))
    draw.text((cbx + 34, body_y), "high\n(dense\nfloaters)", font=v5.font(13), fill=INK, spacing=2)
    draw.text((cbx + 34, body_y + body_h - 48), "low\n(clean)", font=v5.font(13), fill=INK, spacing=2)
    draw.text(
        (left, height - 38),
        "Top row: true source-camera split with continuous held-out wedge and Ours BEV RGB inset. Body: brighter heat = denser free-space floaters.",
        font=v5.font(14),
        fill=INK,
    )

    with readme_path.open("a") as handle:
        handle.write("BEV panel sources used in composite:\n")
        for method in ORDER:
            for angle in ANGLES:
                handle.write(f"  {NICE[method]} {angle} deg: {panel_sources[(method, angle)]}\n")

    out = GA / f"BEV_overlay_{scene}_mapcontext_v6.png"
    canvas.save(out)
    return out


def main():
    scene = "bonsai"
    print("v6 CPU-only generator: publication-oriented source-camera context", flush=True)
    center, axes, crop_ext, ref_path = base.crop_extent(scene)
    geoms = v5.adjusted_geoms(base.build_geoms(scene, center, axes, crop_ext))
    print("camera extent", [round(float(x), 3) for x in geoms[20]["ctx_ext"]], "all cameras", geoms[20]["n_all"], flush=True)

    bgs = {}
    bg_sources = {}
    for angle in ANGLES:
        bgs[angle], bg_sources[angle] = base.load_bg(scene, angle)
    outdir = save_contexts(scene, crop_ext, geoms, bgs, bg_sources, ref_path)
    out = compose(scene, outdir, outdir / "README.txt")
    print("wrote", out, flush=True)
    print("separate", outdir, flush=True)


if __name__ == "__main__":
    main()
