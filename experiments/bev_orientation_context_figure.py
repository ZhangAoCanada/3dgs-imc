#!/usr/bin/env python3
"""Create a bonsai BEV figure with an intuitive orientation-map context row.

Layout:
  - top-left: normal Ours RGB reference of the potted bonsai/tabletop
  - top row: BEV orientation maps, one per gap angle, in the exact BEV crop
  - body: existing 9-method BEV floater-density grid
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
from matplotlib.patches import Wedge
from PIL import Image, ImageDraw
from plyfile import PlyData

from bev_context_figure import (
    BASE,
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
    labeled,
    load_bev_data,
    reference_points,
)
from main_crop import main_box_proj, proj_from_ref

DIAG = Path("/data1/zhangao/outputs/diag/mipnerf360")
BEST_COVERAGE = Path("/data1/zhangao/outputs/best_coverage_gap")
DATA_ROOT = Path("/data1/zhangao/DATA/mipnerf360")


def parse_manifest(path):
    data = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def best_source_dir(scene, angle):
    manifest = BEST_COVERAGE / f"{scene}_angle_{float(angle):.1f}" / "MANIFEST.txt"
    info = parse_manifest(manifest)
    candidates = []
    source = info.get("source_dir")
    if source:
        candidates.append(Path(source))
    candidates.extend(sorted(DIAG.glob(f"{scene}_angle_{float(angle):.1f}*")))
    for path in candidates:
        if (path / "cameras.json").exists():
            return path, manifest
    tried = "\n  ".join(str(p) for p in candidates) or "<none>"
    raise FileNotFoundError(f"No cameras.json source for {scene}@{angle}. Tried:\n  {tried}")


def split_names(scene, angle):
    path = DATA_ROOT / scene / f"test_selection_angle_{float(angle):.1f}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    names = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        stem = Path(line.strip()).stem
        names.add(stem)
    return names, path


def camera_geometry(scene, angle, center, axes, display_center):
    src, manifest = best_source_dir(scene, angle)
    cam_path = src / "cameras.json"
    if not cam_path.exists():
        raise FileNotFoundError(cam_path)
    test, split_path = split_names(scene, angle)
    cams = json.loads(cam_path.read_text())
    pos = []
    names = []
    for cam in cams:
        name = Path(cam["img_name"]).stem
        names.append(name)
        pos.append(cam["position"])
    pos = np.asarray(pos, dtype=np.float32)
    cp = (pos - center) @ axes.T
    is_test = np.asarray([name in test for name in names], dtype=bool)
    if not np.any(is_test):
        raise RuntimeError(f"No cameras matched split {split_path} in {cam_path}")

    # Use directions from the actual camera ring, but draw them on the cropped BEV frame boundary.
    scene_center = np.median(cp, axis=0)
    vec = cp - scene_center
    cam_ang = np.arctan2(vec[:, 1], vec[:, 0])
    test_ang = cam_ang[is_test]
    gap_dir = float(np.arctan2(np.sin(test_ang).mean(), np.cos(test_ang).mean()))
    gap_half = float(np.percentile(np.abs(np.angle(np.exp(1j * (test_ang - gap_dir)))), 85))
    return {
        "train_angles": cam_ang[~is_test],
        "test_angles": cam_ang[is_test],
        "gap_dir": gap_dir,
        "gap_half": gap_half,
        "source_dir": src,
        "manifest": manifest,
        "cameras_json": cam_path,
        "split": split_path,
        "n_train": int((~is_test).sum()),
        "n_test": int(is_test.sum()),
        "display_center": np.asarray(display_center, dtype=float),
    }


def rgb_reference(scene):
    candidates = [
        VM / "cells" / f"{scene}_20" / "renders" / "Ours" / "00005.png",
        VM / "cells" / f"{scene}_40" / "renders" / "Ours" / "00014.png",
        VM / "cells" / f"{scene}_100" / "renders" / "Ours" / "00070.png",
        VM / "cells" / f"{scene}_20" / "renders" / "GT" / "00005.png",
    ]
    for path in candidates:
        if path.exists():
            img = Image.open(path).convert("RGB")
            return img, path, candidates
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No RGB reference found. Tried:\n  {tried}")


def boxed_rgb_reference(scene):
    img, path, candidates = rgb_reference(scene)
    # Fixed box from the original context plan for a recognizable potted-bonsai crop in the reference view.
    box = (234, 78, 442, 286)
    boxed = img.copy()
    draw = ImageDraw.Draw(boxed)
    draw.rectangle(box, outline=(255, 45, 45), width=max(3, img.size[0] // 220))
    crop = img.crop(box).resize((360, 360), RESIZE_BICUBIC)
    full = boxed.copy()
    full.thumbnail((360, 250), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
    canvas = Image.new("RGB", (360, 390), (255, 255, 255))
    labeled_full = labeled(Image.new("RGB", (360, 250), (245, 245, 245)), "RGB reference", sub="red box = crop")
    labeled_full.paste(full, ((360 - full.width) // 2, 30 + (250 - full.height) // 2))
    # Use the crop as the exported reference, but the composite top-left uses the full boxed frame.
    return boxed, crop, labeled_full, path, candidates, box


def draw_orientation_map(ax, ref_crop, ext, geom, angle):
    cx = (ext[0] + ext[1]) / 2.0
    cy = (ext[2] + ext[3]) / 2.0
    half = (ext[1] - ext[0]) / 2.0
    radius = 0.91 * half
    ring_radius = 0.86 * half

    ax.imshow(
        np.clip(ref_crop, 0, 1),
        origin="lower",
        extent=ext,
        cmap="Greys",
        vmin=0,
        vmax=1.15,
        alpha=0.38,
        aspect="equal",
        interpolation=INTERP,
    )
    theta1 = math.degrees(geom["gap_dir"] - geom["gap_half"])
    theta2 = math.degrees(geom["gap_dir"] + geom["gap_half"])
    ax.add_patch(Wedge((cx, cy), radius, theta1, theta2, facecolor="#ff9f1a", alpha=0.26,
                       edgecolor="#ff7a00", linewidth=1.2, zorder=3))
    for arr, color, marker, size, alpha in [
        (geom["train_angles"], "#1f77ff", "o", 10, 0.78),
        (geom["test_angles"], "#ff7a00", "x", 24, 0.95),
    ]:
        xs = cx + ring_radius * np.cos(arr)
        ys = cy + ring_radius * np.sin(arr)
        ax.scatter(xs, ys, s=size, c=color, marker=marker, alpha=alpha, linewidths=1.0, zorder=5)

    # Strong arrow into the held-out sector; this is the thing readers should map to the heatmaps below.
    gd = geom["gap_dir"]
    ax.annotate(
        "",
        xy=(cx + 0.78 * half * np.cos(gd), cy + 0.78 * half * np.sin(gd)),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color="#ff4a00", lw=2.8),
        zorder=6,
    )
    ax.text(0.04, 0.94, f"{angle} deg", transform=ax.transAxes, ha="left", va="top",
            fontsize=11, fontweight="bold", color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=2))
    ax.text(0.04, 0.05, "blue=train  orange=gap", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color="black", bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=2))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#444444")


def save_orientation_png(path, ref_crop, ext, geom, angle):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    draw_orientation_map(ax, ref_crop, ext, geom, angle)
    fig.tight_layout(pad=0.05)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def export_orientation_assets(scene, angles, ref_crop, ext, geoms):
    out_dir = SEP / f"{scene}_orientation"
    out_dir.mkdir(parents=True, exist_ok=True)
    boxed, crop, labeled_full, rgb_path, rgb_candidates, box = boxed_rgb_reference(scene)
    boxed.save(out_dir / "rgb_reference_full_boxed.png")
    crop.save(out_dir / "rgb_reference_crop.png")
    labeled_full.save(out_dir / "rgb_reference_labeled.png")
    for angle in angles:
        save_orientation_png(out_dir / f"orientation_map_{angle}.png", ref_crop, ext, geoms[angle], angle)
    readme = [
        f"Orientation-map context exports for BEV_overlay_{scene}_orientcontext.png",
        "",
        "Top-row maps share the exact BEV crop/extent used by the floater heatmaps.",
        "Gray = SfM scene footprint, blue dots = training camera directions, orange sector/x marks = held-out gap directions.",
        "The RGB reference is only a human-recognizable perspective inset; it is not a projection source.",
        "",
        f"RGB reference: {rgb_path}",
        f"RGB crop box xyxy: {list(box)}",
        "RGB candidates:",
    ]
    readme.extend([f"  {p}" for p in rgb_candidates])
    readme.append("")
    for angle in angles:
        g = geoms[angle]
        readme.extend([
            f"{angle} deg",
            f"  source_dir: {g['source_dir']}",
            f"  best manifest: {g['manifest']}",
            f"  cameras_json: {g['cameras_json']}",
            f"  split: {g['split']}",
            f"  train/test cameras: {g['n_train']} / {g['n_test']}",
            f"  gap_dir_rad: {g['gap_dir']:.6f}",
            f"  gap_half_rad: {g['gap_half']:.6f}",
            "",
        ])
    (out_dir / "README.txt").write_text("\n".join(readme) + "\n")
    return out_dir, labeled_full


def draw_composite(scene, angles, ref_crop, grids, vmax, ext, geoms, rgb_panel):
    nrows = len(ORDER) + 1
    ncols = len(angles) + 1
    fig = plt.figure(figsize=(2.05 + 2.35 * len(angles), 2.35 * len(ORDER) + 2.45))
    gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        width_ratios=[0.92] + [1.0] * len(angles),
        height_ratios=[1.05] + [1.0] * len(ORDER),
        wspace=0.05,
        hspace=0.08,
    )

    ax_ref = fig.add_subplot(gs[0, 0])
    ax_ref.imshow(rgb_panel)
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])
    ax_ref.set_title("What it is", fontsize=11, fontweight="bold")
    for spine in ax_ref.spines.values():
        spine.set_visible(False)

    for col, angle in enumerate(angles, start=1):
        ax = fig.add_subplot(gs[0, col])
        draw_orientation_map(ax, ref_crop, ext, geoms[angle], angle)
        ax.set_title(f"{angle} deg gap", fontsize=12)

    for row, method in enumerate(ORDER, start=1):
        label_ax = fig.add_subplot(gs[row, 0])
        label_ax.axis("off")
        label_ax.text(
            0.98,
            0.5,
            NICE[method],
            ha="right",
            va="center",
            fontsize=11,
            fontweight=("bold" if method == "Ours" else "normal"),
            color=("crimson" if method == "Ours" else "black"),
        )
        for col, angle in enumerate(angles, start=1):
            ax = fig.add_subplot(gs[row, col])
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

    fig.suptitle(
        f"Free-space floater density (top-down BEV) - {scene}    (% = share of model Gaussians in free space)",
        fontsize=13,
        y=0.997,
    )
    fig.text(
        0.52,
        0.018,
        "Top row: normal RGB reference plus BEV orientation maps. Body: brighter heat = denser free-space floaters.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.035, right=0.90, top=0.965, bottom=0.045)

    cax = fig.add_axes([0.918, 0.16, 0.014, 0.60])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="inferno"), cax=cax)
    cb.set_ticks([0.03, 0.97])
    cb.set_ticklabels(["low\n(clean)", "high\n(dense\nfloaters)"])
    cb.ax.tick_params(labelsize=9)
    cb.set_label("free-space floater density -> brighter = more floaters", fontsize=10)

    out = GA / f"BEV_overlay_{scene}_orientcontext.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    return out


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "bonsai"
    if scene != "bonsai":
        raise SystemExit("bev_orientation_context_figure.py is intentionally configured for bonsai only.")

    angles, ref_crop, grids, vmax, ext = load_bev_data(scene)
    ref = reference_points(scene)
    center, axes = proj_from_ref(ref)
    display_center = ((ext[0] + ext[1]) / 2.0, (ext[2] + ext[3]) / 2.0)
    geoms = {angle: camera_geometry(scene, angle, center, axes, display_center) for angle in angles}
    orient_dir, rgb_panel = export_orientation_assets(scene, angles, ref_crop, ext, geoms)
    bev_dir = export_bev_panels(scene, angles, ref_crop, grids, vmax)
    out = draw_composite(scene, angles, ref_crop, grids, vmax, ext, geoms, rgb_panel)
    print(f"wrote {out}")
    print(f"orientation exports -> {orient_dir}")
    print(f"BEV panel exports -> {bev_dir}")


if __name__ == "__main__":
    main()
