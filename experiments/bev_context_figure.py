#!/usr/bin/env python3
# Create a context-enhanced bonsai BEV figure without overwriting the original overlay.

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

from main_crop import main_box_proj, proj_from_ref, px_box

VM = Path("/data1/zhangao/outputs/viz_multiangle")
GA = VM / "global_all"
SEP = VM / "SEPARATE" / "BEV"
BASE = Path("/data1/zhangao/outputs/baselines_multi_angle")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BINS = 320
MARGIN = float(os.environ.get("BEV_MARGIN", "2.8"))
MASS = float(os.environ.get("BEV_MASS", "0.88"))
INTERP = os.environ.get("BEV_INTERP", "bicubic")

ORDER = ["3DGS", "Hier", "ScaffoldGS", "MCMC", "CDGS", "PGSR", "MH-3DGS", "MVGS", "Ours"]
NICE = {
    "3DGS": "3DGS",
    "Hier": "Hier-GS",
    "ScaffoldGS": "Scaffold-GS",
    "MCMC": "3DGS-MCMC",
    "CDGS": "CDGS",
    "PGSR": "PGSR",
    "MH-3DGS": "MH-3DGS",
    "MVGS": "MVGS",
    "Ours": "Ours (GS-LSS)",
}
CONTEXT = {
    20: ("bonsai_20", "00005.png", [234, 78, 442, 286]),
    40: ("bonsai_40", "00014.png", [260, 104, 468, 312]),
    60: ("bonsai_60", "00034.png", [286, 78, 494, 286]),
    80: ("bonsai_80", "00044.png", [260, 52, 468, 260]),
    100: ("bonsai_100", "00070.png", [572, 26, 780, 234]),
}

try:
    RESIZE_BICUBIC = Image.Resampling.BICUBIC
    RESIZE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESIZE_BICUBIC = Image.BICUBIC
    RESIZE_LANCZOS = Image.LANCZOS


def safe(name):
    return str(name).replace("/", "-")


def fnt(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def labeled(img, text, sub=None, accent=False, bar=30):
    img = img.convert("RGB")
    width, height = img.size
    out = Image.new("RGB", (width, height + bar), (15, 15, 15))
    out.paste(img, (0, bar))
    draw = ImageDraw.Draw(out)
    draw.text((6, 5), text, fill=((120, 255, 120) if accent else (240, 240, 240)), font=fnt(20))
    if sub:
        font = fnt(16)
        text_w = draw.textlength(sub, font=font)
        draw.text((width - text_w - 6, 7), sub, fill=(255, 220, 120), font=font)
    return out


def load_manifest():
    path = VM / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def manifest_entry(scene, angle, cell, manifest):
    keys = [f"{scene}@{angle}", f"{scene}@{float(angle)}", cell]
    for key in keys:
        if key in manifest:
            return manifest[key]
    return {}


def context_candidates(scene, angle, cell, view, method, manifest):
    candidates = [VM / "cells" / cell / "renders" / method / view]
    entry = manifest_entry(scene, angle, cell, manifest)
    if method == "Ours":
        rdir = entry.get("ours", {}).get("rdir")
        if rdir:
            candidates.append(Path(rdir) / view)
    elif method == "GT":
        gtdir = entry.get("ours", {}).get("gtdir")
        if gtdir:
            candidates.append(Path(gtdir) / view)
    return candidates


def load_context_image(scene, angle, cell, view, method, manifest):
    tried = context_candidates(scene, angle, cell, view, method, manifest)
    for path in tried:
        if path.exists():
            return Image.open(path).convert("RGB"), path, tried
    joined = "\n  ".join(str(p) for p in tried)
    raise FileNotFoundError(f"Missing {method} context image for {cell} {view}. Tried:\n  {joined}")


def boxed_gt(gt, box):
    out = gt.copy()
    draw = ImageDraw.Draw(out)
    width = max(3, gt.size[0] // 200)
    draw.rectangle(tuple(box), outline=(255, 40, 40), width=width)
    return out


def make_context_pair(gt_boxed, ours_crop, angle):
    target_w = 360
    full_h = 240
    crop_h = 208
    full = gt_boxed.copy()
    full.thumbnail((target_w, full_h), RESIZE_LANCZOS)
    full_canvas = Image.new("RGB", (target_w, full_h), (245, 245, 245))
    full_canvas.paste(full, ((target_w - full.width) // 2, (full_h - full.height) // 2))

    crop = ours_crop.resize((target_w, crop_h), RESIZE_BICUBIC)
    top = labeled(full_canvas, f"{angle} deg GT", sub="red box = crop")
    bottom = labeled(crop, "Ours crop", accent=True)
    gap = 10
    out = Image.new("RGB", (target_w, top.height + gap + bottom.height), (255, 255, 255))
    out.paste(top, (0, 0))
    out.paste(bottom, (0, top.height + gap))
    return out


def export_context_images(scene, manifest):
    out_dir = SEP / f"{scene}_context"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = {}
    readme = [
        f"Context exports for BEV_overlay_{scene}_context.png",
        "",
        "Each angle uses a fixed representative view and crop box.",
        "Image loading policy: cells/<cell>/renders/{GT,Ours}/<view>, then manifest ours {gtdir,rdir}.",
        "",
    ]
    for angle, (cell, view, box) in CONTEXT.items():
        gt, gt_path, gt_tried = load_context_image(scene, angle, cell, view, "GT", manifest)
        ours, ours_path, ours_tried = load_context_image(scene, angle, cell, view, "Ours", manifest)
        if ours.size != gt.size:
            ours = ours.resize(gt.size, RESIZE_BICUBIC)
        box = tuple(box)
        gt_boxed = boxed_gt(gt, box)
        gt_crop = gt.crop(box)
        ours_crop = ours.crop(box)
        pair = make_context_pair(gt_boxed, ours_crop, angle)

        gt_boxed.save(out_dir / f"full_boxed_{angle}.png")
        gt_crop.save(out_dir / f"gt_crop_{angle}.png")
        ours_crop.save(out_dir / f"ours_crop_{angle}.png")
        pair.save(out_dir / f"context_pair_{angle}.png")
        pairs[angle] = pair

        readme.extend([
            f"{angle} deg",
            f"  cell: {cell}",
            f"  view: {view}",
            f"  crop_box_xyxy: {list(box)}",
            f"  GT source: {gt_path}",
            f"  Ours source: {ours_path}",
            "  GT candidates:",
        ])
        readme.extend([f"    {p}" for p in gt_tried])
        readme.append("  Ours candidates:")
        readme.extend([f"    {p}" for p in ours_tried])
        readme.append("")
    (out_dir / "README.txt").write_text("\n".join(readme) + "\n")
    return pairs, out_dir


def reference_points(scene):
    paths = sorted((BASE / scene).glob("*/input.ply"))
    if not paths:
        raise FileNotFoundError(f"No input.ply found under {BASE / scene}")
    ply = PlyData.read(paths[0])
    return np.stack([ply["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float32)


def frac(scene, method, angle):
    path = GA / f"{scene}__{method}.frac.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get(str(angle)) or data.get(str(float(angle)))


def load_bev_data(scene):
    angles = [angle for angle in [20, 40, 60, 80, 100] if (GA / f"{scene}__Ours__{angle}.npy").exists()]
    if not angles:
        raise FileNotFoundError(f"No Ours BEV npy files found for {scene} under {GA}")

    ref = reference_points(scene)
    center, axes = proj_from_ref(ref)
    r2 = (ref - center) @ axes.T
    pad = 0.35
    mn = r2.min(0) - pad * (r2.max(0) - r2.min(0))
    mx = r2.max(0) + pad * (r2.max(0) - r2.min(0))
    ref_grid, _, _ = np.histogram2d(r2[:, 0], r2[:, 1], bins=BINS, range=[[mn[0], mx[0]], [mn[1], mx[1]]])
    ref_grid = (ref_grid.T > 0).astype(float)

    cx, cy, half = main_box_proj(ref, center, axes, mass_frac=MASS, margin=MARGIN)
    i0, i1, j0, j1 = px_box(mn, mx, BINS, cx, cy, half)
    ext = [cx - half, cx + half, cy - half, cy + half]
    ref_crop = ref_grid[i0:i1, j0:j1]

    grids = {}
    for method in ORDER:
        for angle in angles:
            path = GA / f"{scene}__{method}__{angle}.npy"
            if not path.exists():
                grids[(method, angle)] = None
                continue
            full = np.load(path)
            fr = frac(scene, method, angle)
            total = float(full.sum())
            total_mass = (total / fr) if (fr and total > 0) else None
            norm = (full / total_mass) if total_mass else full * 0.0
            grids[(method, angle)] = norm[i0:i1, j0:j1]

    all_values = [grid for grid in grids.values() if grid is not None and grid.size]
    if not all_values:
        raise RuntimeError(f"No BEV grids found for {scene}")
    vmax = np.percentile(np.concatenate([g.ravel() for g in all_values]), 99.5) + 1e-9
    return angles, ref_crop, grids, vmax, ext


def panel_rgb(ref_crop, grid, vmax):
    ref = np.clip(ref_crop.astype(float), 0, 1)
    if grid is None:
        comp = np.dstack([ref * 0.22, ref * 0.22, ref * 0.26])
    else:
        hot = plt.get_cmap("inferno")(np.clip(np.sqrt(np.clip(grid.astype(float), 0, None) / vmax), 0, 1))[..., :3]
        comp = np.clip(hot + ref[..., None] * np.array([0.22, 0.22, 0.26]), 0, 1)
    return Image.fromarray((comp[::-1] * 255).astype(np.uint8))


def export_bev_panels(scene, angles, ref_crop, grids, vmax):
    out_dir = SEP / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in ORDER:
        for angle in angles:
            grid = grids[(method, angle)]
            if grid is None:
                continue
            img = panel_rgb(ref_crop, grid, vmax).resize((360, 360), RESIZE_BICUBIC)
            fr = frac(scene, method, angle)
            sub = f"{fr * 100:.0f}% floaters" if fr is not None else None
            labeled(img, f"{NICE[method]} @{angle} deg", sub=sub, accent=(method == "Ours")).save(
                out_dir / f"{safe(NICE[method])}_{angle}.png"
            )
    return out_dir


def draw_composite(scene, context_pairs, angles, ref_crop, grids, vmax, ext):
    nrows = len(ORDER) + 1
    fig, axes = plt.subplots(
        nrows,
        len(angles),
        figsize=(2.35 * len(angles), 2.35 * len(ORDER) + 2.7),
        gridspec_kw={"height_ratios": [1.28] + [1.0] * len(ORDER)},
    )
    if len(angles) == 1:
        axes = axes[:, None]

    for col, angle in enumerate(angles):
        ax = axes[0, col]
        ax.imshow(context_pairs[angle])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{angle} deg gap", fontsize=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if col == 0:
            ax.set_ylabel("RGB context", fontsize=11, fontweight="bold")

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
        "Top row: representative RGB crops for orientation; BEV maps show model-level free-space floater density.",
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

    out = GA / f"BEV_overlay_{scene}_context.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    return out


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "bonsai"
    if scene != "bonsai":
        raise SystemExit("bev_context_figure.py is intentionally configured for bonsai only.")
    manifest = load_manifest()
    context_pairs, context_dir = export_context_images(scene, manifest)
    angles, ref_crop, grids, vmax, ext = load_bev_data(scene)
    missing_context = sorted(set(angles) - set(context_pairs))
    if missing_context:
        raise RuntimeError(f"Missing context pairs for angles: {missing_context}")
    bev_dir = export_bev_panels(scene, angles, ref_crop, grids, vmax)
    out = draw_composite(scene, context_pairs, angles, ref_crop, grids, vmax, ext)
    print(f"wrote {out}")
    print(f"context exports -> {context_dir}")
    print(f"BEV panel exports -> {bev_dir}")


if __name__ == "__main__":
    main()
