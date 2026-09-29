#!/usr/bin/env python3
"""Improved, INTERPRETABLE all-method BEV for one scene: faint top-down scene footprint (SfM
reference, gray) UNDER each method's free-space floaters (hot), so a viewer can tell what they are
looking at. Reuses the existing global_all/<scene>__<method>__<angle>.npy (made by floaters_export
with a DETERMINISTIC axis-aligned projection from input.ply) and recomputes the matching reference
footprint from the same input.ply. Usage: bev_grid_overlay.py <scene>"""
import os, sys, glob, json
import numpy as np
from plyfile import PlyData
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_crop import proj_from_ref, main_box_proj, px_box
# MARGIN/MASS env loosen the crop (user: "cropped too much, blurry"): a wider box keeps far more native
# floater bins in view so we can drop the heavy upsample+gaussian (the real blur) and let matplotlib's
# bicubic display interpolate sharply. INTERP applies to all panels.
MARGIN = float(os.environ.get("BEV_MARGIN", "2.8"))
MASS = float(os.environ.get("BEV_MASS", "0.88"))
INTERP = os.environ.get("BEV_INTERP", "bicubic")

GA = "/data1/zhangao/outputs/viz_multiangle/global_all"
scene = sys.argv[1] if len(sys.argv) > 1 else "bonsai"
# only angles where OUR data exists, so no "n/a" panels for Ours
ANGLES = [a for a in [20, 40, 60, 80, 100]
          if os.path.exists(f"{GA}/{scene}__Ours__{a}.npy")]
ORDER = ["3DGS", "Hier", "ScaffoldGS", "MCMC", "CDGS", "PGSR", "MH-3DGS", "MVGS", "Ours"]
NICE = {"3DGS": "3DGS", "Hier": "Hier-GS", "ScaffoldGS": "Scaffold-GS", "MCMC": "3DGS-MCMC",
        "CDGS": "CDGS", "PGSR": "PGSR", "MH-3DGS": "MH-3DGS", "MVGS": "MVGS", "Ours": "Ours (GS-LSS)"}
BINS = 320

# reference footprint in the SAME deterministic projection floaters_export used
refp = sorted(glob.glob(f"/data1/zhangao/outputs/baselines_multi_angle/{scene}/*/input.ply"))
ref = np.stack([PlyData.read(refp[0])["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float32)
center, axes = proj_from_ref(ref)
r2 = (ref - center) @ axes.T
pad = 0.35; mn = r2.min(0) - pad*(r2.max(0)-r2.min(0)); mx = r2.max(0) + pad*(r2.max(0)-r2.min(0))
refH, _, _ = np.histogram2d(r2[:, 0], r2[:, 1], bins=BINS, range=[[mn[0], mx[0]], [mn[1], mx[1]]])
refH = (refH.T > 0).astype(float)   # scene footprint mask
# MAIN-REGION crop (object + surrounding floater zone): square box in proj coords -> pixel box
cx, cy, half = main_box_proj(ref, center, axes, mass_frac=MASS, margin=MARGIN)
i0, i1, j0, j1 = px_box(mn, mx, BINS, cx, cy, half)
EXT = [cx-half, cx+half, cy-half, cy+half]
refH = refH[i0:i1, j0:j1]

def frac(m, a):
    f = f"{GA}/{scene}__{m}.frac.json"
    if os.path.exists(f):
        d = json.load(open(f)); return d.get(str(a)) or d.get(str(float(a)))
    return None
# FRACTION-DENSITY normalization: divide each method's floater histogram by its TOTAL model mass
# (= full_grid_sum / floater_fraction) so a grid SUMS to the method's floater fraction. This is
# size-independent and puts opacity-weighted methods and COUNT-weighted Scaffold-GS on one comparable
# scale (Scaffold has no per-Gaussian opacity), while keeping the cross-method comparison (Ours stays
# darkest). Shared vmax across the normalized grids.
grids = {}
for m in ORDER:
    for a in ANGLES:
        p = f"{GA}/{scene}__{m}__{a}.npy"
        if not os.path.exists(p):
            grids[(m, a)] = None; continue
        full = np.load(p); fr = frac(m, a); s = float(full.sum())
        tm = (s/fr) if (fr and s > 0) else None
        gn = (full/tm) if tm else full*0.0
        grids[(m, a)] = gn[i0:i1, j0:j1]
allv = [g for g in grids.values() if g is not None and g.size]
vmax = np.percentile(np.concatenate([g.ravel() for g in allv]), 99.5) + 1e-9
refS = refH.astype(float)   # native footprint; bicubic display interpolation keeps it crisp

fig, axx = plt.subplots(len(ORDER), len(ANGLES), figsize=(2.35*len(ANGLES), 2.35*len(ORDER)))
for i, m in enumerate(ORDER):
    for j, a in enumerate(ANGLES):
        ax = axx[i, j]; g = grids[(m, a)]; ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(np.clip(refS, 0, 1), origin="lower", extent=EXT, cmap="Greys", vmin=0, vmax=1.2,
                  alpha=0.30, aspect="equal", interpolation=INTERP)
        if g is not None:
            gs = np.sqrt(np.clip(g.astype(float), 0, None)/vmax)   # SHARED scale (vmax across all panels)
            ax.imshow(np.clip(gs, 0, 1), origin="lower", extent=EXT, cmap="inferno", vmin=0, vmax=1,
                      alpha=0.92, aspect="equal", interpolation=INTERP)
            fr = frac(m, a)
            if fr is not None:
                ax.text(.96, .04, f"{fr*100:.0f}%", color=("#39ff14" if m == "Ours" else "w"),
                        ha="right", va="bottom", transform=ax.transAxes, fontsize=11, fontweight="bold")
        else:
            ax.text(.5, .5, "n/a", color="gray", ha="center", transform=ax.transAxes)
        ax.set_xlim(cx-half, cx+half); ax.set_ylim(cy-half, cy+half)
        if i == 0: ax.set_title(f"{a}° gap", fontsize=12)
        if j == 0: ax.set_ylabel(NICE[m], fontsize=11, fontweight=("bold" if m == "Ours" else "normal"),
                                 color=("crimson" if m == "Ours" else "black"))
fig.suptitle(f"Free-space floater density (top-down BEV) — {scene}    "
             f"(% = share of model Gaussians in free space)", fontsize=13, y=0.997)
plt.tight_layout(rect=[0, 0, 0.9, 0.985])
# temperature bar: which colour = more / fewer floaters
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
cax = fig.add_axes([0.915, 0.18, 0.014, 0.64])
cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="inferno"), cax=cax)
cb.set_ticks([0.03, 0.97]); cb.set_ticklabels(["low\n(clean)", "high\n(dense\nfloaters)"])
cb.ax.tick_params(labelsize=9)
cb.set_label("free-space floater density  →  brighter = more floaters", fontsize=10)
out = f"{GA}/BEV_overlay_{scene}.png"; plt.savefig(out, dpi=115); plt.close()
print("wrote", out)
