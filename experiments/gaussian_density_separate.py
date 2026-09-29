#!/usr/bin/env python3
"""Save the Gaussian-density plots SEPARATELY: one individual PNG per (method, angle) for a scene,
cropped to the main region (sharp, bicubic) with the gap (red) / train (blue) arrows and the gap/obs
ratio burned in — same look as a single cell of GAUSSIAN_DENSITY_<scene>.png. Also writes a per-method
strip (all angles in a row). Outputs under viz_multiangle/gdens/SEPARATE/<scene>/.
Usage: gaussian_density_separate.py <scene> [a1,a2,..] [ref_ply]"""
import os, sys, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plyfile import PlyData
from main_crop import proj_from_ref, main_box_proj, px_box

MARGIN = float(os.environ.get("DENS_MARGIN", "4.5"))
MASS = float(os.environ.get("DENS_MASS", "0.86"))
INTERP = os.environ.get("DENS_INTERP", "bicubic")
GD = "/data1/zhangao/outputs/viz_multiangle/gdens"
scene = sys.argv[1] if len(sys.argv) > 1 else "truck_resize"
ANGLES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [20, 40, 60, 80, 100]
REFPLY = sys.argv[3] if len(sys.argv) > 3 else (
    f"/data1/zhangao/DATA/tandt/{scene}/sparse/0/points3D.ply" if "resize" in scene
    else f"/data1/zhangao/DATA/mipnerf360/{scene}/sparse/0/points3D.ply")
DET = ["3DGS", "Hier", "ScaffoldGS", "MVGS", "CDGS", "PGSR"]; STO = ["MCMC", "MH-3DGS", "Ours"]
NICE = {"3DGS": "3DGS (det.)", "Hier": "Hier-GS (det.)", "ScaffoldGS": "Scaffold-GS (det.)",
        "MVGS": "MVGS (det.)", "CDGS": "CDGS (det.)", "PGSR": "PGSR (det.)",
        "MCMC": "3DGS-MCMC (stoch.)", "MH-3DGS": "MH-3DGS (stoch.)", "Ours": "Ours / GS-LSS (stoch.)"}
def npy(m, a): return f"{GD}/{scene}__{m}__{a}.dens.npy"
ORDER = [m for m in DET+STO if any(os.path.exists(npy(m, a)) for a in ANGLES)]
geo = json.load(open(f"{GD}/{scene}__GEO.json")); mn = np.array(geo["mn"]); mx = np.array(geo["mx"])
ref = np.stack([PlyData.read(REFPLY)["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float32)
center, axes = proj_from_ref(ref); cx, cy, half = main_box_proj(ref, center, axes, mass_frac=MASS, margin=MARGIN)
refH0 = np.load(f"{GD}/{scene}__REF.npy"); B0 = refH0.shape[0]
i0, i1, j0, j1 = px_box(mn, mx, B0, cx, cy, half)
EXT = [cx-half, cx+half, cy-half, cy+half]; refC = refH0[i0:i1, j0:j1].astype(float)
def cropd(m, a):
    p = npy(m, a); return np.load(p)[i0:i1, j0:j1] if os.path.exists(p) else None
def gapr(m, a):
    f = f"{GD}/{scene}__{m}.gap.json"
    if os.path.exists(f): d = json.load(open(f)); return d.get(f"{a}.0") or d.get(str(a))
    return None
OUT = f"{GD}/SEPARATE/{scene}"; os.makedirs(OUT, exist_ok=True)

def draw(ax, m, a, title=True):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    ax.imshow(np.clip(refC*0.6, 0, 1), origin="lower", extent=EXT, cmap="Greys", vmin=0, vmax=1,
              aspect="equal", interpolation=INTERP)
    g = cropd(m, a)
    if g is not None and (g > 0).any():
        gv = np.sqrt(np.clip(g.astype(float), 0, None)); gv = np.clip(gv/(np.percentile(gv[gv > 0], 98)+1e-9), 0, 1)
        ax.imshow(gv, origin="lower", extent=EXT, cmap="magma", vmin=0, vmax=1, alpha=0.95,
                  aspect="equal", interpolation=INTERP)
    gi = geo["angles"].get(f"{a}.0")
    if gi:
        gd = gi["gap_dir"]; L = 0.78*half
        tr = np.array(gi["train_xy"]); tv = tr.mean(0)-np.array([cx, cy]); trm = np.arctan2(tv[1], tv[0])
        ax.annotate("", xy=(cx+L*np.cos(gd), cy+L*np.sin(gd)), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color="#ff3b3b", lw=3), zorder=6)
        ax.text(cx+L*np.cos(gd), cy+L*np.sin(gd), " gap", color="#ff3b3b", fontsize=11, fontweight="bold", zorder=7)
        ax.annotate("", xy=(cx+0.6*half*np.cos(trm), cy+0.6*half*np.sin(trm)), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color="#3aa0ff", lw=2.2), zorder=6)
    ax.set_xlim(cx-half, cx+half); ax.set_ylim(cy-half, cy+half)
    r = gapr(m, a)
    if r is not None:
        ax.text(.97, .03, f"gap/obs {r:.2f}", color=("#39ff14" if m == "Ours" else "w"),
                ha="right", va="bottom", transform=ax.transAxes, fontsize=12, fontweight="bold")
    if title:
        ax.set_title(f"{NICE[m]} — {a}° gap", fontsize=11,
                     color=("crimson" if m == "Ours" else "black"), fontweight=("bold" if m == "Ours" else "normal"))

n_ind = 0
for m in ORDER:
    # individual per-(method,angle) panels
    for a in ANGLES:
        if cropd(m, a) is None: continue
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        draw(ax, m, a)
        plt.tight_layout()
        plt.savefig(f"{OUT}/{m.replace('/','-')}__{a}.png", dpi=150, facecolor="white"); plt.close(); n_ind += 1
    # per-method strip (all angles)
    av = [a for a in ANGLES if cropd(m, a) is not None]
    if av:
        fig, axx = plt.subplots(1, len(av), figsize=(3.0*len(av), 3.2), squeeze=False)
        for j, a in enumerate(av):
            draw(axx[0, j], m, a, title=False); axx[0, j].set_title(f"{a}° gap", fontsize=12)
        fig.suptitle(f"{NICE[m]} — {scene.replace('_resize','')} Gaussian density vs gap", fontsize=12, y=1.02)
        plt.tight_layout(); plt.savefig(f"{OUT}/STRIP_{m.replace('/','-')}.png", dpi=140,
                                        facecolor="white", bbox_inches="tight"); plt.close()
print(f"{scene}: wrote {n_ind} individual panels + {len(ORDER)} per-method strips -> {OUT}/  (methods={ORDER})")
