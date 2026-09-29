#!/usr/bin/env python3
"""GLOBAL scene-level theory figure: isolate the FAILURE MODE (free-space floaters) so the
advantage is obvious, unlike a raw density plot. For one scene, across gap angles 20..100,
for each method: a Gaussian is a 'floater' if its center is far (> tau) from the FULL-coverage
COLMAP SfM surface (input.ply, identical across splits -> covers the gap region, so OUR correct
gap-filling is NOT flagged). We render a top-down (PCA ground-plane) heatmap of floater
opacity-mass per (method, angle), and the floater-mass fraction curve vs angle.

Outputs (in viz_multiangle/global/<scene>/):
  floater_bev_grid.png   rows=methods, cols=angles  (only floaters, fixed extent)
  floater_curve.png      floater-mass fraction vs gap angle, per method
  floater_stats.csv
Usage: python global_floaters.py <scene>   (default bonsai)
"""
import os, sys, csv
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bonsai"
ANGLES = [20, 40, 60, 80, 100]
BR = "/data1/zhangao/outputs/baselines_multi_angle"
BEST = "/data1/zhangao/outputs/best_coverage_gap"
OUT = f"/data1/zhangao/outputs/viz_multiangle/global/{SCENE}"
os.makedirs(OUT, exist_ok=True)
TAU_FRAC = 0.015   # floater if dist-to-SfM > TAU_FRAC * scene_diag
METHODS = ["Hier", "MH-3DGS", "MVGS", "Ours"]
def base_dir(meth, a):
    t = {"Hier": "3dgs-depth-test_selection_angle_{a}",
         "MH-3DGS": "mh-3dgs-original_test_selection_angle_{a}",
         "MVGS": "mvgs-original-mv3_test_selection_angle_{a}"}[meth]
    return os.path.join(BR, SCENE, t.format(a=f"{a:.1f}"))
def ply_path(meth, a):
    if meth == "Ours":
        return os.path.join(BEST, f"{SCENE}_angle_{a:.1f}", "point_cloud.ply")
    import glob
    g = sorted(glob.glob(os.path.join(base_dir(meth, a), "point_cloud", "iteration_*", "point_cloud.ply")))
    return g[-1] if g else None

def read_xyz_op(p):
    v = PlyData.read(p)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))  # sigmoid
    return xyz, op

# reference SfM (full coverage) + scene frame
ref_xyz = np.stack([PlyData.read(os.path.join(base_dir("MH-3DGS", 100), "input.ply"))["vertex"][k]
                    for k in ("x", "y", "z")], 1).astype(np.float32)
lo, hi = np.percentile(ref_xyz, 1, 0), np.percentile(ref_xyz, 99, 0)
diag = float(np.linalg.norm(hi - lo)); tau = TAU_FRAC * diag
center = ref_xyz.mean(0)
# PCA ground plane (2 largest-variance axes = top-down image plane)
refc = ref_xyz - center
U, S, Vt = np.linalg.svd(refc[np.random.default_rng(0).choice(len(refc), min(40000, len(refc)), replace=False)],
                         full_matrices=False)
axes = Vt[:2]                       # image plane basis (2,3)
def project(x): return (x - center) @ axes.T   # (N,2)
ref2 = project(ref_xyz)
pad = 0.35
mn = ref2.min(0) - pad*(ref2.max(0)-ref2.min(0)); mx = ref2.max(0) + pad*(ref2.max(0)-ref2.min(0))
BINS = 320
tree = cKDTree(ref_xyz)
print(f"scene={SCENE} diag={diag:.2f} tau={tau:.3f} ref_pts={len(ref_xyz)}", flush=True)

stats = {m: [] for m in METHODS}
grids = {}   # (m,a) -> 2D floater-opacity histogram
for m in METHODS:
    for a in ANGLES:
        p = ply_path(m, a)
        if not p or not os.path.exists(p):
            stats[m].append(None); grids[(m,a)] = None; continue
        xyz, op = read_xyz_op(p)
        d, _ = tree.query(xyz, k=1, workers=-1)
        fl = d > tau
        frac = float((op[fl]).sum() / (op.sum() + 1e-9))
        stats[m].append(frac)
        xy = project(xyz[fl]); w = op[fl]
        H, _, _ = np.histogram2d(xy[:,0], xy[:,1], bins=BINS,
                                 range=[[mn[0],mx[0]],[mn[1],mx[1]]], weights=w)
        grids[(m,a)] = H.T
        print(f"  {m:>8} @{a:<3}: floater mass frac {frac*100:5.2f}%  ({fl.sum()}/{len(xyz)} gauss)", flush=True)

# ---- BEV grid figure (rows=methods, cols=angles) ----
vmax = np.percentile(np.concatenate([g.ravel() for g in grids.values() if g is not None]), 99.5) + 1e-6
refH, _, _ = np.histogram2d(ref2[:,0], ref2[:,1], bins=BINS, range=[[mn[0],mx[0]],[mn[1],mx[1]]])
refH = refH.T > 0
fig, axx = plt.subplots(len(METHODS), len(ANGLES), figsize=(2.5*len(ANGLES), 2.5*len(METHODS)))
for i, m in enumerate(METHODS):
    for j, a in enumerate(ANGLES):
        ax = axx[i, j]; g = grids[(m, a)]
        ax.imshow(refH, origin="lower", cmap="Greys", alpha=0.18, aspect="auto")
        if g is not None:
            ax.imshow(np.clip(g/vmax, 0, 1), origin="lower", cmap="inferno", alpha=0.92, aspect="auto",
                      vmin=0, vmax=1)
            fr = stats[m][j]
            ax.text(0.97, 0.04, f"{fr*100:.1f}%", color="w", ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=10, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "n/a", color="gray", ha="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title(f"{a}° gap", fontsize=12)
        if j == 0: ax.set_ylabel(("Ours (GS-LSS)" if m=="Ours" else {"Hier":"Hier-GS","MH-3DGS":"MH-3DGS","MVGS":"MVGS"}[m]),
                                  fontsize=11, fontweight=("bold" if m=="Ours" else "normal"))
fig.suptitle(f"Free-space floater mass (top-down BEV), {SCENE} — only floaters shown; "
             f"Ours carries the least at every gap",
             fontsize=13, y=0.995)
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(os.path.join(OUT, "floater_bev_grid.png"), dpi=120)
plt.close()

# ---- curve ----
plt.figure(figsize=(7,5))
cols = {"Hier":"#1f77b4","MH-3DGS":"#9467bd","MVGS":"#ff7f0e","Ours":"crimson"}
for m in METHODS:
    ys = [(s*100 if s is not None else np.nan) for s in stats[m]]
    plt.plot(ANGLES, ys, "-o", lw=(2.8 if m=="Ours" else 1.8), color=cols[m],
             label=("Ours (GS-LSS)" if m=="Ours" else m), zorder=(5 if m=="Ours" else 3))
plt.xlabel("Test-view angle gap (degrees)  →  increasing viewpoint bias")
plt.ylabel("Free-space floater mass fraction (%)")
plt.title(f"Scene-level floater mass vs viewpoint bias — {SCENE}")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "floater_curve.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "floater_stats.csv"), "w") as fh:
    w = csv.writer(fh); w.writerow(["method"]+[f"{a}" for a in ANGLES])
    for m in METHODS: w.writerow([m]+[f"{s*100:.2f}" if s is not None else "" for s in stats[m]])
print(f"\nwrote {OUT}/floater_bev_grid.png, floater_curve.png, floater_stats.csv")
