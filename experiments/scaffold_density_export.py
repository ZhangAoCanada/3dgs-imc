#!/usr/bin/env python3
"""Add Scaffold-GS to the Gaussian-DENSITY figure (truck). Reconstructs the offset Gaussian centers
(anchor + offset*exp(scale[:3]), no MLP needed) and histograms them in the SAME projection/GEO as the
other methods (count-weighted, since per-Gaussian opacity needs the MLP). Also computes a count-based
gap/obs ratio using the authoritative GEO gap geometry. Usage: scaffold_density_export.py [scene]"""
import os, sys, glob, json
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_crop import proj_from_ref
# env overrides let this run on whichever machine holds the Scaffold PLYs (truck Scaffold is on hust2)
GD = os.environ.get("SDE_GD", "/data1/zhangao/outputs/viz_multiangle/gdens")   # holds GEO + outputs
scene = sys.argv[1] if len(sys.argv) > 1 else "truck_resize"
REFPLY = os.environ.get("SDE_REF") or (
    f"/data1/zhangao/DATA/tandt/{scene}/sparse/0/points3D.ply" if "resize" in scene
    else f"/data1/zhangao/DATA/mipnerf360/{scene}/sparse/0/points3D.ply")
SCAFGLOB = os.environ.get("SDE_SCAFGLOB",
    f"/data1/zhangao/outputs/baselines_multi_angle/{scene}/scaffoldgs-original*test_selection_angle_{{a}}.0/point_cloud/iteration_*/point_cloud.ply")
ANGLES = [20, 40, 60, 80, 100]; BINS = 360
def first(g):
    h = sorted(glob.glob(g)); return h[-1] if h else None
def scaffold_centers(p):
    v = PlyData.read(p)["vertex"]; N = len(v["x"])
    anchor = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    noff = sum(1 for pr in v.properties if pr.name.startswith("f_offset_")); k = noff // 3
    off = np.stack([v[f"f_offset_{i}"] for i in range(noff)], 1).astype(np.float32).reshape(N, 3, k).transpose(0, 2, 1)
    sc = np.stack([v[f"scale_{i}"] for i in range(6)], 1).astype(np.float32)
    cen = (anchor[:, None, :] + off*np.exp(sc[:, :3])[:, None, :]).reshape(-1, 3)
    return cen[np.isfinite(cen).all(1)]
geo = json.load(open(f"{GD}/{scene}__GEO.json")); mn = np.array(geo["mn"]); mx = np.array(geo["mx"])
scenter = np.array(geo["scenter"], np.float32); near = 0.04*geo["diag"]
ref = np.stack([PlyData.read(REFPLY)["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float32)
center, axes = proj_from_ref(ref); tree = cKDTree(ref)
def ang_of(p): d = p - scenter; return np.arctan2(d[..., 1], d[..., 0])
def adiff(a, b): return np.abs(np.angle(np.exp(1j*(a-b))))
gap = {}
for a in ANGLES:
    astr = f"{a:.1f}"
    p = first(SCAFGLOB.replace("{a}", str(a)))
    if not p:
        print(f"MISS {scene}/ScaffoldGS@{a}", flush=True); continue
    cen = scaffold_centers(p); xy = (cen - center) @ axes.T
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=BINS, range=[[mn[0], mx[0]], [mn[1], mx[1]]])
    np.save(f"{GD}/{scene}__ScaffoldGS__{a}.dens.npy", H.T.astype(np.float32))
    if astr in geo["angles"]:
        gm = geo["angles"][astr]["gap_dir"]; hf = geo["angles"][astr]["gap_half"]
        d, _ = tree.query(cen, k=1, workers=-1); ns = d < near; aa = ang_of(xy); ing = adiff(aa, gm) < hf
        gd = (ns & ing).sum()/(2*hf+1e-9); od = (ns & ~ing).sum()/(2*np.pi-2*hf+1e-9)
        gap[astr] = float(gd/(od+1e-9))
    print(f"{scene}/ScaffoldGS@{a}: n={len(cen)} gap/obs(count)={gap.get(astr,0):.2f}", flush=True)
json.dump(gap, open(f"{GD}/{scene}__ScaffoldGS.gap.json", "w")); print("DONE")
