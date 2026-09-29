#!/usr/bin/env python3
"""Add Scaffold-GS to the free-space-floater BEV. Scaffold-GS stores, per anchor, k offset vectors
(f_offset_*) and a 6-d scaling; the offset Gaussian CENTERS are anchor + offset*exp(scale[:3]) and are
recoverable WITHOUT the view-dependent MLP (only per-Gaussian opacity/color need the MLP). So we can
build the floater BEV from the neural-Gaussian centers, COUNT-weighted (no per-Gaussian opacity).
Uses the SAME saved projection as the other methods (global_all/proj/<scene>_proj.json) so it tiles.
Outputs global_all/<scene>__ScaffoldGS__<a>.npy + <scene>__ScaffoldGS.frac.json (count-fraction).
Usage: scaffold_bev_export.py [scene1 scene2 ...]   (default bonsai counter kitchen)"""
import os, sys, glob, json
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
GA = "/data1/zhangao/outputs/viz_multiangle/global_all"
BINS = 320
scenes = sys.argv[1:] or ["bonsai", "counter", "kitchen"]
ANGLES = [20, 40, 60, 80, 100]
def first(g):
    h = sorted(glob.glob(g)); return h[-1] if h else None
def scaffold_centers(p):
    v = PlyData.read(p)["vertex"]; N = len(v["x"])
    anchor = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    noff = sum(1 for pr in v.properties if pr.name.startswith("f_offset_")); k = noff // 3
    off = np.stack([v[f"f_offset_{i}"] for i in range(noff)], 1).astype(np.float32)  # [x0..x{k-1},y..,z..]
    off = off.reshape(N, 3, k).transpose(0, 2, 1)                                    # (N,k,3)
    sc = np.stack([v[f"scale_{i}"] for i in range(6)], 1).astype(np.float32)
    cen = anchor[:, None, :] + off * np.exp(sc[:, :3])[:, None, :]                   # (N,k,3)
    cen = cen.reshape(-1, 3)
    return cen[np.isfinite(cen).all(1)]
for scene in scenes:
    projf = f"{GA}/proj/{scene}_proj.json"
    refp = first(f"/data1/zhangao/outputs/baselines_multi_angle/{scene}/*/input.ply")
    if not os.path.exists(projf) or not refp:
        print(f"SKIP {scene}: missing proj/ref"); continue
    P = json.load(open(projf)); axes = np.array(P["axes"], np.float32); center = np.array(P["center"], np.float32)
    mn = np.array(P["mn"], np.float32); mx = np.array(P["mx"], np.float32); tau = float(P["tau"])
    ref = np.stack([PlyData.read(refp)["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float32)
    tree = cKDTree(ref)
    fr = {}
    for a in ANGLES:
        p = first(f"/data1/zhangao/outputs/baselines_multi_angle/{scene}/scaffoldgs-original*test_selection_angle_{a}.0/point_cloud/iteration_*/point_cloud.ply")
        if not p:
            print(f"MISS {scene}/ScaffoldGS@{a}", flush=True); continue
        cen = scaffold_centers(p)
        d, _ = tree.query(cen, k=1, workers=-1); fl = d > tau
        fr[a] = float(fl.mean())                       # COUNT fraction of neural gaussians in free space
        xy = (cen[fl] - center) @ axes.T
        H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=BINS, range=[[mn[0], mx[0]], [mn[1], mx[1]]])
        np.save(f"{GA}/{scene}__ScaffoldGS__{a}.npy", H.T.astype(np.float32))
        print(f"{scene}/ScaffoldGS@{a}: floater {fr[a]*100:.1f}% (count)  ({fl.sum()}/{len(cen)})", flush=True)
    json.dump(fr, open(f"{GA}/{scene}__ScaffoldGS.frac.json", "w"))
print("DONE")
