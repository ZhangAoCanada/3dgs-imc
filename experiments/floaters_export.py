#!/usr/bin/env python3
"""Compute free-space floater stats + BEV histograms for a set of methods on ONE machine,
using a SHARED projection (so results from different machines tile together). Run locally for
local-ply methods and on hust2 for its methods; combine with floaters_combine.py.

Config JSON (passed as argv[1]):
  { "scene": "bonsai",
    "ref_ply": "<glob to full-coverage SfM input.ply>",
    "angles": [20,40,60,80,100],
    "methods": { "MCMC": "<glob with {a}>", ... },   # {a} -> angle like 100.0
    "proj": "<path to proj json>",   # loaded if exists, else computed from ref_ply and saved
    "out":  "<output dir>" }
Outputs: <out>/<scene>__<method>__<angle>.npy  (BEV float histogram, opacity-weighted floaters)
         <out>/<scene>__<method>.frac.json      ({angle: fraction})
"""
import os, sys, json, glob
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

cfg = json.load(open(sys.argv[1]))
scene = cfg["scene"]; out = cfg["out"]; os.makedirs(out, exist_ok=True)
BINS = 320; TAU_FRAC = 0.015

def first(g):
    h = sorted(glob.glob(g)); return h[-1] if h else None

ref_path = first(cfg["ref_ply"])
ref = np.stack([PlyData.read(ref_path)["vertex"][k] for k in ("x","y","z")], 1).astype(np.float32)

# projection: load if present (shared across machines), else compute deterministically (no SVD)
projf = cfg["proj"]
if os.path.exists(projf):
    P = json.load(open(projf))
    axes = np.array(P["axes"], np.float32); center = np.array(P["center"], np.float32)
    mn = np.array(P["mn"], np.float32); mx = np.array(P["mx"], np.float32)
    tau = P["tau"]
else:
    lo, hi = np.percentile(ref, 1, 0), np.percentile(ref, 99, 0)
    diag = float(np.linalg.norm(hi - lo)); tau = TAU_FRAC * diag
    center = ref.mean(0)
    # deterministic top-down: the 2 world axes with largest extent (machine-independent)
    ext = hi - lo; order = np.argsort(-ext)[:2]
    axes = np.zeros((2, 3), np.float32)
    for i, a in enumerate(order): axes[i, a] = 1.0
    r2 = (ref - center) @ axes.T
    pad = 0.35
    mn = r2.min(0) - pad*(r2.max(0)-r2.min(0)); mx = r2.max(0) + pad*(r2.max(0)-r2.min(0))
    json.dump({"axes": axes.tolist(), "center": center.tolist(), "mn": mn.tolist(),
               "mx": mx.tolist(), "tau": tau, "diag": diag}, open(projf, "w"))

def proj(x): return (x - center) @ axes.T
tree = cKDTree(ref)

# save the SfM reference projected top-down (shared scene context for interpretable BEV overlay)
refxy = proj(ref)
Href, _, _ = np.histogram2d(refxy[:,0], refxy[:,1], bins=BINS,
                            range=[[mn[0],mx[0]],[mn[1],mx[1]]])
np.save(os.path.join(out, f"{scene}__REF.npy"), Href.T.astype(np.float32))

def read(p):
    v = PlyData.read(p)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    names = [pr.name for pr in v.properties]
    if "opacity" in names:
        op = 1.0/(1.0+np.exp(-v["opacity"].astype(np.float32)))
    else:
        op = np.ones(len(xyz), np.float32)   # anchor/surfel PLY: count-based
    fin = np.isfinite(xyz).all(1) & np.isfinite(op)
    return xyz[fin], op[fin]

for label, tmpl in cfg["methods"].items():
    fr = {}
    for a in cfg["angles"]:
        astr = f"{a:.1f}"
        p = first(tmpl.replace("{a}", astr))
        if not p:
            print(f"MISS {scene}/{label}@{a}", flush=True); continue
        xyz, op = read(p)
        d, _ = tree.query(xyz, k=1, workers=-1)
        fl = d > tau
        fr[a] = float(op[fl].sum()/(op.sum()+1e-9))
        xy = proj(xyz[fl])
        H, _, _ = np.histogram2d(xy[:,0], xy[:,1], bins=BINS,
                                 range=[[mn[0],mx[0]],[mn[1],mx[1]]], weights=op[fl])
        np.save(os.path.join(out, f"{scene}__{label}__{a}.npy"), H.T.astype(np.float32))
        print(f"{scene}/{label}@{a}: floater {fr[a]*100:.2f}%  ({fl.sum()}/{len(xyz)})", flush=True)
    json.dump(fr, open(os.path.join(out, f"{scene}__{label}.frac.json"), "w"))
print(f"done {scene}: {list(cfg['methods'])}", flush=True)
