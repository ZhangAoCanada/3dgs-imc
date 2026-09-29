#!/usr/bin/env python3
"""Export top-down Gaussian DENSITY (opacity-weighted, ALL Gaussians) per (scene,method,angle) in a
SHARED deterministic projection, plus the gap geometry (train/test camera positions, gap direction)
so the figure can show that stochastic methods (Ours/MCMC/MH) keep density EVEN into the gap while
deterministic methods (3DGS/Hier/...) bias density to the observed arc. Run locally for local PLYs
and on hust2 for its PLYs (shared proj json). Config JSON argv[1]:
 { scene, ref_ply, angles:[...], methods:{name:"glob with {a}"}, cams_glob:"<dir glob with {a}>",
   test_list:"<test_selection_angle_{a}.txt glob>", proj:"<proj json>", out:"<dir>" }
Outputs: <out>/<scene>__<method>__<angle>.dens.npy (density hist) and __<method>.gap.json ({angle:gap_obs_ratio}),
         <out>/<scene>__GEO.json (proj + scene footprint + per-angle train/test cam xy + gap dir/half)."""
import os, sys, json, glob
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
cfg = json.load(open(sys.argv[1])); scene = cfg["scene"]; out = cfg["out"]; os.makedirs(out, exist_ok=True)
BINS = 360
def first(g):
    h = sorted(glob.glob(g)); return h[-1] if h else None
ref = np.stack([PlyData.read(first(cfg["ref_ply"]))["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float32)
lo, hi = np.percentile(ref, 1, 0), np.percentile(ref, 99, 0); diag = float(np.linalg.norm(hi-lo))
center = ref.mean(0); order = np.argsort(-(hi-lo))[:2]
axes = np.zeros((2, 3), np.float32)
for i, a in enumerate(order): axes[i, a] = 1.0
def proj(x): return (x - center) @ axes.T
refc = proj(ref); scenter = np.median(refc, 0)
pad = 0.30; mn = refc.min(0)-pad*(refc.max(0)-refc.min(0)); mx = refc.max(0)+pad*(refc.max(0)-refc.min(0))
def hist(xy, w=None):
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=BINS, range=[[mn[0], mx[0]], [mn[1], mx[1]]], weights=w)
    return H.T.astype(np.float32)
refH = (hist(refc) > 0).astype(np.float32)
def ang_of(p): d = p - scenter; return np.arctan2(d[..., 1], d[..., 0])
def adiff(a, b): return np.abs(np.angle(np.exp(1j*(a-b))))
# geometry per angle (train/test cam xy, gap dir/half)
geo = {"mn": mn.tolist(), "mx": mx.tolist(), "scenter": scenter.tolist(), "diag": diag, "angles": {}}
np.save(f"{out}/{scene}__REF.npy", refH)
for a in cfg["angles"]:
    astr = f"{a:.1f}"
    camf = first(cfg["cams_glob"].replace("{a}", astr))
    tlist = first(cfg["test_list"].replace("{a}", astr))
    if not camf or not tlist: continue
    cams = json.load(open(camf)); test = set(l.strip().rsplit(".", 1)[0] for l in open(tlist))
    pos = np.array([c["position"] for c in cams], np.float32)
    istest = np.array([c["img_name"].rsplit(".", 1)[0] in test for c in cams])
    cp = proj(pos); ta = ang_of(cp[istest]); gmean = float(np.arctan2(np.sin(ta).mean(), np.cos(ta).mean()))
    half = float(np.percentile(adiff(ta, gmean), 85))
    geo["angles"][astr] = {"train_xy": cp[~istest].tolist(), "test_xy": cp[istest].tolist(),
                           "gap_dir": gmean, "gap_half": half}
json.dump(geo, open(f"{out}/{scene}__GEO.json", "w"))
near = 0.04*diag; tree = cKDTree(ref)
def read(p):
    v = PlyData.read(p)["vertex"]; xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    names = [pr.name for pr in v.properties]
    op = 1/(1+np.exp(-v["opacity"].astype(np.float32))) if "opacity" in names else np.ones(len(xyz), np.float32)
    f = np.isfinite(xyz).all(1) & np.isfinite(op); return xyz[f], op[f]
for label, tmpl in cfg["methods"].items():
    gap = {}
    for a in cfg["angles"]:
        astr = f"{a:.1f}"; p = first(tmpl.replace("{a}", astr))
        if not p: print(f"MISS {scene}/{label}@{a}", flush=True); continue
        xyz, op = read(p); xy = proj(xyz)
        np.save(f"{out}/{scene}__{label}__{a}.dens.npy", hist(xy, op))
        # gap/obs evenness on near-surface gaussians
        if astr in geo["angles"]:
            gm = geo["angles"][astr]["gap_dir"]; hf = geo["angles"][astr]["gap_half"]
            d, _ = tree.query(xyz, k=1, workers=-1); ns = d < near; aa = ang_of(xy); ing = adiff(aa, gm) < hf
            gd = op[ns & ing].sum()/(2*hf+1e-9); od = op[ns & ~ing].sum()/(2*np.pi-2*hf+1e-9)
            gap[astr] = float(gd/(od+1e-9))
        print(f"{scene}/{label}@{a}: n={len(xyz)} gap/obs={gap.get(astr,0):.2f}", flush=True)
    json.dump(gap, open(f"{out}/{scene}__{label}.gap.json", "w"))
print(f"done {scene}: {list(cfg['methods'])}", flush=True)
