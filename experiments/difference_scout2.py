#!/usr/bin/env python3
"""Crop-aware difference scout (v2). For every (cell,view) it searches VARIABLE-SIZE crops and
returns the box maximizing a visible-difference score: where Ours beats the MEDIAN baseline
(robust) in a region where the baselines disagree (high spread) — i.e. an obvious, localized win.
Stage 'coverage_gap' reads local-method renders from the standings manifest; 'discontinuous' reads
the assembled viz_discontinuous/cells. Output: diff2_<stage>.json (ranked).
"""
import json, os, glob, sys
import numpy as np
from PIL import Image

STAGE = sys.argv[1] if len(sys.argv) > 1 else "coverage_gap"
VM = "/data1/zhangao/outputs/viz_multiangle"
VD = "/data1/zhangao/outputs/viz_discontinuous"
MAXV = 30
SCALES = [0.26, 0.36, 0.48, 0.62]

def err(im, gt):
    a = np.asarray(im.convert("RGB"), float)/255; b = np.asarray(gt, float)/255
    return np.abs(a-b).mean(2)

from scipy.ndimage import uniform_filter
def smoothness(gt):
    g = np.asarray(gt.convert("L"), float)/255
    gy, gx = np.gradient(g); grad = uniform_filter(np.hypot(gx, gy), size=15)
    p90 = np.percentile(grad, 90)+1e-6
    return np.clip(1.0 - grad/p90, 0, 1)   # 1 = smooth (low texture, where artifacts pop)

def integral(x):
    return np.pad(np.cumsum(np.cumsum(x,0),1), ((1,0),(1,0)))

def best_crop(weight):
    H, W = weight.shape; ii = integral(weight); best = (-1, None, None)
    for s in SCALES:
        bh = bw = int(min(H, W)*s)
        if bh < 8: continue
        st = max(4, int(min(H, W)*0.06))
        for t in range(0, H-bh+1, st):
            for l in range(0, W-bw+1, st):
                tot = ii[t+bh, l+bw]-ii[t, l+bw]-ii[t+bh, l]+ii[t, l]
                m = tot/(bh*bw)
                if m > best[0]: best = (m, (l, t, l+bw, t+bh), s)
    return best

def score_view(gt, ours, bases):
    oe = err(ours, gt)
    be = np.stack([err(b, gt) for b in bases])
    med = np.median(be, 0); spread = be.std(0)
    sm = smoothness(gt)
    # obvious win: ours beats median baseline, methods disagree, on a region smooth enough
    # that an artifact pops (down-weights pure foliage clutter); + ours actually clean.
    weight = np.clip(med - oe, 0, None) * (0.25 + spread) * (0.4 + 0.9*sm) * (1.0 - np.clip(oe/0.4, 0, 0.8))
    m, box, s = best_crop(weight)
    if box is None: return None
    l, t, r, b = box
    # diagnostics within box
    obox = oe[t:b, l:r].mean(); medbox = med[t:b, l:r].mean()
    beats_all = float((be.min(0)[t:b, l:r] > oe[t:b, l:r]).mean())
    return dict(score=round(float(m), 5), box=[l, t, r, b], scale=s,
                ours_err=round(float(obox), 4), med_err=round(float(medbox), 4),
                beats_all_frac=round(beats_all, 3))

cands = []
if STAGE == "coverage_gap":
    man = json.load(open(f"{VM}/manifest.json"))
    LB = ["Hier", "ScaffoldGS", "MH-3DGS", "MVGS"]
    for cell, info in man.items():
        o = info.get("ours") or {}; od = o.get("rdir")
        bd = {m: info["base"][m]["rdir"] for m in LB if m in info.get("base", {}) and info["base"][m].get("rdir")}
        gtd = None
        for m in LB:
            if m in info.get("base", {}) and info["base"][m].get("gtdir"): gtd = info["base"][m]["gtdir"]; break
        gtd = gtd or o.get("gtdir")
        if not od or not gtd or len(bd) < 2: continue
        views = sorted(set(os.path.basename(p) for p in glob.glob(f"{od}/*.png"))
                       & set.intersection(*[set(os.path.basename(p) for p in glob.glob(f"{d}/*.png")) for d in bd.values()]))
        if not views: continue
        for v in views[::max(1, len(views)//MAXV)]:
            if not os.path.exists(f"{gtd}/{v}"): continue
            try:
                gt = Image.open(f"{gtd}/{v}").convert("RGB"); ref = gt.size
                ours = Image.open(f"{od}/{v}").convert("RGB").resize(ref)
                bases = [Image.open(f"{d}/{v}").convert("RGB").resize(ref) for d in bd.values()]
            except Exception: continue
            r = score_view(gt, ours, bases)
            if r: cands.append(dict(cell=cell.replace("@", "_").replace(".0", ""), manifest_key=cell,
                                    scene=info["scene"], angle=str(int(float(info["angle"]))), view=v, **r))
else:  # discontinuous
    LB = ["3DGS", "Hier", "ScaffoldGS", "MH-3DGS", "MVGS"]
    for d in sorted(glob.glob(f"{VD}/cells/*")):
        scene = os.path.basename(d); rd = f"{d}/renders"
        if not os.path.isdir(f"{rd}/Ours") or not os.path.isdir(f"{rd}/GT"): continue
        bd = {m: f"{rd}/{m}" for m in LB if os.path.isdir(f"{rd}/{m}")}
        views = sorted(os.path.basename(p) for p in glob.glob(f"{rd}/Ours/*.png"))
        for v in views:
            if not os.path.exists(f"{rd}/GT/{v}"): continue
            try:
                gt = Image.open(f"{rd}/GT/{v}").convert("RGB"); ref = gt.size
                ours = Image.open(f"{rd}/Ours/{v}").convert("RGB").resize(ref)
                bases = [Image.open(f"{p}/{v}").convert("RGB").resize(ref) for p in bd.values() if os.path.exists(f"{p}/{v}")]
            except Exception: continue
            if len(bases) < 2: continue
            r = score_view(gt, ours, bases)
            if r: cands.append(dict(cell=scene, scene=scene, angle="disc", view=v, **r))

def rk(c):
    gap = {"100": 1.18, "80": 1.12, "60": 1.05}.get(c["angle"], 1.0)
    return c["score"]*gap
cands.sort(key=rk, reverse=True)
json.dump(cands, open(f"{VM}/diff2_{STAGE}.json", "w"), indent=1)
print(f"[{STAGE}] scored {len(cands)} (cell,view); cells={len(set(c['cell'] for c in cands))}")
print(f"{'cell':>14} {'view':>9} {'score':>7} {'scale':>5} {'beats_all%':>10}")
for c in cands[:30]:
    print(f"{c['cell']:>14} {c['view']:>9} {c['score']:7.4f} {c['scale']:5.2f} {c['beats_all_frac']*100:9.0f}")
