#!/usr/bin/env python3
"""Crop-aware difference scout (v3) — REDESIGNED for human-obvious, readable crops.

Lessons from v2 rejection: v2's `smoothness` term up-weighted FLAT regions ("artifacts pop on
smooth surfaces") which drove crops straight into pure sky / wall / shadow with no semantic anchor,
and it collapsed to the tiniest scale (0.26) so there was no context. The user wants crops that
(a) are READABLE (recognizable structure: edges, contours, objects — not flat white/black, not pure
foliage mush) and (b) show an OBVIOUS localized win (a baseline visibly ghosts/blurs/floats where
Ours is sharp).

New scoring favours a COHERENT baseline-failure blob (low-pass of [median_baseline_err - ours_err],
so scattered foliage noise averages out but a real artifact survives) located on/near readable
STRUCTURE (mid-band edge density of GT), and HARD-REJECTS crops that are saturated (sky/shadow),
flat (low luminance variance), or pure clutter. Bigger scales for context. Emits several
non-overlapping candidates per cell so a downstream vision pass can choose.

Output: diff3_<stage>.json (ranked). Reads local-method renders only (no hust2 needed for scouting).
"""
import json, os, glob, sys
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

STAGE = sys.argv[1] if len(sys.argv) > 1 else "coverage_gap"
VM = "/data1/zhangao/outputs/viz_multiangle"
MANIFEST = sys.argv[2] if len(sys.argv) > 2 else None          # CG: override manifest path
OUT_NAME = sys.argv[3] if len(sys.argv) > 3 else None          # override output basename
VD = "/data1/zhangao/outputs/viz_discontinuous"
MAXV = 100                     # views sampled per cell (sample ~all views -> more distinct crops)
SCALES = [0.40, 0.52, 0.66]   # bigger crops -> more context, readable
# discontinuous: each scene IS one cell -> keep many crops/cell; coverage_gap has a cell per angle.
CAND_PER_CELL = {"coverage_gap": 3, "discontinuous": 16}[STAGE]
TOPN = {"coverage_gap": 24, "discontinuous": 80}[STAGE]
# Method differences are large in coverage_gap (wins ~0.1-0.24) but genuinely subtler in
# discontinuous (wins ~0.02-0.05, localized on edges) -> stage-aware win thresholds. Readability
# filters (saturation/std) stay the same: the user's core complaint was flat/unreadable crops.
THR = {"coverage_gap": dict(sat=0.45, std=0.055, win=0.018, rel=0.16, obox=0.16),
       "discontinuous": dict(sat=0.45, std=0.055, win=0.006, rel=0.08, obox=0.20)}[STAGE]

def luma(im):
    return np.asarray(im.convert("L"), float) / 255

def err(im, gt):
    a = np.asarray(im.convert("RGB"), float) / 255
    b = np.asarray(gt, float) / 255
    return np.abs(a - b).mean(2)

def structure(L):
    """mid-band edge density: enough edges to read the scene, saturating so pure foliage
    doesn't dominate. Returns 0..~1.1 map (low-passed gradient magnitude, normalized)."""
    gy, gx = np.gradient(L)
    g = uniform_filter(np.hypot(gx, gy), size=21)
    p70 = np.percentile(g, 70) + 1e-6
    return np.clip(g / p70, 0, 1.1)

def integral(x):
    return np.pad(np.cumsum(np.cumsum(x, 0), 1), ((1, 0), (1, 0)))

def box_mean(ii, t, l, bh, bw):
    return (ii[t + bh, l + bw] - ii[t, l + bw] - ii[t + bh, l] + ii[t, l]) / (bh * bw)

def ranked_boxes(weight, topn=TOPN):
    """integral-image multi-scale box search; return top-N (mean, box, scale) before NMS."""
    H, W = weight.shape
    ii = integral(weight)
    out = []
    for s in SCALES:
        bh = bw = int(min(H, W) * s)
        if bh < 8 or bh > min(H, W):
            continue
        st = max(6, int(min(H, W) * 0.05))
        for t in range(0, H - bh + 1, st):
            for l in range(0, W - bw + 1, st):
                out.append((box_mean(ii, t, l, bh, bw), (l, t, l + bw, t + bh), s))
    out.sort(key=lambda x: -x[0])
    return out[:topn]

def iou(a, b):
    l = max(a[0], b[0]); t = max(a[1], b[1]); r = min(a[2], b[2]); bo = min(a[3], b[3])
    if r <= l or bo <= t:
        return 0.0
    inter = (r - l) * (bo - t)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua

def valid_crop(L, oe, med, box):
    """Reject sky/shadow/flat/clutter; require a real, visible win in the crop."""
    l, t, r, b = box
    Lc = L[t:b, l:r]
    sat = ((Lc > 0.90) | (Lc < 0.045)).mean()      # near-white sky / near-black shadow
    std = Lc.std()                                   # tonal range -> there is *something* to see
    obox = oe[t:b, l:r].mean(); medbox = med[t:b, l:r].mean()
    win = medbox - obox                              # ours beats median baseline by this L1
    rel = win / (medbox + 1e-6)                      # relative improvement
    ok = (sat < THR["sat"]) and (std > THR["std"]) and (win > THR["win"]) \
        and (rel > THR["rel"]) and (obox < THR["obox"])
    return ok, dict(sat=round(float(sat), 3), std=round(float(std), 3),
                    ours_err=round(float(obox), 4), med_err=round(float(medbox), 4),
                    win=round(float(win), 4), rel_win=round(float(rel), 3))

def score_view(gt, ours, bases):
    L = luma(gt)
    oe = err(ours, gt)
    be = np.stack([err(b, gt) for b in bases])
    med = np.median(be, 0); spread = be.std(0)
    adv = np.clip(med - oe, 0, None)
    # coherent failure: low-pass adv so a real artifact blob survives but foliage speckle cancels
    adv_lp = uniform_filter(adv, size=27)
    struct = structure(L)                           # readable structure (context)
    sat_map = ((L > 0.90) | (L < 0.045)).astype(float)
    nonsat = 1.0 - uniform_filter(sat_map, size=27)  # avoid sky/shadow
    # weight: coherent win, where baselines disagree, on readable structure, not saturated
    weight = adv_lp * (0.35 + spread) * (0.30 + 0.70 * struct) * (nonsat ** 2)
    boxes = ranked_boxes(weight)
    picks = []
    for m, box, s in boxes:
        ok, diag = valid_crop(L, oe, med, box)
        if not ok:
            continue
        if any(iou(box, p["box"]) > 0.30 for p in picks):
            continue
        l, t, r, b = box
        beats_all = float((be.min(0)[t:b, l:r] > oe[t:b, l:r]).mean())
        picks.append(dict(score=round(float(m), 6), box=[l, t, r, b], scale=s,
                          beats_all_frac=round(beats_all, 3), **diag))
        if len(picks) >= CAND_PER_CELL:
            break
    return picks

cands = []
if STAGE == "coverage_gap":
    man = json.load(open(MANIFEST or f"{VM}/manifest.json"))
    LB = ["Hier", "ScaffoldGS", "MH-3DGS", "MVGS"]
    for cell, info in man.items():
        o = info.get("ours") or {}; od = o.get("rdir")
        bd = {m: info["base"][m]["rdir"] for m in LB
              if m in info.get("base", {}) and info["base"][m].get("rdir")}
        gtd = None
        for m in LB:
            if m in info.get("base", {}) and info["base"][m].get("gtdir"):
                gtd = info["base"][m]["gtdir"]; break
        gtd = gtd or o.get("gtdir")
        if not od or not gtd or len(bd) < 2:
            continue
        views = sorted(set(os.path.basename(p) for p in glob.glob(f"{od}/*.png"))
                       & set.intersection(*[set(os.path.basename(p) for p in glob.glob(f"{d}/*.png"))
                                            for d in bd.values()]))
        if not views:
            continue
        for v in views[::max(1, len(views) // MAXV)]:
            if not os.path.exists(f"{gtd}/{v}"):
                continue
            try:
                gt = Image.open(f"{gtd}/{v}").convert("RGB"); ref = gt.size
                ours = Image.open(f"{od}/{v}").convert("RGB").resize(ref)
                bases = [Image.open(f"{d}/{v}").convert("RGB").resize(ref) for d in bd.values()]
            except Exception:
                continue
            for r in score_view(gt, ours, bases):
                cands.append(dict(cell=cell.replace("@", "_").replace(".0", ""), manifest_key=cell,
                                  scene=info["scene"], angle=str(int(float(info["angle"]))), view=v, **r))
else:  # discontinuous
    LB = ["3DGS", "Hier", "ScaffoldGS", "MH-3DGS", "MVGS"]
    SCENES = os.environ.get("SCOUT_SCENES")
    SCENES = set(SCENES.split(",")) if SCENES else None
    for d in sorted(glob.glob(f"{VD}/cells/*")):
        scene = os.path.basename(d); rd = f"{d}/renders"
        if SCENES and scene not in SCENES: continue
        if not os.path.isdir(f"{rd}/Ours") or not os.path.isdir(f"{rd}/GT"):
            continue
        bd = {m: f"{rd}/{m}" for m in LB if os.path.isdir(f"{rd}/{m}")}
        views = sorted(os.path.basename(p) for p in glob.glob(f"{rd}/Ours/*.png"))
        for v in views:
            if not os.path.exists(f"{rd}/GT/{v}"):
                continue
            try:
                gt = Image.open(f"{rd}/GT/{v}").convert("RGB"); ref = gt.size
                ours = Image.open(f"{rd}/Ours/{v}").convert("RGB").resize(ref)
                bases = [Image.open(f"{p}/{v}").convert("RGB").resize(ref)
                         for p in bd.values() if os.path.exists(f"{p}/{v}")]
            except Exception:
                continue
            if len(bases) < 2:
                continue
            for r in score_view(gt, ours, bases):
                cands.append(dict(cell=scene, scene=scene, angle="disc", view=v, **r))

def rk(c):
    gap = {"100": 1.18, "80": 1.12, "60": 1.05}.get(c["angle"], 1.0)
    return c["score"] * gap

cands.sort(key=rk, reverse=True)
json.dump(cands, open(f"{VM}/{OUT_NAME or ('diff3_' + STAGE)}.json", "w"), indent=1)
print(f"[{STAGE}] valid candidates: {len(cands)}; cells={len(set(c['cell'] for c in cands))}; "
      f"scenes={sorted(set(c['scene'] for c in cands))}")
import collections
print("scale use:", dict(collections.Counter(c['scale'] for c in cands)))
print(f"{'cell':>16} {'view':>9} {'score':>8} {'scale':>5} {'win':>6} {'relw':>5} {'sat':>5} {'std':>5} {'beats%':>6}")
for c in cands[:30]:
    print(f"{c['cell']:>16} {c['view']:>9} {c['score']:8.4f} {c['scale']:5.2f} "
          f"{c['win']:6.3f} {c['rel_win']:5.2f} {c['sat']:5.2f} {c['std']:5.2f} {c['beats_all_frac']*100:5.0f}")
