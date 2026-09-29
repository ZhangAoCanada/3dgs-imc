#!/usr/bin/env python3
"""Per-cell standings for the coverage-gap (multi-angle) stage, to pick scenes for
qualitative comparison. For each (scene, angle):
  OURS  = best-PSNR run that HAS rendered test images, scanned from the live diag trees
          (server + WSL mirror) and the collected best_coverage_gap set.
  BASE  = each baseline's per-scene cell (3dgs-depth=Hier, MH-3DGS, MVGS, ScaffoldGS),
          metrics read from results.json, renders from test/ours_*/renders.
Emits a printed standings table + a JSON manifest (render dirs + per-view metric files)
for every cell, and a ranked shortlist where OURS beats the best baseline by a margin,
preferring large angle gaps (80/100). The manifest drives the render-assembly step.
"""
import csv, glob, json, os, re
from collections import defaultdict

OUT = "/data1/zhangao/outputs"
DIAG_ROOTS = [f"{OUT}/diag/mipnerf360",
              "/data1/zhangao/repos/3dgs-imc-codex/outputs/wsl_results/diag/mipnerf360"]
BASE_ROOT = f"{OUT}/baselines_multi_angle"
BEST_CG = f"{OUT}/best_coverage_gap"
DEST = f"{OUT}/viz_multiangle"
os.makedirs(DEST, exist_ok=True)

SCENES = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"]
ANGLES = [20.0, 40.0, 60.0, 80.0, 100.0]
# baseline method -> per-scene cell dirname template (note hyphen vs underscore differences)
BASE_METHODS = {
    "Hier":     "3dgs-depth-test_selection_angle_{a}",      # 3dgs-depth (paper: Hier-GS)
    "ScaffoldGS":"scaffoldgs-original-test_selection_angle_{a}",
    "MH-3DGS":  "mh-3dgs-original_test_selection_angle_{a}",
    "MVGS":     "mvgs-original-mv3_test_selection_angle_{a}",
}

def render_dir(cell_dir):
    g = sorted(glob.glob(os.path.join(cell_dir, "test", "ours_*", "renders")))
    return g[-1] if g else None
def gt_dir(cell_dir):
    g = sorted(glob.glob(os.path.join(cell_dir, "test", "ours_*", "gt")))
    return g[-1] if g else None
def depth_dir(cell_dir):
    g = sorted(glob.glob(os.path.join(cell_dir, "test", "ours_*", "depth")))
    return g[-1] if g else None
def n_png(d):
    return len(glob.glob(os.path.join(d, "*.png"))) if d and os.path.isdir(d) else 0

# ---------- OURS: best run with renders per (scene, angle) ----------
RUNPAT = re.compile(r"^(?P<scene>.+?)_angle_(?P<a>[0-9.]+)_(?P<tag>.+)_i(?P<it>\d+)$")
ours = {}  # (scene,a) -> dict(psnr,ssim,lpips,dir,rdir,gtdir,ddir,nv,tag)
def consider(d, scene, a):
    mc = os.path.join(d, "metrics.csv")
    if not os.path.exists(mc): return
    best = None
    try:
        for r in csv.DictReader(open(mc)):
            if not r.get("method", "").startswith("ours_"): continue
            try: p = float(r["PSNR"]); s = float(r["SSIM"]); l = float(r["LPIPS"])
            except: continue
            if best is None or p > best[0]: best = (p, s, l)
    except: return
    if best is None: return
    rd = render_dir(d)
    if not rd or n_png(rd) == 0: return  # deliverable-grade only
    k = (scene, a)
    if k not in ours or best[0] > ours[k]["psnr"]:
        ours[k] = dict(psnr=best[0], ssim=best[1], lpips=best[2], dir=d,
                       rdir=rd, gtdir=gt_dir(d), ddir=depth_dir(d), nv=n_png(rd),
                       tag=os.path.basename(d))
for root in DIAG_ROOTS:
    for d in glob.glob(os.path.join(root, "*_angle_*_i*")):
        if not os.path.isdir(d): continue
        m = RUNPAT.match(os.path.basename(d))
        if not m or m["scene"] not in SCENES: continue
        consider(d, m["scene"], float(m["a"]))
# fold in collected best_coverage_gap source dirs (already curated bests)
for sub in glob.glob(os.path.join(BEST_CG, "*_angle_*")):
    man = os.path.join(sub, "MANIFEST.txt")
    if not os.path.exists(man): continue
    info = dict(l.strip().split("=", 1) for l in open(man) if "=" in l)
    sd = info.get("source_dir", "")
    mm = re.search(r"^(.+)@([0-9.]+)$", info.get("cell", ""))
    if sd and os.path.isdir(sd) and mm and mm.group(1) in SCENES:
        consider(sd, mm.group(1), float(mm.group(2)))

# ---------- BASELINES ----------
def base_metrics(cell_dir):
    rj = os.path.join(cell_dir, "results.json")
    if not os.path.exists(rj): return None
    try: d = json.load(open(rj))
    except: return None
    key = "ours_30000" if "ours_30000" in d else (sorted(d)[-1] if d else None)
    if not key: return None
    v = d[key]
    return dict(psnr=v.get("PSNR"), ssim=v.get("SSIM"), lpips=v.get("LPIPS"))

base = defaultdict(dict)  # (scene,a) -> method -> dict(psnr,ssim,lpips,rdir,gtdir,nv,dir)
for scene in SCENES:
    for a in ANGLES:
        astr = f"{a:.1f}"
        for meth, tmpl in BASE_METHODS.items():
            cd = os.path.join(BASE_ROOT, scene, tmpl.format(a=astr))
            if not os.path.isdir(cd): continue
            mt = base_metrics(cd)
            if not mt or mt["psnr"] is None: continue
            rd = render_dir(cd)
            base[(scene, a)][meth] = dict(**mt, rdir=rd, gtdir=gt_dir(cd), nv=n_png(rd), dir=cd)

# ---------- standings + manifest ----------
rows = []
manifest = {}
for scene in SCENES:
    for a in ANGLES:
        k = (scene, a)
        o = ours.get(k)
        bm = base.get(k, {})
        if not o or not bm: continue
        comp = {m: v for m, v in bm.items() if v["psnr"] is not None}
        if not comp: continue
        best_m, best_v = max(comp.items(), key=lambda t: t[1]["psnr"])
        margin = o["psnr"] - best_v["psnr"]
        # per-metric vs best-PSNR competitor
        rows.append(dict(scene=scene, angle=a, ours=o["psnr"], ours_ssim=o["ssim"], ours_lpips=o["lpips"],
                         best_base=best_m, base_psnr=best_v["psnr"], base_ssim=best_v["ssim"],
                         base_lpips=best_v["lpips"], margin=margin, nv_ours=o["nv"],
                         scaf=comp.get("ScaffoldGS", {}).get("psnr"),
                         hier=comp.get("Hier", {}).get("psnr"),
                         mh=comp.get("MH-3DGS", {}).get("psnr"),
                         mvgs=comp.get("MVGS", {}).get("psnr")))
        manifest[f"{scene}@{int(a)}"] = dict(
            scene=scene, angle=a, margin=margin, ours=o, base=comp)

json.dump(manifest, open(os.path.join(DEST, "manifest.json"), "w"), indent=1, default=str)

# print
def f(x, w=6, p=2):
    return (f"%{w}.{p}f" % x) if isinstance(x, (int, float)) else f"{'--':>{w}}"
rows.sort(key=lambda r: (-r["margin"]))
print(f"{'cell':>13} | {'OURS':>6} | {'Hier':>6} {'Scaf':>6} {'MH':>6} {'MVGS':>6} | {'bestB':>8} {'marg':>6} | nv")
print("-"*90)
for r in rows:
    print(f"{r['scene']+'@'+str(int(r['angle'])):>13} | {f(r['ours'])} | "
          f"{f(r['hier'])} {f(r['scaf'])} {f(r['mh'])} {f(r['mvgs'])} | "
          f"{r['best_base'][:8]:>8} {r['margin']:+6.2f} | {r['nv_ours']}")

# shortlist: ours wins, prefer big gaps. rank = margin but boost 80/100.
def score(r):
    bonus = {80.0: 0.6, 100.0: 1.0, 60.0: 0.2}.get(r["angle"], 0.0)
    return r["margin"] + bonus
win = [r for r in rows if r["margin"] > 0]
win.sort(key=score, reverse=True)
print(f"\n=== {len(win)} winning cells; shortlist (margin>0, big-gap-weighted) ===")
for r in win[:18]:
    print(f"  {r['scene']}@{int(r['angle'])}: ours {r['ours']:.2f} vs {r['best_base']} {r['base_psnr']:.2f}  (+{r['margin']:.2f} dB)")
print(f"\ntotal cells compared: {len(rows)}  | manifest -> {DEST}/manifest.json")
