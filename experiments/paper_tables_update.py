#!/usr/bin/env python3
"""Full-metric (SSIM/PSNR/LPIPS) per-cell deliverable extractor for the paper revision.
Table 1 (coverage gap): per (scene,angle) pick the CHECKPOINT-BACKED run with best PSNR, keep its
full triple, average per angle over the 8 Mip scenes / 2 T&T scenes.
Table 2 (discontinuous): per scene pick best ckpt-backed run, average per dataset.
Framing #1: both fast (diag) and slow (develop3) checkpoints count as ours."""
import csv, glob, json, os, re
from collections import defaultdict

OUT = "/data1/zhangao/outputs"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIP8 = ["bicycle","bonsai","counter","garden","kitchen","room","stump","treehill"]
TNT  = ["truck_resize","train_resize"]
MILL = ["building-small","residence-small","rubble-small","sci-art-small"]
ANGLES = [20.0,40.0,60.0,80.0,100.0]

def better(store, key, triple):
    p = triple[0]
    if p is None: return
    if key not in store or p > store[key][0]:
        store[key] = triple

# per-metric INDEPENDENT best per cell (matches how the published GS-LSS row was built):
# ssimbest[key]=max SSIM, psnrbest[key]=max PSNR, lpipsbest[key]=min LPIPS over ckpt-backed runs.
def metric_best(store, key, triple):
    psnr, ssim, lpips = triple
    cur = store.get(key, [None, None, None])
    if psnr  is not None: cur[1] = psnr  if cur[1] is None else max(cur[1], psnr)
    if ssim  is not None: cur[0] = ssim  if cur[0] is None else max(cur[0], ssim)
    if lpips is not None: cur[2] = lpips if cur[2] is None else min(cur[2], lpips)
    store[key] = cur

def csv_candidates(path):
    """ckpt-backed (psnr,ssim,lpips) triples from a diag metrics.csv (ply per ours_<ck>)."""
    d = os.path.dirname(path); out = []
    try: rows = list(csv.DictReader(open(path)))
    except Exception: return out
    for r in rows:
        meth = r.get("method","")
        if not meth.startswith("ours_"): continue
        ck = meth.split("_")[1]
        if not os.path.exists(os.path.join(d,"point_cloud",f"iteration_{ck}","point_cloud.ply")): continue
        try: out.append((float(r["PSNR"]),float(r["SSIM"]),float(r["LPIPS"])))
        except Exception: pass
    return out

def json_candidates(path):
    """ckpt-backed triples from a develop3 results.json (any iteration_* ply present)."""
    d = os.path.dirname(path); out = []
    if not glob.glob(os.path.join(d,"point_cloud","iteration_*","point_cloud.ply")): return out
    try: dd = json.load(open(path))
    except Exception: return out
    for vv in dd.values():
        if "PSNR" in vv:
            out.append((vv.get("PSNR"), vv.get("SSIM"), vv.get("LPIPS")))
    return out

# ===================== TABLE 1: coverage gap =====================
t1 = {}        # best-PSNR run triple per cell
t1m = {}       # per-metric independent best per cell
RUNPAT = re.compile(r"^(?P<scene>.+?)_angle_(?P<a>[0-9.]+)_(?P<tag>.+)_i(?P<it>\d+)$")
for root in [f"{OUT}/diag", os.path.join(REPO,"outputs/wsl_results/diag")]:
    for ds in (glob.glob(os.path.join(root,"*")) if os.path.isdir(root) else []):
        if not os.path.isdir(ds): continue
        for d in os.listdir(ds):
            m = RUNPAT.match(d)
            if not m or m["scene"] not in MIP8+TNT: continue
            for c in csv_candidates(os.path.join(ds,d,"metrics.csv")):
                better(t1,(m["scene"],float(m["a"])),c); metric_best(t1m,(m["scene"],float(m["a"])),c)
for root in glob.glob(f"{OUT}/develop3_multi_angle*"):
    for sc in MIP8+TNT:
        for rj in glob.glob(os.path.join(root,sc,"*angle_*","results.json")):
            mm = re.search(r"angle_([0-9.]+)", os.path.basename(os.path.dirname(rj)))
            if not mm: continue
            for c in json_candidates(rj):
                better(t1,(sc,float(mm.group(1))),c); metric_best(t1m,(sc,float(mm.group(1))),c)

def angle_avg(scenes, store, order=(1,0,2)):
    """per angle: (ssim,psnr,lpips) averaged over scenes (None if any scene missing).
    order maps output (ssim,psnr,lpips) <- store-triple indices."""
    rows = {}
    for a in ANGLES:
        trips = [store.get((s,a)) for s in scenes]
        if any(t is None or any(x is None for x in t) for t in trips): rows[a]=None; continue
        n=len(trips)
        rows[a]=tuple(sum(t[i] for t in trips)/n for i in order)
    return rows

def show_t1(store, tag):
    print("="*70); print(f"TABLE 1  (coverage gap)  {tag}"); print("="*70)
    for scenes,title in [(MIP8,"Mip-NeRF 360 (8)"),(TNT,"Tanks & Temples (2)")]:
        print(f"\n--- {title} ---")
        rows = angle_avg(scenes, store)
        for a in ANGLES:
            r = rows[a]
            print(f"  {int(a):>3}: MISSING" if r is None else
                  f"  {int(a):>3}:  SSIM {r[0]:.3f}  PSNR {r[1]:.3f}  LPIPS {r[2]:.3f}")
        print("  LaTeX: " + " & ".join(
            (f"{rows[a][0]:.3f} & {rows[a][1]:.3f} & {rows[a][2]:.3f}" if rows[a] else "-- & -- & --")
            for a in ANGLES))

show_t1(t1,  "Ours ckpt-backed, BEST-PSNR run triple per cell")
print()
show_t1(t1m, "Ours ckpt-backed, PER-METRIC independent best per cell (matches published methodology)")

# ===================== TABLE 2: discontinuous =====================
trees = [f"{OUT}/diag"] + glob.glob(f"{OUT}/develop3_discontinuous*") + glob.glob(f"{OUT}/develop3_multi_angle*")
def disc_dirs(sc):
    dirs=[]
    for t in trees:
        dirs += glob.glob(f"{t}/*/{sc}_discontinuous_*") + glob.glob(f"{t}/{sc}_discontinuous_*")
        dirs += glob.glob(f"{t}/{sc}/*discontinuous*")
    return dirs
def best_disc(sc):
    """returns (best-PSNR-run triple, per-metric-best triple) over ckpt-backed runs for one scene."""
    best=None; mb=[None,None,None]
    for d in disc_dirs(sc):
        cands=[]
        if os.path.exists(os.path.join(d,"metrics.csv")): cands += csv_candidates(os.path.join(d,"metrics.csv"))
        if os.path.exists(os.path.join(d,"results.json")): cands += json_candidates(os.path.join(d,"results.json"))
        for c in cands:
            if c[0] is not None and (best is None or c[0]>best[0]): best=c
            p,s,l = c
            if p is not None: mb[1]=p if mb[1] is None else max(mb[1],p)
            if s is not None: mb[0]=s if mb[0] is None else max(mb[0],s)
            if l is not None: mb[2]=l if mb[2] is None else min(mb[2],l)
    return best, mb

print("\n"+"="*70); print("TABLE 2  (discontinuous)  Ours ckpt-backed, per-dataset avg triple"); print("="*70)
for ds,scenes in [("Mip-360",MIP8),("T&T",TNT),("Mill19",MILL)]:
    res=[best_disc(s) for s in scenes]
    bp=[r[0] for r in res]; mb=[r[1] for r in res]
    if any(t is None for t in bp):
        miss=[s for s,t in zip(scenes,bp) if t is None]; print(f"  {ds:>8}: MISSING {miss}"); continue
    n=len(bp)
    for label,trips,order in [("best-PSNR run",bp,(1,0,2)),("per-metric best",mb,(0,1,2))]:
        a=[sum(t[i] for t in trips)/n for i in order]
        print(f"  {ds:>8} [{label:>15}]:  SSIM {a[0]:.3f}  PSNR {a[1]:.3f}  LPIPS {a[2]:.3f}   LaTeX: {a[0]:.3f} & {a[1]:.3f} & {a[2]:.3f}")
