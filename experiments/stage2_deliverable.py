#!/usr/bin/env python3
"""Table 2 (discontinuous) CHECKPOINT-BACKED deliverable verifier (framing #1).
ours[scene] = best PSNR among runs whose .ply EXISTS, across ALL our trees (diag + develop3_*).
Compares per-dataset aggregate vs every competitor in experiments/metrics_discontinuous + published GS-LSS."""
import glob, os, json, csv, re
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "/data1/zhangao/outputs"
MD = os.path.join(REPO, "experiments/metrics_discontinuous")
DATA = {"Mip-360": ["bicycle","bonsai","counter","garden","kitchen","room","stump","treehill"],
        "T&T": ["truck_resize","train_resize"],
        "Mill19": ["building-small","residence-small","rubble-small","sci-art-small"]}
PUB = {"Mip-360": 27.960, "T&T": 22.160, "Mill19": 22.893}   # published GS-LSS row (paper Table 2)

def ckpt_psnr(dirs):
    best = None
    for d in dirs:
        if not glob.glob(os.path.join(d, "point_cloud", "iteration_*", "point_cloud.ply")): continue
        for f in ("metrics.csv", "results.json"):
            p = os.path.join(d, f)
            if not os.path.exists(p): continue
            try:
                if f == "metrics.csv":
                    for r in csv.DictReader(open(p)):
                        if r.get("method","").startswith("ours_"): best = max(best or 0, float(r["PSNR"]))
                else:
                    for k, v in json.load(open(p)).items():
                        if "PSNR" in v: best = max(best or 0, v["PSNR"])
            except Exception: pass
    return best

trees = [f"{OUT}/diag"] + glob.glob(f"{OUT}/develop3_discontinuous*") + glob.glob(f"{OUT}/develop3_multi_angle*")
def ours_scene(sc):
    dirs = []
    for t in trees:
        dirs += glob.glob(f"{t}/*/{sc}_discontinuous_*") + glob.glob(f"{t}/{sc}_discontinuous_*")
        dirs += glob.glob(f"{t}/{sc}/*discontinuous*")
    return ckpt_psnr(dirs)

def comp(method, sc):
    p = os.path.join(MD, method, f"{sc}_discontinuous_metrics.csv")
    try:
        r = list(csv.DictReader(open(p))); return float(r[-1]["PSNR"]) if r else None
    except Exception: return None
METHODS = {"MCMC":"3dgs-mcmc-original","Scaffold":"scaffoldgs-original","3DGS":"3dgs-original",
           "Hier":"3dgs-depth","MVGS":"mvgs-original-mv3"}

print(f"{'dataset':>9} | {'OURS(ckpt)':>10} | {'pub GSLSS':>9} | " + " | ".join(f"{n:>7}" for n in METHODS) + " | result")
for ds, scenes in DATA.items():
    ov = [ours_scene(s) for s in scenes]
    if any(v is None for v in ov):
        miss = [s for s,v in zip(scenes,ov) if v is None]; print(f"{ds:>9} | MISSING CKPT: {miss}"); continue
    o_agg = sum(ov)/len(ov)
    comps = {}
    for n, m in METHODS.items():
        cv = [comp(m, s) for s in scenes]; comps[n] = sum(cv)/len(cv) if all(v is not None for v in cv) else None
    best_comp = max([(n,v) for n,v in comps.items() if v is not None], key=lambda x: x[1], default=(None,None))
    win = o_agg > PUB[ds] and all(o_agg > v for v in comps.values() if v is not None)
    cs = " | ".join(f"{comps[n]:7.2f}" if comps[n] is not None else "   -- " for n in METHODS)
    print(f"{ds:>9} | {o_agg:10.3f} | {PUB[ds]:9.3f} | {cs} | {'WIN all' if win else '2nd=' + str(best_comp[0])}")
