#!/usr/bin/env python3
"""Stage-2 (Table 2, discontinuous trajectories) AGGREGATE standing.
Per scene: our best = max(old develop3_discontinuous run, any new diag *_discontinuous_* run).
Per dataset (Mip9 / T&T2 / Mill4): mean(ours) vs mean(each competitor). Bar = beat 3DGS-MCMC."""
import csv, glob, json, os, re
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = "/data1/zhangao/outputs"
MD = os.path.join(REPO, "experiments/metrics_discontinuous")
OLD = os.path.join(OUTD, "develop3_discontinuous_woantialiasing")
DIAG = os.path.join(OUTD, "diag")
DATASETS = {   # paper Table 2 Mip aggregate = MIP8 (flowers EXCLUDED; verified vs verify_paper_tables2.py MCMC=27.517)
    "Mip-360(MIP8)": ["bicycle","bonsai","counter","garden","kitchen","room","stump","treehill"],
    "T&T":     ["train_resize","truck_resize"],
    "Mill19":  ["building-small","residence-small","rubble-small","sci-art-small"],
}
def csv_psnr(path):
    try: rows=list(csv.DictReader(open(path)))
    except Exception: return None
    best=None
    for r in rows:
        try: p=float(r["PSNR"])
        except Exception: continue
        if best is None or p>best: best=p
    return best

# our OLD per-scene (results.json ours_30000)
old={}
for sc in sum(DATASETS.values(),[]):
    for rj in glob.glob(os.path.join(OLD, sc, "*", "results.json")):
        try: d=json.load(open(rj))
        except Exception: continue
        k="ours_30000" if "ours_30000" in d else (sorted(d)[-1] if d else None)
        if k and "PSNR" in d[k]:
            v=d[k]["PSNR"]; old[sc]=max(old.get(sc,-1),v)
# our NEW per-scene (diag *_discontinuous_* metrics, any subdir/dataset)
new=defaultdict(lambda:-1)
for mc in glob.glob(os.path.join(DIAG,"*","*_discontinuous_*","metrics.csv")):
    d=os.path.basename(os.path.dirname(mc)); m=re.match(r"^(.+?)_discontinuous_",d)
    if not m: continue
    sc=m.group(1); v=csv_psnr(mc)
    if v is not None and v>new[sc]: new[sc]=v
ours={sc:max(old.get(sc,-1), new.get(sc,-1)) for sc in sum(DATASETS.values(),[])}
ours={k:v for k,v in ours.items() if v>0}
newflag={sc:(new.get(sc,-1)>=old.get(sc,-1) and new.get(sc,-1)>0) for sc in ours}

# competitors
METHODS=["3dgs-mcmc-original","scaffoldgs-original","3dgs-original","3dgs-depth","mvgs-original-mv3","cdgs-original"]
comp={m:{} for m in METHODS}
for m in METHODS:
    for sc in sum(DATASETS.values(),[]):
        f=os.path.join(MD,m,f"{sc}_discontinuous_metrics.csv")
        if os.path.isfile(f):
            v=csv_psnr(f);
            if v is not None: comp[m][sc]=v

NAME={"3dgs-mcmc-original":"MCMC","scaffoldgs-original":"Scaf","3dgs-original":"3DGS","3dgs-depth":"Hier","mvgs-original-mv3":"MVGS","cdgs-original":"cdgs"}
for ds,scenes in DATASETS.items():
    have=[s for s in scenes if s in ours]
    print(f"\n=== {ds} ({len(have)}/{len(scenes)} scenes have our results) ===")
    print(f"{'scene':>16} | {'ours':>7} | {'MCMC':>7} | {'d':>6} | src")
    for s in scenes:
        o=ours.get(s); mc=comp["3dgs-mcmc-original"].get(s)
        if o is None: print(f"{s:>16} |   --    |  {mc if mc else '--'}  |   --   | (no ours)"); continue
        dd=o-(mc if mc else 0); src="NEW" if newflag.get(s) else "old"
        print(f"{s:>16} | {o:>7.3f} | {(mc if mc else 0):>7.3f} | {dd:>+6.3f} | {src}")
    oa=sum(ours[s] for s in have)/len(have) if have else 0
    print(f"  AGG ours={oa:.3f}", end="")
    for m in METHODS:
        cc=[s for s in have if s in comp[m]]
        if cc:
            ca=sum(comp[m][s] for s in cc)/len(cc)
            print(f" | {NAME[m]}={ca:.3f}({ '+' if oa>=ca else ''}{oa-ca:.3f})", end="")
    print()
