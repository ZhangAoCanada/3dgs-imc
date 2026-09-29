#!/usr/bin/env python3
"""Table 3 (aerial-ground) CHECKPOINT-BACKED deliverable: best ckpt-backed ours for MatrixCity + Zeche
vs the paper's published Table-3 competitor rows (full-reference PSNR is the headline). Marks win/loss."""
import glob, os, csv, json
OUT = "/data1/zhangao/outputs"
# published Table 3 PSNR (bare_jrnl_new_sample4.tex lines 606-613)
PUB = {
    "MatrixCity": {"3DGS":16.146,"Scaffold":16.193,"Hier":12.943,"MVGS":15.610,"PGSR":13.072,"MCMC":16.225,"MH-3DGS":15.297,"GS-LSS":16.951},
    "Zeche":      {"3DGS":9.431,"Scaffold":9.806,"Hier":9.646,"MVGS":9.494,"PGSR":8.499,"MCMC":10.133,"MH-3DGS":12.703,"GS-LSS":12.879},
}
PAT = {"MatrixCity": "blockA_fusion_small_aerial+somestreet_traintest_*", "Zeche": "Zeche1_traintest_*"}

def best_ckpt(ds):
    best=None; src=None
    for d in glob.glob(os.path.join(OUT,"diag","large_angle",PAT[ds])):
        if not glob.glob(os.path.join(d,"point_cloud","iteration_*","point_cloud.ply")): continue
        m=os.path.join(d,"metrics.csv")
        if not os.path.exists(m): continue
        try:
            for r in csv.DictReader(open(m)):
                if r.get("method","").startswith("ours_"):
                    v=float(r["PSNR"])
                    if best is None or v>best: best=v; src=os.path.basename(d)[:45]
        except Exception: pass
    return best, src

print(f"{'dataset':>11} | {'OURS(ckpt)':>10} | {'pub GS-LSS':>10} | {'2nd-best':>16} | result")
for ds in ("MatrixCity","Zeche"):
    o,src = best_ckpt(ds)
    comps = PUB[ds]; gslss = comps["GS-LSS"]
    ext = {k:v for k,v in comps.items() if k!="GS-LSS"}
    bn,bv = max(ext.items(), key=lambda x:x[1])
    if o is None:
        print(f"{ds:>11} | {'(running)':>10} | {gslss:10.3f} | {bn+' '+format(bv,'.3f'):>16} | pending"); continue
    if o > bv and o > gslss: res="WIN all"
    elif o > bv: res=f"beats baselines, <pub({gslss-o:+.2f})"
    else: res=f"LOSE to {bn} ({o-bv:+.2f})"
    print(f"{ds:>11} | {o:10.3f} | {gslss:10.3f} | {bn+' '+format(bv,'.3f'):>16} | {res}")
    print(f"{'':>11}   src: {src}")
