#!/usr/bin/env python3
"""Generate per-pick all-method RGB + error strips (single best view) for vision curation."""
import os, sys, json
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compose_figure import load, labeled, NAME, METHODS
from compose_errfig import err_heat
def hstrip(imgs,pad=4):
    H=max(i.size[1] for i in imgs); imgs=[i if i.size[1]==H else i.resize((int(i.size[0]*H/i.size[1]),H)) for i in imgs]
    W=sum(i.size[0] for i in imgs)+pad*(len(imgs)-1); out=Image.new("RGB",(W,H),(15,15,15)); x=0
    for i in imgs: out.paste(i,(x,0)); x+=i.size[0]+pad
    return out
VM="/data1/zhangao/outputs/viz_multiangle"
OUT=f"{VM}/curation_strips"; os.makedirs(OUT,exist_ok=True)
picks=json.load(open(f"{VM}/fig_picks_scout.json"))["picks"]
methods=METHODS["coverage_gap"]
for p in picks:
    cell=p["cell"]; v=p["best_view"]; gt=load("coverage_gap",cell,"GT",v)
    if gt is None: continue
    ref=gt.size
    ims={m:load("coverage_gap",cell,m,v) for m in methods}
    ims={m:(im.resize(ref) if im and im.size!=ref else im) for m,im in ims.items()}
    rgb=hstrip([labeled(ims[m] if ims[m] else Image.new("RGB",ref,(40,40,40)),NAME[m]) for m in methods])
    rgb.save(f"{OUT}/{cell}_rgb.png")
    errs={m:np.abs(np.asarray(ims[m].convert("RGB"),float)/255-np.asarray(gt,float)/255).mean(2) for m in methods if m!="GT" and ims[m]}
    vmax=float(np.percentile(np.stack(list(errs.values())),99))
    ep=[labeled(gt,"GT")]+[labeled(err_heat(ims[m],gt,vmax) if ims[m] else Image.new("RGB",ref,(40,40,40)),NAME[m]) for m in methods if m!="GT"]
    hstrip(ep).save(f"{OUT}/{cell}_err.png")
    print("strips",cell)
print("done ->",OUT)
