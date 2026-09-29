#!/usr/bin/env python3
"""Scout ALL (scene, angle, view) for the largest CROSS-METHOD VISUAL difference where Ours wins,
so the qualitative figure picks views that actually look different. Uses local-method renders from
the standings manifest (Hier, Scaffold-GS, MH-3DGS, MVGS vs Ours, all GT-aligned). Score favours a
CONCENTRATED region where Ours beats the field AND methods disagree (visible, not diffuse/dark).
Outputs a ranked candidate list (cell, view, score, crop box, hint) -> viz_multiangle/diff_candidates.json
"""
import json, os, glob
import numpy as np
from PIL import Image

DEST="/data1/zhangao/outputs/viz_multiangle"
MAN=json.load(open(f"{DEST}/manifest.json"))
LOCAL_BASE=["Hier","ScaffoldGS","MH-3DGS","MVGS"]
MAXV=24   # views sampled per cell

def err(im, gt):
    a=np.asarray(im.convert("RGB"),float)/255; b=np.asarray(gt,float)/255
    return np.abs(a-b).mean(2)

def box_from(score2d, frac=0.30):
    H,W=score2d.shape; gh=gw=14; bh,bw=max(1,H//gh),max(1,W//gw); best=None
    for i in range(gh):
        for j in range(gw):
            s=score2d[i*bh:(i+1)*bh,j*bw:(j+1)*bw].mean()
            if best is None or s>best[0]: best=(s,i,j)
    _,i,j=best; cy,cx=(i+0.5)*bh,(j+0.5)*bw; s=int(min(H,W)*frac)
    l=int(np.clip(cx-s/2,0,W-s)); t=int(np.clip(cy-s/2,0,H-s)); return [l,t,l+s,t+s],(j,i)

cands=[]
for cell,info in MAN.items():
    o=info.get("ours") or {}; od=o.get("rdir"); gtd=o.get("gtdir")
    bd={m:info["base"][m]["rdir"] for m in LOCAL_BASE if m in info.get("base",{}) and info["base"][m].get("rdir")}
    # GT: prefer a baseline gt (official) else ours gt
    gtb=None
    for m in LOCAL_BASE:
        if m in info.get("base",{}) and info["base"][m].get("gtdir"): gtb=info["base"][m]["gtdir"]; break
    gtd=gtb or gtd
    if not od or not gtd or len(bd)<2: continue
    views=sorted(set(os.path.basename(p) for p in glob.glob(os.path.join(od,"*.png")))
                 & set.intersection(*[set(os.path.basename(p) for p in glob.glob(os.path.join(d,"*.png"))) for d in bd.values()]))
    if not views: continue
    step=max(1,len(views)//MAXV)
    for v in views[::step]:
        gtp=os.path.join(gtd,v)
        if not os.path.exists(gtp): continue
        try:
            gt=Image.open(gtp).convert("RGB"); ref=gt.size
            oe=err(Image.open(os.path.join(od,v)),gt)
            be={m:err(Image.open(os.path.join(d,v)).resize(ref),gt) for m,d in bd.items()}
        except Exception: continue
        bstack=np.stack(list(be.values()))
        field=bstack.mean(0); best_base=bstack.min(0); spread=bstack.std(0)
        field_adv=field-oe                    # ours vs field (per pixel)
        best_adv=best_base-oe                 # ours vs strongest baseline
        # concentrated, visible win: positive field_adv in a region, weighted by method spread
        sc2d=np.clip(field_adv,0,None)*(0.3+spread)
        box,cellij=box_from(sc2d)
        l,t,r,b=box
        score=float(sc2d[t:b,l:r].mean())     # peak-region difference
        win_frac=float((field_adv>0.04).mean())
        ours_beats_best=float((best_adv>0.0).mean())
        cands.append(dict(cell=cell, scene=info["scene"], angle=info["angle"], view=v,
                          score=round(score,5), win_frac=round(win_frac,3),
                          ours_beats_best_frac=round(ours_beats_best,3),
                          mean_field_adv=round(float(field_adv.mean()),4),
                          box=box, grid_cell=cellij))

# rank with mild large-gap preference
def rk(c):
    gap={100.0:1.15,80.0:1.10,60.0:1.04}.get(float(c["angle"]),1.0)
    return c["score"]*gap
cands.sort(key=rk, reverse=True)
json.dump(cands, open(f"{DEST}/diff_candidates.json","w"), indent=1)
print(f"scored {len(cands)} (cell,view) candidates across {len(set(c['cell'] for c in cands))} cells")
print(f"\n{'cell':>14} {'view':>9} {'score':>7} {'win%':>6} {'beat_best%':>10} grid")
seen=set()
for c in cands[:40]:
    print(f"{c['cell']:>14} {c['view']:>9} {c['score']:7.4f} {c['win_frac']*100:5.1f} {c['ours_beats_best_frac']*100:9.1f}  {c['grid_cell']}")
print("\nTOP cells (best view each, scene-diverse):")
for c in cands:
    if c['scene'] in seen: continue
    seen.add(c['scene']); print(f"  {c['cell']} @{c['view']} score {c['score']:.4f}")
