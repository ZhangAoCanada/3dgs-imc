#!/usr/bin/env python3
"""Curate the strongest qualitative examples into paper-ready composites.
For each picked (cell, view): stack [RGB concat] / [error heatmaps] / [depth compare] vertically
into one figure. Also copy the global floater figures. Picks chosen by viewing the error maps:
indoor bonsai scenes show the clearest floater wins."""
import os, shutil
from PIL import Image
DEST = "/data1/zhangao/outputs/viz_multiangle"
CUR = os.path.join(DEST, "curated"); os.makedirs(CUR, exist_ok=True)
PICKS = [("bonsai_100", 69), ("bonsai_100", 75), ("bonsai_60", 55)]

def vstack(imgs, pad=6, bg=(0,0,0)):
    W = max(i.size[0] for i in imgs)
    imgs = [i if i.size[0]==W else i.resize((W,int(i.size[1]*W/i.size[0]))) for i in imgs]
    H = sum(i.size[1] for i in imgs)+pad*(len(imgs)-1)
    out = Image.new("RGB",(W,H),bg); y=0
    for i in imgs: out.paste(i,(0,y)); y+=i.size[1]+pad
    return out

for cell, v in PICKS:
    parts=[]
    for sub in ("concat_full","errmap"):
        p=os.path.join(DEST,"cells",cell,sub,f"{v:05d}.png")
        if os.path.exists(p): parts.append(Image.open(p).convert("RGB"))
    dp=os.path.join(DEST,"depth",cell,"compare",f"{v:05d}.png")
    if os.path.exists(dp): parts.append(Image.open(dp).convert("RGB"))
    if parts:
        comp=vstack(parts)
        comp.save(os.path.join(CUR,f"{cell}_{v:05d}_composite.png"))
        print(f"composite {cell}@{v:05d}: {len(parts)} rows -> {comp.size}")

# copy global floater showcase
for src in ["global/bonsai/floater_bev_grid.png","global/bonsai/floater_curve.png",
            "global/floater_summary_bar.png","global/floater_summary.csv",
            "global/kitchen/floater_bev_grid.png","global/stump/floater_bev_grid.png"]:
    s=os.path.join(DEST,src)
    if os.path.exists(s): shutil.copy(s, os.path.join(CUR, src.replace("/","__")))
print("curated ->", CUR)
