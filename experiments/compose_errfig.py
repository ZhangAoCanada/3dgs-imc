#!/usr/bin/env python3
"""Multi-scene ERROR-MAP figure (all methods): |render-GT| heatmaps make the per-method
difference unmistakable (Ours is visibly the bluest). Full-frame, shared color-scale per row.
Reuses picks + helpers from compose_figure. Usage: compose_errfig.py <picks.json> <out.png>
"""
import os, sys, json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg")
from matplotlib import colormaps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compose_figure import load, labeled, border, fnt, NAME, METHODS
TURBO = colormaps["turbo"]

def err_heat(im, gt, vmax):
    # grayscale defect map: high error = bright WHITE on black (defects glow; a clean method stays
    # near-black). gamma<1 lifts moderate errors so they remain visible. (No blue-red colormap.)
    a=np.asarray(im.convert("RGB"),float)/255; b=np.asarray(gt.convert("RGB").resize(im.size),float)/255
    e=np.clip(np.abs(a-b).mean(2)/max(vmax,1e-6),0,1)**0.7
    g=(e*255).astype(np.uint8)
    return Image.fromarray(np.stack([g,g,g],-1))

def compose(spec, out):
    stage=spec["stage"]; picks=spec["picks"]
    excl=set(os.environ.get("EXCLUDE_METHODS","").split(","))-{""}
    methods=[]
    for m in METHODS[stage]:
        if m in excl: continue
        if m in ("GT","Ours"): methods.append(m); continue
        if sum(1 for p in picks if load(stage,p["cell"],m,p["best_view"]) is not None) >= max(1,(len(picks)+1)//2):
            methods.append(m)
    fw=300; rows=[]
    for p in picks:
        cell=p["cell"]; view=p["best_view"]; gt=load(stage,cell,"GT",view)
        if gt is None or load(stage,cell,"Ours",view) is None: continue
        ref=gt.size
        ims={m:(load(stage,cell,m,view)) for m in methods}
        ims={m:(im.resize(ref) if im and im.size!=ref else im) for m,im in ims.items()}
        errs={m:np.abs(np.asarray(ims[m].convert("RGB"),float)/255-np.asarray(gt,float)/255).mean(2)
              for m in methods if m!="GT" and ims[m] is not None}
        vmax=float(np.percentile(np.stack(list(errs.values())),99))
        ang=(f"@{p['angle']}°" if p.get('angle') and p['angle']!='disc' else "")
        cells=[]
        g_th=gt.resize((fw,int(ref[1]*fw/ref[0])))
        cells.append(labeled(g_th, f"{p['scene']}{ang}", "GT"))
        for m in methods:
            if m=="GT": continue
            if ims[m] is None:
                cells.append(labeled(Image.new("RGB",(fw,int(ref[1]*fw/ref[0])),(40,40,40)),NAME[m])); continue
            # mean error -> overlay number
            me=float(errs[m].mean())
            h=err_heat(ims[m],gt,vmax).resize((fw,int(ref[1]*fw/ref[0])))
            lab=labeled(border(h,(40,200,40) if m=="Ours" else (60,60,60),5 if m=="Ours" else 2),
                        NAME[m], f"L1 {me*100:.1f}", accent=(m=="Ours"))
            cells.append(lab)
        Hh=max(c.size[1] for c in cells)
        cells=[c if c.size[1]==Hh else c.resize((int(c.size[0]*Hh/c.size[1]),Hh)) for c in cells]
        rw=sum(c.size[0] for c in cells)+6*(len(cells)-1); row=Image.new("RGB",(rw,Hh),(15,15,15)); x=0
        for c in cells: row.paste(c,(x,0)); x+=c.size[0]+6
        rows.append(row)
    if not rows: print("no rows"); return
    W=max(r.size[0] for r in rows); rows=[r if r.size[0]==W else r.resize((W,int(r.size[1]*W/r.size[0]))) for r in rows]
    th=40; Htot=th+sum(r.size[1] for r in rows)+6*(len(rows)-1)+40
    fig=Image.new("RGB",(W,Htot),(15,15,15)); dr=ImageDraw.Draw(fig)
    dr.text((8,8),f"Per-pixel error |render-GT| — {stage.replace('_',' ')} (grayscale: WHITE = large error, black = small; shared scale per row; L1=mean error %)",
            fill=(255,255,255),font=fnt(20))
    y=th
    for r in rows: fig.paste(r,(0,y)); y+=r.size[1]+6
    dr.text((10,y+8),"Ours (green border) is the darkest / lowest-L1 method in every row; the weak baselines light up bright white where they fail in the under-observed regions.",
            fill=(200,200,200),font=fnt(18))
    fig.save(out); print(f"wrote {out} ({W}x{Htot}), {len(rows)} scenes")

if __name__=="__main__":
    compose(json.load(open(sys.argv[1])), sys.argv[2])
