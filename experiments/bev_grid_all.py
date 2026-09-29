#!/usr/bin/env python3
"""All-method co-registered BEV floater grid for one scene: rows = methods, cols = gap angles.
Reads the per-(scene,method,angle) histograms in global_all/. Shared colormap. Usage:
bev_grid_all.py <scene>"""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GA="/data1/zhangao/outputs/viz_multiangle/global_all"
scene=sys.argv[1] if len(sys.argv)>1 else "bonsai"
ANGLES=[20,40,60,80,100]
ORDER=["Hier","MCMC","CDGS","PGSR","MH-3DGS","MVGS","Ours"]
NICE={"Hier":"Hier-GS","MCMC":"3DGS-MCMC","CDGS":"CDGS","PGSR":"PGSR","MH-3DGS":"MH-3DGS","MVGS":"MVGS","Ours":"Ours (GS-LSS)"}

def frac(m,a):
    f=f"{GA}/{scene}__{m}.frac.json"
    if os.path.exists(f):
        d=json.load(open(f)); return d.get(str(a)) or d.get(str(float(a)))
    return None
grids={}
for m in ORDER:
    for a in ANGLES:
        p=f"{GA}/{scene}__{m}__{a}.npy"
        grids[(m,a)]=np.load(p) if os.path.exists(p) else None
allv=[g for g in grids.values() if g is not None]
if not allv: print("no data for",scene); sys.exit()
vmax=np.percentile(np.concatenate([g.ravel() for g in allv]),99.5)+1e-9
fig,axx=plt.subplots(len(ORDER),len(ANGLES),figsize=(2.3*len(ANGLES),2.3*len(ORDER)))
for i,m in enumerate(ORDER):
    for j,a in enumerate(ANGLES):
        ax=axx[i,j]; g=grids[(m,a)]; ax.set_xticks([]); ax.set_yticks([])
        if g is not None:
            ax.imshow(np.clip(g/vmax,0,1),origin="lower",cmap="inferno",vmin=0,vmax=1,aspect="auto")
            fr=frac(m,a)
            if fr is not None: ax.text(.96,.04,f"{fr*100:.0f}%",color="w",ha="right",va="bottom",transform=ax.transAxes,fontsize=10,fontweight="bold")
        else: ax.text(.5,.5,"n/a",color="gray",ha="center",transform=ax.transAxes)
        if i==0: ax.set_title(f"{a}° gap",fontsize=12)
        if j==0: ax.set_ylabel(NICE[m],fontsize=11,fontweight=("bold" if m=="Ours" else "normal"),color=("crimson" if m=="Ours" else "black"))
fig.suptitle(f"Free-space floater mass (top-down BEV), {scene} — all methods × gap angles; only floaters shown (warm=dense). Ours stays cleanest.",fontsize=13,y=0.997)
plt.tight_layout(rect=[0,0,1,0.985])
out=f"{GA}/BEV_allmethods_{scene}.png"; plt.savefig(out,dpi=115); plt.close()
print("wrote",out)
