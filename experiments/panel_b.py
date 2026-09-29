#!/usr/bin/env python3
"""Panel B — signed density-deviation (BEV). For one scene, each method's free-space floater
density MINUS the across-method mean, diverging colormap (RdBu_r: red = MORE floaters than the
field average here, blue = FEWER / cleaner). Ours & MH-3DGS read blue (clean); CDGS/PGSR/Hier/MVGS
read red. References the field mean (not Ours), so it is not circular. Usage: panel_b.py <scene>"""
import os, sys, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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
def load(m,a):
    p=f"{GA}/{scene}__{m}__{a}.npy"
    if not os.path.exists(p): return None
    h=gaussian_filter(np.load(p),2.0); fr=frac(m,a)
    if fr is None or fr<=0: return None
    total=h.sum()/fr                      # floater_mass / fraction = total Gaussian mass
    return h/(total+1e-9)                 # floater density normalized by TOTAL mass (integrates to fraction)

# per angle: deviation of each method's (count-fair) floater density from the across-method mean
dev={}; present={}
for a in ANGLES:
    norm={m:load(m,a) for m in ORDER}
    norm={m:g for m,g in norm.items() if g is not None}
    if not norm: continue
    mean=np.mean(np.stack(list(norm.values())),0)
    for m in norm: dev[(m,a)]=norm[m]-mean
    present[a]=list(norm)
# symmetric scale
alld=np.concatenate([d.ravel() for d in dev.values()]) if dev else np.array([0])
vlim=np.percentile(np.abs(alld),99.5)+1e-12
fig,axx=plt.subplots(len(ORDER),len(ANGLES),figsize=(2.3*len(ANGLES),2.3*len(ORDER)))
for i,m in enumerate(ORDER):
    for j,a in enumerate(ANGLES):
        ax=axx[i,j]; ax.set_xticks([]); ax.set_yticks([])
        if (m,a) in dev:
            ax.imshow(np.clip(dev[(m,a)]/vlim,-1,1),origin="lower",cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
        else: ax.text(.5,.5,"n/a",color="gray",ha="center",transform=ax.transAxes)
        if i==0: ax.set_title(f"{a}° gap",fontsize=12)
        if j==0: ax.set_ylabel(NICE[m],fontsize=11,fontweight=("bold" if m=="Ours" else "normal"),color=("crimson" if m=="Ours" else "black"))
fig.suptitle(f"Panel B — signed floater-density deviation vs method mean, {scene}  (red = more floaters than average, blue = cleaner)",fontsize=13,y=0.997)
plt.tight_layout(rect=[0,0,1,0.985])
out=f"{GA}/PANELB_density_deviation_{scene}.png"; plt.savefig(out,dpi=115); plt.close()
print("wrote",out)
