#!/usr/bin/env python3
"""Discontinuous-stage global floater figure: free-space floater mass per (scene, method),
referenced to full-coverage SfM. Standard-Gaussian methods only (3DGS, Hier, MCMC, MH-3DGS,
MVGS, Ours). BEV grid (rows=methods x cols=scenes) + summary bar + CSV."""
import os, glob, csv
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_discontinuous import find_ours, BR  # reuse resolver

OUT="/data1/zhangao/outputs"; G=f"{OUT}/viz_discontinuous/global"; os.makedirs(G,exist_ok=True)
SCENES=["bicycle","bonsai","counter","garden","kitchen","room","stump","treehill","flowers"]
METHODS=["3DGS","Hier","MCMC","MH-3DGS","MVGS","Ours"]
SUB={"3DGS":"3dgs-original","Hier":"3dgs-depth","MCMC":"3dgs-mcmc-original-1000000",
     "MH-3DGS":"mh-3dgs-original","MVGS":"mvgs-original-mv3"}
TAU_FRAC=0.015
def plypath(scene,m):
    if m=="Ours":
        d=find_ours(scene);
        if not d: return None
        g=sorted(glob.glob(os.path.join(d,"point_cloud","iteration_*","point_cloud.ply"))); return g[-1] if g else None
    g=sorted(glob.glob(os.path.join(BR,scene,SUB[m],"point_cloud","iteration_*","point_cloud.ply"))); return g[-1] if g else None
def read(p):
    v=PlyData.read(p)["vertex"]; xyz=np.stack([v["x"],v["y"],v["z"]],1).astype(np.float32)
    op=1/(1+np.exp(-v["opacity"].astype(np.float32)))
    fin=np.isfinite(xyz).all(1)&np.isfinite(op)
    return xyz[fin],op[fin]

stats={m:{} for m in METHODS}; grids={}
extent={}
for scene in SCENES:
    refp=os.path.join(BR,scene,"mh-3dgs-original","input.ply")
    if not os.path.exists(refp): continue
    ref=np.stack([PlyData.read(refp)["vertex"][k] for k in("x","y","z")],1).astype(np.float32)
    lo,hi=np.percentile(ref,1,0),np.percentile(ref,99,0); diag=float(np.linalg.norm(hi-lo)); tau=TAU_FRAC*diag
    c=ref.mean(0); rc=ref-c
    Vt=np.linalg.svd(rc[np.random.default_rng(0).choice(len(rc),min(40000,len(rc)),replace=False)],full_matrices=False)[2]
    ax=Vt[:2]; proj=lambda x:(x-c)@ax.T; r2=proj(ref)
    pad=0.35; mn=r2.min(0)-pad*(r2.max(0)-r2.min(0)); mx=r2.max(0)+pad*(r2.max(0)-r2.min(0)); extent[scene]=(mn,mx,ax,c)
    tree=cKDTree(ref)
    for m in METHODS:
        p=plypath(scene,m)
        if not p or not os.path.exists(p): grids[(m,scene)]=None; continue
        xyz,op=read(p); d,_=tree.query(xyz,k=1,workers=-1); fl=d>tau
        stats[m][scene]=float(op[fl].sum()/(op.sum()+1e-9))
        xy=proj(xyz[fl]); H,_,_=np.histogram2d(xy[:,0],xy[:,1],bins=300,range=[[mn[0],mx[0]],[mn[1],mx[1]]],weights=op[fl])
        grids[(m,scene)]=H.T
    print(f"{scene}: "+"  ".join(f"{m} {stats[m].get(scene,float('nan'))*100:.1f}%" for m in METHODS),flush=True)

# summary bar
means={m:np.mean([stats[m][s] for s in stats[m]]) for m in METHODS if stats[m]}
plt.figure(figsize=(6.5,4.5))
cols={"3DGS":"#8c564b","Hier":"#1f77b4","MCMC":"#2ca02c","MH-3DGS":"#9467bd","MVGS":"#ff7f0e","Ours":"crimson"}
bars=plt.bar([("Ours (GS-LSS)" if m=="Ours" else m) for m in METHODS],[means[m]*100 for m in METHODS],color=[cols[m] for m in METHODS])
for b,m in zip(bars,METHODS): plt.text(b.get_x()+b.get_width()/2,b.get_height()*100 if False else means[m]*100+0.2,f"{means[m]*100:.1f}%",ha="center",fontweight="bold")
plt.ylabel("Free-space floater mass fraction (%)"); plt.title("Discontinuous stage: scene-level floaters (mean over 9 scenes)\nreferenced to full-coverage SfM; lower = cleaner")
plt.tight_layout(); plt.savefig(f"{G}/floater_summary_bar.png",dpi=130); plt.close()
# BEV grid rows=methods cols=scenes
vmax=np.percentile(np.concatenate([g.ravel() for g in grids.values() if g is not None]),99.5)+1e-6
fig,axx=plt.subplots(len(METHODS),len(SCENES),figsize=(1.8*len(SCENES),1.8*len(METHODS)))
for i,m in enumerate(METHODS):
    for j,sc in enumerate(SCENES):
        a=axx[i,j]; g=grids.get((m,sc)); a.set_xticks([]); a.set_yticks([])
        if g is not None:
            a.imshow(np.clip(g/vmax,0,1),origin="lower",cmap="inferno",vmin=0,vmax=1,aspect="auto")
            a.text(0.96,0.04,f"{stats[m].get(sc,0)*100:.0f}%",color="w",ha="right",va="bottom",transform=a.transAxes,fontsize=8,fontweight="bold")
        else: a.text(0.5,0.5,"n/a",color="gray",ha="center",transform=a.transAxes,fontsize=8)
        if i==0: a.set_title(sc,fontsize=9)
        if j==0: a.set_ylabel(("Ours" if m=="Ours" else m),fontsize=9,fontweight=("bold" if m=="Ours" else "normal"))
fig.suptitle("Discontinuous: free-space floater mass (top-down BEV) — Ours stays cleanest",fontsize=12,y=0.997)
plt.tight_layout(rect=[0,0,1,0.985]); plt.savefig(f"{G}/floater_bev_grid.png",dpi=115); plt.close()
with open(f"{G}/floater_stats.csv","w") as fh:
    w=csv.writer(fh); w.writerow(["scene"]+METHODS)
    for sc in SCENES: w.writerow([sc]+[f"{stats[m].get(sc,float('nan'))*100:.2f}" for m in METHODS])
    w.writerow(["MEAN"]+[f"{means[m]*100:.2f}" for m in METHODS])
print("\n=== disc floater means ==="); [print(f"  {m:>8}: {means[m]*100:.2f}%") for m in METHODS]
print(f"-> {G}")
