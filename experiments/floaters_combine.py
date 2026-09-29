#!/usr/bin/env python3
"""Combine local + hust2 floater fractions into an ALL-METHOD bar + per-angle curve.
Local methods (vs full-coverage SfM, same frame): Ours, Hier, MH-3DGS, MVGS.
hust2 methods (self-referenced to each cell's own SfM): MCMC, CDGS, PGSR.
Scaffold-GS excluded (anchor PLY has no per-Gaussian opacity)."""
import os, glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GA="/data1/zhangao/outputs/viz_multiangle/global_all"
SCENES=["bicycle","bonsai","counter","garden","kitchen","room","stump","treehill","flowers"]
ANGLES=[20,40,60,80,100]
LOCAL=["Ours","Hier","MH-3DGS","MVGS"]
HUST=["MCMC","CDGS","PGSR"]
ORDER=["Hier","MCMC","CDGS","PGSR","MH-3DGS","MVGS","Ours"]
COL={"Hier":"#1f77b4","MCMC":"#2ca02c","CDGS":"#17becf","PGSR":"#8c564b","MH-3DGS":"#9467bd","MVGS":"#ff7f0e","Ours":"crimson"}

frac={m:{} for m in ORDER}   # method -> {(scene,angle): val}
# all methods now have {scene}__{method}.frac.json in GA (local + pulled hust2, same ref/proj)
for sc in SCENES:
    for m in ORDER:
        f=f"{GA}/{sc}__{m}.frac.json"
        if os.path.exists(f):
            for a,v in json.load(open(f)).items(): frac[m][(sc,int(float(a)))]=v

def mean(m):
    vs=list(frac[m].values()); return float(np.mean(vs)) if vs else float('nan')
def ang_mean(m,a):
    vs=[frac[m][(s,a)] for s in SCENES if (s,a) in frac[m]]; return float(np.mean(vs)) if vs else float('nan')

means={m:mean(m) for m in ORDER}
# bar
plt.figure(figsize=(7.5,4.6))
bars=plt.bar([("Ours (GS-LSS)" if m=="Ours" else m) for m in ORDER],[means[m]*100 for m in ORDER],
             color=[COL[m] for m in ORDER])
for b,m in zip(bars,ORDER): plt.text(b.get_x()+b.get_width()/2,means[m]*100+0.4,f"{means[m]*100:.1f}%",ha="center",fontweight="bold",fontsize=9)
plt.ylabel("Free-space floater mass fraction (%)")
plt.title("Scene-level floaters across ALL methods (mean over 9 scenes × gap angles)\nreferenced to full-coverage SfM; lower = cleaner")
plt.tight_layout(); plt.savefig(f"{GA}/floater_bar_allmethods.png",dpi=130); plt.close()
# curve
plt.figure(figsize=(7.5,5))
for m in ORDER:
    ys=[ang_mean(m,a)*100 for a in ANGLES]
    plt.plot(ANGLES,ys,"-o",lw=(3 if m=="Ours" else 1.7),color=COL[m],label=("Ours (GS-LSS)" if m=="Ours" else m),zorder=(5 if m=="Ours" else 3))
plt.xlabel("Test-view angle gap (degrees)  →  increasing viewpoint bias"); plt.ylabel("Free-space floater mass fraction (%)")
plt.title("Floater mass vs viewpoint bias — all methods (mean over 9 scenes)")
plt.grid(alpha=0.3); plt.legend(ncol=2,fontsize=9); plt.tight_layout()
plt.savefig(f"{GA}/floater_curve_allmethods.png",dpi=130); plt.close()
# csv
with open(f"{GA}/floater_allmethods.csv","w") as fh:
    fh.write("method,mean,"+",".join(f"a{a}" for a in ANGLES)+"\n")
    for m in ORDER: fh.write(f"{m},{means[m]*100:.2f},"+",".join(f"{ang_mean(m,a)*100:.2f}" for a in ANGLES)+"\n")
print("=== ALL-METHOD floater means (%) ===")
for m in ORDER: print(f"  {m:>9}: {means[m]*100:5.2f}")
print(f"-> {GA}/floater_bar_allmethods.png, floater_curve_allmethods.png, floater_allmethods.csv")
print("NOTE: Ours/Hier/MH/MVGS/MCMC vs the same full-coverage SfM; CDGS/PGSR self-referenced to their own SfM init (different COLMAP) — fractions comparable, spatial frames differ.")
