#!/usr/bin/env python3
"""Assemble all 9 methods for the coverage-gap figure rows (from fig_coverage_gap_*.json).
Local methods copied from manifest source dirs; MCMC/CDGS pulled from hust2 (00xxx names);
PGSR staged + remapped by GT content-match. Run on the dev machine (uses ssh master cm-hust2)."""
import json, os, glob, shutil, subprocess, sys
import numpy as np
from PIL import Image
VM="/data1/zhangao/outputs/viz_multiangle"
SSHOPT="-o ClearAllForwardings=yes -o ControlMaster=auto -o ControlPath=/tmp/cm3-%r -o ControlPersist=600 -o ServerAliveInterval=15 -o ServerAliveCountMax=4".split()
PAT={"MCMC":"3dgs-mcmc-original-*","CDGS":"cdgs-original","PGSR":"pgsr2-original-*"}
man=json.load(open(f"{VM}/manifest.json"))
picks=[]
for f in sorted(glob.glob(f"{VM}/fig_coverage_gap_*.json")):
    picks += json.load(open(f))["picks"]
# dedup (cell,view)
seen=set(); rows=[]
for p in picks:
    k=(p["cell"],p["best_view"])
    if k in seen: continue
    seen.add(k); rows.append(p)
print(f"{len(rows)} unique (cell,view) to assemble")
subprocess.run(["ssh",*SSHOPT,"-fN","hust2"],stderr=subprocess.DEVNULL)
def first(g):
    h=sorted(glob.glob(g)); return h[-1] if h else None
# 1. local copies
for p in rows:
    key=p.get("manifest_key") or p["cell"].replace("_","@")  # fallback
    info=man.get(key) or man.get(p["cell"].replace("_","@")+".0")
    if not info: print("NO MANIFEST",key); continue
    o=info["ours"]; base=info["base"]; v=p["best_view"]; cd=f"{VM}/cells/{p['cell']}/renders"
    gtd=None
    for m in ("MH-3DGS","MVGS","Hier","ScaffoldGS"):
        if m in base and base[m].get("gtdir"): gtd=base[m]["gtdir"]; break
    gtd=gtd or o.get("gtdir")
    srcs={"GT":gtd,"Ours":o.get("rdir"),"Hier":base.get("Hier",{}).get("rdir"),
          "ScaffoldGS":base.get("ScaffoldGS",{}).get("rdir"),"MH-3DGS":base.get("MH-3DGS",{}).get("rdir"),
          "MVGS":base.get("MVGS",{}).get("rdir")}
    for m,sd in srcs.items():
        os.makedirs(f"{cd}/{m}",exist_ok=True)
        if os.path.exists(f"{cd}/{m}/{v}"): continue
        if sd and os.path.exists(f"{sd}/{v}"): shutil.copy(f"{sd}/{v}",f"{cd}/{m}/{v}")
# 2. MCMC/CDGS single-file pulls
for p in rows:
    v=p["best_view"]; sc=p["scene"]; ang=p["angle"]; cd=f"{VM}/cells/{p['cell']}/renders"
    for m in ("MCMC","CDGS"):
        if os.path.exists(f"{cd}/{m}/{v}"): continue
        rdir=subprocess.run(["ssh",*SSHOPT,"hust2",f"ls -d /data1/zhangao/outputs/baselines_multi_angle/{sc}/{PAT[m]}_test_selection_angle_{ang}.0/test/ours_*/renders 2>/dev/null|head -1"],
                            capture_output=True,text=True,stdin=subprocess.DEVNULL).stdout.strip()
        if not rdir: continue
        os.makedirs(f"{cd}/{m}",exist_ok=True)
        subprocess.run(["rsync","-a","--timeout=60","-e",f"ssh {' '.join(SSHOPT)}",f"hust2:{rdir}/{v}",f"{cd}/{m}/{v}"],stdin=subprocess.DEVNULL)
# 3. PGSR: stage gt+renders per distinct cell, then remap
cells=sorted(set((p["cell"],p["scene"],p["angle"]) for p in rows))
for cell,sc,ang in cells:
    st=f"/tmp/pgsr_stage/{cell}"
    if os.path.isdir(f"{st}/renders") and os.listdir(f"{st}/renders"): continue
    base=subprocess.run(["ssh",*SSHOPT,"hust2",f"ls -d /data1/zhangao/outputs/baselines_multi_angle/{sc}/{PAT['PGSR']}_test_selection_angle_{ang}.0/test/ours_* 2>/dev/null|head -1"],
                        capture_output=True,text=True,stdin=subprocess.DEVNULL).stdout.strip()
    if not base: print("no pgsr",cell); continue
    for sub in ("gt","renders"):
        os.makedirs(f"{st}/{sub}",exist_ok=True)
        for _ in range(3):
            r=subprocess.run(["rsync","-a","--timeout=60","-e",f"ssh {' '.join(SSHOPT)}",f"hust2:{base}/{sub}/",f"{st}/{sub}/"],stdin=subprocess.DEVNULL)
            if r.returncode==0: break
def arr(p,sz): return np.asarray(Image.open(p).convert("RGB").resize(sz),float)
for p in rows:
    cell=p["cell"]; v=p["best_view"]; cd=f"{VM}/cells/{cell}/renders"
    if os.path.exists(f"{cd}/PGSR/{v}"): continue
    gtp=f"{cd}/GT/{v}"; pg=sorted(glob.glob(f"/tmp/pgsr_stage/{cell}/gt/*.png"))
    if not os.path.exists(gtp) or not pg: continue
    t=np.asarray(Image.open(gtp).convert("RGB"),float); sz=Image.open(gtp).size
    best=min(((np.mean((arr(g,sz)-t)**2),os.path.basename(g)) for g in pg), key=lambda x:x[0])
    rp=f"/tmp/pgsr_stage/{cell}/renders/{best[1]}"
    if best[0]<200 and os.path.exists(rp):
        os.makedirs(f"{cd}/PGSR",exist_ok=True); shutil.copy(rp,f"{cd}/PGSR/{v}")
# report
M=["GT","Hier","MCMC","CDGS","PGSR","ScaffoldGS","MH-3DGS","MVGS","Ours"]
miss=0
for p in rows:
    m=[x for x in M if not os.path.exists(f"{VM}/cells/{p['cell']}/renders/{x}/{p['best_view']}")]
    if m: print(f"  {p['cell']}/{p['best_view']} missing {m}"); miss+=len(m)
print(f"assembly done; {miss} missing method-views")
