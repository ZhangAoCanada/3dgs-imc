#!/usr/bin/env python3
"""Discontinuous-trajectory stage: standings + qualitative (concat + error maps).
OURS = best-PSNR disc run with ply+renders (scan develop3_discontinuous* + diag).
Baselines (6): 3DGS, Hier(3dgs-depth), Scaffold-GS, MCMC, MH-3DGS, MVGS.
All view-aligned (GT md5 identical, 20-view discontinuous split). Output: viz_discontinuous/.
"""
import os, glob, json, csv, re, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg")
from matplotlib import colormaps

OUT = "/data1/zhangao/outputs"
BR = f"{OUT}/baselines_discontinuous"
DEST = f"{OUT}/viz_discontinuous"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TURBO = colormaps["turbo"]
SCENES = ["bicycle","bonsai","counter","garden","kitchen","room","stump","treehill","flowers"]
BASE_METHODS = {  # label -> subdir
    "3DGS":"3dgs-original", "Hier":"3dgs-depth", "ScaffoldGS":"scaffoldgs-original",
    "MCMC":"3dgs-mcmc-original-1000000", "MH-3DGS":"mh-3dgs-original", "MVGS":"mvgs-original-mv3"}
ORDER = ["GT","3DGS","Hier","ScaffoldGS","MCMC","MH-3DGS","MVGS","Ours"]
NAME = {"GT":"Ground Truth","3DGS":"3DGS","Hier":"Hier-GS","ScaffoldGS":"Scaffold-GS",
        "MCMC":"3DGS-MCMC","MH-3DGS":"MH-3DGS","MVGS":"MVGS","Ours":"Ours (GS-LSS)"}
K_HERO = 6

def fnt(s):
    try: return ImageFont.truetype(FONT,s)
    except: return ImageFont.load_default()
def label(img,t,sub=None,good=False):
    W,H=img.size; bar=max(24,H//13); sz=int(bar*0.58)
    cv=Image.new("RGB",(W,H+bar),(15,15,15)); cv.paste(img,(0,bar)); dr=ImageDraw.Draw(cv)
    dr.text((5,bar*0.12),t,fill=(255,255,255),font=fnt(sz))
    if sub:
        c=(120,230,120) if good else (230,200,120); tw=dr.textlength(sub,font=fnt(int(sz*0.85)))
        dr.text((W-tw-5,bar*0.16),sub,fill=c,font=fnt(int(sz*0.85)))
    return cv
def hstrip(imgs,pad=4):
    H=max(i.size[1] for i in imgs); imgs=[i if i.size[1]==H else i.resize((int(i.size[0]*H/i.size[1]),H)) for i in imgs]
    W=sum(i.size[0] for i in imgs)+pad*(len(imgs)-1); out=Image.new("RGB",(W,H),(15,15,15)); x=0
    for i in imgs: out.paste(i,(x,0)); x+=i.size[0]+pad
    return out
def err_heat(r,g,vmax):
    a=np.asarray(r.convert("RGB"),float)/255; b=np.asarray(g.convert("RGB").resize(r.size),float)/255
    e=np.clip(np.abs(a-b).mean(2)/max(vmax,1e-6),0,1)
    return Image.fromarray((TURBO(e)[...,:3]*255).astype(np.uint8))
def rdir(d):
    g=sorted(glob.glob(os.path.join(d,"test","ours_*","renders"))); return g[-1] if g else None
def gdir(d):
    g=sorted(glob.glob(os.path.join(d,"test","ours_*","gt"))); return g[-1] if g else None
def metrics(d):
    rj=os.path.join(d,"results.json")
    if os.path.exists(rj):
        try:
            j=json.load(open(rj)); k="ours_30000" if "ours_30000" in j else sorted(j)[-1]
            return j[k].get("PSNR"),j[k].get("SSIM"),j[k].get("LPIPS")
        except: pass
    mc=os.path.join(d,"metrics.csv")
    if os.path.exists(mc):
        best=None
        for r in csv.DictReader(open(mc)):
            if r.get("method","").startswith("ours_"):
                try: best=max(best or (0,0,0),(float(r["PSNR"]),float(r["SSIM"]),float(r["LPIPS"])),key=lambda t:t[0])
                except: pass
        return best if best else (None,None,None)
    return (None,None,None)
def per_view(d):
    pj=os.path.join(d,"per_view.json")
    if not os.path.exists(pj): return {}
    j=json.load(open(pj)); k=list(j)[-1]; return j[k].get("PSNR",{})

# ---- resolve OURS best per scene ----
def find_ours(scene):
    cands=[]
    for dv in glob.glob(f"{OUT}/develop3_discontinuous*"):
        for cell in glob.glob(os.path.join(dv,scene,"*")):
            if glob.glob(os.path.join(cell,"point_cloud","iteration_*","point_cloud.ply")) and rdir(cell):
                p=metrics(cell)[0]
                if p: cands.append((p,cell))
    for d in glob.glob(f"{OUT}/diag/mipnerf360/{scene}_discontinuous_*_i*"):
        if glob.glob(os.path.join(d,"point_cloud","iteration_*","point_cloud.ply")) and rdir(d):
            p=metrics(d)[0]
            if p: cands.append((p,d))
    return max(cands)[1] if cands else None

if __name__=="__main__":
    os.makedirs(DEST,exist_ok=True)
    rows=[]
    for scene in SCENES:
        od=find_ours(scene)
        if not od: print(f"{scene}: no ours"); continue
        op,os_,ol=metrics(od); ord_=rdir(od)
        bcells={m:os.path.join(BR,scene,sd) for m,sd in BASE_METHODS.items() if os.path.isdir(os.path.join(BR,scene,sd))}
        bmet={m:metrics(c) for m,c in bcells.items()}
        bmet={m:v for m,v in bmet.items() if v[0] is not None}
        if not bmet: print(f"{scene}: no baselines"); continue
        bestm,bestv=max(bmet.items(),key=lambda t:t[1][0]); margin=op-bestv[0]
        fieldm=op-np.mean([v[0] for v in bmet.values()])
        rows.append((scene,op,bestm,bestv[0],margin,fieldm,{m:v[0] for m,v in bmet.items()}))
        # assembly
        gtd=gdir(bcells[bestm]);
        methods=[m for m in ORDER if m=="GT" or m in bcells or m=="Ours"]
        dirs={"Ours":ord_}; [dirs.__setitem__(m,rdir(c)) for m,c in bcells.items()]
        sets=[set(os.path.basename(p) for p in glob.glob(os.path.join(dirs[m],"*.png"))) for m in dirs if dirs[m]]
        common=sorted(set.intersection(*sets)) if sets else []
        if not common: continue
        pv={m:per_view(c) for m,c in (list(bcells.items())+[("Ours",od)])}
        def adv(v):
            o=pv.get("Ours",{}).get(v); bs=[pv.get(m,{}).get(v) for m in bcells]; bs=[x for x in bs if x is not None]
            return (o-max(bs)) if (o is not None and bs) else -999
        hero=[v for _,v in sorted(((adv(v),v) for v in common),reverse=True)[:K_HERO]]
        cd=os.path.join(DEST,"cells",scene)
        for s in ("renders","concat_full","errmap"): os.makedirs(os.path.join(cd,s),exist_ok=True)
        for m in methods:
            md=os.path.join(cd,"renders",m); os.makedirs(md,exist_ok=True)
            src=gtd if m=="GT" else dirs.get(m)
            if not src: continue
            for v in common:
                if os.path.exists(os.path.join(src,v)): shutil.copy(os.path.join(src,v),os.path.join(md,v))
        ref=Image.open(os.path.join(gtd,common[0])).convert("RGB").size
        for v in hero:
            gt=Image.open(os.path.join(gtd,v)).convert("RGB")
            ims={m:(gt if m=="GT" else (Image.open(os.path.join(dirs[m],v)).convert("RGB").resize(ref) if dirs.get(m) and os.path.exists(os.path.join(dirs[m],v)) else None)) for m in methods}
            if any(ims[m] is None for m in methods): continue
            panels=[label(ims[m],NAME[m],(f"{pv.get(m,{}).get(v):.1f}dB" if pv.get(m,{}).get(v) and m!="GT" else None),good=(m=="Ours")) for m in methods]
            hstrip(panels).save(os.path.join(cd,"concat_full",v))
            errs={m:np.abs(np.asarray(ims[m].convert("RGB"),float)/255-np.asarray(gt.resize(ref),float)/255).mean(2) for m in methods if m!="GT"}
            vmax=float(np.percentile(np.stack(list(errs.values())),99))
            ep=[label(gt,"Ground Truth")]+[label(err_heat(ims[m],gt,vmax),NAME[m],good=(m=="Ours")) for m in methods if m!="GT"]
            hstrip(ep).save(os.path.join(cd,"errmap",v))
        print(f"{scene}: ours {op:.2f} vs {bestm} {bestv[0]:.2f} (+{margin:.2f} best / +{fieldm:.2f} field); {len(hero)} hero -> {cd}")
    # standings
    rows.sort(key=lambda r:-r[5])
    print("\n=== DISCONTINUOUS standings (sorted by field margin) ===")
    print(f"{'scene':>10} | {'OURS':>6} | {'best baseline':>16} {'+best':>6} {'+field':>6}")
    with open(os.path.join(DEST,"standings.csv"),"w") as fh:
        w=csv.writer(fh); w.writerow(["scene","ours","best_baseline","best_psnr","margin_best","margin_field"])
        for sc,op,bm,bv,mg,fm,allb in rows:
            print(f"{sc:>10} | {op:6.2f} | {bm:>16} {mg:+6.2f} {fm:+6.2f}")
            w.writerow([sc,f"{op:.3f}",bm,f"{bv:.3f}",f"{mg:.3f}",f"{fm:.3f}"])
    print(f"\n-> {DEST}")
