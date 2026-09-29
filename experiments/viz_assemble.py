#!/usr/bin/env python3
"""Assemble qualitative comparison material for the coverage-gap stage.
Reads viz_multiangle/manifest.json (from viz_standings.py). Selects cells where OURS beats
the FIELD (large-gap-weighted), picks hero views by per-view PSNR advantage, and writes:
  cells/<cell>/renders/<Method>/*.png        all method renders (+ GT), view-aligned
  cells/<cell>/concat_full/<view>.png        labeled side-by-side, all methods, per-view PSNR
  cells/<cell>/concat_crop/<view>.png        zoom on the region where OURS wins most
  cells/<cell>/errmap/<view>.png             |render-GT| heatmaps, all methods (theory viz)
  contact_sheet.png                          one hero view per selected cell, stacked
  SELECTION.md                               the shortlist + rationale
Method panel order: GT, Hier, ScaffoldGS, MH-3DGS, MVGS, Ours (ours last).
"""
import json, os, glob, shutil, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm

DEST = "/data1/zhangao/outputs/viz_multiangle"
MAN = json.load(open(os.path.join(DEST, "manifest.json")))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ORDER = ["GT", "Hier", "ScaffoldGS", "MH-3DGS", "MVGS", "Ours"]
NAME = {"GT": "Ground Truth", "Hier": "Hier-GS", "ScaffoldGS": "Scaffold-GS",
        "MH-3DGS": "MH-3DGS", "MVGS": "MVGS", "Ours": "Ours (GS-LSS)"}
K_HERO = int(os.environ.get("K_HERO", "8"))
N_CELLS = int(os.environ.get("N_CELLS", "12"))
MAX_PER_SCENE = 2
ONLY = os.environ.get("ONLY_CELLS", "").split(",") if os.environ.get("ONLY_CELLS") else None

def load_pv(d):
    """per_view.json -> {view: {PSNR,SSIM,LPIPS}}"""
    pj = os.path.join(os.path.dirname(os.path.dirname(d)), "..", "per_view.json")
    # d is .../test/ours_XXXX/renders ; per_view.json is at run root
    root = d
    for _ in range(3): root = os.path.dirname(root)
    pj = os.path.join(root, "per_view.json")
    if not os.path.exists(pj): return {}
    raw = json.load(open(pj)); key = list(raw)[-1]
    out = {}
    psnr = raw[key].get("PSNR", {})
    for v in psnr: out[v] = {"PSNR": psnr[v]}
    return out

def fnt(sz):
    try: return ImageFont.truetype(FONT, sz)
    except: return ImageFont.load_default()

def label(img, text, sub=None):
    """add a label bar on top of an image"""
    W, H = img.size
    bar = max(26, H // 16)
    sz = int(bar * 0.62)
    canvas = Image.new("RGB", (W, H + bar), (15, 15, 15))
    canvas.paste(img, (0, bar))
    dr = ImageDraw.Draw(canvas)
    dr.text((6, bar * 0.12), text, fill=(255, 255, 255), font=fnt(sz))
    if sub:
        tw = dr.textlength(sub, font=fnt(int(sz*0.85)))
        dr.text((W - tw - 6, bar * 0.16), sub, fill=(120, 230, 120), font=fnt(int(sz*0.85)))
    return canvas

def hstrip(imgs, pad=4):
    H = max(i.size[1] for i in imgs)
    imgs = [i if i.size[1] == H else i.resize((int(i.size[0]*H/i.size[1]), H)) for i in imgs]
    W = sum(i.size[0] for i in imgs) + pad*(len(imgs)-1)
    out = Image.new("RGB", (W, H), (15, 15, 15)); x = 0
    for i in imgs:
        out.paste(i, (x, 0)); x += i.size[0] + pad
    return out

def err_heat(render, gt, vmax):
    r = np.asarray(render.convert("RGB"), float)/255
    g = np.asarray(gt.convert("RGB").resize(render.size), float)/255
    e = np.abs(r-g).mean(2)
    e = np.clip(e/max(vmax,1e-6), 0, 1)
    rgb = (cm.get_cmap("turbo")(e)[..., :3]*255).astype(np.uint8)
    return Image.fromarray(rgb)

def err_arr(render, gt):
    r = np.asarray(render.convert("RGB"), float)/255
    g = np.asarray(gt.convert("RGB").resize(render.size), float)/255
    return np.abs(r-g).mean(2)

def find_win_box(adv, frac=0.34):
    """coarse-pool the advantage map, return crop box (l,t,r,b) around max-advantage block."""
    H, W = adv.shape
    gh, gw = 12, 12
    bh, bw = H//gh, W//gw
    if bh == 0 or bw == 0: return None
    best=None
    for i in range(gh):
        for j in range(gw):
            m = adv[i*bh:(i+1)*bh, j*bw:(j+1)*bw].mean()
            if best is None or m>best[0]: best=(m, i, j)
    _, i, j = best
    cy, cx = (i+0.5)*bh, (j+0.5)*bw
    s = int(min(H, W)*frac)
    l = int(np.clip(cx - s/2, 0, W-s)); t = int(np.clip(cy - s/2, 0, H-s))
    return (l, t, l+s, t+s)

def open_view(d, v, ref_size=None):
    p = os.path.join(d, v)
    if not os.path.exists(p): return None
    im = Image.open(p).convert("RGB")
    if ref_size and im.size != ref_size: im = im.resize(ref_size)
    return im

# ---------- cell selection ----------
def field_stats(cell):
    info = MAN[cell]
    o = info["ours"]; comp = info["base"]
    bp = [float(v["psnr"]) for v in comp.values() if v.get("psnr") is not None]
    if not bp: return None
    op = float(o["psnr"])
    angle = float(info["angle"])
    return dict(cell=cell, scene=info["scene"], angle=angle, ours=op,
                margin_best=op-max(bp), margin_field=op-(sum(bp)/len(bp)),
                n_beat1=sum(1 for x in bp if op-x>=1.0), nbase=len(bp))

def select():
    stats=[s for s in (field_stats(c) for c in MAN) if s and s["margin_best"]>-0.05]
    def sc(s):
        gap={100.0:1.0,80.0:0.7,60.0:0.35}.get(s["angle"],0.0)
        return 0.55*s["margin_best"]+0.45*s["margin_field"]+0.5*gap+0.25*s["n_beat1"]
    stats.sort(key=sc, reverse=True)
    chosen=[]; per_scene={}
    for s in stats:
        if per_scene.get(s["scene"],0)>=MAX_PER_SCENE: continue
        chosen.append(s); per_scene[s["scene"]]=per_scene.get(s["scene"],0)+1
        if len(chosen)>=N_CELLS: break
    # guarantee large-gap presence
    return chosen

# ---------- per-cell assembly ----------
def assemble(cell):
    info = MAN[cell]
    o = info["ours"]; comp = info["base"]
    dirs = {"Ours": o["rdir"]}
    for m, v in comp.items():
        if v.get("rdir"): dirs[m] = v["rdir"]
    gtd = None
    for m in ("MH-3DGS","MVGS","Hier","ScaffoldGS"):
        if m in comp and comp[m].get("gtdir"): gtd = comp[m]["gtdir"]; break
    if gtd is None: gtd = o.get("gtdir")
    methods = [m for m in ORDER if m=="GT" or m in dirs]
    # common views across all present sources
    sets=[set(os.path.basename(p) for p in glob.glob(os.path.join(dirs[m],"*.png"))) for m in methods if m!="GT"]
    common = sorted(set.intersection(*sets)) if sets else []
    if not common: return None
    # per-view advantage (ours - best baseline) from per_view.json
    pv = {m: load_pv(dirs[m]) for m in dirs}
    def adv_psnr(v):
        op = pv.get("Ours",{}).get(v,{}).get("PSNR")
        bps = [pv.get(m,{}).get(v,{}).get("PSNR") for m in comp]
        bps = [x for x in bps if x is not None]
        if op is None or not bps: return None
        return op - max(bps)
    scored = [(adv_psnr(v) if adv_psnr(v) is not None else -999, v) for v in common]
    scored.sort(reverse=True)
    hero = [v for _, v in scored[:K_HERO]]

    cd = os.path.join(DEST, "cells", cell.replace("@","_"))
    for sub in ("renders","concat_full","concat_crop","errmap"):
        os.makedirs(os.path.join(cd, sub), exist_ok=True)
    # copy ALL renders (per method) + GT
    for m in methods:
        md = os.path.join(cd,"renders", m); os.makedirs(md, exist_ok=True)
        src = gtd if m=="GT" else dirs[m]
        for v in common:
            sp=os.path.join(src,v)
            if os.path.exists(sp): shutil.copy(sp, os.path.join(md,v))
    # ref size from GT
    ref = Image.open(os.path.join(gtd, common[0])).convert("RGB").size
    # hero strips
    hero_full={}
    for v in hero:
        gt = open_view(gtd, v, ref)
        imgs={"GT":gt}
        for m in methods:
            if m=="GT": continue
            imgs[m]=open_view(dirs[m], v, ref)
        if any(imgs[m] is None for m in methods): continue
        # full labeled strip
        panels=[]
        for m in methods:
            pp = pv.get(m,{}).get(v,{}).get("PSNR")
            sub = (f"{pp:.2f} dB" if pp is not None else None) if m!="GT" else None
            panels.append(label(imgs[m], NAME[m], sub))
        strip = hstrip(panels)
        strip.save(os.path.join(cd,"concat_full", v))
        hero_full[v]=strip
        # error heatmaps (shared vmax across methods)
        errs={m: err_arr(imgs[m], gt) for m in methods if m!="GT"}
        vmax=float(np.percentile(np.stack(list(errs.values())), 99))
        epanels=[label(gt,"Ground Truth")]
        for m in methods:
            if m=="GT": continue
            epanels.append(label(err_heat(imgs[m], gt, vmax), NAME[m]))
        hstrip(epanels).save(os.path.join(cd,"errmap", v))
        # crop on win region: adv = best_baseline_err - ours_err
        be = np.min(np.stack([errs[m] for m in errs if m!="Ours"]),0) if len(errs)>1 else None
        if be is not None and "Ours" in errs:
            advmap = be - errs["Ours"]
            box = find_win_box(advmap)
            if box:
                cps=[]
                for m in methods:
                    cim = imgs[m].crop(box).resize((360,360), Image.NEAREST)
                    pp = pv.get(m,{}).get(v,{}).get("PSNR")
                    cps.append(label(cim, NAME[m], (f"{pp:.2f} dB" if pp is not None and m!="GT" else None)))
                hstrip(cps).save(os.path.join(cd,"concat_crop", v))
    # write hero list
    with open(os.path.join(cd,"hero.txt"),"w") as fh:
        fh.write(f"cell={cell}  ours_tag={o.get('tag')}\n")
        for a,v in scored[:K_HERO]:
            fh.write(f"  {v}  per-view adv = {a:+.2f} dB\n")
    return dict(cell=cell, cd=cd, hero=hero, hero_full=hero_full, methods=methods,
                best_view=(hero[0] if hero else None))

if __name__=="__main__":
    if ONLY:
        chosen=[field_stats(c) for c in ONLY if c in MAN]
    else:
        chosen=select()
    print("=== SELECTED CELLS ===")
    for s in chosen:
        print(f"  {s['cell']:>14}  ours={s['ours']:.2f}  +{s['margin_best']:.2f}(best) "
              f"+{s['margin_field']:.2f}(field)  beats>=1dB:{s['n_beat1']}/{s['nbase']}")
    results=[]
    for s in chosen:
        r=assemble(s["cell"])
        if r: results.append((s,r)); print(f"  done {s['cell']}: {len(r['hero'])} hero views -> {r['cd']}")
    # contact sheet: best hero view per cell (full strip), stacked
    strips=[]
    for s,r in results:
        if r["best_view"] and r["best_view"] in r["hero_full"]:
            st=r["hero_full"][r["best_view"]]
            # title bar with cell name
            W,_=st.size; bar=34
            cv=Image.new("RGB",(W,st.size[1]+bar),(0,0,0))
            cv.paste(st,(0,bar)); dr=ImageDraw.Draw(cv)
            dr.text((8,4),f"{s['cell']}  (ours +{s['margin_best']:.2f} dB vs best baseline; +{s['margin_field']:.2f} vs field)",
                    fill=(255,230,120),font=fnt(22))
            strips.append(cv)
    if strips:
        W=max(s.size[0] for s in strips)
        strips=[s if s.size[0]==W else s.resize((W,int(s.size[1]*W/s.size[0]))) for s in strips]
        Htot=sum(s.size[1] for s in strips)+6*(len(strips)-1)
        sheet=Image.new("RGB",(W,Htot),(0,0,0)); y=0
        for s in strips: sheet.paste(s,(0,y)); y+=s.size[1]+6
        sheet.save(os.path.join(DEST,"contact_sheet.png"))
        print(f"\ncontact sheet -> {DEST}/contact_sheet.png  ({W}x{Htot})")
    # SELECTION.md
    with open(os.path.join(DEST,"SELECTION.md"),"w") as fh:
        fh.write("# Coverage-gap qualitative selection\n\n")
        fh.write("Panel order: GT | Hier-GS | Scaffold-GS | MH-3DGS | MVGS | Ours.\n\n")
        fh.write("| cell | ours PSNR | +best | +field | beats>=1dB | hero views |\n|---|---|---|---|---|---|\n")
        for s,r in results:
            fh.write(f"| {s['cell']} | {s['ours']:.2f} | +{s['margin_best']:.2f} | +{s['margin_field']:.2f} | {s['n_beat1']}/{s['nbase']} | {len(r['hero'])} |\n")
    print(f"selection -> {DEST}/SELECTION.md")
