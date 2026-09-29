#!/usr/bin/env python3
"""Compose a publication multi-scene qualitative figure from curated picks.
Input JSON: {"stage": "...", "picks": [{cell,best_view,scene,angle,ours_psnr,best_baseline,
margin_best,win_description}], "caption": "..."}  (one file per stage).
Layout: rows = scenes; col 0 = full GT (small) with red box on the crop region + scene label;
cols 1..M = per-method CROP of the win region, labeled with method + per-view PSNR; Ours gets a
green border. Column headers on top, caption at bottom. Also saves a full-frame variant.
Usage: compose_figure.py <picks.json> <out.png>
"""
import os, sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELLS = {"coverage_gap": "/data1/zhangao/outputs/viz_multiangle/cells",
         "discontinuous": "/data1/zhangao/outputs/viz_discontinuous/cells"}
METHODS = {"coverage_gap": ["GT","3DGS","Hier","MCMC","CDGS","PGSR","ScaffoldGS","MH-3DGS","MVGS","Ours"],
           "discontinuous": ["GT","3DGS","Hier","MCMC","CDGS","PGSR","ScaffoldGS","MH-3DGS","MVGS","Ours"]}
NAME = {"GT":"Ground Truth","3DGS":"3DGS","Hier":"Hier-GS","ScaffoldGS":"Scaffold-GS",
        "MCMC":"3DGS-MCMC","CDGS":"CDGS","PGSR":"PGSR","MH-3DGS":"MH-3DGS","MVGS":"MVGS","Ours":"Ours (GS-LSS)"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CROP = 300          # output crop px
FULLW = 300         # full-GT thumbnail width
def fnt(s):
    try: return ImageFont.truetype(FONT, s)
    except: return ImageFont.load_default()

def psnr(a, b):
    a = np.asarray(a, float)/255; b = np.asarray(b.resize(a.shape[1::-1]) if b.size!=a.shape[1::-1] else b, float)/255
    m = np.mean((a-b)**2); return 99.0 if m<=0 else -10*np.log10(m)

GRID = {"top-left":(0,0),"top-center":(1,0),"top-right":(2,0),"upper":(1,0),
        "center-left":(0,1),"center":(1,1),"center-right":(2,1),"middle":(1,1),
        "bottom-left":(0,2),"bottom-center":(1,2),"bottom-right":(2,2),"lower":(1,2)}
def hint_cell(desc):
    """map a free-text win description to a coarse 3x3 grid cell (col,row in 0..2), or None."""
    d = (desc or "").lower()
    col = 1; row = 1; hit = False
    if "left" in d: col = 0; hit = True
    elif "right" in d: col = 2; hit = True
    if any(k in d for k in ("top","upper","sky","ceiling","background","wall")): row = 0; hit = True
    elif any(k in d for k in ("bottom","lower","ground","floor","pavement","path","foreground","forest floor")): row = 2; hit = True
    if "center" in d or "central" in d or "middle" in d:
        if "left" not in d and "right" not in d: col = 1
        hit = True
    return (col, row) if hit else None

def win_box(ours, gt, base_errs, frac=0.34, hint=None):
    """crop region: where Ours beats baselines, biased to bright+textured areas; a text hint
    (3x3 grid cell from the curator's description) softly steers the search to that region."""
    g = np.asarray(gt.convert("L"), float)/255
    oe = np.abs(np.asarray(ours.convert("RGB"),float)/255 - np.asarray(gt.convert("RGB"),float)/255).mean(2)
    be = np.min(np.stack(base_errs), 0) if base_errs else oe*0
    adv = be - oe
    H, W = adv.shape
    gy, gx = np.gradient(g); grad = np.hypot(gx, gy)
    bright = np.clip(g, 0.12, 1.0) ** 0.5          # downweight near-black regions
    score = np.clip(adv, 0, None) * (0.25 + grad) * bright
    if hint is not None:
        cx0 = (hint[0]+0.5)/3*W; cy0 = (hint[1]+0.5)/3*H
        yy, xx = np.mgrid[0:H, 0:W]
        prior = np.exp(-(((xx-cx0)/(0.45*W))**2 + ((yy-cy0)/(0.45*H))**2))
        score = score * (0.35 + prior)             # soft steer toward hinted region
    gh = gw = 16; bh, bw = max(1,H//gh), max(1,W//gw)
    best = None
    for i in range(gh):
        for j in range(gw):
            s = score[i*bh:(i+1)*bh, j*bw:(j+1)*bw].mean()
            if best is None or s > best[0]: best = (s, i, j)
    _, i, j = best
    cy, cx = (i+0.5)*bh, (j+0.5)*bw
    s = int(min(H, W)*frac)
    l = int(np.clip(cx-s/2, 0, W-s)); t = int(np.clip(cy-s/2, 0, H-s))
    return (l, t, l+s, t+s)

def load(stage, cell, meth, view):
    p = os.path.join(CELLS[stage], cell, "renders", meth, view)
    return Image.open(p).convert("RGB") if os.path.exists(p) else None

def border(img, color, w=5):
    out = Image.new("RGB",(img.size[0]+2*w,img.size[1]+2*w),color); out.paste(img,(w,w)); return out

def labeled(img, text, sub=None, accent=False):
    W,H=img.size; bar=34; sz=20
    cv=Image.new("RGB",(W,H+bar),(20,20,20)); cv.paste(img,(0,bar)); dr=ImageDraw.Draw(cv)
    dr.text((6,7),text,fill=((140,255,140) if accent else (255,255,255)),font=fnt(sz))
    if sub:
        tw=dr.textlength(sub,font=fnt(17)); dr.text((W-tw-6,9),sub,fill=(255,225,120),font=fnt(17))
    return cv

def compose(spec, out, mode="crop"):
    stage=spec["stage"]; picks=spec["picks"]
    excl=set(os.environ.get("EXCLUDE_METHODS","").split(","))-{""}
    # keep a method column only if present for >= half the picked scenes (drops all-empty cols, e.g. MCMC)
    methods=[]
    for m in METHODS[stage]:
        if m in excl: continue
        if m in ("GT","Ours"): methods.append(m); continue
        present=sum(1 for p in picks if load(stage,p["cell"],m,p["best_view"]) is not None)
        if present >= max(1, (len(picks)+1)//2): methods.append(m)
    rows=[]
    for p in picks:
        cell=p["cell"]; view=p["best_view"]
        gt=load(stage,cell,"GT",view)
        ims={m:load(stage,cell,m,view) for m in methods}
        if gt is None or ims.get("Ours") is None:
            print("skip (missing)",cell,view); continue
        ref=gt.size
        ims={m:(im.resize(ref) if im and im.size!=ref else im) for m,im in ims.items()}
        base_errs=[np.abs(np.asarray(ims[m],float)/255-np.asarray(gt,float)/255).mean(2)
                   for m in methods if m not in ("GT","Ours") and ims[m] is not None]
        if p.get("box"):
            box=tuple(p["box"])   # scout-provided validated win region
        else:
            box=win_box(ims["Ours"], gt, base_errs, hint=hint_cell(p.get("win_description")))
        ang = (f"@{p['angle']}°" if p.get('angle') and p['angle']!='disc' else "")
        cells=[]
        if mode=="full":
            fw=380
            for m in methods:
                im=ims[m]
                if im is None:
                    cells.append(labeled(Image.new("RGB",(fw,int(ref[1]*fw/ref[0])),(40,40,40)),NAME[m])); continue
                show=im.copy()
                if m=="GT":
                    ImageDraw.Draw(show).rectangle(box,outline=(255,40,40),width=max(3,ref[0]//180))
                th=show.resize((fw,int(ref[1]*fw/ref[0])), Image.LANCZOS)
                ps=psnr(im,gt); sub=(None if m=="GT" else f"{ps:.1f}dB")
                lab=labeled(border(th,(40,200,40) if m=="Ours" else (60,60,60),5 if m=="Ours" else 2),
                            (f"{p['scene']}{ang}  "+NAME[m]) if m==methods[0] else NAME[m], sub, accent=(m=="Ours"))
                cells.append(lab)
        else:
            fg=gt.copy(); ImageDraw.Draw(fg).rectangle(box,outline=(255,40,40),width=max(3,ref[0]//200))
            fg=fg.resize((FULLW,int(ref[1]*FULLW/ref[0])))
            cells.append(labeled(fg, f"{p['scene']}{ang}", p.get("label") or f"+{p.get('margin_best',0):.2f}dB"))
            for m in methods:
                im=ims[m]
                if im is None:
                    cells.append(labeled(Image.new("RGB",(CROP,CROP),(40,40,40)),NAME[m])); continue
                cr=im.crop(box).resize((CROP,CROP), Image.LANCZOS)
                ps=psnr(im.crop(box), gt.crop(box))
                lab=labeled(border(cr,(40,200,40) if m=="Ours" else (60,60,60), 5 if m=="Ours" else 2),
                            NAME[m], (None if m=="GT" else f"{ps:.1f}dB"), accent=(m=="Ours"))
                cells.append(lab)
        # assemble row (align heights)
        Hh=max(c.size[1] for c in cells)
        cells=[c if c.size[1]==Hh else c.resize((int(c.size[0]*Hh/c.size[1]),Hh)) for c in cells]
        rw=sum(c.size[0] for c in cells)+6*(len(cells)-1)
        row=Image.new("RGB",(rw,Hh),(15,15,15)); x=0
        for c in cells: row.paste(c,(x,0)); x+=c.size[0]+6
        rows.append(row)
    if not rows: print("no rows"); return
    W=max(r.size[0] for r in rows)
    rows=[r if r.size[0]==W else r.resize((W,int(r.size[1]*W/r.size[0]))) for r in rows]
    title_h=40; cap=spec.get("caption","")
    # wrap caption
    dr0=ImageDraw.Draw(Image.new("RGB",(10,10))); f=fnt(18); words=cap.split(); lines=[]; cur=""
    for w in words:
        if dr0.textlength((cur+" "+w).strip(),font=f)<W-20: cur=(cur+" "+w).strip()
        else: lines.append(cur); cur=w
    if cur: lines.append(cur)
    cap_h=24*len(lines)+16
    Htot=title_h+sum(r.size[1] for r in rows)+6*(len(rows)-1)+cap_h
    fig=Image.new("RGB",(W,Htot),(15,15,15)); dr=ImageDraw.Draw(fig)
    sub=("red box = detail region; per-panel PSNR (full frame)" if mode=="full"
         else "crop = region where Ours wins (red box on GT); per-panel PSNR on crop")
    dr.text((8,8),f"Qualitative comparison — {stage.replace('_',' ')} setting (held-out biased views; {sub})",
            fill=(255,255,255),font=fnt(20))
    y=title_h
    for r in rows: fig.paste(r,(0,y)); y+=r.size[1]+6
    for i,ln in enumerate(lines): dr.text((10,y+8+i*24),ln,fill=(200,200,200),font=fnt(18))
    fig.save(out); print(f"wrote {out} ({W}x{Htot}), {len(rows)} scenes")

if __name__=="__main__":
    mode = sys.argv[3] if len(sys.argv)>3 else "crop"
    compose(json.load(open(sys.argv[1])), sys.argv[2], mode)
