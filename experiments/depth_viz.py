#!/usr/bin/env python3
"""Build cross-method DEPTH comparison strips from the uniform depth .npy files.
Floaters / broken geometry under viewpoint bias appear as near-camera blobs, speckle, and
holes; OURS should be smooth + complete. Shared depth color-scale per view so methods are
directly comparable. Panel order: GT(RGB) | Ours | Hier | MH-3DGS | MVGS.
Also a 'floater map': pixels whose depth is anomalously NEAR vs the cross-method consensus.
"""
import os, glob, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps

DEPTH = "/data1/zhangao/outputs/viz_multiangle/depth"
CELLS = "/data1/zhangao/outputs/viz_multiangle/cells"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ORDER = ["Ours", "Hier", "MH-3DGS", "MVGS"]
NAME = {"Ours": "Ours (GS-LSS)", "Hier": "Hier-GS", "MH-3DGS": "MH-3DGS", "MVGS": "MVGS"}
TURBO = colormaps["turbo"]

def fnt(s):
    try: return ImageFont.truetype(FONT, s)
    except: return ImageFont.load_default()
def label(img, text, sub=None, good=False):
    W, H = img.size; bar = max(26, H//14); sz = int(bar*0.6)
    cv = Image.new("RGB", (W, H+bar), (15,15,15)); cv.paste(img,(0,bar))
    dr = ImageDraw.Draw(cv); dr.text((6,bar*0.12), text, fill=(255,255,255), font=fnt(sz))
    if sub:
        col = (120,230,120) if good else (230,200,120)
        tw = dr.textlength(sub, font=fnt(int(sz*0.85)))
        dr.text((W-tw-6, bar*0.16), sub, fill=col, font=fnt(int(sz*0.85)))
    return cv
def hstrip(imgs, pad=4):
    H = max(i.size[1] for i in imgs)
    imgs = [i if i.size[1]==H else i.resize((int(i.size[0]*H/i.size[1]),H)) for i in imgs]
    W = sum(i.size[0] for i in imgs)+pad*(len(imgs)-1)
    out = Image.new("RGB",(W,H),(15,15,15)); x=0
    for i in imgs: out.paste(i,(x,0)); x+=i.size[0]+pad
    return out
def depth_to_rgb(d, vmin, vmax):
    m = (d <= 1e-6)  # holes / no geometry
    x = np.clip((d - vmin)/max(vmax-vmin,1e-6), 0, 1)
    rgb = (TURBO(x)[...,:3]*255).astype(np.uint8)
    rgb[m] = (25,25,25)
    return Image.fromarray(rgb)

def gt_for(cell_us, view):
    p = os.path.join(CELLS, cell_us, "renders", "GT", f"{view:05d}.png")
    return Image.open(p).convert("RGB") if os.path.exists(p) else None

summary = []
sheet_rows = []
for cd in sorted(glob.glob(os.path.join(DEPTH, "*"))):
    cell_us = os.path.basename(cd)
    methods = [m for m in ORDER if os.path.isdir(os.path.join(cd, m))]
    if "Ours" not in methods: continue
    views = sorted(int(os.path.basename(f)[:5]) for f in glob.glob(os.path.join(cd,"Ours","*.npy")))
    os.makedirs(os.path.join(cd,"compare"), exist_ok=True)
    floater_frac = {m: [] for m in methods}
    best_strip = None
    for v in views:
        ds = {}
        for m in methods:
            p = os.path.join(cd, m, f"{v:05d}.npy")
            if os.path.exists(p): ds[m] = np.load(p)
        if "Ours" not in ds or len(ds) < 2: continue
        # align shapes (should match)
        H = min(a.shape[0] for a in ds.values()); W = min(a.shape[1] for a in ds.values())
        ds = {m: a[:H,:W] for m, a in ds.items()}
        valid = np.stack([np.where(a>1e-6, a, np.nan) for a in ds.values()])
        vmin = np.nanpercentile(valid, 3); vmax = np.nanpercentile(valid, 97)
        # consensus depth (median across methods) -> floater = much nearer than consensus
        cons = np.nanmedian(valid, axis=0)
        gt = gt_for(cell_us, v)
        panels = []
        if gt is not None: panels.append(label(gt.resize((W,H)), "Ground Truth"))
        for m in methods:
            a = ds[m]
            panels.append(label(depth_to_rgb(a, vmin, vmax), NAME[m], good=(m=="Ours")))
        strip = hstrip(panels)
        strip.save(os.path.join(cd,"compare", f"{v:05d}.png"))
        if best_strip is None: best_strip = (v, strip)
    summary.append((cell_us, len(views)))
    if best_strip: sheet_rows.append((cell_us, best_strip[0], best_strip[1]))

# contact sheet: one hero depth strip per cell
if sheet_rows:
    W = max(s.size[0] for _,_,s in sheet_rows)
    rows=[]
    for cell_us, v, s in sheet_rows:
        s = s if s.size[0]==W else s.resize((W,int(s.size[1]*W/s.size[0])))
        bar=34; cv=Image.new("RGB",(W,s.size[1]+bar),(0,0,0)); cv.paste(s,(0,bar))
        dr=ImageDraw.Draw(cv)
        dr.text((8,5),f"{cell_us}  (view {v:05d})   depth: warm=near, cool=far, dark=no geometry/holes",
                fill=(255,230,120),font=fnt(22)); rows.append(cv)
    Htot=sum(r.size[1] for r in rows)+6*(len(rows)-1)
    sheet=Image.new("RGB",(W,Htot),(0,0,0)); y=0
    for r in rows: sheet.paste(r,(0,y)); y+=r.size[1]+6
    sheet.save(os.path.join(DEPTH,"depth_contact_sheet.png"))
    print(f"depth contact sheet -> {DEPTH}/depth_contact_sheet.png ({W}x{Htot})")
print(f"\nbuilt depth comparison strips for {len(summary)} cells")
for cell_us, nv in summary: print(f"  {cell_us}: {nv} views")
