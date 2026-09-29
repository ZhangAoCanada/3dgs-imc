#!/usr/bin/env python3
"""Export every figure's images SEPARATELY: per row (scene/view), the full original image and the
labeled crop for each method, plus the GT (with the crop box marked). Also exports each BEV panel
(per scene/method/angle) separately. Organized under viz_multiangle/SEPARATE/.
Usage: export_separate.py            (does coverage_gap, discontinuous, and BEV)
"""
import json, os, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compose_figure import load, NAME, METHODS

VM = "/data1/zhangao/outputs/viz_multiangle"
OUT = f"{VM}/SEPARATE"
GA = f"{VM}/global_all"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
def fnt(s):
    try: return ImageFont.truetype(FONT, s)
    except: return ImageFont.load_default()

def labeled(img, text, accent=False, sub=None):
    W, H = img.size; bar = 30
    cv = Image.new("RGB", (W, H + bar), (15, 15, 15)); cv.paste(img, (0, bar))
    dr = ImageDraw.Draw(cv)
    dr.text((6, 5), text, fill=((120, 255, 120) if accent else (240, 240, 240)), font=fnt(20))
    if sub:
        tw = dr.textlength(sub, font=fnt(16)); dr.text((W-tw-6, 7), sub, fill=(255, 220, 120), font=fnt(16))
    return cv

def safe(s): return str(s).replace("/", "-")

def export_stage(stage, nf):
    figdirs = []
    for i in range(1, nf+1):
        spec = json.load(open(f"{VM}/fig_{stage}_{i}.json"))
        tag = "FIG_CG" if stage == "coverage_gap" else "FIG_DISC"
        for r, p in enumerate(spec["picks"], 1):
            cell, view, box = p["cell"], p["best_view"], tuple(p["box"])
            ang = p.get("angle"); scene = p["scene"]
            vstem = view.replace(".png", "")
            rowdir = f"{OUT}/{stage}/{tag}_{i}/row{r:02d}_{safe(scene)}_{ang}_{vstem}"
            os.makedirs(rowdir, exist_ok=True)
            gt = load(stage, cell, "GT", view)
            if gt is None: continue
            ref = gt.size
            # GT full with red box + GT crop
            full = gt.copy(); ImageDraw.Draw(full).rectangle(box, outline=(255, 40, 40), width=max(3, ref[0]//200))
            full.save(f"{rowdir}/GT_full_boxed.png")
            gt.save(f"{rowdir}/GT_full.png")
            gt.crop(box).save(f"{rowdir}/GT_crop.png")
            labeled(gt.crop(box), "Ground Truth").save(f"{rowdir}/GT_crop_labeled.png")
            for m in METHODS[stage]:
                if m == "GT": continue
                im = load(stage, cell, m, view)
                if im is None: continue
                if im.size != ref: im = im.resize(ref)
                im.save(f"{rowdir}/{safe(NAME[m])}_full.png")
                cr = im.crop(box)
                cr.save(f"{rowdir}/{safe(NAME[m])}_crop.png")
                labeled(cr, NAME[m], accent=(m == "Ours")).save(f"{rowdir}/{safe(NAME[m])}_crop_labeled.png")
            # caption
            with open(f"{rowdir}/caption.txt", "w") as fh:
                fh.write(f"{scene}@{ang}  view={view}  box={list(box)}\n{p.get('label','')}\n{p.get('win_description','')}\n")
            figdirs.append(rowdir)
    print(f"{stage}: exported {len(figdirs)} rows under {OUT}/{stage}/")

# ---- BEV panels ----
from plyfile import PlyData
import matplotlib; matplotlib.use("Agg")
import matplotlib.cm as cm
from main_crop import proj_from_ref, main_box_proj, px_box
from scipy.ndimage import zoom as _zoom, gaussian_filter as _gf
_BEVMARGIN = float(os.environ.get("BEV_MARGIN", "2.8")); _BEVMASS = float(os.environ.get("BEV_MASS", "0.88"))
def _smooth(a, target=360):
    # gentle cubic upsample (the wide crop already keeps many native bins) — avoids the old heavy blur
    if a.size == 0 or min(a.shape) < 2: return a
    f = max(1.0, target/max(a.shape)); return _gf(_zoom(a.astype(float), f, order=3), sigma=max(0.6, f*0.35))
def bev_crop_box(scene, bins=320):
    """main-region pixel crop (i0,i1,j0,j1) for this scene's BEV histograms."""
    refp = sorted(glob.glob(f"/data1/zhangao/outputs/baselines_multi_angle/{scene}/*/input.ply"))
    if not refp: return None
    ref = np.stack([PlyData.read(refp[0])["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float32)
    center, axes = proj_from_ref(ref); r2 = (ref-center)@axes.T
    pad = 0.35; mn = r2.min(0)-pad*(r2.max(0)-r2.min(0)); mx = r2.max(0)+pad*(r2.max(0)-r2.min(0))
    cx, cy, half = main_box_proj(ref, center, axes, mass_frac=_BEVMASS, margin=_BEVMARGIN)
    return px_box(mn, mx, bins, cx, cy, half)
def bev_refH(scene, bins=320):
    refp = sorted(glob.glob(f"/data1/zhangao/outputs/baselines_multi_angle/{scene}/*/input.ply"))
    if not refp: return None
    ref = np.stack([PlyData.read(refp[0])["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float32)
    lo, hi = np.percentile(ref, 1, 0), np.percentile(ref, 99, 0)
    center = ref.mean(0); order = np.argsort(-(hi-lo))[:2]
    axes = np.zeros((2, 3), np.float32)
    for i, a in enumerate(order): axes[i, a] = 1.0
    r2 = (ref - center) @ axes.T
    pad = 0.35; mn = r2.min(0)-pad*(r2.max(0)-r2.min(0)); mx = r2.max(0)+pad*(r2.max(0)-r2.min(0))
    H, _, _ = np.histogram2d(r2[:, 0], r2[:, 1], bins=bins, range=[[mn[0], mx[0]], [mn[1], mx[1]]])
    return (H.T > 0).astype(float)

def export_bev(scenes):
    ORDER = ["3DGS", "Hier", "ScaffoldGS", "MCMC", "CDGS", "PGSR", "MH-3DGS", "MVGS", "Ours"]
    turbo = cm.get_cmap("inferno"); greys = cm.get_cmap("Greys")
    for scene in scenes:
        refH = bev_refH(scene); box = bev_crop_box(scene)
        if box: i0, i1, j0, j1 = box; refH = refH[i0:i1, j0:j1] if refH is not None else None
        refS = _smooth(refH) if refH is not None else None
        angles = [a for a in [20, 40, 60, 80, 100] if os.path.exists(f"{GA}/{scene}__Ours__{a}.npy")]
        sd = f"{OUT}/BEV/{scene}"; os.makedirs(sd, exist_ok=True)
        fr = {}
        for m in ORDER:
            f = f"{GA}/{scene}__{m}.frac.json"
            if os.path.exists(f): fr[m] = json.load(open(f))
        # FRACTION-DENSITY normalization (matches bev_grid_overlay): grid / total-model-mass
        def fracv(m, a): return (fr.get(m, {}).get(str(a)) or fr.get(m, {}).get(str(float(a))))
        def norm(m, a):
            p = f"{GA}/{scene}__{m}__{a}.npy"
            if not os.path.exists(p): return None
            full = np.load(p); fv = fracv(m, a); s = float(full.sum())
            tm = (s/fv) if (fv and s > 0) else None
            return (full/tm) if tm else full*0.0
        allv = [norm(m, a) for m in ORDER for a in angles if norm(m, a) is not None]
        if not allv: continue
        vmax = np.percentile(np.concatenate([g.ravel() for g in allv]), 99.5) + 1e-9
        for m in ORDER:
            for a in angles:
                gn = norm(m, a)
                if gn is None: continue
                g = gn
                if box: g = g[i0:i1, j0:j1]
                g = np.clip(_smooth(np.sqrt(np.clip(g, 0, None)/vmax)), 0, 1)   # shared scale, cropped, smoothed
                rS = refS if refS is not None else np.zeros_like(g)
                hot = turbo(g)[..., :3]                       # inferno: black where empty, hot at floaters
                comp = np.clip(hot + np.clip(rS, 0, 1)[..., None]*np.array([0.22, 0.22, 0.26]), 0, 1)  # +faint gray scene
                img = Image.fromarray((comp[::-1]*255).astype(np.uint8))  # origin lower
                frac = (fr.get(m, {}).get(str(a)) or fr.get(m, {}).get(str(float(a))))
                sub = (f"{frac*100:.0f}% floaters" if frac is not None else None)
                labeled(img.resize((360, 360), Image.BICUBIC), f"{NAME[m]} @{a}°", accent=(m == "Ours"), sub=sub).save(
                    f"{sd}/{safe(NAME[m])}_{a}.png")
        print(f"BEV {scene}: panels for {ORDER} x {angles} -> {sd}/")

if __name__ == "__main__":
    export_stage("coverage_gap", 3)
    export_stage("discontinuous", 2)
    export_bev(["bonsai", "counter", "kitchen"])
    print("DONE -> " + OUT)
