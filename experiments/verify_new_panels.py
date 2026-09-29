#!/usr/bin/env python3
"""Build verification strips for the NEWLY-added method panels: per figure spec, one row per pick
showing  GT-full(box) | GT-crop | <NEW>-full | <NEW>-crop , so each new panel can be checked for
correct viewpoint (matches GT) and real (not collapsed/black). NEW = 3DGS for coverage_gap, PGSR for
discontinuous. Writes viz_multiangle/VERIFY/<TAG>_<n>.png. Usage: verify_new_panels.py"""
import json, os, glob, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compose_figure import load
VM = "/data1/zhangao/outputs/viz_multiangle"
OUT = f"{VM}/VERIFY"; os.makedirs(OUT, exist_ok=True)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
def fnt(s):
    try: return ImageFont.truetype(FONT, s)
    except: return ImageFont.load_default()
NEW = {"coverage_gap": "3DGS", "discontinuous": "PGSR"}

def panel(im, label, sub=None, sz=(300, 225)):
    if im is None: im = Image.new("RGB", sz, (50, 50, 50))
    im = im.convert("RGB").resize(sz); bar = 26
    cv = Image.new("RGB", (sz[0], sz[1] + bar), (15, 15, 15)); cv.paste(im, (0, bar))
    dr = ImageDraw.Draw(cv); dr.text((4, 4), label, fill=(235, 235, 235), font=fnt(15))
    if sub:
        tw = dr.textlength(sub, font=fnt(13)); dr.text((sz[0]-tw-4, 6), sub, fill=(255, 215, 110), font=fnt(13))
    return cv

def mse(a, b):
    a = np.asarray(a.convert("RGB").resize((128, 128)), float)
    b = np.asarray(b.convert("RGB").resize((128, 128)), float)
    return float(np.mean((a-b)**2))

for stage, tag in [("coverage_gap", "CG"), ("discontinuous", "DISC")]:
    nm = NEW[stage]
    for spec in sorted(glob.glob(f"{VM}/fig_{stage}_*.json")):
        n = spec.rsplit("_", 1)[1].split(".")[0]
        picks = json.load(open(spec))["picks"]
        rows = []
        for p in picks:
            cell, v, box = p["cell"], p["best_view"], tuple(p.get("box") or (0, 0, 0, 0))
            gt = load(stage, cell, "GT", v); new = load(stage, cell, nm, v)
            if gt is None: continue
            if new is not None and new.size != gt.size: new = new.resize(gt.size)
            def crop(im): return im.crop(box) if (im and box[2] > box[0]) else im
            mu = (np.asarray(new.convert("RGB"), float).mean() if new is not None else -1)
            cells = [panel(gt, f"{p['scene']}@{p.get('angle')}  GT"),
                     panel(crop(gt), "GT crop"),
                     panel(new, nm + (" full" if new is not None else " MISSING"),
                           sub=(f"mean{mu:.0f}" if new is not None else None)),
                     panel(crop(new), nm + " crop")]
            w = sum(c.width for c in cells) + 18; h = cells[0].height
            row = Image.new("RGB", (w, h), (15, 15, 15)); x = 0
            for c in cells: row.paste(c, (x, 0)); x += c.width + 6
            rows.append(row)
        if not rows: continue
        W = max(r.width for r in rows); H = sum(r.height for r in rows) + 6*len(rows) + 28
        fig = Image.new("RGB", (W, H), (15, 15, 15)); dr = ImageDraw.Draw(fig)
        dr.text((6, 6), f"VERIFY {tag}_{n}: new method = {nm}  (each row: GT | GT-crop | {nm} | {nm}-crop)",
                fill=(255, 255, 255), font=fnt(17))
        y = 28
        for r in rows: fig.paste(r, (0, y)); y += r.height + 6
        outp = f"{OUT}/{tag}_{n}.png"; fig.save(outp); print("wrote", outp, fig.size)
print("VERIFY STRIPS DONE")
