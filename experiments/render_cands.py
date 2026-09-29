#!/usr/bin/env python3
"""Render candidate comparison strips for vision curation.
Reads diff3_<stage>.json; for each scene keeps up to TOPK varied candidates (NMS on (cell,view)+box),
renders ONE labeled image per candidate: [full GT w/ red box | GT crop | Ours crop | local-baseline
crops]. Saves /tmp/cand_<stage>/<scene>/c###.png and writes cand_<stage>.json (id -> meta) so the
vision pass and the selector share the same ids. Local methods only (sufficient to judge the win;
hust2 methods are pulled later for the final compose).
Usage: render_cands.py <stage> [topk_per_scene]
"""
import json, os, sys, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

STAGE = sys.argv[1]
TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 14
VM = "/data1/zhangao/outputs/viz_multiangle"
VD = "/data1/zhangao/outputs/viz_discontinuous"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# overrides (for the T&T mini-pipeline so it doesn't collide with the mip set)
DIFF3 = os.environ.get("RC_DIFF3", f"diff3_{STAGE}")
MANIFEST = os.environ.get("RC_MANIFEST", f"{VM}/manifest.json")
INDEX_NAME = os.environ.get("RC_INDEX", f"cand_{STAGE}")
IDPFX = os.environ.get("RC_IDPFX", "c")
OUT = os.environ.get("RC_OUT", f"/tmp/cand_{STAGE}")
CR = 340   # crop display size
NAME = {"GT": "GT", "Ours": "Ours", "Hier": "Hier-GS", "ScaffoldGS": "Scaffold",
        "MH-3DGS": "MH-3DGS", "MVGS": "MVGS", "3DGS": "3DGS"}
LOCAL = {"coverage_gap": ["GT", "Ours", "Hier", "ScaffoldGS", "MH-3DGS", "MVGS"],
         "discontinuous": ["GT", "Ours", "3DGS", "Hier", "ScaffoldGS", "MH-3DGS", "MVGS"]}[STAGE]

def fnt(s):
    try: return ImageFont.truetype(FONT, s)
    except: return ImageFont.load_default()

def iou(a, b):
    l = max(a[0], b[0]); t = max(a[1], b[1]); r = min(a[2], b[2]); bo = min(a[3], b[3])
    if r <= l or bo <= t: return 0.0
    inter = (r - l) * (bo - t)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua

def srcdirs(c):
    """return {method: render_dir} + gtdir for this candidate."""
    if STAGE == "coverage_gap":
        man = json.load(open(MANIFEST))
        info = man[c["manifest_key"]]
        d = {"Ours": info["ours"]["rdir"]}
        for m in ("Hier", "ScaffoldGS", "MH-3DGS", "MVGS"):
            if m in info.get("base", {}): d[m] = info["base"][m]["rdir"]
        gtd = None
        for m in ("Hier", "ScaffoldGS", "MH-3DGS", "MVGS"):
            if m in info.get("base", {}) and info["base"][m].get("gtdir"): gtd = info["base"][m]["gtdir"]; break
        d["GT"] = gtd or info["ours"]["gtdir"]
        return d
    else:
        rd = f"{VD}/cells/{c['scene']}/renders"
        return {m: f"{rd}/{m}" for m in LOCAL if os.path.isdir(f"{rd}/{m}")}

def lab(img, text, accent=False):
    W, H = img.size; bar = 30
    cv = Image.new("RGB", (W, H + bar), (18, 18, 18)); cv.paste(img, (0, bar))
    ImageDraw.Draw(cv).text((5, 5), text, fill=((120, 255, 120) if accent else (235, 235, 235)), font=fnt(20))
    return cv

def border(img, color, w=4):
    o = Image.new("RGB", (img.size[0] + 2*w, img.size[1] + 2*w), color); o.paste(img, (w, w)); return o

cands = json.load(open(f"{VM}/{DIFF3}.json"))
# Surface candidates PER CELL (= per (scene,angle) for coverage_gap) so every angle is represented,
# not just the gap-weighted @100. Then cap per scene. This gives the downstream selector the angle
# variety it needs to spread a scene across different figures (fixes cross-figure repetition).
PER_CELL = {"coverage_gap": 6, "discontinuous": 36}[STAGE]
PER_SCENE = {"coverage_gap": 36, "discontinuous": 36}[STAGE]
by_cell = defaultdict(list)
for c in cands:
    by_cell[c["cell"]].append(c)   # cands already rank-sorted
by_scene = defaultdict(list)
for cell in by_cell:
    keptc = []
    for c in by_cell[cell]:
        if any(k["view"] == c["view"] and iou(k["box"], c["box"]) > 0.4 for k in keptc):
            continue
        keptc.append(c)
        if len(keptc) >= PER_CELL:
            break
    by_scene[keptc[0]["scene"]].extend(keptc) if keptc else None

os.makedirs(OUT, exist_ok=True)
index = {}
cid = 0
for scene in sorted(by_scene):
    kept = sorted(by_scene[scene], key=lambda c: -c["score"])[:PER_SCENE]
    sd = os.path.join(OUT, scene); os.makedirs(sd, exist_ok=True)
    dirs = None
    for c in kept:
        if STAGE == "coverage_gap": dirs = srcdirs(c)
        elif dirs is None: dirs = srcdirs(c)
        v = c["view"]; box = tuple(c["box"])
        try:
            gt = Image.open(f"{dirs['GT']}/{v}").convert("RGB"); ref = gt.size
        except Exception:
            continue
        # full GT with red box
        fw = 360; fh = int(ref[1] * fw / ref[0])
        full = gt.copy(); ImageDraw.Draw(full).rectangle(box, outline=(255, 40, 40), width=max(3, ref[0]//180))
        full = full.resize((fw, fh), Image.LANCZOS)
        cells = [lab(full, f"{scene}" + (f"@{c['angle']}" if c['angle'] != 'disc' else " disc"))]
        for m in LOCAL:
            sdp = dirs.get(m)
            if not sdp or not os.path.exists(f"{sdp}/{v}"):
                cells.append(lab(Image.new("RGB", (CR, CR), (40, 40, 40)), NAME[m])); continue
            im = Image.open(f"{sdp}/{v}").convert("RGB")
            if im.size != ref: im = im.resize(ref)
            cr = im.crop(box).resize((CR, CR), Image.LANCZOS)
            cells.append(lab(border(cr, (40, 200, 40) if m == "Ours" else (70, 70, 70), 4 if m == "Ours" else 1),
                             NAME[m], accent=(m == "Ours")))
        Hh = max(x.size[1] for x in cells)
        cells = [x if x.size[1] == Hh else x.resize((int(x.size[0]*Hh/x.size[1]), Hh)) for x in cells]
        W = sum(x.size[0] for x in cells) + 5*(len(cells)-1)
        strip = Image.new("RGB", (W, Hh), (12, 12, 12)); x = 0
        for cc in cells: strip.paste(cc, (x, 0)); x += cc.size[0] + 5
        p = os.path.join(sd, f"{IDPFX}{cid:03d}.png"); strip.save(p)
        index[f"{IDPFX}{cid:03d}"] = dict(id=f"{IDPFX}{cid:03d}", path=p, scene=scene, cell=c["cell"],
                                    angle=c["angle"], view=v, box=list(box),
                                    manifest_key=c.get("manifest_key"), score=c["score"],
                                    win=c["win"], rel_win=c["rel_win"], beats_all_frac=c["beats_all_frac"])
        cid += 1
    print(f"{scene}: {len([k for k in index.values() if k['scene']==scene])} candidate strips")

json.dump(index, open(f"{VM}/{INDEX_NAME}.json", "w"), indent=1)
print(f"wrote {cid} candidate strips -> {OUT} ; index {VM}/{INDEX_NAME}.json")
