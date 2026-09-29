#!/usr/bin/env python3
"""Build figure specs from the graded vision pool, SPREADING crops across figures so no
(scene,angle) clusters in one figure and the figures don't overlap. Each scene's distinct crops
(1 per view, large-angle-first) are distributed round-robin to different figures; within a figure
scenes are diverse (<=2 per scene, distinct angles) and every view is globally unique. User-approved
disc rows are locked one-per-figure. Inputs: /tmp/judge_pool.json, /tmp/confirmed_all.json,
cand_<stage>.json. Usage: select_graded.py <stage> <n_figs> <rows_per_fig>
"""
import json, sys, os
from collections import defaultdict, Counter
VM = "/data1/zhangao/outputs/viz_multiangle"
stage = sys.argv[1]; NF = int(sys.argv[2]); RPF = int(sys.argv[3])
SEL_SCENES = set(os.environ["SEL_SCENES"].split(",")) if os.environ.get("SEL_SCENES") else None
SEL_PINS = os.environ.get("SEL_PINS")   # json list of user-pinned anchor crops

idx = json.load(open(f"{VM}/cand_{stage}.json"))
pool = [r for r in json.load(open("/tmp/judge_pool.json")) if r["stage"] == stage]
conf_list = [c for c in json.load(open("/tmp/confirmed_all.json")) if c["stage"] == stage]
confirmed = {(c["scene"], c["id"]) for c in conf_list}
capmap = {c["id"]: c.get("caption", "") for c in conf_list}

LOCKS = {"discontinuous": [("counter", "00019.png", [62, 0, 196, 134]),
                           ("garden", "00001.png", [950, 0, 1168, 218])]}

def gapw(angle):
    return {"100": 1.30, "80": 1.18, "60": 1.07, "40": 0.95}.get(str(angle), 0.85 if stage == "coverage_gap" else 1.0)

def grade(r, conf):
    return ((100 if conf else 0) + 10 * r.get("obvious_win", 0) + 3 * r.get("readable", 0)
            + (5 if r.get("recommend") else 0)) * gapw(idx[r["id"]]["angle"])

def iou(a, b):
    l = max(a[0], b[0]); t = max(a[1], b[1]); rr = min(a[2], b[2]); bo = min(a[3], b[3])
    if rr <= l or bo <= t: return 0.0
    inter = (rr - l) * (bo - t); return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

# usable items (readable + a win), 1 best crop per (cell,view) -> maximize distinct pictures
best_by_view = {}
for r in pool:
    m = idx.get(r["id"]);
    if not m: continue
    conf = (r["scene"], r["id"]) in confirmed
    if not (r.get("readable", 0) >= 2 and (r.get("obvious_win", 0) >= 1 or conf)): continue
    it = dict(r, cell=m["cell"], view=m["view"], box=m["box"], angle=str(m["angle"]),
              manifest_key=m.get("manifest_key"), confirmed=conf, grade=grade(r, conf),
              caption=(capmap.get(r["id"]) or r.get("caption", "")))
    k = (it["cell"], it["view"])
    if k not in best_by_view or it["grade"] > best_by_view[k]["grade"]:
        best_by_view[k] = it
items = list(best_by_view.values())
if SEL_SCENES:
    items = [it for it in items if it["scene"] in SEL_SCENES]

by_scene = defaultdict(list)
for it in items: by_scene[it["scene"]].append(it)
for s in by_scene: by_scene[s].sort(key=lambda r: -r["grade"])

def mkpick(it, locked=False):
    label = (f"{it['angle']}°" if str(it.get('angle')) not in ('disc', 'None') else "disc")
    if it.get("pinned"): label += " · ★ kept"
    elif locked: label += " · exemplar"
    elif it.get("confirmed"): label += f" · win {it['obvious_win']}/3 ✓"
    elif it.get("obvious_win"): label += f" · win {it['obvious_win']}/3"
    return dict(cell=it["cell"], manifest_key=it.get("manifest_key"), best_view=it["view"],
                scene=it["scene"], angle=str(it.get("angle")), box=it["box"], label=label,
                win_description=it.get("caption", ""), artifact=it.get("artifact", ""),
                obvious_win=it.get("obvious_win"), grade=round(it.get("grade", 0), 1))

figs = [[] for _ in range(NF)]
fig_scene = [Counter() for _ in range(NF)]
fig_cellang = [set() for _ in range(NF)]   # (scene,angle) present per figure
cellang_figs = Counter()                    # how many figures each (scene,angle) appears in
used_view = set()
TARGET = NF * RPF
def place(it, f, locked=False):
    if (it["cell"], it["view"]) in used_view: return False
    figs[f].append(mkpick(it, locked)); fig_scene[f][it["scene"]] += 1
    fig_cellang[f].add((it["scene"], it["angle"])); cellang_figs[(it["scene"], it["angle"])] += 1
    used_view.add((it["cell"], it["view"])); return True

# 1) locked disc rows, one per figure
for li, (cell, view, box) in enumerate(LOCKS.get(stage, [])):
    if li >= NF: break
    place(dict(cell=cell, view=view, box=box, scene=cell, angle="disc", manifest_key=None,
               caption="user-selected exemplar", obvious_win=3, grade=999, confirmed=True), li, locked=True)

# 1b) user-pinned anchor crops -> force include, distributed across figures (your liked rows)
if SEL_PINS:
    for pi, p in enumerate(json.load(open(SEL_PINS))):
        view = p.get("best_view") or p.get("view")
        it = dict(cell=p["cell"], view=view, box=p["box"], scene=p["scene"],
                  angle=str(p.get("angle")), manifest_key=p.get("manifest_key"),
                  caption=p.get("win_description") or p.get("caption", ""),
                  obvious_win=p.get("obvious_win", 3), confirmed=False, pinned=True, grade=5000)
        if (it["cell"], it["view"]) in used_view: continue
        target = None
        for off in range(NF):
            ff = (pi + off) % NF
            if len(figs[ff]) < RPF and fig_scene[ff][it["scene"]] < 2 \
               and (it["scene"], it["angle"]) not in fig_cellang[ff]:
                target = ff; break
        if target is not None: place(it, target)

# 2) balanced cell-based round-robin: each (scene,angle) cell contributes its distinct crops (best
#    first) to DIFFERENT figures, so no cell exceeds MAXCELL figures and every appearance is a
#    distinct view. Rounds r=0,1,2 place 1st/2nd/3rd-best crop of each cell into the emptiest
#    eligible figure (scene<=2 per figure, no (scene,angle) repeat within a figure).
by_cellang = defaultdict(list)
for it in items: by_cellang[(it["scene"], it["angle"])].append(it)
for k in by_cellang: by_cellang[k].sort(key=lambda r: -r["grade"])
cells_sorted = sorted(by_cellang, key=lambda k: -by_cellang[k][0]["grade"])
def eligible(it):
    return [f for f in range(NF) if len(figs[f]) < RPF and fig_scene[f][it["scene"]] < 2
            and (it["scene"], it["angle"]) not in fig_cellang[f]]
MAXCELL = 3
for r in range(MAXCELL):
    for k in cells_sorted:
        if sum(len(f) for f in figs) >= TARGET: break
        lst = by_cellang[k]
        if r >= len(lst): continue
        it = lst[r]
        if (it["cell"], it["view"]) in used_view: continue
        elig = eligible(it)
        if not elig: continue
        place(it, min(elig, key=lambda f: (len(figs[f]), fig_scene[f][it["scene"]], f)))

# 3) pad any short figure: prefer crops whose (scene,angle) appears in the FEWEST figures so far
def pad_key(it):
    return (cellang_figs[(it["scene"], it["angle"])], -it["grade"])
for f in range(NF):
    while len(figs[f]) < RPF:
        progressed = False
        for it in sorted(items, key=pad_key):   # prefer a 2nd distinct VIEW of a scene (<=2/figure)
            if (it["cell"], it["view"]) in used_view: continue
            if fig_scene[f][it["scene"]] >= 2: continue
            # for coverage_gap keep (scene,angle) unique in a figure; disc has one angle so allow it
            if stage != "discontinuous" and (it["scene"], it["angle"]) in fig_cellang[f]: continue
            place(it, f); progressed = True; break
        if not progressed:   # last resort: any unused distinct view
            for it in sorted(items, key=pad_key):
                if (it["cell"], it["view"]) in used_view: continue
                place(it, f); progressed = True; break
        if not progressed: break

for i, fig in enumerate(figs):
    fig = sorted(fig, key=lambda p: -(p.get("grade", 0)))
    json.dump(dict(stage=stage, picks=fig, caption=""), open(f"{VM}/fig_{stage}_{i+1}.json", "w"), indent=1)
    print(f"fig {i+1} ({len(fig)} rows): " +
          ", ".join(f"{p['scene']}@{p['angle']}" + (f"w{p['obvious_win']}" if p.get('obvious_win') is not None else "")
                    for p in fig))
# cross-figure overlap report
allrows = [(p["scene"], p["angle"]) for f in figs for p in f]
sa = Counter(allrows)
print(f"\n{stage}: usable distinct crops={len(items)}; placed={len(used_view)}")
print("  (scene,angle) appearing in >2 figures:", {k: v for k, v in sa.items() if v > 2})
