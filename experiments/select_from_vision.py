#!/usr/bin/env python3
"""Build figure specs from vision-confirmed crops.
Inputs: confirmed_<stage>.json (from curate_vision workflow: [{id,scene,angle,readable,obvious_win,
score,caption,artifact,...}]) and cand_<stage>.json (id->{cell,view,box,manifest_key,...}).
coverage_gap: 5 figs x 10 rows, scene-diverse + large-angle-first. discontinuous: 3 figs x 10 rows,
scene-diverse, with user-locked good rows force-included. Pads from the scout pool (diff3) if a
scene is thin, so every figure still gets 10 readable rows. Writes fig_<stage>_<i>.json.
Usage: select_from_vision.py <stage> <n_figs> <rows_per_fig>
"""
import json, os, sys
from collections import defaultdict
VM = "/data1/zhangao/outputs/viz_multiangle"
stage = sys.argv[1]; NF = int(sys.argv[2]); RPF = int(sys.argv[3])

idx = json.load(open(f"{VM}/cand_{stage}.json"))
conf = json.load(open(f"{VM}/confirmed_{stage}.json"))
diff3 = json.load(open(f"{VM}/diff3_{stage}.json"))   # fallback pool (rank-sorted)

# user-approved good discontinuous rows -> force include with their original boxes, one per figure
LOCKS = {"discontinuous": [("counter", "00019.png", [62, 0, 196, 134]),
                           ("garden", "00001.png", [950, 0, 1168, 218]),
                           ("bicycle", "00013.png", [294, 0, 507, 213])]}

def gapw(angle, base):
    return base * {"100": 1.30, "80": 1.18, "60": 1.07}.get(str(angle), 1.0)

def mkpick(meta, vis=None):
    label = (f"{meta['angle']}°" if str(meta.get('angle')) not in ('disc', 'None') else "disc")
    if vis:
        label += f" · win {vis['obvious_win']}/3"
    return dict(cell=meta["cell"], manifest_key=meta.get("manifest_key"), best_view=meta["view"],
                scene=meta["scene"], angle=str(meta.get("angle")), box=meta["box"], label=label,
                win_description=(vis or {}).get("caption", ""),
                artifact=(vis or {}).get("artifact", ""),
                obvious_win=(vis or {}).get("obvious_win"))

# 1) confirmed crops grouped by scene, ranked (vision score, then large-angle, then scout score)
sc_by_id = {c["id"]: c for c in conf}
rows = []
for c in conf:
    meta = idx.get(c["id"])
    if not meta: continue
    rows.append(dict(meta=meta, vis=c,
                     rank=gapw(meta.get("angle"), 1.0) * (c["score"] + 0.01 * meta.get("score", 0))))
by_scene = defaultdict(list)
for r in rows: by_scene[r["meta"]["scene"]].append(r)
for s in by_scene: by_scene[s].sort(key=lambda r: -r["rank"])

# 2) fallback pool from diff3 (already non-flat & ours-wins), keyed by scene, excluding chosen
def diff3_meta(c):
    return dict(cell=c["cell"], manifest_key=c.get("manifest_key"), view=c["view"],
                scene=c["scene"], angle=c["angle"], box=c["box"], score=c.get("score", 0))
fb_by_scene = defaultdict(list)
for c in diff3:
    fb_by_scene[c["scene"]].append(diff3_meta(c))

scenes = sorted(by_scene, key=lambda s: -(by_scene[s][0]["rank"] if by_scene[s] else 0))
all_scenes = sorted(set(list(by_scene) + list(fb_by_scene)))

used = set()   # (cell, view)
def take(meta, vis, fig):
    k = (meta["cell"], meta["view"])
    if k in used: return False
    used.add(k); fig.append(mkpick(meta, vis)); return True

figs = [[] for _ in range(NF)]

# 2a) force-locked rows (discontinuous) with their user-approved boxes, one per figure in order
for li, (cell, view, box) in enumerate(LOCKS.get(stage, [])):
    if li >= NF: break
    meta = dict(cell=cell, manifest_key=None, view=view, scene=cell, angle="disc", box=box, score=0)
    vis = dict(caption="user-selected exemplar", artifact="", obvious_win=3, readable=3)
    take(meta, vis, figs[li])

# 3) scene-diverse fill: figure f gets the f-th best distinct (scene) confirmed crop
order = sorted(all_scenes, key=lambda s: -(by_scene[s][0]["rank"] if by_scene.get(s) else 0))
for f in range(NF):
    for s in order:
        if any(p["scene"] == s for p in figs[f]):  # scene already in this figure (e.g. locked)
            continue
        pool = by_scene.get(s, [])
        placed = False
        # pick the best not-yet-used confirmed crop of this scene, prefer distinct angle across figs
        chosen_angles = set(p["angle"] for fg in figs for p in fg if p["scene"] == s)
        for r in pool:
            if (r["meta"]["cell"], r["meta"]["view"]) in used: continue
            if stage == "coverage_gap" and str(r["meta"]["angle"]) in chosen_angles and len(pool) > NF:
                continue
            if take(r["meta"], r["vis"], figs[f]): placed = True; break
        if not placed:  # fallback to scout pool for this scene
            for m in fb_by_scene.get(s, []):
                if (m["cell"], m["view"]) in used: continue
                if take(m, None, figs[f]): break

# 4) pad every figure to RPF using confirmed extras first, then scout pool (large-angle weighted)
extras = sorted(rows, key=lambda r: -gapw(r["meta"].get("angle"), r["rank"]))
for f in range(NF):
    if len(figs[f]) >= RPF: continue
    for r in extras:
        if len(figs[f]) >= RPF: break
        if (r["meta"]["cell"], r["meta"]["view"]) in used: continue
        take(r["meta"], r["vis"], figs[f])
    if len(figs[f]) < RPF:
        flat_fb = []
        for s in all_scenes: flat_fb += fb_by_scene.get(s, [])
        flat_fb.sort(key=lambda m: -gapw(m.get("angle"), m.get("score", 0)))
        for m in flat_fb:
            if len(figs[f]) >= RPF: break
            if (m["cell"], m["view"]) in used: continue
            take(m, None, figs[f])

for i, fig in enumerate(figs):
    fig = sorted(fig, key=lambda p: -gapw(p["angle"], (p.get("obvious_win") or 0) + 1))
    json.dump(dict(stage=stage, picks=fig, caption=""), open(f"{VM}/fig_{stage}_{i+1}.json", "w"), indent=1)
    desc = ", ".join(f"{p['scene']}@{p['angle']}" + (f"(w{p['obvious_win']})" if p.get('obvious_win') else "")
                     for p in fig)
    print(f"fig {i+1} ({len(fig)} rows): {desc}")
print(f"wrote {NF} fig specs for {stage}; confirmed pool={len(conf)}, used={len(used)}")
