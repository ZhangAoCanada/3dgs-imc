#!/usr/bin/env python3
"""Select diverse high-difference rows from diff2_<stage>.json and split into N figures.
Picks distinct views per scene (scene-spread so each figure shows many different scenes),
large-angle-weighted for coverage_gap. Writes fig_<stage>_<i>.json (compose-ready).
Usage: select_split.py <stage> <n_figs> <rows_per_fig>
"""
import json, os, sys
from collections import defaultdict
VM="/data1/zhangao/outputs/viz_multiangle"
stage=sys.argv[1]; NF=int(sys.argv[2]); RPF=int(sys.argv[3]); N=NF*RPF
cands=json.load(open(f"{VM}/diff2_{stage}.json"))
# per (scene) ordered best views (distinct view, non-overlapping boxes)
by_scene=defaultdict(list)
for c in cands: by_scene[c["scene"]].append(c)
def overlap(a,b):
    l=max(a[0],b[0]); t=max(a[1],b[1]); r=min(a[2],b[2]); bo=min(a[3],b[3])
    if r<=l or bo<=t: return 0.0
    inter=(r-l)*(bo-t); ua=(a[2]-a[0])*(a[3]-a[1]); return inter/ua
chosen_per_scene={}
for sc,lst in by_scene.items():
    lst=sorted(lst,key=lambda c:-c["score"]); keep=[]
    if stage=="coverage_gap":
        # one best view per (scene,angle) cell -> angle diversity across figures
        seen_cell=set()
        for c in lst:
            if c["cell"] in seen_cell: continue
            seen_cell.add(c["cell"]); keep.append(c)
    else:
        for c in lst:
            if any(k["cell"]==c["cell"] and k["view"]==c["view"] for k in keep): continue
            if any(k["cell"]==c["cell"] and overlap(k["box"],c["box"])>0.5 for k in keep): continue
            keep.append(c)
            if len(keep)>=NF+2: break
    chosen_per_scene[sc]=keep
# Each figure should be SCENE-DIVERSE: figure f gets the f-th best distinct view of EACH scene.
# That gives ~one row per scene per figure (all different scenes), with views varying across figures.
scenes=sorted(chosen_per_scene, key=lambda s:-max(c["score"] for c in chosen_per_scene[s]))
figs=[[] for _ in range(NF)]
for f in range(NF):
    for s in scenes:
        if f < len(chosen_per_scene[s]): figs[f].append(chosen_per_scene[s][f])
# pad each figure to RPF using a global EXTRAS pool (2nd/3rd non-overlapping views per cell,
# gap-weighted so large angles are preferred), avoiding duplicate (cell,view).
def gapw(c):
    return c["score"]*{"100":1.18,"80":1.12,"60":1.05}.get(c["angle"],1.0)
used=set()
for f in range(NF):
    for c in figs[f]: used.add((c["cell"],c["view"]))
extras=[]
for sc,lst in by_scene.items():
    percell={}
    for c in sorted(lst,key=lambda c:-gapw(c)):
        if (c["cell"],c["view"]) in used: continue
        n=percell.get(c["cell"],0)
        if n>=2: continue   # up to 2 extra views per cell
        if any(k["cell"]==c["cell"] and overlap(k["box"],c["box"])>0.55 for k in extras if k["cell"]==c["cell"]): continue
        extras.append(c); percell[c["cell"]]=n+1
extras.sort(key=lambda c:-gapw(c))
ei=0
for f in range(NF):
    figs[f]=sorted(figs[f], key=lambda c:-gapw(c))
    while len(figs[f])<RPF and ei<len(extras):
        c=extras[ei]; ei+=1
        if (c["cell"],c["view"]) in used: continue
        figs[f].append(c); used.add((c["cell"],c["view"]))
for i,fig in enumerate(figs):
    picks=[]
    for c in fig:
        picks.append(dict(cell=c["cell"], manifest_key=c.get("manifest_key"), best_view=c["view"],
                          scene=c["scene"], angle=c["angle"], box=c["box"],
                          label=(f"{c['angle']}° · {int(c['beats_all_frac']*100)}%px" if stage=="coverage_gap"
                                 else f"disc · {int(c['beats_all_frac']*100)}%px"),
                          win_description=""))
    json.dump(dict(stage=stage,picks=picks,caption=""),open(f"{VM}/fig_{stage}_{i+1}.json","w"),indent=1)
    print(f"fig {i+1}: "+", ".join(f"{p['scene']}@{p['angle']}/{p['best_view'][:5]}" for p in picks))
print(f"wrote {NF} figure specs for {stage}")
