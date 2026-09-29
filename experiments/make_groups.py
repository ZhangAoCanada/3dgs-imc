#!/usr/bin/env python3
"""Rebuild the inlined GROUPS array in curate_vision.js from the current cand_<stage>.json indexes.
Chunks each scene's candidate ids into <=CHUNK per judge agent (pool is large now). Stable: regex-
replaces `const GROUPS = [...]` in place so Workflow can be re-run with scriptPath."""
import json, re, os, sys
VM = "/data1/zhangao/outputs/viz_multiangle"
JS = "/data1/zhangao/repos/3dgs-imc-codex/experiments/curate_vision.js"
CHUNK = 16
ONLY = sys.argv[1] if len(sys.argv) > 1 else None   # optional stage filter
groups = []
for stage in ([ONLY] if ONLY else ["coverage_gap", "discontinuous"]):
    idx = json.load(open(f"{VM}/cand_{stage}.json"))
    byscene = {}
    for v in idx.values(): byscene.setdefault(v["scene"], []).append(v["id"])
    for scene, ids in byscene.items():
        ids = sorted(ids)
        for j in range(0, len(ids), CHUNK):
            groups.append(dict(stage=stage, scene=scene, dir=f"/tmp/cand_{stage}/{scene}", ids=ids[j:j+CHUNK]))
arr = json.dumps(groups)
src = open(JS).read()
src2 = re.sub(r"const GROUPS = \[.*?\]\n", "const GROUPS = " + arr + "\n", src, count=1, flags=re.S)
assert src2 != src, "GROUPS replace failed"
open(JS, "w").write(src2)
print(f"wrote {len(groups)} groups ({sum(len(g['ids']) for g in groups)} candidates) into curate_vision.js")
for stage in ("coverage_gap", "discontinuous"):
    n = sum(len(g["ids"]) for g in groups if g["stage"] == stage)
    print(f"  {stage}: {n} candidates in {sum(1 for g in groups if g['stage']==stage)} judge groups")
