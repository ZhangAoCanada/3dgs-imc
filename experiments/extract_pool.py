#!/usr/bin/env python3
"""Extract the full judge pool (every ranked candidate with stage/scene) from a curate_vision
workflow's agent transcripts -> /tmp/judge_pool.json. Usage: extract_pool.py <workflow_dir>"""
import json, glob, re, sys
WF = sys.argv[1]
def walk(o, name, key):
    out = []
    if isinstance(o, dict):
        if o.get("name") == name and isinstance(o.get("input"), dict) and key in o["input"]: out.append(o["input"])
        for v in o.values(): out += walk(v, name, key)
    elif isinstance(o, list):
        for v in o: out += walk(v, name, key)
    return out
pool = []
for f in glob.glob(f"{WF}/*.jsonl"):
    txt = open(f).read()
    if '"ranked"' not in txt: continue
    m = re.search(r"/tmp/cand_(coverage_gap|discontinuous)/([a-z]+)/", txt)
    if not m: continue
    stage, scene = m.group(1), m.group(2)
    for ln in txt.splitlines():
        if '"ranked"' not in ln: continue
        try: ev = json.loads(ln)
        except Exception: continue
        for inp in walk(ev, "StructuredOutput", "ranked"):
            for r in inp["ranked"]:
                r = dict(r); r["stage"] = stage; r["scene"] = scene; pool.append(r)
seen = {}
for r in pool: seen[(r["stage"], r["scene"], r["id"])] = r
pool = list(seen.values())
json.dump(pool, open("/tmp/judge_pool.json", "w"))
import collections
print("judge pool:", len(pool))
for stage in ("coverage_gap", "discontinuous"):
    P = [r for r in pool if r["stage"] == stage]
    good = [r for r in P if r.get("readable", 0) >= 2 and r.get("obvious_win", 0) >= 2]
    print(f"  {stage}: {len(P)} judged, good(rd>=2,ow>=2)={len(good)} per scene:",
          dict(collections.Counter(r["scene"] for r in good)))
