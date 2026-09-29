#!/usr/bin/env python3
"""Stage-1 standing: our HARVESTED best per cell (best_coverage_gap/INDEX.csv) vs the per-cell
best competitor (7 baselines + old GS-LSS develop3 oracle). Lists losers as tuning targets.
Paper Table 1 = 8 Mip scenes (NO flowers) + T&T; we report the 8 here and flowers separately."""
import csv, glob, json, os, re
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MA = os.path.join(REPO, "experiments/metrics_multiangle")
ANGLES = [20.0, 40.0, 60.0, 80.0, 100.0]
MIP8 = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "treehill"]
INDEX = "/data1/zhangao/outputs/best_coverage_gap/INDEX.csv"

def psnr_of(path):
    try:
        rows = list(csv.DictReader(open(path)))
        for r in rows:
            if r.get("method") == "ours_15000":
                return float(r["PSNR"])
        return float(rows[-1]["PSNR"]) if rows else None
    except Exception:
        return None

# our current best per cell from the harvest
OURS = {}
for r in csv.DictReader(open(INDEX)):
    sc, a = r["cell"].split("@")
    OURS[(sc, float(a))] = (float(r["PSNR"]), r["source_run"], r["checkpoint"], r["ply_present"])

def load_flat(d, suf="_metrics.csv"):
    out = {}
    pat = re.compile(r"^(.*)_test_selection_angle_([0-9.]+)" + re.escape(suf) + r"$")
    if not os.path.isdir(d): return out
    for fn in os.listdir(d):
        m = pat.match(fn)
        if m:
            v = psnr_of(os.path.join(d, fn))
            if v is not None: out[(m.group(1), float(m.group(2)))] = v
    return out
def load_depth():
    out = {}
    base = os.path.join(MA, "3dgs-depth")
    if not os.path.isdir(base): return out
    for sub in os.listdir(base):
        m = re.match(r"^3dgs-depth-test_selection_angle_([0-9.]+)$", sub)
        if not m: continue
        for fn in os.listdir(os.path.join(base, sub)):
            m2 = re.match(r"^(.*)_metrics\.csv$", fn)
            if m2:
                v = psnr_of(os.path.join(base, sub, fn))
                if v is not None: out[(m2.group(1), float(m.group(1)))] = v
    return out
BASE = {"3DGS": load_flat(os.path.join(MA, "3dgs-original")), "Scaf": load_flat(os.path.join(MA, "scaffoldgs-original")),
        "Hier": load_depth(), "MVGS": load_flat(os.path.join(MA, "mvgs-original-mv3")),
        "PGSR": load_flat(os.path.join(MA, "pgsr2-original"), ".csv"), "MCMC": load_flat(os.path.join(MA, "3dgs-mcmc-original")),
        "MH3D": load_flat(os.path.join(MA, "mh-3dgs-original"))}

cells = defaultdict(list)
for root in glob.glob("/data1/zhangao/outputs/develop3_multi_angle*"):
    for rj in glob.glob(os.path.join(root, "*", "*", "results.json")):
        sc = os.path.basename(os.path.dirname(os.path.dirname(rj)))
        m = re.search(r"angle_([0-9.]+)", os.path.basename(os.path.dirname(rj)))
        if not m or sc not in MIP8: continue
        try: d = json.load(open(rj))
        except: continue
        k = "ours_30000" if "ours_30000" in d else (sorted(d)[-1] if d else None)
        if k and "PSNR" in d[k]: cells[(sc, float(m.group(1)))].append(d[k]["PSNR"])
for dd in [os.path.join(MA, "develop3_anti"), os.path.join(MA, "develop3_woanti")]:
    if os.path.isdir(dd):
        for fn in os.listdir(dd):
            m = re.match(r"^(.*)_test_selection_angle_([0-9.]+)_metrics\.csv$", fn)
            if m and m.group(1) in MIP8:
                v = psnr_of(os.path.join(dd, fn))
                if v is not None: cells[(m.group(1), float(m.group(2)))].append(v)
OLD = {k: max(v) for k, v in cells.items()}

print(f"{'cell':>13} | {'ours':>7} {'src':>26} {'ckpt':>6} ply | {'best comp':>16} | verdict")
win = lose = 0; losers = []
for sc in MIP8:
    for a in ANGLES:
        if (sc, a) not in OURS: continue
        ob, src, ck, ply = OURS[(sc, a)]
        comp = {n: b[(sc, a)] for n, b in BASE.items() if (sc, a) in b}
        if (sc, a) in OLD: comp["oldGSLSS"] = OLD[(sc, a)]
        if not comp: continue
        cn, cv = max(comp.items(), key=lambda t: t[1])
        d = ob - cv
        if d >= 0: win += 1; verdict = "WIN "
        else: lose += 1; verdict = "lose"; losers.append((sc, a, d, cn, cv, src, ck, ply))
        print(f"{sc+'@'+str(int(a)):>13} | {ob:>7.2f} {src[:26]:>26} {ck.replace('iter_',''):>6} {ply[0]:>3} | {cn:>8} {cv:>6.2f} | {verdict} {d:+.2f}")
print(f"\nMIP8 (Table-1 scenes): WIN {win} | LOSE {lose}")
print("\n=== LOSERS (tuning targets, by margin) ===")
for sc, a, d, cn, cv, src, ck, ply in sorted(losers, key=lambda t: t[2]):
    print(f"  {sc}@{int(a)}: behind {cn} by {-d:.2f} (ours {OURS[(sc,a)][0]:.2f} from {src}@{ck}, target {cv:.2f})")
