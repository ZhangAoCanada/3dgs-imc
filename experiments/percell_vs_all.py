#!/usr/bin/env python3
"""Per-cell: does our best (A=shipped / B=MH, both NON-fresh) beat ALL methods incl old GS-LSS oracle?
Cells that lose -> per-scene tuning targets. Reads the 15k non-fresh A/B sweep + baselines + develop3 oracle."""
import csv, glob, json, os, re
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = "/data1/zhangao/outputs/diag/mipnerf360"
WR = os.path.join(REPO, "outputs/wsl_results/diag/mipnerf360")
MA = os.path.join(REPO, "experiments/metrics_multiangle")
ANGLES = [20.0, 40.0, 60.0, 80.0, 100.0]
MIP9 = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "treehill", "flowers"]

def psnr_of(path):
    try:
        rows = list(csv.DictReader(open(path)))
        for r in rows:
            if r.get("method") == "ours_15000":
                return float(r["PSNR"])
        return float(rows[-1]["PSNR"]) if rows else None
    except Exception:
        return None

def ours(sc, a, cfg):
    for base in (SD, WR):
        p = os.path.join(base, f"{sc}_angle_{a}_{cfg}_{sc}_{a}_i15000", "metrics.csv")
        v = psnr_of(p)
        if v is not None:
            return v
    return None

# baselines (PSNR), per cell
def load_flat(d, suf="_metrics.csv"):
    out = {}
    pat = re.compile(r"^(.*)_test_selection_angle_([0-9.]+)" + re.escape(suf) + r"$")
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        m = pat.match(fn)
        if m:
            v = psnr_of(os.path.join(d, fn))
            if v is not None:
                out[(m.group(1), float(m.group(2)))] = v
    return out
def load_depth():
    out = {}
    base = os.path.join(MA, "3dgs-depth")
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

# old GS-LSS oracle (max over develop3 runs)
cells = defaultdict(list)
for root in glob.glob("/data1/zhangao/outputs/develop3_multi_angle*"):
    for rj in glob.glob(os.path.join(root, "*", "*", "results.json")):
        sc = os.path.basename(os.path.dirname(os.path.dirname(rj)))
        m = re.search(r"angle_([0-9.]+)", os.path.basename(os.path.dirname(rj)))
        if not m or sc not in MIP9: continue
        try: d = json.load(open(rj))
        except: continue
        k = "ours_30000" if "ours_30000" in d else (sorted(d)[-1] if d else None)
        if k and "PSNR" in d[k]: cells[(sc, float(m.group(1)))].append(d[k]["PSNR"])
for dd in [os.path.join(MA, "develop3_anti"), os.path.join(MA, "develop3_woanti")]:
    if os.path.isdir(dd):
        for fn in os.listdir(dd):
            m = re.match(r"^(.*)_test_selection_angle_([0-9.]+)_metrics\.csv$", fn)
            if m and m.group(1) in MIP9:
                v = psnr_of(os.path.join(dd, fn))
                if v is not None: cells[(m.group(1), float(m.group(2)))].append(v)
OLD = {k: max(v) for k, v in cells.items()}

print("PER-CELL: ours(A=shipped, B=MH; both non-fresh) vs best competitor (incl old GS-LSS oracle)")
print(f"{'cell':>13} | {'A':>6} {'B':>6} | {'our best':>8} | {'best competitor':>22} | verdict")
win = lose = pend = 0
losers = []
for sc in MIP9:
    for a in ANGLES:
        A = ours(sc, a, "A"); B = ours(sc, a, "B")
        if A is None and B is None:
            pend += 1; continue
        ob = max([x for x in (A, B) if x is not None])
        comp = {**{n: b[(sc, a)] for n, b in BASE.items() if (sc, a) in b}}
        if (sc, a) in OLD: comp["oldGSLSS"] = OLD[(sc, a)]
        if not comp: continue
        cn, cv = max(comp.items(), key=lambda t: t[1])
        d = ob - cv
        verdict = "WIN " if d >= 0 else "lose"
        if d >= 0: win += 1
        else: lose += 1; losers.append((sc, a, d, cn, cv, A, B))
        astr = f"{A:.2f}" if A else "  -- "
        bstr = f"{B:.2f}" if B else "  -- "
        print(f"{sc+'@'+str(int(a)):>13} | {astr:>6} {bstr:>6} | {ob:>8.2f} | {cn:>10} {cv:>8.2f} | {verdict} {d:+.2f}")
print(f"\nWIN {win} | LOSE {lose} | pending {pend}  (of completed cells)")
print("\n=== PER-SCENE TUNING TARGETS (cells we lose, by margin) ===")
for sc, a, d, cn, cv, A, B in sorted(losers, key=lambda t: t[2]):
    print(f"  {sc}@{int(a)}: behind {cn} by {-d:.2f}  (ours A={A and round(A,2)} B={B and round(B,2)}, target {cv:.2f})")
