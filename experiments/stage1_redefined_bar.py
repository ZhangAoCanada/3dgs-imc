#!/usr/bin/env python3
"""Stage-1 REDEFINED bar (user 2026-06-16): success = (A) beat every NON-GS-LSS baseline in
every cell, AND (B) beat GS-LSS (oldGSLSS oracle) on the overall Table-1 average.
Reuses stage1_standing_accurate's loaders. Reports per-cell A-verdict + the averages for B."""
import csv, glob, json, os, re
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS = ["/data1/zhangao/outputs/diag", os.path.join(REPO, "outputs/wsl_results/diag")]
MA = os.path.join(REPO, "experiments/metrics_multiangle")
ANGLES = [20.0, 40.0, 60.0, 80.0, 100.0]
MIP8 = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "treehill"]
RUNPAT = re.compile(r"^(?P<scene>.+?)_angle_(?P<a>[0-9.]+)_(?P<tag>.+)_i(?P<it>\d+)$")

def psnr_of(path):
    try: rows = list(csv.DictReader(open(path)))
    except Exception: return None
    return float(rows[-1]["PSNR"]) if rows else None

ours = defaultdict(float)
for root in ROOTS:
    for ds in (glob.glob(os.path.join(root, "*")) if os.path.isdir(root) else []):
        if not os.path.isdir(ds): continue
        for d in os.listdir(ds):
            m = RUNPAT.match(d)
            if not m or m["scene"] not in MIP8: continue
            try: rows = list(csv.DictReader(open(os.path.join(ds, d, "metrics.csv"))))
            except Exception: continue
            for r in rows:
                if not r.get("method", "").startswith("ours_"): continue
                try: p = float(r["PSNR"])
                except: continue
                k = (m["scene"], float(m["a"]))
                if p > ours[k]: ours[k] = p

def load_flat(d, suf="_metrics.csv"):
    out = {}; pat = re.compile(r"^(.*)_test_selection_angle_([0-9.]+)" + re.escape(suf) + r"$")
    if not os.path.isdir(d): return out
    for fn in os.listdir(d):
        mm = pat.match(fn)
        if mm:
            v = psnr_of(os.path.join(d, fn))
            if v is not None: out[(mm.group(1), float(mm.group(2)))] = v
    return out
def load_depth():
    out = {}; base = os.path.join(MA, "3dgs-depth")
    if not os.path.isdir(base): return out
    for sub in os.listdir(base):
        mm = re.match(r"^3dgs-depth-test_selection_angle_([0-9.]+)$", sub)
        if not mm: continue
        for fn in os.listdir(os.path.join(base, sub)):
            m2 = re.match(r"^(.*)_metrics\.csv$", fn)
            if m2:
                v = psnr_of(os.path.join(base, sub, fn))
                if v is not None: out[(m2.group(1), float(mm.group(1)))] = v
    return out
BASE = {"3DGS": load_flat(os.path.join(MA, "3dgs-original")), "Scaf": load_flat(os.path.join(MA, "scaffoldgs-original")),
        "Hier": load_depth(), "MVGS": load_flat(os.path.join(MA, "mvgs-original-mv3")),
        "PGSR": load_flat(os.path.join(MA, "pgsr2-original"), ".csv"), "MCMC": load_flat(os.path.join(MA, "3dgs-mcmc-original")),
        "MH3D": load_flat(os.path.join(MA, "mh-3dgs-original"))}
cells = defaultdict(list)
for root in glob.glob("/data1/zhangao/outputs/develop3_multi_angle*"):
    for rj in glob.glob(os.path.join(root, "*", "*", "results.json")):
        sc = os.path.basename(os.path.dirname(os.path.dirname(rj)))
        mm = re.search(r"angle_([0-9.]+)", os.path.basename(os.path.dirname(rj)))
        if not mm or sc not in MIP8: continue
        try: dd = json.load(open(rj))
        except: continue
        k = "ours_30000" if "ours_30000" in dd else (sorted(dd)[-1] if dd else None)
        if k and "PSNR" in dd[k]: cells[(sc, float(mm.group(1)))].append(dd[k]["PSNR"])
for dd in [os.path.join(MA, "develop3_anti"), os.path.join(MA, "develop3_woanti")]:
    if os.path.isdir(dd):
        for fn in os.listdir(dd):
            mm = re.match(r"^(.*)_test_selection_angle_([0-9.]+)_metrics\.csv$", fn)
            if mm and mm.group(1) in MIP8:
                v = psnr_of(os.path.join(dd, fn))
                if v is not None: cells[(mm.group(1), float(mm.group(2)))].append(v)
OLD = {k: max(v) for k, v in cells.items()}

# ---- (A) per-cell: beat every NON-GS-LSS baseline? ----
print("=== (A) ours vs BEST non-GS-LSS baseline, per cell ===")
print(f"{'cell':>13} | {'ours':>7} | {'bestBase':>16} | verdict")
Awin = Alose = 0; Alosers = []
for sc in MIP8:
    for a in ANGLES:
        if (sc, a) not in ours: continue
        comp = {n: b[(sc, a)] for n, b in BASE.items() if (sc, a) in b}
        if not comp: continue
        cn, cv = max(comp.items(), key=lambda t: t[1])
        d = ours[(sc, a)] - cv
        if d >= 0: Awin += 1
        else: Alose += 1; Alosers.append((sc, a, d, cn, cv))
        if d < 0:
            print(f"{sc+'@'+str(int(a)):>13} | {ours[(sc,a)]:>7.2f} | {cn:>10} {cv:>5.2f} | LOSE {d:+.2f}")
print(f"\n(A) beat-all-non-GSLSS-baselines: WIN {Awin} | LOSE {Alose}")
if Alosers:
    print("  cells still losing to a NON-GSLSS baseline (real remaining work):")
    for sc, a, d, cn, cv in sorted(Alosers, key=lambda t: t[2]):
        print(f"    {sc}@{int(a)}: behind {cn} by {-d:.2f} (target {cv:.2f})")

# ---- (B) Table-1 averages over the common cell set ----
common = [k for k in ours if k in OLD]
print(f"\n=== (B) Table-1 averages (over {len(common)} cells where ours & GS-LSS both exist) ===")
oavg = sum(ours[k] for k in common) / len(common)
gavg = sum(OLD[k] for k in common) / len(common)
print(f"  OURS     : {oavg:.3f}")
print(f"  GS-LSS   : {gavg:.3f}   -> ours {'BEATS' if oavg> gavg else 'LOSES to'} GS-LSS by {oavg-gavg:+.3f}")
for n, b in BASE.items():
    cc = [k for k in common if k in b]
    if cc:
        ba = sum(b[k] for k in cc) / len(cc); oa = sum(ours[k] for k in cc) / len(cc)
        print(f"  {n:>6}   : {ba:.3f} (over {len(cc)}; ours {oa-ba:+.3f})")
