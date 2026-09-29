#!/usr/bin/env python3
"""ACCURATE Stage-1 standing: per cell, the TRUE max-PSNR over ALL our runs/checkpoints
(ignoring ply presence — collect_best_checkpoints is ply-biased and under-counts cells whose
best run's ply was pruned), vs the per-cell best competitor (7 baselines + frozen develop3 oracle).
Flags cells that are WON but whose best checkpoint lacks a ply (=> re-render to recover the deliverable).
Paper Table 1 = 8 Mip scenes (NO flowers) + T&T; this reports the 8 Mip scenes."""
import csv, glob, json, os, re
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS = ["/data1/zhangao/outputs/diag", os.path.join(REPO, "outputs/wsl_results/diag")]
MA = os.path.join(REPO, "experiments/metrics_multiangle")
ANGLES = [20.0, 40.0, 60.0, 80.0, 100.0]
MIP8 = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "treehill"]
RUNPAT = re.compile(r"^(?P<scene>.+?)_angle_(?P<a>[0-9.]+)_(?P<tag>.+)_i(?P<it>\d+)$")

def psnr_of(path, method=None):
    try:
        rows = list(csv.DictReader(open(path)))
    except Exception:
        return None
    if method:
        for r in rows:
            if r.get("method") == method:
                try: return float(r["PSNR"])
                except: return None
        return None
    return float(rows[-1]["PSNR"]) if rows else None

# ---- OURS: scan every run dir, every ours_<ck> checkpoint, record (psnr, ssim, lpips, tag, ck, ply) ----
ours = defaultdict(list)   # (scene, angle) -> list of dicts
for root in ROOTS:
    for ds in (glob.glob(os.path.join(root, "*")) if os.path.isdir(root) else []):
        if not os.path.isdir(ds): continue
        for d in os.listdir(ds):
            m = RUNPAT.match(d)
            if not m or m["scene"] not in MIP8: continue
            mc = os.path.join(ds, d, "metrics.csv")
            try: rows = list(csv.DictReader(open(mc)))
            except Exception: continue
            for r in rows:
                meth = r.get("method", "")
                if not meth.startswith("ours_"): continue
                ck = meth.split("_")[1]
                ply = os.path.join(ds, d, "point_cloud", f"iteration_{ck}", "point_cloud.ply")
                try: p = float(r["PSNR"])
                except: continue
                ours[(m["scene"], float(m["a"]))].append(
                    dict(psnr=p, tag=m["tag"], ck=ck, ply=os.path.exists(ply), dir=os.path.join(ds, d)))

# ---- competitors ----
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

print(f"{'cell':>13} | {'TRUEbest':>8} {'ck':>6} ply | {'best comp':>16} | verdict | best-with-ply")
win = lose = 0; losers = []; reready = []
for sc in MIP8:
    for a in ANGLES:
        recs = ours.get((sc, a), [])
        if not recs: continue
        best = max(recs, key=lambda r: r["psnr"])
        wply = [r for r in recs if r["ply"]]
        bestply = max(wply, key=lambda r: r["psnr"]) if wply else None
        comp = {n: b[(sc, a)] for n, b in BASE.items() if (sc, a) in b}
        if (sc, a) in OLD: comp["oldGSLSS"] = OLD[(sc, a)]
        if not comp: continue
        cn, cv = max(comp.items(), key=lambda t: t[1])
        d = best["psnr"] - cv
        if d >= 0: win += 1; verdict = "WIN "
        else: lose += 1; verdict = "lose"; losers.append((sc, a, d, cn, cv))
        bp = f"{bestply['psnr']:.2f}@{bestply['ck']}({bestply['tag'][:14]})" if bestply else "NO-PLY!"
        # flag: WON on true best but that checkpoint has no ply -> re-render to recover deliverable
        flag = ""
        if d >= 0 and not best["ply"]:
            flag = " <= ply MISSING (re-render)"; reready.append((sc, a, best))
        print(f"{sc+'@'+str(int(a)):>13} | {best['psnr']:>8.2f} {best['ck']:>6} {('Y' if best['ply'] else 'N'):>3} | {cn:>8} {cv:>6.2f} | {verdict} {d:+.2f} | {bp}{flag}")
print(f"\nMIP8: WIN {win} | LOSE {lose}   (true max-PSNR, ply-independent)")
if reready:
    print(f"\n=== WON but best ckpt has NO ply ({len(reready)}) — re-render to recover the deliverable ===")
    for sc, a, b in reready:
        print(f"  {sc}@{int(a)}: {b['psnr']:.2f}@{b['ck']} tag={b['tag']}  dir={b['dir']}")
print("\n=== TRUE LOSERS (still need tuning), by margin ===")
for sc, a, d, cn, cv in sorted(losers, key=lambda t: t[2]):
    print(f"  {sc}@{int(a)}: behind {cn} by {-d:.2f} (target {cv:.2f})")
