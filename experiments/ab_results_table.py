#!/usr/bin/env python3
"""Final A/B(/T*) sweep results table vs all baselines + old GS-LSS oracle.

Ours per cell = best PSNR over A_ (shipped), B_ (MH), and any T*_ (per-scene tuned) runs,
mirroring the published GS-LSS row's per-cell-best-over-runs methodology.
Aggregates use the paper's MIP8 convention (8 Mip-NeRF 360 scenes, NO flowers);
flowers cells are reported separately.

Usage: python experiments/ab_results_table.py [--iter 15000] [--md out.md]
"""
import argparse, csv, glob, json, os, re
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = "/data1/zhangao/outputs/diag/mipnerf360"
WR = os.path.join(REPO, "outputs/wsl_results/diag/mipnerf360")
MA = os.path.join(REPO, "experiments/metrics_multiangle")
ANGLES = [20.0, 40.0, 60.0, 80.0, 100.0]
MIP8 = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "treehill"]
MIP9 = MIP8 + ["flowers"]
METRICS = ("PSNR", "SSIM", "LPIPS")

def row_of(path, method=None):
    """-> {PSNR,SSIM,LPIPS} for the requested method row (or last row)."""
    try:
        rows = list(csv.DictReader(open(path)))
        if not rows:
            return None
        r = None
        if method:
            for x in rows:
                if x.get("method") == method:
                    r = x; break
        if r is None:
            r = rows[-1]
        return {m: float(r[m]) for m in METRICS}
    except Exception:
        return None

FINAL = False  # set by --final: read the locked 30K dual-ckpt runs (F_ dirs), row ours_{it}

def ours_variants(sc, a, it):
    """All our runs for a cell -> {variant_tag: metrics}."""
    out = {}
    if FINAL:
        # the final 30K run lives in _i30000 with BOTH ours_15000 and ours_30000 rows;
        # read the ckpt requested via --iter (15000 = mid-schedule snapshot, 30000 = final).
        for base in (SD, WR):
            p = os.path.join(base, f"{sc}_angle_{a}_F_{sc}_{a}_i30000", "metrics.csv")
            v = row_of(p, f"ours_{it}")
            if v is not None:
                out["F"] = v
        return out
    # dir = {sc}_angle_{a}_{PREFIX}_{sc}_{a}_i{it}; PREFIX in {A,B,T1,T2cap,...}
    pat = re.compile(rf"^{re.escape(sc)}_angle_{a}_([A-Za-z][A-Za-z0-9]*)_{re.escape(sc)}_{a}_i{it}$")
    for base in (SD, WR):
        if not os.path.isdir(base):
            continue
        for d in os.listdir(base):
            m = pat.match(d)
            if not m:
                continue
            v = row_of(os.path.join(base, d, "metrics.csv"), f"ours_{it}")
            if v is not None:
                out[m.group(1)] = v   # A / B / T1 / T2cap / ...
    return out

# ---------------- baselines (full metric rows, 30k) ----------------
def load_flat(d, suf="_metrics.csv"):
    out = {}
    pat = re.compile(r"^(.*)_test_selection_angle_([0-9.]+)" + re.escape(suf) + r"$")
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        m = pat.match(fn)
        if m:
            v = row_of(os.path.join(d, fn))
            if v is not None:
                out[(m.group(1), float(m.group(2)))] = v
    return out

def load_depth():
    out = {}
    base = os.path.join(MA, "3dgs-depth")
    if not os.path.isdir(base):
        return out
    for sub in os.listdir(base):
        m = re.match(r"^3dgs-depth-test_selection_angle_([0-9.]+)$", sub)
        if not m:
            continue
        for fn in os.listdir(os.path.join(base, sub)):
            m2 = re.match(r"^(.*)_metrics\.csv$", fn)
            if m2:
                v = row_of(os.path.join(base, sub, fn))
                if v is not None:
                    out[(m2.group(1), float(m.group(1)))] = v
    return out

BASELINES = [
    ("3DGS",        load_flat(os.path.join(MA, "3dgs-original"))),
    ("Scaffold-GS", load_flat(os.path.join(MA, "scaffoldgs-original"))),
    ("Hier-GS",     load_depth()),
    ("MVGS",        load_flat(os.path.join(MA, "mvgs-original-mv3"))),
    ("PGSR",        load_flat(os.path.join(MA, "pgsr2-original"), ".csv")),
    ("3DGS-MCMC",   load_flat(os.path.join(MA, "3dgs-mcmc-original"))),
    ("MH-3DGS",     load_flat(os.path.join(MA, "mh-3dgs-original"))),
]

# ---------------- old GS-LSS oracle (per-cell best-PSNR run; its SSIM/LPIPS) ----------------
def load_old():
    cells = defaultdict(list)
    for root in glob.glob("/data1/zhangao/outputs/develop3_multi_angle*"):
        for rj in glob.glob(os.path.join(root, "*", "*", "results.json")):
            sc = os.path.basename(os.path.dirname(os.path.dirname(rj)))
            m = re.search(r"angle_([0-9.]+)", os.path.basename(os.path.dirname(rj)))
            if not m or sc not in MIP9:
                continue
            try:
                d = json.load(open(rj))
            except Exception:
                continue
            k = "ours_30000" if "ours_30000" in d else (sorted(d)[-1] if d else None)
            if k and all(mm in d[k] for mm in METRICS):
                cells[(sc, float(m.group(1)))].append({mm: d[k][mm] for mm in METRICS})
    for dd in [os.path.join(MA, "develop3_anti"), os.path.join(MA, "develop3_woanti")]:
        if not os.path.isdir(dd):
            continue
        for fn in os.listdir(dd):
            m = re.match(r"^(.*)_test_selection_angle_([0-9.]+)_metrics\.csv$", fn)
            if m and m.group(1) in MIP9:
                v = row_of(os.path.join(dd, fn))
                if v is not None:
                    cells[(m.group(1), float(m.group(2)))].append(v)
    return {k: max(v, key=lambda r: r["PSNR"]) for k, v in cells.items()}

OLD = load_old()

def agg(cells_dict, scenes, a):
    """Mean over scenes at angle a; None unless ALL scenes present."""
    vals = [cells_dict.get((sc, a)) for sc in scenes]
    if any(v is None for v in vals):
        return None
    return {m: sum(v[m] for v in vals) / len(vals) for m in METRICS}

def fmt(v, m):
    if v is None:
        return "  --  "
    return f"{v[m]:6.3f}" if m != "PSNR" else f"{v[m]:6.2f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=15000)
    ap.add_argument("--final", action="store_true", help="read the locked 30K dual-ckpt F_ runs")
    ap.add_argument("--md", default=None, help="also write a markdown file")
    args = ap.parse_args()
    it = args.iter
    global FINAL
    FINAL = args.final

    oursA, oursB, oursT, oursBest, bestsrc = {}, {}, {}, {}, {}
    for sc in MIP9:
        for a in ANGLES:
            var = ours_variants(sc, a, it)
            if "A" in var: oursA[(sc, a)] = var["A"]
            if "B" in var: oursB[(sc, a)] = var["B"]
            tk = [k for k in var if k.startswith("T")]
            if tk:
                tb = max(tk, key=lambda k: var[k]["PSNR"])
                oursT[(sc, a)] = var[tb]
            if var:
                bk = max(var, key=lambda k: var[k]["PSNR"])
                oursBest[(sc, a)] = var[bk]; bestsrc[(sc, a)] = bk

    methods = BASELINES + [("GS-LSS(old,oracle)", OLD),
                           (f"Ours-A@{it//1000}k", oursA),
                           (f"Ours-best@{it//1000}k", oursBest)]

    lines = []
    P = lines.append
    P(f"==== AGGREGATE (MIP8, paper convention — no flowers) @ iter {it} ====")
    for m in METRICS:
        P(f"\n--- {m} ---")
        hdr = f"{'method':>20} |" + "".join(f" {int(a):>5}°" for a in ANGLES)
        P(hdr); P("-" * len(hdr))
        for name, cd in methods:
            row = f"{name:>20} |"
            for a in ANGLES:
                row += " " + fmt(agg(cd, MIP8, a), m)
            P(row)

    P(f"\n==== PER-CELL (ours best vs best competitor, PSNR) @ iter {it} ====")
    P(f"{'cell':>13} | {'A':>6} {'B':>6} {'T':>6} | {'best':>6} src | {'top competitor':>24} | margin")
    win = lose = pend = 0
    for sc in MIP9:
        for a in ANGLES:
            ob = oursBest.get((sc, a))
            comp = {n: cd[(sc, a)] for n, cd in BASELINES if (sc, a) in cd}
            if (sc, a) in OLD:
                comp["GS-LSS(old)"] = OLD[(sc, a)]
            if ob is None:
                pend += 1; continue
            if not comp:
                continue
            cn = max(comp, key=lambda n: comp[n]["PSNR"])
            d = ob["PSNR"] - comp[cn]["PSNR"]
            win += d >= 0; lose += d < 0
            av, bv, tv = oursA.get((sc, a)), oursB.get((sc, a)), oursT.get((sc, a))
            P(f"{sc+'@'+str(int(a)):>13} | "
              + (f"{av['PSNR']:6.2f}" if av else "   -- ") + " "
              + (f"{bv['PSNR']:6.2f}" if bv else "   -- ") + " "
              + (f"{tv['PSNR']:6.2f}" if tv else "   -- ") + " | "
              + f"{ob['PSNR']:6.2f} {bestsrc[(sc,a)]:>3} | "
              + f"{cn:>17} {comp[cn]['PSNR']:6.2f} | "
              + ("WIN " if d >= 0 else "lose") + f" {d:+.2f}")
    P(f"\nWIN {win} | lose {lose} | pending {pend}")

    P("\n==== FLOWERS (excluded from paper MIP8 averages) ====")
    for a in ANGLES:
        ob = oursBest.get(("flowers", a))
        old = OLD.get(("flowers", a))
        P(f"  flowers@{int(a)}: ours " + (f"{ob['PSNR']:.2f}" if ob else "--")
          + "  old-oracle " + (f"{old['PSNR']:.2f}" if old else "--"))

    out = "\n".join(lines)
    print(out)
    if args.md:
        with open(args.md, "w") as f:
            f.write("```\n" + out + "\n```\n")

if __name__ == "__main__":
    main()
