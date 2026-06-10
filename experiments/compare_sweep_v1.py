#!/usr/bin/env python3
"""Compare sweep_results_v1 (new one-config run.sh) against verified baselines
and the old per-cell-best GS-LSS (paper oracle), per cell and on average."""
import csv, glob, json, os, re
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MA = os.path.join(REPO, "experiments", "metrics_multiangle")
SWEEP = os.path.join(REPO, "sweep_results_v1", "mipnerf360")

ANGLES = [20.0, 40.0, 60.0, 80.0, 100.0]
MIP8 = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "treehill"]
MIP9 = MIP8 + ["flowers"]
INDOOR = {"bonsai", "counter", "kitchen", "room"}

def read_csv_last(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    r = rows[-1]
    return (float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"]))

def load_flat(d, suffix="_metrics.csv"):
    out = {}
    pat = re.compile(r"^(.*)_test_selection_angle_([0-9.]+)" + re.escape(suffix) + r"$")
    for fn in os.listdir(d):
        m = pat.match(fn)
        if m:
            v = read_csv_last(os.path.join(d, fn))
            if v:
                out[(m.group(1), float(m.group(2)))] = v
    return out

def load_depth():
    out = {}
    base = os.path.join(MA, "3dgs-depth")
    for sub in os.listdir(base):
        m = re.match(r"^3dgs-depth-test_selection_angle_([0-9.]+)$", sub)
        if not m:
            continue
        for fn in os.listdir(os.path.join(base, sub)):
            m2 = re.match(r"^(.*)_metrics\.csv$", fn)
            if m2:
                v = read_csv_last(os.path.join(base, sub, fn))
                if v:
                    out[(m2.group(1), float(m.group(1)))] = v
    return out

BASELINES = {
    "3DGS": load_flat(os.path.join(MA, "3dgs-original")),
    "Scaffold": load_flat(os.path.join(MA, "scaffoldgs-original")),
    "Hier(depth)": load_depth(),
    "MVGS": load_flat(os.path.join(MA, "mvgs-original-mv3")),
    "PGSR": load_flat(os.path.join(MA, "pgsr2-original"), suffix=".csv"),
    "MCMC": load_flat(os.path.join(MA, "3dgs-mcmc-original")),
    "MH": load_flat(os.path.join(MA, "mh-3dgs-original")),
}

# old GS-LSS oracle: per-cell best over all old tuned runs (results.json + csv dirs)
old_cells = defaultdict(list)
for root in ["/data1/zhangao/outputs/develop3_multi_angle",
             "/data1/zhangao/outputs/develop3_multi_angle2",
             "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing",
             "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing2"]:
    for rj in glob.glob(os.path.join(root, "*", "*", "results.json")):
        sc = os.path.basename(os.path.dirname(os.path.dirname(rj)))
        m = re.search(r"test_selection_angle_([0-9.]+)", os.path.basename(os.path.dirname(rj)))
        if not m or sc not in MIP9:
            continue
        try:
            data = json.load(open(rj))
        except Exception:
            continue
        key = "ours_30000" if "ours_30000" in data else (sorted(data)[-1] if data else None)
        if key and all(k in data[key] for k in ("SSIM", "PSNR", "LPIPS")):
            d = data[key]
            old_cells[(sc, float(m.group(1)))].append((d["SSIM"], d["PSNR"], d["LPIPS"]))
for d in [os.path.join(MA, "develop3_anti"), os.path.join(MA, "develop3_woanti"),
          "/data1/zhangao/outputs/develop3_multi_angle2/metrics/develop3",
          "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing/metrics/develop3",
          "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing2/metrics/develop3_1e0",
          "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing2/metrics/develop3_1e-2"]:
    if not os.path.isdir(d):
        continue
    for fn in os.listdir(d):
        m = re.match(r"^(.*)_test_selection_angle_([0-9.]+)_metrics\.csv$", fn)
        if m and m.group(1) in MIP9:
            v = read_csv_last(os.path.join(d, fn))
            if v:
                old_cells[(m.group(1), float(m.group(2)))].append(v)
OLD = {k: max(vs, key=lambda t: t[1]) for k, vs in old_cells.items()}

# new sweep
NEW = {}
for d in glob.glob(os.path.join(SWEEP, "*_i30000*")):
    m = re.match(r"^(.*)_angle_([0-9.]+)_i30000(_aa)?$", os.path.basename(d))
    if not m:
        continue
    v = read_csv_last(os.path.join(d, "metrics.csv"))
    if v:
        NEW[(m.group(1), float(m.group(2)))] = v

def avg(data, scenes, ang):
    vals = [data[(sc, ang)] for sc in scenes if (sc, ang) in data]
    if not vals:
        return None
    n = len(vals)
    return tuple(sum(v[i] for v in vals) / n for i in range(3))

# ---------------- dataset-average table ----------------
for scenes, label in [(MIP8, "MIP8 (paper protocol, no flowers)"), (MIP9, "MIP9 (all scenes)")]:
    print(f"\n================== {label}: averages ==================")
    methods = list(BASELINES.items()) + [("OLD-GSLSS(oracle)", OLD), ("NEW-run.sh", NEW)]
    print(f"{'method':>18} | " + " | ".join(f"{int(a)}° S/P/L" + " " * 8 for a in ANGLES))
    for name, data in methods:
        cells = []
        for ang in ANGLES:
            v = avg(data, scenes, ang)
            cells.append("%.3f/%6.3f/%.3f" % v if v else "      missing     ")
        print(f"{name:>18} | " + " | ".join(cells))

# ---------------- per-cell win/loss vs best baseline ----------------
print("\n================== per-cell: NEW vs best baseline (PSNR) ==================")
wins = defaultdict(int); losses = defaultdict(int)
rows = []
for ang in ANGLES:
    for sc in MIP9:
        if (sc, ang) not in NEW:
            continue
        nv = NEW[(sc, ang)]
        cands = [(n, b[(sc, ang)]) for n, b in BASELINES.items() if (sc, ang) in b]
        if not cands:
            continue
        bb_name, bb = max(cands, key=lambda t: t[1][1])
        ov = OLD.get((sc, ang))
        d = nv[1] - bb[1]
        rows.append((d, sc, ang, nv, bb_name, bb, ov))
        if d >= 0:
            wins[ang] += 1
        else:
            losses[ang] += 1
print(f"{'scene':>10} {'ang':>5} | {'NEW psnr':>8} | {'best-baseline':>20} | {'dPSNR':>7} | {'old-oracle':>10} | {'new-old':>8}")
for d, sc, ang, nv, bbn, bb, ov in sorted(rows):
    print(f"{sc:>10} {ang:>5} | {nv[1]:8.3f} | {bbn:>12} {bb[1]:7.3f} | {d:+7.3f} | {ov[1] if ov else float('nan'):10.3f} | {nv[1]-ov[1] if ov else float('nan'):+8.3f}")
print("\nper-angle win/loss vs best baseline (9 scenes):",
      {int(a): f"{wins[a]}W/{losses[a]}L" for a in ANGLES})
tot_w = sum(wins.values()); tot_l = sum(losses.values())
print(f"TOTAL: {tot_w}W / {tot_l}L over {tot_w+tot_l} cells")
