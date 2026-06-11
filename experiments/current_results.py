#!/usr/bin/env python3
"""Current best results per cell (priority: WSL final > room nview3 validated > v2 > v1),
compared per-cell and on MIP8 averages against baselines and the published GS-LSS row."""
import csv, glob, os, re
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MA = os.path.join(REPO, "experiments", "metrics_multiangle")
ANGLES = [20.0, 40.0, 60.0, 80.0, 100.0]
MIP8 = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "treehill"]
MIP9 = MIP8 + ["flowers"]

def rows_of(path):
    try:
        return list(csv.DictReader(open(path)))
    except Exception:
        return []

def triple(r):
    return (float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"]))

def pick30k(path):
    rs = rows_of(path)
    for r in rs:
        if r.get("method") == "ours_30000":
            return triple(r)
    return triple(rs[-1]) if rs else None

OURS, SRC = {}, {}
def put(sc, ang, v, src):
    if v and (sc, ang) not in OURS:
        OURS[(sc, ang)] = v
        SRC[(sc, ang)] = src

# 1. WSL final cells
for d in glob.glob(f"{REPO}/outputs/wsl_results/sweep_results_final/mipnerf360/*_i30000*"):
    m = re.match(r"^(.*)_angle_([0-9.]+)_i30000(_aa)?$", os.path.basename(d))
    if m:
        put(m.group(1), float(m.group(2)), pick30k(os.path.join(d, "metrics.csv")), "FINAL")
# 2. room nview3 validated (server dual-ckpt + @60 confirm)
for a in ("20.0", "40.0", "80.0", "100.0"):
    put("room", float(a), pick30k(f"{REPO}/outputs/sweep_results/mipnerf360/room_angle_{a}_i30000_aa/metrics.csv"), "ROOMv")
put("room", 60.0, pick30k("/data1/zhangao/outputs/diag/mipnerf360/room_angle_60.0_nview3_confirm_i30000/metrics.csv"), "ROOMv")
# 3. v2 server partial sweep
for d in glob.glob(f"{REPO}/outputs/sweep_results/mipnerf360/*_i30000*"):
    m = re.match(r"^(.*)_angle_([0-9.]+)_i30000(_aa)?$", os.path.basename(d))
    if m and m.group(1) != "room":
        put(m.group(1), float(m.group(2)), pick30k(os.path.join(d, "metrics.csv")), "v2")
# 4. v1 fill
for d in glob.glob(f"{REPO}/sweep_results_v1/mipnerf360/*_i30000*"):
    m = re.match(r"^(.*)_angle_([0-9.]+)_i30000(_aa)?$", os.path.basename(d))
    if m:
        put(m.group(1), float(m.group(2)), pick30k(os.path.join(d, "metrics.csv")), "v1")

# baselines
def load_flat(d, suffix="_metrics.csv"):
    out = {}
    pat = re.compile(r"^(.*)_test_selection_angle_([0-9.]+)" + re.escape(suffix) + r"$")
    for fn in os.listdir(d):
        m = pat.match(fn)
        if m:
            rs = rows_of(os.path.join(d, fn))
            if rs:
                out[(m.group(1), float(m.group(2)))] = triple(rs[-1])
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
                rs = rows_of(os.path.join(base, sub, fn))
                if rs:
                    out[(m2.group(1), float(m.group(1)))] = triple(rs[-1])
    return out
BASE = {
    "3DGS": load_flat(os.path.join(MA, "3dgs-original")),
    "Scaffold": load_flat(os.path.join(MA, "scaffoldgs-original")),
    "Hier": load_depth(),
    "MVGS": load_flat(os.path.join(MA, "mvgs-original-mv3")),
    "PGSR": load_flat(os.path.join(MA, "pgsr2-original"), suffix=".csv"),
    "MCMC": load_flat(os.path.join(MA, "3dgs-mcmc-original")),
    "MH": load_flat(os.path.join(MA, "mh-3dgs-original")),
}
PUB = {20: (0.803, 25.646, 0.193), 40: (0.745, 22.823, 0.241), 60: (0.699, 21.406, 0.292),
       80: (0.660, 20.048, 0.322), 100: (0.617, 18.555, 0.352)}

print("=== per-cell PSNR: ours vs best baseline (MIP9) ===")
print(f"{'scene':>9} | " + " | ".join(f"{int(a):>17}" for a in ANGLES))
wins = 0; cells = 0
for sc in MIP9:
    row = []
    for ang in ANGLES:
        v = OURS.get((sc, ang))
        cands = [(n, b[(sc, ang)][1]) for n, b in BASE.items() if (sc, ang) in b]
        if not v or not cands:
            row.append(f"{'—':>17}")
            continue
        bn, bp = max(cands, key=lambda t: t[1])
        mark = "+" if v[1] >= bp else "-"
        if v[1] >= bp:
            wins += 1
        cells += 1
        row.append(f"{v[1]:6.2f}{mark}{bn[:4]:>5}{bp:6.2f}")
    print(f"{sc:>9} | " + " | ".join(row))
print(f"cells won vs best baseline: {wins}/{cells}")

print("\n=== MIP8 averages (S/P/L): ours vs published GS-LSS vs best external ===")
for ang in ANGLES:
    vals = [OURS[(sc, ang)] for sc in MIP8 if (sc, ang) in OURS]
    n = len(vals)
    avg = tuple(sum(v[i] for v in vals) / n for i in range(3))
    p = PUB[int(ang)]
    ext = {}
    for nm, b in BASE.items():
        bv = [b[(sc, ang)] for sc in MIP8 if (sc, ang) in b]
        if len(bv) == 8:
            ext[nm] = tuple(sum(v[i] for v in bv) / 8 for i in range(3))
    bestP = max(ext.items(), key=lambda t: t[1][1])
    bestS = max(ext.items(), key=lambda t: t[1][0])
    bestL = min(ext.items(), key=lambda t: t[1][2])
    print(f"{int(ang):>4}°: ours {avg[0]:.3f}/{avg[1]:6.3f}/{avg[2]:.3f} (n={n}) | "
          f"pub {p[0]:.3f}/{p[1]:6.3f}/{p[2]:.3f} (dP {avg[1]-p[1]:+.2f}) | "
          f"extP {bestP[0]}:{bestP[1][1]:6.3f} (dP {avg[1]-bestP[1][1]:+.2f}) "
          f"extS {bestS[0]}:{bestS[1][0]:.3f} ({avg[0]-bestS[1][0]:+.3f}) "
          f"extL {bestL[0]}:{bestL[1][2]:.3f} ({bestL[1][2]-avg[2]:+.3f})")
print("\nsources used:", {s: sum(1 for k in SRC if SRC[k] == s) for s in set(SRC.values())})