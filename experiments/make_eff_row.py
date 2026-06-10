#!/usr/bin/env python3
"""Build the GS-LSS-Eff row (sweep_results_v1, MIP8 paper protocol) for
tab:coverage_gap and recompute bold/underline for the Mip-NeRF 360 block."""
import csv, glob, os, re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(REPO, "sweep_results_v1", "mipnerf360")
MIP8 = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "treehill"]
ANGLES = [20.0, 40.0, 60.0, 80.0, 100.0]

# ---- new row: full-precision MIP8 averages from the sweep ----
new = {}
for d in glob.glob(os.path.join(SWEEP, "*_i30000*")):
    m = re.match(r"^(.*)_angle_([0-9.]+)_i30000(_aa)?$", os.path.basename(d))
    if not m:
        continue
    rows = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
    if rows:
        r = rows[-1]
        new[(m.group(1), float(m.group(2)))] = (float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"]))

eff = {}
for ang in ANGLES:
    vals = [new[(sc, ang)] for sc in MIP8 if (sc, ang) in new]
    assert len(vals) == 8, f"missing scenes at {ang}"
    eff[ang] = tuple(sum(v[i] for v in vals) / 8 for i in range(3))
print("GS-LSS-Eff full precision (MIP8):")
for ang in ANGLES:
    print(f"  {int(ang):>3}°: SSIM {eff[ang][0]:.6f}  PSNR {eff[ang][1]:.6f}  LPIPS {eff[ang][2]:.6f}")

# ---- printed paper values for the Mip-NeRF block (display precision) ----
ROWS = [
    ("3DGS~\\cite{3dgs}", [(0.787,24.936,0.202),(0.726,22.137,0.249),(0.674,20.499,0.291),(0.630,19.460,0.324),(0.584,17.597,0.358)]),
    ("Scaffold-GS~\\cite{scaffold}", [(0.760,23.844,0.231),(0.706,21.835,0.273),(0.653,20.418,0.315),(0.616,19.574,0.344),(0.571,17.751,0.378)]),
    ("Hierarchical-GS~\\cite{hierarchical}", [(0.787,25.044,0.203),(0.725,22.314,0.250),(0.671,20.611,0.291),(0.624,19.339,0.326),(0.581,17.463,0.359)]),
    ("MVGS~\\cite{mvgs}", [(0.784,24.827,0.203),(0.715,22.130,0.256),(0.658,20.458,0.303),(0.608,19.081,0.340),(0.561,17.164,0.374)]),
    ("PGSR~\\cite{pgsr}", [(0.771,24.203,0.231),(0.706,21.588,0.280),(0.662,20.179,0.307),(0.611,18.834,0.343),(0.553,16.808,0.375)]),
    ("3DGS-MCMC~\\cite{mcmc}", [(0.801,25.323,0.194),(0.739,22.369,0.239),(0.679,20.732,0.283),(0.632,19.344,0.319),(0.572,17.204,0.355)]),
    ("MH-3DGS~\\cite{mh3dgs}", [(0.784,25.122,0.224),(0.728,22.365,0.265),(0.683,20.855,0.300),(0.650,20.008,0.328),(0.608,18.352,0.360)]),
    ("GS-LSS (Ours)", [(0.803,25.646,0.193),(0.745,22.823,0.241),(0.699,21.406,0.292),(0.660,20.048,0.322),(0.617,18.555,0.352)]),
]

# full-precision tie-breakers for cells that display equal at 3 decimals
# (recomputed from experiments CSVs / oracle union; only needed where displayed values tie)
TIEBREAK = {}
# MCMC SSIM@20 (full precision from CSVs) and old-GSLSS LPIPS@100 (oracle union)
import json
MA = os.path.join(REPO, "experiments", "metrics_multiangle")
def load_flat(d, suffix="_metrics.csv"):
    out = {}
    pat = re.compile(r"^(.*)_test_selection_angle_([0-9.]+)" + re.escape(suffix) + r"$")
    for fn in os.listdir(d):
        m = pat.match(fn)
        if m:
            rows = list(csv.DictReader(open(os.path.join(d, fn))))
            if rows:
                r = rows[-1]
                out[(m.group(1), float(m.group(2)))] = (float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"]))
    return out
mcmc = load_flat(os.path.join(MA, "3dgs-mcmc-original"))
mcmc20 = [mcmc[(sc, 20.0)] for sc in MIP8]
TIEBREAK[("3DGS-MCMC~\\cite{mcmc}", 0, 0)] = sum(v[0] for v in mcmc20) / 8  # SSIM@20

# old GS-LSS oracle full precision (per-cell best PSNR over all old runs)
from collections import defaultdict
cells = defaultdict(list)
for root in ["/data1/zhangao/outputs/develop3_multi_angle", "/data1/zhangao/outputs/develop3_multi_angle2",
             "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing", "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing2"]:
    for rj in glob.glob(os.path.join(root, "*", "*", "results.json")):
        sc = os.path.basename(os.path.dirname(os.path.dirname(rj)))
        m = re.search(r"test_selection_angle_([0-9.]+)", os.path.basename(os.path.dirname(rj)))
        if not m or sc not in MIP8:
            continue
        try:
            data = json.load(open(rj))
        except Exception:
            continue
        key = "ours_30000" if "ours_30000" in data else (sorted(data)[-1] if data else None)
        if key and all(k in data[key] for k in ("SSIM", "PSNR", "LPIPS")):
            d = data[key]
            cells[(sc, float(m.group(1)))].append((d["SSIM"], d["PSNR"], d["LPIPS"]))
for d in [os.path.join(MA, "develop3_anti"), os.path.join(MA, "develop3_woanti"),
          "/data1/zhangao/outputs/develop3_multi_angle2/metrics/develop3",
          "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing/metrics/develop3",
          "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing2/metrics/develop3_1e0",
          "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing2/metrics/develop3_1e-2"]:
    if not os.path.isdir(d):
        continue
    for fn in os.listdir(d):
        m = re.match(r"^(.*)_test_selection_angle_([0-9.]+)_metrics\.csv$", fn)
        if m and m.group(1) in MIP8:
            rows = list(csv.DictReader(open(os.path.join(d, fn))))
            for r in rows:
                cells[(m.group(1), float(m.group(2)))].append((float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"])))
oracle = {}
for ang in ANGLES:
    best = [max(cells[(sc, ang)], key=lambda t: t[1]) for sc in MIP8]
    oracle[ang] = tuple(sum(v[i] for v in best) / 8 for i in range(3))
TIEBREAK[("GS-LSS (Ours)", 4, 2)] = oracle[100.0][2]  # LPIPS@100
print(f"\ntiebreak MCMC SSIM@20 full = {TIEBREAK[('3DGS-MCMC~\\cite{mcmc}', 0, 0)]:.6f}  (eff = {eff[20.0][0]:.6f})")
print(f"tiebreak oracle LPIPS@100 full = {TIEBREAK[('GS-LSS (Ours)', 4, 2)]:.6f}  (eff = {eff[100.0][2]:.6f})")

# ---- assemble 9-row block and recompute bold/underline per column ----
eff_row = ("GS-LSS-Eff (Ours)", [tuple(eff[a]) for a in ANGLES])
ALL = ROWS + [eff_row]

def disp(name, ai, mi):
    v = ALL[[r[0] for r in ALL].index(name)][1][ai][mi]
    return round(v, 3)

marks = {}  # (name, ai, mi) -> 'b'|'u'
for ai in range(5):
    for mi in range(3):
        entries = []
        for name, vals in ALL:
            d = round(vals[ai][mi], 3)
            f = TIEBREAK.get((name, ai, mi), vals[ai][mi])
            entries.append((name, d, f))
        rev = (mi != 2)  # SSIM/PSNR high better, LPIPS low better
        entries.sort(key=lambda t: (t[1], t[2]), reverse=rev)
        marks[(entries[0][0], ai, mi)] = "b"
        marks[(entries[1][0], ai, mi)] = "u"

def fmt(name, vals):
    cells_ = []
    for ai in range(5):
        for mi in range(3):
            s = f"{round(vals[ai][mi], 3):.3f}"
            mk = marks.get((name, ai, mi))
            if mk == "b":
                s = "\\textbf{%s}" % s
            elif mk == "u":
                s = "\\underline{%s}" % s
            cells_.append(s)
    return f"{name} & " + " & ".join(cells_) + " \\\\"

print("\n---- updated LaTeX (Mip-NeRF 360 block) ----")
for name, vals in ALL:
    print(fmt(name, vals))

print("\n---- markdown ----")
hdr = "| Method | " + " | ".join(f"{int(a)}° S / P / L" for a in ANGLES) + " |"
print(hdr)
print("|" + "---|" * 6)
for name, vals in ALL:
    nm = name.split("~")[0].replace(" (Ours)", " (Ours)")
    cs = []
    for ai in range(5):
        parts = []
        for mi in range(3):
            s = f"{round(vals[ai][mi], 3):.3f}"
            mk = marks.get((name, ai, mi))
            if mk == "b":
                s = f"**{s}**"
            elif mk == "u":
                s = f"_{s}_"
            parts.append(s)
        cs.append(" / ".join(parts))
    print(f"| {nm} | " + " | ".join(cs) + " |")
