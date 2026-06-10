#!/usr/bin/env python3
"""Verify experiments/metrics_multiangle against paper Table tab:coverage_gap.

Recomputes per-(method,dataset,angle) scene-averages of SSIM/PSNR/LPIPS from the
raw metric CSVs and diffs them against the numbers printed in
writings/GS-LSS/bare_jrnl_new_sample4.tex.
"""
import csv, os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
MA = os.path.join(ROOT, "metrics_multiangle")

ANGLES = [20.0, 40.0, 60.0, 80.0, 100.0]
MIP8 = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "treehill"]
MIP9 = MIP8 + ["flowers"]
TNT = ["train_resize", "truck_resize"]
INDOOR = {"bonsai", "counter", "kitchen", "room"}

# ---------------- paper values: {method: {dataset: {angle: (ssim, psnr, lpips)}}} ----------------
PAPER = {
    "3DGS": {
        "mip": {20: (0.787, 24.936, 0.202), 40: (0.726, 22.137, 0.249), 60: (0.674, 20.499, 0.291), 80: (0.630, 19.460, 0.324), 100: (0.584, 17.597, 0.358)},
        "tnt": {20: (0.771, 20.411, 0.209), 40: (0.741, 19.464, 0.234), 60: (0.695, 18.651, 0.269), 80: (0.663, 17.945, 0.294), 100: (0.632, 17.422, 0.323)},
    },
    "Scaffold-GS": {
        "mip": {20: (0.760, 23.844, 0.231), 40: (0.706, 21.835, 0.273), 60: (0.653, 20.418, 0.315), 80: (0.616, 19.574, 0.344), 100: (0.571, 17.751, 0.378)},
        "tnt": {20: (0.744, 19.331, 0.222), 40: (0.743, 19.945, 0.232), 60: (0.692, 18.658, 0.269), 80: (0.653, 17.589, 0.301), 100: (0.615, 17.082, 0.334)},
    },
    "Hierarchical-GS": {
        "mip": {20: (0.787, 25.044, 0.203), 40: (0.725, 22.314, 0.250), 60: (0.671, 20.611, 0.291), 80: (0.624, 19.339, 0.326), 100: (0.581, 17.463, 0.359)},
        "tnt": {20: (0.766, 20.418, 0.208), 40: (0.743, 19.640, 0.229), 60: (0.694, 18.728, 0.267), 80: (0.657, 17.811, 0.296), 100: (0.630, 17.226, 0.321)},
    },
    "MVGS": {
        "mip": {20: (0.784, 24.827, 0.203), 40: (0.715, 22.130, 0.256), 60: (0.658, 20.458, 0.303), 80: (0.608, 19.081, 0.340), 100: (0.561, 17.164, 0.374)},
        "tnt": {20: (0.756, 20.211, 0.199), 40: (0.721, 19.150, 0.229), 60: (0.668, 18.323, 0.275), 80: (0.625, 17.549, 0.309), 100: (0.590, 16.875, 0.340)},
    },
    "PGSR": {
        "mip": {20: (0.771, 24.203, 0.231), 40: (0.706, 21.588, 0.280), 60: (0.662, 20.179, 0.307), 80: (0.611, 18.834, 0.343), 100: (0.553, 16.808, 0.375)},
        "tnt": {20: (0.765, 20.119, 0.234), 40: (0.737, 19.182, 0.258), 60: (0.687, 18.020, 0.295), 80: (0.653, 17.305, 0.313), 100: (0.634, 17.206, 0.338)},
    },
    "3DGS-MCMC": {
        "mip": {20: (0.801, 25.323, 0.194), 40: (0.739, 22.369, 0.239), 60: (0.679, 20.732, 0.283), 80: (0.632, 19.344, 0.319), 100: (0.572, 17.204, 0.355)},
        "tnt": {20: (0.782, 21.026, 0.202), 40: (0.754, 20.018, 0.220), 60: (0.702, 18.973, 0.257), 80: (0.665, 18.014, 0.285), 100: (0.632, 17.406, 0.317)},
    },
    "MH-3DGS": {
        "mip": {20: (0.784, 25.122, 0.224), 40: (0.728, 22.365, 0.265), 60: (0.683, 20.855, 0.300), 80: (0.650, 20.008, 0.328), 100: (0.608, 18.352, 0.360)},
        "tnt": {20: (0.754, 19.967, 0.221), 40: (0.726, 19.205, 0.241), 60: (0.649, 17.404, 0.291), 80: (0.652, 17.299, 0.298), 100: (0.612, 16.627, 0.331)},
    },
    "GS-LSS": {
        "mip": {20: (0.803, 25.646, 0.193), 40: (0.745, 22.823, 0.241), 60: (0.699, 21.406, 0.292), 80: (0.660, 20.048, 0.322), 100: (0.617, 18.555, 0.352)},
        "tnt": {20: (0.786, 21.451, 0.191), 40: (0.754, 20.253, 0.211), 60: (0.708, 19.345, 0.247), 80: (0.669, 18.113, 0.276), 100: (0.641, 17.777, 0.304)},
    },
}

# ---------------- loaders ----------------
def read_metric_csv(path):
    """Return (ssim, psnr, lpips) from one metrics csv (last data row)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    r = rows[-1]
    return (float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"]))

def load_flat(method_dir, suffix="_metrics.csv"):
    """Standard layout: <scene>_test_selection_angle_<a><suffix>."""
    out = {}  # (scene, angle) -> (ssim,psnr,lpips)
    pat = re.compile(r"^(.*)_test_selection_angle_([0-9.]+)" + re.escape(suffix) + r"$")
    d = os.path.join(MA, method_dir)
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        m = pat.match(fn)
        if not m:
            continue
        sc, ang = m.group(1), float(m.group(2))
        v = read_metric_csv(os.path.join(d, fn))
        if v:
            out[(sc, ang)] = v
    return out

def load_3dgs_depth():
    """Layout: 3dgs-depth-test_selection_angle_<a>/<scene>_metrics.csv"""
    out = {}
    base = os.path.join(MA, "3dgs-depth")
    for sub in os.listdir(base):
        m = re.match(r"^3dgs-depth-test_selection_angle_([0-9.]+)$", sub)
        if not m:
            continue
        ang = float(m.group(1))
        subdir = os.path.join(base, sub)
        for fn in os.listdir(subdir):
            m2 = re.match(r"^(.*)_metrics\.csv$", fn)
            if not m2:
                continue
            v = read_metric_csv(os.path.join(subdir, fn))
            if v:
                out[(m2.group(1), ang)] = v
    return out

METHODS = {
    "3dgs-original": load_flat("3dgs-original"),
    "scaffoldgs-original": load_flat("scaffoldgs-original"),
    "cdgs-original": load_flat("cdgs-original"),
    "3dgs-depth": load_3dgs_depth(),
    "mvgs-original-mv3": load_flat("mvgs-original-mv3"),
    "pgsr2-original": load_flat("pgsr2-original", suffix=".csv"),
    "3dgs-mcmc-original": load_flat("3dgs-mcmc-original"),
    "mh-3dgs-original": load_flat("mh-3dgs-original"),
    "develop3_anti": load_flat("develop3_anti"),
    "develop3_woanti": load_flat("develop3_woanti"),
}

# GS-LSS candidate: AA indoor / no-AA outdoor mix
mix = {}
for (sc, ang), v in METHODS["develop3_woanti"].items():
    mix[(sc, ang)] = v
for (sc, ang), v in METHODS["develop3_anti"].items():
    if sc in INDOOR:
        mix[(sc, ang)] = v
METHODS["develop3_mix(aa-indoor)"] = mix

# per-cell best of anti/woanti (PSNR-based)
best = {}
for key in set(METHODS["develop3_anti"]) | set(METHODS["develop3_woanti"]):
    cands = [METHODS[m][key] for m in ("develop3_anti", "develop3_woanti") if key in METHODS[m]]
    best[key] = max(cands, key=lambda t: t[1])
METHODS["develop3_best(anti|woanti)"] = best

def avg(data, scenes, ang):
    vals = [data[(sc, ang)] for sc in scenes if (sc, ang) in data]
    if not vals:
        return None, 0
    n = len(vals)
    return tuple(sum(v[i] for v in vals) / n for i in range(3)), n

def fmt(v):
    return "%.3f/%.3f/%.3f" % v if v else "  (missing)  "

def show(paper_name, method_keys, dataset, scenes):
    print(f"\n### paper row: {paper_name}  [{dataset}: {len(scenes)} scenes] ###")
    print(f"{'angle':>5} | {'paper (S/P/L)':>21} | " + " | ".join(f"{k:>34}" for k in method_keys))
    for ang in ANGLES:
        p = PAPER[paper_name][dataset][int(ang)]
        cells = []
        for k in method_keys:
            v, n = avg(METHODS[k], scenes, ang)
            if v is None:
                cells.append(f"{'missing':>34}")
            else:
                d = max(abs(v[i] - p[i]) for i in range(3))
                flag = "==" if d < 0.0006 else ("~" if d < 0.006 else "XX")
                cells.append(f"{fmt(v)} n={n} {flag:>2} dP={v[1]-p[1]:+0.3f}"[:40].rjust(34))
        print(f"{ang:>5} | {fmt(p):>21} | " + " | ".join(cells))

ROWMAP = [
    ("3DGS", ["3dgs-original"]),
    ("Scaffold-GS", ["scaffoldgs-original"]),
    ("Hierarchical-GS", ["cdgs-original", "3dgs-depth"]),
    ("MVGS", ["mvgs-original-mv3"]),
    ("PGSR", ["pgsr2-original"]),
    ("3DGS-MCMC", ["3dgs-mcmc-original"]),
    ("MH-3DGS", ["mh-3dgs-original"]),
    ("GS-LSS", ["develop3_anti", "develop3_woanti", "develop3_mix(aa-indoor)", "develop3_best(anti|woanti)"]),
]

scene_set = MIP9 if "--mip9" in sys.argv else MIP8
label = "MIP9(with flowers)" if "--mip9" in sys.argv else "MIP8(no flowers)"
print(f"================ Mip-NeRF 360 [{label}] ================")
for paper_name, keys in ROWMAP:
    show(paper_name, keys, "mip", scene_set)

print("\n================ Tanks & Temples ================")
for paper_name, keys in ROWMAP:
    show(paper_name, keys, "tnt", TNT)
