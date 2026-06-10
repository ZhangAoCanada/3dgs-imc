#!/usr/bin/env python3
"""Verify tab:discontinuous, tab:ablation_CPF (eps sweep) and tab:detailed_metrics
(8 trials) against experiments/metrics_discontinuous and metrics_ablation."""
import csv, os, re
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(ROOT, "metrics_discontinuous")
MA = os.path.join(ROOT, "metrics_ablation")

MIP9 = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"]
MIP8 = [s for s in MIP9 if s != "flowers"]
TNT = ["train_resize", "truck_resize"]
MILL = ["building-small", "residence-small", "rubble-small", "sci-art-small"]

PAPER_DISC = {  # method -> {dataset: (ssim,psnr,lpips)}
    "3DGS":            {"mip": (0.826, 26.979, 0.169), "tnt": (0.778, 20.914, 0.216), "mill": (0.754, 21.503, 0.217)},
    "Scaffold-GS":     {"mip": (0.783, 25.356, 0.205), "tnt": (0.745, 19.496, 0.237), "mill": (0.755, 21.081, 0.226)},
    "Hierarchical-GS": {"mip": (0.827, 27.046, 0.168), "tnt": (0.787, 21.521, 0.210), "mill": (0.762, 22.263, 0.210)},
    "MVGS":            {"mip": (0.828, 27.288, 0.164), "tnt": (0.776, 20.744, 0.196), "mill": (0.746, 21.311, 0.227)},
    "PGSR":            {"mip": (0.821, 26.269, 0.188), "tnt": (0.786, 21.093, 0.231), "mill": (0.717, 20.336, 0.265)},
    "3DGS-MCMC":       {"mip": (0.843, 27.517, 0.153), "tnt": (0.799, 21.955, 0.196), "mill": (0.766, 22.532, 0.207)},
    "MH-3DGS":         {"mip": (0.817, 26.895, 0.191), "tnt": (0.767, 20.895, 0.218), "mill": (0.747, 22.394, 0.249)},
    "GS-LSS":          {"mip": (0.844, 27.960, 0.152), "tnt": (0.802, 22.160, 0.187), "mill": (0.775, 22.893, 0.198)},
}

def read_rows(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []

def load_dir(d):
    """scene -> (ssim,psnr,lpips); scene = filename prefix."""
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        if not fn.endswith(".csv"):
            continue
        sc = re.sub(r"(_discontinuous)?(_metrics)?\.csv$", "", fn)
        rows = read_rows(os.path.join(d, fn))
        if rows:
            r = rows[-1]
            out[sc] = (float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"]))
    return out

def load_3dgs_depth_disc():
    out = {}
    base = os.path.join(MD, "3dgs-depth")
    if not os.path.isdir(base):
        return out
    # could be flat or nested; handle both
    for entry in os.listdir(base):
        p = os.path.join(base, entry)
        if os.path.isdir(p):
            for fn in os.listdir(p):
                if fn.endswith(".csv"):
                    sc = re.sub(r"(_discontinuous)?(_metrics)?\.csv$", "", fn)
                    rows = read_rows(os.path.join(p, fn))
                    if rows:
                        r = rows[-1]
                        out[sc] = (float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"]))
        elif entry.endswith(".csv"):
            sc = re.sub(r"(_discontinuous)?(_metrics)?\.csv$", "", entry)
            rows = read_rows(p)
            if rows:
                r = rows[-1]
                out[sc] = (float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"]))
    return out

METHODS = {
    "3dgs-original": load_dir(os.path.join(MD, "3dgs-original")),
    "scaffoldgs-original": load_dir(os.path.join(MD, "scaffoldgs-original")),
    "3dgs-depth": load_3dgs_depth_disc(),
    "cdgs-original": load_dir(os.path.join(MD, "cdgs-original")),
    "mvgs-original-mv3": load_dir(os.path.join(MD, "mvgs-original-mv3")),
    "pgsr-original": load_dir(os.path.join(MD, "pgsr-original")),
    "pgsr2-original": load_dir(os.path.join(MD, "pgsr2-original")),
    "3dgs-mcmc-original": load_dir(os.path.join(MD, "3dgs-mcmc-original")),
    "mh-3dgs-original": load_dir(os.path.join(MD, "mh-3dgs-original")),
    "develop3_antialiasing": load_dir(os.path.join(MD, "develop3_antialiasing")),
    "develop3_woantialiasing": load_dir(os.path.join(MD, "develop3_woantialiasing")),
}
best = {}
for sc in set(METHODS["develop3_antialiasing"]) | set(METHODS["develop3_woantialiasing"]):
    cands = [METHODS[m][sc] for m in ("develop3_antialiasing", "develop3_woantialiasing") if sc in METHODS[m]]
    best[sc] = max(cands, key=lambda t: t[1])
METHODS["develop3_best"] = best

def avg(data, scenes):
    vals = [data[sc] for sc in scenes if sc in data]
    if not vals:
        return None, 0
    n = len(vals)
    return tuple(sum(v[i] for v in vals) / n for i in range(3)), n

ROWMAP = [
    ("3DGS", ["3dgs-original"]),
    ("Scaffold-GS", ["scaffoldgs-original"]),
    ("Hierarchical-GS", ["3dgs-depth", "cdgs-original"]),
    ("MVGS", ["mvgs-original-mv3"]),
    ("PGSR", ["pgsr-original", "pgsr2-original"]),
    ("3DGS-MCMC", ["3dgs-mcmc-original"]),
    ("MH-3DGS", ["mh-3dgs-original"]),
    ("GS-LSS", ["develop3_antialiasing", "develop3_woantialiasing", "develop3_best"]),
]

print("================ tab:discontinuous ================")
for ds_name, scenes_opts in [("mip", [MIP9, MIP8]), ("tnt", [TNT]), ("mill", [MILL])]:
    print(f"\n--- dataset: {ds_name} ---")
    for paper_name, keys in ROWMAP:
        p = PAPER_DISC[paper_name][ds_name]
        line = f"{paper_name:>16} paper={p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f} || "
        for k in keys:
            for scenes in scenes_opts:
                v, n = avg(METHODS[k], scenes)
                if v is None:
                    line += f"{k}: missing | "
                    continue
                d = max(abs(v[i] - p[i]) for i in range(3))
                flag = "==" if d < 0.0006 else ("~" if d < 0.006 else "XX")
                line += f"{k}[n={n}]: {v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f} {flag} | "
        print(line)

# ---------------- ablation: eps sweep (tab:ablation_CPF), avg over train+truck @60 ----------------
PAPER_EPS = {0.1: (0.7058, 19.1690, 0.2458), 0.25: (0.7070, 19.3959, 0.2459), 0.5: (0.7058, 19.1871, 0.2485),
             0.75: (0.7033, 19.1725, 0.2488), 1.0: (0.7004, 18.9240, 0.2550)}
print("\n================ tab:ablation_CPF eps sweep (T&T@60, 4 decimals) ================")
for d in ["develop3_eps", "develop3_eps_partial1e0", "develop3_eps_partial1e0_2"]:
    print(f"--- {d} ---")
    base = os.path.join(MA, d)
    if not os.path.isdir(base):
        continue
    for eps in [0.1, 0.25, 0.5, 0.75, 1.0]:
        vals = []
        for sc in TNT:
            p_ = os.path.join(base, f"{sc}_{eps}_metrics.csv")
            rows = read_rows(p_)
            if rows:
                r = rows[-1]
                vals.append((float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"])))
        if len(vals) != 2:
            print(f"  eps={eps}: missing ({len(vals)})")
            continue
        v = tuple(sum(x[i] for x in vals) / 2 for i in range(3))
        p = PAPER_EPS[eps]
        d4 = max(abs(v[i] - p[i]) for i in range(3))
        flag = "==" if d4 < 0.00006 else ("~" if d4 < 0.005 else "XX")
        print(f"  eps={eps}: paper={p[0]:.4f}/{p[1]:.4f}/{p[2]:.4f}  got={v[0]:.4f}/{v[1]:.4f}/{v[2]:.4f}  {flag}")

# ---------------- trials (tab:detailed_metrics) ----------------
PAPER_TRIALS = [
    (0.7052, 19.3795, 0.2476), (0.7030, 19.1766, 0.2514), (0.7070, 19.3917, 0.2486),
    (0.7040, 19.3068, 0.2461), (0.7075, 19.3454, 0.2474), (0.7063, 19.1804, 0.2487),
    (0.7125, 19.5701, 0.2502), (0.7117, 19.5996, 0.2471),
]
print("\n================ tab:detailed_metrics 8 trials (T&T@60) ================")
cand_trials = []
for t in (1, 2, 3):
    vals = []
    for sc in TNT:
        rows = read_rows(os.path.join(MA, "develop3_trials", f"{sc}_trial{t}_metrics.csv"))
        if rows:
            r = rows[-1]
            vals.append((float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"])))
    if len(vals) == 2:
        cand_trials.append((f"develop3_trials/trial{t}", tuple(sum(x[i] for x in vals) / 2 for i in range(3))))
for t in (1, 2, 3, 4, 5):
    vals = []
    d = os.path.join(MA, "trials2", f"develop3_{t}")
    if os.path.isdir(d):
        for fn in os.listdir(d):
            rows = read_rows(os.path.join(d, fn))
            if rows:
                r = rows[-1]
                vals.append((float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"])))
    if len(vals) == 2:
        cand_trials.append((f"trials2/develop3_{t}", tuple(sum(x[i] for x in vals) / 2 for i in range(3))))

for name, v in cand_trials:
    hit = ""
    for i, p in enumerate(PAPER_TRIALS):
        if max(abs(v[j] - p[j]) for j in range(3)) < 0.00006:
            hit = f"== paper Trial {i+1}"
            break
    print(f"  {name}: {v[0]:.4f}/{v[1]:.4f}/{v[2]:.4f}  {hit or 'NO MATCH'}")
