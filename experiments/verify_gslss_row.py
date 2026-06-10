#!/usr/bin/env python3
"""Try to reproduce the paper's GS-LSS Mip-NeRF 360 coverage-gap row from all
develop3_multi_angle* runs on disk (per-(scene,angle) best over configs)."""
import json, os, re, glob
from collections import defaultdict

ROOTS = [
    "/data1/zhangao/outputs/develop3_multi_angle",
    "/data1/zhangao/outputs/develop3_multi_angle2",
    "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing",
    "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing2",
]
MIP8 = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump", "treehill"]
ANGLES = [20.0, 40.0, 60.0, 80.0, 100.0]
PAPER = {20: (0.803, 25.646, 0.193), 40: (0.745, 22.823, 0.241), 60: (0.699, 21.406, 0.292),
         80: (0.660, 20.048, 0.322), 100: (0.617, 18.555, 0.352)}

cells = defaultdict(list)  # (scene, angle) -> list of (ssim, psnr, lpips, src)
n_json = 0
for root in ROOTS:
    for sc_dir in sorted(glob.glob(os.path.join(root, "*"))):
        sc = os.path.basename(sc_dir)
        if sc not in MIP8 + ["flowers", "train_resize", "truck_resize"]:
            continue
        for rj in glob.glob(os.path.join(sc_dir, "*", "results.json")):
            run = os.path.basename(os.path.dirname(rj))
            m = re.search(r"test_selection_angle_([0-9.]+)", run)
            if not m:
                continue
            ang = float(m.group(1))
            try:
                with open(rj) as f:
                    data = json.load(f)
            except Exception:
                continue
            key = "ours_30000" if "ours_30000" in data else (sorted(data.keys())[-1] if data else None)
            if not key:
                continue
            d = data[key]
            if not all(k in d for k in ("SSIM", "PSNR", "LPIPS")):
                continue
            cells[(sc, ang)].append((d["SSIM"], d["PSNR"], d["LPIPS"], f"{os.path.basename(root)}/{run}"))
            n_json += 1

print(f"total results.json parsed: {n_json}")

# union with all known develop3 metric CSV collections
import csv as _csv
CSV_DIRS = [
    "/data1/zhangao/repos/3dgs-imc-codex/experiments/metrics_multiangle/develop3_anti",
    "/data1/zhangao/repos/3dgs-imc-codex/experiments/metrics_multiangle/develop3_woanti",
    "/data1/zhangao/outputs/develop3_multi_angle2/metrics/develop3",
    "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing/metrics/develop3",
    "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing2/metrics/develop3_1e0",
    "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing2/metrics/develop3_1e-2",
]
n_csv = 0
pat = re.compile(r"^(.*)_test_selection_angle_([0-9.]+)_metrics\.csv$")
for d in CSV_DIRS:
    if not os.path.isdir(d):
        continue
    for fn in os.listdir(d):
        m = pat.match(fn)
        if not m:
            continue
        sc, ang = m.group(1), float(m.group(2))
        try:
            with open(os.path.join(d, fn)) as f:
                rows = list(_csv.DictReader(f))
        except Exception:
            continue
        for r in rows:
            try:
                cells[(sc, ang)].append((float(r["SSIM"]), float(r["PSNR"]), float(r["LPIPS"]),
                                         f"csv:{os.path.basename(d)}/{fn}"))
                n_csv += 1
            except Exception:
                pass
print(f"csv rows added: {n_csv}")
print(f"distinct (scene,angle) cells: {len(cells)}")

# de-duplicate identical metric tuples per cell
for k in cells:
    seen, uniq = set(), []
    for v in cells[k]:
        t = (round(v[0], 6), round(v[1], 5), round(v[2], 6))
        if t not in seen:
            seen.add(t)
            uniq.append(v)
    cells[k] = uniq

print(f"\n{'angle':>5} | {'paper':>21} | {'per-cell BEST-PSNR avg':>23} | {'dPSNR':>7} | n_scenes | configs/cell(min-max)")
for ang in ANGLES:
    best = {}
    counts = []
    for sc in MIP8:
        cands = cells.get((sc, ang), [])
        if cands:
            best[sc] = max(cands, key=lambda t: t[1])
            counts.append(len(cands))
    if not best:
        print(f"{ang:>5} | no data")
        continue
    n = len(best)
    avg = tuple(sum(v[i] for v in best.values()) / n for i in range(3))
    p = PAPER[int(ang)]
    # per-metric independent best (max SSIM, max PSNR, min LPIPS per cell)
    pm = [0.0, 0.0, 0.0]
    for sc in best:
        cands = cells[(sc, ang)]
        pm[0] += max(c[0] for c in cands)
        pm[1] += max(c[1] for c in cands)
        pm[2] += min(c[2] for c in cands)
    pm = tuple(x / n for x in pm)
    print(f"{ang:>5} | {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f} | {avg[0]:.3f}/{avg[1]:.3f}/{avg[2]:.3f}   | {avg[1]-p[1]:+0.3f} | {n:>8} | {min(counts)}-{max(counts)}   per-metric-best: {pm[0]:.3f}/{pm[1]:.3f}/{pm[2]:.3f} (dP {pm[1]-p[1]:+0.3f})")

# per-scene detail at 60 and 100 (where gap was largest)
for ang in (60.0, 100.0):
    print(f"\n--- per-scene best @ {ang} ---")
    for sc in MIP8:
        cands = cells.get((sc, ang), [])
        if not cands:
            print(f"{sc:>10}: MISSING")
            continue
        b = max(cands, key=lambda t: t[1])
        print(f"{sc:>10}: PSNR {b[1]:7.3f}  SSIM {b[0]:.3f}  LPIPS {b[2]:.3f}  ({len(cands)} cfgs)  {b[3][:90]}")
