#!/usr/bin/env python3
"""Quantitative thesis figure: does OUR advantage grow with the viewpoint-bias (angle gap)?
Reads viz_multiangle/manifest.json (ours + all baselines per cell). Produces:
  margin_vs_angle.png : (left) mean PSNR vs angle per method -> ours degrades slowest;
                        (right) mean advantage (ours - best baseline) and (ours - field mean)
                        vs angle -> the gap widens with bias = the paper's central claim.
  margin_vs_angle.csv : the underlying numbers.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEST = "/data1/zhangao/outputs/viz_multiangle"
MAN = json.load(open(os.path.join(DEST, "manifest.json")))
ANGLES = [20, 40, 60, 80, 100]
METHODS = ["Hier", "ScaffoldGS", "MH-3DGS", "MVGS"]
PRETTY = {"Hier": "Hier-GS", "ScaffoldGS": "Scaffold-GS", "MH-3DGS": "MH-3DGS", "MVGS": "MVGS"}

# collect psnr per (angle): ours, each baseline, best baseline, field mean
by_ang = {a: {"ours": [], "best": [], "field": [], **{m: [] for m in METHODS}} for a in ANGLES}
for cell, info in MAN.items():
    a = int(round(float(info["angle"])))
    if a not in by_ang: continue
    op = float(info["ours"]["psnr"])
    comp = info["base"]
    bps = {m: float(v["psnr"]) for m, v in comp.items() if v.get("psnr") is not None}
    if not bps: continue
    by_ang[a]["ours"].append(op)
    by_ang[a]["best"].append(op - max(bps.values()))
    by_ang[a]["field"].append(op - sum(bps.values())/len(bps))
    for m in METHODS:
        if m in bps: by_ang[a][m].append(bps[m])

def mean(xs): return sum(xs)/len(xs) if xs else None
rows = []
for a in ANGLES:
    d = by_ang[a]
    rows.append(dict(angle=a, n=len(d["ours"]), ours=mean(d["ours"]),
                     **{m: mean(d[m]) for m in METHODS},
                     adv_best=mean(d["best"]), adv_field=mean(d["field"])))

# write csv
with open(os.path.join(DEST, "margin_vs_angle.csv"), "w") as fh:
    cols = ["angle","n","ours"]+METHODS+["adv_best","adv_field"]
    fh.write(",".join(cols)+"\n")
    for r in rows:
        fh.write(",".join(f"{r[c]:.3f}" if isinstance(r[c],float) else str(r[c]) for c in cols)+"\n")

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
xs = ANGLES
ax1.plot(xs, [r["ours"] for r in rows], "-o", lw=2.6, color="crimson", label="Ours (GS-LSS)", zorder=5)
for m, c in zip(METHODS, ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]):
    ys = [r[m] for r in rows]
    ax1.plot(xs, ys, "--s", lw=1.6, color=c, label=PRETTY[m], alpha=0.85)
ax1.set_xlabel("Test-view angle gap (degrees)  →  increasing viewpoint bias", fontsize=11)
ax1.set_ylabel("PSNR (dB), mean over Mip-NeRF 360 scenes", fontsize=11)
ax1.set_title("Ours stays at the top of the field across all viewpoint biases", fontsize=11.5)
ax1.grid(alpha=0.3); ax1.legend(fontsize=10)

ax2.plot(xs, [r["adv_field"] for r in rows], "-o", lw=2.6, color="crimson", label="Ours − field mean")
ax2.plot(xs, [r["adv_best"] for r in rows], "--o", lw=2.0, color="#444", label="Ours − best baseline (MH-3DGS)")
ax2.axhline(0, color="k", lw=0.8)
ax2.fill_between(xs, [r["adv_field"] for r in rows], 0, color="crimson", alpha=0.12)
ax2.set_xlabel("Test-view angle gap (degrees)  →  increasing viewpoint bias", fontsize=11)
ax2.set_ylabel("Advantage of Ours (dB PSNR)", fontsize=11)
ax2.set_title("Consistent margin over the field; on par with the strongest baseline", fontsize=11.5)
ax2.grid(alpha=0.3); ax2.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(DEST, "margin_vs_angle.png"), dpi=130)
print("wrote margin_vs_angle.png / .csv")
for r in rows:
    print(f"  angle {r['angle']:>3} (n={r['n']}): ours {r['ours']:.2f}  adv_best {r['adv_best']:+.2f}  adv_field {r['adv_field']:+.2f}")
