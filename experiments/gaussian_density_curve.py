#!/usr/bin/env python3
"""Gap/observed Gaussian-density ratio vs coverage-gap angle (the single clearest density figure).
Reads <scene>__<method>.gap.json from gdens/. Deterministic methods drawn dashed (they starve the gap,
ratio<1), stochastic solid (they keep density even into the gap, ratio~1). Usage: gaussian_density_curve.py <scene>"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
GD = "/data1/zhangao/outputs/viz_multiangle/gdens"
scene = sys.argv[1] if len(sys.argv) > 1 else "truck_resize"
ANGLES = [20, 40, 60, 80, 100]
# (method, nice, color, is_stochastic)
SPEC = [("3DGS", "3DGS (det.)", "#444444", False),
        ("Hier", "Hier-GS (det.)", "#1f77b4", False),
        ("ScaffoldGS", "Scaffold-GS (det.)", "#e377c2", False),
        ("MVGS", "MVGS (det.)", "#ff7f0e", False),
        ("CDGS", "CDGS (det.)", "#17becf", False),
        ("PGSR", "PGSR (det.)", "#8c564b", False),
        ("MCMC", "3DGS-MCMC (stoch.)", "#2ca02c", True),
        ("MH-3DGS", "MH-3DGS (stoch.)", "#9467bd", True),
        ("Ours", "Ours (GS-LSS) (stoch.)", "#d62728", True)]
def gj(m):
    f = f"{GD}/{scene}__{m}.gap.json"
    if not os.path.exists(f): return None
    d = json.load(open(f)); return [d.get(f"{a}.0") or d.get(str(a)) for a in ANGLES]
plt.figure(figsize=(9, 6))
for m, nice, c, sto in SPEC:
    y = gj(m)
    if y is None or all(v is None for v in y): continue
    xs = [a for a, v in zip(ANGLES, y) if v is not None]; ys = [v for v in y if v is not None]
    plt.plot(xs, ys, marker="o", color=c, label=nice,
             lw=(3.2 if m == "Ours" else 2.0), ls=("-" if sto else "--"),
             zorder=(5 if m == "Ours" else 3), markersize=(7 if m == "Ours" else 5))
plt.axhline(1.0, color="gray", ls=":", lw=1); plt.text(21, 1.005, "even (gap=observed)", color="gray", fontsize=10)
plt.xlabel("Coverage-gap angle (deg)  →  stronger viewpoint bias", fontsize=12)
plt.ylabel("Gap / Observed Gaussian-density ratio", fontsize=12)
plt.title(f"Gap-region Gaussian density vs viewpoint bias — {scene.replace('_resize','').capitalize()} (T&T)\n"
          "Stochastic methods keep density even into the gap; deterministic methods starve it", fontsize=13)
plt.legend(ncol=2, fontsize=10, loc="upper left"); plt.grid(alpha=0.3); plt.ylim(0.3, 1.2)
plt.tight_layout()
out = f"{GD}/GAUSSIAN_DENSITY_curve_{scene.replace('_resize','')}.png"
plt.savefig(out, dpi=130); plt.close(); print("wrote", out)
