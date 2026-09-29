#!/usr/bin/env python3
"""Free-space floater-mass fraction (Table-1 approach #2), exact paper/global_floaters.py definition:
a Gaussian is a floater if its center is > tau from the FULL-coverage COLMAP SfM surface (input.ply,
identical across splits -> correct gap-FILLING is NOT flagged, only true free-space floaters).
floater-mass fraction = sum(opacity of floaters) / sum(opacity all)   [opacity = sigmoid].
Usage: python tools/floater_metric.py <ref_input.ply> <method_map.json> [tau_frac=0.015] [ours_key]
"""
import sys, json
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

def read_xyz(p):
    v = PlyData.read(p)["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)

def read_xyz_op(p):
    v = PlyData.read(p)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    return xyz, op

ref_ply, mapj = sys.argv[1], sys.argv[2]
tau_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.015
ours_key = sys.argv[4] if len(sys.argv) > 4 else "GS-LSS(ours)"
methods = json.load(open(mapj))

ref = read_xyz(ref_ply)
lo, hi = np.percentile(ref, 1, 0), np.percentile(ref, 99, 0)
diag = float(np.linalg.norm(hi - lo)); tau = tau_frac * diag
tree = cKDTree(ref)
print(f"[ref] {len(ref):,} SfM pts  diag={diag:.2f}  tau={tau:.3f} ({tau_frac*100:.1f}% diag)\n")

res = {}
for m, p in methods.items():
    try:
        xyz, op = read_xyz_op(p)
    except Exception as e:
        print(f"  {m:16} MISSING ({e})"); continue
    d, _ = tree.query(xyz, k=1, workers=-1)
    fl = d > tau
    res[m] = (float(op[fl].sum() / (op.sum() + 1e-9)) * 100, int(fl.sum()), len(xyz))

for m in sorted(res, key=lambda k: res[k][0]):
    f, nf, n = res[m]
    star = "  <-- ours" if m == ours_key else ""
    print(f"  {m:16} floater-mass {f:6.2f}%   ({nf:,}/{n:,} gauss){star}")

if ours_key in res:
    o = res[ours_key][0]
    base = {m: v[0] for m, v in res.items() if m != ours_key}
    if base:
        bb = min(base, key=base.get)             # best (lowest-floater) baseline
        print(f"\n[GAP] ours {o:.2f}% vs best baseline {bb} {base[bb]:.2f}%  "
              f"=> {base[bb]-o:+.2f} pts abs, {base[bb]/max(o,1e-9):.2f}x relative")
