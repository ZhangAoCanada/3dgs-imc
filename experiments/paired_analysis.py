#!/usr/bin/env python3
"""Live aggregator + paired-comparison analysis for the seed-based speedup.

Reads all diag metrics (server + synced WSL), prints:
  1. SEED CALIBRATION: residual paired noise (same-seed repeats) vs the independent-run spread.
     If residual << independent spread, paired comparison is valid -> 1 pair replaces a seed study.
  2. PAIRED MH EFFECTS: MH-on minus MH-off at the same seed, per cell.
"""
import csv, glob, os, re, sys

DIRS = [
    "/data1/zhangao/outputs/diag/mipnerf360",
    "/data1/zhangao/repos/3dgs-imc-codex/outputs/wsl_results/diag/mipnerf360",  # synced WSL (if present)
]

def psnr(tag, it=15000):
    for d in DIRS:
        f = os.path.join(d, f"{tag}_i{it}", "metrics.csv")
        if os.path.isfile(f):
            rows = list(csv.DictReader(open(f)))
            for r in rows:
                if r.get("method") == f"ours_{it}":
                    return float(r["PSNR"])
            if rows:
                return float(rows[-1]["PSNR"])
    return None

def show(label, tags):
    vals = [(t, psnr(f"room_angle_100.0_{t}")) for t in tags]
    vals = [(t, v) for t, v in vals if v is not None]
    if not vals:
        print(f"  {label}: (pending)")
        return None
    ps = [v for _, v in vals]
    spread = max(ps) - min(ps)
    print(f"  {label}: " + ", ".join(f"{t.split('_')[-1]}={v:.3f}" for t, v in vals) +
          f"   spread={spread:.3f}")
    return ps

print("=== INDEPENDENT-RUN SPREAD (no seed, room@100, the variance we are fighting) ===")
indep = []
for d in DIRS:
    for f in glob.glob(os.path.join(d, "room_angle_100.0_*_i15000", "metrics.csv")):
        tag = os.path.basename(os.path.dirname(f))
        if any(x in tag for x in ["cal_", "_s1", "_s2", "_s3", "_s4", "_s5", "mh15k", "mhdens", "aggr", "nv3_mh", "p_rm100"]):
            rows = list(csv.DictReader(open(f)))
            if rows:
                try:
                    indep.append((tag, float(rows[-1]["PSNR"])))
                except Exception:
                    pass
seen = set(); uind = []
for t, v in indep:
    k = round(v, 3)
    if k not in seen:
        seen.add(k); uind.append((t, v))
if uind:
    ps = [v for _, v in uind]
    print(f"  n={len(ps)} runs, range {min(ps):.3f}..{max(ps):.3f}  SPREAD={max(ps)-min(ps):.3f} dB")

print("\n=== SEED CALIBRATION ===")
off = show("MH-off repeats SEED=42 (SERVER)", ["cal_off_a", "cal_off_b"])
on = show("MH-on  repeats SEED=42 (SERVER)", ["cal_on_a", "cal_on_b"])

print("\n=== RESIDUAL NOISE FLOOR (same config, same seed) ===")
if off and len(off) == 2:
    r = abs(off[0] - off[1]); print(f"  MH-off SEED=42 off_a vs off_b: {r:.3f} dB")
if on and len(on) == 2:
    r = abs(on[0] - on[1]); print(f"  MH-on  SEED=42 on_a vs on_b:   {r:.3f} dB (tests isolated-MH-generator pairing)")

print("\n=== PAIRED MH EFFECT per seed (on - off, same seed = clean treatment effect) ===")
effects = []
# SEED=42 from server
o42 = psnr("room_angle_100.0_cal_off_a"); n42 = psnr("room_angle_100.0_cal_on_a")
if o42 and n42:
    effects.append(("42(server)", n42 - o42)); print(f"  SEED=42: {n42 - o42:+.3f}  (on {n42:.3f} - off {o42:.3f})")
# SEED=42 from WSL
ow = psnr("room_angle_100.0_cal_wsl_off_a"); nw = psnr("room_angle_100.0_cal_wsl_on_a")
if ow and nw:
    effects.append(("42(wsl)", nw - ow)); print(f"  SEED=42 (WSL): {nw - ow:+.3f}  (on {nw:.3f} - off {ow:.3f})")
# SEED=7, SEED=123 from WSL
for s in ("7", "123"):
    o = psnr(f"room_angle_100.0_cal_s{s}_off"); n = psnr(f"room_angle_100.0_cal_s{s}_on")
    if o and n:
        effects.append((f"{s}", n - o)); print(f"  SEED={s}: {n - o:+.3f}  (on {n:.3f} - off {o:.3f})")

print("\n=== VERDICT ===")
if off and len(off) == 2:
    r = abs(off[0] - off[1])
    if r < 0.15:
        print(f"  SEED WORKS: residual {r:.3f} << independent spread 1.03. Paired comparison VALID -> 1 pair replaces a 10-seed study.")
    else:
        print(f"  residual {r:.3f} larger than hoped; pairing partial. Use 2 pairs for borderline cells.")
if len(effects) >= 2:
    es = [e for _, e in effects]
    mean_e = sum(es) / len(es)
    print(f"  MH effect across {len(es)} seed-pairs: mean {mean_e:+.3f}, range {min(es):+.3f}..{max(es):+.3f}")
    if min(es) > 0.1:
        print(f"  -> MH ROBUSTLY HELPS room@100 (all seed-pairs positive). The +1.14 single-draw was inflated; true ~{mean_e:+.2f}.")
    elif max(es) < 0.1:
        print("  -> MH does NOT help room@100 (all pairs ~0/negative). Earlier 'gains' were lucky draws.")
    else:
        print("  -> MIXED across seeds: MH effect on room@100 is marginal/inconsistent.")
