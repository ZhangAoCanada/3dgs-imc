#!/usr/bin/env python3
"""Harvest Tables 5/6/7 from the tables567 output tree once runs complete.
T5 (ablation): per DBSCAN eps {0.1,0.25,0.5,0.75,1.0} + net {MLP=relu,SIREN=eps0.25}, avg over truck+train.
T6 (stability): 10 truck trials + mean + variance.
T7 (timing): ours = T7_ours_fast training_time.txt (server, FAST); baselines harvested separately.
Metrics from <run>/metrics.csv (ours_30000 row): SSIM,PSNR,LPIPS,BRISQUE,NIQE,PIQE."""
import csv, glob, os, sys, statistics

OUT = sys.argv[1] if len(sys.argv) > 1 else "/data1/zhangao/outputs/tables567"
TD = os.path.join(OUT, "tandt")
COLS = ["SSIM","PSNR","LPIPS","BRISQUE","NIQE","PIQE"]

def metrics(tag):
    """return dict of the ours_ (prefer ours_30000) row for a run tag, or None."""
    mp = os.path.join(TD, f"truck_resize_discontinuous_{tag}_i30000")
    # tag already encodes scene for T5; this helper takes a full run-dir basename stem
    return None

def read_run(scene, tag):
    d = os.path.join(TD, f"{scene}_discontinuous_{tag}_i30000")
    p = os.path.join(d, "metrics.csv")
    if not os.path.exists(p): return None
    rows = [r for r in csv.DictReader(open(p)) if r.get("method","").startswith("ours_")]
    if not rows: return None
    row = next((r for r in rows if r["method"]=="ours_30000"), rows[-1])
    try: return {c: float(row[c]) for c in COLS}
    except Exception: return None

def avg2(tag_truck, tag_train):
    a = read_run("truck_resize", tag_truck); b = read_run("train_resize", tag_train)
    if a is None or b is None: return None, (a is not None, b is not None)
    return {c: (a[c]+b[c])/2 for c in COLS}, (True, True)

def fmt(d, dec=(4,4,4,4,4,4)):
    return " & ".join(f"{d[c]:.{dec[i]}f}" for i,c in enumerate(COLS)) if d else "-- (incomplete)"

print("="*64); print("TABLE 5  (geometric field + CPF ablation, T&T discontinuous, avg truck+train)"); print("="*64)
print("-- DBSCAN eps --")
for e in ["0.1","0.25","0.5","0.75","1.0"]:
    d,_ = avg2(f"T5_eps{e}_truck", f"T5_eps{e}_train")
    print(f"  eps={e:<4}: {fmt(d)}")
print("-- CPF net --")
mlp,_ = avg2("T5_mlp_truck","T5_mlp_train")
siren,_ = avg2("T5_eps0.25_truck","T5_eps0.25_train")  # SIREN = eps0.25 default net
print(f"  MLP        : {fmt(mlp)}")
print(f"  SIREN-style: {fmt(siren)}")

print("\n"+"="*64); print("TABLE 6  (stability, 10 truck trials, T&T discontinuous)"); print("="*64)
trials = []
for i in range(1,11):
    d = read_run("truck_resize", f"T6_trial{i:02d}")
    if d: trials.append(d); print(f"  Trial {i:>2}: {fmt(d)}")
    else: print(f"  Trial {i:>2}: (pending)")
if trials:
    mean = {c: statistics.mean(t[c] for t in trials) for c in COLS}
    var  = {c: (statistics.pvariance(t[c] for t in trials) if len(trials)>1 else 0.0) for c in COLS}
    print(f"  Mean    : {fmt(mean)}")
    print(f"  Var     : " + " & ".join(f"{var[c]:.3e}" for c in COLS))

print("\n"+"="*64); print("TABLE 7  (training time, truck discontinuous)"); print("="*64)
def tt(path):
    try:
        s = open(path).read()
        import re; m = re.search(r"([0-9.]+)\s*seconds", s); return float(m.group(1)) if m else None
    except Exception: return None
ours_t = tt(os.path.join(TD, "truck_resize_discontinuous_T7_ours_fast_i30000", "training_time.txt"))
print(f"  GS-LSS (Ours, FAST/server): {ours_t/60:.2f} min" if ours_t else "  GS-LSS (Ours): pending")
print("  (baselines harvested from hust2: see harvest_t7_baselines)")
