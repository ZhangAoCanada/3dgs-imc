#!/usr/bin/env python3
"""Gap-drain forensics: compare WHERE the Gaussian opacity mass sits (train-visible vs
gap region) between the fast-pipeline room@60 model and an old slow-pipeline model
of the same cap. Visibility = projection into camera frustum (no occlusion; same
proxy for both models, so differences are comparable)."""
import json, sys
import numpy as np
from plyfile import PlyData

FAST = "/data1/zhangao/outputs/diag/mipnerf360/room_angle_60.0_noaa_nofloater_i30000"
OLD = "/data1/zhangao/outputs/develop3_multi_angle2/room/develop3_allviews_local_eps0.5_minsamples500_depth0.3_wholeonly_rangescale1.0_partial1e0_sigma1e-6_sigma-detach_nview3_1200000_test_selection_angle_60.0_0.001"
TESTLIST = "/data1/zhangao/DATA/mipnerf360/room/test_selection_angle_60.0.txt"

def load_ply(path, trained=True):
    ply = PlyData.read(path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    if trained:
        op = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    else:
        op = np.ones(len(xyz), dtype=np.float32)
    return xyz, op

def load_cams(model_dir):
    cams = json.load(open(f"{model_dir}/cameras.json"))
    test_names = {n.rsplit(".", 1)[0] for n in open(TESTLIST).read().split()}
    train, test = [], []
    for c in cams:
        R = np.array(c["rotation"], dtype=np.float32)       # cam-to-world
        pos = np.array(c["position"], dtype=np.float32)
        entry = (R, pos, c["fx"], c["fy"], c["width"], c["height"])
        (test if c["img_name"].rsplit(".", 1)[0] in test_names else train).append(entry)
    return train, test

def vis_counts(xyz, cams):
    n = np.zeros(len(xyz), dtype=np.int32)
    for R, pos, fx, fy, W, H in cams:
        Xc = (xyz - pos) @ R                                 # world -> cam (R is C2W, use right-mult by R = R^T @ x)
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * Xc[:, 0] / z + W / 2.0
            v = fy * Xc[:, 1] / z + H / 2.0
        n += ((z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).astype(np.int32)
    return n

def analyze(tag, xyz, op, train_cams, test_cams):
    nt = vis_counts(xyz, train_cams)
    ns = vis_counts(xyz, test_cams)
    total_mass = op.sum()
    test_rel = ns >= 1                                       # matters for test rendering
    print(f"\n==== {tag}: N={len(xyz):,}  opacity mass={total_mass:,.0f} ====")
    for T in (0, 2, 5):
        gap = test_rel & (nt <= T)
        print(f"  gap(test-vis & train-vis<={T}): count={gap.sum():>9,} ({100*gap.sum()/len(xyz):5.2f}%)"
              f"  mass={op[gap].sum():>10,.0f} ({100*op[gap].sum()/total_mass:5.2f}% of total)"
              f"  mean-op={op[gap].mean() if gap.any() else 0:.3f}")
    well = test_rel & (nt > 5)
    print(f"  test-vis & train-vis>5 :       count={well.sum():>9,}  mass={op[well].sum():>10,.0f}  mean-op={op[well].mean():.3f}")
    # train-visibility histogram of test-relevant opacity mass
    bins = [0, 1, 3, 6, 11, 21, 51, 1000]
    hist = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = test_rel & (nt >= lo) & (nt < hi)
        hist.append(op[m].sum())
    hist = np.array(hist)
    print(f"  test-relevant mass by train-vis bins {bins[:-1]}: " +
          " ".join(f"{100*h/hist.sum():.1f}%" for h in hist))

FAST_P2 = "/data1/zhangao/outputs/diag/mipnerf360/room_angle_60.0_aa_nofloater_i30000"
OLD_WOAA = "/data1/zhangao/outputs/develop3_multi_angle_woantialiasing2/room/develop3_allviews_local_eps0.5_minsamples500_depth0.3_wholeonly_rangescale1_partial1e0_sigma1e-6_sigma-detach_nview3_800000_test_selection_angle_60.0_0.001"

train_cams, test_cams = load_cams(FAST)
print(f"cams: train={len(train_cams)} test={len(test_cams)}")

MODELS = [
    ("FAST P1 AA=0 (15.61, 1.2M)", f"{FAST}/point_cloud/iteration_30000/point_cloud.ply", True),
    ("FAST P2 AA=1 (15.35, 1.2M)", f"{FAST_P2}/point_cloud/iteration_30000/point_cloud.ply", True),
    ("OLD anti AA=1 (17.27, 1.2M)", f"{OLD}/point_cloud/iteration_30000/point_cloud.ply", True),
    ("OLD woanti2 AA=0 (17.12, 800k)", f"{OLD_WOAA}/point_cloud/iteration_30000/point_cloud.ply", True),
    ("SFM input (reference)", f"{FAST}/input.ply", False),
]
for tag, path, trained in MODELS:
    try:
        xyz, op = load_ply(path, trained=trained)
    except Exception as e:
        print(f"\n==== {tag}: SKIP ({e})")
        continue
    analyze(tag, xyz, op, train_cams, test_cams)
