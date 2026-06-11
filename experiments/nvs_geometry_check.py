#!/usr/bin/env python3
"""Localize the NVS inb=0 bug with pure geometry (no GPU): project SfM points through
the synth-pose and source-camera math used by NVSupervisor and report overlap counts."""
import json, math, sys
sys.path.insert(0, "/data1/zhangao/repos/3dgs-imc-codex")
import numpy as np
from plyfile import PlyData
from scene_experimental_branch3.gap_poses import find_gap_sector, synth_gap_poses

MODEL = "/data1/zhangao/outputs/diag/mipnerf360/room_angle_100.0_rm100_nv3_rgbonly15k_i15000"
TESTLIST = "/data1/zhangao/DATA/mipnerf360/room/test_selection_angle_100.0.txt"

cams = json.load(open(f"{MODEL}/cameras.json"))
test_names = {n.rsplit(".", 1)[0] for n in open(TESTLIST).read().split()}
train = [c for c in cams if c["img_name"].rsplit(".", 1)[0] not in test_names]
centers = np.array([c["position"] for c in train])
looks = np.array([np.array(c["rotation"])[:, 2] for c in train])
sec = find_gap_sector(centers, looks)
print(f"train cams={len(train)} gap={math.degrees(sec['gap_width']):.1f}deg")

ply = PlyData.read(f"{MODEL}/input.ply")["vertex"]
P = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64)
print(f"sfm points: {len(P)}")

def project(pts, R_c2w, pos, fx, fy, W, H):
    Xc = (pts - pos) @ R_c2w           # = R_c2w.T @ x per row
    z = Xc[:, 2]
    u = Xc[:, 0] / np.clip(z, 1e-9, None) * fx + W / 2
    v = Xc[:, 1] / np.clip(z, 1e-9, None) * fy + H / 2
    inb = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return inb, z

ref = train[0]
W, H, fx, fy = ref["width"], ref["height"], ref["fx"], ref["fy"]
O, e1, e2 = sec["center"], sec["e1"], sec["e2"]
tr_az = sec["az"]

for i, (pos, R) in enumerate(synth_gap_poses(sec, num=4)):
    inb_s, _ = project(P, R, pos, fx, fy, W, H)
    a = math.atan2((pos - O) @ e2, (pos - O) @ e1)
    d = np.abs((tr_az - a + np.pi) % (2 * np.pi) - np.pi)
    src_idx = np.argsort(d)[:2]
    msg = f"pose{i}: sees {inb_s.sum():6d} sfm pts | "
    vis = P[inb_s]
    for k in src_idx:
        c = train[int(k)]
        inb_k, _ = project(vis, np.array(c["rotation"]), np.array(c["position"]),
                           c["fx"], c["fy"], c["width"], c["height"])
        msg += f"src{k}(az {math.degrees(d[k]):.0f}deg away): {inb_k.sum()}/{len(vis)} | "
    print(msg)

# sanity: a real train cam vs its neighbor (identity-style check)
c0, c1 = train[0], train[1]
inb0, _ = project(P, np.array(c0["rotation"]), np.array(c0["position"]), c0["fx"], c0["fy"], c0["width"], c0["height"])
vis = P[inb0]
inb01, _ = project(vis, np.array(c1["rotation"]), np.array(c1["position"]), c1["fx"], c1["fy"], c1["width"], c1["height"])
print(f"sanity train0->train1 overlap: {inb01.sum()}/{len(vis)}")
