#!/usr/bin/env python3
"""Generate leak-proof PatchMatch configs for a (scene, angle): reference images and
per-reference source lists contain ONLY train-split images (no __auto__, so COLMAP can
never select a test image). Sources = K nearest train cameras by position.

usage: make_mvs_cfgs.py <scene> <angle> <outdir>
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scene_experimental_branch3.colmap_loader import read_extrinsics_binary, qvec2rotmat

DATA = "/data1/zhangao/DATA/mipnerf360"

scene, angle, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
test_names = set(open(f"{DATA}/{scene}/test_selection_angle_{angle}.txt").read().split())
images = read_extrinsics_binary(f"{DATA}/{scene}/sparse/0/images.bin")

names, centers = [], []
for img in images.values():
    if img.name in test_names:
        continue
    R = qvec2rotmat(img.qvec)
    names.append(img.name)
    centers.append((-R.T @ img.tvec))
centers = np.array(centers)
print(f"{scene}@{angle}: {len(names)} train images ({len(test_names)} test excluded)")

K = 10
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "patch-match.cfg"), "w") as f:
    for i, nm in enumerate(names):
        d = np.linalg.norm(centers - centers[i], axis=1)
        src = [names[j] for j in np.argsort(d)[1:K + 1]]
        f.write(nm + "\n")
        f.write(", ".join(src) + "\n")
with open(os.path.join(outdir, "fusion.cfg"), "w") as f:
    for nm in names:
        f.write(nm + "\n")
print(f"wrote {outdir}/patch-match.cfg and fusion.cfg")
