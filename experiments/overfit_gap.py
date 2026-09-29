#!/usr/bin/env python3
"""Perspective-overfitting test: how much does each method degrade from the views it SAW
(train) to the held-out biased gap views (test)? Renders sampled train + all test views with
ONE uniform renderer (no exposure) and computes PSNR consistently, so the GAP = train_PSNR -
test_PSNR is a fair, capacity-agnostic measure of overfitting to the biased training set.
Smaller gap = generalizes better = the paper's central claim.

Usage: CUDA_VISIBLE_DEVICES=1 python overfit_gap.py <depth_cells.json>
"""
import os, sys, json, math, random
import numpy as np, torch
from argparse import Namespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene_experimental_branch3 import Scene, GaussianModel
from gaussian_renderer import render

CELLS = json.load(open(sys.argv[1]))
DEST = "/data1/zhangao/outputs/viz_multiangle"
PIPE = Namespace(debug=False, antialiasing=False, compute_cov3D_python=False, convert_SHs_python=False)
BG = torch.tensor([0., 0., 0.], device="cuda")
N_TRAIN = 24
random.seed(0)

def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse) if mse > 0 else 99.0

def mp(cell, cam_mp):
    return Namespace(sh_degree=3, source_path=cell["source_path"], model_path=cam_mp,
                     images=cell["images"], depths="", resolution=1, white_background=False,
                     test_path="", test_selection_path=cell["test_selection"],
                     train_test_exp=False, data_device="cpu", eval=False, init_type="sfm",
                     depth_init_points=0, depth_init_stride=32, depth_init_views=0,
                     cap_max=8000000, load_iteration=0, load_model_path="")

@torch.no_grad()
def cell_gap(name, cell):
    cam_mp = os.path.dirname(os.path.dirname(os.path.dirname(cell["plys"]["MH-3DGS"])))
    g0 = GaussianModel(3)
    scene = Scene(mp(cell, cam_mp), g0, load_iteration=-1, shuffle=False)
    train = scene.getTrainCameras(); test = scene.getTestCameras()
    tr = random.sample(train, min(N_TRAIN, len(train)))
    out = {}
    for meth, ply in cell["plys"].items():
        if not os.path.exists(ply): continue
        g = GaussianModel(3); g.load_ply(ply); g.active_sh_degree = g.max_sh_degree
        def mean_psnr(cams):
            vs = []
            for c in cams:
                r = render(c, g, PIPE, BG)["render"].clamp(0, 1)
                gt = c.original_image.to(r.device)[:3]
                if gt.shape[-2:] != r.shape[-2:]:
                    r = torch.nn.functional.interpolate(r[None], size=gt.shape[-2:], mode="bilinear")[0]
                vs.append(psnr(r, gt))
            return float(np.mean(vs)) if vs else None
        ptr = mean_psnr(tr); pte = mean_psnr(test)
        out[meth] = dict(train=ptr, test=pte, gap=(ptr - pte) if (ptr and pte) else None)
        del g; torch.cuda.empty_cache()
        print(f"  [{name}/{meth}] train {ptr:.2f}  test {pte:.2f}  gap {ptr-pte:+.2f}", flush=True)
    del g0, scene; torch.cuda.empty_cache()
    return out

if __name__ == "__main__":
    res = {}
    for name, cell in CELLS.items():
        print(f"[{name}]", flush=True)
        try: res[name] = cell_gap(name, cell)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[{name}] FAIL {e}", flush=True)
    json.dump(res, open(os.path.join(DEST, "overfit_gap.json"), "w"), indent=1)
    # aggregate
    methods = ["Ours", "Hier", "MH-3DGS", "MVGS"]
    print("\n=== OVERFITTING GAP (train PSNR - test PSNR), lower = generalizes better ===")
    agg = {m: [] for m in methods}
    for name, d in res.items():
        for m in methods:
            if m in d and d[m]["gap"] is not None: agg[m].append(d[m]["gap"])
    for m in methods:
        if agg[m]: print(f"  {m:>9}: mean gap {np.mean(agg[m]):+.2f} dB  (n={len(agg[m])})")
