#!/usr/bin/env python3
"""Uniform novel-view DEPTH + RGB renderer for cross-method floater comparison.
For each (scene, angle) cell, build the test cameras ONCE via the codex Scene loader
(deterministic test_selection_angle split -> identical viewpoints/order as the stored
renders), then swap each method's STANDARD 3DGS PLY in and render. Saves raw depth (.npy)
and the rasterized RGB (for self-validation) per hero view, per method.

Methods covered (standard PLY only): Ours, Hier(3dgs-depth), MH-3DGS, MVGS.
(ScaffoldGS uses anchor PLYs -> not rendered here.)

Usage: CUDA_VISIBLE_DEVICES=1 python render_depth_uniform.py <cells.json> [--validate]
cells.json: {cell: {source_path, test_selection, images, hero:[idx...], plys:{Method:path}}}
"""
import os, sys, json, glob
import numpy as np
import torch
from argparse import Namespace
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene_experimental_branch3 import Scene, GaussianModel
from gaussian_renderer import render

CELLS = json.load(open(sys.argv[1]))
VALIDATE = "--validate" in sys.argv
DEST = "/data1/zhangao/outputs/viz_multiangle/depth"
os.makedirs(DEST, exist_ok=True)
PIPE = Namespace(debug=False, antialiasing=False, compute_cov3D_python=False, convert_SHs_python=False)
BG = torch.tensor([0., 0., 0.], device="cuda")

def model_params(cell, cam_model_path):
    # ModelParams-like namespace consumed by Scene/dataset_readers
    return Namespace(sh_degree=3, source_path=cell["source_path"], model_path=cam_model_path,
                     images=cell["images"], depths="", resolution=1, white_background=False,
                     test_path="", test_selection_path=cell["test_selection"],
                     train_test_exp=False, data_device="cpu", eval=False,
                     init_type="sfm", depth_init_points=0, depth_init_stride=32, depth_init_views=0,
                     cap_max=3000000, load_iteration=0, load_model_path="")

@torch.no_grad()
def run_cell(name, cell):
    od = os.path.join(DEST, name.replace("@", "_"))
    os.makedirs(od, exist_ok=True)
    # camera source = MH cell (always has point_cloud) -> Scene loads its cams + gaussians
    cam_mp = os.path.dirname(os.path.dirname(cell["plys"]["MH-3DGS"]))  # .../<cell>/point_cloud/iter -> cell dir
    cam_mp = os.path.dirname(cam_mp)
    gaussians = GaussianModel(3)
    scene = Scene(model_params(cell, cam_mp), gaussians, load_iteration=-1, shuffle=False)
    test = scene.getTestCameras()
    print(f"[{name}] {len(test)} test cams; hero idx {cell['hero']}", flush=True)
    hero = set(cell["hero"])
    sel = [(i, c) for i, c in enumerate(test) if i in hero]
    for meth, ply in cell["plys"].items():
        if not os.path.exists(ply):
            print(f"  [{name}/{meth}] MISSING ply {ply}", flush=True); continue
        g = GaussianModel(3); g.load_ply(ply)
        g.active_sh_degree = g.max_sh_degree
        md = os.path.join(od, meth); os.makedirs(md, exist_ok=True)
        for i, cam in sel:
            pkg = render(cam, g, PIPE, BG)
            depth = pkg["depth"].detach().squeeze().cpu().numpy().astype(np.float32)  # (H,W)
            np.save(os.path.join(md, f"{i:05d}.npy"), depth)
            if VALIDATE:
                rgb = (pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(rgb).save(os.path.join(md, f"{i:05d}_rgb.png"))
        print(f"  [{name}/{meth}] {g.get_xyz.shape[0]} gauss -> {len(sel)} depth maps", flush=True)
        del g; torch.cuda.empty_cache()
    del gaussians, scene; torch.cuda.empty_cache()

if __name__ == "__main__":
    for name, cell in CELLS.items():
        try:
            run_cell(name, cell)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{name}] FAILED: {e}", flush=True)
