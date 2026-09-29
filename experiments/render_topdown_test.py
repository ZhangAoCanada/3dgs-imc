#!/usr/bin/env python3
"""TEST: rasterized top-down + oblique render of a Gaussian PLY, aligned to the floater BEV crop.
Goal: confirm a REAL render (vs bevcontext's mushy color-projection) is recognizable from above.
Reuses main_crop projection + getWorld2View2/getProjectionMatrix + gaussian_renderer.render."""
import os, sys, math, glob
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, '/data1/zhangao/repos/3dgs-imc-codex')
sys.path.insert(0, '/data1/zhangao/repos/3dgs-imc-codex/experiments')
from scene_experimental_branch3 import GaussianModel
from gaussian_renderer import render
from argparse import Namespace
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from main_crop import proj_from_ref, main_box_proj
from plyfile import PlyData

PLY = sys.argv[1] if len(sys.argv) > 1 else '/data1/zhangao/outputs/best_coverage_gap/bonsai_angle_100.0/point_cloud.ply'
SCENE = sys.argv[2] if len(sys.argv) > 2 else 'bonsai'
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else '/tmp/topdown_test'
os.makedirs(OUTDIR, exist_ok=True)
W = H = 1000
PIPE = Namespace(debug=False, antialiasing=False, compute_cov3D_python=False, convert_SHs_python=False)

# projection + crop (match bev_grid_overlay exactly)
refp = sorted(glob.glob(f'/data1/zhangao/outputs/baselines_multi_angle/{SCENE}/*/input.ply'))
ref = np.stack([PlyData.read(refp[0])["vertex"][k] for k in ("x","y","z")],1).astype(np.float32)
center, axes = proj_from_ref(ref)
cx, cy, half = main_box_proj(ref, center, axes, mass_frac=0.88, margin=2.8)
ax0, ax1 = axes[0], axes[1]            # ax0 = X-ish (proj u), ax1 = Z-ish (proj v); up = the remaining axis
up_axis = np.cross(ax0, ax1)            # world up direction (sign to be tested)
crop_world = center + ax0*cx + ax1*cy
print("center", center, "axes", axes.tolist(), "up_axis", up_axis.tolist(), "cx,cy,half", round(cx,2), round(cy,2), round(half,2))
print("crop_world", crop_world.tolist())
print("ref Y(min/1%/50%/99%/max):", [round(float(np.percentile(ref[:,1],p)),2) for p in (0,1,50,99,100)])

# load gaussians, print Y range (ceiling check)
g = GaussianModel(3); g.load_ply(PLY); g.active_sh_degree = g.max_sh_degree
gy = g.get_xyz.detach()[:,1].cpu().numpy()
print("gauss Y(min/1%/50%/99%/max):", [round(float(np.percentile(gy,p)),2) for p in (0,1,50,99,100)], "N", g.get_xyz.shape[0])

def make_cam(cam_pos, forward, down, fov_deg, w, h):
    forward = forward/np.linalg.norm(forward)
    down = down/np.linalg.norm(down)
    right = np.cross(down, forward); right /= np.linalg.norm(right)
    # re-orthogonalize down
    down = np.cross(forward, right)
    R_c2w = np.stack([right, down, forward], axis=1).astype(np.float32)  # cols = cam axes in world (OpenCV)
    t_w2c = (-R_c2w.T @ cam_pos).astype(np.float32)
    fov = math.radians(fov_deg)
    znear = max(0.1, float(np.linalg.norm(cam_pos - crop_world)) - 4*half)
    zfar  = float(np.linalg.norm(cam_pos - crop_world)) + 4*half
    wv = torch.tensor(getWorld2View2(R_c2w, t_w2c)).transpose(0,1).cuda().float()
    proj = getProjectionMatrix(znear, zfar, fov, fov).transpose(0,1).cuda().float()
    fp = (wv.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    class MC:
        pass
    c = MC(); c.image_width=w; c.image_height=h; c.FoVy=fov; c.FoVx=fov
    c.znear=znear; c.zfar=zfar; c.world_view_transform=wv; c.full_proj_transform=fp
    vi = torch.inverse(wv); c.camera_center = vi[3,:3]
    return c

def do_render(cam, bg, tag):
    BG = torch.tensor(bg, device="cuda").float()
    with torch.no_grad():
        pkg = render(cam, g, PIPE, BG)
        rgb = pkg["render"].clamp(0,1).permute(1,2,0).cpu().numpy()
    Image.fromarray((rgb*255).astype(np.uint8)).save(f"{OUTDIR}/{tag}.png")
    print("wrote", tag, "shape", rgb.shape)

# 1) STRAIGHT top-down (orthographic-ish, narrow FoV), look along -up_axis from above
D = half / math.tan(math.radians(3.0))   # narrow 6deg FoV
cam_pos = crop_world + up_axis*D
do_render(make_cam(cam_pos, -up_axis, ax1, 6.0, W, H), [1,1,1], "topdown_white")
do_render(make_cam(cam_pos, -up_axis, ax1, 6.0, W, H), [0,0,0], "topdown_black")

# 2) OBLIQUE bird's-eye (recognizable 3/4 view): elevate along up_axis, offset along -ax1, look at crop
D2 = 3.0*half
cam_pos2 = crop_world + up_axis*D2 - ax1*2.2*half
forward2 = crop_world - cam_pos2;
do_render(make_cam(cam_pos2, forward2, -up_axis, 45.0, W, H), [1,1,1], "oblique_white")

# 3) opposite up sign straight-down (in case up_axis points down)
cam_pos3 = crop_world - up_axis*D
do_render(make_cam(cam_pos3, up_axis, ax1, 6.0, W, H), [1,1,1], "topdown_flip_white")
print("DONE")
