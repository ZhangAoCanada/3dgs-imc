#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_gsplat
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
from scene_experimental_branch3.gaussian_model import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
import copy
import numpy as np
import cv2
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def write_video(video_path, frames, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frames.shape[2], frames.shape[1]))
    for i in range(frames.shape[0]):
        frame = frames[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


def generate_viewpoints(raw_view, up_dist=0.1, lookat_dist=0.3):
    """
    Generate a camera viewpoint that is 0.1m higher than the current viewpoint.
    Both viewpoints look at the same point, which is 1m in front of the current viewpoint.

    Parameters:
    view (NamedTuple): The current camera view with rotation matrix R and translation vector T
    """
    view = copy.deepcopy(raw_view)
    view.T[1] += up_dist

    rotation_angle = -np.arctan2(up_dist, lookat_dist)
    relative_rotation = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
        [0, np.sin(rotation_angle), np.cos(rotation_angle)]])
    view.R = np.matmul(view.R, relative_rotation)

    view.world_view_transform = torch.tensor(getWorld2View2(view.R, view.T, view.trans, view.scale)).transpose(0, 1).cuda()
    view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda()
    view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
    view.camera_center = view.world_view_transform.inverse()[3, :3]
    return view


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, if_render=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    video_path = os.path.join(model_path, name, "ours_{}".format(iteration), "videos")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(video_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = render_pkg["render"]
        inv_depth = render_pkg["depth"]
        gt = view.original_image[0:3, :, :]

        if view.invdepthmap is not None:
            inv_depth /= inv_depth.max()
            mono_invdepth = view.invdepthmap.cuda()
            mono_invdepth /= mono_invdepth.max()
            depth_mask = view.depth_mask.cuda()
            masked_inv_depth = depth_mask * inv_depth
            render_pkg_gsplat = render_gsplat(view, gaussians)
            gsplat_depth = render_pkg_gsplat["depth"]
            gsplat_depth /= gsplat_depth.max()
            gsplat_depth_masked = depth_mask * gsplat_depth
            depth_show = torch.cat([mono_invdepth, inv_depth, masked_inv_depth, gsplat_depth, gsplat_depth_masked], dim=1)
        else:
            depth_show = inv_depth

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        if if_render:
            up_dists = [i * 0.01 for i in range(50)]
            all_renderings = []
            for up_dist in up_dists:
                view_new = generate_viewpoints(view, up_dist=up_dist)
                render_pkg_new = render(view_new, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
                rendering_new = render_pkg_new["render"]
                depth_show_new = render_pkg_new["depth"].repeat(3, 1, 1)
                rendering_new = torch.cat([rendering_new, depth_show_new], dim=2)
                all_renderings.append(rendering_new.unsqueeze(0).permute(0, 2, 3, 1))
            all_renderings = torch.cat(all_renderings, dim=0)
            all_renderings = (all_renderings * 255).to(torch.uint8)

            write_video(os.path.join(video_path, '{0:05d}'.format(idx) + ".mp4"), all_renderings.cpu().numpy(), 30)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth_show, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, if_render: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, if_render=if_render)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, if_render=if_render)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--if_render", action="store_true")
    args = get_combined_args(parser) # loading presaved "cfg_args"
    print("Rendering " + args.model_path)

    args.train_test_exp = False

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.if_render)