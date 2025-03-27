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
import math
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


def generate_viewpoints(raw_view, up_dist=0.1, lookat_dist=0.3, forward_dist=0.0):
    """
    Generate a camera viewpoint that is 0.1m higher than the current viewpoint.
    Both viewpoints look at the same point, which is 1m in front of the current viewpoint.

    Parameters:
    view (NamedTuple): The current camera view with rotation matrix R and translation vector T
    """
    view = copy.deepcopy(raw_view)
    relative_translation = torch.tensor([0, up_dist, forward_dist]).cuda()
    rotation_angle = math.atan(up_dist / lookat_dist)
    rotation_angle = torch.tensor(rotation_angle)
    relative_rotation = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(rotation_angle), -torch.sin(rotation_angle)],
        [0, torch.sin(rotation_angle), torch.cos(rotation_angle)]]).cuda()
    old_w2c = view.world_view_transform.transpose(0, 1)

    new_rotation = torch.eye(4).cuda()
    new_rotation[:3, :3] = relative_rotation
    old_w2c[:3, 3] += relative_translation
    new_w2c = new_rotation @ old_w2c

    # relative_transform = torch.eye(4).cuda()
    # relative_transform[:3, :3] = relative_rotation
    # relative_transform[:3, 3] = relative_translation
    # new_w2c = relative_transform @ old_w2c

    view.world_view_transform = new_w2c.transpose(0, 1).cuda()
    view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda()
    view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
    view.camera_center = view.world_view_transform.inverse()[3, :3]
    return view


def camera_trajectory(view, gaussians, pipeline, background, train_test_exp, separate_sh, up_dists, lookat_dists, forward_dists):
    assert len(up_dists) == len(lookat_dists) == len(forward_dists)
    all_renderings = []
    for i, up_dist in enumerate(up_dists):
        lookat_dist = lookat_dists[i]
        forward_dist = forward_dists[i]
        view_new = generate_viewpoints(view, up_dist=up_dist, lookat_dist=lookat_dist, forward_dist=forward_dist)
        render_pkg_new = render(view_new, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering_new = render_pkg_new["render"]
        # depth_show_new = render_pkg_new["depth"].repeat(3, 1, 1)
        depth_show_new = (1 / (render_pkg_new["depth"] + 1e-6)).repeat(3, 1, 1)
        rendering_new = torch.cat([rendering_new, depth_show_new], dim=2)
        all_renderings.append(rendering_new.unsqueeze(0).permute(0, 2, 3, 1))
    all_renderings = torch.cat(all_renderings, dim=0)
    all_renderings = (all_renderings * 255).to(torch.uint8)
    return all_renderings


def camera_render(view, gaussians, pipeline, background, train_test_exp, separate_sh, up_dist, lookat_dist, forward_dist):
    view_new = generate_viewpoints(view, up_dist=up_dist, lookat_dist=lookat_dist, forward_dist=forward_dist)
    render_pkg_new = render(view_new, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
    rendering_new = render_pkg_new["render"]
    depth_show_new = render_pkg_new["depth"].repeat(3, 1, 1)
    torchvision.utils.save_image(rendering_new, "tmp/tmp.png")


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
        if idx not in [206, 344, 375, 507]:
            continue
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
            up_dists = [i * 0.03 for i in range(50, 0, -1)] + [0.0] * 50
            lookat_dists = [0.5] * 100
            forward_dists = [0.0] * 50 + [i * -0.03 for i in range(50)]
            all_renderings = camera_trajectory(view, gaussians, pipeline, background, train_test_exp, separate_sh, up_dists, lookat_dists, forward_dists)
            # camera_render(view, gaussians, pipeline, background, train_test_exp, separate_sh, 0.1, 0.3, 0.0)
            # write_video(os.path.join("./tmp", '{0:05d}'.format(idx) + ".mp4"), all_renderings.cpu().numpy(), 30)
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