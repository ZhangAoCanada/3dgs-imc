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
import sys
sys.path.append("dust3r")

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.reloc_utils import compute_relocation_cuda
from utils.sh_utils import eval_sh
from scene_experimental_branch2.imc_experimental import NetworksA
from scene_experimental_branch2.pntfunc_experimental import NetworkAnother
from scene_experimental_branch2 import diff_operators
from scene_experimental_branch2.utility import write_summary

# from dust3r.inference import inference
# from dust3r.model import AsymmetricCroCo3DStereo
# from dust3r.utils.image import load_images
# from dust3r.image_pairs import make_pairs
# from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
# from dust3r.demo import get_3D_model_from_scene, _convert_scene_output_to_glb
import shutil
import math
from glob import glob
from gaussian_renderer import render, render_gsplat
import torchvision
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression, RANSACRegressor
from tqdm import tqdm


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2)*0.1)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        ###################### NOTE: change properties #######################
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        # self._xyz = fused_point_cloud.clone()
        # self._opacity = opacities.clone()

        #################### NOTE: hyper-param ########################
        self.max_num = 100000
        self.net_mode = "mlp" # "mlp" or "fft"
        self.net_type = "sine" # "sine" or "relu"
        self.net = NetworksA(
            in_features=3, 
            out_features=4, 
            type=self.net_type, 
            mode=self.net_mode, 
            )
        self.net.cuda()
        self.nn_gt_pts = None
        self.aligned_depth_dict = {}
        self.downsample_ratio = 4
        self.voxel_size = 0.001
        self.eps = 0.04 # 0.05
        self.min_samples = 100 # 100

        # self.max_num = 1000
        # self.net = NetworkAnother(
        #     xyz_bounds=[xyz_lowerbound, xyz_upperbound],
        #     batch_size=self.max_num,
        # )
        # self.net.cuda()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}, 
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)

        #################### NOTE: hyper-param ########################
        self.imc_experimental_optimizer = torch.optim.Adam(lr=1e-4, params=self.net.parameters())
        self.imc_experimental_optim_scheduler = torch.optim.lr_scheduler.StepLR(self.imc_experimental_optimizer, step_size=1000, gamma=0.5)
        # self.imc_experimental_optim_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.imc_experimental_optimizer, gamma=0.9)
        self.partial_scaling = training_args.partial_scaling
        self.n_jobs = training_args.n_jobs

        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init, training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations)


    ######################################################################
    ######################################################################
    ######################################################################
    def nntrain(self, tb_writer, iteration):
        if self.nn_gt_pts is None:
            return None
        self.net.train()
        assert self.nn_gt_pts.shape[1] == 9
        xyz, color, norm = torch.split(self.nn_gt_pts, [3, 3, 3], dim=1)
        mask = None
        if xyz.shape[0] > self.max_num:
            mask = torch.zeros(xyz.shape[0], dtype=torch.bool)
            random_index = torch.randperm(xyz.shape[0])[:self.max_num]
            mask[random_index] = True
            xyz = xyz[mask]
            color = color[mask]
            norm = norm[mask]
        opacity = torch.ones((xyz.shape[0], 1), device="cuda")
        off_xyz, off_color, off_norm, off_opacity = self.spawn_randompnts(xyz.shape[0], self.voxel_size)
        net_in = torch.cat([xyz, off_xyz], dim=0)
        res = self.net(net_in)
        gt_opacity = torch.cat([opacity, off_opacity], dim=0)
        gt_color = torch.cat([color, off_color], dim=0)
        gt_norm = torch.cat([norm, off_norm], dim=0)
        l = self.nn_l(res, gt_opacity, gt_color, gt_norm)
        write_summary(self.net, tb_writer, iteration)
        return l
    
    def nntrain_view(self, view, pipe, bg, tb_writer, iteration, align=True):
        if self.net.xyz_lowerbound is None or self.net.xyz_upperbound is None:
            return None
        H, W, focals, c2w = self.viewcam_properties(view, self.downsample_ratio)
        with torch.no_grad():
            pts, aligned_depth = self.filter_depth_and_align(view, pipe, bg, align=align, debug=False, single_view=True)
        if pts is None:
            return None
        if align:
            assert aligned_depth is not None
        assert pts.shape[0] == H * W
        self.net.train()
        assert pts.shape[1] == 9
        xyz, color, norm = torch.split(pts, [3, 3, 3], dim=1)
        mask = None
        if xyz.shape[0] > self.max_num:
            mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device="cuda")
            random_index = torch.randperm(xyz.shape[0])[:self.max_num]
            mask[random_index] = True
            xyz = xyz[mask]
            color = color[mask]
            norm = norm[mask]
        else:
            mask = torch.ones(xyz.shape[0], dtype=torch.bool, device="cuda")
        opacity = torch.ones((xyz.shape[0], 1), device="cuda")
        off_xyz, off_color, off_norm, off_opacity = self.spawn_randomraypnts(mask, aligned_depth, H, W, focals, c2w, times=1, voxel_size=self.voxel_size)
        net_in = torch.cat([xyz, off_xyz], dim=0)
        res = self.net(net_in)
        gt_opacity = torch.cat([opacity, off_opacity], dim=0)
        gt_color = torch.cat([color, off_color], dim=0)
        gt_norm = torch.cat([norm, off_norm], dim=0)
        l = self.nn_l(res, gt_opacity, gt_color, gt_norm)
        write_summary(self.net, tb_writer, iteration)
        return l
    
    def nn_l(self, pred, gt_opacity, gt_color, gt_norm):
        pred_color = pred['rgb']
        pred_opacity = pred['sigma']
        net_in = pred['net_in']
        gradient = diff_operators.gradient(pred_opacity, net_in)

        opacity_constraint = torch.where(gt_opacity != 0, F.l1_loss(pred_opacity, gt_opacity), torch.zeros_like(pred_opacity)).mean()
        color_constraint = torch.where(gt_opacity != 0, F.l1_loss(pred_color, gt_color), torch.zeros_like(pred_color)).mean()
        normal_constraint = torch.where(gt_opacity != 0, 1. - F.cosine_similarity(gradient, gt_norm, dim=-1)[..., None], torch.zeros_like(gradient)).mean()
        gradient_constraint = torch.abs(gradient.norm(dim=-1) - 1).mean()
        inter_constraint = torch.where(gt_opacity != 0, torch.zeros_like(pred_opacity), pred_opacity).mean()

        l = opacity_constraint * 3e3 + \
            color_constraint * 1e3 + \
            normal_constraint * 1e2 + \
            inter_constraint * 1e2 + \
            gradient_constraint * 5e1
        return l

    def partial_l(self, tb_writer, iteration):
        if self.nn_gt_pts is None:
            return 0.0
        xyz = self.get_xyz
        if xyz.shape[0] > self.max_num:
            mask = torch.zeros(xyz.shape[0], dtype=torch.bool)
            random_index = torch.randperm(xyz.shape[0])[:self.max_num]
            mask[random_index] = True
            net_in = xyz[mask]
        pred = self.net(net_in)
        # l = 100 * (1.0 - pred['sigma']).mean()
        l = (1.0 - pred['sigma'])
        grad = diff_operators.gradient(l, pred['net_in'])
        net_scale = (self.net.xyz_upperbound - self.net.xyz_lowerbound)
        with torch.no_grad():
            xyz_upper = self.net.xyz_upperbound
            xyz_lower = self.net.xyz_lowerbound
            diff_upper = net_in - xyz_upper
            diff_lower = net_in - xyz_lower
            scaling = torch.ones_like(net_in)
            scaling = torch.where(diff_upper > 0, torch.abs(diff_upper) * 0.01, 1)
            scaling = torch.where(diff_lower < 0, torch.abs(diff_lower) * 0.01, 1)
            scaling = torch.where(scaling < 1, 1, scaling)
            grad = grad * net_scale * scaling
            # grad = grad * net_scale
            grad = grad * self.partial_scaling
            self._xyz[mask].add_(grad)
        if tb_writer is not None:
            tb_writer.add_scalar("nn_l/partial_grad_min", grad.min(), iteration)
            tb_writer.add_scalar("nn_l/partial_grad_max", grad.max(), iteration)
        ### NOTE: original size ###
        l = l.mean()
        return l

    def spawn_randompnts(self, num_pnts=100000, voxel_size=0.01):
        upper_bound = self.net.xyz_upperbound
        lower_bound = self.net.xyz_lowerbound
        off_pnts = torch.rand((num_pnts, 3), device="cuda") * (upper_bound - lower_bound) + lower_bound
        # # voxelization and remove duplicates with self.nn_gt_pts
        # off_pnts = torch.round(off_pnts / voxel_size) * voxel_size
        # off_pnts = torch.unique(off_pnts, dim=0)
        # num_pnts = off_pnts.shape[0]
        # get parameters
        off_xyz = nn.Parameter(off_pnts.contiguous().requires_grad_(True))
        off_opacity = torch.zeros((num_pnts, 1), device="cuda")
        off_color = torch.rand((num_pnts, 3), device="cuda")
        off_norm = torch.rand((num_pnts, 3), device="cuda")
        return off_xyz, off_color, off_norm, off_opacity
    
    def spawn_randomraypnts(self, mask, depth, H, W, focals, c2w, times=1, voxel_size=0.01):
        times = int(times)
        fx, fy = focals
        i, j = torch.meshgrid(torch.arange(W, device="cuda"), torch.arange(H, device="cuda"), indexing='xy')
        i, j = i.reshape(-1), j.reshape(-1)
        dirs = torch.stack([(i-W*.5)/fx, (j-H*.5)/fy, torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
        rays_o = c2w[:3,-1].expand(rays_d.shape)

        lowbound = min(self.net.xyz_lowerbound, 0.0).unsqueeze(0).expand(rays_o.shape)
        upbound = max(self.net.xyz_upperbound, 0.0).unsqueeze(0).expand(rays_o.shape)
        lowdist = (lowbound - rays_o) / (rays_d + 1e-6)
        updist = (upbound - rays_o) / (rays_d + 1e-6)

        rand_dist = torch.rand(times, H*W, 1, device="cuda") * (updist - lowdist) + lowdist
        off_pts = rays_o.unsqueeze(0) + (rays_d.unsqueeze(0) * rand_dist)
        mask = mask.view(1, -1).expand(times, -1)
        off_pts = off_pts[mask].reshape(-1, 3)

        off_pts = off_pts.reshape(-1, 3)
        off_xyz = nn.Parameter(off_pts.contiguous().requires_grad_(True))
        off_opacity = torch.zeros((off_xyz.shape[0], 1), device="cuda")
        off_color = torch.rand((off_xyz.shape[0], 3), device="cuda")
        off_norm = torch.rand((off_xyz.shape[0], 3), device="cuda")
        return off_xyz, off_color, off_norm, off_opacity
    
    def save_nn(self, dir_path):
        torch.save(self.net.state_dict(), os.path.join(dir_path, "net.pth"))
    
    def load_nn(self, dir_path):
        self.net.load_state_dict(torch.load(os.path.join(dir_path, "net.pth")))

    def add_noise(self, noise, mask=None):
        if mask is None:
            self._xyz.add_(noise)
        else:
            self._xyz[mask].add_(noise[mask])

    def remove_nan_grad(self, ):
        self._xyz.grad[torch.isnan(self._xyz.grad)] = 0.0
        self._xyz.grad[torch.isinf(self._xyz.grad)] = 0.0
        for param in self.net.parameters():
            param.grad[torch.isnan(param.grad)] = 0.0
            param.grad[torch.isinf(param.grad)] = 0.0
        
    def update_nnpts(self, all_views, pipe, bg, align=True, single_view=False, gaussians_bound=False):
        if gaussians_bound:
            self.net.find_boundary(self.get_xyz.detach().clone(), extend_factor=0.0)
        else:
            self.nn_gt_pts = self.pointdepth(all_views, pipe, bg, align, debug=False, single_view=single_view)
            if self.nn_gt_pts is not None:
                # self.net.find_boundary(self.get_xyz.detach().clone(), extend_factor=0.0)
                self.net.find_boundary(self.nn_gt_pts)
    ######################################################################
    ######################################################################
    ######################################################################
    # def nntrain(self, viewpoint_cam):
    #     self.net.train()
    #     nnl = self.net(viewpoint_cam, train=True)
    #     return nnl
    
    # def nntrainpts(self, viewpoint_cam):
    #     self.net.train()
    #     with torch.no_grad():
    #         render_pkg = render_gsplat(viewpoint_cam, self)
    #         render_depth = render_pkg["depth"]
    #         render_depth = render_depth.detach().clone()
    #     nnl = self.net.train_pts(viewpoint_cam, render_depth)
    #     return nnl
    
    # def nnrender(self, viewpoint_cam):
    #     self.net.eval()
    #     rgb, inv_depth, acc = self.net(viewpoint_cam, train=False)
    #     rgb = torch.clamp(rgb, 0.0, 1.0)
    #     inv_depth = torch.clamp(inv_depth, 0.0, 1.0)
    #     acc = torch.clamp(acc, 0.0, 1.0)
    #     return rgb, inv_depth, acc
    
    def pointdepth(self, all_views, pipe, bg, align=True, debug=True, single_view=False):
        if debug:
            self.nn_gt_pts = None
            self.aligned_depth_dict = {}
            self.downsample_ratio = 4
            self.voxel_size = 0.01
            self.eps = 0.04 # 0.04
            self.min_samples = 50 # 100
            all_views = all_views[:100]

        all_pts = torch.empty(0, device="cuda")
        if debug:
            if os.path.exists("tmp"):
                shutil.rmtree("tmp")
            os.makedirs("tmp", exist_ok=True)
        with torch.no_grad():
            for i in tqdm(range(len(all_views))):
                view = all_views[i]
                # if int(view.image_name.split(".")[0]) >= 1171:
                #     continue
                # if int(view.image_name.split(".")[0]) in [248, 249]:
                #     continue
                pt_clr, _ = self.filter_depth_and_align(view, pipe, bg, align=align, debug=debug, single_view=single_view)
                if pt_clr is None:
                    continue
                # voxelization and unique with all points
                all_pts = torch.cat([all_pts, pt_clr], dim=0)
                all_pts = self.unique_pts(all_pts, self.voxel_size)
        if debug:
            # save to ply
            ply_path = os.path.join("tmp", "pointdepth.ply")
            self.save_to_ply(all_pts[..., :6], ply_path)
            print("debugging")
            # # for debugging
            # xyz_ = self.get_xyz.detach().clone()
            # xyz_ = self.unique_pts(xyz_, 0.03)
            # colors_ = torch.zeros_like(xyz_).float()
            # xyz_ = torch.cat([xyz_, colors_], dim=1)
            # all_pts = torch.cat([xyz_, all_pts[..., :6]], dim=0) 
            # ply_path = os.path.join("tmp", "pointall.ply")
            # self.save_to_ply(all_pts, ply_path)
            # print("debugging")
        if all_pts.shape[0] == 0:
            return None
        return all_pts
    
    def filter_depth_and_align(self, view, pipe, bg, align=True, debug=True, single_view=False):
        align_depth = None
        mask = None
        # render_pkg = render(view, self, pipe, bg, use_trained_exp=False, separate_sh=False)
        render_pkg_gsplat = render_gsplat(view, self)
        render_depth_gsplat = render_pkg_gsplat["depth"]
        gt = view.original_image.cuda()
        mono_invdepth = view.invdepthmap.cuda()
        # downsampling
        render_depth_gsplat = F.interpolate(render_depth_gsplat.unsqueeze(0), size=None, scale_factor=1/self.downsample_ratio, mode='bilinear', align_corners=False).squeeze(0)
        gt = F.interpolate(gt.unsqueeze(0), size=None, scale_factor=1/self.downsample_ratio, mode='bilinear', align_corners=False).squeeze(0)
        mono_invdepth = F.interpolate(mono_invdepth.unsqueeze(0), size=None, scale_factor=1/self.downsample_ratio, mode='bilinear', align_corners=False).squeeze(0)
        H, W, focals, c2w = self.viewcam_properties(view, self.downsample_ratio)
        # get points 
        pts = self.depth_proj(render_depth_gsplat, H, W, focals, c2w)
        pts_np = pts.detach().cpu().numpy()
        # dbscan
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=self.n_jobs).fit(pts_np)
        labels = db.labels_
        labels = torch.tensor(labels, device="cuda", dtype=torch.float)
        mask = labels != -1
        if mask.sum() == 0:
            return None, None
        ### NOTE: alternatives to dbscan ###
        render_invdepth_gsplat = 1.0 / (render_depth_gsplat + 1e-6)
        mask_threshold = 1e-3
        mask = torch.zeros(H*W, dtype=torch.bool, device="cuda")
        mask = torch.where(render_invdepth_gsplat > mask_threshold, True, False)
        ####################################
        # TODO: try to align mono_depth with render_depth_gsplat
        if align:
            align_depth = self.depth_align(render_depth_gsplat, mono_invdepth, mask.reshape(1, H, W), debug=False)
            self.aligned_depth_dict[view.image_name] = align_depth
            pts = self.depth_proj(align_depth, H, W, focals, c2w)
        ### **************************************************** ###
        elif not single_view:
            align_depth = self.depth_align(render_depth_gsplat, mono_invdepth, mask.reshape(1, H, W), debug=False)
            self.aligned_depth_dict[view.image_name] = align_depth
        ### **************************************************** ###
        # compute normal map
        if align_depth is None:
            mono_depth = 1.0 / (mono_invdepth + 1e-4)
            scale_factor = 1.0
        else:
            mono_depth = align_depth
            scale_factor = 100.0
        mono_normal = self.depth2norm(mono_depth, scale_factor=scale_factor).squeeze(0)
        # visualize
        if debug:
            mask_show = mask.reshape(H, W).float().unsqueeze(0).repeat(3, 1, 1) if mask is not None else torch.ones(3, H, W, device="cuda").float()
            mono_depth /= mono_depth.max()
            mono_depth_show = mono_depth.repeat(3, 1, 1)
            mono_normal_show = mono_normal
            img_show = torch.cat([mono_depth_show, mono_normal_show, mask_show], dim=1)
            torchvision.utils.save_image(img_show, f"tmp/{view.image_name}")
        # keep the shape
        colors = gt.reshape(3, -1).t()
        normals = mono_normal.reshape(3, -1).t()
        # get valid points
        if not align:
            pts = pts[mask]
            colors = colors[mask]
            normals = normals[mask]
        pt_clr = torch.cat([pts, colors, normals], dim=1)
        return pt_clr, align_depth
    
    def depth_align(self, render_depth, mono_invdepth, mask=None, debug=True):
        """
        Mono_depth is normally with unkown scale. The function here is to compute the correct scale with render_depth, so that the two depth maps can be aligned. render_depth is somehow noisy. Therefore, we need to use mask to filter out the noisy points.

        @param render_depth: [1, H, W]
        @param mono_invdepth: [1, H, W]
        @param mask: [1, H, W]
        """
        if mask is None:
            mask = torch.ones_like(render_depth, dtype=torch.bool, device="cuda")
        render_invdepth = 1.0 / (render_depth + 1e-6)
        render_inv = render_invdepth[mask].reshape(-1).detach().cpu().numpy()
        mono_inv = mono_invdepth[mask].reshape(-1).detach().cpu().numpy()
        # find linear transformation from mono_inv to render_inv with RANSAC
        mono_inv = mono_inv.reshape(-1, 1)
        render_inv = render_inv.reshape(-1, 1)
        # # ransac = RANSACRegressor(LinearRegression(n_jobs=-1), min_samples=0.8, residual_threshold=0.1, max_trials=100)
        ransac = LinearRegression(n_jobs=self.n_jobs)
        ransac.fit(mono_inv, render_inv)
        _, H, W = render_depth.shape
        align_invdepth = ransac.predict(mono_invdepth.reshape(-1, 1).detach().cpu().numpy())
        align_invdepth = torch.from_numpy(align_invdepth).reshape(1, H, W).float().cuda()
        align_depth = 1.0 / (align_invdepth + 1e-6)

        if debug:
            align_depth_show = (align_depth - align_depth.min()) / (align_depth.max() - align_depth.min())
            render_depth_show = (render_depth - align_depth.min()) / (align_depth.max() - align_depth.min())
            render_depth_show = torch.clamp(render_depth_show, 0.0, 1.0)
            image_show = torch.cat([render_depth_show, align_depth_show], dim=1) 
            torchvision.utils.save_image(image_show, "tmp/align_depth.png")
        return align_depth
    
    def depth2norm(self, depth, scale_factor=1.0):
        """
        Transform depth map to normal map
        """
        if depth.dim() == 3:
            depth = depth.unsqueeze(0)
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device="cuda")
        sobel_y = sobel_x.t()
        kernel_x = sobel_x.view(1, 1, 3, 3)
        kernel_y = sobel_y.view(1, 1, 3, 3)
        grad_x = F.conv2d(depth, kernel_x, padding=1, groups=depth.size(0))
        grad_y = F.conv2d(depth, kernel_y, padding=1, groups=depth.size(0))
        normal = torch.cat([
            -scale_factor * grad_x,
            -scale_factor * grad_y,
            torch.ones_like(depth)
        ], dim=1)
        norm = torch.sqrt(torch.sum(normal ** 2, dim=1, keepdim=True) + 1e-6)
        normal = normal / norm
        normal = normal * 0.5 + 0.5
        return normal
    
    def unique_pts(self, points, voxel_size):
        points[:, :3] = torch.round(points[:, :3] / voxel_size) * voxel_size
        points = torch.unique(points, dim=0)
        return points
    
    def viewcam_properties(self, viewpoint_cam, downsample_ratio=1.0):
        fovx = viewpoint_cam.FoVx
        fovy = viewpoint_cam.FoVy
        W = viewpoint_cam.image_width // downsample_ratio
        H = viewpoint_cam.image_height // downsample_ratio
        fx = W / (2 * math.tan(fovx / 2))
        fy = H / (2 * math.tan(fovy / 2))
        K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device="cuda")
        w2c = viewpoint_cam.world_view_transform.transpose(0, 1)
        c2w = torch.inverse(w2c) 
        return H, W, [fx, fy], c2w
    
    def depth_proj(self, depth, H, W, focals, c2w):
        # fx, fy = focals
        # i, j = torch.meshgrid(torch.arange(W, device="cuda"), torch.arange(H, device="cuda"), indexing='xy')
        # i, j = i.reshape(-1), j.reshape(-1)
        # dirs = torch.stack([(i-W*.5)/fx, (j-H*.5)/fy, torch.ones_like(i)], -1)
        # rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
        # rays_o = c2w[:3,-1].expand(rays_d.shape)
        # pts = rays_o + rays_d * depth.reshape(1, -1).t()
        fx, fy = focals
        x = torch.arange(W, device="cuda").repeat(H, 1).reshape(-1)
        y = torch.arange(H, device="cuda").repeat(W, 1).t().reshape(-1)
        z = depth[0].reshape(-1)
        pts = torch.stack([x, y, z], 1)
        pts[:, 0] = (pts[:, 0] - W * 0.5) / fx * pts[:, 2]
        pts[:, 1] = (pts[:, 1] - H * 0.5) / fy * pts[:, 2]
        pts = torch.cat([pts, torch.ones(pts.shape[0], 1, device="cuda")], 1)
        pts = torch.matmul(c2w, pts.t()).t()
        pts = pts[:, :3]
        return pts

    def save_to_ply(self, all_pts, ply_path):
        all_pts = all_pts.detach().cpu().numpy()
        with open(ply_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex {}\n".format(all_pts.shape[0]))
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(all_pts.shape[0]):
                x, y, z, r, g, b = all_pts[i]
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
        print("[INFO] save to ply done.")
    
    # def point3r(self, all_views):
    #     device = 'cuda'
    #     batch_size = 1
    #     schedule = 'linear'
    #     lr = 0.01
    #     niter = 300
    #     outdir = "/data2/zhangao/repos/dust3r/tmp"
    #     if os.path.exists(outdir):
    #         shutil.rmtree(outdir)
    #     os.makedirs(outdir, exist_ok=True) 
    #     model_name = "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    #     model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    #     img_dir = "/home/ZHANGAo_2024/3dgs/data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/train/input_cached"
    #     all_img_fs = sorted(glob(os.path.join(img_dir, "*.png")))
    #     all_img_fs = [os.path.basename(f) for f in all_img_fs]
    #     all_images = []
    #     all_focals = []
    #     all_c2w = []
    #     all_pts3d = []
    #     all_mask = []
    #     all_views = all_views[:5]
    #     for vid, viewpoint_cam in enumerate(all_views):
    #         print(f"************ processing {vid}/{len(all_views)} ************")
    #         image_name, H, W, focals, w2c, c2w = self.cam_info(viewpoint_cam)
    #         if image_name not in all_img_fs:
    #             continue
    #         idx = all_img_fs.index(image_name)
    #         image_name_id = int(image_name.split(".")[0])
    #         img_fs = [os.path.join(img_dir, image_name)]
    #         idx_start = max(0, idx - 3)
    #         idx_end = min(len(all_img_fs), idx + 3)
    #         for i in range(idx_start, idx_end):
    #             image_pair_name = all_img_fs[i]
    #             image_pair_name_id = int(image_pair_name.split(".")[0])
    #             if image_pair_name_id == image_name_id:
    #                 continue
    #             if abs(image_pair_name_id - image_name_id) < 10:
    #                 img_fs.append(os.path.join(img_dir, image_pair_name))
            
    #         images = load_images(img_fs, size=512)
    #         pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    #         output = inference(pairs, model, device, batch_size=batch_size)
    #         view1, pred1 = output['view1'], output['pred1']
    #         view2, pred2 = output['view2'], output['pred2']
    #         scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    #         loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    #         # get_3D_model_from_scene(outdir, False, scene, as_pointcloud=True)
    #         imgs = scene.imgs
    #         focals = scene.get_focals()
    #         poses = scene.get_im_poses()
    #         pts3d = scene.get_pts3d()
    #         confidence_masks = scene.get_masks()

    #         fovx = viewpoint_cam.FoVx
    #         fovy = viewpoint_cam.FoVy
    #         new_H, new_W = imgs[0].shape[0], imgs[0].shape[1]
    #         fx = new_W / (2 * math.tan(fovx / 2))
    #         fy = new_H / (2 * math.tan(fovy / 2))

    #         new_focal = fx
    #         pnts = pts3d[0] * focals[0] / new_focal
    #         pnts = pnts.reshape(-1, 3)
    #         pnts = torch.cat([pnts, torch.ones(pnts.shape[0], 1, device=device)], dim=1)
    #         # pnts = torch.matmul(poses[0], pnts.t()).t() 
    #         pnts = torch.matmul(c2w, pnts.t()).t()
    #         pnts = pnts[:, :3].reshape(new_H, new_W, 3)

    #         all_images.append(imgs[0])
    #         all_focals.append(focals[0].detach().cpu().numpy())
    #         # all_c2w.append(poses[0].cpu())
    #         all_c2w.append(c2w.detach().cpu().numpy())
    #         all_pts3d.append(pnts.detach().cpu().numpy())
    #         all_mask.append(confidence_masks[0].detach().cpu().numpy())
        
    #     _convert_scene_output_to_glb(outdir, all_images, all_pts3d, all_mask, all_focals, all_c2w, as_pointcloud=True)
        
    #     all_pts3d = [pt.reshape(-1, 3) for pt in all_pts3d]
    #     all_pts3d = np.concatenate(all_pts3d, axis=0)
    #     all_images = [img.reshape(-1, 3) for img in all_images]
    #     all_images = np.concatenate(all_images, axis=0)
    #     # save all_pts3d and all_images to ply
    #     ply_path = os.path.join(outdir, "pts.ply")
    #     # write ply with rgb
    #     with open(ply_path, 'w') as f:
    #         f.write("ply\n")
    #         f.write("format ascii 1.0\n")
    #         f.write("element vertex {}\n".format(all_pts3d.shape[0]))
    #         f.write("property float x\n")
    #         f.write("property float y\n")
    #         f.write("property float z\n")
    #         f.write("property uchar red\n")
    #         f.write("property uchar green\n")
    #         f.write("property uchar blue\n")
    #         f.write("end_header\n")
    #         for i in range(all_pts3d.shape[0]):
    #             x, y, z = all_pts3d[i]
    #             r, g, b = all_images[i]
    #             r, g, b = int(r * 255), int(g * 255), int(b * 255)
    #             f.write(f"{x} {y} {z} {r} {g} {b}\n")

    #     print("debugging")
    ######################################################################
    ######################################################################
    ######################################################################

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp=False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, reset_params=True):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation}

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] 

        torch.cuda.empty_cache()
        
        return optimizable_tensors

    
    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))

        return self._xyz[idxs], self._features_dc[idxs], self._features_rest[idxs], new_opacity, new_scaling, self._rotation[idxs]


    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio
    

    def relocate_gs(self, dead_mask=None):

        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        
        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx) 
        

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1) 
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz, 
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation 
        ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs




