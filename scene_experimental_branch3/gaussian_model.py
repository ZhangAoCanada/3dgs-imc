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
from scene_experimental_branch3.imc_experimental import NetworksA
from scene_experimental_branch3 import diff_operators
from scene_experimental_branch3.utility import write_summary
from scene_experimental_branch3.torch_ransac import RANSAC, LinearRegressionModel, ScalingRegressionModel
import shutil
import math
from glob import glob
from gaussian_renderer import render, render_gsplat
import torchvision
import matplotlib.pyplot as plt
import cuml, cudf
from cuml.cluster import DBSCAN
from tqdm import tqdm



def build_rotation_grad(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda', requires_grad=True)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation_grad(s, r):
    R = build_rotation(r)

    s1 = s[:,0].view(-1, 1, 1)
    s2 = s[:,1].view(-1, 1, 1)
    s3 = s[:,2].view(-1, 1, 1)

    row1 = torch.cat([s1, torch.zeros_like(s1), torch.zeros_like(s1)], dim=2)
    row2 = torch.cat([torch.zeros_like(s2), s2, torch.zeros_like(s2)], dim=2)
    row3 = torch.cat([torch.zeros_like(s3), torch.zeros_like(s3), s3], dim=2)
    L = torch.cat([row1, row2, row3], dim=1)

    L = R @ L
    return L



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
            # in_features=3, #7, 
            # out_features=4, 
            type=self.net_type, 
            mode=self.net_mode, 
            )
        self.net.cuda()
        self.nn_gt_pts = None
        self.aligned_depth_dict = {}

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

        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init, training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations)

        #################### NOTE: hyper-param ########################
        self.imc_experimental_optimizer = torch.optim.Adam(lr=1e-4, params=self.net.parameters())
        # self.imc_experimental_optim_scheduler = torch.optim.lr_scheduler.StepLR(self.imc_experimental_optimizer, step_size=1000, gamma=0.5)
        # self.imc_experimental_optim_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.imc_experimental_optimizer, gamma=0.9)
        self.partial_scaling = training_args.partial_scaling
        self.sigma_scaling = training_args.sigma_scaling
        self.downsample_ratio = training_args.dds_ratio
        self.voxel_size = training_args.dvoxel_size
        self.eps = training_args.eps
        self.min_samples = training_args.min_samples
        self.dbscan_portion = training_args.dbscan_portion
        self.nn_type = training_args.nn_type
        self.bound_type = training_args.bound_type
        self.minmax = training_args.minmax
        self.range_scale = training_args.range_scale
        print("******************* self.minmax: ", self.minmax)

    ######################################################################
    ######################################################################
    ######################################################################
    ######################################################################
    ######################################################################
    def derivatives(self, view, tb_writer, iteration):
        if iteration < 2400:
            return None, None
        # if iteration < 10:
        #     return None, None
        # self.net.find_boundary(self.get_xyz.detach().clone(), extend_factor=0.0)

        gs_mask = self.mask_pts(self.get_xyz.shape[0])
        gs_xyz, gs_opacity, probability = self.sample_xyz(mask=gs_mask)
        gs_color = self.get_color(view, mask=gs_mask)
        # gs_norm = self.get_norm()
        attributes = torch.cat([gs_opacity, gs_color], dim=-1)
        # gs_pnts = torch.cat([gs_xyz, gs_opacity, gs_color], dim=1)
        # res = self.net(gs_pnts)
        res = self.net(gs_xyz)#, attributes=attributes)
        # pred_color_diff = res['rgb']
        # pred_opacity_diff = res['sigma']
        # pred_opacity = gs_pnts[..., 3:4] + pred_opacity_diff
        pred_color = res['rgb']
        pred_opacity = res['sigma']
        l = 1.0 - pred_opacity

        grad = torch.autograd.grad(l, [self._xyz, self._rotation, self._scaling], grad_outputs=torch.ones_like(l), create_graph=True)
        net_scale = (self.net.xyz_upperbound - self.net.xyz_lowerbound)

        # grad_net_in = diff_operators.gradient(l, res['net_in']) * (self.net.xyz_upperbound - self.net.xyz_lowerbound)
        # grad = torch.autograd.grad(gs_xyz, [self._xyz, self._rotation, self._scaling], grad_outputs=grad_net_in, create_graph=True)
        # grad = [gd * self.net.xyz_upperbound for gd in grad]
        # net_scale = 1.0

        # grad = diff_operators.gradient(l, res['net_in'])
        # net_scale = (self.net.xyz_upperbound - self.net.xyz_lowerbound)
        with torch.no_grad():
            if self.bound_type == "local":
                # xyz = self._xyz[gs_mask].detach().clone()
                # upperdist = xyz - self.net.xyz_upperbound
                # lowerdist = self.net.xyz_lowerbound - xyz
                # xyz_scaling = torch.ones_like(xyz)
                # xyz_scaling = torch.where(upperdist > 0, torch.abs(upperdist) * 0.01, 1.)
                # xyz_scaling = torch.where(lowerdist < 0, torch.abs(lowerdist) * 0.01, 1.)
                xyz = self._xyz[gs_mask].detach().clone()
                xyz = (xyz - self.net.xyz_lowerbound) / (self.net.xyz_upperbound - self.net.xyz_lowerbound) * 2.0 - 1.0
                xyz_scaling = torch.ones_like(xyz)
                xyz_scaling = torch.where(xyz > 1.0, torch.max(torch.abs(xyz / 1.0), xyz_scaling), xyz_scaling)
                xyz_scaling = torch.where(xyz < -1.0, torch.max(torch.abs(xyz / -1.0), xyz_scaling), xyz_scaling)
            else:
                xyz_scaling = 1.0

            # grad = [gd * probability.unsqueeze(-1) * self.partial_scaling for gd in grad]
            # grad = [gd * self.partial_scaling for gd in grad]

            # grad_xyz = grad[0][gs_mask] * net_scale * xyz_scaling
            grad_xyz = grad[0][gs_mask] * xyz_scaling * self.partial_scaling
            grad_rot = grad[1][gs_mask] * self.partial_scaling * self.sigma_scaling
            grad_scale = grad[2][gs_mask] * self.partial_scaling * self.sigma_scaling

            self._xyz[gs_mask].add_(grad_xyz)
            self._rotation[gs_mask].add_(grad_rot)
            self._scaling[gs_mask].add_(grad_scale)

            # # self._opacity[gs_mask].add_(self.inverse_opacity_activation(pred_opacity - gs_opacity))
            # self._opacity[gs_mask].copy_(self.inverse_opacity_activation(pred_opacity))
            # # NOTE: update the features
            # color_diff = RGB2SH(pred_color)
            # features_diff = torch.zeros((color_diff.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            # features_diff[:, :3, 0] = color_diff
            # features_diff[:, 3:, 1:] = 0.0
            # # self._features_dc[gs_mask].add_(features_diff[:,:,0:1].transpose(1, 2))
            # self._features_dc[gs_mask].copy_(features_diff[:,:,0:1].transpose(1, 2))
            if tb_writer is not None:
                tb_writer.add_scalar("nn_l/partial_grad_min", grad_xyz.min(), iteration)
                tb_writer.add_scalar("nn_l/partial_grad_max", grad_xyz.max(), iteration)
        
        # # TODO: add noise, w.r.t to covariance or L
        # pred_noise = res['xyz_noise']
        # xyz_noise = torch.bmm(covariance[gs_mask].detach().clone(), pred_noise.unsqueeze(-1)).squeeze(-1)
        # noise_full = torch.zeros((self._xyz.shape[0], 3), device="cuda")
        # noise_full[gs_mask] = xyz_noise
        # return noise_full, gs_mask

    def compute_diff(self, view, tb_writer, iteration):
        with torch.no_grad():
            # gs_mask = self.mask_pts(gs_xyz.shape[0], num_scale=10)
            gs_xyz, gs_opacity, probability = self.sample_xyz()
            gs_color = self.get_color(view)
            # gs_xyz, gs_opacity = self.get_xyz, self.get_opacity
            if iteration > 2400:
                # res = self.net(gs_xyz[gs_mask])
                # opacity_diff_mask = res['sigma'] - gs_opacity[gs_mask]
                # color_diff_mask = res['rgb'] - gs_color[gs_mask]
                # # random select
                # opacity_diff = torch.zeros_like(gs_opacity)
                # color_diff = torch.zeros_like(gs_color)
                # opacity_diff[gs_mask] = opacity_diff_mask
                # color_diff[gs_mask] = color_diff_mask
                res = self.net(gs_xyz)
                opacity_diff = res['sigma'] - gs_opacity
                color_diff = res['rgb'] - gs_color
            else:
                opacity_diff = 1.0 - gs_opacity
                color_diff = torch.zeros_like(gs_color) + 1e-3
        
        # opacity_diff_abs = torch.abs(opacity_diff)
        # color_diff_abs = torch.abs(color_diff)
        opacity_diff_abs = torch.abs(opacity_diff) * probability
        color_diff_abs = torch.abs(color_diff) * probability
        return opacity_diff_abs, color_diff_abs
        
    def train_continuous(self, all_views, view, pipe, bg, tb_writer, iteration):
        if iteration < 2000:
            return None
        elif iteration % 1000 == 0:
            self.update_nnpts(all_views, pipe, bg, align=False)
            # self.max_num += 10
        if self.net.xyz_lowerbound is None or self.net.xyz_upperbound is None:
            return None
        self.net.train()
        if self.nn_type == "allviews":
            pts = self.nn_gt_pts
        else:
            with torch.no_grad():
                pts = self.filter_depth_and_align(view, pipe, bg, align=True, debug=False, aligndepth=True)
        if pts is None:
            return None
        assert pts.shape[1] == 9
        mask = self.mask_pts(pts.shape[0])
        pts = pts[mask].clone().requires_grad_(True)
        xyz, color, norm = torch.split(pts, [3, 3, 3], dim=1)
        opacity = torch.ones((xyz.shape[0], 1), device="cuda")
        # if self.nn_type == "allviews":
        #     off_xyz, off_color, off_norm, off_opacity = self.spawn_randompnts(xyz.shape[0], self.voxel_size)
        # else:
        #     H, W, focals, c2w = self.viewcam_properties(view, self.downsample_ratio)
        #     off_xyz, off_color, off_norm, off_opacity = self.spawn_randomraypnts(mask, H, W, focals, c2w, voxel_size=self.voxel_size)

        pert_xyz, pert_color, pert_norm, pert_opacity = self.perturb_pnts_whole(xyz, color, norm, opacity)
        # if self.minmax == "whole":
        #     pert_xyz, pert_color, pert_norm, pert_opacity = self.perturb_pnts_whole(xyz, color, norm, opacity)
        # else:
        #     pert_xyz, pert_color, pert_norm, pert_opacity = self.perturb_pnts(xyz, color, norm, opacity)

        gt_opacity = torch.cat([opacity, pert_opacity], dim=0)
        gt_color = torch.cat([color, pert_color], dim=0)
        gt_norm = torch.cat([norm, pert_norm], dim=0)
        net_in = torch.cat([xyz, pert_xyz], dim=0)
        # gt_opacity = torch.cat([opacity, pert_opacity, off_opacity], dim=0)
        # gt_color = torch.cat([color, pert_color, off_color], dim=0)
        # gt_norm = torch.cat([norm, pert_norm, off_norm], dim=0)
        # net_in = torch.cat([xyz, pert_xyz, off_xyz], dim=0)

        # gt_opacity = torch.cat([opacity, off_opacity], dim=0)
        # gt_color = torch.cat([color, off_color], dim=0)
        # gt_norm = torch.cat([norm, off_norm], dim=0)
        # net_in = torch.cat([xyz, off_xyz], dim=0)

        res = self.net(net_in)
        l = self.loss_fn(res, gt_opacity, gt_color, gt_norm)
        if iteration % 500 == 0:
            write_summary(self.net, tb_writer, iteration)
        if tb_writer is not None:
            tb_writer.add_scalar("nn_l/nnl", l, iteration)
        return l
    
    def perturb_pnts(self, xyz, color, norm, opacity, range_scale=1.):
        # # randn_probs = (1 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * randn_vals ** 2)
        # randn_vals = torch.randn_like(opacity)
        # randn_probs = torch.exp(-0.5 * randn_vals ** 2)
        randn_vals = torch.randn_like(xyz)
        randn_probs = torch.exp(-0.5 * torch.bmm(randn_vals.unsqueeze(1), randn_vals.unsqueeze(2)).squeeze(-1))

        # dist_range = (self.net.xyz_upperbound - self.net.xyz_lowerbound) * range_scale
        if self.minmax == "min":
            dist_range = torch.min(torch.abs(self.net.xyz_upperbound - xyz).min(dim=-1, keepdim=True).values, torch.abs(self.net.xyz_lowerbound - xyz).min(dim=-1, keepdim=True).values) * range_scale
        elif self.minmax == "max":
            # dist_range = torch.min(torch.abs(self.net.xyz_upperbound - xyz).min(dim=-1, keepdim=True).values, torch.abs(self.net.xyz_lowerbound - xyz).min(dim=-1, keepdim=True).values) * range_scale
            dist_range = torch.max(torch.abs(self.net.xyz_upperbound - xyz).max(dim=-1, keepdim=True).values, torch.abs(self.net.xyz_lowerbound - xyz).max(dim=-1, keepdim=True).values) * range_scale
        else:
            dist_range = (self.net.xyz_upperbound - self.net.xyz_lowerbound) * range_scale

        pert_xyz = xyz.clone() + randn_vals * norm * dist_range
        pert_color = color.clone()
        pert_norm = norm.clone()
        pert_opacity = opacity.clone() * randn_probs
        return pert_xyz, pert_color, pert_norm, pert_opacity

    def perturb_pnts_whole(self, xyz, color, norm, opacity):
        rand_ = torch.rand_like(xyz).requires_grad_(True)
        xyz_ = (xyz - self.net.xyz_lowerbound) / (self.net.xyz_upperbound - self.net.xyz_lowerbound)
        rand_vals = rand_ * (self.net.xyz_upperbound - self.net.xyz_lowerbound) + self.net.xyz_lowerbound
        if self.minmax == "whole" or self.minmax == "wholeonly":
            diff_vals = rand_vals - xyz
        else:
            diff_vals = (rand_ - xyz_) * self.range_scale
        rand_probs = torch.exp(-0.5 * torch.bmm(diff_vals.unsqueeze(1), diff_vals.unsqueeze(2)).squeeze(-1))
        pert_xyz = rand_vals
        pert_color = color.clone()
        pert_norm = norm.clone()
        pert_opacity = opacity.clone() * rand_probs
        return pert_xyz, pert_color, pert_norm, pert_opacity

    def loss_fn(self, pred, gt_opacity, gt_color, gt_norm):
        net_in = pred['net_in']
        pred_color = pred['rgb']
        pred_opacity = pred['sigma']

        # gradient = diff_operators.gradient(pred_opacity, net_in)
        # opacity_constraint = torch.where(gt_opacity != 0, F.l1_loss(pred_opacity, gt_opacity), torch.zeros_like(pred_opacity)).mean()
        # color_constraint = torch.where(gt_opacity != 0, F.l1_loss(pred_color, gt_color), torch.zeros_like(pred_color)).mean()
        # normal_constraint = torch.where(gt_opacity != 0, 1. - F.cosine_similarity(gradient, gt_norm, dim=-1)[..., None], torch.zeros_like(gradient)).mean()
        # gradient_constraint = torch.abs(gradient.norm(dim=-1) - 1).mean()
        # inter_constraint = torch.where(gt_opacity != 0, torch.zeros_like(pred_opacity), pred_opacity).mean()

        # l = opacity_constraint * 3e3 + \
        #     color_constraint * 1e3 + \
        #     normal_constraint * 1e2 + \
        #     inter_constraint * 1e2 + \
        #     gradient_constraint * 5e1

        gradient = diff_operators.gradient(pred_opacity, net_in)
        opacity_constraint = F.l1_loss(pred_opacity, gt_opacity).mean()
        color_constraint = F.l1_loss(pred_color, gt_color).mean()
        normal_constraint = (1. - F.cosine_similarity(gradient, gt_norm, dim=-1)[..., None]).mean()
        gradient_constraint = torch.abs(gradient.norm(dim=-1) - 1).mean()
        l = opacity_constraint * 3e3 + \
            color_constraint * 1e3 + \
            normal_constraint * 1e2 + \
            gradient_constraint * 5e1

        return l
    
    def mask_pts(self, pts_num, num_scale=1.0):
        if pts_num > self.max_num * num_scale:
            mask = torch.zeros(pts_num, dtype=torch.bool, device="cuda")
            random_index = torch.randperm(pts_num)[:self.max_num]
            mask[random_index] = True
            return mask
        else:
            return torch.ones(pts_num, dtype=torch.bool, device="cuda")

    def sample_xyz(self, mask=None, scaling_modifier=1.0):
        if mask is None:
            mask = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        means = self.get_xyz[mask]
        # scaling = self.get_scaling
        # rot = self.get_rotation
        # L = build_scaling_rotation_grad(scaling, rot)
        L = build_scaling_rotation(self.get_scaling[mask], self.get_rotation[mask])
        covariance = L @ L.transpose(1, 2)
        # TODO: is this L the same as the one above?
        # L = torch.linalg.cholesky(covariance)
        z = torch.randn_like(means) * scaling_modifier
        #######################################################
        # epsilon = torch.bmm(z.unsqueeze(1), L.transpose(1, 2)).squeeze(1)
        # epsilon = torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)
        epsilon = torch.bmm(covariance, z.unsqueeze(-1)).squeeze(-1)
        ########################################################
        samples = means + epsilon
        # NOTE: compute (x - μ)^T · Σ^(-1) · (x - μ)
        # cov_inv = torch.linalg.inv(covariance)
        cov_inv = torch.inverse(covariance + 1e-6 * torch.eye(3, device="cuda"))
        mahalanobis_dist = torch.sum(torch.bmm(epsilon.unsqueeze(1), cov_inv).squeeze(1) * epsilon, dim=1)
        probs = torch.exp(-0.5 * mahalanobis_dist)
        opacities = self.get_opacity[mask] * probs[..., None]
        return samples, opacities, probs
    
    def get_color(self, view, mask=None):
        if mask is None:
            mask = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        shs_view = self.get_features[mask].transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        dir_pp = (self.get_xyz[mask] - view.camera_center.repeat(self.get_features[mask].shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return colors_precomp
    
    def get_norm(self, ):
        scaling = self.get_scaling
        rot = self.get_rotation
        norm_axis = torch.argmin(scaling, dim=1)
        norm = torch.zeros_like(scaling)
        norm[torch.arange(scaling.shape[0]), norm_axis] = 1
        rot_mat = build_rotation(rot)
        normal = torch.bmm(norm.unsqueeze(1), rot_mat.transpose(1, 2)).squeeze(1)
        normal = F.normalize(normal, p=2, dim=-1)
        return normal

    def spawn_randompnts(self, num_pnts=100000, voxel_size=0.01):
        upper_bound = self.net.xyz_upperbound
        lower_bound = self.net.xyz_lowerbound
        off_pnts = torch.rand((num_pnts, 3), device="cuda") * (upper_bound - lower_bound) + lower_bound
        off_xyz = nn.Parameter(off_pnts.contiguous().requires_grad_(True))
        off_opacity = torch.zeros((num_pnts, 1), device="cuda")
        off_color = torch.rand((num_pnts, 3), device="cuda")
        off_norm = torch.rand((num_pnts, 3), device="cuda")
        return off_xyz, off_color, off_norm, off_opacity
    
    def spawn_randomraypnts(self, mask, H, W, focals, c2w, times=1, voxel_size=0.01):
        times = int(times)
        fx, fy = focals
        i, j = torch.meshgrid(torch.arange(W, device="cuda"), torch.arange(H, device="cuda"), indexing='xy')
        i, j = i.reshape(-1), j.reshape(-1)
        dirs = torch.stack([(i-W*.5)/fx, (j-H*.5)/fy, torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        near = 0.01
        far = torch.norm(self.net.xyz_upperbound.expand(3) - self.net.xyz_lowerbound.expand(3))
        rand_dist = torch.rand(times, H*W, 1, device="cuda") * (far - near) + near
        off_pts = rays_o.unsqueeze(0) + (rays_d.unsqueeze(0) * rand_dist)
        
        mask = mask.view(1, -1).expand(times, -1)
        off_pts = off_pts[mask].reshape(-1, 3)
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
        
    def update_nnpts(self, all_views, pipe, bg, align=True):
        assert self.nn_type in ["singleview", "allviews"]
        assert self.bound_type in ["local", "global"]
        aligndepth = True if self.nn_type == "allviews" else False

        if self.nn_type == "allviews" or self.bound_type == "local":
            self.nn_gt_pts = self.pointdepth(all_views, pipe, bg, align, debug=False, aligndepth=aligndepth)
            if self.nn_gt_pts is None:
                return
        if self.bound_type == "local":
            self.net.find_boundary(self.nn_gt_pts)
        else:
            self.net.find_boundary(self.get_xyz.detach().clone(), extend_factor=0.0)

    def pointdepth(self, all_views, pipe, bg, align=True, debug=True, aligndepth=False):
        if debug:
            self.nn_gt_pts = None
            self.aligned_depth_dict = {}
            self.downsample_ratio = 4
            self.voxel_size = 0.001
            self.eps = 0.04 # 0.04
            self.min_samples = 50 # 100
            all_views = all_views[:100]

        all_pts = torch.empty(0, device="cuda")
        with torch.no_grad():
            for i in tqdm(range(len(all_views))):
                view = all_views[i]
                pt_clr = self.filter_depth_and_align(view, pipe, bg, align=align, debug=debug, aligndepth=aligndepth)
                if pt_clr is None:
                    continue
                pt_clr = pt_clr.detach().clone()
                all_pts = torch.cat([all_pts, pt_clr], dim=0)
                all_pts = self.unique_pts(all_pts, self.voxel_size)
        if debug:
            # save to ply
            ply_path = os.path.join("tmp", "pointdepth.ply")
            self.save_to_ply(all_pts[..., :6], ply_path)
            print("debugging")
        if all_pts.shape[0] == 0:
            return None
        return all_pts
    
    def filter_depth_and_align(self, view, pipe, bg, align=True, debug=True, aligndepth=False):
        # if int(view.image_name.split(".")[0]) in [248, 249]:
        if int(view.image_name.split(".")[0]) >= 1171:
            return None
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

        # TODO: try to align mono_depth with render_depth_gsplat
        if align:
            # NOTE: dbscan with gpu
            pts = self.depth_proj(render_depth_gsplat, H, W, focals, c2w)
            pts_cudf = cudf.DataFrame(pts.detach().clone().cpu().numpy())
            db = DBSCAN(
                eps=self.eps, 
                min_samples=self.min_samples, 
                max_mbytes_per_batch=5000
                    ).fit(pts_cudf, out_dtype='int32')
                    # ).fit(pts_cudf, out_dtype='int64')
            labels = db.labels_
            labels = torch.tensor(labels, device="cuda", dtype=torch.float)
            mask = labels.detach().clone() != -1
            del db, labels, pts_cudf
            # torch.cuda.empty_cache()

            if mask.sum() < self.dbscan_portion * H * W:
                return None
            align_depth = self.depth_align(render_depth_gsplat, mono_invdepth, mask.reshape(1, H, W), debug=debug, name=view.image_name)
            
            if align_depth is None:
                return None
            self.aligned_depth_dict[view.image_name] = align_depth.detach().clone()
            pts = self.depth_proj(align_depth, H, W, focals, c2w)
        else:
            # NOTE: dbscan with gpu
            pts = self.depth_proj(render_depth_gsplat, H, W, focals, c2w)
            pts_cudf = cudf.DataFrame(pts.detach().clone().cpu().numpy())
            db = DBSCAN(
                eps=self.eps, 
                min_samples=self.min_samples, 
                max_mbytes_per_batch=5000
                    ).fit(pts_cudf, out_dtype='int32')
                    # ).fit(pts_cudf, out_dtype='int64')
            labels = db.labels_
            labels = torch.tensor(labels, device="cuda", dtype=torch.float)
            mask = labels.detach().clone() != -1
            del db, labels, pts_cudf
            torch.cuda.empty_cache()

            if mask.sum() < self.dbscan_portion * H * W:
                return None
            if aligndepth:
                align_depth = self.depth_align(render_depth_gsplat, mono_invdepth, mask.reshape(1, H, W), debug=debug)
                if align_depth is None:
                    return None
                if align_depth is not None:
                    self.aligned_depth_dict[view.image_name] = align_depth.detach().clone()

        # compute normal map
        mono_depth = align_depth if align_depth is not None else 1.0 / (mono_invdepth + 1e-6)
        scale_factor = 100.0 if align_depth is not None else 1.0
        mono_normal = self.depth2norm(mono_depth, scale_factor=scale_factor).squeeze(0)
        # visualize
        if debug:
            mask_show = mask.reshape(H, W).float().unsqueeze(0).repeat(3, 1, 1) if mask is not None else torch.ones(3, H, W, device="cuda").float()
            mono_depth_show = (mono_depth / mono_depth.max()).repeat(3, 1, 1)
            img_show = torch.cat([mono_depth_show, mono_normal, mask_show], dim=1)
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
        return pt_clr
    
    def depth_align(self, render_depth, mono_invdepth, mask=None, debug=True, name=None):
        """
        Mono_depth is normally with unkown scale. The function here is to compute the correct scale with render_depth, so that the two depth maps can be aligned. render_depth is somehow noisy. Therefore, we need to use mask to filter out the noisy points.

        @param render_depth: [1, H, W]
        @param mono_invdepth: [1, H, W]
        @param mask: [1, H, W]
        """
        if mask is None:
            mask = torch.ones_like(render_depth, dtype=torch.bool, device="cuda")

        # NOTE: remove the zero points in render_depth
        mask_nonzero = torch.where(render_depth < 1e-6, torch.zeros_like(render_depth, dtype=torch.bool), torch.ones_like(render_depth, dtype=torch.bool))
        mask = mask & mask_nonzero

        render_invdepth = 1.0 / (render_depth + 1e-6)
        render_inv = render_invdepth[mask].reshape(-1, 1)
        mono_inv = mono_invdepth[mask].reshape(-1, 1)
        linear = LinearRegressionModel().fit(mono_inv, render_inv)
        if linear.weights[0][0] < 0:
            return None
        _, H, W = render_depth.shape
        align_invdepth = linear.predict(mono_invdepth.reshape(-1, 1)).reshape(1, H, W)
        if align_invdepth.min() < 0:
            return None

        align_depth = 1.0 / (align_invdepth + 1e-6)
        if debug:
            align_depth_show = (align_depth - align_depth.min()) / (align_depth.max() - align_depth.min())
            render_depth_show = (render_depth - align_depth.min()) / (align_depth.max() - align_depth.min())
            render_depth_show = torch.clamp(render_depth_show, 0.0, 1.0)
            image_show = torch.cat([render_depth_show, align_depth_show], dim=1) 
            if name is not None:
                torchvision.utils.save_image(image_show, f"tmp/aligned_{name}.png")
            else:
                torchvision.utils.save_image(image_show, f"tmp/align_depth.png")
        return align_depth

    # def depth_align_ransac(self, render_depth, mono_invdepth, mask=None, debug=True):
    #     """
    #     Mono_depth is normally with unkown scale. The function here is to compute the correct scale with render_depth, so that the two depth maps can be aligned. render_depth is somehow noisy. Therefore, we need to use mask to filter out the noisy points.

    #     @param render_depth: [1, H, W]
    #     @param mono_invdepth: [1, H, W]
    #     @param mask: [1, H, W]
    #     """
    #     if mask is None:
    #         mask = torch.ones_like(render_depth, dtype=torch.bool, device="cuda")
    #     render_invdepth = 1.0 / (render_depth + 1e-2)
    #     render_inv = render_invdepth[mask].reshape(-1, 1)
    #     mono_inv = mono_invdepth[mask].reshape(-1, 1)
    #     ransac = RANSAC(
    #                 model_class=LinearRegressionModel, 
    #                 max_iterations=100, 
    #                 threshold=0.01, 
    #                 min_samples=0.5,
    #             )
    #     ransac.fit(mono_inv, render_inv)
    #     if ransac.best_model.weights[0][0] < 0:
    #         return None
    #     _, H, W = render_depth.shape
    #     align_invdepth = ransac.predict(mono_invdepth.reshape(-1, 1)).reshape(1, H, W)
    #     align_depth = 1.0 / (align_invdepth + 1e-6)
    #     if debug:
    #         align_depth_show = (align_depth - align_depth.min()) / (align_depth.max() - align_depth.min())
    #         render_depth_show = (render_depth - align_depth.min()) / (align_depth.max() - align_depth.min())
    #         render_depth_show = torch.clamp(render_depth_show, 0.0, 1.0)
    #         image_show = torch.cat([render_depth_show, align_depth_show], dim=1) 
    #         torchvision.utils.save_image(image_show, "tmp/align_depth.png")
    #     return align_depth
    
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
    ######################################################################
    ######################################################################
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



