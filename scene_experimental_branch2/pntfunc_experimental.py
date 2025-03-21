import os, sys
import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import math

from sklearn import metrics
from sklearn.cluster import DBSCAN


class NetworkAnother(nn.Module):
    """
    Our target is to create a network that map (x, y, z) to (r, g, b, a)
    pos_enc: ["none", "basic", "pos", "gaussian"]
    """
    def __init__(self, 
                 xyz_bounds, 
                 in_channels=3, 
                 out_channels=4, 
                 hidden_features=256, 
                 hidden_layers=5, 
                 batch_size=10000, 
                 pos_enc="basic"):
        super(NetworkAnother, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.find_boundary(xyz_bounds)

        self.pos_enc = pos_enc
        if self.pos_enc != "none":
            self.pos_enc = PosGaussianEncoding(in_channels=in_channels, mode=pos_enc)
            self.in_channels = self.pos_enc.enc_channels

        # Define the network
        self.layers = []
        self.layers.append(nn.Linear(self.in_channels, hidden_features))
        self.layers.append(nn.ReLU())
        for i in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_features, out_channels))
        self.net = nn.Sequential(*self.layers)

        self.sigma_act = nn.ReLU()
        self.rgb_act = nn.Sigmoid()
        self.downsample_ratio = 4
        # self.batch_size = batch_size
        self.batch_size = 10000
        self.near = 0.1
        self.far = 20.0
        self.N_samples = 100
        self.pts = None
    
    def process(self, x):
        if self.pos_enc != "none":
            x_enc = self.pos_enc(x)
        else:
            x_enc = x
        raw = self.net(x_enc)

        rgb, sigma_a = raw[:, :3], raw[:, 3]
        sigma_pred = self.sigma_act(sigma_a)
        rgb_pred = self.rgb_act(rgb)

        return {
                'rgb': rgb_pred,
                'sigma': sigma_pred
            }
    
    def forward(self, viewpoint_cam, train=True):
        H, W, focals, c2w = self.viewcam_properties(viewpoint_cam)
        rays_o, rays_d = self.get_rays(H, W, focals, c2w)
        if train:
            gt_image = viewpoint_cam.original_image.cuda()
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            rays_o, rays_d, gt_image, gt_invdepth = self.batchify_rays(rays_o, rays_d, gt_image, mono_invdepth)
        rgb_map, inv_depth_map, acc_map = self.render_rays(rays_o, rays_d)
        if train:
            loss = self.loss_fn(rgb_map, inv_depth_map, acc_map, gt_image, gt_invdepth)
            return loss
        rgb_map = rgb_map.reshape(H, W, 3).permute(2, 0, 1)
        inv_depth_map = inv_depth_map.reshape(H, W).unsqueeze(0)
        acc_map = acc_map.reshape(H, W).unsqueeze(0)
        return rgb_map, inv_depth_map, acc_map
    
    def train_pts(self, viewpoint_cam, render_depth):
        H, W, focals, c2w = self.viewcam_properties(viewpoint_cam)
        rays_o, rays_d = self.get_rays(H, W, focals, c2w)
        rendepth = F.interpolate(render_depth.unsqueeze(0), size=None, scale_factor=1/self.downsample_ratio, mode='bilinear', align_corners=False).squeeze(0)
        pts = self.depth_proj(rendepth, rays_o, rays_d)
        pts_np = pts.detach().cpu().numpy()
        db = DBSCAN(eps=0.05, min_samples=100).fit(pts_np)
        labels = db.labels_
        labels = torch.tensor(labels, device="cuda", dtype=torch.float)
        mask = labels != -1
        mask = mask.reshape(H, W)
        gt_image = viewpoint_cam.original_image.cuda()
        rays_o, rays_d, gt_rgb, gt_alpha = self.batchify_pts(rays_o, rays_d, gt_image, pts, mask)
    
    def batchify_pts(self, rays_o, rays_d, gt_image, pts, mask):
        return
    
    def depth_proj(self, depth, rays_o, rays_d):
        pts = rays_o + rays_d * depth.reshape(1, -1).t()
        return pts
    
    def viewcam_properties(self, viewpoint_cam):
        fovx = viewpoint_cam.FoVx
        fovy = viewpoint_cam.FoVy
        W = viewpoint_cam.image_width // self.downsample_ratio
        H = viewpoint_cam.image_height // self.downsample_ratio
        fx = W / (2 * math.tan(fovx / 2))
        fy = H / (2 * math.tan(fovy / 2))
        K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device="cuda")
        w2c = viewpoint_cam.world_view_transform.transpose(0, 1)
        c2w = torch.inverse(w2c) 
        return H, W, [fx, fy], c2w
    
    def get_rays(self, H, W, focals, c2w, rand=False):
        i, j = torch.meshgrid(torch.arange(W, device="cuda"), torch.arange(H, device="cuda"))
        i, j = i.reshape(-1), j.reshape(-1)
        if rand:
            i += torch.rand_like(i)
            j += torch.rand_like(j)
        dirs = torch.stack([(i-W*.5)/focals[0], -(j-H*.5)/focals[1], -torch.ones_like(i)], 1)
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        return rays_o, rays_d
    
    def batchify_rays(self, rays_o, rays_d, gt_image, mono_invdepth):
        """
        select number of rays to process in parallel
        """
        # downsample 
        gt_image = F.interpolate(gt_image.unsqueeze(0), size=None, scale_factor=1/self.downsample_ratio, mode='bilinear', align_corners=False).squeeze(0)
        gt = gt_image.reshape(3, -1).T
        gt_invdepth = F.interpolate(mono_invdepth.unsqueeze(0), size=None, scale_factor=1/self.downsample_ratio, mode='bilinear', align_corners=False).squeeze(0)
        gt_invdepth = gt_invdepth.reshape(-1)
        assert gt.shape[0] == rays_o.shape[0]
        # random index for batch
        if gt.shape[0] > self.batch_size:
            rand_idx = torch.randperm(gt.shape[0])
            rays_o = rays_o[rand_idx[:self.batch_size]]
            rays_d = rays_d[rand_idx[:self.batch_size]]
            gt = gt[rand_idx[:self.batch_size]]
            gt_invdepth = gt_invdepth[rand_idx[:self.batch_size]]
        return rays_o, rays_d, gt, gt_invdepth

    def render_rays(self, rays_o, rays_d, rand=False):
        if rand:
            z_vals += torch.rand(list(rays_o.shape[0]) + [self.N_samples], device="cuda") * (self.far - self.near) / self.N_samples
        else:
            z_vals = torch.linspace(self.near, self.far, self.N_samples, device="cuda")
            z_vals = z_vals.expand(rays_o.shape[0], self.N_samples)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts_flat = pts.reshape(-1, 3)
        num = self.batch_size * self.N_samples
        if pts_flat.shape[0] > num:
            pts_flat = torch.split(pts_flat, num, dim=0)
            rgb = []
            sigma = []
            for pts_b in pts_flat:
                raw = self.process(pts_b)
                rgb.append(raw['rgb'])
                sigma.append(raw['sigma'])
            rgb = torch.cat(rgb, 0)
            sigma = torch.cat(sigma, 0)
        else:
            raw = self.process(pts_flat)
            rgb = raw['rgb']#.reshape(pts.shape[0], N_samples, 3)
            sigma = raw['sigma']#.reshape(pts.shape[0], N_samples)
        rgb = rgb.reshape(pts.shape[0], self.N_samples, 3)
        sigma = sigma.reshape(pts.shape[0], self.N_samples)
        ### NOTE: do volume rendering
        dist = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.tensor([1e10], device="cuda").expand(rays_o.shape[0], 1)], -1)
        alpha = 1. - torch.exp(-sigma * dist)
        trans_raw = torch.min(torch.tensor([1.], device="cuda"), 1. - alpha + 1e-10)
        trans = torch.cat([torch.ones_like(trans_raw[..., :1]), trans_raw[..., :-1]], -1)
        weights = alpha * torch.cumprod(trans, -1)
        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        inv_depth_map = torch.sum(weights / (z_vals+1e-6), -1)
        acc_map = torch.sum(weights, -1)
        return rgb_map, inv_depth_map, acc_map
    
    def loss_fn(self, rgb_map, inv_depth_map, acc_map, gt_image, invdepth_gt):
        rgb_loss = F.mse_loss(rgb_map, gt_image)
        ### NOTE: normalize depth ###
        invdepth_min = inv_depth_map.min()
        invdepth_max = inv_depth_map.max()
        invdepth_gt_min = invdepth_gt.min()
        invdepth_gt_max = invdepth_gt.max()
        inv_depth_map = (inv_depth_map - invdepth_min) / (invdepth_max - invdepth_min)
        invdepth_gt = (invdepth_gt - invdepth_gt_min) / (invdepth_gt_max - invdepth_gt_min)
        depth_loss = F.mse_loss(inv_depth_map, invdepth_gt)
        # acc_loss = F.mse_loss(acc_map, torch.ones_like(acc_map))
        return rgb_loss + depth_loss

    def find_boundary(self, xyz_bounds):
        self.xyz_lowerbound, self.xyz_upperbound = xyz_bounds
        self.xyz_upperbound = self.xyz_upperbound.max()
        self.xyz_lowerbound = self.xyz_lowerbound.min()
        # self.boundary = self.xyz_upperbound - self.xyz_lowerbound
        # self.xyz_lowerbound -= 0.1 * self.boundary
        # self.xyz_upperbound += 0.1 * self.boundary



class PosGaussianEncoding(nn.Module):
    """
    mode: ["basic", "pos", "gaussian"]
    """
    def __init__(self, in_channels=3, gaussian_scales=38, seed=8, max_posenc_log_scale=8, mode="gaussian"):
        super().__init__()
        if mode == "basic":
            self.bvals = torch.eye(in_channels).cuda()
            self.avals = torch.ones(in_channels).cuda()
            self.enc_channels = 2 * len(self.bvals)
        elif mode == "pos":
            self.bvals = 2. ** torch.arange(max_posenc_log_scale).float().cuda()
            self.bvals = torch.reshape(torch.eye(in_channels) * self.bvals[:, None, None], [len(self.bvals)*in_channels, in_channels])
            self.avals = torch.ones(self.bvals.shape[0]).cuda()
            self.enc_channels = 2 * len(self.bvals)
        elif mode == "gausian":
            self.scales = gaussian_scales
            self.seed = seed
            torch.manual_seed(seed)
            self.bvals = torch.randn(gaussian_scales, in_channels).cuda()
            self.avals = torch.ones(gaussian_scales).cuda()
            self.enc_channels = 2 * len(self.bvals)
        else:
            raise NotImplementedError
    
    def forward(self, x):
        """
        x: (N, in_channels)
        """
        pts_flat = torch.cat([self.avals * torch.sin(x @ self.bvals.T), self.avals * torch.cos(x @ self.bvals.T)], -1)
        return pts_flat

