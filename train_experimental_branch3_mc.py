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

import os
import shutil
import json
import math
import torch
import torch.nn.functional as F
from random import randint, shuffle
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_noise
import sys
from scene_experimental_branch3 import Scene, GaussianModel
from scene_experimental_branch3.gaussian_model import build_scaling_rotation
# from scene import Scene, GaussianModel
# from scene.gaussian_model import build_scaling_rotation
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
import time


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    load_iteration = getattr(dataset, "load_iteration", 0)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration if load_iteration else None)
    if scene.loaded_iter and not checkpoint:
        first_iter = scene.loaded_iter
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # if opt.nndebug:
    #     if os.path.exists("tmp"):
    #         shutil.rmtree("tmp")
    #     os.makedirs("tmp", exist_ok=True)

    random_view_num = opt.random_view_num
    stochastic_stop_logged = False

    print(f"[INFO] num train views: {len(scene.getTrainCameras())}, num test views: {len(scene.getTestCameras())}")

    timing_enabled = getattr(opt, "timing_breakdown", False)
    timing_sync = getattr(opt, "timing_sync", False)
    timing_breakdown = {
        "view_selection": 0.0,
        "cpf_derivatives": 0.0,
        "render_loss": 0.0,
        "cpf_training": 0.0,
        "backward": 0.0,
        "report_save": 0.0,
        "noise_densify": 0.0,
        "optimizer": 0.0,
        "iterations": 0,
    }

    def timing_mark():
        if timing_enabled and timing_sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    def timing_add(name, start_time):
        if not timing_enabled:
            return
        if timing_sync:
            torch.cuda.synchronize()
        timing_breakdown[name] += time.perf_counter() - start_time

    total_time_start = time.time()

    for iteration in range(first_iter, opt.iterations + 1):
        iter_timing_start = timing_mark()
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # # Pick a random Camera
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()
        #     viewpoint_indices = list(range(len(viewpoint_stack)))
        # rand_idx = randint(0, len(viewpoint_indices) - 1)
        # viewpoint_cam = viewpoint_stack.pop(rand_idx)
        # vind = viewpoint_indices.pop(rand_idx)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        if random_view_num <= 1:
            rand_idx = randint(0, len(viewpoint_stack) - 1)
            viewpoint_cameras = [viewpoint_stack.pop(rand_idx)]
        else:
            shuffle(viewpoint_stack)
            actual_view_num = min(random_view_num, len(viewpoint_stack))
            viewpoint_cameras = viewpoint_stack[:actual_view_num]
            viewpoint_stack = viewpoint_stack[actual_view_num:]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        timing_add("view_selection", iter_timing_start)

        ############### NOTE: IMC ###############
        # gaussians.derivatives(viewpoint_cam, tb_writer, iteration)
        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)

        segment_start = timing_mark()
        noise, noise_mask = gaussians.derivatives(viewpoint_cameras[0], tb_writer, iteration)
        timing_add("cpf_derivatives", segment_start)

        segment_start = timing_mark()
        total_l = 0.0
        for viewpoint_cam in viewpoint_cameras:
            render_pkg = render_noise(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, noise=noise)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            residual_cache_active = (
                int(getattr(opt, "cpf_residual_cache", 0)) != 0
                and iteration >= int(getattr(opt, "cpf_residual_cache_start_iter", 0))
                and (int(getattr(opt, "cpf_residual_cache_until_iter", 0)) <= 0 or iteration <= int(getattr(opt, "cpf_residual_cache_until_iter", 0)))
                and iteration % max(1, int(getattr(opt, "cpf_residual_cache_interval", 1))) == 0
            )
            if residual_cache_active:
                gaussians.cache_residual_candidates(
                    viewpoint_cam,
                    image,
                    gt_image,
                    max_pixels=int(getattr(opt, "cpf_residual_cache_pixels", 1024)),
                )
            
            ################# NOTE: MCMC ##################
            loss = loss + args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
            loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()

            # Depth regularization
            Ll1depth_pure = 0.0
            current_depth_l1_weight = depth_l1_weight(iteration)
            depth_l1_stop_iter = getattr(opt, "depth_l1_stop_iter", 0)
            if depth_l1_stop_iter > 0 and iteration > depth_l1_stop_iter:
                current_depth_l1_weight = 0.0
            if current_depth_l1_weight > 0 and viewpoint_cam.depth_reliable:
                if (
                    int(getattr(opt, "depth_align_sfm_on_demand", 0)) != 0
                    and viewpoint_cam.image_name not in gaussians.aligned_depth_dict
                    and viewpoint_cam.invdepthmap is not None
                ):
                    mono_for_align = viewpoint_cam.invdepthmap.cuda()
                    H, W = mono_for_align.shape[-2:]
                    fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2))
                    fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2))
                    aligned_depth = gaussians.depth_align_sfm(mono_for_align, viewpoint_cam, H, W, [fx, fy])
                    if aligned_depth is not None:
                        gaussians.aligned_depth_dict[viewpoint_cam.image_name] = aligned_depth.detach()

                if viewpoint_cam.image_name not in gaussians.aligned_depth_dict:
                    invDepth = render_pkg["depth"]
                    mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                    depth_mask = viewpoint_cam.depth_mask.cuda()

                    if opt.depth_normalize:
                        mono_min = mono_invdepth.min()
                        mono_max = mono_invdepth.max()
                        inDepth_min = invDepth.min()
                        inDepth_max = invDepth.max()
                        mono_invdepth = (mono_invdepth - mono_min) / (mono_max - mono_min + 1e-6)
                        invDepth = (invDepth - inDepth_min) / (inDepth_max - inDepth_min + 1e-6)
                    Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                else:
                    aligned_depth = gaussians.aligned_depth_dict[viewpoint_cam.image_name]
                    downsample = gaussians.downsample_ratio
                    # aligned_depth = F.interpolate(aligned_depth.unsqueeze(0), size=None, scale_factor=downsample, mode='bilinear', align_corners=False).squeeze(0)
                    aligned_depth = F.interpolate(aligned_depth.unsqueeze(0), size=viewpoint_cam.invdepthmap.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
                    aligned_invdepth = 1.0 / (aligned_depth + 1e-4)
                    render_invdepth = render_pkg["depth"]
                    depth_mask = viewpoint_cam.depth_mask.cuda()
                    Ll1depth_pure = torch.abs((render_invdepth - aligned_invdepth) * depth_mask).mean()
                Ll1depth = current_depth_l1_weight * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0

            total_l += loss
        timing_add("render_loss", segment_start)
        
        loss = total_l / len(viewpoint_cameras)

        ############### NOTE: IMC ###############
        segment_start = timing_mark()
        continuous_l = gaussians.train_continuous(scene.getTrainCameras().copy(), viewpoint_cam, pipe, background, tb_writer, iteration)
        timing_add("cpf_training", segment_start)
        if continuous_l is not None:
            loss += continuous_l
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        loss = torch.where(torch.isinf(loss), torch.zeros_like(loss), loss)

        segment_start = timing_mark()
        loss.backward()

        ############### NOTE: IMC ###############
        if iteration > 2450:
            gaussians.remove_nan_grad()
        guided_stats_active = (
            getattr(opt, "cpf_densify", False)
            and int(getattr(opt, "cpf_densify_guided_clone", 0)) != 0
            and iteration < opt.densify_until_iter
            and iteration > opt.densify_from_iter
        )
        if guided_stats_active:
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        timing_add("backward", segment_start)

        iter_end.record()

        if timing_enabled:
            timing_breakdown["iterations"] += 1
        segment_start = timing_mark()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                if iteration >= opt.iterations:
                    total_time = time.time() - total_time_start
                    print(f"\n[INFO] Total training time: {total_time:.2f}seconds")
                    time_path = os.path.join(dataset.model_path, "training_time.txt")
                    with open(time_path, 'w') as time_f:
                        time_f.write(f"Total training time: {total_time:.2f}seconds\n")
                    if timing_enabled:
                        measured_time = sum(v for k, v in timing_breakdown.items() if k != "iterations")
                        breakdown = {
                            "total_training_seconds": total_time,
                            "measured_segment_seconds": measured_time,
                            "timing_sync": timing_sync,
                            "iterations": timing_breakdown["iterations"],
                            "segments": {},
                        }
                        for name, seconds in timing_breakdown.items():
                            if name == "iterations":
                                continue
                            breakdown["segments"][name] = {
                                "seconds": seconds,
                                "percent_of_training": 100.0 * seconds / total_time if total_time > 0 else 0.0,
                                "percent_of_measured": 100.0 * seconds / measured_time if measured_time > 0 else 0.0,
                            }
                        timing_path = os.path.join(dataset.model_path, "timing_breakdown.json")
                        with open(timing_path, 'w') as timing_f:
                            json.dump(breakdown, timing_f, indent=2, sort_keys=True)
                        print(f"[INFO] Saved timing breakdown to {timing_path}")
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                ############### NOTE: IMC ###############
                gaussians.save_nn(scene.model_path)
            timing_add("report_save", segment_start)

            ############### NOTE: IMC ###############
            segment_start = timing_mark()
            # L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
            # actual_covariance = L @ L.transpose(1, 2)
            # def op_sigmoid(x, k=100, x0=0.995):
            #     return 1 / (1 + torch.exp(-k * (x - x0)))
            # noise_mcmc = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
            # noise_mcmc = torch.bmm(actual_covariance, noise_mcmc.unsqueeze(-1)).squeeze(-1)

            noise_mcmc = torch.zeros_like(gaussians._xyz)
            if noise is not None:
                noise_mcmc[noise_mask] = noise[noise_mask]
            gaussians._xyz.add_(noise_mcmc)

            ############### NOTE: Densification ###############
            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                cpf_densify_active = (
                    getattr(opt, "cpf_densify", False)
                    and iteration >= getattr(opt, "cpf_densify_start_iter", 0)
                    and (getattr(opt, "cpf_densify_until_iter", 0) <= 0 or iteration <= getattr(opt, "cpf_densify_until_iter", 0))
                )
                if cpf_densify_active:
                    use_cpf_relocate = int(getattr(opt, "cpf_densify_relocate", 1)) != 0
                    use_cpf_birth = int(getattr(opt, "cpf_densify_birth", 1)) != 0
                    if use_cpf_relocate:
                        relocated = gaussians.relocate_gs_cpf(dead_mask=dead_mask)
                        if not relocated:
                            gaussians.relocate_gs(dead_mask=dead_mask)
                    else:
                        gaussians.relocate_gs(dead_mask=dead_mask)
                    if use_cpf_birth:
                        current_num_points = gaussians.get_xyz.shape[0]
                        target_num = min(args.cap_max, int(1.05 * current_num_points))
                        target_add = max(0, target_num - current_num_points)
                        if int(getattr(opt, "cpf_densify_guided_clone", 0)) != 0:
                            added = gaussians.add_new_gs_cpf_guided(cap_max=args.cap_max, num_gs=target_add)
                        else:
                            added = gaussians.add_new_gs_cpf(cap_max=args.cap_max)
                        remaining = max(0, target_add - max(0, added))
                        if remaining > 0:
                            gaussians.add_new_gs(cap_max=args.cap_max, num_gs=remaining)
                    else:
                        gaussians.add_new_gs(cap_max=args.cap_max)
                else:
                    gaussians.relocate_gs(dead_mask=dead_mask)
                    gaussians.add_new_gs(cap_max=args.cap_max)


            timing_add("noise_densify", segment_start)

            segment_start = timing_mark()
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                # gaussians.optimizer.step()
                # gaussians.optimizer.zero_grad(set_to_none = True)

                ############### NOTE: IMC ###############
                torch.nn.utils.clip_grad_norm_(gaussians.net.parameters(), 1.)
                gaussians.imc_experimental_optimizer.step()
                gaussians.imc_experimental_optimizer.zero_grad()
                # gaussians.imc_experimental_optim_scheduler.step()

                # ################# NOTE: IMC ##################
                # if opt.noise_method == "imc" or opt.noise_method == "imc2":
                #     opacity_diff, color_diff = gaussians.compute_diff(viewpoint_cam, tb_writer, iteration)
                #     L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                #     actual_covariance = L @ L.transpose(1, 2)

                #     def op_sigmoid(x, k=100, x0=0.995):
                #         return 1 / (1 + torch.exp(-k * (x - x0)))
                    
                #     noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(opacity_diff))*args.noise_lr*xyz_lr
                #     noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                #     gaussians._xyz.add_(noise)
                # ################# NOTE: MCMC ##################
                stochastic_stop_iter = getattr(opt, "stochastic_stop_iter", 0)
                stochastic_active = stochastic_stop_iter <= 0 or iteration <= stochastic_stop_iter
                if stochastic_active:
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)

                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))
                    
                    noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)
                elif not stochastic_stop_logged:
                    print(f"[INFO] Stochastic covariance noise stopped at iteration {iteration} by stochastic_stop_iter={stochastic_stop_iter}")
                    stochastic_stop_logged = True
            timing_add("optimizer", segment_start)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        ############### NOTE: IMC ###############
        # if tb_writer and iteration > 2400:
        #     tb_writer.add_scalar('nn_l/partial_l', partial_l.item(), iteration)

        # if opt.nn_type == "allviews":
        #     ### NOTE: all views
        #     if iteration >= 2000 and iteration % 1000 == 0:
        #         gaussians.update_nnpts(scene.getTrainCameras().copy(), pipe, background, align=False, nn_type=opt.nn_type, bound_type=opt.bound_type)
        #     nnl = gaussians.nntrain(tb_writer, iteration)
        # elif opt.nn_type == "singleview":
        #     ### NOTE: single view
        #     if iteration >= 2000 and iteration % 1000 == 0:
        #         gaussians.update_nnpts(scene.getTrainCameras().copy(), pipe, background, align=False, nn_type=opt.nn_type, bound_type=opt.bound_type)
        #     nnl = gaussians.nntrain_view(viewpoint_cam, pipe, bg, tb_writer, iteration)
        # else:
        #     raise NotImplementedError

        # if nnl is None:
        #     continue
        # nnl.backward()
        # gaussians.exposure_optimizer.zero_grad(set_to_none = True)
        # gaussians.optimizer.zero_grad(set_to_none = True)

        # if tb_writer:
        #     tb_writer.add_scalar('nn_l/nnl', nnl.item(), iteration)
        
        # if iteration % 10000 == 0:
        #     gaussians.save_nn(scene.model_path)



def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    inv_depth = render_pkg["depth"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    ############### NOTE: IMC ###############
                    # rgb, nn_inv_depth, acc = scene.gaussians.nnrender(viewpoint)

                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/inv_depth".format(viewpoint.image_name), inv_depth[None], global_step=iteration)

                        ############### NOTE: IMC ###############
                        # tb_writer.add_images(config['name'] + "_view_{}/nn_rgb".format(viewpoint.image_name), rgb[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/nn_inv_depth".format(viewpoint.image_name), nn_inv_depth[None], global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i * 10000 for i in range(100)])
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000, 60000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[i * 10000 for i in range(100)])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    if args.model_path:
        os.makedirs(args.model_path, exist_ok=True)
        with open(os.path.join(args.model_path, "full_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        with open(os.path.join(args.model_path, "command.txt"), "w") as f:
            f.write(" ".join(sys.argv) + "\n")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
