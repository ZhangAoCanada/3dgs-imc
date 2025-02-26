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
import json
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_noise
import sys
from scene_experimental import Scene, GaussianModel
from scene_experimental.gaussian_model import build_scaling_rotation
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


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
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

    random_num = opt.experimental_randomview_num
    for iteration in range(first_iter, opt.iterations + 1):
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

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam_this = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        ############### NOTE: IMC ###############
        noise, noise_mask = gaussians.imc_process_experimental(viewpoint_cam_this)

        l_total = 0.0
        viewpoint_tmp = scene.getTrainCameras().copy()
        np.random.shuffle(viewpoint_tmp)
        viewpoint_random = viewpoint_tmp[:random_num]
        viewpoint_random.append(viewpoint_cam_this)
        for idx, viewpoint_cam in enumerate(viewpoint_random):
            render_pkg = render_noise(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, noise=noise)
            # gaussians._xyz.add_(noise)
            xyz_lr = gaussians.xyz_scheduler_args(iteration)

            # render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)

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

            # Depth regularization
            Ll1depth_pure = 0.0
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                if opt.depth_normalize:
                    mono_min = mono_invdepth.min()
                    mono_max = mono_invdepth.max()
                    inDepth_min = invDepth.min()
                    inDepth_max = invDepth.max()
                    mono_invdepth_normalized = (mono_invdepth - mono_min) / (mono_max - mono_min + 1e-6)
                    invDepth_normalized = (invDepth - inDepth_min) / (inDepth_max - inDepth_min + 1e-6)
                    Ll1depth_pure = torch.abs((invDepth_normalized  - mono_invdepth_normalized) * depth_mask).mean()
                else:
                    Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()

                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0

            l_total += loss

        loss = l_total / len(viewpoint_random)
        loss.backward()

        if opt.depth_normalize:
            del mono_min, mono_max, inDepth_min, inDepth_max, mono_invdepth_normalized, invDepth_normalized

        ############### NOTE: IMC ###############
        gaussians.remove_nan_grad()

        iter_end.record()

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
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            ############### NOTE: Densification ###############
            # if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #     dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
            #     gaussians.relocate_gs(dead_mask=dead_mask)
            #     gaussians.add_new_gs(cap_max=args.cap_max)

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

                ############### NOTE: IMC ###############
                # gaussians.detach_param()
                gaussians.add_noise(noise, noise_mask)
                # filter_mask = radii > 0
                # gaussians.add_noise(noise, filter_mask)

                ################# NOTE: MCMC ##################
                # L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                # actual_covariance = L @ L.transpose(1, 2)

                # def op_sigmoid(x, k=100, x0=0.995):
                #     return 1 / (1 + torch.exp(-k * (x - x0)))
                
                # noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                # noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                # gaussians._xyz.add_(noise)

            ############### NOTE: Densification ###############
            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                gaussians.relocate_gs(dead_mask=dead_mask)
                gaussians.add_new_gs(cap_max=args.cap_max)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        ############### NOTE: IMC ###############
        # noise, noise_mask = gaussians.imc_process_experimental(viewpoint_cam_this)

        # nl_total = 0.0
        # np.random.shuffle(viewpoint_tmp)
        # viewpoint_random = viewpoint_tmp[:random_num]
        # viewpoint_random.append(viewpoint_cam_this)
        # for idx, viewpoint_cam in enumerate(viewpoint_random):
        #     render_pkg_noise = render_noise(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, noise=noise, noise_mask=noise_mask)
        #     image_noise, viewspace_point_tensor_noise, visibility_filter_noise, radii_noise = render_pkg_noise["render"], render_pkg_noise["viewspace_points"], render_pkg_noise["visibility_filter"], render_pkg_noise["radii"]

        #     Ll1_noise = l1_loss(image_noise, gt_image)
        #     if FUSED_SSIM_AVAILABLE:
        #         ssim_value_noise = fused_ssim(image_noise.unsqueeze(0), gt_image.unsqueeze(0))
        #     else:
        #         ssim_value_noise = ssim(image_noise, gt_image)

        #     psnr_noise = psnr(image_noise, gt_image).mean()
        #     loss_noise = (1.0 - opt.lambda_dssim) * Ll1_noise \
        #             + opt.lambda_dssim * (1.0 - ssim_value_noise) \
        #             + psnr(image, gt_image).mean().detach().clone() - psnr_noise \
        #             + ssim_value.detach().clone() - ssim_value_noise

        #     # Depth regularization
        #     Ll1depth_noise = 0.0
        #     if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
        #         invDepth = render_pkg_noise["depth"]
        #         mono_invdepth = viewpoint_cam.invdepthmap.cuda()
        #         depth_mask = viewpoint_cam.depth_mask.cuda()

        #         Ll1depth_noise = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
        #         Ll1depth = depth_l1_weight(iteration) * Ll1depth_noise 
        #         loss_noise += Ll1depth
        #         Ll1depth = Ll1depth.item()
        #     else:
        #         Ll1depth = 0

        #     # loss_noise.backward()
        #     nl_total += loss_noise

        # nl_total /= len(viewpoint_random)
        # nl_total.backward()
        # gaussians.remove_nan_grad()

        # if tb_writer:
        #     tb_writer.add_scalar('trainoise_loss_patches/l1_loss_noise', Ll1_noise.item(), iteration)
        #     tb_writer.add_scalar('trainoise_loss_patches/psnr_noise', psnr_noise.item(), iteration)
        #     tb_writer.add_scalar('trainoise_loss_patches/ssim_noise', ssim_value_noise.item(), iteration)
        #     tb_writer.add_scalar('trainoise_loss_patches/total_loss_noise', nl_total.item(), iteration)

        # torch.nn.utils.clip_grad_norm_(gaussians.net.parameters(), 1.)
        # gaussians.imc_experimental_optimizer.step()
        # gaussians.imc_experimental_optimizer.zero_grad()
        # gaussians.exposure_optimizer.zero_grad(set_to_none = True)
        # gaussians.optimizer.zero_grad(set_to_none = True)

        # # if iteration > 500:
        # with torch.no_grad():
        #     gaussians.add_noise(noise, noise_mask)
        #     # filter_mask = radii > 0
        #     # gaussians.add_noise(noise, filter_mask)




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
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i * 1000 for i in range(100)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--depthnorm", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])

    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    if args.depthnorm:
        args.depth_normalize = True
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
