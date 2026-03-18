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

from pathlib import Path
import os
import csv
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import pyiqa


def brisque(img):
    """
    Blind/Referenceless Image Spatial Quality Evaluator using PyIQA.
    
    Lower values generally indicate better perceptual quality.
    
    Args:
        img (torch.Tensor): Input image tensor of shape [B, C, H, W] in range [0, 1]
    
    Returns:
        torch.Tensor: Quality score with shape [B, 1]
    """
    # Create PyIQA BRISQUE metric
    iqa_metric = pyiqa.create_metric('brisque')
    
    # Process batches directly with PyIQA
    with torch.no_grad():
        # PyIQA expects image in range [0, 1]
        scores = iqa_metric(img)
        
        # Ensure output shape is [B, 1]
        if scores.ndim == 1:
            scores = scores.unsqueeze(1)
            
    return scores


def niqe(img):
    """
    Natural Image Quality Evaluator using PyIQA.
    
    Lower values indicate better perceptual quality.
    
    Args:
        img (torch.Tensor): Input image tensor of shape [B, C, H, W] in range [0, 1]
    
    Returns:
        torch.Tensor: Quality score with shape [B, 1]
    """
    # Create PyIQA NIQE metric
    iqa_metric = pyiqa.create_metric('niqe')
    
    # Process batches directly with PyIQA
    with torch.no_grad():
        # PyIQA expects image in range [0, 1]
        scores = iqa_metric(img)
        
        # Ensure output shape is [B, 1]
        if scores.ndim == 1:
            scores = scores.unsqueeze(1)
            
    return scores


def piqe(img):
    """
    Perception based Image Quality Evaluator using pre-built wheels when available.
    
    Lower values indicate better perceptual quality.
    
    Args:
        img (torch.Tensor): Input image tensor of shape [B, C, H, W] in range [0, 1]
    
    Returns:
        torch.Tensor: Quality score with shape [B, 1]
    """
    batch_size = img.shape[0]
    scores = torch.zeros(batch_size, 1, device=img.device)
    
    # Try to import PyIQA library (pre-built wheel)
    iqa_metric = pyiqa.create_metric('piqe')
    
    # PyIQA can process batches directly
    with torch.no_grad():
        # PyIQA expects image in range [0, 1]
        scores = iqa_metric(img)
        
        # Ensure output shape is [B, 1]
        if scores.ndim == 1:
            scores = scores.unsqueeze(1)
            
    return scores

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        # renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        # gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, keyword="test", metrics_save_path=None):
    print(f"***** [INFO] Evaluating {keyword} *****")

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    csv_rows = []
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / keyword

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                brisques = []
                niqes = []
                piqes = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    # ssims.append(ssim(renders[idx], gts[idx]))
                    # psnrs.append(psnr(renders[idx], gts[idx]))
                    # lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    ssims.append(ssim(renders[idx].cuda(), gts[idx].cuda()))
                    psnrs.append(psnr(renders[idx].cuda(), gts[idx].cuda()))
                    lpipss.append(lpips(renders[idx].cuda(), gts[idx].cuda(), net_type='vgg'))
                    brisques.append(brisque(renders[idx].cuda()))
                    niqes.append(niqe(renders[idx].cuda()))
                    piqes.append(piqe(renders[idx].cuda()))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  BRISQUE: {:>12.7f}".format(torch.tensor(brisques).mean(), ".5"))
                print("  NIQE: {:>12.7f}".format(torch.tensor(niqes).mean(), ".5"))
                print("  PIQE: {:>12.7f}".format(torch.tensor(piqes).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "BRISQUE": torch.tensor(brisques).mean().item(),
                                                        "NIQE": torch.tensor(niqes).mean().item(),
                                                        "PIQE": torch.tensor(piqes).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}, 
                                                            "BRISQUE": {name: brisque for brisque, name in zip(torch.tensor(brisques).tolist(), image_names)},
                                                            "NIQE": {name: niqe for niqe, name in zip(torch.tensor(niqes).tolist(), image_names)},
                                                            "PIQE": {name: piqe for piqe, name in zip(torch.tensor(piqes).tolist(), image_names)}})

                csv_rows.append({
                    "scene": scene_dir,
                    "keyword": keyword,
                    "method": method,
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                    "BRISQUE": torch.tensor(brisques).mean().item(),
                    "NIQE": torch.tensor(niqes).mean().item(),
                    "PIQE": torch.tensor(piqes).mean().item(),
                })

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

    if metrics_save_path is not None:
        metrics_save_path = Path(metrics_save_path)
        metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["scene", "keyword", "method", "SSIM", "PSNR", "LPIPS", "BRISQUE", "NIQE", "PIQE"]
        with open(metrics_save_path, 'w', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[INFO] Saved organized metrics CSV to: {metrics_save_path}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--metrics_save_path', required=True, type=str)
    args = parser.parse_args()
    # evaluate(args.model_paths, "train")
    evaluate(args.model_paths, "test", args.metrics_save_path)