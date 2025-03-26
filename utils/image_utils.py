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
import cv2
import numpy as np
import pyiqa

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))



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