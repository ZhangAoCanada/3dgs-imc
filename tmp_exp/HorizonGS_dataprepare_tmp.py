import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from PIL import Image as PILImage
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
import collections
import struct
import argparse
import shutil

from tmp_exp.tools.read_write_model import (
    Camera, Point3D, Image, write_cameras_text, write_images_text, write_points3D_text, write_cameras_binary, write_images_binary, write_points3D_binary, read_points3D_binary
)
from tmp_exp.tools.utils import list_images, get_camera_id, list_jsons

from tqdm import tqdm


def readColmapSceneInfo(path):
    train_images, train_cameras, train_points3D = {}, {}, {}
    test_images, test_cameras, test_points3D = {}, {}, {}
    image_path = os.path.join(path, "images")
    train_path = os.path.join(path, "train2")
    test_path = os.path.join(path, "test2")
    train_image_path = os.path.join(train_path, "images")
    test_image_path = os.path.join(test_path, "images")
    train_sparse_path = os.path.join(train_path, "sparse/0")
    test_sparse_path = os.path.join(test_path, "sparse/0")
    os.makedirs(train_image_path, exist_ok=True)
    os.makedirs(test_image_path, exist_ok=True)
    os.makedirs(train_sparse_path, exist_ok=True)
    os.makedirs(test_sparse_path, exist_ok=True)
    train_mask_path = os.path.join(train_path, "masks")
    test_mask_path = os.path.join(test_path, "masks")
    os.makedirs(train_mask_path, exist_ok=True)
    os.makedirs(test_mask_path, exist_ok=True)

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    for idx, key in tqdm(enumerate(cam_extrinsics)):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        if intr.model!="SIMPLE_PINHOLE" and intr.model!="PINHOLE":
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        try:
            image_base = os.path.basename(extr.name)
            matches = re.findall(r'(\d+)', image_base)
            if matches:
                 image_num = int(matches[-1])
            else:
                 print(f"Warning: No number found in {extr.name}, skipping.")
                 continue
        except Exception as e:
            print(f"Error processing {extr.name}: {e}")
            continue

        # Calculate resize parameters
        MAX_SIZE = 20000
        w, h = intr.width, intr.height
        scale = 1.0
        if max(w, h) > MAX_SIZE:
             scale = MAX_SIZE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        new_params = np.array(intr.params)
        if intr.model == "SIMPLE_PINHOLE": # f, cx, cy
            new_params[0] *= scale
            new_params[1] *= scale
            new_params[2] *= scale
        elif intr.model == "PINHOLE": # fx, fy, cx, cy
            new_params[0] *= scale
            new_params[1] *= scale
            new_params[2] *= scale
            new_params[3] *= scale

        if "aerial" in extr.name:
            #############################################################
            # # if not ((219 <= image_num <= 229) or (183 <= image_num <= 193) or (135 <= image_num <= 145) or (100 <= image_num <= 110)):
            # if not ((263 <= image_num <= 269) or (184 <= image_num <= 192) or (136 <= image_num <= 144) or (290 <= image_num <= 295)):
            #     continue
            # if 136 <= image_num <= 144 and "aerial_z" in extr.name:
            #     continue
            # if 184 <= image_num <= 192 and "aerial_z" in extr.name:
            #     continue
            #############################################################
            # if not ((126 <= image_num <= 136) or (104 <= image_num <= 114) or (165 <= image_num <= 185)):
            if not ((128 <= image_num <= 133) or (107 <= image_num <= 112) or (165 <= image_num <= 172) or (177 <= image_num <= 185)):
                continue
            if 185 >= image_num >= 177 and "aerial_y" in extr.name:
                continue
            elif 165<= image_num <= 172 and "aerial_z" in extr.name:
                continue
            elif 128 <= image_num <= 133 and "aerial_z" in extr.name:
                continue
            elif 107 <= image_num <= 112 and "aerial_y" in extr.name:
                continue
            #############################################################

            train_cameras[intr.id] = Camera(id=intr.id, model=intr.model, width=new_w, height=new_h, params=new_params)
            
            new_xys = extr.xys * scale
            
            train_images[extr.id] = Image(
                id=extr.id,
                qvec=extr.qvec,
                tvec=extr.tvec,
                camera_id=extr.camera_id,
                name=extr.name,
                xys=new_xys,
                point3D_ids=extr.point3D_ids
            )
            train_dst = os.path.join(train_image_path, extr.name)
            os.makedirs(os.path.dirname(train_dst), exist_ok=True)
            
            src_img_path = os.path.join(image_path, extr.name)
            if scale != 1.0:
                 with PILImage.open(src_img_path) as img:
                     img = img.resize((new_w, new_h), PILImage.LANCZOS)
                     img.save(train_dst)
            else:
                 shutil.copy2(src_img_path, train_dst)
            
            train_mask_src = os.path.join(image_path.replace("images", "masks"), extr.name)
            train_mask_dst = os.path.join(train_mask_path, extr.name)
            os.makedirs(os.path.dirname(train_mask_dst), exist_ok=True)
            
            if os.path.exists(train_mask_src):
                if scale != 1.0:
                    with PILImage.open(train_mask_src) as img:
                        img = img.resize((new_w, new_h), PILImage.NEAREST)
                        img.save(train_mask_dst)
                else:
                    shutil.copy2(train_mask_src, train_mask_dst)
        elif "street" in extr.name:
            #############################################################
            # if not (330 <= image_num <= 350):
            #     continue
            #############################################################
            if "street_cam3" in extr.name or "street_cam4" in extr.name or "street_cam1" in extr.name:
                continue
            if not (405 <= image_num <= 425):
                continue
            #############################################################
            test_cameras[intr.id] = Camera(id=intr.id, model=intr.model, width=new_w, height=new_h, params=new_params)
            
            new_xys = extr.xys * scale
            
            test_images[extr.id] = Image(
                id=extr.id,
                qvec=extr.qvec,
                tvec=extr.tvec,
                camera_id=extr.camera_id,
                name=extr.name,
                xys=new_xys,
                point3D_ids=extr.point3D_ids
            )
            test_dst = os.path.join(test_image_path, extr.name)
            os.makedirs(os.path.dirname(test_dst), exist_ok=True)
            
            src_img_path = os.path.join(image_path, extr.name)
            if scale != 1.0:
                 with PILImage.open(src_img_path) as img:
                     img = img.resize((new_w, new_h), PILImage.LANCZOS)
                     img.save(test_dst)
            else:
                 shutil.copy2(src_img_path, test_dst)

            test_mask_src = os.path.join(image_path.replace("images", "masks"), extr.name)
            test_mask_dst = os.path.join(test_mask_path, extr.name)
            os.makedirs(os.path.dirname(test_mask_dst), exist_ok=True)
            
            if os.path.exists(test_mask_src):
                if scale != 1.0:
                    with PILImage.open(test_mask_src) as img:
                        img = img.resize((new_w, new_h), PILImage.NEAREST)
                        img.save(test_mask_dst)
                else:
                    shutil.copy2(test_mask_src, test_mask_dst)
        else:
            assert False, "Colmap camera name not handled: only air and ground cameras supported!"

    points3D = read_points3D_binary(os.path.join(path, "sparse/0", "points3D.bin"))

    print("[INFO] num points: {}".format(len(points3D)))

    # Pre-compute sets of image IDs for faster lookup
    train_image_ids_set = set(train_images.keys())
    test_image_ids_set = set(test_images.keys())

    for i, (point_id, point) in tqdm(enumerate(points3D.items())):
        assert point_id == point.id

        # Check if point is visible in any training image
        is_in_train = False
        for img_id in point.image_ids:
            if img_id in train_image_ids_set:
                is_in_train = True
                break
        
        if is_in_train:
            train_points3D[point_id] = Point3D(
                id=point.id,
                xyz=point.xyz,
                rgb=point.rgb,
                error=point.error,
                image_ids=point.image_ids,
                point2D_idxs=point.point2D_idxs
            )
        
        # Check if point is visible in any testing image
        is_in_test = False
        for img_id in point.image_ids:
            if img_id in test_image_ids_set:
                is_in_test = True
                break

        if is_in_test:
            test_points3D[point_id] = Point3D(
                id=point.id,
                xyz=point.xyz,
                rgb=point.rgb,
                error=point.error,
                image_ids=point.image_ids,
                point2D_idxs=point.point2D_idxs
            )

    # save train
    write_cameras_binary(train_cameras, os.path.join(train_sparse_path, "cameras.bin"))
    write_images_binary(train_images, os.path.join(train_sparse_path, "images.bin"))
    write_points3D_binary(train_points3D, os.path.join(train_sparse_path, "points3D.bin"))
    # save test
    write_cameras_binary(test_cameras, os.path.join(test_sparse_path, "cameras.bin"))
    write_images_binary(test_images, os.path.join(test_sparse_path, "images.bin"))
    write_points3D_binary(test_points3D, os.path.join(test_sparse_path, "points3D.bin"))



if __name__ == "__main__":
    # path = "/mnt/c/Users/Aooooo/Documents/DATA/HorizonGS/real/road"
    path = "/mnt/c/Users/Aooooo/Documents/DATA/HorizonGS/real/park"
    readColmapSceneInfo(path)