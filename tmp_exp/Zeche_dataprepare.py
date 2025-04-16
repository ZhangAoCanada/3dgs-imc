import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
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

from tmp_exp.tools.read_write_model import (
    Camera, Point3D, Image, write_cameras_text, write_images_text, write_points3D_text, write_cameras_binary, write_images_binary, write_points3D_binary, read_points3D_binary
)
from tmp_exp.tools.utils import list_images, get_camera_id, list_jsons

from tqdm import tqdm


def readColmapSceneInfo(path):
    train_images, train_cameras, train_points3D = {}, {}, {}
    test_images, test_cameras, test_points3D = {}, {}, {}
    image_path = os.path.join(path, "images")
    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")
    train_image_path = os.path.join(train_path, "images")
    test_image_path = os.path.join(test_path, "images")
    train_sparse_path = os.path.join(train_path, "sparse/0")
    test_sparse_path = os.path.join(test_path, "sparse/0")
    os.makedirs(train_image_path, exist_ok=True)
    os.makedirs(test_image_path, exist_ok=True)
    os.makedirs(train_sparse_path, exist_ok=True)
    os.makedirs(test_sparse_path, exist_ok=True)

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
        
        if "air" in extr.name:
            train_cameras[intr.id] = Camera(id=intr.id, model=intr.model, width=intr.width, height=intr.height, params=intr.params)
            train_images[extr.id] = Image(
                id=extr.id,
                qvec=extr.qvec,
                tvec=extr.tvec,
                camera_id=extr.camera_id,
                name=extr.name,
                xys=extr.xys,
                point3D_ids=extr.point3D_ids
            )
            os.system("cp {} {}".format(os.path.join(image_path, extr.name), os.path.join(train_image_path, extr.name)))
            # print("cp {} {}".format(os.path.join(image_path, extr.name), os.path.join(train_image_path, extr.name)))
        elif "ground" in extr.name:
            test_cameras[intr.id] = Camera(id=intr.id, model=intr.model, width=intr.width, height=intr.height, params=intr.params)
            test_images[extr.id] = Image(
                id=extr.id,
                qvec=extr.qvec,
                tvec=extr.tvec,
                camera_id=extr.camera_id,
                name=extr.name,
                xys=extr.xys,
                point3D_ids=extr.point3D_ids
            )
            os.system("cp {} {}".format(os.path.join(image_path, extr.name), os.path.join(test_image_path, extr.name)))
        else:
            assert False, "Colmap camera name not handled: only air and ground cameras supported!"

    points3D = read_points3D_binary(os.path.join(path, "sparse/0", "points3D.bin"))

    print("[INFO] num points: {}".format(len(points3D)))
    for i, (point_id, point) in tqdm(enumerate(points3D.items())):
        assert point_id == point.id
        train_points3D[point_id] = Point3D(
            id=point.id,
            xyz=point.xyz,
            rgb=point.rgb,
            error=point.error,
            image_ids=point.image_ids,
            point2D_idxs=point.point2D_idxs
        )
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
    path = "/data/zhangao/VastGaussian/Zeche2"
    readColmapSceneInfo(path)