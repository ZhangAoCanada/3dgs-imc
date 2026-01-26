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
import shutil

from tmp_exp.tools.read_write_model import (
    Camera, Point3D, Image, write_cameras_text, write_images_text, write_points3D_text, write_cameras_binary, write_images_binary, write_points3D_binary, read_points3D_binary
)
from tmp_exp.tools.utils import list_images, get_camera_id, list_jsons

from tqdm import tqdm


def _estimate_ground_plane(points: np.ndarray):
    if points.shape[0] < 3:
        raise ValueError("Not enough points to estimate ground plane.")
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    return center, normal


def _estimate_ground_plane_ransac(points: np.ndarray, threshold=0.5, iterations=200):
    if points.shape[0] < 3:
        raise ValueError("Not enough points to estimate ground plane.")
    best_inliers = None
    best_count = -1
    rng = np.random.default_rng(0)

    for _ in range(iterations):
        idx = rng.choice(points.shape[0], size=3, replace=False)
        p1, p2, p3 = points[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        dists = np.abs((points - p1) @ normal)
        inliers = dists < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        return _estimate_ground_plane(points)

    inlier_points = points[best_inliers]
    center = inlier_points.mean(axis=0)
    centered = inlier_points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    return center, normal


def _plane_basis(normal: np.ndarray):
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u = u / (np.linalg.norm(u) + 1e-9)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-9)
    return u, v


def _project_to_plane(points: np.ndarray, center: np.ndarray, u: np.ndarray, v: np.ndarray):
    centered = points - center
    x = centered @ u
    y = centered @ v
    return np.stack([x, y], axis=1)


def _select_central_indices(coords_2d: np.ndarray, max_count: int):
    if coords_2d.shape[0] <= max_count:
        return np.arange(coords_2d.shape[0])
    med = np.median(coords_2d, axis=0)
    dists = np.linalg.norm(coords_2d - med[None, :], axis=1)
    return np.argsort(dists)[:max_count]


def _resolve_image_src(image_root: str, rel_name: str):
    direct = os.path.join(image_root, rel_name)
    if os.path.exists(direct):
        return direct
    basename = os.path.basename(rel_name)
    for root, _, files in os.walk(image_root):
        if basename in files:
            return os.path.join(root, basename)
    raise FileNotFoundError(f"Image not found: {rel_name} under {image_root}")


def readColmapSceneInfo(path, max_train_images=300, max_street_images=100):
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

    aerial_keys = []
    street_keys = []
    for idx, key in tqdm(enumerate(cam_extrinsics)):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        if intr.model != "SIMPLE_PINHOLE" and intr.model != "PINHOLE":
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        if "aerial" in extr.name:
            aerial_keys.append(key)
        elif "street" in extr.name:
            street_keys.append(key)
        else:
            assert False, "Colmap camera name not handled: only air and ground cameras supported!"

    if len(aerial_keys) == 0:
        raise ValueError("No aerial cameras found.")

    aerial_tvecs = np.array([cam_extrinsics[k].tvec for k in aerial_keys], dtype=np.float64)
    ground_center, ground_normal = _estimate_ground_plane_ransac(aerial_tvecs)
    u, v = _plane_basis(ground_normal)
    aerial_coords_2d = _project_to_plane(aerial_tvecs, ground_center, u, v)
    aerial_center_2d = np.median(aerial_coords_2d, axis=0)
    aerial_dists = np.linalg.norm(aerial_coords_2d - aerial_center_2d[None, :], axis=1)
    selected_idx = np.argsort(aerial_dists)[:min(max_train_images, aerial_coords_2d.shape[0])]
    selected_aerial_keys = [aerial_keys[i] for i in selected_idx]
    aerial_radius = float(aerial_dists[selected_idx].max()) if len(selected_idx) > 0 else 0.0
    street_radius = aerial_radius * 0.5

    selected_street_keys = street_keys
    if len(street_keys) > 0 and street_radius > 0:
        street_tvecs = np.array([cam_extrinsics[k].tvec for k in street_keys], dtype=np.float64)
        street_coords_2d = _project_to_plane(street_tvecs, ground_center, u, v)
        street_dists = np.linalg.norm(street_coords_2d - aerial_center_2d[None, :], axis=1)
        keep_mask = street_dists <= street_radius
        filtered_keys = [k for k, keep in zip(street_keys, keep_mask) if keep]
        if len(filtered_keys) > max_street_images:
            filtered_dists = np.array([d for d, keep in zip(street_dists, keep_mask) if keep])
            order = np.argsort(filtered_dists)[:max_street_images]
            selected_street_keys = [filtered_keys[i] for i in order]
        else:
            selected_street_keys = filtered_keys

    for key in tqdm(selected_aerial_keys, desc="Copy train (aerial)"):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        try:
            src = _resolve_image_src(image_path, extr.name)
        except FileNotFoundError:
            print(f"[WARN] missing image, skip: {extr.name}")
            continue
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
        train_dst = os.path.join(train_image_path, extr.name)
        os.makedirs(os.path.dirname(train_dst), exist_ok=True)
        shutil.copy2(src, train_dst)
        train_mask_src = os.path.join(image_path.replace("images", "masks"), extr.name)
        train_mask_dst = os.path.join(train_mask_path, extr.name)
        os.makedirs(os.path.dirname(train_mask_dst), exist_ok=True)
        shutil.copy2(train_mask_src, train_mask_dst)

    for key in tqdm(selected_street_keys, desc="Copy test (street)"):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        try:
            src = _resolve_image_src(image_path, extr.name)
        except FileNotFoundError:
            print(f"[WARN] missing image, skip: {extr.name}")
            continue
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
        test_dst = os.path.join(test_image_path, extr.name)
        os.makedirs(os.path.dirname(test_dst), exist_ok=True)
        shutil.copy2(src, test_dst)
        test_mask_src = os.path.join(image_path.replace("images", "masks"), extr.name)
        test_mask_dst = os.path.join(test_mask_path, extr.name)
        os.makedirs(os.path.dirname(test_mask_dst), exist_ok=True)
        shutil.copy2(test_mask_src, test_mask_dst)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/mnt/c/Users/Aooooo/Documents/DATA/HorizonGS/real/road")
    parser.add_argument("--max_train_images", type=int, default=300)
    parser.add_argument("--max_street_images", type=int, default=50)
    args = parser.parse_args()
    readColmapSceneInfo(args.path, max_train_images=args.max_train_images, max_street_images=args.max_street_images)