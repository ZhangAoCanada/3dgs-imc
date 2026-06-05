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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readSelectionNames(selection_path):
    if selection_path == "" or not os.path.exists(selection_path):
        return set()
    with open(selection_path, "r") as f:
        return {line.strip() for line in f if line.strip()}

def augmentPointCloudWithDepth(pcd, cam_infos, depth_init_points=0, depth_init_stride=32, depth_init_views=0, skip_image_names=None):
    if pcd is None or depth_init_points <= 0:
        return pcd

    skip_image_names = skip_image_names or set()
    stride = max(1, int(depth_init_stride))
    candidates = [
        cam for cam in cam_infos
        if cam.image_name not in skip_image_names
        and cam.depth_path != "" and os.path.exists(cam.depth_path)
        and os.path.exists(cam.image_path)
    ]
    candidates = sorted(candidates, key=lambda cam: cam.image_name)
    if depth_init_views > 0 and len(candidates) > depth_init_views:
        ids = np.linspace(0, len(candidates) - 1, int(depth_init_views), dtype=np.int64)
        candidates = [candidates[i] for i in ids]

    if len(candidates) == 0:
        print("Depth init requested, but no train cameras with depth were found.")
        return pcd

    xyz_chunks = []
    color_chunks = []
    for cam in candidates:
        image = np.asarray(Image.open(cam.image_path).convert("RGB"), dtype=np.float32) / 255.0
        invdepth = cv2.imread(cam.depth_path, -1)
        if invdepth is None:
            continue
        invdepth = invdepth.astype(np.float32) / float(2**16)
        if invdepth.ndim != 2:
            invdepth = invdepth[..., 0]

        if cam.depth_params is not None:
            scale = cam.depth_params.get("scale", 0)
            med_scale = cam.depth_params.get("med_scale", scale)
            if med_scale > 0 and (scale < 0.2 * med_scale or scale > 5 * med_scale):
                continue
            if scale > 0:
                invdepth = invdepth * scale + cam.depth_params.get("offset", 0)

        h, w = invdepth.shape
        if image.shape[0] != h or image.shape[1] != w:
            image = np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0

        yy, xx = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
        inv_samples = invdepth[yy, xx]
        valid = np.isfinite(inv_samples) & (inv_samples > 1e-4)
        if valid.sum() == 0:
            continue

        z = 1.0 / np.maximum(inv_samples[valid], 1e-4)
        x_pix = xx[valid].astype(np.float32) + 0.5
        y_pix = yy[valid].astype(np.float32) + 0.5
        fx = fov2focal(cam.FovX, w)
        fy = fov2focal(cam.FovY, h)
        x = (x_pix - w * 0.5) / fx * z
        y = (y_pix - h * 0.5) / fy * z
        cam_points = np.stack([x, y, z, np.ones_like(z)], axis=-1)
        c2w = np.linalg.inv(getWorld2View2(cam.R, cam.T))
        world_points = (cam_points @ c2w.T)[:, :3]
        colors = image[yy[valid], xx[valid]]
        xyz_chunks.append(world_points.astype(np.float32))
        color_chunks.append(colors.astype(np.float32))

    if len(xyz_chunks) == 0:
        print("Depth init requested, but no valid depth samples were projected.")
        return pcd

    depth_xyz = np.concatenate(xyz_chunks, axis=0)
    depth_rgb = np.concatenate(color_chunks, axis=0)

    sfm_xyz = np.asarray(pcd.points)
    lo = np.percentile(sfm_xyz, 0.5, axis=0)
    hi = np.percentile(sfm_xyz, 99.5, axis=0)
    margin = (hi - lo) * 0.25 + 1e-3
    keep = np.all((depth_xyz >= (lo - margin)) & (depth_xyz <= (hi + margin)), axis=1)
    if keep.any():
        depth_xyz = depth_xyz[keep]
        depth_rgb = depth_rgb[keep]
    else:
        depth_xyz = depth_xyz[:0]
        depth_rgb = depth_rgb[:0]

    if depth_xyz.shape[0] > depth_init_points:
        rng = np.random.default_rng(0)
        ids = rng.choice(depth_xyz.shape[0], int(depth_init_points), replace=False)
        depth_xyz = depth_xyz[ids]
        depth_rgb = depth_rgb[ids]

    if depth_xyz.shape[0] == 0:
        print("Depth init requested, but all projected samples were filtered out.")
        return pcd

    points = np.concatenate([sfm_xyz, depth_xyz], axis=0).astype(np.float32)
    colors = np.concatenate([np.asarray(pcd.colors), depth_rgb], axis=0).astype(np.float32)
    normals = np.concatenate([np.asarray(pcd.normals), np.zeros_like(depth_xyz, dtype=np.float32)], axis=0).astype(np.float32)
    print(f"Depth init added {depth_xyz.shape[0]} train-view points from {len(candidates)} cameras (stride={stride}).")
    return BasicPointCloud(points=points, colors=colors, normals=normals)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8, init_type="sfm", num_pts=5000000, depth_init_points=0, depth_init_stride=32, depth_init_views=0, test_selection_path=""):
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

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if init_type == "sfm":
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    elif init_type == "random":
        ply_path = os.path.join(path, "random.ply")
        print(f"Generating random point cloud ({num_pts})...")
        
        xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"]* 3*2 -(nerf_normalization["radius"]*3)
        
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        print("Please specify a correct init_type: random or sfm")
        exit(0)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    if depth_init_points > 0 and depths != "" and pcd is not None:
        selected_image_names = readSelectionNames(test_selection_path)
        pcd = augmentPointCloudWithDepth(
            pcd, train_cam_infos, depth_init_points=depth_init_points,
            depth_init_stride=depth_init_stride, depth_init_views=depth_init_views,
            skip_image_names=selected_image_names)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}