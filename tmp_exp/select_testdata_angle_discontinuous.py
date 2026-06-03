import os
import sys
import argparse
import numpy as np

# Add the parent directory to sys.path to allow imports from scene and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)

def readColmapSceneInfo(path):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    return cam_extrinsics, cam_intrinsics


def readColmapPoints(path):
    try:
        points_file = os.path.join(path, "sparse/0", "points3D.bin")
        xyzs, rgbs, errors = read_points3D_binary(points_file)
    except Exception:
        points_file = os.path.join(path, "sparse/0", "points3D.txt")
        xyzs, rgbs, errors = read_points3D_text(points_file)
    return xyzs, rgbs, errors


def fit_plane_pca(points):
    center = np.mean(points, axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return center, normal


def project_to_plane(vec, normal):
    return vec - normal * np.dot(vec, normal)


def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.arccos(cos_val)


def set_axes_equal(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def draw_camera_frustum(ax, center, R, color, scale, aspect_ratio):
    depth = scale
    half_w = scale * 0.5
    half_h = half_w / (aspect_ratio + 1e-12)

    corners_cam = np.array([
        [-half_w, -half_h, depth],
        [half_w, -half_h, depth],
        [half_w, half_h, depth],
        [-half_w, half_h, depth],
        [0.0, 0.0, 0.0],
    ])

    corners_world = (R.T @ corners_cam.T).T + center

    # Frustum edges
    edges = [
        (4, 0), (4, 1), (4, 2), (4, 3),
        (0, 1), (1, 2), (2, 3), (3, 0),
    ]
    for i, j in edges:
        ax.plot(
            [corners_world[i, 0], corners_world[j, 0]],
            [corners_world[i, 1], corners_world[j, 1]],
            [corners_world[i, 2], corners_world[j, 2]],
            color=color,
            linewidth=1.0,
        )

    # Camera axes (COLMAP-like style)
    axis_scale = scale * 0.6
    axes_cam = np.array([
        [axis_scale, 0.0, 0.0],
        [0.0, axis_scale, 0.0],
        [0.0, 0.0, axis_scale],
    ])
    axes_world = (R.T @ axes_cam.T).T + center
    colors = ["r", "g", "b"]
    for axis_idx in range(3):
        ax.plot(
            [center[0], axes_world[axis_idx, 0]],
            [center[1], axes_world[axis_idx, 1]],
            [center[2], axes_world[axis_idx, 2]],
            color=colors[axis_idx],
            linewidth=1.0,
        )


def rotation_from_a_to_b(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        # 180 degree rotation around any orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v = v / (np.linalg.norm(v) + 1e-12)
        vx = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])
        return np.eye(3) + 2.0 * (vx @ vx)
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s ** 2))

def main():
    parser = argparse.ArgumentParser(description="Select viewpoints by angular threshold on a fitted bottom plane for multiple targets.")
    parser.add_argument("--source_path", "-s", required=True, type=str, help="Path to the source directory containing sparse/0/")
    parser.add_argument("--image_name", "-i", required=True, type=str, nargs='+', help="Name(s) of target image(s)")
    parser.add_argument("--angle", "-a", required=True, type=float, nargs='+', help="Angular threshold(s) in degrees")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to plot camera poses and 3D points")

    args = parser.parse_args()

    if not os.path.exists(args.source_path):
        print(f"Error: Source path {args.source_path} does not exist.")
        sys.exit(1)

    print(f"Reading colmap info from {args.source_path}...")
    try:
        extrinsics, intrinsics = readColmapSceneInfo(args.source_path)
    except Exception as e:
        print(f"Error reading colmap info: {e}")
        sys.exit(1)

    target_centers = {}
    all_cameras = []

    # Iterate through all cameras to find targets and compute centers
    for _, cam in extrinsics.items():
        R = qvec2rotmat(cam.qvec)
        t = cam.tvec
        center = -R.T @ t

        if cam.name in args.image_name:
            target_centers[cam.name] = center

        all_cameras.append((cam.name, cam.camera_id, center, R))

    missing_targets = [name for name in args.image_name if name not in target_centers]
    if missing_targets:
        print(f"Error: Target image(s) not found in the dataset: {missing_targets}")
        sys.exit(1)

    for target_name in args.image_name:
        print(f"Found target image '{target_name}' at {target_centers[target_name]}")

    # Fit bottom plane via PCA
    all_centers = np.array([c for _, _, c, _ in all_cameras])
    plane_center, plane_normal = fit_plane_pca(all_centers)

    # Compute target vectors for each target image
    target_vecs = {}
    for target_name, target_center in target_centers.items():
        # Find the rotation matrix for this target
        target_rotation = None
        for _, cam in extrinsics.items():
            if cam.name == target_name:
                target_rotation = qvec2rotmat(cam.qvec)
                break
        
        # Use projected target vector as reference direction
        target_vec = project_to_plane(target_center - plane_center, plane_normal)
        if np.linalg.norm(target_vec) < 1e-8 and target_rotation is not None:
            target_forward = target_rotation.T @ np.array([0.0, 0.0, 1.0])
            target_vec = project_to_plane(target_forward, plane_normal)
        target_vecs[target_name] = target_vec

    all_selected = {}
    for angle_deg in args.angle:
        angle_threshold = np.deg2rad(angle_deg)
        selected_names_set = set()
        
        # For each target, select neighbors within angle threshold
        for target_name, target_vec in target_vecs.items():
            selected = []
            for name, cam_id, center, R in all_cameras:
                vec = project_to_plane(center - plane_center, plane_normal)
                angle = angle_between(vec, target_vec)
                if angle <= angle_threshold:
                    selected.append((name, cam_id, center, R, angle))
            
            selected.sort(key=lambda x: x[4])
            selected_names_set.update([name for name, _, _, _, _ in selected])
            
            print(f"Target '{target_name}': Selected {len(selected)} images with angle <= {angle_deg} degrees")
            if len(selected) <= 20:
                for name, _, _, _, angle in selected:
                    print(f"  {name}: {np.rad2deg(angle):.3f} deg")
        
        all_selected[angle_deg] = sorted(list(selected_names_set))
        
        print(f"Total unique images for angle {angle_deg}: {len(all_selected[angle_deg])}")
        
        # angle_tag = str(angle_deg).replace(".", "p")
        angle_tag = str(angle_deg*2)
        output_file = os.path.join(args.source_path, f"discontinuous.txt")
        with open(output_file, "w") as f:
            for name in all_selected[angle_deg]:
                f.write(f"{name}\n")

        print(f"Saved selected image names to {output_file}")

    if args.debug:
        print("Plotting camera poses and points...")
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("Error: matplotlib is required for debug mode. Please install it.")
            return

        # Align visualization so the bottom plane is the default plotting plane
        R_align = rotation_from_a_to_b(plane_normal, np.array([0.0, 0.0, 1.0]))

        # Scene scale for camera frustums
        scene_scale = np.median(np.linalg.norm(all_centers - plane_center, axis=1))
        cam_scale = max(scene_scale * 0.05, 1e-3)

        plots_dir = os.path.join(args.source_path, "test_selection_discontinuous")
        os.makedirs(plots_dir, exist_ok=True)

        for angle_deg in args.angle:
            selected_names = set(all_selected[angle_deg])
            target_names_set = set(args.image_name)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            for name, cam_id, center, R in all_cameras:
                if name in target_names_set:
                    color = "green"
                elif name in selected_names:
                    color = "red"
                else:
                    color = "blue"

                cam = intrinsics.get(cam_id, None)
                if cam is not None:
                    aspect_ratio = cam.width / max(cam.height, 1)
                else:
                    aspect_ratio = 1.0

                center_vis = R_align @ center
                R_vis = R @ R_align.T
                draw_camera_frustum(ax, center_vis, R_vis, color, cam_scale, aspect_ratio)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            target_names_str = ", ".join(args.image_name)
            ax.set_title(f"Angle-based selection for {len(args.image_name)} target(s) (angle={angle_deg})")
            set_axes_equal(ax)

            # angle_tag = str(angle_deg).replace(".", "p")
            angle_tag = str(angle_deg*2)
            fig_path = os.path.join(plots_dir, f"discontinuous.png")
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {fig_path}")

            topdown_path = os.path.join(plots_dir, f"discontinuous_topdown.png")
            ax.view_init(elev=90, azim=-90)
            fig.savefig(topdown_path, dpi=200, bbox_inches="tight")
            print(f"Saved top-down plot to {topdown_path}")
            plt.close(fig)
        # plt.show()

if __name__ == "__main__":
    main()
