import os
import sys
import argparse
import numpy as np

# Add the parent directory to sys.path to allow imports from scene and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary

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


def fit_plane_pca(points):
    center = np.mean(points, axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return center, normal


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
    parser = argparse.ArgumentParser(description="Select N nearest viewpoints to a given image.")
    parser.add_argument("--source_path", "-s", required=True, type=str, help="Path to the source directory containing sparse/0/")
    parser.add_argument("--image_name", "-i", required=True, type=str, nargs='+', help="Name(s) of target image(s)")
    parser.add_argument("--n", "-n", required=True, type=int, nargs='+', help="Number(s) of nearest neighbors to select")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to plot camera poses")
    
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
    
    # Iterate through all cameras to find target and compute centers
    for cam_id, cam in extrinsics.items():
        R = qvec2rotmat(cam.qvec)
        t = cam.tvec
        center = -R.T @ t

        # center = cam.tvec
        
        if cam.name in args.image_name:
            target_centers[cam.name] = center
        
        all_cameras.append((cam.name, cam.camera_id, center, R))
        
    missing_targets = [name for name in args.image_name if name not in target_centers]
    if missing_targets:
        print(f"Error: Target image(s) not found in the dataset: {missing_targets}")
        sys.exit(1)
    
    for target_name in args.image_name:
        print(f"Found target image '{target_name}' at {target_centers[target_name]}")
    
    # Calculate and sort distances from each target
    distances_by_target = {}
    for target_name, target_center in target_centers.items():
        distances = []
        for name, _, center, _ in all_cameras:
            dist = np.linalg.norm(center - target_center)
            distances.append((name, center, dist))
        distances.sort(key=lambda x: x[2])
        distances_by_target[target_name] = distances
    
    # Process each n in the list
    for n in args.n:
        # For each target, select top N and merge into a test set
        selected_names_set = set()
        for target_name in args.image_name:
            selected = distances_by_target[target_name][:n]
            selected_names_set.update([name for name, _, _ in selected])
        selected_names = sorted(list(selected_names_set))
        
        print(f"Selected {len(selected_names)} unique images for n={n} from {len(args.image_name)} target(s).")
        for target_name in args.image_name:
            selected = distances_by_target[target_name][:n]
            print(f"  Target '{target_name}': {len(selected)} images")
            if len(selected) <= 20:
                for name, center, dist in selected:
                    print(f"    {name}: {dist:.4f}")
            
        output_file = os.path.join(args.source_path, f"discontinuous.txt")
        with open(output_file, "w") as f:
            for name in selected_names:
                f.write(f"{name}\n")
                
        print(f"Saved selected image names to {output_file}")

    if args.debug:
        print("Plotting camera poses...")
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("Error: matplotlib is required for debug mode. Please install it.")
            return

        all_centers = np.array([c for _, _, c, _ in all_cameras])
        plane_center, plane_normal = fit_plane_pca(all_centers)
        R_align = rotation_from_a_to_b(plane_normal, np.array([0.0, 0.0, 1.0]))

        scene_scale = np.median(np.linalg.norm(all_centers - plane_center, axis=1))
        cam_scale = max(scene_scale * 0.05, 1e-3)

        plots_dir = os.path.join(args.source_path, "test_selection_discontinuous")
        os.makedirs(plots_dir, exist_ok=True)

        for n in args.n:
            selected_names = set()
            for target_name in args.image_name:
                selected = distances_by_target[target_name][:n]
                selected_names.update([name for name, _, _ in selected])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            for name, cam_id, center, R in all_cameras:
                if name in target_centers:
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
            ax.set_title(f"Nearest Neighbors for {len(args.image_name)} targets (n={n})")
            set_axes_equal(ax)

            fig_path = os.path.join(plots_dir, f"discontinuous.png")
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {fig_path}")

            topdown_path = os.path.join(plots_dir, f"discontinuous_topdown.png")
            ax.view_init(elev=90, azim=-90)
            fig.savefig(topdown_path, dpi=200, bbox_inches="tight")
            print(f"Saved top-down plot to {topdown_path}")
            plt.close(fig)

if __name__ == "__main__":
    main()
