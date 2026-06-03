import os
import sys
import argparse
import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None

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

    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        plane = Poly3DCollection([[corners_world[0], corners_world[1], corners_world[2], corners_world[3]]])
        plane.set_facecolor(color)
        plane.set_alpha(0.22)
        plane.set_edgecolor("none")
        plane.set_zorder(8)
        ax.add_collection3d(plane)
    except Exception:
        pass

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
            linewidth=1.5,
            zorder=5,
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


def make_lookat_rotation_world_to_camera(camera_center, target, world_up=np.array([0.0, 0.0, 1.0])):
    forward = target - camera_center
    forward = forward / (np.linalg.norm(forward) + 1e-12)

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-12)

    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-12)

    c2w = np.stack([right, up, forward], axis=1)
    return c2w.T


def read_ply_points_rgb(ply_path):
    if PlyData is None:
        raise ImportError("plyfile is not installed. Please install it with: pip install plyfile")

    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]

    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )

    prop_names = set(vertex.data.dtype.names or [])
    if {"red", "green", "blue"}.issubset(prop_names):
        rgb = np.stack(
            [
                np.asarray(vertex["red"], dtype=np.float32),
                np.asarray(vertex["green"], dtype=np.float32),
                np.asarray(vertex["blue"], dtype=np.float32),
            ],
            axis=1,
        )
        if np.max(rgb) > 1.0:
            rgb = rgb / 255.0
    else:
        rgb = None

    return xyz, rgb


def filter_points_inside_camera_circle(points_vis, points_rgb, centers_vis):
    if points_vis is None or points_vis.shape[0] == 0:
        return points_vis, points_rgb
    if centers_vis is None or centers_vis.shape[0] == 0:
        return points_vis, points_rgb

    cam_xy = centers_vis[:, :2]
    circle_center_xy = np.mean(cam_xy, axis=0)
    cam_dists = np.linalg.norm(cam_xy - circle_center_xy[None, :], axis=1)
    radius = np.max(cam_dists)

    point_xy = points_vis[:, :2]
    point_dists = np.linalg.norm(point_xy - circle_center_xy[None, :], axis=1)
    mask = point_dists <= radius

    filtered_points = points_vis[mask]
    filtered_rgb = points_rgb[mask] if points_rgb is not None else None
    return filtered_points, filtered_rgb, circle_center_xy, radius


def sparsify_cameras_for_plot(cameras, R_align, must_keep_names=None, grid_ratio=0.12):
    if len(cameras) <= 2:
        return cameras

    if must_keep_names is None:
        must_keep_names = set()

    centers = np.array([center for _, _, center, _ in cameras])
    centers_vis = (R_align @ centers.T).T
    min_xy = np.min(centers_vis[:, :2], axis=0)
    span_xy = np.ptp(centers_vis[:, :2], axis=0)
    extent_xy = max(np.max(span_xy), 1e-6)
    cell_size = max(extent_xy * grid_ratio, 1e-6)

    scene_center_xy = np.mean(centers_vis[:, :2], axis=0)
    kept_by_cell = {}

    for idx, (name, cam_id, center, R) in enumerate(cameras):
        cxy = centers_vis[idx, :2]
        cell = tuple(np.floor((cxy - min_xy) / cell_size).astype(np.int64))
        must_keep = 1 if name in must_keep_names else 0
        dist = float(np.linalg.norm(cxy - scene_center_xy))

        if cell not in kept_by_cell:
            kept_by_cell[cell] = (must_keep, dist, (name, cam_id, center, R))
        else:
            prev_must_keep, prev_dist, _ = kept_by_cell[cell]
            if (must_keep > prev_must_keep) or (must_keep == prev_must_keep and dist < prev_dist):
                kept_by_cell[cell] = (must_keep, dist, (name, cam_id, center, R))

    sparse_cameras = [entry[2] for entry in kept_by_cell.values()]

    existing_names = {name for name, _, _, _ in sparse_cameras}
    for name, cam_id, center, R in cameras:
        if name in must_keep_names and name not in existing_names:
            sparse_cameras.append((name, cam_id, center, R))

    return sparse_cameras

def main():
    parser = argparse.ArgumentParser(description="Select viewpoints by angular threshold on a fitted bottom plane.")
    parser.add_argument("--source_path", "-s", default="/mnt/c/Users/Aooooo/Documents/DATA/tandt_db/tandt/truck_resize", type=str, help="Path to the source directory containing sparse/0/")
    parser.add_argument("--image_name", "-i", default="000013.jpg", type=str, help="Name of the target image")
    parser.add_argument("--angle", "-a", default="45", type=float, nargs='+', help="Angular threshold(s) in degrees")
    parser.add_argument("--ply_path", type=str, default="/mnt/c/Users/Aooooo/Documents/DATA/tandt_db/tandt/truck_resize/sparse/0/points3D.ply", help="Path to point cloud PLY for visualization")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to plot camera poses and 3D points")

    args = parser.parse_args()

    if np.isscalar(args.angle):
        args.angle = [float(args.angle)]
    else:
        args.angle = [float(a) for a in args.angle]

    if not os.path.exists(args.source_path):
        print(f"Error: Source path {args.source_path} does not exist.")
        sys.exit(1)

    print(f"Reading colmap info from {args.source_path}...")
    try:
        extrinsics, intrinsics = readColmapSceneInfo(args.source_path)
    except Exception as e:
        print(f"Error reading colmap info: {e}")
        sys.exit(1)

    target_center = None
    target_rotation = None
    all_cameras = []

    # Iterate through all cameras to find target and compute centers
    for _, cam in extrinsics.items():
        R = qvec2rotmat(cam.qvec)
        t = cam.tvec
        center = -R.T @ t

        if cam.name == args.image_name:
            target_center = center
            target_rotation = R

        all_cameras.append((cam.name, cam.camera_id, center, R))

    if target_center is None:
        print(f"Error: Target image '{args.image_name}' not found in the dataset.")
        sys.exit(1)

    print(f"Found target image '{args.image_name}' at {target_center}")

    # Fit bottom plane via PCA
    all_centers = np.array([c for _, _, c, _ in all_cameras])
    plane_center, plane_normal = fit_plane_pca(all_centers)

    # Use projected target vector as reference direction
    target_vec = project_to_plane(target_center - plane_center, plane_normal)
    if np.linalg.norm(target_vec) < 1e-8 and target_rotation is not None:
        target_forward = target_rotation.T @ np.array([0.0, 0.0, 1.0])
        target_vec = project_to_plane(target_forward, plane_normal)

    all_selected = {}
    for angle_deg in args.angle:
        angle_threshold = np.deg2rad(angle_deg)
        selected = []
        for name, cam_id, center, R in all_cameras:
            vec = project_to_plane(center - plane_center, plane_normal)
            angle = angle_between(vec, target_vec)
            if angle <= angle_threshold:
                selected.append((name, cam_id, center, R, angle))

        selected.sort(key=lambda x: x[4])
        all_selected[angle_deg] = selected

        print(f"Selected {len(selected)} images with angle <= {angle_deg} degrees:")
        if len(selected) <= 20:
            for name, _, _, _, angle in selected:
                print(f"  {name}: {np.rad2deg(angle):.3f} deg")

        # # angle_tag = str(angle_deg).replace(".", "p")
        # angle_tag = str(angle_deg*2)
        # output_file = os.path.join(args.source_path, f"test_selection_angle_{angle_tag}.txt")
        # with open(output_file, "w") as f:
        #     for name, _, _, _, _ in selected:
        #         f.write(f"{name}\n")

        # print(f"Saved selected image names to {output_file}")

    print("Plotting camera poses and points...")
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Error: matplotlib is required for debug mode. Please install it.")
        return

    # Align visualization so the bottom plane is the default plotting plane
    R_align = rotation_from_a_to_b(plane_normal, np.array([0.0, 0.0, 1.0]))

    centers_vis = (R_align @ all_centers.T).T

    points_vis = None
    points_rgb = None
    if args.ply_path and os.path.exists(args.ply_path):
        try:
            points_xyz, points_rgb = read_ply_points_rgb(args.ply_path)
            points_vis = (R_align @ points_xyz.T).T
            n_before = points_vis.shape[0]
            points_vis, points_rgb, circle_center_xy, circle_radius = filter_points_inside_camera_circle(
                points_vis, points_rgb, centers_vis
            )
            n_after = points_vis.shape[0]
            print(f"Loaded point cloud from {args.ply_path} with {n_before} points")
            print(f"Filtered by camera circle (top-down, max radius): center={circle_center_xy}, radius={circle_radius:.4f}")
            print(f"Filtered point cloud for visualization: kept {n_after} / {n_before} points")
        except Exception as e:
            print(f"Warning: Failed to read PLY '{args.ply_path}': {e}")
    else:
        print(f"Warning: PLY path does not exist: {args.ply_path}")

    # Scene scale for camera frustums (adaptive to what is plotted)
    scene_scale = np.median(np.linalg.norm(all_centers - plane_center, axis=1))
    if points_vis is not None and points_vis.shape[0] > 0:
        points_min = np.percentile(points_vis, 1.0, axis=0)
        points_max = np.percentile(points_vis, 99.0, axis=0)
        points_extent = np.max(points_max - points_min)
    else:
        points_extent = 0.0

    centers_extent = np.max(np.ptp(centers_vis, axis=0)) if centers_vis.shape[0] > 0 else 0.0
    extent_scale = max(points_extent, centers_extent)
    cam_scale = max(scene_scale * 0.09, extent_scale * 0.035, 1e-3)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # plots_dir = os.path.join(script_dir, "test_selection_angleplots")
    # plots_dir = os.path.join(script_dir, "tmp")
    plots_dir = os.path.join("./tmp")
    os.makedirs(plots_dir, exist_ok=True)

    for angle_deg in args.angle:
        selected = all_selected[angle_deg]
        selected_names = set([name for name, _, _, _, _ in selected])
        must_keep_names = set(selected_names)
        must_keep_names.add(args.image_name)
        cameras_to_draw = sparsify_cameras_for_plot(all_cameras, R_align, must_keep_names=must_keep_names, grid_ratio=0.1)
        print(f"Drawing {len(cameras_to_draw)} sparse cameras (from {len(all_cameras)}) for angle {angle_deg}")

        fig = plt.figure(figsize=(10, 10), facecolor=(0.0, 0.0, 0.0, 0.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.computed_zorder = False
        ax.set_facecolor((0.0, 0.0, 0.0, 0.0))

        if points_vis is not None:
            max_points = 200000
            if points_vis.shape[0] > max_points:
                idx = np.linspace(0, points_vis.shape[0] - 1, max_points, dtype=np.int64)
                pts = points_vis[idx]
                cols = points_rgb[idx] if points_rgb is not None else None
            else:
                pts = points_vis
                cols = points_rgb

            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=cols if cols is not None else "gray",
                s=1.5,
                alpha=0.3,
                linewidths=0,
                depthshade=False,
                zorder=1,
            )

        center_x, center_y, center_z = [], [], []
        center_colors = []
        for name, cam_id, center, R in cameras_to_draw:
            if name == args.image_name or name in selected_names:
                color = "blue"
            else:
                color = "red"

            cam = intrinsics.get(cam_id, None)
            if cam is not None:
                aspect_ratio = cam.width / max(cam.height, 1)
            else:
                aspect_ratio = 1.0

            center_vis = R_align @ center
            R_vis = R @ R_align.T
            draw_camera_frustum(ax, center_vis, R_vis, color, cam_scale * 1.0, aspect_ratio)

            center_x.append(center_vis[0])
            center_y.append(center_vis[1])
            center_z.append(center_vis[2])
            center_colors.append(color)

        ax.scatter(center_x, center_y, center_z, c=center_colors, s=1.2, alpha=1.0, depthshade=False, zorder=12)

        set_axes_equal(ax)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

        # angle_tag = str(angle_deg).replace(".", "p")
        angle_tag = str(angle_deg*2)
        fig_path = os.path.join(plots_dir, f"test_selection_angle_{angle_tag}.png")
        fig.savefig(fig_path, dpi=1000, bbox_inches="tight", pad_inches=0.0, transparent=True)
        print(f"Saved plot to {fig_path}")

        topdown_path = os.path.join(plots_dir, f"test_selection_angle_{angle_tag}_topdown.png")
        ax.view_init(elev=90, azim=-90)
        fig.savefig(topdown_path, dpi=1000, bbox_inches="tight", pad_inches=0.0, transparent=True)
        print(f"Saved top-down plot to {topdown_path}")
        plt.close(fig)
        # plt.show()

    # Render additional figure: only synthetic sky cameras looking at the scene center
    sky_fig = plt.figure(figsize=(10, 10), facecolor=(0.0, 0.0, 0.0, 0.0))
    sky_ax = sky_fig.add_subplot(111, projection="3d")
    sky_ax.computed_zorder = False
    sky_ax.set_facecolor((0.0, 0.0, 0.0, 0.0))

    scene_center_vis = np.mean(centers_vis, axis=0)
    xy_dists = np.linalg.norm(centers_vis[:, :2] - scene_center_vis[:2][None, :], axis=1)
    sky_radius = max(np.max(xy_dists) * 1.25, cam_scale * 10.0)
    sky_height = np.max(centers_vis[:, 2]) + sky_radius * 0.9

    grid_rows = 4
    grid_cols = 6
    grid_width = 2.0 * sky_radius
    grid_height = 2.0 * sky_radius
    xs = np.linspace(scene_center_vis[0] - 0.5 * grid_width, scene_center_vis[0] + 0.5 * grid_width, grid_cols)
    ys = np.linspace(scene_center_vis[1] - 0.5 * grid_height, scene_center_vis[1] + 0.5 * grid_height, grid_rows)

    sky_color = "red"
    for y in ys:
        for x in xs:
            sky_center = np.array([x, y, sky_height])
            sky_rotation = make_lookat_rotation_world_to_camera(sky_center, scene_center_vis)
            draw_camera_frustum(sky_ax, sky_center, sky_rotation, sky_color, cam_scale * 3.0, aspect_ratio=1.0)

    set_axes_equal(sky_ax)
    sky_ax.grid(False)
    sky_ax.set_xticks([])
    sky_ax.set_yticks([])
    sky_ax.set_zticks([])
    sky_ax.set_axis_off()
    sky_fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    tmp_fig_path = os.path.join(plots_dir, "tmp.png")
    sky_fig.savefig(tmp_fig_path, dpi=1000, bbox_inches="tight", pad_inches=0.0, transparent=True)
    print(f"Saved sky camera plot to {tmp_fig_path}")
    plt.close(sky_fig)

if __name__ == "__main__":
    main()
