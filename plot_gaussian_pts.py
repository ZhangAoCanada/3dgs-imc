import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import numpy as np
from scipy.ndimage import gaussian_filter
from plyfile import PlyData

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
)


# Zeroth-order SH basis constant
SH_C0 = 0.28209479177387814


def load_gaussian_centers(ply_path):
    """Load Gaussian center positions and colors from a .ply file."""
    plydata = PlyData.read(ply_path)
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    # Extract colors from SH DC coefficients
    colors = np.stack(
        (
            np.asarray(plydata.elements[0]["f_dc_0"]),
            np.asarray(plydata.elements[0]["f_dc_1"]),
            np.asarray(plydata.elements[0]["f_dc_2"]),
        ),
        axis=1,
    )
    # Convert SH DC to RGB and clamp to [0, 1]
    colors = np.clip(colors * SH_C0 + 0.5, 0.0, 1.0)
    return xyz, colors


def load_camera_poses(sparse_path):
    """Load camera positions, orientations, and names from COLMAP sparse reconstruction."""
    bin_path = os.path.join(sparse_path, "images.bin")
    txt_path = os.path.join(sparse_path, "images.txt")

    if os.path.exists(bin_path):
        images = read_extrinsics_binary(bin_path)
    elif os.path.exists(txt_path):
        images = read_extrinsics_text(txt_path)
    else:
        raise FileNotFoundError(f"No COLMAP images file found in {sparse_path}")

    names = []
    positions = []
    forward_dirs = []
    for img in images.values():
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        # Camera position in world coordinates: C = -R^T @ t
        cam_pos = -R.T @ t
        # Camera forward direction (z-axis of camera in world frame)
        cam_fwd = R.T @ np.array([0, 0, 1])
        names.append(img.name)
        positions.append(cam_pos)
        forward_dirs.append(cam_fwd)

    return names, np.array(positions), np.array(forward_dirs)


def draw_camera_frustum(ax, cu, cv, fu, fv, color, size=0.4, lw=1.2):
    """Draw a small camera frustum icon at (cu, cv) pointing toward (fu, fv)."""
    du = fu - cu
    dv = fv - cv
    length = np.hypot(du, dv)
    if length < 1e-9:
        return
    du /= length
    dv /= length
    # Perpendicular direction
    pu, pv = -dv, du

    # Frustum: a triangle (apex at camera center, base in front)
    apex = np.array([cu, cv])
    base_center = apex + np.array([du, dv]) * size
    half_w = size * 0.45
    left = base_center + np.array([pu, pv]) * half_w
    right = base_center - np.array([pu, pv]) * half_w

    tri = Polygon(
        [apex, left, right],
        closed=True,
        facecolor=color,
        edgecolor="white",
        linewidth=lw,
        alpha=0.95,
        zorder=10,
    )
    ax.add_patch(tri)


def find_ground_plane(cam_positions):
    """Find the ground plane normal via PCA on camera positions.

    Assumes cameras roughly lie on a plane (e.g. all at similar height).
    The ground-plane normal is the direction of least variance.
    """
    centroid = cam_positions.mean(axis=0)
    centered = cam_positions - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    # The last principal component is the normal to the best-fit plane
    normal = Vt[2]
    # Make sure the normal points "up" (positive y by convention, but we
    # choose the direction that has the largest absolute component and
    # orient it positively)
    if normal[1] < 0:
        normal = -normal
    return normal, centroid


def project_to_ground_plane(points, normal, centroid):
    """Project 3D points onto a 2D coordinate system defined by the ground plane.

    Returns 2D coordinates where:
      - u axis: first principal direction in the plane
      - v axis: second principal direction in the plane (perpendicular to u and normal)
    """
    # Build an orthonormal basis for the plane
    # Pick an arbitrary vector not parallel to normal
    arbitrary = np.array([1, 0, 0])
    if abs(np.dot(arbitrary, normal)) > 0.9:
        arbitrary = np.array([0, 1, 0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    centered = points - centroid
    coords_u = centered @ u
    coords_v = centered @ v
    return coords_u, coords_v


def find_ply_path(model_path):
    """Find the .ply file in model_path.

    Search order:
      1. model_path/point_cloud/iteration_*/point_cloud.ply  (pick highest iteration)
      2. Any .ply file recursively under model_path (pick first found)
    """
    # Try the standard 3DGS layout first
    pattern = os.path.join(model_path, "point_cloud", "iteration_*", "point_cloud.ply")
    candidates = glob.glob(pattern)
    if candidates:
        def iter_num(p):
            dirname = os.path.basename(os.path.dirname(p))
            return int(dirname.split("_")[-1])
        return max(candidates, key=iter_num)

    # Fallback: any .ply file under model_path
    candidates = glob.glob(os.path.join(model_path, "**", "*.ply"), recursive=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No .ply file found under {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Gaussian centers and camera poses in top-down view.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained 3DGS model directory.")
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source data directory (contains sparse/0).")
    parser.add_argument("--max_gaussians", type=int, default=100000,
                        help="Max number of Gaussians to plot (randomly sampled if exceeded). "
                             "Only used when --no_density is set.")
    parser.add_argument("--density_bins", type=int, default=512,
                        help="Number of voxel cells per axis for the density heatmap.")
    parser.add_argument("--density_gamma", type=float, default=0.3,
                        help="Power-norm gamma (<1 stretches sparse end, >1 stretches dense end).")
    parser.add_argument("--no_density", action="store_true",
                        help="Use scatter plot instead of density heatmap.")
    parser.add_argument("--dist_thresh", type=float, default=None,
                        help="Max distance from nearest camera to keep a Gaussian. "
                             "Defaults to 2x the max camera-to-camera-centroid distance.")
    parser.add_argument("--test_selection_path", type=str, default=None,
                        help="Path to a txt file listing test image names (one per line).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path. Defaults to <model_path>/topdown_view.png.")
    args = parser.parse_args()

    # --- Load data ---
    ply_path = find_ply_path(args.model_path)
    print(f"Loading Gaussians from {ply_path}")
    gaussians, colors = load_gaussian_centers(ply_path)
    print(f"  Loaded {len(gaussians)} Gaussians")

    sparse_path = os.path.join(args.source_path, "sparse", "0")
    print(f"Loading cameras from {sparse_path}")
    cam_names, cam_positions, cam_forwards = load_camera_poses(sparse_path)
    print(f"  Loaded {len(cam_positions)} cameras")

    # --- Identify test cameras ---
    if args.test_selection_path is not None:
        with open(args.test_selection_path, "r") as f:
            test_names = {line.strip() for line in f if line.strip()}
        is_test = np.array([n in test_names for n in cam_names])
    else:
        is_test = np.zeros(len(cam_names), dtype=bool)
    print(f"  Train cameras: {(~is_test).sum()}, Test cameras: {is_test.sum()}")

    # --- Filter Gaussians by distance to nearest camera ---
    cam_center = cam_positions.mean(axis=0)
    cam_dists = np.linalg.norm(cam_positions - cam_center, axis=1)
    dist_thresh = args.dist_thresh if args.dist_thresh is not None else 1.0 * cam_dists.max()

    # For each Gaussian, compute distance to the nearest camera
    # Process in chunks to avoid huge memory allocation
    chunk_size = 50000
    min_dists = np.empty(len(gaussians))
    for start in range(0, len(gaussians), chunk_size):
        end = min(start + chunk_size, len(gaussians))
        # (chunk, 3) - (n_cams, 3) -> broadcast via (chunk, 1, 3) - (1, n_cams, 3)
        diffs = gaussians[start:end, None, :] - cam_positions[None, :, :]
        min_dists[start:end] = np.linalg.norm(diffs, axis=2).min(axis=1)

    keep_mask = min_dists <= dist_thresh
    gaussians = gaussians[keep_mask]
    colors = colors[keep_mask]
    print(f"  Kept {len(gaussians)}/{keep_mask.size} Gaussians within dist_thresh={dist_thresh:.2f}")

    # --- Find ground plane from camera poses ---
    normal, centroid = find_ground_plane(cam_positions)
    print(f"  Ground plane normal: {normal}")

    # --- Project to 2D ---
    cam_u, cam_v = project_to_ground_plane(cam_positions, normal, centroid)
    gauss_u, gauss_v = project_to_ground_plane(gaussians, normal, centroid)

    # --- Compute frustum size adaptively ---
    all_cam_uv = np.column_stack([cam_u, cam_v])
    cam_span = np.ptp(all_cam_uv, axis=0).max()
    frustum_size = cam_span * 0.04  # ~4% of camera spread

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))

    if not args.no_density:
        # --- 2D voxelization on the top-down projected plane ---
        pad = cam_span * 0.15
        u_min, u_max = cam_u.min() - pad, cam_u.max() + pad
        v_min, v_max = cam_v.min() - pad, cam_v.max() + pad
        nbins = args.density_bins

        # Assign each Gaussian to a voxel cell
        u_idx = np.clip(((gauss_u - u_min) / (u_max - u_min) * nbins).astype(int), 0, nbins - 1)
        v_idx = np.clip(((gauss_v - v_min) / (v_max - v_min) * nbins).astype(int), 0, nbins - 1)

        # Count occupancy per voxel cell
        voxel_grid = np.zeros((nbins, nbins), dtype=np.float64)
        np.add.at(voxel_grid, (v_idx, u_idx), 1)

        # Light Gaussian smoothing for visual continuity
        sigma = nbins / 200
        voxel_smooth = gaussian_filter(voxel_grid, sigma=sigma)

        # Mask empty regions as NaN
        voxel_plot = voxel_smooth.copy()
        voxel_plot[voxel_plot < 0.5] = np.nan

        # Power-norm: gamma < 1 stretches the sparse (low-count) end of the colormap
        vmax = np.nanpercentile(voxel_plot, 99)
        im = ax.imshow(
            voxel_plot,
            origin="lower",
            extent=[u_min, u_max, v_min, v_max],
            aspect="equal",
            cmap="inferno",
            norm=mcolors.PowerNorm(gamma=args.density_gamma, vmin=1, vmax=vmax),
            interpolation="bilinear",
            rasterized=True,
        )

        # Overlay contour lines at density percentiles
        u_centers = np.linspace(u_min, u_max, nbins)
        v_centers = np.linspace(v_min, v_max, nbins)
        valid_vals = voxel_smooth[voxel_smooth >= 0.5]
        if len(valid_vals) > 100:
            contour_levels = np.percentile(valid_vals, [10, 25, 50, 75, 90])
            contour_levels = np.unique(contour_levels)
            if len(contour_levels) >= 2:
                ax.contour(
                    u_centers, v_centers, voxel_smooth,
                    levels=contour_levels,
                    colors="white", linewidths=0.5, alpha=0.4,
                )

        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Gaussians per voxel cell", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # Print voxel stats
        occ_cells = np.sum(voxel_grid >= 1)
        print(f"  Voxel grid: {nbins}x{nbins}, occupied cells: {occ_cells}, "
              f"max count: {voxel_grid.max():.0f}, "
              f"median (occupied): {np.median(voxel_grid[voxel_grid >= 1]):.1f}")
    else:
        # --- Scatter fallback ---
        if len(gaussians) > args.max_gaussians:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(gaussians), args.max_gaussians, replace=False)
            gu, gv = gauss_u[idx], gauss_v[idx]
            gc = colors[idx]
        else:
            gu, gv, gc = gauss_u, gauss_v, colors
        ax.scatter(gu, gv, s=0.1, alpha=0.3, c=gc, rasterized=True)

    # Draw camera frustums
    fwd_u, fwd_v = project_to_ground_plane(cam_positions + cam_forwards * 0.3, normal, centroid)
    train_color = "#E74C3C"  # red
    test_color = "#2980B9"   # blue

    for i in range(len(cam_positions)):
        color = test_color if is_test[i] else train_color
        draw_camera_frustum(ax, cam_u[i], cam_v[i], fwd_u[i], fwd_v[i],
                            color=color, size=frustum_size)

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    output_path = args.output or os.path.join(args.model_path, "topdown_view.png")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()