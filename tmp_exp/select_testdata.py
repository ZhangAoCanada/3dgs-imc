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
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    return cam_extrinsics

def main():
    parser = argparse.ArgumentParser(description="Select N nearest viewpoints to a given image.")
    parser.add_argument("--source_path", "-s", required=True, type=str, help="Path to the source directory containing sparse/0/")
    parser.add_argument("--image_name", "-i", required=True, type=str, help="Name of the target image")
    parser.add_argument("--n", "-n", required=True, type=int, nargs='+', help="Number(s) of nearest neighbors to select")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to plot camera poses")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_path):
        print(f"Error: Source path {args.source_path} does not exist.")
        sys.exit(1)
        
    print(f"Reading colmap info from {args.source_path}...")
    try:
        extrinsics = readColmapSceneInfo(args.source_path)
    except Exception as e:
        print(f"Error reading colmap info: {e}")
        sys.exit(1)
        
    target_center = None
    all_cameras = []
    
    # Iterate through all cameras to find target and compute centers
    for cam_id, cam in extrinsics.items():
        R = qvec2rotmat(cam.qvec)
        t = cam.tvec
        center = -R.T @ t

        # center = cam.tvec
        
        if cam.name == args.image_name:
            target_center = center
        
        all_cameras.append((cam.name, center))
        
    if target_center is None:
        print(f"Error: Target image '{args.image_name}' not found in the dataset.")
        sys.exit(1)
        
    print(f"Found target image '{args.image_name}' at {target_center}")
    
    # Calculate distances from target
    distances = []
    for name, center in all_cameras:
        # Include the target image itself in the distance calculation
        dist = np.linalg.norm(center - target_center)
        distances.append((name, center, dist))
        
    # Sort by distance
    distances.sort(key=lambda x: x[2])
    
    # Process each n in the list
    for n in args.n:
        # Select top N (which will include the target itself as the first element since dist=0)
        selected = distances[:n]
        
        print(f"Selected {len(selected)} nearest images for n={n}:")
        if len(selected) <= 20: # Only print if list is small enough
            for name, center, dist in selected:
                print(f"  {name}: {dist:.4f}")
            
        output_file = os.path.join(args.source_path, f"test_selection_{n}.txt")
        with open(output_file, "w") as f:
            for name, _, _ in selected:
                f.write(f"{name}\n")
                
        print(f"Saved selected image names to {output_file}")

    # Use the largest n for visualization
    max_n = max(args.n)
    selected = distances[:max_n]

    if args.debug:
        print("Plotting camera poses...")
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("Error: matplotlib is required for debug mode. Please install it.")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Collect points
        selected_centers = np.array([c for _, c, _ in selected])
        other_centers = []
        selected_names = set([name for name, _, _ in selected])
        
        for name, center in all_cameras:
            if name not in selected_names:
                other_centers.append(center)
        
        other_centers = np.array(other_centers)
        
        # Plot other cameras
        if len(other_centers) > 0:
            ax.scatter(other_centers[:, 0], other_centers[:, 1], other_centers[:, 2], c='blue', marker='.', s=5, alpha=0.3, label='Others')
            
        # Plot selected cameras
        if len(selected_centers) > 0:
            ax.scatter(selected_centers[:, 0], selected_centers[:, 1], selected_centers[:, 2], c='green', marker='^', s=50, label='Selected')
            
        # Plot target camera (emphasize it even if it is in selected)
        ax.scatter(target_center[0], target_center[1], target_center[2], c='red', marker='*', s=100, label='Target')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title(f"Nearest Neighbors for {args.image_name}")
        plt.show()

if __name__ == "__main__":
    main()
