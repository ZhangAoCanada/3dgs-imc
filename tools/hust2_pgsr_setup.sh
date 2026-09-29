#!/usr/bin/env bash
# Build a PGSR conda env on hust2 by cloning gaussian_splatting (torch+cuda+toolchain) then
# pip-installing PGSR's CUDA submodules + light deps. pytorch3d skipped (mesh-export only, not training).
set -uo pipefail
C=/opt/anaconda3/bin/conda
P=/data1/ZHANGAo_2024/PGSR
echo "[pgsr-setup $(date +%T)] start"
$C env list | grep -q "^pgsr " || { echo "cloning gaussian_splatting -> pgsr"; $C create -y --clone gaussian_splatting -n pgsr; }
echo "building diff-plane-rasterization + simple-knn ..."
$C run -n pgsr pip install "$P/submodules/diff-plane-rasterization" 2>&1 | tail -3
$C run -n pgsr pip install "$P/submodules/simple-knn" 2>&1 | tail -3
echo "installing light deps ..."
$C run -n pgsr pip install open3d plyfile opencv-python lpips trimesh 2>&1 | tail -2
echo "verify:"
$C run -n pgsr python -c "import torch,diff_plane_rasterization,simple_knn._C; print('PGSR_ENV_OK torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | tail -2
echo "[pgsr-setup $(date +%T)] done"
