#!/usr/bin/env bash
# Full ours-FAST timing on WSL (RTX 5090) with the display-freeze fix:
# garbage_collection_threshold:0.9 keeps PyTorch from returning memory to the driver during
# densification reallocs -> no cudaMalloc/cudaFree storms -> no display stalls. Real FAST config
# (densify 200, cap2M, refresh 16). ext4 data (off 9p). Watchdog on.
cd /mnt/c/Users/Aooooo/Documents/wsl_repos/3dgs-imc-codex || exit 1
pgrep -f wsl_watchdog.sh >/dev/null || nohup bash tools/wsl_watchdog.sh >/home/ooooo/wsl_watchdog.out 2>&1 &
export CUDA_VISIBLE_DEVICES=0 NVSMI=/usr/lib/wsl/lib/nvidia-smi
export DATA_ROOT=/home/ooooo/DATA DATASET=tandt OUT_ROOT=/home/ooooo/outputs/tables567
export CONDA_EXE=/home/ooooo/miniconda3/bin/conda ENV_NAME=3dgs-imc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9
export SCENE=truck_resize SPLIT=discontinuous SLOW=0 ANGLE=0 IMAGES=images DEPTHS=dav2
export CAP_MAX=2000000 EPS=0.25 CPF_MAX_REFRESH_VIEWS=16
export ITER=30000 ITERATIONS=30000 SAVE_ITERATIONS=30000 RUN_EVAL=0 TAG=T7_ours_fast
exec bash diag_run.sh
