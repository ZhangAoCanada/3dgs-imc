#!/usr/bin/env bash
# Launch the Table-5/6 ours worker on WSL (RTX 5090). Sets WSL-specific paths/conda, then runs
# run_t56.sh sequentially (1 job at a time -> low crash risk). JOBS overridable via env.
cd /mnt/c/Users/Aooooo/Documents/wsl_repos/3dgs-imc-codex || exit 1
export CUDA_VISIBLE_DEVICES=0
export FREE_NEED=${FREE_NEED:-16000}
export REPO_DIR=/mnt/c/Users/Aooooo/Documents/wsl_repos/3dgs-imc-codex
export DATA_ROOT=/home/ooooo/DATA          # ext4 (native), NOT /mnt/c 9p — avoids WSL I/O stalls
export OUT_ROOT=/home/ooooo/outputs/tables567
export CONDA_EXE=/home/ooooo/miniconda3/bin/conda
export ENV_NAME=3dgs-imc
export NVSMI=/usr/lib/wsl/lib/nvidia-smi
# WSL crash fix: cap CPF geometry-field refresh to 16 views (was 0=all ~220) so the periodic
# update_nnpts GPU burst doesn't starve the Windows display on the shared 5090. (server keeps default.)
export CPF_MAX_REFRESH_VIEWS=16
export JOBS="${JOBS:-12 13 14 15 16 17 18 19 20 21 22}"
exec bash tools/run_t56.sh
