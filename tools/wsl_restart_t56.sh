#!/usr/bin/env bash
# Restart the WSL ours worker with a new JOBS set (default: all 22, Table-5 first). Kills the old
# worker + any in-flight training, then relaunches. Named so pkill -f run_t56 won't self-match.
pkill -f run_t56.sh 2>/dev/null
pkill -f train_experimental_branch3 2>/dev/null
pkill -f "render.py" 2>/dev/null
sleep 3
cd /mnt/c/Users/Aooooo/Documents/wsl_repos/3dgs-imc-codex || exit 1
export JOBS="${JOBS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22}"
nohup bash tools/wsl_launch_t56.sh > /home/ooooo/run_t56_wsl.log 2>&1 &
echo "restarted wsl worker pid $! JOBS=$JOBS"
