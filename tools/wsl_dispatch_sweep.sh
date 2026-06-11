#!/usr/bin/env bash
# Server-side: dispatch final-sweep scenes to the WSL machine (ssh alias `wsl`), detached.
#   tools/wsl_dispatch_sweep.sh <scene> [scene ...]
# Runs run.sh per scene at the 5 coverage angles with dual checkpoints (run.sh default),
# WSL paths, GPU 0, conda env 3dgs-imc. Logs: outputs/wsl_lane_sweep.log on the WSL side.
set -euo pipefail
SCENES="$*"
[ -n "$SCENES" ] || { echo "usage: $0 <scene> [scene...]"; exit 1; }

WREPO=/mnt/c/Users/Aooooo/Documents/wsl_repos/3dgs-imc-codex
WOUT=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs

# shellcheck disable=SC2087
ssh -o BatchMode=yes wsl bash -s <<EOF
set -u
cd "$WREPO"
cat > /tmp/wsl_lane_runner.sh <<'RUNNER'
#!/usr/bin/env bash
cd "$WREPO"
export CUDA_VISIBLE_DEVICES=0
export SKIP_UVM_PREFLIGHT=1
export CONDA_EXE="\$HOME/miniconda3/bin/conda"
export DATA_ROOT=/mnt/c/Users/Aooooo/Documents/DATA
export OUT_ROOT="$WOUT/sweep_models_final"
export SWEEP_RESULTS="$WOUT/sweep_results_final"
for sc in $SCENES; do
    echo "===== [wsl-lane] scene \$sc ====="
    SCENES="\$sc" bash run.sh
done
echo "[wsl-lane] ALL DONE: $SCENES"
RUNNER
chmod +x /tmp/wsl_lane_runner.sh
nohup setsid bash /tmp/wsl_lane_runner.sh > "$WOUT/wsl_lane_sweep.log" 2>&1 &
sleep 2
echo "dispatched: $SCENES"
tail -2 "$WOUT/wsl_lane_sweep.log" 2>/dev/null || true
EOF
