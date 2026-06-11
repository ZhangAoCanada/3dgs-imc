#!/usr/bin/env bash
# Server-side: enqueue a job for the WSL worker.
#   tools/wsl_enqueue.sh <name> KEY=VAL [KEY=VAL ...]
# Probe example:
#   tools/wsl_enqueue.sh rm100_colorreg30k KIND=diag SCENE=room ANGLE=100.0 TAG=colorreg30k \
#       CAP_MAX=1200000 ANTIALIASING=1 CPF_FIELD_OPACITY_REG=0.1 RANDOM_VIEW_NUM=3 \
#       CPF_FIELD_RGB_LOSS=1e3 CPF_FIELD_COLOR_REG=0.2 ITERATIONS=30000
# Final-cell example (run.sh path, dual checkpoints automatic):
#   tools/wsl_enqueue.sh sweep_garden KIND=run SCENES=garden ANGLES="20.0 40.0 60.0 80.0 100.0"
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
QDIR="$REPO/outputs/wsl_queue"
mkdir -p "$QDIR/pending" "$QDIR/done" "$QDIR/failed"
name=$1; shift
jf="$QDIR/pending/${name}.job"
{
    echo "# queued $(date '+%F %T') on $(hostname)"
    for kv in "$@"; do echo "$kv"; done
} > "$jf"
echo "queued: $jf"
cat "$jf"
