#!/usr/bin/env bash
# Per-dir harvester: pull WSL T&T + Mill19 checkpoints to the server for the
# Table-1/2 deliverable. SAFE = enumerate dirs via cheap `ls`, then rsync EACH
# dir individually. NEVER a whole-tree rsync (the 134GB tree-scan crashed WSL).
set -uo pipefail
SSH="ssh wsl"
WROOT=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/diag
SROOT=/data1/zhangao/outputs/diag
LOG=/data1/zhangao/outputs/wsl_harvest.log
INTERVAL=${INTERVAL:-1200}
echo "[harvest $(date '+%F %T')] start interval=${INTERVAL}s" >>"$LOG"
while true; do
  for DS in tandt DATA-small; do
    # cheap remote enumeration of finished dirs (have a .ply) — no tree scan
    dirs=$(timeout 40 $SSH "ls -d $WROOT/$DS/*/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | sed 's#/point_cloud/iteration.*##' | sort -u" 2>/dev/null)
    for d in $dirs; do
      name=$(basename "$d")
      # skip if server already has this dir with a ply (idempotent)
      if ls "$SROOT/$DS/$name"/point_cloud/iteration_*/point_cloud.ply >/dev/null 2>&1; then continue; fi
      mkdir -p "$SROOT/$DS"
      if timeout 600 rsync -a -e ssh "wsl:$d/" "$SROOT/$DS/$name/" 2>>"$LOG"; then
        echo "[harvest $(date '+%F %T')] pulled $DS/$name" >>"$LOG"
      else
        echo "[harvest $(date '+%F %T')] FAILED $DS/$name" >>"$LOG"
      fi
    done
  done
  sleep "$INTERVAL"
done
