#!/usr/bin/env bash
# Drop a job into the persistent probe worker's queue.
#   tools/qjob.sh <name> KEY=VAL [KEY=VAL ...]
# Values may contain spaces if quoted at the call site, e.g.
#   tools/qjob.sh rm100_confirm SCENE=room ANGLE=100.0 CAP_MAX=1200000 'SAVE_ITERATIONS=15000 30000'
# Each KEY=VAL is written %q-quoted so the worker's `source` parses it exactly.
set -euo pipefail
QDIR=${QDIR:-/data1/zhangao/outputs/probe_queue}
mkdir -p "$QDIR/pending"
name=$1; shift
jf="$QDIR/pending/$name.job"
: > "$jf"
for kv in "$@"; do
    k=${kv%%=*}; v=${kv#*=}
    printf '%s=%q\n' "$k" "$v" >> "$jf"
done
echo "queued: $name -> $jf"