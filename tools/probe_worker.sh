#!/usr/bin/env bash
# Persistent probe worker: continuously drains a job queue keeping MAX_SLOTS training jobs
# on the GPU at all times. Never exits until a STOP sentinel — so the GPU never idles at
# round boundaries (the recurring failure of the per-round launch scripts).
#   start: nohup setsid bash tools/probe_worker.sh >/dev/null 2>&1 &
#   stop : touch /data1/zhangao/outputs/probe_queue/STOP   (finishes running jobs, then exits)
#   queue: tools/qjob.sh <name> KEY=VAL ...   (drops a .job into pending/)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, portable across machines
QDIR=${QDIR:-/data1/zhangao/outputs/probe_queue}
PEND="$QDIR/pending"; RUN="$QDIR/running"; DONE="$QDIR/done"; FAIL="$QDIR/failed"; LOGS="$QDIR/logs"
LOG="$QDIR/worker.log"
MAX_SLOTS=${MAX_SLOTS:-2}
GPU_FREE_MIN=${GPU_FREE_MIN:-16000}  # require this many MiB FREE on GPU before starting a new job
                                     # (>= the largest single-scene footprint; auto-serializes big caps)
RAM_MIN=${RAM_MIN:-20}        # need this many GB MemAvailable to start a new job
WORKER_GPU=${WORKER_GPU:-1}
SETTLE=${SETTLE:-45}          # seconds to let a launched job ramp its GPU memory before next slot decision
NVSMI=${NVSMI:-$(command -v nvidia-smi 2>/dev/null || echo /usr/lib/wsl/lib/nvidia-smi)}  # WSL: not on PATH
mkdir -p "$PEND" "$RUN" "$DONE" "$FAIL" "$LOGS"
echo $$ > "$QDIR/worker.pid"
log(){ echo "[worker $(date '+%F %T')] $*" >> "$LOG"; }
log "started pid=$$ slots=$MAX_SLOTS gpu_free_min=${GPU_FREE_MIN}MiB ram_min=${RAM_MIN}G gpu=$WORKER_GPU settle=${SETTLE}s"

# count train AND render AND metrics procs: a job's render/eval phase still loads the GPU,
# so the next job's training must NOT start until the previous job's render+eval finishes.
# Otherwise (with MAX_SLOTS=1) render(A)+train(B) briefly overlap on the GPU — a memory spike
# that can wedge nvidia-uvm on WSL. Counting render/metrics makes it truly one GPU-op at a time.
# MATCH BOTH PIPELINES: train_experimental_branch3.py (slow repro) AND ..._mc.py (fast).
# The old pattern only matched _mc.py, so when draining SLOW jobs n_train returned 0 and the
# launch loop spawned jobs without bound until the GPU-free gate tripped (3 concurrent slow
# trains -> nvidia-uvm wedge -> WSL crash, x3). [._] matches both the dot and the _mc form.
n_train(){ pgrep -af 'train_experimental_branch3[._]|render\.py|metrics\.py' | grep -v 'conda run' | grep -cv 'grep' || true; }
gpu_free(){ "$NVSMI" --query-gpu=memory.free --format=csv,noheader,nounits -i "$WORKER_GPU" 2>/dev/null | head -1 || echo 0; }
ram_avail(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }

# Reconcile any leftover running/ jobs from a previous (killed) worker: a job whose result
# metrics already exist -> done; otherwise -> back to pending for a fresh run. Prevents stuck
# running/ files (which also blinded the old queue-file monitors).
reconcile(){
    for jf in "$RUN"/*.job; do
        [ -e "$jf" ] || continue
        local name; name=$(basename "$jf" .job)
        local sc ang it
        sc=$(grep -E '^SCENE=' "$jf" | head -1 | cut -d= -f2)
        ang=$(grep -E '^ANGLE=' "$jf" | head -1 | cut -d= -f2)
        it=$(grep -E '^ITERATIONS=' "$jf" | head -1 | cut -d= -f2); it=${it:-30000}
        local out="${OUT_ROOT:-/data1/zhangao/outputs/diag}/mipnerf360/${sc}_angle_${ang}_${name}_i${it}/metrics.csv"
        # require a REAL result row (ours_), not just an empty header file: a churned/
        # killed job leaves a header-only metrics.csv, which the old existence check
        # wrongly accepted as done.
        if [ -f "$out" ] && grep -q 'ours_' "$out" 2>/dev/null; then mv "$jf" "$DONE/" 2>/dev/null; log "RECONCILE $name -> done (real metrics)"
        else mv "$jf" "$PEND/" 2>/dev/null; log "RECONCILE $name -> pending (no real metrics)"; fi
    done
}
reconcile

run_job(){
    local jf=$1 name; name=$(basename "$jf" .job)
    mv "$jf" "$RUN/$name.job" 2>/dev/null || return 0
    log "START $name"
    (
        set -a
        # shellcheck disable=SC1090
        source "$RUN/$name.job"      # KEY=VAL lines (quoted safely by qjob.sh); provides SCENE/ANGLE/CAP_MAX/...
        set +a
        export TAG="$name" CUDA_VISIBLE_DEVICES="$WORKER_GPU"
        bash diag_run.sh
    ) > "$LOGS/$name.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then mv "$RUN/$name.job" "$DONE/" 2>/dev/null; else mv "$RUN/$name.job" "$FAIL/" 2>/dev/null; fi
    log "END $name rc=$rc"
}

while [ ! -f "$QDIR/STOP" ]; do
    while [ "$(n_train)" -lt "$MAX_SLOTS" ] && [ "$(ram_avail)" -ge "$RAM_MIN" ]; do
        jf=$(ls -1 "$PEND"/*.job 2>/dev/null | sort | head -1)
        [ -z "$jf" ] && break
        # PER-JOB memory gate: a big-cap scene (>=3M Gaussians ~22GB) needs the GPU nearly empty
        # (run 1-wide); a small-cap scene (~11GB) only needs room to pair. A single fixed threshold
        # can't do both, so derive the requirement from THIS job's CAP_MAX. (The watchdog is the
        # backstop; this prevents the OOM in the first place.)
        cap=$(grep -E '^CAP_MAX=' "$jf" 2>/dev/null | head -1 | cut -d= -f2)
        need="${GPU_FREE_MIN}"; [ "${cap:-0}" -ge 3000000 ] && need=23000
        if [ "$(gpu_free)" -lt "$need" ]; then break; fi
        before=$(n_train)
        run_job "$jf" &
        for _ in $(seq 1 50); do [ "$(n_train)" -gt "$before" ] && break; sleep 3; done   # wait until its python is live
        # MEMORY-PLATEAU WAIT: poll gpu_free until the new job stops claiming memory (two stable
        # reads), so a big-cap job has fully ramped (~22GB) before the next slot decision. A fixed
        # settle was too short for 5M-Gaussian loads -> two big scenes raced past the per-job gate.
        prev=-99999
        for _ in $(seq 1 48); do   # up to ~240s
            cur=$(gpu_free)
            d=$((prev - cur)); [ "$d" -lt 0 ] && d=$((-d))
            [ "$d" -lt 800 ] && break
            prev=$cur; sleep 5
        done
    done
    sleep 15
done
log "STOP sentinel seen — exiting (running jobs continue)"
rm -f "$QDIR/worker.pid"
