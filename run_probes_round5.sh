#!/usr/bin/env bash
# Round-5: nview3 TRANSFER + REGRESSION probes (15k protocol), each cell as a (locked, nview3)
# pair so comparisons stay within-fidelity. GPU 1 only, memory-aware.
#   treehill@80  (under-fill deficit cell, outdoor: AA=0 floater=0, cap 3.3M)   transfer
#   counter@80   (winning indoor cell, AA=1 floater=0.1, cap 1M)                regression check
#   bicycle@20   (winning outdoor cell, cap 5M -> ~22GB peak: STRICTLY 1-wide)  regression check
# 30k refs (locked, this machine v2): treehill@80=?, counter@80=20.700, bicycle@20=22.644.
set -uo pipefail
cd /data1/zhangao/repos/3dgs-imc-codex
DIAG_OUT=/data1/zhangao/outputs/diag/mipnerf360
LOGDIR=/data1/zhangao/outputs/probe_r3fast_logs
mkdir -p "$LOGDIR"

P15K="ITERATIONS=15000 DENSIFY_UNTIL_ITER=12500 DENSIFICATION_INTERVAL=100 NN_START_ITER=1250 NN_DERIVATIVES_ITER=1500 NN_UPDATE_INTERVAL=1500"

n_train() { pgrep -af 'train_experimental_branch3_mc.py' | grep -v 'conda run' | grep -cv 'grep' || true; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1 || echo 0; }
wait_until() { local max_tr=$1 max_mem=$2; while [ "$(n_train)" -ge "$max_tr" ] || [ "$(gpu_used)" -gt "$max_mem" ]; do sleep 60; done; }

# wait for round-4 to fully exit
while pgrep -f run_probes_round4.sh >/dev/null 2>&1; do sleep 60; done

launch_bg() {  # launch_bg <scene> <angle> <cap> <aa> <floater> <tag> <extra...>
    local sc=$1 ang=$2 cap=$3 aa=$4 fl=$5 tag=$6; shift 6
    local before; before=$(n_train)
    echo "[r5 $(date +%H:%M)] launching $tag (trainings=$before, gpu=$(gpu_used)MiB)"
    ( export SCENE=$sc ANGLE=$ang TAG="$tag" CAP_MAX=$cap ANTIALIASING=$aa CPF_FIELD_OPACITY_REG=$fl
      for kv in $P15K "$@"; do [ -n "$kv" ] && export "${kv?}"; done
      bash diag_run.sh ) > "$LOGDIR/$tag.log" 2>&1 &
    for _ in $(seq 1 36); do [ "$(n_train)" -gt "$before" ] && break; sleep 5; done
}

# --- treehill@80 pair (2-wide; each ~13-14GB peak) ---
wait_until 1 8000
launch_bg treehill 80.0 3300000 0 0 th80_locked15k ""
wait_until 2 12000
launch_bg treehill 80.0 3300000 0 0 th80_nv3_15k RANDOM_VIEW_NUM=3
wait

# --- counter@80 pair (2-wide; ~11GB each) ---
wait_until 1 8000
launch_bg counter 80.0 1000000 1 0.1 ct80_locked15k ""
wait_until 2 14000
launch_bg counter 80.0 1000000 1 0.1 ct80_nv3_15k RANDOM_VIEW_NUM=3
wait

# --- bicycle@20 pair (STRICTLY 1-wide: ~22GB peak each) ---
wait_until 1 4000
launch_bg bicycle 20.0 5000000 0 0 bi20_locked15k ""
wait
wait_until 1 4000
launch_bg bicycle 20.0 5000000 0 0 bi20_nv3_15k RANDOM_VIEW_NUM=3
wait

echo "================ ROUND-5 RESULTS (15k pairs) ================"
for t in th80_locked15k th80_nv3_15k ct80_locked15k ct80_nv3_15k bi20_locked15k bi20_nv3_15k; do
    case $t in th80*) sc=treehill; ang=80.0;; ct80*) sc=counter; ang=80.0;; bi20*) sc=bicycle; ang=20.0;; esac
    f="$DIAG_OUT/${sc}_angle_${ang}_${t}_i15000/metrics.csv"
    echo "== $t"; awk -F, 'NR==2{printf "  SSIM %.3f PSNR %.3f LPIPS %.3f\n", $4,$5,$6}' "$f" 2>/dev/null || echo "  (missing)"
done
echo "refs(30k locked): counter@80=20.700(v2) bicycle@20=22.644(v2) treehill@80=14.215(v1) room nview3@30k=16.634"
