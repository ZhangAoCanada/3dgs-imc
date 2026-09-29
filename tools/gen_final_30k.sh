#!/usr/bin/env bash
# Queue the 30K FINAL run: every cell at the full 30K schedule, DUAL-checkpoint (15000+30000),
# with the per-cell-BEST config locked from the 15K tuning sweep. Big scenes -> server seq_runner,
# small -> WSL worker; the bidirectional sentinel balances. Produces the two final table rows.
#   bash tools/gen_final_30k.sh        # queue all 45 cells
#   DRY=1 bash tools/gen_final_30k.sh  # print what would be queued, queue nothing
set -euo pipefail
SEED=${SEED:-42}
DRY=${DRY:-0}
# FULL 30K schedule (diag_run.sh 30K defaults; NOT the compressed 15K dev protocol). Non-fresh NN.
P30K='ITERATIONS=30000 DENSIFY_FROM_ITER=500 DENSIFY_UNTIL_ITER=25000 DENSIFICATION_INTERVAL=200 NN_START_ITER=2500 NN_DERIVATIVES_ITER=3000 NN_UPDATE_INTERVAL=3000 RUN_EVAL=1 METRICS_SKIP_NOREF=1'
ANGLES="20.0 40.0 60.0 80.0 100.0"
SVR_DIAG=/data1/zhangao/outputs/diag/mipnerf360
WSL_DIAG=/data1/zhangao/repos/3dgs-imc-codex/outputs/wsl_results/diag/mipnerf360

scene_meta() { case "$1" in
    bicycle) echo "outdoor 5000000 server";; stump) echo "outdoor 4200000 server";;
    garden)  echo "outdoor 4000000 server";; treehill) echo "outdoor 3300000 server";;
    flowers) echo "outdoor 3000000 server";;
    kitchen) echo "indoor 1500000 wsl";; room) echo "indoor 1200000 wsl";;
    bonsai)  echo "indoor 1000000 wsl";; counter) echo "indoor 1000000 wsl";;
esac; }

# per-cell-BEST override locked from 15K tuning (default = A shipped, empty override).
# A later CAP_MAX=/MH_/STOCHASTIC_ line wins over the defaults when the job file is sourced.
override() { case "${1}_${2}" in
    bicycle_100.0|bonsai_100.0|stump_100.0|room_100.0|treehill_80.0) echo "MH_RELOC=1 MH_INTERVAL=150 MH_START_ITER=6000 MH_BATCH=1024";;  # aggressive MH (B/T1 won)
    bonsai_80.0|kitchen_80.0|stump_80.0) echo "MH_RELOC=1 MH_INTERVAL=300 MH_START_ITER=6000 MH_BATCH=512";;                              # standard MH (B won)
    bonsai_20.0|room_40.0) echo "CAP_MAX=2000000";;                                                                                       # capacity 2M (T1/T2cap won)
    room_60.0) echo "STOCHASTIC_STOP_ITER=24000";;                                                                                        # noise anneal @80% of 30K (T1 won)
    *) echo "";;
esac; }

queue_server() { tools/qjob.sh "$@"; }
queue_wsl() { ssh -o BatchMode=yes wsl "cd /mnt/c/Users/Aooooo/Documents/wsl_repos/3dgs-imc-codex && QDIR=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/dev_queue tools/qjob.sh $*"; }

n=0
for sc in bicycle stump garden treehill flowers kitchen room bonsai counter; do
    read -r cls cap lane <<<"$(scene_meta "$sc")"
    aa=0; floater=0; nview=1
    [ "$cls" = indoor ] && { aa=1; floater=0.1; }
    [ "$sc" = room ] && nview=3
    for a in $ANGLES; do
        tag="F_${sc}_${a}"
        # idempotent skip: 30K metrics row already present
        for D in "$SVR_DIAG" "$WSL_DIAG"; do
            f="$D/${sc}_angle_${a}_${tag}_i30000/metrics.csv"
            [ -f "$f" ] && grep -q ours_30000 "$f" 2>/dev/null && { echo "skip $tag (30k done)"; continue 2; }
        done
        ov="$(override "$sc" "$a")"
        env="SCENE=$sc ANGLE=$a CAP_MAX=$cap ANTIALIASING=$aa CPF_FIELD_OPACITY_REG=$floater RANDOM_VIEW_NUM=$nview SEED=$SEED $P30K $ov"
        if [ "$DRY" = 1 ]; then
            printf "%-16s [%s] %s\n" "$tag" "$lane" "${ov:-A(shipped)}"
        else
            if [ "$lane" = server ]; then queue_server "$tag" $env "SAVE_ITERATIONS=15000 30000"
            else queue_wsl "$tag" $env "'SAVE_ITERATIONS=15000 30000'"; fi
        fi
        n=$((n+1))
    done
done
echo "[gen_final_30k] $([ "$DRY" = 1 ] && echo 'WOULD queue' || echo 'queued') $n cells"
