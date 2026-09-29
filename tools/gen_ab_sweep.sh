#!/usr/bin/env bash
# Generate the paired A/B verification sweep (15k, SEED=42), RAM-balanced:
#   big-cap scenes -> server probe_queue ; small-cap -> WSL dev_queue.
#   Config A (shipped): per-scene AA/floater/nview, NO MH.
#   Config B (MH):      A + MH at the chosen severe angles (env MH_ANGLES, default "80.0 100.0").
# Idempotent: skips a cell whose metrics already exist (server or WSL diag).
#   tools/gen_ab_sweep.sh A      # queue Config A (all 45 cells)
#   tools/gen_ab_sweep.sh B      # queue Config B (MH cells only, severe angles)
set -euo pipefail
WHICH=${1:?A or B}
MH_ANGLES=${MH_ANGLES:-"80.0 100.0"}
# MH params: severity-scaled (aggressive @100, standard @80) — refined from dev findings.
SEED=${SEED:-42}
# FRESHNESS REVERTED (NN_UPDATE_INTERVAL back to 1500): paired SEED=42 test showed it HURTS
# (room@60 fresh 15.83 vs nofresh 16.48 = -0.66; kitchen@20 -0.07). The +0.61 that justified it
# was a single lucky run. Base = the validated non-fresh shipped config.
P15K='ITERATIONS=15000 DENSIFY_UNTIL_ITER=12500 DENSIFICATION_INTERVAL=100 NN_START_ITER=1250 NN_DERIVATIVES_ITER=1500 NN_UPDATE_INTERVAL=1500 RUN_EVAL=1 METRICS_SKIP_NOREF=1'
ANGLES="20.0 40.0 60.0 80.0 100.0"
SVR_DIAG=/data1/zhangao/outputs/diag/mipnerf360
WSL_DIAG=/data1/zhangao/repos/3dgs-imc-codex/outputs/wsl_results/diag/mipnerf360

scene_meta() { # echo: class cap lane
    case "$1" in
        bicycle) echo "outdoor 5000000 server" ;; stump) echo "outdoor 4200000 server" ;;
        garden)  echo "outdoor 4000000 server" ;; treehill) echo "outdoor 3300000 server" ;;
        flowers) echo "outdoor 3000000 server" ;;
        kitchen) echo "indoor 1500000 wsl" ;; room) echo "indoor 1200000 wsl" ;;
        bonsai)  echo "indoor 1000000 wsl" ;; counter) echo "indoor 1000000 wsl" ;;
    esac
}
in_list() { case " $2 " in *" $1 "*) return 0;; *) return 1;; esac; }

queue_server() { tools/qjob.sh "$@"; }
queue_wsl() { ssh -o BatchMode=yes wsl "cd /mnt/c/Users/Aooooo/Documents/wsl_repos/3dgs-imc-codex && QDIR=/mnt/c/Users/Aooooo/Documents/wsl_repos/outputs/dev_queue tools/qjob.sh $*"; }

n=0
for sc in bicycle stump garden treehill flowers kitchen room bonsai counter; do
    read -r cls cap lane <<<"$(scene_meta "$sc")"
    aa=0; floater=0; nview=1
    [ "$cls" = indoor ] && { aa=1; floater=0.1; }
    [ "$sc" = room ] && nview=3
    for a in $ANGLES; do
        [ "$WHICH" = B ] && ! in_list "$a" "$MH_ANGLES" && continue   # B only at MH angles
        tag="${WHICH}_${sc}_${a}"
        # idempotent skip
        [ -f "$SVR_DIAG/${sc}_angle_${a}_${tag}_i15000/metrics.csv" ] && { echo "skip $tag (done)"; continue; }
        [ -f "$WSL_DIAG/${sc}_angle_${a}_${tag}_i15000/metrics.csv" ] && { echo "skip $tag (done)"; continue; }
        env="SCENE=$sc ANGLE=$a CAP_MAX=$cap ANTIALIASING=$aa CPF_FIELD_OPACITY_REG=$floater RANDOM_VIEW_NUM=$nview SEED=$SEED $P15K"
        if [ "$WHICH" = B ]; then
            if [ "$a" = "100.0" ]; then env="$env MH_RELOC=1 MH_INTERVAL=150 MH_START_ITER=3000 MH_BATCH=1024"
            else env="$env MH_RELOC=1 MH_INTERVAL=300 MH_START_ITER=3000 MH_BATCH=512"; fi
        fi
        if [ "$lane" = server ]; then queue_server "$tag" $env; else queue_wsl "$tag" $env; fi
        n=$((n+1))
    done
done
echo "[gen_ab_sweep $WHICH] queued $n cells"
