#!/usr/bin/env bash
# One gamma-sweep arm on one GPU.
#
#   bash flow/run_sweep_arm.sh <gpu> <gamma> [seed]
#   bash flow/run_sweep_arm.sh 3 0.5          # -> gamma0.5
#   bash flow/run_sweep_arm.sh 3 0.5 1        # -> gamma0.5_s1 (training seed 1)
#
# env: P4, CFG (config), SCORES (score file), EXTRA (more trainer flags).
# Every arm, gamma=0 included, trains on the pi-capped weights from
# cap_scores.py. The eval reference is still the config's own cache.
set -euo pipefail

GPU=${1:?usage: run_sweep_arm.sh <gpu> <gamma>}
G=${2:?usage: run_sweep_arm.sh <gpu> <gamma>}
S=${3:-}

P4=${P4:-/path/to/workdir/results/iclr/phase4}
CFG=${CFG:-configs/cellflux_percrop_bbbc_fp.yaml}
SCORES=${SCORES:-$P4/cache/crop_scores_cap0.95.parquet}
cd "$(dirname "$0")/.."

[ -f "$SCORES" ] || { echo "no $SCORES -- run cap_scores.py first"; exit 1; }

echo "== gpu $GPU  gamma=$G  seed=${S:-config}  scores=$SCORES  -> gamma$G${S:+_s$S} =="
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPU" python flow/train_cellflux_percrop.py \
    --config "$CFG" \
    --gamma "$G" \
    --crop_scores "$SCORES" \
    ${S:+--seed "$S"} \
    ${EXTRA:-}
