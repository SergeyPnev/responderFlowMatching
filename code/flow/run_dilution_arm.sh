#!/usr/bin/env bash
# One dilution arm.
#
#   bash flow/run_dilution_arm.sh <gpu> <q> <gamma>
#   bash flow/run_dilution_arm.sh 5 0.25 1
#
# env: SETS (dilution sets dir), CFG (config), SCORES (score file name inside
# the q dir), SUFFIX (arm name suffix), EXTRA (more trainer flags).
# Everything except --gamma and the q paths is identical across arms.
set -euo pipefail

GPU=${1:?usage: run_dilution_arm.sh <gpu> <q> <gamma>}
Q=${2:?usage: run_dilution_arm.sh <gpu> <q> <gamma>}
G=${3:?usage: run_dilution_arm.sh <gpu> <q> <gamma>}

SETS=${SETS:-/path/to/workdir/results/iclr/phase4/dilution/sets_final}
CFG=${CFG:-configs/cellflux_percrop_bbbc_fp.yaml}
# SCORES / SUFFIX select a control weighting (dilution_controls.py): the same
# set with one score column rewritten.
SCORES=${SCORES:-crop_scores.parquet}
SUFFIX=${SUFFIX:-}
cd "$(dirname "$0")/.."

[ -d "$SETS/q$Q" ] || { echo "no such set: $SETS/q$Q"; exit 1; }

echo "== gpu $GPU  q=$Q  gamma=$G  scores=$SCORES  -> dil_q${Q}_g${G}${SUFFIX} =="
CUDA_VISIBLE_DEVICES="$GPU" python flow/train_cellflux_percrop.py \
    --config "$CFG" \
    --gamma "$G" \
    --out_suffix "dil_q${Q}_g${G}${SUFFIX}" \
    --flow_index    "$SETS/q$Q/flow_index.parquet" \
    --crop_scores   "$SETS/q$Q/$SCORES" \
    --crop_universe "$SETS/q$Q/crop_universe.parquet" \
    --train_splits train \
    --control_splits train \
    --eval_units "$SETS/study_units.csv" \
    ${EXTRA:-}
