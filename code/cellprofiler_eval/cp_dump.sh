#!/usr/bin/env bash
# Dump every arm's images for CellProfiler into one shared dir.
#
#   bash cellprofiler_eval/cp_dump.sh [gpu]                        # dilution
#   ARMS="gamma0 gamma0.25 gamma0.5 gamma1 gamma2" UNITS= \
#     DUMP=$P4/cp/dump_sweep bash cellprofiler_eval/cp_dump.sh 0   # the sweep
#
# ARMS   space-separated; the first also writes real / recon / control
# UNITS  eval_units csv (default the dilution study units; empty = all units)
# CFG    the config
# DUMP   output dir
# EXTRA  more eval_flow flags, e.g. --source_noise
# CH     channel names in stored order: actin, tubulin, DNA
#
# real / recon / control are shared by every arm and written once. Every later
# arm adds only its own gen set under gen__<arm>/; load_data.csv is appended
# and deduped on (crop_id, population, arm), so re-running an arm is safe.
set -euo pipefail

GPU=${1:-0}
ROOT=${ROOT:-/path/to/workdir}
P4=${P4:-$ROOT/results/iclr/phase4}
SETS=${SETS:-$P4/dilution/sets_final}
DUMP=${DUMP:-$P4/cp/dump_s12}           # never the s=0.2 dump in $P4/cp/dump
CFG=${CFG:-configs/cellflux_percrop_bbbc_fp.yaml}
CH=${CH:-Actin,Tubulin,DNA}
UNITS=${UNITS-$SETS/study_units.csv}
EXTRA=${EXTRA:-}
ARMS=${ARMS:-"dil_q0.1_g0 dil_q0.1_g1 dil_q0.25_g0 dil_q0.25_g1 dil_q0.5_g0
      dil_q0.5_g1 dil_q0.75_g0 dil_q0.75_g1 dil_q0.9_g0 dil_q0.9_g1
      dil_q0.1_g1_oracle dil_q0.1_g1_shuffled dil_q0.25_g1_oracle"}
cd "$(dirname "$0")/.."

# shellcheck disable=SC2086
set -- $ARMS
FIRST=$1; shift; REST="$*"
U=${UNITS:+--eval_units $UNITS}

[[ -z $UNITS || -f $UNITS ]] || {
  echo "no $UNITS -- wrong SETS / UNITS?" >&2; exit 1; }
[[ -f $CFG ]] || { echo "no $CFG (run from the code folder)" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=$GPU
echo "gpu $GPU  ->  $DUMP"

# The first arm also writes real / recon / control, so it always runs; images
# already on disk are not rewritten and the csv dedupes.
# --out on every call keeps these FID-less rows out of $OUT/eval_metrics.csv.
printf '\n\033[1m== %s (+ real / recon / control) ==\033[0m\n' "$FIRST"
# shellcheck disable=SC2086
python evaluation/eval_flow.py --config "$CFG" --arms "$FIRST" \
    --dump_images "$DUMP" --dump_channels "$CH" --no_inception $U $EXTRA \
    --out "$DUMP/eval_metrics_dump.csv"

for arm in $REST; do
  if [[ -d $DUMP/gen__$arm && -n $(ls -A "$DUMP/gen__$arm" 2>/dev/null) ]]; then
    echo "== $arm: gen__$arm already populated, skipping (rm -r it to redo)"
    continue
  fi
  printf '\n\033[1m== %s ==\033[0m\n' "$arm"
  # shellcheck disable=SC2086
  python evaluation/eval_flow.py --config "$CFG" --arms "$arm" \
      --dump_images "$DUMP" --dump_populations gen --dump_channels "$CH" \
      --no_inception $U $EXTRA --out "$DUMP/eval_metrics_dump_$arm.csv"
done

echo
python - "$DUMP" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(f"{sys.argv[1]}/load_data.csv")
print(d.groupby(["Metadata_population", "Metadata_arm"]).size().to_string())
print(f"\n{len(d)} image sets total")
EOF

echo; echo "next: bash cellprofiler_eval/cp_measure.sh"

next, in the CellProfiler env:
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  bash cellprofiler_eval/cp_run.sh $DUMP $P4/cp/bbbc021_crops.cppipe $P4/cp/out 128
EOF
