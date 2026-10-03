#!/usr/bin/env bash
# CellProfiler over the dump (whole-crop features), then per-crop features and
# per-unit effect sizes.
#
#   bash cellprofiler_eval/cp_measure.sh [jobs]
#
# CellProfiler runs in CP_ENV (default `cp`) through `conda run`; the feature
# scripts run in the calling env.
#
# env: ROOT, P4, DUMP, OUT, FEAT, FILT, EFF (paths), PIPE (pipeline, built if
# missing), CH (channel names), CP_ENV, FORCE=1 (delete a non-empty OUT).
set -euo pipefail

JOBS=${1:-32}
ROOT=${ROOT:-/path/to/workdir}
P4=${P4:-$ROOT/results/iclr/phase4}
DUMP=${DUMP:-$P4/cp/dump_s12}
PIPE=${PIPE:-$DUMP/wholecrop.cppipe}
export CSV_NAME=load_data.csv CP_TABLES=Image.csv
OUT=${OUT:-$P4/cp/out}
FEAT=${FEAT:-$P4/cp/percrop.parquet}
FILT=${FILT:-$P4/cp/feature_filter.csv}
EFF=${EFF:-$P4/cp/effects.csv}
CH=${CH:-Actin,Tubulin,DNA}
CP_ENV=${CP_ENV:-cp}
cd "$(dirname "$0")/.."

[[ -f $DUMP/load_data.csv ]] || {
  echo "no $DUMP/load_data.csv -- run cp_dump.sh first" >&2; exit 1; }

if [[ ! -f $PIPE ]]; then
  conda run -n "$CP_ENV" --no-capture-output python \
      cellprofiler_eval/cp_pipeline.py --segmentation none \
      --channels "$CH" --out "$PIPE"
fi

# Never merge two runs into one out dir: ExportToSpreadsheet refuses to
# overwrite, and stale csvs would be aggregated as a partial run.
if [[ -d $OUT ]] && [[ -n $(ls -A "$OUT" 2>/dev/null) ]]; then
  if [[ ${FORCE:-0} = 1 ]]; then
    echo "FORCE=1: rm -rf $OUT"; rm -rf "$OUT"
  else
    echo "$OUT exists and is not empty. Delete it (or FORCE=1) -- a rerun into" >&2
    echo "an existing out dir silently mixes chunks from two runs." >&2
    exit 1
  fi
fi

# Two caps on JOBS. Memory: one JVM per chunk, ~1-1.5 GB resident. CPU: with
# the thread pools pinned to 1, each chunk wants a physical core.
avail=$(free -g | awk '/^Mem:/ {print $7}')
mem_cap=$(( avail / 2 ))
per_core=$(lscpu | awk -F: '/^Thread\(s\) per core/ {gsub(/ /, "", $2); print $2}')
cpu_cap=$(( $(nproc) / ${per_core:-1} ))
echo "$(nproc) logical cores / $cpu_cap physical, ${avail} GB available"
echo "  caps: cpu $cpu_cap, memory $mem_cap  (load now:$(cut -d' ' -f1-3 /proc/loadavg | sed 's/^/ /'))"
for cap in $cpu_cap $mem_cap; do
  if (( cap > 0 && JOBS > cap )); then
    echo "  capping $JOBS -> $cap"; JOBS=$cap
  fi
done

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

printf '\n\033[1m== CellProfiler (%s env, %s jobs) ==\033[0m\n' "$CP_ENV" "$JOBS"
conda run -n "$CP_ENV" --no-capture-output \
    bash cellprofiler_eval/cp_run.sh "$DUMP" "$PIPE" "$OUT" "$JOBS"

printf '\n\033[1m== features and effect sizes ==\033[0m\n'
python cellprofiler_eval/cp_features.py aggregate \
       --cp_out "$OUT/chunk_*" --out "$FEAT"
python cellprofiler_eval/cp_features.py roundtrip --features "$FEAT" \
       --out "$FILT"
python cellprofiler_eval/cp_features.py effects --features "$FEAT" \
       --filter "$FILT" --out "$EFF"
echo; echo "wrote: $FEAT  $FILT  $EFF"
