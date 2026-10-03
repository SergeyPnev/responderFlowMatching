#!/usr/bin/env bash
# Headless CellProfiler over a cp_io dump, chunked across cores.
#
#   conda activate <the env where CellProfiler is installed>
#   bash cellprofiler_eval/cp_run.sh DUMP_DIR PIPELINE.cppipe OUT_DIR [N_JOBS]
#
# env: CSV_NAME (image-set csv in the dump), CP_TMP (temp dir), SETS_PER_CHUNK,
# CP_TABLES (tables every chunk must write).
# Chunks are separate CellProfiler processes over disjoint image-set ranges,
# each writing its own output dir. cp_features.py aggregate joins them on
# crop_id, never on ImageNumber, which restarts inside every chunk.
set -euo pipefail

DUMP=${1:?usage: cp_run.sh DUMP_DIR PIPELINE.cppipe OUT_DIR [N_JOBS]}
PIPE=${2:?}
OUT=${3:?}
JOBS=${4:-$(nproc)}
# CSV_NAME=load_data_cellpose.csv for a --segmentation cellpose pipeline, which
# needs the mask columns
CSV="$DUMP/${CSV_NAME:-load_data.csv}"

# Each chunk streams its measurements into an HDF5 in the temp dir, and a full
# temp dir crashes the JVM. Keep the temp next to the output; SETS_PER_CHUNK
# caps how many image sets are in flight.
TMP=${CP_TMP:-$OUT/tmp}
export TMPDIR="$TMP"
ulimit -c 0                    # a JVM crash otherwise drops a core in the cwd

command -v cellprofiler >/dev/null || {
  echo "cellprofiler not on PATH -- activate the conda env first" >&2; exit 1; }
[[ -f $CSV ]] || { echo "no $CSV (run eval_flow.py --dump_images first)" >&2; exit 1; }
[[ -f $PIPE ]] || { echo "no $PIPE (build it with cp_pipeline.py)" >&2; exit 1; }

# One BLAS/OpenMP thread per chunk: the parallelism is the chunking.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

N=$(( $(wc -l < "$CSV") - 1 ))
(( N > 0 )) || { echo "$CSV has no rows" >&2; exit 1; }
CHUNK=${SETS_PER_CHUNK:-$(( (N + JOBS - 1) / JOBS ))}
NCHUNK=$(( (N + CHUNK - 1) / CHUNK ))
mkdir -p "$OUT" "$TMP"
echo "cellprofiler $(cellprofiler --version 2>&1 | head -1)"
echo "$N image sets, $CHUNK per chunk, $NCHUNK chunks, $JOBS at a time -> $OUT"
echo "temp: $TMP  ($(df -h --output=avail "$TMP" 2>/dev/null | tail -1 | tr -d ' ') free)"

pids=(); k=0; fail=0
# The chunks are background jobs; kill them when this script is killed.
cleanup() { trap - INT TERM EXIT; (( ${#pids[@]} )) && kill "${pids[@]}" 2>/dev/null
            rm -rf "$TMP"; return 0; }
trap cleanup INT TERM EXIT

# Wait out the current wave and delete its temp files.
drain() { local p; for p in "${pids[@]}"; do wait "$p" || fail=1; done
          pids=(); rm -rf "$TMP"/chunk_*; }

for (( first=1; first<=N; first+=CHUNK )); do
  last=$(( first + CHUNK - 1 )); (( last > N )) && last=$N
  d="$OUT/chunk_$(printf %03d "$k")"; mkdir -p "$d" "$TMP/chunk_$k"
  echo "  chunk $k/$(( NCHUNK - 1 )): image sets $first..$last -> $d"
  cellprofiler -c -r -p "$PIPE" --data-file="$CSV" -o "$d" -t "$TMP/chunk_$k" \
               -f "$first" -l "$last" > "$d/cp.log" 2>&1 &
  pids+=($!); k=$(( k + 1 ))
  (( ${#pids[@]} >= JOBS )) && drain
done
(( ${#pids[@]} )) && drain

if (( fail )); then
  echo "at least one chunk failed; last 30 lines of each failing log:" >&2
  for d in "$OUT"/chunk_*; do
    grep -qiE "error|traceback" "$d/cp.log" && {
      echo "--- $d/cp.log"; tail -30 "$d/cp.log"; }
  done >&2
  exit 1
fi

# CellProfiler exits 0 even when a module raises mid-run or post_run fails, and
# a failed ExportToSpreadsheet leaves an empty csv. Check logs and outputs.
bad=0
for d in "$OUT"/chunk_*; do
  if grep -qE "Error detected during run|Failed to complete post_run" "$d/cp.log"; then
    echo "!! $d: CellProfiler reported an error but exited 0:" >&2
    grep -A6 -E "Error detected during run|Failed to complete post_run" \
        "$d/cp.log" | head -20 >&2
    bad=1
  fi
  for f in ${CP_TABLES:-Image.csv Nuclei.csv}; do
    [[ -s "$d/$f" ]] || { echo "!! $d/$f missing or empty" >&2; bad=1; }
  done
done
(( bad )) && { echo "-> fix the pipeline and rerun; do NOT aggregate this." >&2; exit 1; }

rows=$(cat "$OUT"/chunk_*/Image.csv 2>/dev/null | grep -cv '^ImageNumber' || true)
echo "done: $rows image rows across $k chunks"
echo
echo "next:"
echo "  python cellprofiler_eval/cp_features.py aggregate --cp_out '$OUT/chunk_*' --out percrop_features.parquet"
echo "  python cellprofiler_eval/cp_features.py roundtrip --features percrop_features.parquet --out feature_filter.csv"
echo "  python cellprofiler_eval/cp_features.py effects   --features percrop_features.parquet --filter feature_filter.csv"
