#!/usr/bin/env bash
# CPG / RxRx1: everything between the MorphEm features and trained gamma arms,
# then the arms in series on one GPU.
#
#   bash flow/gamma_arms.sh <cpg|rxrx1> <gpu> <gamma> [<gamma> ...]
#   bash flow/gamma_arms.sh cpg 0 0 2            # prep, then gamma0, gamma2
#   bash flow/gamma_arms.sh cpg 1 0.5 1 0.25     # waits for the prep, then 3 arms
#
# 1. Prep, under one lock per dataset; each step is skipped when its output
#    exists: scorer run (CPU) -> flow index -> score cache -> pi cap ->
#    latents (GPU) -> trainer dry run -> pheno heads (GPU).
# 2. Arms: a complete arm is skipped, a partial one restarts from scratch
#    (--force), a lock per arm stops two sessions training the same dir, and a
#    failed arm does not stop the others.
#
# env: ROOT (work dir), F (embeddings), R (HiDDEN results), IMG (image dir),
# CFG (config), TMPDIR.
# Needs the MorphEm h5 + covariates under $F/<ds> and the frozen VAE checkpoint
# the config names.
set -uo pipefail          # NOT -e: one failed arm must not kill the rest
cd "$(dirname "$0")/.."

USAGE="usage: bash flow/gamma_arms.sh <cpg|rxrx1> <gpu> <gamma> [<gamma> ...]"
DS=${1:?$USAGE}
GPU=${2:?$USAGE}
shift 2
[ $# -ge 1 ] || { echo "$USAGE"; exit 1; }

ROOT=${ROOT:-/path/to/workdir}
D=$ROOT/data/IMPA_sources/datasets
F=${F:-$ROOT/results/iclr/embeddings}
R=${R:-$ROOT/results/iclr/hidden_v4b}
case $DS in
    cpg)   IMG=${IMG:-$D/cpg0000_u2os_normalized_segmented_large}
           TAG=_broad_state ;;
    rxrx1) IMG=${IMG:-$D/rxrx1}
           TAG=_batch_negctl ;;
    *)     echo "$USAGE"; exit 1 ;;
esac
CFG=${CFG:-configs/cellflux_percrop_$DS.yaml}
SC=v4_${DS}_morphem_optics
export TMPDIR=${TMPDIR:-$ROOT/tmp}

# every path the trainer reads comes from the config, so the two cannot drift
read -r OUT FI LAT VAE PH < <(python -c "
import yaml; c = yaml.safe_load(open('$CFG'))
print(c['out_dir'], c['flow_index'], c['latents'], c['vae']['ckpt'],
      c['eval']['pheno_dir'])")
[ -n "$PH" ] || { echo "! could not read the paths from $CFG"; exit 1; }
P4=$(dirname "$FI")
SCORES=$P4/cache/crop_scores_cap0.95.parquet
MAN=$R/manifest_${DS}_f0_split_iclr$TAG.parquet
LOCKS=$OUT/.locks
mkdir -p "$LOCKS" "$P4" "$TMPDIR"

echo "== $DS  gpu $GPU  gammas: $*  $(date '+%m-%d %H:%M') =="
echo "   config $CFG  out $OUT  p4 $P4"
nvidia-smi -i "$GPU" --query-gpu=index,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null

run() {     # run <name> <cmd...>; a failure stops the session
    local n=$1; shift
    echo
    echo "-- $n  $(date '+%m-%d %H:%M')"
    "$@" || { echo "! $n FAILED (output above)"; exit 1; }
}

for x in "$F/$DS/morphem_ind.h5" "$F/$DS/crop_covariates_v4.parquet" "$VAE"; do
    [ -f "$x" ] || { echo "! missing $x"; exit 1; }
done

K=$(python -c "
from scorer.scorer_config import FROZEN; print(FROZEN['$SC']['n_pcs'])")
RUN=$R/$DS/morphem_pca${K}_qcoptics

# ---- 1. prep ------------------------------------------------------------- #
exec 8>"$P4/.prep.lock"
echo "-- prep (K=$K; waits here if another session holds it)"
flock 8

if [ -f "$RUN/units_manifest.parquet" ]; then
    echo "   scorer: $RUN exists -- skipped"
else
    run scorer bash -c \
        "eval \"\$(python scorer/scorer_config.py --config $SC --argv)\""
fi

if [ -f "$FI" ]; then
    echo "   flow_index: exists -- skipped"
else
    run flow_index python flow/flow_index.py --dataset "$DS" --manifest "$MAN" \
        --img_dir "$IMG" --out "$FI" --check_paths 500
fi

if [ -f "$P4/cache/crop_scores.parquet" ]; then
    echo "   cache_scores: exists -- skipped"
else
    run cache_scores bash -c "python scorer/cache_scores.py \
        --run_dir $RUN --manifest $MAN --out $P4/cache --config $SC"
fi

if [ -f "$SCORES" ]; then
    echo "   cap_scores: exists -- skipped"
else
    run cap_scores python scorer/cap_scores.py --cache "$P4/cache" --cap 0.95
fi

# latents_meta.json is written after latents.npy, so it marks a finished pass
if [ -f "$LAT/latents_meta.json" ]; then
    echo "   latents: exists -- skipped"
else
    X=""; [ -f "$LAT/latents.npy" ] && X=--force
    run latents env CUDA_VISIBLE_DEVICES="$GPU" python autoencoder/precompute_latents.py \
        --index "$FI" --vae_config "$CFG" --out "$LAT" $X
fi

run dry_run python flow/train_cellflux_percrop.py --config "$CFG" --gamma 0 \
    --crop_scores "$SCORES" --dry_run

if [ -f "$PH/heads.pt" ]; then
    echo "   pheno: exists -- skipped"
else
    run pheno env CUDA_VISIBLE_DEVICES="$GPU" \
        python evaluation/pheno_eval.py fit --config "$CFG"
fi
exec 8>&-

# ---- 3. arms ------------------------------------------------------------- #
for G in "$@"; do
    ARM=gamma$(python -c "print(f'{float($G):g}')")
    echo
    echo "== $ARM  $(date '+%m-%d %H:%M') =="
    ST=$(python flow/arm_state.py --config "$CFG" --arm "$OUT/$ARM") \
        && RC=0 || RC=$?
    if [ "$RC" = 0 ]; then
        echo "   $ST -- skipped"; continue
    fi
    exec 9>"$LOCKS/$ARM.lock"
    if ! flock -n 9; then
        echo "   another session is training $ARM -- skipped"; continue
    fi
    X=""; [ "$RC" = 1 ] && X=--force
    echo "   $ST -- starting${X:+ from scratch}"
    if P4=$P4 CFG=$CFG SCORES=$SCORES EXTRA=$X \
        bash flow/run_sweep_arm.sh "$GPU" "$G"; then
        echo "   done: $(python flow/arm_state.py --config "$CFG" --arm "$OUT/$ARM")"
    else
        echo "   ! $ARM FAILED (output above) -- going on with the next gamma"
    fi
    exec 9>&-
done

echo
echo "== session done $(date '+%m-%d %H:%M') =="
for d in "$OUT"/gamma*; do
    [ -d "$d" ] && printf '   %-12s %s\n' "$(basename "$d")" \
        "$(python flow/arm_state.py --config "$CFG" --arm "$d")"
done
