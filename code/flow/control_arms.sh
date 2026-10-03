#!/usr/bin/env bash
# The control arms of a gamma arm: same data, same pairing, one thing changed.
#
#   bash flow/control_arms.sh <cpg|rxrx1|bbbc021> <gpu> <mode> [<mode> ...]
#   bash flow/control_arms.sh rxrx1 0 shuffled
#   bash flow/control_arms.sh cpg 1 sampler well_mean
#   SEED=1 bash flow/control_arms.sh rxrx1 2 shuffled    # -> gamma1_shuffled_s1
#   G=2 bash flow/control_arms.sh bbbc021 0 shuffled     # -> gamma2_shuffled (fp)
#   NORM=unit bash flow/control_arms.sh cpg 0 well_mean  # -> gamma1_well_mean_unitnorm
#
# env:
#   SEED   training seed only (init, data order, pairs); eval keeps the config's
#   G      gamma (default 1; the arm name carries it)
#   NORM   unit = normalise the weights to mean 1 per unit instead of per
#          (unit, plate) cell; adds _unitnorm to the arm name. well_mean needs
#          it: with one treated well per cell its weight is exactly 1.
#   CFG ROOT F R   config and data roots
# bbbc021 means the fingerprint config (configs/cellflux_percrop_bbbc_fp.yaml).
#
# modes:
#   shuffled          s permuted within each (unit, plate) cell: same ESS and
#                     w_max, random assignment
#   well_mean         every crop carries its well's mean s
#   nuisance          s refit from crop covariates alone
#   sampler           real weights, drawn p ~ w with an unweighted loss
#   real              the real scores, weighted loss; only with NORM=unit, as
#                     the matched comparator for well_mean
#   shuffled_sampler  the shuffled scores drawn as in `sampler`; reuses the
#                     shuffled score file
#
# Each arm lands in <out_dir>/gamma<G>_<mode> and evaluates itself. A score
# variant is built once and reused. Needs the prep gamma_arms.sh does (flow
# index, score cache, latents, pheno heads).
set -uo pipefail
cd "$(dirname "$0")/.."

USAGE="usage: bash flow/control_arms.sh <cpg|rxrx1|bbbc021> <gpu> <mode> [<mode> ...]"
DS=${1:?$USAGE}
GPU=${2:?$USAGE}
shift 2
[ $# -ge 1 ] || { echo "$USAGE"; exit 1; }

SEED=${SEED:-}
G=${G:-1}
NORM=${NORM:-}
NSUF=${NORM:+_${NORM}norm}
ROOT=${ROOT:-/path/to/workdir}
F=${F:-$ROOT/results/iclr/embeddings}
R=${R:-$ROOT/results/iclr/hidden_v4b}
case $DS in
    cpg)   FEATS=cells_per_field,foreground_frac ;;
    rxrx1) FEATS=cells_per_field,edge_dist,foreground_frac ;;
    bbbc021) FEATS= ;;
    *)     echo "$USAGE"; exit 1 ;;
esac
DEFCFG=configs/cellflux_percrop_$DS.yaml
[ "$DS" = bbbc021 ] && DEFCFG=configs/cellflux_percrop_bbbc_fp.yaml
CFG=${CFG:-$DEFCFG}
read -r OUT FI < <(python -c "
import yaml; c = yaml.safe_load(open('$CFG'))
print(c['out_dir'], c['flow_index'])")
[ -n "$FI" ] || { echo "! could not read the paths from $CFG"; exit 1; }
P4=$(dirname "$FI")
CACHE=$P4/cache
BASE=$CACHE/crop_scores_cap0.95.parquet
RUN=$R/$DS/morphem_pca50_qcoptics
LOCKS=$OUT/.locks
mkdir -p "$LOCKS"
[ -f "$BASE" ] || { echo "! no $BASE -- run gamma_arms.sh first"; exit 1; }

echo "== $DS  gpu $GPU  modes: $*  $(date '+%m-%d %H:%M') =="

for MODE in "$@"; do
    ARM=gamma${G}_$MODE$NSUF${SEED:+_s$SEED}
    [ "$MODE" = real ] && ARM=gamma$G$NSUF${SEED:+_s$SEED}
    if [ "$MODE" = real ] && [ -z "$NORM" ]; then
        echo "! mode real without NORM is gamma$G itself -- skipped"; continue
    fi
    if [ "$MODE" = nuisance ] && [ -z "$FEATS" ]; then
        echo "! nuisance has no feature list for $DS"; continue
    fi
    # which score variant this mode trains on: the real scores for `sampler`,
    # `shuffled` for `shuffled_sampler`, else the mode's own variant
    case $MODE in
        sampler|real)     VARIANT= ;;
        shuffled_sampler) VARIANT=shuffled ;;
        *)                VARIANT=$MODE ;;
    esac
    SCORES=$BASE
    if [ -n "$VARIANT" ]; then
        SCORES=$CACHE/crop_scores_$VARIANT.parquet
        if [ -f "$SCORES" ]; then
            echo "   $VARIANT scores: exist -- skipped"
        else
            echo
            echo "-- building $VARIANT scores  $(date '+%m-%d %H:%M')"
            EX=()
            [ "$VARIANT" = nuisance ] && EX=(-c "$F/$DS/crop_covariates_v4.parquet"
                                             -r "$RUN" --features "$FEATS")
            python scorer/weight_variants.py -s "$BASE" -m "$VARIANT" -o "$SCORES" \
                "${EX[@]}" || { echo "! $VARIANT build FAILED"; continue; }
        fi
    fi

    echo
    echo "== $ARM  $(date '+%m-%d %H:%M') =="
    ST=$(python flow/arm_state.py --config "$CFG" --arm "$OUT/$ARM") && RC=0 || RC=$?
    if [ "$RC" = 0 ]; then
        echo "   $ST -- skipped"; continue
    fi
    exec 9>"$LOCKS/$ARM.lock"
    if ! flock -n 9; then
        echo "   another session is training $ARM -- skipped"; continue
    fi
    X="--out_suffix $ARM"
    [ "$RC" = 1 ] && X="$X --force"
    case $MODE in sampler|shuffled_sampler) X="$X --sampler" ;; esac
    [ -n "$SEED" ] && X="$X --seed $SEED"
    [ -n "$NORM" ] && X="$X --weight_norm $NORM"
    echo "   $ST -- starting"
    if CUDA_VISIBLE_DEVICES="$GPU" python flow/train_cellflux_percrop.py \
            --config "$CFG" --gamma "$G" --crop_scores "$SCORES" $X; then
        echo "   done: $(python flow/arm_state.py --config "$CFG" --arm "$OUT/$ARM")"
    else
        echo "   ! $ARM FAILED (output above) -- going on with the next mode"
    fi
    exec 9>&-
done

echo
echo "== session done $(date '+%m-%d %H:%M') =="
