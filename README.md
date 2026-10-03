# Responder-weighted flow matching for cell-painting perturbation images

Not every cell in a treated well responds to the perturbation. This code

1. estimates a **per-crop responder score** `s` from precomputed embeddings, by
   cross-fitting a classifier of treated crops against same-plate control crops,
2. trains a **conditional latent flow** (control crop → treated crop) whose loss is
   weighted per crop by `w ∝ s^γ`, normalised to mean 1 inside each
   (perturbation, plate) cell, so `γ = 0` is the unweighted baseline,
3. evaluates the generated crops with distribution metrics (FID, KID, energy
   distance, PRDC), mechanism-of-action / perturbation classifiers and
   CellProfiler feature effect sizes, including a **dilution study** in which a
   known fraction of each treated set is replaced by control crops.

Datasets: BBBC021 (3 channels), cpg0000 / JUMP (5 channels) and RxRx1 (6 channels).

## Layout

All code is in `code/`

| folder | contents |
|---|---|
| `features/` | `extract_features.py` (per-crop embeddings: MorphEm or DINOv2-g, one grayscale pass per channel), `covariates.py` (imaging covariates for QC) |
| `scorer/` | the responder scorer (`manifest.py`, `core.py`, `run_hidden.py`), its per-dataset configuration (`scorer_config.py`), the score table the flow trains against (`cache_scores.py`, `cap_scores.py`) and control weightings such as shuffled scores (`weight_variants.py`) |
| `autoencoder/` | the KL autoencoder (`vae.py`, `losses.py`, `train.py`, `eval.py`) and the latent cache (`precompute_latents.py`) |
| `flow/` | the conditional latent flow (`dit.py`, `flow_matching.py`, `crop_flow_dataset.py`, `flow_index.py`, `train_cellflux_percrop.py`) and the launchers `run_sweep_arm.sh`, `control_arms.sh`, `run_dilution_arm.sh`, `gamma_arms.sh` |
| `evaluation/` | generation and metrics (`eval_flow.py`, `dist_metrics.py`, `inception_feats.py`), MoA and perturbation classifiers (`moa_eval.py`, `cellflux_moa.py`, `pheno_eval.py`), evaluation-set tagging (`tag_test_controls.py`) |
| `dilution_study/` | plan and build the diluted training sets, and their control weightings |
| `cellprofiler_eval/` | whole-crop CellProfiler features of real and generated crops, and per-unit effect sizes |
| `configs/`, `datasets/` | one autoencoder and one flow config per dataset; the crop dataset classes |

`splits/` (next to `code/`) holds the fold files for cpg0000 and RxRx1.

## Setup

```bash
conda create -n flow python=3.10 -y && conda activate flow
pip install -r requirements.txt
```

Pretrained embedding models are downloaded from the Hugging Face hub on first
use (`facebook/dinov2-giant`, `CaicedoLab/MorphEm`)

### CellProfiler environment

CellProfiler 4.2.8 runs in a separate environment (default name `cp`,
`CP_ENV` overrides it); `cp_measure.sh` switches into it with `conda run`.

```bash
conda create -n cp python=3.9 -y && conda activate cp
conda install -c conda-forge openjdk=11 -y
printf 'numpy<2\n' > $CONDA_PREFIX/constraints.txt
export PIP_CONSTRAINT=$CONDA_PREFIX/constraints.txt
pip install cellprofiler-core==4.2.8
pip install --no-deps cellprofiler==4.2.8
pip install mahotas "inflect<7" imageio joblib "scikit-learn<1" six jinja2 \
            requests "sentry-sdk==0.18.0" "tifffile<2022.4.22" pillow pyyaml
```

Run the metric scripts (`cp_features.py` and everything after it) from the main
environment, not from `cp`.

## Data and paths

Every config and script uses one root, written as `/path/to/workdir` in the
YAML files and read from `$ROOT` by the shell scripts (`$ICLR_ROOT` by
`scorer/scorer_config.py`):

```
$ROOT/data/IMPA_sources/datasets/
    bbbc021_all/                               96x96 single-cell crops (.npy) + metadata/
    cpg0000_u2os_normalized_segmented_large/
    rxrx1/
$ROOT/results/iclr/                            everything this code writes
```

The crops are the single-cell crops distributed with IMPA and CellFlux. Each
dataset directory needs `metadata/split_iclr/` with its fold files:

| dataset | fold files |
|---|---|
| BBBC021 | `ds_bbbc021_train_iid.csv`, `ds_bbbc021_val_iid.csv`, `ds_bbbc021_test_ood_dmso.csv` |
| cpg0000 | `ds_cpg_train_fold0.csv`, `ds_cpg_val_fold0.csv` (in `splits/cpg0000/`, gzipped) |
| RxRx1 | `ds_rxrx1_train_fold0.csv`, `ds_rxrx1_val_fold0.csv` (in `splits/rxrx1/`) |

The flow is conditioned on public perturbation embedding tables
(`emb_fp.csv` for BBBC021, `cpg0000_combined_embeddings.csv`,
`rxrx1_gene2vec_embeddings.csv`); the `text_emb` key of each flow config points
at one of them. Replace `/path/to/workdir` and `/path/to/src` in
`code/configs/*.yaml` with your own locations before running.

BBBC021 crops are stored with channels in the order actin, tubulin, DNA.

## Pipeline

Commands are given for BBBC021. cpg0000 and RxRx1 differ only in the dataset
name, image directory and config. Each script's `--help` and header list the
remaining options.

```bash
cd code
export ROOT=/path/to/workdir
export ICLR_ROOT=$ROOT          # the root scorer/scorer_config.py reads
D=$ROOT/data/IMPA_sources/datasets
F=$ROOT/results/iclr/embeddings
R=$ROOT/results/iclr/hidden_v4b
P4=$ROOT/results/iclr/phase4
```

### 1. Embeddings and crop covariates

```bash
python features/extract_features.py --model morphem --dataset bbbc021 \
    --img_dir $D/bbbc021_all --fold_path $D/bbbc021_all/metadata/split_iclr \
    --output_h5 $F/bbbc021/morphem_ind.h5

python features/covariates.py --dataset bbbc021 --img_dir $D/bbbc021_all \
    --fold_path $D/bbbc021_all/metadata/split_iclr \
    --out $F/bbbc021/crop_covariates_v4.parquet --workers 16
```

`--dry_run` reports which fold files and crops would be read without loading a
model. `--model dinov2g` writes DINOv2-g embeddings instead.

### 2. Responder scorer

Per (perturbation, plate): robust per-feature normalisation on the plate's
controls, PCA fitted inside each fold, class-balanced logistic regression,
folds grouped by well, out-of-fold predictions only, well-level bootstrap for
intervals. The run configuration per dataset is in `scorer_config.py`.

```bash
eval "$(python scorer/scorer_config.py --config v4_bbbc021_morphem_optics --argv --out_dir $R)"
```

Outputs under `$R/bbbc021/morphem_pca50_qcoptics/`: per-unit AUROC and responder
fraction with intervals, and per-crop scores in
`scores/*.parquet`. Configurations for the other datasets:
`v4_cpg_morphem_optics`, `v4_rxrx1_morphem_optics`.

### 3. Autoencoder

```bash
python autoencoder/train.py --config configs/vae_bbbc.yaml          # vae_cpg.yaml, vae_rxrx.yaml
python autoencoder/eval.py  --config configs/vae_bbbc.yaml --ckpt <out_dir>/ckpt/last.pt --use_ema
python autoencoder/ckpt_at_step.py --help                           # copy the checkpoint a flow config names
```

### 4. Flow inputs: index, score cache, latents

```bash
MAN=$R/manifest_bbbc021_f0_split_iclr.parquet
python flow/flow_index.py --manifest $MAN --img_dir $D/bbbc021_all \
    --fold_path $D/bbbc021_all/metadata/split_iclr \
    --out $P4/flow_index_bbbc021.parquet --check_paths 500
python scorer/cache_scores.py --run_dir $R/bbbc021/morphem_pca50_qcoptics \
    --manifest $MAN --out $P4/cache --config v4_bbbc021_morphem_optics
python scorer/cap_scores.py --cache $P4/cache --cap 0.95
CUDA_VISIBLE_DEVICES=0 python autoencoder/precompute_latents.py --index $P4/flow_index_bbbc021.parquet \
    --vae_config configs/cellflux_percrop_bbbc_fp.yaml \
    --out $P4/latents_bbbc021_e227
```

For cpg0000 and RxRx1, `bash flow/gamma_arms.sh <cpg|rxrx1> <gpu> <gamma> ...` runs
this whole section (scorer, index, cache, latents, classifier heads) and then
trains the listed arms.

### 5. Training the arms

```bash
export CFG=configs/cellflux_percrop_bbbc_fp.yaml
bash flow/run_sweep_arm.sh <gpu> <gamma> [seed]             # gamma in {0, 0.25, 0.5, 1, 2}
bash flow/control_arms.sh bbbc021 <gpu> shuffled            # scores permuted within (unit, plate)
bash flow/control_arms.sh cpg <gpu> sampler well_mean       # other weight controls, see its header
```

Each arm writes `<out_dir>/gamma<γ>[_suffix]/` with checkpoints, the effective
sample size per epoch and, at the end, its own evaluation. The BBBC021
config is the fingerprint-conditioned one, `configs/cellflux_percrop_bbbc_fp.yaml`.

### 6. Dilution study

```bash
python dilution_study/dilution_plan.py  --p4 $P4 --index $P4/flow_index_bbbc021.parquet --out $P4/dilution/plan
python dilution_study/dilution_build.py --p4 $P4 --run_dir $R/bbbc021/morphem_pca50_qcoptics \
    --index $P4/flow_index_bbbc021.parquet --out $P4/dilution/sets_final
bash flow/run_dilution_arm.sh <gpu> <q> <gamma>             # q in {0.1, 0.25, 0.5, 0.75, 0.9}
python dilution_study/dilution_controls.py --help                     # oracle / shuffled weights
```

### 7. Evaluation

```bash
python evaluation/moa_eval.py --config $CFG                        # fit the MoA head on real crops (BBBC021)
python evaluation/pheno_eval.py fit --config configs/cellflux_percrop_cpg.yaml   # perturbation heads (cpg0000, RxRx1)
CUDA_VISIBLE_DEVICES=0 python evaluation/eval_flow.py --config $CFG --arms gamma0,gamma1,gamma2 \
    --dump_feats $ROOT/results/iclr/feats/bbbc021_iid --tag m2
python evaluation/dist_metrics.py table --dir $ROOT/results/iclr/feats/bbbc021_iid --out table.csv
```

`eval_flow.py` integrates the flow from same-plate control crops (Heun, 50
function evaluations, classifier-free guidance) and writes FID / KID / energy /
PRDC and classifier metrics per arm; `dist_metrics.py` re-estimates the
distribution metrics from the stored features with well-level bootstrap
intervals.

To score the BBBC021 arms on the public CellFlux evaluation crops,
`evaluation/tag_test_controls.py` marks those crops on the flow index and writes
the matching eval configs (`--ood_config` is CellFlux's `eval_bbbc_ood.yaml`,
which lists the held-out compounds); `eval_flow.py --source_noise` then
evaluates in that regime, and `--moa_head` accepts either our head or
CellFlux's released classifier.

CellProfiler effect sizes:

```bash
bash cellprofiler_eval/cp_dump.sh <gpu>                 # real / reconstructed / generated crops as images
bash cellprofiler_eval/cp_measure.sh 32                 # CellProfiler, per-crop features, effect sizes
```
