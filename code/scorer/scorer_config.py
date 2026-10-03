"""The scorer configs, and the hashes that identify which scores a run used.

    python scorer/scorer_config.py --show                 # the config and its hashes
    python scorer/scorer_config.py --argv                 # the exact command line

Three hashes are stamped onto every cached score file:

    scorer_config_hash   the score-affecting knobs (--n_pcs, --qc, --folds, ...)
    scorer_code_hash     core.py and the QC block of run_hidden.py, as text
    data_index_hash      the crop universe

Paths are hashed by basename, so the hash does not depend on the mount point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, Iterable

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ICLR_ROOT", "/path/to/workdir")

# --------------------------------------------------------------------------- #
# The frozen config
# --------------------------------------------------------------------------- #
# Every knob that changes a score, and nothing else. --qc optics is the primary
# setting; --qc none is the sensitivity check, frozen alongside it.
FROZEN: Dict[str, Dict[str, Any]] = {
    "v4_bbbc021_morphem_optics": {
        "dataset": "bbbc021",
        "embedding": "morphem",
        "emb_dir": f"{ROOT}/results/iclr/embeddings",
        "fold_path": f"{ROOT}/data/IMPA_sources/datasets/bbbc021_all/metadata/split_iclr",
        "fold": 0,
        "covariates": f"{ROOT}/results/iclr/embeddings/bbbc021/crop_covariates_v4.parquet",
        "image_csv": None,
        "moa_csv": None,
        "control": "DMSO",
        "n_pcs": 50,
        "no_pca": False,
        "C": 1.0,
        "n_splits": 5,
        "seed": 0,
        "unit": "auto",
        "pooling": "pooled",
        "min_treated_wells": 2,
        "min_treated_crops": 50,
        "max_control_crops": 8000,
        "units": None,
        "folds": "lowo",
        "qc": "optics",
        "qc_q": 0.001,
    },
}
# The sensitivity arm differs from the primary in exactly one knob.
FROZEN["v4_bbbc021_morphem_none"] = {**FROZEN["v4_bbbc021_morphem_optics"],
                                     "qc": "none"}

# CPG and RxRx1: the same estimator and QC, with the dataset-specific inputs.
# Controls are CONTROL because manifest.py relabels STATE / ANNOT controls
# (perturbation_id.py); RxRx1's `plate` is the experiment. n_pcs is set per
# dataset from its own K sweep (MorphEm's D is 384*C, so K does not transfer).
_D = f"{ROOT}/data/IMPA_sources/datasets"
_E = f"{ROOT}/results/iclr/embeddings"
FROZEN["v4_cpg_morphem_optics"] = {
    **FROZEN["v4_bbbc021_morphem_optics"],
    "dataset": "cpg",
    "fold_path": f"{_D}/cpg0000_u2os_normalized_segmented_large/metadata/split_iclr",
    "covariates": f"{_E}/cpg/crop_covariates_v4.parquet",
    "control": "CONTROL",
    # picked by the K sweep on this dataset
    "n_pcs": 50,
    # no crop floor: the flow trains on every perturbation, so each needs a score
    "min_treated_crops": 0,
}
FROZEN["v4_rxrx1_morphem_optics"] = {
    **FROZEN["v4_cpg_morphem_optics"],
    "dataset": "rxrx1",
    "fold_path": f"{_D}/rxrx1/metadata/split_iclr",
    "covariates": f"{_E}/rxrx1/crop_covariates_v4.parquet",
    # picked by the K sweep on this dataset
    "n_pcs": 50,
    "min_treated_crops": 50,
}

DEFAULT = "v4_bbbc021_morphem_optics"

# Paths are compared and hashed by basename; see the module docstring.
PATH_KEYS = ("emb_dir", "fold_path", "covariates", "image_csv", "moa_csv")


# --------------------------------------------------------------------------- #
ABSENT = "<absent>"


def _canonical(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The score-affecting knobs, path-independent, in a stable order.

    A knob the config does not record reads as ``<absent>``, never as the
    frozen value, so an older run's config cannot hash as a match.
    """
    out = {}
    for k in sorted(FROZEN[DEFAULT]):
        v = cfg.get(k, ABSENT)
        if k in PATH_KEYS and v not in (None, ABSENT):
            v = os.path.basename(os.path.normpath(str(v)))
        out[k] = v
    return out


def _sha(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def config_hash(cfg: Dict[str, Any]) -> str:
    return _sha(json.dumps(_canonical(cfg), sort_keys=True))


def _qc_source() -> str:
    """The QC block out of run_hidden.py, read as text.

    QC decides which crops are scored, so it enters the code hash. Only the
    two objects that implement it (``QC_FAMILIES``, ``qc_defect``) are read,
    textually, to avoid importing run_hidden and its dependencies.
    """
    src = open(os.path.join(HERE, "run_hidden.py")).read()
    out = []
    for start in ("QC_FAMILIES = {", "def qc_defect("):
        i = src.index(start)
        j = src.index("\ndef ", i + len(start))
        out.append(src[i:j])
    return "\n".join(out)


def code_hash(files: Iterable[str] = ("core.py",)) -> str:
    """Digest of the scoring source: the estimator, plus the QC block that
    decides which crops it sees."""
    h = hashlib.sha256()
    for f in sorted(files):
        with open(os.path.join(HERE, f), "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    h.update(hashlib.sha256(_qc_source().encode()).digest())
    return h.hexdigest()[:16]


def data_index_hash(crop_ids: Iterable[str]) -> str:
    """Digest of the crop universe: sorted, deduplicated crop ids
    (order-independent)."""
    ids = sorted(set(map(str, crop_ids)))
    h = hashlib.sha256()
    h.update(str(len(ids)).encode())
    for i in ids:
        h.update(i.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def stamp(name: str = DEFAULT, crop_ids: Iterable[str] | None = None) -> Dict[str, str]:
    """The provenance block written into every cached score file."""
    cfg = FROZEN[name]
    out = {
        "scorer_config_name": name,
        "scorer_config_hash": config_hash(cfg),
        "scorer_code_hash": code_hash(),
    }
    if crop_ids is not None:
        out["data_index_hash"] = data_index_hash(crop_ids)
    return out


def argv_of(name: str = DEFAULT, out_dir: str = f"{ROOT}/results/iclr/hidden_v4b",
            extra: str = "") -> str:
    """The exact run_hidden.py command line this config denotes."""
    c = FROZEN[name]
    parts = [
        "python scorer/run_hidden.py",
        f"--dataset {c['dataset']}", f"--embedding {c['embedding']}",
        f"--emb_dir {c['emb_dir']}", f"--fold_path {c['fold_path']}",
        f"--fold {c['fold']}", f"--covariates {c['covariates']}",
        f"--control {c['control']}", f"--n_pcs {c['n_pcs']}",
        f"--C {c['C']}", f"--n_splits {c['n_splits']}", f"--seed {c['seed']}",
        f"--unit {c['unit']}", f"--pooling {c['pooling']}",
        f"--min_treated_wells {c['min_treated_wells']}",
        f"--min_treated_crops {c['min_treated_crops']}",
        f"--max_control_crops {c['max_control_crops']}",
        f"--folds {c['folds']}", f"--qc {c['qc']}", f"--qc_q {c['qc_q']}",
        f"--out_dir {out_dir}",
    ]
    if c["no_pca"]:
        parts.append("--no_pca")
    if extra:
        parts.append(extra)
    return " \\\n    ".join(parts)


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=DEFAULT, choices=sorted(FROZEN))
    p.add_argument("--show", action="store_true")
    p.add_argument("--argv", action="store_true")
    p.add_argument("--out_dir", default=f"{ROOT}/results/iclr/hidden_v4b")
    p.add_argument("--extra", default="",
                   help="appended to --argv's command")
    a = p.parse_args()

    if a.argv:
        print(argv_of(a.config, a.out_dir, a.extra))
        return 0
    for nm in ([a.config] if a.show else sorted(FROZEN)):
        print(f"== {nm} ==")
        print(json.dumps(_canonical(FROZEN[nm]), indent=2, sort_keys=True))
        for k, v in stamp(nm).items():
            print(f"{k:<22} {v}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
