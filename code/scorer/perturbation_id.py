"""What a perturbation is, and what a control is, per dataset.

    bbbc021  perturbation = CPD_NAME, control = DMSO by name
    cpg      perturbation = BROAD_SAMPLE, control = STATE == "control" on every
             plate type; all controls are relabelled CONTROL
    rxrx1    perturbation = CPD_NAME (siRNA gene), control = ANNOT ==
             "negative_control" relabelled CONTROL; ANNOT == "positive_control"
             is its own role, named ``<gene>__positive_control`` because a
             gene can be both treated and a positive control

CPG's CPD_NAME is a gene/target name shared by a compound, its CRISPR guides
and its ORF, so units are keyed on BROAD_SAMPLE (as in CellFlux). The original
control label is kept in ``control_label``.
"""
from __future__ import annotations

import pandas as pd

CONTROL = "CONTROL"


def apply_identity(dataset: str, df: pd.DataFrame,
                   treatment_col: str = "CPD_NAME") -> pd.DataFrame:
    """Add ``pert``, ``pert_role`` (treated / control / positive_control) and
    ``control_label`` to a frame of raw fold-CSV rows. bbbc021 gets
    ``pert = treatment_col`` and no role (its control is matched by name)."""
    out = df.copy()
    if dataset == "cpg":
        miss = {"BROAD_SAMPLE", "STATE"} - set(out.columns)
        if miss:
            raise SystemExit(f"cpg fold CSVs lack {sorted(miss)}: the "
                             f"perturbation is BROAD_SAMPLE and the control is "
                             f"STATE (split_iclr carries both)")
        state = out["STATE"].astype(str).str.lower()
        bad = sorted(set(state) - {"control", "trt"})
        if bad:
            raise SystemExit(f"cpg STATE values {bad}; expected control / trt")
        ctl = state.eq("control")
        out["pert_role"] = ctl.map({True: "control", False: "treated"})
        out["control_label"] = out["BROAD_SAMPLE"].where(ctl)
        out["pert"] = out["BROAD_SAMPLE"].astype(str).where(~ctl, CONTROL)
    elif dataset == "rxrx1":
        if "ANNOT" not in out.columns:
            raise SystemExit("rxrx1 fold CSVs lack ANNOT (negative_control / "
                             "positive_control / treated)")
        annot = out["ANNOT"].astype(str)
        bad = sorted(set(annot) - {"treated", "negative_control",
                                   "positive_control"})
        if bad:
            raise SystemExit(f"rxrx1 ANNOT values {bad}; expected treated / "
                             f"negative_control / positive_control")
        role = annot.map({"treated": "treated", "negative_control": "control",
                          "positive_control": "positive_control"})
        ctl = role.eq("control")
        out["pert_role"] = role
        out["control_label"] = out[treatment_col].where(ctl)
        name = out[treatment_col].astype(str)
        name = name.where(~role.eq("positive_control"),
                          name + "__positive_control")
        out["pert"] = name.where(~ctl, CONTROL)
    else:
        out["pert"] = out[treatment_col]
    return out
