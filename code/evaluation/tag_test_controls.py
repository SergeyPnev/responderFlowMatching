"""CellFlux's eval sets, marked on our flow index, so the latent arms can be
evaluated in CellFlux's regime.

Writes a copy of the flow index with

    split cf_test       <- CellFlux's test-split DMSO controls
    split cf_ood        <- CellFlux's OOD test crops (SPLIT == test, 8 OOD compounds)
    split cf_ood_train  <- the OOD compounds' SPLIT == train crops, on which a
                           MoA head for the OOD question is fitted

and copies of our config pointing at it (--out_config, --out_config_ood,
--out_config_moa_ood; the last is for moa_eval.py, not eval_flow.py).

    python evaluation/tag_test_controls.py --flow_index flow_index_bbbc021.parquet \\
        --cellflux_index bbbc021_df_all.csv \\
        --config cellflux_percrop_bbbc.yaml \\
        --out_index flow_index_cftest.parquet \\
        --out_config cfg_cftest.yaml --out_config_ood cfg_cftest_ood.yaml
"""
import argparse
import os

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def relabel(idx, keys, y, name, strict: bool = True):
    hit = idx["crop_id"].astype(str).isin(keys)
    if not strict:
        # a partial match is expected here: our index does not carry every
        # row of theirs
        bad = int((idx.loc[hit, "y"] != y).sum())
        if bad:
            raise SystemExit(f"{name}: {bad} crop(s) with y != {y}")
        print(f"{name}: {int(hit.sum())} of {len(keys)} CellFlux crops found, "
              f"from our splits {idx.loc[hit, 'split'].value_counts().to_dict()}"
              f", on {idx.loc[hit, 'plate'].nunique()} plates")
        idx.loc[hit, "split"] = name
        return
    if hit.sum() != len(keys) or (idx.loc[hit, "y"] != y).any():
        raise SystemExit(f"{name}: {len(keys)} CellFlux crops, {int(hit.sum())} "
                         f"found in the flow index, "
                         f"{int((idx.loc[hit, 'y'] != y).sum())} with y != {y}")
    print(f"{name}: {len(keys)} crops, from our splits "
          f"{idx.loc[hit, 'split'].value_counts().to_dict()}, on "
          f"{idx.loc[hit, 'plate'].nunique()} plates")
    idx.loc[hit, "split"] = name


def write_config(cfg, path, **keys):
    yaml.safe_dump(dict(cfg, **keys), open(path, "w"), sort_keys=False)
    print(f"-> {path}  {keys}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--flow_index", required=True)
    p.add_argument("--cellflux_index", required=True,
                   help="bbbc021_df_all.csv, either STATE spelling")
    p.add_argument("--config", required=True,
                   help="our per-crop config (read only)")
    p.add_argument("--out_index", required=True)
    p.add_argument("--out_config", required=True)
    p.add_argument("--out_config_ood", default=None)
    p.add_argument("--out_config_moa_ood", default=None,
                   help="config for moa_eval.py: fit the head on the OOD "
                        "compounds themselves (split cf_ood_train) and report "
                        "it on cf_ood, as CellFlux's checkpoint_ood.pth is")
    p.add_argument("--ood_config", default=os.path.join(HERE, "configs",
                                                        "eval_bbbc_ood.yaml"),
                   help="their OOD eval config; its mol_list minus DMSO is the "
                        "OOD compound set")
    a = p.parse_args()

    cf = pd.read_csv(a.cellflux_index, index_col=0)
    state = cf["STATE"].astype(str)
    test = cf["SPLIT"] == "test"
    ood = [m for m in yaml.safe_load(open(a.ood_config))["mol_list"]
           if m != "DMSO"]
    ctl = set(cf.loc[test & state.isin(["control", "0"]), "SAMPLE_KEY"].astype(str))
    is_trt = state.isin(["trt", "1"])
    ood_keys = set(cf.loc[test & is_trt
                          & cf["CPD_NAME"].isin(ood), "SAMPLE_KEY"].astype(str))
    # their SPLIT partitions crops, so these are disjoint from ood_keys
    ood_tr_keys = set(cf.loc[(cf["SPLIT"] == "train") & is_trt
                             & cf["CPD_NAME"].isin(ood), "SAMPLE_KEY"].astype(str))

    idx = pd.read_parquet(a.flow_index)
    relabel(idx, ctl, 0, "cf_test")
    relabel(idx, ood_keys, 1, "cf_ood")
    if a.out_config_moa_ood:
        relabel(idx, ood_tr_keys, 1, "cf_ood_train", strict=False)
        n = int((idx["split"] == "cf_ood_train").sum())
        if n < 500:
            print(f"! only {n} cf_ood_train crops -- a MoA head fitted on that "
                  f"few is not worth reading")
    idx.to_parquet(a.out_index, index=False)
    print(f"-> {a.out_index}")

    cfg = yaml.safe_load(open(a.config))
    write_config(cfg, a.out_config, eval_flow_index=a.out_index,
                 eval_control_splits=["cf_test"])
    if a.out_config_ood:
        write_config(cfg, a.out_config_ood, eval_flow_index=a.out_index,
                     eval_split="cf_ood", eval_control_splits=["cf_test"])
    if a.out_config_moa_ood:
        # for moa_eval.py only: with eval_flow, these train_splits would put
        # the head's training crops in the flow's target set
        write_config(cfg, a.out_config_moa_ood, flow_index=a.out_index,
                     train_splits=["cf_ood_train"], eval_split="cf_ood",
                     eval_control_splits=["cf_test"])


if __name__ == "__main__":
    main()
