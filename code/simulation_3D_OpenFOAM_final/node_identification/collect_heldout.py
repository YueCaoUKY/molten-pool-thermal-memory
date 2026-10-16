#!/usr/bin/env python3
"""Collect the held-out protocol results into one table.

Writes results/identification_metrics_heldout.csv (the same column names as
identification_metrics_current.csv, filled with held-out values) and
results/heldout_summary.json with the model-comparison rows.

    python collect_heldout.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
BASE = OUT / "baselines"
CASES = [c for c in (OUT / "paper_cases.txt").read_text().split() if c]
MATERIAL = {"steel": "Low-carbon steel", "ss304": "AISI 304L"}
POOL, NODELTA = "_v_pool_simpleD_p60", "_nodelta_p60"


def read(tag, case):
    return json.loads((OUT / (case + tag) / "metrics.json").read_text())


def rows():
    out = []
    for case in CASES:
        material, condition = case.split("_")[1], int(case.split("_")[2])
        a, b = read(POOL, case), read(NODELTA, case)
        out.append(dict(
            material=MATERIAL[material], condition=condition, case=case,
            r2_with_delta=a["r2_global_heldout"], r2_without_delta=b["r2_global_heldout"],
            delta_r2=a["r2_global_heldout"] - b["r2_global_heldout"],
            spearman_delta_Tw=a["delta_spearman_heldout"],
            rmse_with_delta=a["nrmse_std_heldout"], rmse_without_delta=b["nrmse_std_heldout"],
            r2_train_with_delta=a["r2_global_train"], r2_train_without_delta=b["r2_global_train"],
            spearman_full=a["delta_spearman"], t_split_s=a["t_split_s"],
            samples=a["n_heldout"], samples_train=a["n_train"],
            r2_per_state_with=json.dumps(a["r2_per_state_heldout"]),
            r2_per_state_without=json.dumps(b["r2_per_state_heldout"]),
        ))
    return pd.DataFrame(out)


def comparison(exclude_constant=True):
    """Median and worst held-out accuracy of every model class."""
    keep = [c for c in CASES if "normal_const" not in c] if exclude_constant else CASES
    table = {}
    node = {"Proposed two-state NODE": POOL, "NODE without the memory state": NODELTA}
    for name, tag in node.items():
        r2 = [read(tag, c)["r2_global_heldout"] for c in keep]
        rmse = [read(tag, c)["nrmse_std_heldout"] for c in keep]
        table[name] = dict(median_r2=float(np.median(r2)), worst_r2=float(np.min(r2)),
                           median_rmse=float(np.median(rmse)), worst_rmse=float(np.max(rmse)),
                           n_params=read(tag, keep[0])["n_params"], per_record={c: v for c, v in zip(keep, r2)})
    sel = json.loads((BASE / "structure_selection_p60.json").read_text())
    label = {"arx": "Vector ARX", "lstm": "LSTM", "transformer": "Transformer"}
    for model in ("arx", "lstm", "transformer"):
        info = [json.loads((BASE / f"{c}_{model}_p60.json").read_text()) for c in keep]
        r2 = [x["r2_heldout"] for x in info]
        rmse = [x["nrmse_std_heldout"] for x in info]
        table[label[model]] = dict(median_r2=float(np.median(r2)), worst_r2=float(np.min(r2)),
                                   median_rmse=float(np.median(rmse)), worst_rmse=float(np.max(rmse)),
                                   n_params=int(info[0]["n_params"]), chosen=sel[model]["chosen"],
                                   median_one_step_r2=float(np.median([x["r2_one_heldout"] for x in info])),
                                   per_record={c: v for c, v in zip(keep, r2)})
    return dict(records=keep, excluded=[c for c in CASES if c not in keep], models=table)


def main():
    table = rows()
    table.to_csv(OUT / "identification_metrics_heldout.csv", index=False)
    summary = dict(train_frac=0.6, comparison_without_constant=comparison(True),
                   comparison_all_records=comparison(False))
    (OUT / "heldout_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    cols = ["material", "condition", "r2_with_delta", "r2_without_delta", "rmse_with_delta",
            "rmse_without_delta", "spearman_delta_Tw", "r2_train_with_delta", "t_split_s"]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:10.4g}"))
    for name, v in summary["comparison_without_constant"]["models"].items():
        print(f"{name:34s} median R2 {v['median_r2']:8.3f}  worst {v['worst_r2']:9.3f}  "
              f"median RMSE {v['median_rmse']:6.3f}  params {v['n_params']}")


if __name__ == "__main__":
    main()
