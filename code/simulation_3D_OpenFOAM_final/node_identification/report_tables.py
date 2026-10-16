#!/usr/bin/env python3
"""Markdown tables for the held-out protocol report (reads results/ only).

    python report_tables.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CONST = {2}                                   # constant-input conditions: no held-out variance


def per_record():
    t = pd.read_csv(OUT / "identification_metrics_heldout.csv")
    lines = ["| Material | Condition | Split (s) | Identification $R^2$ | Held-out $R^2$ | "
             "Held-out $R^2$ without $\\delta$ | Held-out RMSE | Held-out RMSE without $\\delta$ | "
             "Held-out $\\rho$ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in t.iterrows():
        mark = "$^\\dagger$" if int(r["condition"]) in CONST else ""
        lines.append(f"| {r['material']} | {int(r['condition'])}{mark} | {r['t_split_s']:.1f} | "
                     f"{r['r2_train_with_delta']:.3f} | {r['r2_with_delta']:.3f} | "
                     f"{r['r2_without_delta']:.3f} | {r['rmse_with_delta']:.3f} | "
                     f"{r['rmse_without_delta']:.3f} | {r['spearman_delta_Tw']:.3f} |")
    return "\n".join(lines)


def structure(tag):
    """The chosen candidate tag as it is named in the paper."""
    parts = [x for x in tag.strip("_").split("_") if x != "p60"]
    names = {"na": "$n_a$", "nb": "$n_b$", "h": "hidden width", "w": "window", "d": "width"}
    out = []
    for part in parts:
        key = part.rstrip("0123456789")
        out.append(f"{names.get(key, key)} {part[len(key):]}")
    return ", ".join(out)


def comparison():
    s = json.loads((OUT / "heldout_summary.json").read_text())["comparison_without_constant"]
    order = ["Proposed two-state NODE", "NODE without the memory state", "Vector ARX", "LSTM",
             "Transformer"]
    def init_of(name, v):
        """Samples of measured pool state each structure needs before it can run."""
        if "chosen" not in v:
            return "1 sample"
        parts = {x.rstrip("0123456789"): int(x[len(x.rstrip("0123456789")):])
                 for x in v["chosen"].strip("_").split("_") if x != "p60"}
        if name == "Vector ARX":
            n = max(parts["na"], parts["nb"])
        elif name == "Transformer":
            n = parts["w"]
        else:
            return "1 sample"
        return f"{n} samples"
    lines = ["| Model | Measured initialization | Parameters | Median held-out $R^2$ (worst) | "
             "Median held-out RMSE (worst) |", "|---|---:|---:|---:|---:|"]
    for name in order:
        v = s["models"][name]
        chosen = f" ({structure(v['chosen'])})" if "chosen" in v else ""
        lines.append(f"| {name}{chosen} | {init_of(name, v)} | {v['n_params']:,} | "
                     f"{v['median_r2']:.3f} ({v['worst_r2']:.3f}) | "
                     f"{v['median_rmse']:.3f} ({v['worst_rmse']:.3f}) |")
    return "\n".join(lines), s


def step():
    r = json.loads((OUT / "step_test_p60.json").read_text())
    lines = ["| Model | Held-out $R^2$ | Final-interval RMSE |", "|---|---:|---:|"]
    for name, v in sorted(r["models"].items(), key=lambda kv: -kv[1]["held_out_r2"]):
        lines.append(f"| {name} | {v['held_out_r2']:.3f} | {v['final_steady_state_error']:.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print("## Per record\n")
    print(per_record())
    table, s = comparison()
    print("\n## Model comparison (constant-input records excluded)\n")
    print(table)
    print("\nexcluded:", s["excluded"])
    print("\nmedian one-step held-out R2:",
          {k: round(v["median_one_step_r2"], 4) for k, v in s["models"].items()
           if "median_one_step_r2" in v})
    if (OUT / "step_test_p60.json").exists():
        print("\n## A -> B -> A test\n")
        print(step())
