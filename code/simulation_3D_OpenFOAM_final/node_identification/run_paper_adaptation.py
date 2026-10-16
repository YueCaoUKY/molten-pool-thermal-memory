#!/usr/bin/env python3
"""Adaptation over the paper set, within each material only.

The model identified on design 1 (ordinary welding speed, varying input) is the
source. Each other design of the same material is a target: three positive
scalars (the memory state's starting value and two scales) are calibrated on the
first 20 % of the target record, and the rest is predicted from the input alone.
Cross-material transfer is not attempted - the two materials have different
melting behaviour, so it would not be a meaningful test.

Both memory states are run, the pool-driven one and the proposed δ, so the two
can be compared on identical pairs.

    python run_paper_adaptation.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PY = sys.executable
MATERIALS = ("steel", "ss304")
DESIGNS = ("2_normal_const", "3_mid_var", "4_slow_cycles", "5_slow_cooled")


def main():
    for mat in MATERIALS:
        source = f"PP_{mat}_1_normal_var"
        targets = [f"PP_{mat}_{d}" for d in DESIGNS]
        cmd = [PY, "adapt_test.py", "--source", source, "--targets", *targets,
               "--variants", "pool", "base", "--threads", "4"]
        print("$ " + " ".join(cmd), flush=True)
        subprocess.call(cmd, cwd=str(HERE))

    rows = list(json.loads((HERE / "results" / "adaptation_scores.json").read_text()).values())
    df = pd.DataFrame([r for r in rows if r["source"].startswith("PP_")])
    if not len(df):
        return
    pd.set_option("display.width", 200)
    cols = ["source", "target", "variant", "prefix_s", "d0", "drive_scale", "rate_scale",
            "r2_none", "r2_adapted", "r2_full", "mem_spearman"]
    print("\n== adaptation, within material ==")
    print(df[cols].to_string(index=False))
    print("\nmeans by memory state:")
    print(df.groupby("variant")[["r2_none", "r2_adapted", "r2_full"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
