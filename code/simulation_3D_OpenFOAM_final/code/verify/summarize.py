#!/usr/bin/env python3
"""Write verification/verification_summary.md from verification_summary.csv
(one table per check, pass/fail counts first)."""
from __future__ import annotations

import pandas as pd

from vlib import SUMMARY, VER_DIR

TITLES = {
    "V2": "V2 zero-input cooling (S0 cooldown)",
    "V3": "V3 source normalisation",
    "V4": "V4 analytic limit (exact moving-source plate solution)",
    "V5": "V5 boundary audit (uniform-temperature limit)",
    "V6": "V6 phase audit (latent heat)",
    "V7": "V7 mesh and time-step convergence",
    "V8": "V8 2D comparison (energy)",
    "EC": "Energy closure",
    "SH": "Solver health (thermal-only, converged phase loop)",
    "OUT": "Outputs (0.1 s grid; AdditiveFOAM meltPoolDimensions vs independent recomputation)",
    "DS": "Source-depth sensitivity (documented study, no criteria)",
    "V9": "V9 literature plausibility",
}


def fmt(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{v:.4g}"


def main():
    df = pd.read_csv(SUMMARY)
    judged = df[df["pass"].astype(str).isin(["True", "False"])]
    n_pass = int((judged["pass"].astype(str) == "True").sum())
    lines = ["# Verification summary (generated)", "",
             f"Generated from `verification_summary.csv` by `code/verify/summarize.py`. "
             f"{n_pass} of {len(judged)} judged metrics pass; rows marked `info` carry no criterion.", ""]
    for chk in sorted(df["check"].unique(), key=lambda c: list(TITLES).index(c) if c in TITLES else 99):
        sub = df[df["check"] == chk]
        lines += [f"## {TITLES.get(chk, chk)}", "", "| case | metric | value | criterion | result | note |",
                  "|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            res = {"True": "PASS", "False": "**FAIL**"}.get(str(r["pass"]), "info")
            note = "" if pd.isna(r["note"]) else str(r["note"]).replace("|", "/")
            lines.append(f"| {r['case_id']} | {r['metric']} | {fmt(r['value'])} | {r['criterion']} | {res} | {note} |")
        lines.append("")
    (VER_DIR / "verification_summary.md").write_text("\n".join(lines))
    print(f"{n_pass}/{len(judged)} judged metrics pass -> {VER_DIR / 'verification_summary.md'}")


if __name__ == "__main__":
    main()
