"""Shared helpers for the verification scripts (plan section 5).

Every check writes rows (check, case_id, metric, value, criterion, pass, note)
to verification/verification_summary.csv; re-running a check replaces its rows.
Pass criteria are fixed in each script before its comparison is computed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mp3d import PROJECT_DIR, RESULTS_DIR, RUN_ROOT  # noqa: E402,F401
from mp3d.case_writer import load_config  # noqa: E402

VER_DIR = PROJECT_DIR / "verification"
FIG_DIR = VER_DIR / "figures"
SUMMARY = VER_DIR / "verification_summary.csv"
FIELDS = ["check", "case_id", "metric", "value", "criterion", "pass", "note", "recorded"]


def case_config(case_id: str) -> dict:
    return load_config(RESULTS_DIR / case_id / "config_resolved.yaml")


def signals(case_id: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / case_id / "signals_0p1s.csv")


def record(rows: list[dict]) -> None:
    VER_DIR.mkdir(exist_ok=True)
    new = pd.DataFrame(rows)
    new["recorded"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    for f in FIELDS:
        if f not in new:
            new[f] = ""
    if SUMMARY.exists():
        old = pd.read_csv(SUMMARY)
        key = ["check", "case_id", "metric"]
        keep = ~old.set_index(key).index.isin(new.set_index(key).index)
        new = pd.concat([old[keep], new[FIELDS]], ignore_index=True)
    new[FIELDS].sort_values(["check", "case_id", "metric"]).to_csv(SUMMARY, index=False)
    for r in rows:
        flag = {True: "PASS", False: "FAIL"}.get(r.get("pass"), "info")
        print(f"  [{flag:4s}] {r['check']:4s} {r['case_id']:28s} {r['metric']:34s} "
              f"{r['value']:.6g}  ({r.get('criterion', '')})")
