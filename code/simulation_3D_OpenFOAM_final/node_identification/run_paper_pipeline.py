#!/usr/bin/env python3
"""Unattended driver for the paper set: simulations, features, identification, scoring.

  stage sim    every case in results/paper_cases.txt: run_case.py then
               extract_features.py, two simulations at a time (12 ranks each)
  stage fit    four fits per case: the pool-driven memory state from two seeds,
               the proposed δ, and no memory state; six fits at a time
  stage score  delta_variants.py score for the whole set

    python run_paper_pipeline.py sim fit score
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
import os
# interpreter for the worker processes; override with MP_PYTHON
PY = os.environ.get("MP_PYTHON", sys.executable)
LOGS = BENCH / "logs" / "paper"
CASES = [c for c in (HERE / "results" / "paper_cases.txt").read_text().split() if c]


def sh(cmd, log, cwd):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        return subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(cwd))


def simulate(case):
    log = LOGS / f"{case}.sim.log"
    rc = sh(["python3", BENCH / "code" / "run_case.py", BENCH / "configs" / f"{case}.yaml",
             "--steps", "all", "--nprocs", "12"], log, BENCH)
    if rc == 0:
        rc = sh(["python3", "extract_features.py", case], log, HERE)
    print(f"{case} sim rc={rc}", flush=True)
    return rc


def fit_jobs(case):
    sys.path.insert(0, str(HERE))
    from delta_variants import window_for
    t0, t1 = window_for(case)
    w = ["--window", str(t0), str(t1)]
    return [
        (f"{case} pool", [PY, "delta_variants.py", "fit", case, "pool", "--threads", "2"]),
        (f"{case} pool_s7", [PY, "delta_variants.py", "fit", case, "pool", "--threads", "2",
                             "--seed", "7", "--suffix", "_s7"]),
        (f"{case} base", [PY, "identify_3d.py", case, "--threads", "2", *w]),
        (f"{case} nodelta", [PY, "identify_3d.py", case, "--threads", "2", "--no-delta", "--tag", "_nodelta", *w]),
    ]


def run_fit(job):
    name, cmd = job
    rc = sh(cmd, LOGS / f"{name.split()[0]}.fit.log", HERE)
    print(f"{name} rc={rc}", flush=True)
    return rc


def main():
    stages = sys.argv[1:] or ["sim", "fit", "score"]
    if "sim" in stages:
        print(f"== simulations: {len(CASES)} cases, two at a time", flush=True)
        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(simulate, CASES))
    if "fit" in stages:
        jobs = [j for c in CASES for j in fit_jobs(c)]
        print(f"== identification: {len(jobs)} fits, six at a time", flush=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(run_fit, jobs))
    if "score" in stages:
        print("== scoring", flush=True)
        subprocess.call([PY, "delta_variants.py", "score", "--cases", *CASES,
                         "--variants", "base", "pool"], cwd=str(HERE))


if __name__ == "__main__":
    main()
