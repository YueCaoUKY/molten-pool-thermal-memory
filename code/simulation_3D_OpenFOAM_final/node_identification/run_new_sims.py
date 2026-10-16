#!/usr/bin/env python3
"""Run the added simulations: solve, collect, post-process and extract pool features.

Three cases at a time on six ranks each, so eighteen of the twenty-four cores are used and the
machine stays usable.

    python run_new_sims.py [case ...]
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
LOGS = BENCH / "logs" / "new"
NPROCS, JOBS = 6, 3
CASES = [c for c in (HERE / "results" / "new_cases.txt").read_text().split() if c]


def sh(cmd, log, cwd):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        return subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(cwd))


def simulate(case):
    t0 = time.time()
    log = LOGS / f"{case}.log"
    rc = sh(["python3", BENCH / "code" / "run_case.py", BENCH / "configs" / f"{case}.yaml",
             "--steps", "all", "--nprocs", str(NPROCS)], log, BENCH)
    if rc == 0:
        rc = sh(["python3", "extract_features.py", case], log, HERE)
    print(f"{case} rc={rc} {(time.time() - t0) / 60:.0f} min", flush=True)
    return rc


def main():
    cases = sys.argv[1:] or CASES
    print(f"== {len(cases)} cases, {JOBS} at a time on {NPROCS} ranks each", flush=True)
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        rcs = list(ex.map(simulate, cases))
    print("failures:", sum(r != 0 for r in rcs), flush=True)


if __name__ == "__main__":
    main()
