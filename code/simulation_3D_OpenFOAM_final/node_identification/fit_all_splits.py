#!/usr/bin/env python3
"""Identify every record with every model class at two identification shares.

Splits: 60 % and 50 % of each record. Models: the proposed two-state NODE, the same NODE without
the memory state, and ARX, LSTM and Transformer at the structures selected earlier
(n_a = n_b = 2; hidden width 64; window 16 and width 64). Fits that already exist are skipped.

    python fit_all_splits.py
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os
# interpreter for the worker processes; override with MP_PYTHON
PY = os.environ.get("MP_PYTHON", sys.executable)
LOGS = HERE / "logs" / "splits"
JOBS, THREADS = 6, 2
SPLITS = {"_p60": "0.6", "_p50": "0.5"}
BASE = {"arx": ["--order", "2", "2"], "lstm": ["--size", "64"],
        "transformer": ["--window", "16", "--size", "64"]}

sys.path.insert(0, str(HERE))
from delta_variants import window_for  # noqa: E402

CASES = [c for c in (HERE / "results" / "paper_cases.txt").read_text().split() if c] + \
        [c for c in (HERE / "results" / "new_cases.txt").read_text().split() if c]


def sh(name, cmd):
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(LOGS / f"{name}.log", "w") as f:
        rc = subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE))
    print(f"{name} rc={rc} {time.time() - t0:.0f}s", flush=True)
    return rc


def jobs():
    out = []
    for tag, frac in SPLITS.items():
        for case in CASES:
            t0, t1 = window_for(case)
            if not (HERE / "results" / f"{case}_v_pool_simpleD{tag}" / "metrics.json").exists():
                out.append((f"{case}{tag}_pool",
                            [PY, "delta_variants.py", "fit", case, "pool", "--loss", "mse",
                             "--bounds", "min_only", "--suffix", f"_simpleD{tag}",
                             "--train-frac", frac, "--threads", str(THREADS)]))
            if not (HERE / "results" / f"{case}_nodelta{tag}" / "metrics.json").exists():
                out.append((f"{case}{tag}_nodelta",
                            [PY, "identify_3d.py", case, "--no-delta", "--tag", f"_nodelta{tag}",
                             "--train-frac", frac, "--window", str(t0), str(t1),
                             "--threads", str(THREADS)]))
            for model, args in BASE.items():
                if not (HERE / "results" / "baselines" / f"{case}_{model}{tag}.json").exists():
                    out.append((f"{case}{tag}_{model}",
                                [PY, "baselines.py", "fit", case, model, "--tag", tag,
                                 "--train-frac", frac, "--threads", str(THREADS), *args]))
    # the long records first, so the tail of the queue is short
    return sorted(out, key=lambda j: -window_for(j[0].split("_p")[0])[1])


def main():
    q = jobs()
    print(f"== {len(q)} fits still missing, {JOBS} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        rcs = list(ex.map(lambda j: sh(*j), q))
    print("failures:", sum(r != 0 for r in rcs), flush=True)


if __name__ == "__main__":
    main()
