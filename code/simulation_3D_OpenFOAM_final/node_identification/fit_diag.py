#!/usr/bin/env python3
"""Re-identify every record with the diagonal pool-rate matrix, at both identification shares.

Only the NODE and its ablation change; the comparison models are unaffected.

    python fit_diag.py
"""
import subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os
# interpreter for the worker processes; override with MP_PYTHON
PY = os.environ.get("MP_PYTHON", sys.executable)
LOGS = HERE / "logs" / "diag"
JOBS, THREADS = 16, 1
V2 = {"PP_steel_speed_free": "PP_steel_speed_free_v2",
      "PP_steel_speed_cooled": "PP_steel_speed_cooled_v2"}
CASES = [V2.get(c, c) for c in
         (f"PP_{m}_{d}" for m in ("steel", "ss304")
          for d in ("4_slow_cycles", "5_slow_cooled", "speed_free", "speed_cooled",
                    "both_free", "both_cooled"))]
sys.path.insert(0, str(HERE))
from delta_variants import window_for  # noqa: E402


def sh(name, cmd):
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(LOGS / f"{name}.log", "w") as f:
        rc = subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE))
    print(f"{name} rc={rc} {time.time() - t0:.0f}s", flush=True)
    return rc


def main():
    jobs = []
    for tag, frac in (("_p60", "0.6"), ("_p50", "0.5")):
        for c in CASES:
            t0, t1 = window_for(c)
            jobs.append((f"{c}{tag}_pool",
                         [PY, "delta_variants.py", "fit", c, "pool", "--loss", "mse",
                          "--bounds", "min_only", "--suffix", f"_simpleD{tag}",
                          "--train-frac", frac, "--threads", str(THREADS)]))
            jobs.append((f"{c}{tag}_nodelta",
                         [PY, "identify_3d.py", c, "--no-delta", "--tag", f"_nodelta{tag}",
                          "--train-frac", frac, "--window", str(t0), str(t1),
                          "--threads", str(THREADS)]))
    jobs.sort(key=lambda j: -window_for(j[0].split("_p")[0])[1])
    print(f"== {len(jobs)} fits, {JOBS} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        rcs = list(ex.map(lambda j: sh(*j), jobs))
    print("failures:", sum(r != 0 for r in rcs), flush=True)


if __name__ == "__main__":
    main()
