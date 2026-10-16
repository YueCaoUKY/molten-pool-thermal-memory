#!/usr/bin/env python3
"""Within-record held-out protocol for Section 5: identify on the first 60 % of every record and
score the remaining 40 % without any state correction at the boundary.

  stage node       the proposed two-state NODE (pool memory state, tag _v_pool_simpleD_p60) and its
                   no-memory ablation (tag _nodelta_p60) on all ten records
  stage baselines  the ARX, LSTM and Transformer structure sweep under the same protocol
                   (tags ..._p60), then the selection by held-out R^2
  stage step       the A -> B -> A long-horizon test driven by the 60 % source models

    python run_heldout.py node baselines step
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
LOGS = HERE / "logs" / "heldout"
FRAC = "0.6"          # overridden by --frac
TAG = "_p60"          # derived from FRAC
CASES = [c for c in (HERE / "results" / "paper_cases.txt").read_text().split() if c]

sys.path.insert(0, str(HERE))
from delta_variants import window_for  # noqa: E402


def sh(name, cmd):
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(LOGS / f"{name}.log", "w") as f:
        f.write(f"$ {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        rc = subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE))
    print(f"{name} rc={rc} {time.time() - t0:.0f}s", flush=True)
    return rc


def node_jobs():
    jobs = []
    for case in CASES:
        t0, t1 = window_for(case)
        jobs.append((f"{case}_pool", [PY, "delta_variants.py", "fit", case, "pool", "--loss", "mse",
                                     "--bounds", "min_only", "--suffix", f"_simpleD{TAG}",
                                     "--train-frac", FRAC, "--threads", "2"]))
        jobs.append((f"{case}_nodelta", [PY, "identify_3d.py", case, "--no-delta", "--tag", f"_nodelta{TAG}",
                                         "--train-frac", FRAC, "--window", str(t0), str(t1),
                                         "--threads", "2"]))
    # the long records first, so the tail of the queue is short
    return sorted(jobs, key=lambda j: -(window_for(j[0].rsplit("_", 1)[0])[1]))


def main():
    global FRAC, TAG
    argv = list(sys.argv[1:])
    if "--frac" in argv:
        i = argv.index("--frac")
        FRAC = argv[i + 1]
        del argv[i:i + 2]
    TAG = f"_p{round(float(FRAC) * 100):d}"
    models = None
    if "--models" in argv:
        i = argv.index("--models")
        models = argv[i + 1].split(",")
        del argv[i:i + 2]
    stages = argv or ["node", "baselines", "step"]
    print(f"== train fraction {FRAC}, tag {TAG}", flush=True)
    if "node" in stages:
        jobs = node_jobs()
        if models:
            jobs = [j for j in jobs if j[0].rsplit("_", 1)[1] in models]
        print(f"== NODE: {len(jobs)} fits, eight at a time", flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda j: sh(*j), jobs))
    if "baselines" in stages:
        print("== baselines: structure sweep under the held-out protocol", flush=True)
        vary = [c for c in CASES if "normal_const" not in c]
        extra = ["--models", *models] if models else []
        sh(f"sweep_run{TAG}", [PY, "structure_sweep.py", "run", "--train-frac", FRAC, "--suffix", TAG,
                               "--workers", "10", *extra])
        sh(f"sweep_select{TAG}", [PY, "structure_sweep.py", "select", "--suffix", TAG,
                                  "--key", "r2_heldout", "--score-cases", *vary, *extra])
    if "step" in stages:
        print("== A -> B -> A long-horizon test with the 60 % source models", flush=True)
        sh(f"step_test{TAG}", [PY, "step_test.py", "--tag", TAG, "--out", f"step_test{TAG}"])


if __name__ == "__main__":
    main()
