#!/usr/bin/env python3
"""Select one structure per comparison model over all ten records.

Each candidate structure is identified on every record; the structure with the highest median
complete-record R^2 over the ten records is kept, and its ten fits become the reported results
(results/baselines/PP_*_<model>.{json,npz,pt}). The candidates:

Rule: the highest median among structures with R^2 > 0 on every record.

  ARX          (na, nb) in (1,1) (2,2) (4,4) (8,8) (16,8) (32,16) (64,16)
  LSTM         hidden width 16, 32, 64
  Transformer  context window 16, 32, 64 frames x model width 32, 64

    python structure_sweep.py run        # all fits, ten at a time
    python structure_sweep.py select     # choose, copy the chosen fits to the reported names
"""
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "baselines"
SWEEP = OUT / "sweep"
PY = sys.executable
CASES = [c for c in (HERE / "results" / "paper_cases.txt").read_text().split() if c]
CANDIDATES = {
    "arx": [(f"_na{a}_nb{b}", ["--order", str(a), str(b)])
            for a, b in ((1, 1), (2, 2), (4, 4), (8, 8), (16, 8), (32, 16), (64, 16))],
    "lstm": [(f"_h{h}", ["--size", str(h)]) for h in (16, 32, 64)],
    "transformer": [(f"_w{w}_d{d}", ["--window", str(w), "--size", str(d)])
                    for w in (16, 32, 64) for d in (32, 64)],
}


def run_one(job):
    case, model, tag, args, train_frac = job
    if (SWEEP / f"{case}_{model}{tag}.json").exists():
        return 0
    t0 = time.time()
    extra = ["--train-frac", str(train_frac)] if train_frac < 1.0 else []
    with open(HERE / "logs" / "simple" / f"sweep_{case}_{model}{tag}.log", "w") as f:
        rc = subprocess.call([PY, "baselines.py", "fit", case, model, "--threads", "2", "--tag", tag,
                              *extra, *args], stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE))
    for ext in ("json", "npz", "pt"):
        src = OUT / f"{case}_{model}{tag}.{ext}"
        if src.exists():
            shutil.move(str(src), str(SWEEP / src.name))
    print(f"{case} {model}{tag} rc={rc} {time.time() - t0:.0f}s", flush=True)
    return rc


def run(train_frac=1.0, suffix="", workers=10, models=None):
    SWEEP.mkdir(parents=True, exist_ok=True)
    (HERE / "logs" / "simple").mkdir(parents=True, exist_ok=True)
    chosen = {m: c for m, c in CANDIDATES.items() if models is None or m in models}
    jobs = [(c, m, tag + suffix, args, train_frac)
            for m, cands in chosen.items() for tag, args in cands for c in CASES]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rcs = list(ex.map(run_one, jobs))
    print("failures:", sum(r != 0 for r in rcs))


BOUND = 10.0        # a held-out standardised RMSE above this counts as an unbounded prediction


def select(suffix="", key="r2_free", score_cases=None, models=None):
    """Keep one structure per model class.

    The score is the median of `key` over `score_cases` (all records by default). A structure whose
    free-running prediction leaves the bounded range on any record is not usable, whichever records
    are scored; among the usable ones the highest median wins, and ties on positivity are broken the
    same way as before.
    """
    old = OUT / "superseded_per_record"
    old.mkdir(exist_ok=True)
    table = {}
    for model, cands in CANDIDATES.items():
        if models is not None and model not in models:
            continue
        scores = {}
        scored = score_cases or CASES
        for tag0, _ in cands:
            tag = tag0 + suffix
            info = {c: json.loads((SWEEP / f"{c}_{model}{tag}.json").read_text()) for c in CASES}
            r2 = [info[c][key] for c in scored]
            spread = [info[c].get("nrmse_std_heldout", info[c].get("nrmse_std", 0.0)) for c in CASES]
            scores[tag] = dict(median=float(np.median(r2)), worst=float(np.min(r2)),
                               max_rmse=float(np.max(spread)), bounded=bool(np.max(spread) < BOUND),
                               per_record=r2)
        # A structure that leaves the bounded range on any record is not usable; among the usable
        # ones the highest median over the scored records wins.
        usable = [t for t in scores if scores[t]["bounded"]] or list(scores)
        positive = [t for t in usable if scores[t]["worst"] > 0.0]
        best = max(positive or usable, key=lambda t: scores[t]["median"])
        table[model] = dict(chosen=best, scored_records=scored, candidates=scores)
        for c in CASES:
            for ext in ("json", "npz", "pt"):
                cur = OUT / f"{c}_{model}{suffix}.{ext}"
                if cur.exists():
                    if (old / cur.name).exists():        # the per-record originals are already kept
                        cur.unlink()
                    else:
                        shutil.move(str(cur), str(old / cur.name))
                src = SWEEP / f"{c}_{model}{best}.{ext}"
                if src.exists():
                    shutil.copy(str(src), str(cur))
        print(f"{model}: chosen {best}  median {scores[best]['median']:.4f}  worst {scores[best]['worst']:.4f}")
        for tag, v in scores.items():
            flag = "" if v["bounded"] else "   unbounded"
            print(f"   {tag:16s} median {v['median']:12.4g}  worst {v['worst']:12.4g}"
                  f"  max RMSE {v['max_rmse']:10.4g}{flag}")
    (OUT / f"structure_selection{suffix}.json").write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "select"])
    ap.add_argument("--train-frac", type=float, default=1.0, dest="train_frac")
    ap.add_argument("--suffix", default="", help="appended to every candidate tag and to the reported names")
    ap.add_argument("--key", default="", help="selection metric; default r2_free, or r2_heldout when split")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--score-cases", nargs="*", default=None, dest="score_cases",
                    help="records the median is taken over; the default is every record")
    ap.add_argument("--models", nargs="*", default=None, choices=list(CANDIDATES),
                    help="restrict to these model classes; the default is all three")
    a = ap.parse_args()
    key = a.key or ("r2_heldout" if a.train_frac < 1.0 else "r2_free")
    if a.cmd == "run":
        run(a.train_frac, a.suffix, a.workers, a.models)
    else:
        select(a.suffix, key, a.score_cases, a.models)
