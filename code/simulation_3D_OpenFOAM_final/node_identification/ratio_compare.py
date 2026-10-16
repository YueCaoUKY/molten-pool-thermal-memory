#!/usr/bin/env python3
"""Compare the proposed NODE with the selected ARX across train/test ratios.

For every ratio the held-out accuracy is recomputed from the saved fits, so all ratios use the
same definitions. Two measures are reported:
  R2    on the held-out interval, against that interval's own mean (the usual definition)
  RMSE  on the held-out interval, in units of the WHOLE-record standard deviation of each
        coordinate. This normalization does not depend on the split, so it is comparable
        across ratios; R2 is not, because the evaluated interval changes.

    python ratio_compare.py [0.6 0.5 0.4 0.3]
"""
import json
import sys
from pathlib import Path

import numpy as np

import baselines as B
import identify_3d as I
from delta_variants import window_for

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CASES = [c for c in (OUT / "paper_cases.txt").read_text().split() if c]
VARY = [c for c in CASES if "normal_const" not in c]


def scores(z_true, z_pred, held, sigma_full):
    """Both measures use coordinates scaled by the WHOLE-record standard deviation, so the seven
    coordinates contribute comparably and neither measure depends on where the split falls."""
    t = z_true[held] / sigma_full
    p = z_pred[held] / sigma_full
    sse = np.sum((p - t) ** 2, axis=0)
    sst = np.sum((t - t.mean(axis=0)) ** 2, axis=0)
    return (float(1.0 - sse.sum() / max(sst.sum(), 1e-12)),
            float(np.sqrt(np.mean((p - t) ** 2))))


def node_scores(case, tag, sigma_full, n):
    d = np.load(OUT / f"{case}{tag}" / "fit.npz", allow_pickle=True)
    held = np.zeros(len(d["t"]), bool)
    held[n:] = True
    if "fit" in d.files:
        held &= d["fit"].astype(bool)
    return scores(d["z_true"], d["z_pred"], held, sigma_full)


def split_index(case, tag):
    d = np.load(OUT / f"{case}_v_pool_simpleD{tag}" / "fit.npz", allow_pickle=True)
    return int(d["n_train"])


def arx_scores(case, tag, sigma_full, n):
    a = np.load(OUT / "baselines" / f"{case}_arx{tag}.npz")
    s, z_raw, u_np, *_ = I.load_record(case, I.STATE_NAMES, window_for(case), False)
    zn = ((z_raw - a["z_mean"]) / a["z_std"]).astype(np.float64)
    pred, _ = B.arx_predict(a["theta"], zn, u_np, int(a["na"]), int(a["nb"]), True)
    z_pred = pred * a["z_std"] + a["z_mean"]
    held = np.zeros(len(z_raw), bool)
    held[n:] = True
    return scores(z_raw, z_pred, held, sigma_full) + (f"({int(a['na'])}, {int(a['nb'])})",)


def main():
    fracs = [float(x) for x in sys.argv[1:]] or [0.6, 0.5, 0.4, 0.3]
    table = {}
    for frac in fracs:
        tag = f"_p{round(frac * 100):d}"
        rows = {}
        for case in CASES:
            _, z_raw, *_ = I.load_record(case, I.STATE_NAMES, window_for(case), False)
            sigma = np.maximum(z_raw.std(axis=0), 1e-9)
            try:
                n = split_index(case, tag)
                node = node_scores(case, f"_v_pool_simpleD{tag}", sigma, n)
                nod = node_scores(case, f"_nodelta{tag}", sigma, n)
                arx = arx_scores(case, tag, sigma, n)
            except FileNotFoundError as e:
                print(f"  missing for {case}{tag}: {Path(str(e).split(': ')[-1]).name}")
                continue
            rows[case] = dict(node_r2=node[0], node_rmse=node[1], nodelta_r2=nod[0],
                              nodelta_rmse=nod[1], arx_r2=arx[0], arx_rmse=arx[1], arx_order=arx[2])
        table[tag] = rows
        if not rows:
            continue
        vary = [c for c in VARY if c in rows]
        med = lambda k: float(np.median([rows[c][k] for c in vary]))
        wins = sum(rows[c]["node_r2"] > rows[c]["arx_r2"] for c in vary)
        print(f"\n== train {frac:.0%} / test {1 - frac:.0%}   ARX order "
              f"{rows[vary[0]]['arx_order']}   (medians over the {len(vary)} varying-input records)")
        print(f"   held-out R2    NODE {med('node_r2'):7.3f}   ARX {med('arx_r2'):7.3f}   "
              f"no-delta {med('nodelta_r2'):7.3f}")
        print(f"   held-out RMSE  NODE {med('node_rmse'):7.3f}   ARX {med('arx_rmse'):7.3f}   "
              f"no-delta {med('nodelta_rmse'):7.3f}   (whole-record sigma)")
        print(f"   NODE beats ARX on {wins} of {len(vary)} records")
        for c in vary:
            r = rows[c]
            mark = "  NODE" if r["node_r2"] > r["arx_r2"] else "   ARX"
            print(f"     {c.replace('PP_',''):24s} NODE {r['node_r2']:8.3f} / {r['node_rmse']:5.3f}"
                  f"   ARX {r['arx_r2']:8.3f} / {r['arx_rmse']:5.3f} {mark}")
    (OUT / "ratio_comparison.json").write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main()
