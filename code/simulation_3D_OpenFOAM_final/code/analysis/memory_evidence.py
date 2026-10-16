#!/usr/bin/env python3
"""Is the observed surface pool a Markov state? (implementation plan §7)

Observed surface state z, the five coordinates of plan §6:
    [log(top melt area), top length, top width, peak top temperature,
     ROI-mean top temperature]   (liquidus 1800 K; ROI 10 x 10 mm on the torch)
standardised with the S0 heating samples. The inputs are current and speed.
The hidden heat-state candidates, withheld from the surface, are:
    delta_H (plate enthalpy), the full-thickness ring 5-15 mm from the torch,
    and the bottom-surface temperature under the torch.
The target is the subsequent surface motion dz = (z(t + tau) - z(t)) / tau, for
tau = 0.5, 1 and 2 s. Samples are 0.1 s apart and have: the arc on for >= 1 s,
a pool present, and the same current and speed over [t, t + tau].

Test 1 (matched states; the plan's wording): pairs with the same input and
|z_a - z_b| <= EPS_Z (standardised), from different cases or >= 2 s apart
within a case. For each pair: D = |dz_a - dz_b|, and the hidden-state
differences. The script reports the Spearman correlation and the median D in
the lowest and highest tercile of each hidden-state difference, with a
one-sided Mann-Whitney test.

Test 2 (predictive information): leave-one-case-out k-nearest-neighbour
regression of dz from features
    A = [z, inputs]                          (surface-only, the Markov hypothesis)
    B = [z, inputs, hidden]                  (with a hidden heat state)
    C = [z, inputs, hidden shuffled]         (control with the same dimension)
It reports the pooled mean-squared prediction error of the held-out cases. A
Markov surface state predicts B ~ A ~ C; a non-Markov one gives B < A and
B < C.

No learned dynamic model is involved and no case is selected or excluded.

    python code/analysis/memory_evidence.py [C3r|C3]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mp3d import PROJECT_DIR, RESULTS_DIR  # noqa: E402

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "C3r"          # dataset variant: C3r (calibrated source) or C3
CASES = [f"{PREFIX}_{c}" for c in ("S0", "P0", "H0", "Iminus", "Iplus", "Vminus", "Vplus")]
ZCOLS = ["log_area", "top_length_liq_mm", "top_width_liq_mm", "T_top_max_K", "T_top_roi_mean_K"]
HIDDEN = {"delta_H_K": "plate enthalpy delta_H", "dHK_ring_5_15mm_K": "ring 5-15 mm",
          "T_bottom_under_torch_K": "bottom T under torch"}
TAUS = [0.5, 1.0, 2.0]
EPS_Z = 0.15
K_NN = 10
SEED = 20260913
OUT = PROJECT_DIR / "analysis" / PREFIX


def load(case):
    s = pd.read_csv(RESULTS_DIR / case / "signals_0p1s.csv")
    s["log_area"] = np.log(np.maximum(s["top_area_liq_mm2"], 1e-3))
    on = (s["I_A"] > 0).to_numpy()
    I, U = s["I_A"].to_numpy(), s["u_mm_s"].to_numpy()
    since = np.zeros(len(s))
    ahead = np.zeros(len(s))                      # time the same input continues
    for i in range(1, len(s)):
        since[i] = since[i - 1] + 0.1 if on[i] and on[i - 1] else 0.0
    for i in range(len(s) - 2, -1, -1):
        ahead[i] = ahead[i + 1] + 0.1 if (I[i + 1] == I[i] and U[i + 1] == U[i]) else 0.0
    s["t_since_on"], s["t_same_ahead"] = since, ahead
    return s


def samples_for(data, mu, sd, hsd, tau):
    k = int(round(tau / 0.1))
    rows = []
    for c, s in data.items():
        Z = ((s[ZCOLS] - mu) / sd).to_numpy()
        ok = ((s["I_A"] > 0) & (s["top_area_liq_mm2"] > 1.0) & (s["t_since_on"] >= 1.0)
              & (s["t_same_ahead"] >= tau - 1e-9)).to_numpy()
        for i in np.flatnonzero(ok[:len(s) - k]):
            rows.append(dict(case=c, t=float(s["t_s"].iloc[i]), I=float(s["I_A"].iloc[i]),
                             u=float(s["u_mm_s"].iloc[i]), z=Z[i], dz=(Z[i + k] - Z[i]) / tau,
                             **{h: float(s[h].iloc[i]) / hsd[h] for h in HIDDEN}))
    return rows


def matched_pairs(S, tau):
    Z = np.array([r["z"] for r in S])
    tree = cKDTree(Z)
    out = []
    for a, b in tree.query_pairs(EPS_Z):
        A, B = S[a], S[b]
        if A["I"] != B["I"] or A["u"] != B["u"]:
            continue
        if A["case"] == B["case"] and abs(A["t"] - B["t"]) < 2.0:
            continue
        out.append(dict(case_a=A["case"], t_a=A["t"], case_b=B["case"], t_b=B["t"],
                        dist_z=float(np.linalg.norm(A["z"] - B["z"])),
                        D=float(np.linalg.norm(A["dz"] - B["dz"])),
                        **{f"abs_d_{h}": abs(A[h] - B[h]) for h in HIDDEN}))
    return pd.DataFrame(out)


def knn_mse(S, feats, rng):
    """Leave-one-case-out k-NN regression MSE of dz for the given feature builder."""
    cases = sorted({r["case"] for r in S})
    X = feats(S, rng)
    Y = np.array([r["dz"] for r in S])
    grp = np.array([r["case"] for r in S])
    se, n = 0.0, 0
    for c in cases:
        tr, te = grp != c, grp == c
        if te.sum() == 0 or tr.sum() < K_NN:
            continue
        tree = cKDTree(X[tr])
        d, idx = tree.query(X[te], k=K_NN)
        w = 1.0 / np.maximum(d, 1e-6)
        pred = np.einsum("ij,ijk->ik", w, Y[tr][idx]) / w.sum(axis=1, keepdims=True)
        se += float(((pred - Y[te]) ** 2).sum())
        n += te.sum() * Y.shape[1]
    return se / n


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    data = {c: load(c) for c in CASES if (RESULTS_DIR / c / "signals_0p1s.csv").exists()}
    s0 = data[f"{PREFIX}_S0"]
    ref = s0[(s0["I_A"] > 0) & (s0["top_area_liq_mm2"] > 1.0)]
    mu, sd = ref[ZCOLS].mean(), ref[ZCOLS].std()
    hsd = {h: float(ref[h].std()) for h in HIDDEN}
    summary = {"eps_z": EPS_Z, "k_nn": K_NN, "cases": list(data), "standardisation_mean": mu.to_dict(),
               "standardisation_std": sd.to_dict(), "hidden_std_S0": hsd, "per_tau": {}}
    knn_rows = []
    pair_store = {}
    for tau in TAUS:
        S = samples_for(data, mu, sd, hsd, tau)
        P = matched_pairs(S, tau)
        P.to_csv(OUT / f"matched_pairs_tau{tau:g}.csv", index=False, float_format="%.5g")
        pair_store[tau] = P
        res = {"n_samples": len(S), "n_pairs": int(len(P)),
               "pairs_cross_case": int((P["case_a"] != P["case_b"]).sum()) if len(P) else 0}
        for h in HIDDEN:
            col = f"abs_d_{h}"
            if len(P) > 10:
                rho, p = spearmanr(P[col], P["D"])
                lo, hi = P[col].quantile([1 / 3, 2 / 3])
                a, b = P.loc[P[col] <= lo, "D"], P.loc[P[col] >= hi, "D"]
                mw = mannwhitneyu(b, a, alternative="greater")
                res[h] = {"spearman": float(rho), "p": float(p), "median_D_low_tercile": float(a.median()),
                          "median_D_high_tercile": float(b.median()), "mannwhitney_p_high_gt_low": float(mw.pvalue)}

        def surf(Sx, _r):
            return np.array([np.concatenate([r["z"], [r["I"] / 200.0, r["u"] / 3.5]]) for r in Sx])

        base = knn_mse(S, surf, rng)
        knn_rows.append(dict(tau_s=tau, features="A: surface + input", mse=base, rel_change_vs_A=0.0))
        res["knn_mse_A"] = base
        for h in list(HIDDEN) + ["all"]:
            hs = list(HIDDEN) if h == "all" else [h]

            def with_h(Sx, _r, hs=hs):
                return np.column_stack([surf(Sx, _r), np.array([[r[q] for q in hs] for r in Sx])])

            def with_shuf(Sx, r_, hs=hs):
                H = np.array([[r[q] for q in hs] for r in Sx])
                return np.column_stack([surf(Sx, r_), H[r_.permutation(len(H))]])

            mB, mC = knn_mse(S, with_h, rng), knn_mse(S, with_shuf, rng)
            label = "all three" if h == "all" else HIDDEN[h]
            knn_rows.append(dict(tau_s=tau, features=f"B: + {label}", mse=mB, rel_change_vs_A=mB / base - 1))
            knn_rows.append(dict(tau_s=tau, features=f"C: + {label} (shuffled)", mse=mC, rel_change_vs_A=mC / base - 1))
            res[f"knn_mse_B_{h}"], res[f"knn_mse_C_{h}"] = mB, mC
        summary["per_tau"][f"{tau:g}"] = res
        print(f"tau {tau:g} s: {len(S)} samples, {len(P)} pairs ({res['pairs_cross_case']} cross-case); "
              + "; ".join(f"{HIDDEN[h]} rho {res[h]['spearman']:.2f}" for h in HIDDEN if h in res))
    K = pd.DataFrame(knn_rows)
    K.to_csv(OUT / "knn_predictability.csv", index=False, float_format="%.5g")
    (OUT / "memory_evidence_summary.json").write_text(json.dumps(summary, indent=1))
    print(K.to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for i, tau in enumerate(TAUS):
        P = pair_store[tau]
        if len(P):
            col = "abs_d_dHK_ring_5_15mm_K"
            bins = np.quantile(P[col], np.linspace(0, 1, 6))
            mids = 0.5 * (bins[1:] + bins[:-1])
            med = [P.loc[(P[col] >= a) & (P[col] <= b), "D"].median() for a, b in zip(bins[:-1], bins[1:])]
            ax[0].plot(mids * hsd["dHK_ring_5_15mm_K"], med, "o-", label=f"tau {tau:g} s ({len(P)} pairs)")
    ax[0].set(xlabel="|ring-signal difference| between matched states (K)",
              ylabel="median divergence of surface motion (std/s)",
              title=f"matched surface states (|dz| <= {EPS_Z} std, same input)")
    ax[0].legend(fontsize=7)
    for i, tau in enumerate(TAUS):
        sub = K[(K.tau_s == tau) & K.features.str.startswith("B")]
        ctl = K[(K.tau_s == tau) & K.features.str.startswith("C")]
        x = np.arange(len(sub)) + 0.25 * i
        ax[1].bar(x, 100 * sub["rel_change_vs_A"], 0.12, color=f"C{i}", label=f"with hidden, tau {tau:g} s")
        ax[1].bar(x + 0.12, 100 * ctl["rel_change_vs_A"], 0.12, color=f"C{i}", alpha=0.35,
                  label=f"shuffled control, tau {tau:g} s")
    ax[1].set_xticks(np.arange(len(HIDDEN) + 1) + 0.3, [HIDDEN[h] for h in HIDDEN] + ["all three"], fontsize=7)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(ylabel="change in held-out MSE vs surface-only (%)",
              title="leave-one-case-out k-NN prediction of surface motion")
    ax[1].legend(fontsize=6)
    P = pair_store[TAUS[-1]]
    cross = P[(P.case_a != P.case_b)] if len(P) else P
    if len(cross):
        best = cross.sort_values("abs_d_dHK_ring_5_15mm_K", ascending=False).head(20).sort_values("dist_z").iloc[0]
        for (case, t0), ls in (((best.case_a, best.t_a), "-"), ((best.case_b, best.t_b), "--")):
            s = data[case]
            w = (s["t_s"] >= t0 - 1.0) & (s["t_s"] <= t0 + 3.0)
            zz = (s.loc[w, ZCOLS] - mu) / sd
            for n, col in enumerate(ZCOLS):
                ax[2].plot(s.loc[w, "t_s"] - t0, zz[col], ls, color=f"C{n}", lw=1, label=col if ls == "-" else None)
        ax[2].axvline(0, color="0.6", lw=0.6)
        ax[2].set(xlabel="time from matched instant (s)", ylabel="standardised surface coordinate",
                  title=f"{best.case_a} t={best.t_a:g} s (solid) vs {best.case_b} t={best.t_b:g} s (dashed)\n"
                        f"|dz| = {best.dist_z:.2f} std; ring difference "
                        f"{best['abs_d_dHK_ring_5_15mm_K'] * hsd['dHK_ring_5_15mm_K']:.0f} K")
        ax[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "memory_evidence.png", dpi=150)


if __name__ == "__main__":
    main()
