#!/usr/bin/env python3
"""sec6_experiments.part_3b in numpy (no retraining, forward passes only).

    python numpy_port/run_3b_numpy.py WAAM video_3     # from Coding Project/representation_ViT_state
    python numpy_port/run_3b_numpy.py GMAW exp_05

Same steps as part_3b: identified rollout of the two-state model, delta frozen at t_f, t_f taken
as the earliest of 20/30/40/50 % of the record whose median nearest-recorded-state distance over
t > t_f stays within 1.5x that of the identified run; decoding by the trained structure decoder
plus the residual layer retrieved from recorded frames.

The port is checked against gen_{process}_{record}.npz, written by the torch code for the same model.
results_sec6/3b_WAAM_video_3 and 3b_GMAW_exp_05 were produced this way (2026-10-16), because PyTorch
was not available where they ran; 3b_GTAW_Iden_02 comes from the torch code itself. With PyTorch,
`python -c "import sec6_experiments as X; X.part_3b('GMAW', 'exp_05')"` runs the original code on the
same model and should give the same t_f and arrays to within float rounding.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import npt  # noqa: E402

R = Path(__file__).resolve().parent.parent          # Coding Project/representation_ViT_state
OUT = R / "results_sec6"
OUT.mkdir(exist_ok=True)
CFG = {"WAAM": dict(smooth=25, ids=("video_1", "video_3", "video_5")),
       "GMAW": dict(smooth=24, ids=("exp_01", "exp_02", "exp_03", "exp_04", "exp_05", "exp_06",
                                    "exp_07", "exp_09", "exp_10")),
       "GTAW": dict(smooth=20, ids=("Iden_06", "Iden_01", "Iden_02", "Iden_07", "Iden_08", "Iden_09",
                                    "Iden_10", "Iden_11", "Iden_12", "Iden_13"))}
PROCESS, SID = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("WAAM", "video_3")
SMOOTH, IDS = CFG[PROCESS]["smooth"], CFG[PROCESS]["ids"]
SHARES = (0.20, 0.30, 0.40, 0.50)


def torch_std(x):
    return np.maximum(x.std(0, ddof=1), 1e-6)          # torch.std is unbiased; clamp(min=1e-6)


def nearest_distance(z, bz):
    mu, sd = bz.mean(0), torch_std(bz)
    a, b = (z - mu) / sd, (bz - mu) / sd
    d = np.sqrt(np.maximum((a ** 2).sum(1)[:, None] - 2 * a @ b.T + (b ** 2).sum(1)[None], 0))
    return d.min(1)


def retrieve(z_pred, z_bank, bounds, tol=2.5):
    """delta_study.retrieve, unchanged in logic."""
    mu, sd = z_bank.mean(0), torch_std(z_bank)
    a, b = (z_pred - mu) / sd, (z_bank - mu) / sd
    d = np.sqrt(np.maximum((a ** 2).sum(1)[:, None] - 2 * a @ b.T + (b ** 2).sum(1)[None], 0))
    ends = {e - 1 for e in bounds[1:]}
    picked, p = [], None
    for t in range(len(z_pred)):
        best = int(d[t].argmin())
        if p is not None and (p + 1) < len(z_bank) and p not in ends:
            nxt = p + 1
            p = nxt if d[t, nxt] <= tol * d[t, best] else best
        else:
            p = best
        picked.append(p)
    return np.array(picked)


def decode(raw, dec, vq=None, field=None, chunk=16):
    st, full = [], []
    for i in range(0, len(raw), chunk):
        base = dec(raw[i:i + chunk])
        st.append(base[:, 0])
        if field is not None:
            full.append((base + vq.decode_indices(field[i:i + chunk]))[:, 0])
    return np.concatenate(st), (np.concatenate(full) if field is not None else None)


def main():
    t0 = time.time()
    gen_dir = R / ("results_generator" if PROCESS == "GTAW" else f"results_generator_{PROCESS}")
    ck = npt.load_pt(R / f"results_node_aligned_{PROCESS}" / f"{SID}_A_m1_e1200.pt")
    model = npt.AlignedNODE(ck["model"])
    mean, std, dt = ck["mean"].astype(np.float32), ck["std"].astype(np.float32), float(ck["dt"])
    states = npt.load_pt(gen_dir / "states.pt")
    codes = npt.load_pt(gen_dir / "codes.pt")
    # the identification record as node_aligned.fit stored it: smoothed state, input and speed
    fit = np.load(R / f"results_node_aligned_{PROCESS}" / f"{SID}_A_m1_e1200.npz")
    z_obs = fit["z_true"].astype(np.float32)
    n = len(z_obs)
    u = fit["u"].astype(np.float32).reshape(n, -1)
    speed = fit["speed"][:n].astype(np.float32)
    zr = states[SID].astype(np.float32)[:n]
    assert np.allclose(z_obs, uniform_filter1d(zr, SMOOTH, axis=0, mode="nearest"), atol=1e-5)
    zn = ((z_obs - mean) / std).astype(np.float32)

    z_id, d_id = model.rollout(zn[0], u, dt)
    raw_id = (z_id * std + mean).astype(np.float32)

    ref = np.load(R / "results_sec6" / f"gen_{PROCESS}_{SID}.npz")
    chk = dict(max_abs_z=float(np.abs(raw_id[:len(ref["A_z"])] - ref["A_z"]).max()),
               max_abs_delta=float(np.abs(d_id[:len(ref["A_delta"])] - ref["A_delta"]).max()))
    print("check vs torch rollout:", chk, flush=True)

    bz = np.concatenate([uniform_filter1d(states[s].astype(np.float32), SMOOTH, axis=0,
                                          mode="nearest") for s in IDS]).astype(np.float32)
    bc = np.concatenate([codes[s] for s in IDS])
    bounds = np.cumsum([0] + [len(codes[s]) for s in IDS]).tolist()
    dist_id = nearest_distance(raw_id, bz)

    grid = []
    for s in SHARES:
        k = int(s * n)
        d_fr = d_id.copy(); d_fr[k:] = d_id[k]
        z_fr = model.rollout_forced(zn[0], u, d_fr, dt)
        raw_fr = (z_fr * std + mean).astype(np.float32)
        dist_fr = nearest_distance(raw_fr, bz)
        grid.append(dict(share=s, t_f=k * dt, k=k,
                         med_dist_id=float(np.median(dist_id[k:])),
                         med_dist_frozen=float(np.median(dist_fr[k:])),
                         ratio=float(np.median(dist_fr[k:]) / max(np.median(dist_id[k:]), 1e-9))))
        print(f"  share {s:.0%}: t_f {k*dt:.2f} s, ratio {grid[-1]['ratio']:.3f}", flush=True)
    ok = [g for g in grid if g["ratio"] <= 1.5]
    chosen = ok[0] if ok else min(grid, key=lambda g: g["ratio"])
    k = chosen["k"]
    d_fr = d_id.copy(); d_fr[k:] = d_id[k]
    z_fr = model.rollout_forced(zn[0], u, d_fr, dt)
    raw_fr = (z_fr * std + mean).astype(np.float32)

    a = npt.load_pt(gen_dir / "stage_a.pt")
    dec = npt.StructureDecoder(a["dec"], a["mean_frame"])
    vq = npt.ResidualVQDecoder(npt.load_pt(gen_dir / "stage_b.pt")["net"])

    pick_id, pick_fr = retrieve(raw_id, bz, bounds), retrieve(raw_fr, bz, bounds)
    st_id, full_id = decode(raw_id, dec, vq, bc[pick_id])
    print(f"identified decoded, {time.time()-t0:.0f}s", flush=True)
    st_fr, full_fr = decode(raw_fr, dec, vq, bc[pick_fr])
    print(f"frozen decoded, {time.time()-t0:.0f}s", flush=True)
    st_obs, _ = decode(z_obs, dec)
    m = min(len(ref["A_gen"]), n)
    chk.update(max_abs_struct=float(np.abs(st_id[:m] - ref["A_struct"][:m]).max()),
               max_abs_gen=float(np.abs(full_id[:m] - ref["A_gen"][:m]).max()),
               mean_abs_gen=float(np.abs(full_id[:m] - ref["A_gen"][:m]).mean()))
    print("check vs torch images:", chk, flush=True)
    real = ref["real"]
    n_img = min(len(real), n)

    E_id = np.abs(st_id - st_obs).mean(axis=(1, 2))
    E_fr = np.abs(st_fr - st_obs).mean(axis=(1, 2))
    E_pair = float(np.abs(st_id[k:] - st_fr[k:]).mean())
    res = dict(process=PROCESS, record=SID, grid=grid, chosen=chosen,
               E_identified=float(E_id[k:].mean()), E_frozen=float(E_fr[k:].mean()),
               E_between_runs=E_pair,
               note=("computed with a numpy port of the forward pass (PyTorch unavailable); "
                     f"checked against gen_{PROCESS}_{SID}.npz written by the torch code"),
               port_check=chk)
    (OUT / f"3b_{PROCESS}_{SID}.json").write_text(json.dumps(res, indent=1) + "\n")
    np.savez_compressed(OUT / f"3b_{PROCESS}_{SID}.npz",
                        t=np.arange(n) * dt, t_f=chosen["t_f"], k=k, speed=speed,
                        z_obs=z_obs, z_id=raw_id, z_frozen=raw_fr,
                        delta_id=d_id, delta_frozen=d_fr,
                        struct_id=st_id, struct_frozen=st_fr, struct_obs=st_obs,
                        gen_id=full_id[:n_img], gen_frozen=full_fr[:n_img], real=real[:n_img],
                        E_id=E_id, E_frozen=E_fr, dt=dt)
    print(f"3b {PROCESS} {SID}: t_f = {chosen['t_f']:.2f} s (share {chosen['share']:.0%}, "
          f"ratio {chosen['ratio']:.2f}), E_id {res['E_identified']:.4f} "
          f"E_frozen {res['E_frozen']:.4f} E_pair {E_pair:.4f}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
