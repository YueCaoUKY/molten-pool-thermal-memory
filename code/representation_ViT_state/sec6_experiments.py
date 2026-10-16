#!/usr/bin/env python3
"""Sections 3a-3d of the Section 6 task, on the models identified by node_aligned.py.

Nothing in the representation is retrained. The structure decoder D, the residual codebook and
the retrieval of delta_study are used as they stand.

  3a  generation from the recorded input alone
  3b  the thermal state held fixed from t_f onwards
  3c  variants A and B on the WAAM records
  3d  a designed speed programme that returns to its first level

Each part writes one npz into results_sec6/ for the figure scripts to read.

    python sec6_experiments.py 3a 3b 3c 3d
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import node_aligned as A            # noqa: E402  the aligned model and its data layer
import generator_figures as G      # noqa: E402  the trained representation
import visual_generator as V       # noqa: E402
from delta_study import retrieve   # noqa: E402

OUT = HERE / "results_sec6"
OUT.mkdir(exist_ok=True)
N = A.N


# --------------------------------------------------------------------------- loading
def load_model(process: str, sid: str, variant: str = "A_m1", epochs: int = 1200):
    ck = torch.load(HERE / f"results_node_aligned_{process}" / f"{sid}_{variant}_e{epochs}.pt",
                    weights_only=False)
    model = A.AlignedNODE(6, 1, delta_dim=1, use_delta=(variant != "B_no_delta"))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["mean"], ck["std"], ck["dt"]


def record(process: str, sid: str):
    N.configure(process)
    N.SMOOTH = A.SMOOTH[process]
    return N.records()[sid], N.DT


@torch.no_grad()
def rollout_forced(model, z0, u, delta, dt):
    """The pool equation driven by a thermal trajectory that is supplied rather than integrated."""
    z, out = z0, []
    for k in range(u.shape[0]):
        out.append(z)
        if k == u.shape[0] - 1:
            break
        z_eq, gamma, _ = model._terms(z, delta[k], u[k])
        z = z_eq + (z - z_eq) * torch.exp(-gamma * dt)
    return torch.stack(out)


# --------------------------------------------------------------------------- representation
def bank(process: str, smooth: int):
    """Every recorded state of the process and its code field, as delta_study builds them."""
    V.configure(process)
    codes = torch.load(V.OUT / "codes.pt", weights_only=False)
    zs = torch.load(V.OUT / "states.pt", weights_only=False)
    bz, bc = [], []
    for sid in V.IDS:
        zf = uniform_filter1d(zs[sid].numpy(), smooth, axis=0, mode="nearest")
        bz.append(torch.tensor(zf.astype(np.float32)))
        bc.append(codes[sid])
    bounds = np.cumsum([0] + [len(c) for c in bc]).tolist()
    return torch.cat(bz), torch.cat(bc), bounds


@torch.no_grad()
def decode(raw_states: torch.Tensor, dec, vq=None, field=None, chunk=64):
    """Structure images D(z), and the full frame when a code field is supplied."""
    st, full = [], []
    for i in range(0, len(raw_states), chunk):
        base = dec(raw_states[i:i + chunk].to(V.DEVICE))
        st.append(base.cpu().numpy()[:, 0])
        if field is not None:
            full.append((base + vq.decode_indices(field[i:i + chunk])).cpu().numpy()[:, 0])
    return np.concatenate(st), (np.concatenate(full) if field is not None else None)


def nearest_distance(z_raw: torch.Tensor, bank_z: torch.Tensor) -> np.ndarray:
    """Distance of each state to the closest recorded state, in the bank's own standardisation."""
    mu, sd = bank_z.mean(0), bank_z.std(0).clamp(min=1e-6)
    d = torch.cdist((z_raw - mu) / sd, (bank_z - mu) / sd)
    return d.min(dim=1).values.numpy()


# --------------------------------------------------------------------------- 3a
def part_3a(process="GMAW", sid="exp_05"):
    t0 = time.time()
    model, mean, std, dt = load_model(process, sid)
    rec, _ = record(process, sid)
    smooth = A.SMOOTH[process]
    zn = torch.tensor(((rec["z"] - mean) / std).astype(np.float32))
    u = torch.tensor(rec["u"].astype(np.float32))
    with torch.no_grad():
        z_hat, d_hat = model.rollout(zn[0], u, dt)
    raw_pred = torch.tensor((z_hat.numpy() * std + mean).astype(np.float32))
    raw_obs = torch.tensor(rec["z"].astype(np.float32))

    bz, bc, bounds = bank(process, smooth)
    dec, vq, *_ = G.models()
    field = bc[retrieve(raw_pred, bz, bc, bounds)].to(V.DEVICE)
    struct, full = decode(raw_pred, dec, vq, field)
    struct_obs, _ = decode(raw_obs, dec)
    frames, _ = V.load_cache()
    real = frames[sid].float().numpy()[:, 0] / 255.0

    n = min(len(real), len(full))
    np.savez_compressed(OUT / f"3a_{process}_{sid}.npz",
                        t=np.arange(n) * dt, speed=rec["speed"][:n], u=rec["u"][:n],
                        z_obs=rec["z"][:n], z_pred=raw_pred.numpy()[:n], delta=d_hat.numpy()[:n],
                        gen=full[:n], struct=struct[:n], struct_obs=struct_obs[:n], real=real[:n],
                        dt=dt)
    print(f"3a {process} {sid}: {n} frames, {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- 3b
def part_3b(process="GTAW", sid="Iden_02", shares=(0.20, 0.30, 0.40, 0.50)):
    t0 = time.time()
    model, mean, std, dt = load_model(process, sid)
    rec, _ = record(process, sid)
    smooth = A.SMOOTH[process]
    zn = torch.tensor(((rec["z"] - mean) / std).astype(np.float32))
    u = torch.tensor(rec["u"].astype(np.float32))
    n = len(zn)
    with torch.no_grad():
        z_id, d_id = model.rollout(zn[0], u, dt)
    bz, bc, bounds = bank(process, smooth)
    dec, vq, *_ = G.models()

    raw_id = torch.tensor((z_id.numpy() * std + mean).astype(np.float32))
    dist_id = nearest_distance(raw_id, bz)

    grid = []
    for s in shares:
        k = int(s * n)
        d_fr = d_id.clone()
        d_fr[k:] = d_id[k]
        z_fr = rollout_forced(model, zn[0], u, d_fr, dt)
        raw_fr = torch.tensor((z_fr.numpy() * std + mean).astype(np.float32))
        dist_fr = nearest_distance(raw_fr, bz)
        grid.append(dict(share=s, t_f=k * dt, k=k,
                         med_dist_id=float(np.median(dist_id[k:])),
                         med_dist_frozen=float(np.median(dist_fr[k:])),
                         ratio=float(np.median(dist_fr[k:]) / max(np.median(dist_id[k:]), 1e-9))))
    ok = [g for g in grid if g["ratio"] <= 1.5]
    chosen = ok[0] if ok else min(grid, key=lambda g: g["ratio"])
    k = chosen["k"]

    d_fr = d_id.clone()
    d_fr[k:] = d_id[k]
    z_fr = rollout_forced(model, zn[0], u, d_fr, dt)
    raw_fr = torch.tensor((z_fr.numpy() * std + mean).astype(np.float32))
    raw_obs = torch.tensor(rec["z"].astype(np.float32))

    st_id, full_id = decode(raw_id, dec, vq, bc[retrieve(raw_id, bz, bc, bounds)].to(V.DEVICE))
    st_fr, full_fr = decode(raw_fr, dec, vq, bc[retrieve(raw_fr, bz, bc, bounds)].to(V.DEVICE))
    st_obs, _ = decode(raw_obs, dec)
    frames, _ = V.load_cache()
    real = frames[sid].float().numpy()[:, 0] / 255.0

    # E_run(t): mean absolute difference of the structure images against the recorded state
    E_id = np.abs(st_id - st_obs).mean(axis=(1, 2))
    E_fr = np.abs(st_fr - st_obs).mean(axis=(1, 2))
    E_pair = float(np.abs(st_id[k:] - st_fr[k:]).mean())

    res = dict(process=process, record=sid, grid=grid, chosen=chosen,
               E_identified=float(E_id[k:].mean()), E_frozen=float(E_fr[k:].mean()),
               E_between_runs=E_pair)
    (OUT / f"3b_{process}_{sid}.json").write_text(json.dumps(res, indent=1) + "\n")
    np.savez_compressed(OUT / f"3b_{process}_{sid}.npz",
                        t=np.arange(n) * dt, t_f=chosen["t_f"], k=k, speed=rec["speed"][:n],
                        z_obs=rec["z"], z_id=raw_id.numpy(), z_frozen=raw_fr.numpy(),
                        delta_id=d_id.numpy(), delta_frozen=d_fr.numpy(),
                        struct_id=st_id, struct_frozen=st_fr, struct_obs=st_obs,
                        gen_id=full_id, gen_frozen=full_fr, real=real[:n],
                        E_id=E_id, E_frozen=E_fr, dt=dt)
    print(f"3b {process} {sid}: t_f = {chosen['t_f']:.1f} s (share {chosen['share']:.0%}, "
          f"ratio {chosen['ratio']:.2f}), E_id {res['E_identified']:.4f} "
          f"E_frozen {res['E_frozen']:.4f} E_pair {E_pair:.4f}, {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- 3c
def part_3c(process="WAAM"):
    t0 = time.time()
    out = {}
    for sid in ("video_1", "video_3", "video_5"):
        rec, dt = record(process, sid)
        for v in ("A_m1", "B_no_delta"):
            d = np.load(HERE / f"results_node_aligned_{process}" / f"{sid}_{v}_e1200.npz")
            out[f"{sid}_{v}_pred"] = d["z_pred"]
            out[f"{sid}_{v}_delta"] = d["delta"]
        out[f"{sid}_obs"] = rec["z"]
        out[f"{sid}_t"] = np.arange(len(rec["z"])) * dt
    np.savez_compressed(OUT / "3c_WAAM.npz", **out)
    print(f"3c WAAM: three records, {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- 3d
def part_3d(process="GTAW", sid="Iden_02", hold=20.0, retry_hold=12.0):
    t0 = time.time()
    rec, dt = record(process, sid)
    smooth = A.SMOOTH[process]
    spec = N.SPEC
    v_hi, v_lo = (float(np.quantile(rec["speed"], q)) for q in (0.75, 0.25))
    bz, bc, bounds = bank(process, smooth)
    dec, vq, *_ = G.models()
    frames, _ = V.load_cache()

    # the first state of a recorded frame whose speed is close to v1
    j = int(np.argmin(np.abs(rec["speed"] - v_hi)))

    def run(hold_s):
        m = int(round(hold_s / dt))
        v = np.concatenate([np.full(m, v_hi), np.full(m, v_lo), np.full(m, v_hi)])
        u_np = ((v - spec.u_min) / (spec.u_max - spec.u_min)).astype(np.float32)[:, None]
        u = torch.tensor(u_np)
        res = {}
        for v_name, variant in (("A", "A_m1"), ("B", "B_no_delta")):
            model, mean, std, _ = load_model(process, sid, variant)
            z0 = torch.tensor(((rec["z"][j] - mean) / std).astype(np.float32))
            with torch.no_grad():
                z_hat, d_hat = model.rollout(z0, u, dt)
            raw = torch.tensor((z_hat.numpy() * std + mean).astype(np.float32))
            res[v_name] = dict(z=raw, delta=d_hat.numpy(), dist=nearest_distance(raw, bz))
        return m, v, u_np, res

    m, v, u_np, res = run(hold)
    # The same nearest-neighbour criterion as 3b. The reference cannot be the recorded states
    # themselves, which lie in the bank and are at zero distance from it; it is the identified
    # rollout on the real record, that is, how far a legitimate model trajectory already sits
    # from the recorded states.
    model_ref, mean_r, std_r, _ = load_model(process, sid, "A_m1")
    zn_ref = torch.tensor(((rec["z"] - mean_r) / std_r).astype(np.float32))
    with torch.no_grad():
        z_ref, _ = model_ref.rollout(zn_ref[0],
                                     torch.tensor(rec["u"].astype(np.float32)), dt)
    ref = float(np.median(nearest_distance(
        torch.tensor((z_ref.numpy() * std_r + mean_r).astype(np.float32)), bz)))
    worst = max(float(np.median(res[k]["dist"])) for k in res)
    attempts = [dict(hold_s=m * dt, median_nn_distance=worst, within_1p5x=bool(worst <= 1.5 * ref))]
    retried = False
    if worst > 1.5 * ref:                      # one retry only, with shorter holds
        retried = True
        m, v, u_np, res = run(retry_hold)
        worst = max(float(np.median(res[k]["dist"])) for k in res)
        attempts.append(dict(hold_s=m * dt, median_nn_distance=worst,
                             within_1p5x=bool(worst <= 1.5 * ref)))

    w = int(round(2.0 / dt))                      # the last 2 s of a hold
    first = slice(m - w, m)
    back = slice(3 * m - w, 3 * m)
    summary = dict(process=process, record=sid, v1_mm_s=v_hi, v2_mm_s=v_lo,
                   hold_s=m * dt, retried=retried, attempts=attempts,
                   median_nn_distance_identified_rollout=ref,
                   median_nn_distance_designed=worst,
                   within_1p5x=bool(worst <= 1.5 * ref))
    store = dict(t=np.arange(3 * m) * dt, speed=v, u=u_np, first=np.arange(m - w, m),
                 back=np.arange(3 * m - w, 3 * m), dt=dt)
    for name in ("A", "B"):
        z = res[name]["z"].numpy()
        diff = z[back].mean(0) - z[first].mean(0)
        summary[f"{name}_state_shift_norm"] = float(np.linalg.norm(diff))
        summary[f"{name}_state_shift"] = [float(x) for x in diff]
        st, _ = decode(res[name]["z"], dec)
        store[f"{name}_z"] = z
        store[f"{name}_delta"] = res[name]["delta"]
        store[f"{name}_struct"] = st
        store[f"{name}_dist"] = res[name]["dist"]
        summary[f"{name}_struct_shift_mean_abs"] = float(
            np.abs(st[back].mean(0) - st[first].mean(0)).mean())
    (OUT / f"3d_{process}_{sid}.json").write_text(json.dumps(summary, indent=1) + "\n")
    np.savez_compressed(OUT / f"3d_{process}_{sid}.npz", **store)
    print(f"3d {process} {sid}: v1 {v_hi:.2f} v2 {v_lo:.2f} mm/s, hold {m*dt:.0f} s"
          f"{' (retry)' if retried else ''}; shift A {summary['A_state_shift_norm']:.3f} "
          f"B {summary['B_state_shift_norm']:.3f}; nn {worst:.2f} vs identified {ref:.2f} "
          f"({'within' if summary['within_1p5x'] else 'outside'} 1.5x), "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["3a", "3b", "3c", "3d"]
    for p in todo:
        {"3a": part_3a, "3b": part_3b, "3c": part_3c, "3d": part_3d}[p]()
