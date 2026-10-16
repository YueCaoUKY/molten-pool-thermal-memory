#!/usr/bin/env python3
"""Common dynamic models fitted on the same records, with the same protocol as the NODE.

Models: vector ARX (classical linear identification), an LSTM and a Transformer (black-box
sequence models). The NODE without the hidden state is already available from identify_3d.py.

Every model receives the same record, the same input and pool-state definitions, the same
standardisation, the same initial frame and the same optimiser settings; only the state
definition and the loss differ (ARX is a one-step least-squares fit in closed form). Two
accuracies are reported: one step ahead from measured states, and the free-running prediction
of the whole record from the input sequence alone.

    python baselines.py fit CASE arx|lstm|transformer [--size 32]
    python baselines.py adapt SOURCE TARGET MODEL
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import identify_3d as I
from delta_variants import window_for

HERE = Path(__file__).resolve().parent
OUT = I.OUT / "baselines"
EPOCHS, LR, ROLL = 250, 3e-3, 20
ARX_ORDERS = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 8), (32, 16), (64, 16)]
PREFIX, STEPS = 0.40, 300      # calibration share of the target record


def record(case, p_ref=None, v_ref=None):
    """The record as the NODE sees it; with p_ref/v_ref the source normalisation is carried over."""
    s, z_raw, u_np, cur, p, v, fitm, resets = I.load_record(case, I.STATE_NAMES, window_for(case), False)
    if p_ref is not None:
        u_np = np.column_stack([p / p_ref, v / v_ref]).astype(np.float32)
    return z_raw, u_np, float(p.max()), float(v.max())


def standardise(z_raw, mean=None, std=None):
    if mean is None:
        mean, std = z_raw.mean(axis=0), np.maximum(z_raw.std(axis=0), 1e-5)
    return ((z_raw - mean) / std).astype(np.float32), mean, std


def r2(zn, zhat):
    return float(I.metrics(np.asarray(zn), np.asarray(zhat))["latent_r2_global"])


# --------------------------------------------------------------------------- vector ARX
def arx_design(zn, u, na, nb, n_fit=None):
    n, d = zn.shape
    n_fit = n if n_fit is None else int(n_fit)
    k0 = max(na, nb)
    rows = [np.concatenate([zn[k - na + 1:k + 1][::-1].ravel(), u[k - nb + 1:k + 1][::-1].ravel(), [1.0]])
            for k in range(k0, n_fit - 1)]
    return np.asarray(rows, dtype=np.float64), zn[k0 + 1:n_fit].astype(np.float64), k0


def arx_fit(zn, u, na, nb, n_fit=None):
    X, Y, k0 = arx_design(zn, u, na, nb, n_fit)
    theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return theta, k0


def arx_predict(theta, zn, u, na, nb, free):
    n, d = zn.shape
    k0 = max(na, nb)
    hist = [zn[k].copy() for k in range(n)]
    out = np.array(zn, dtype=np.float64, copy=True)
    for k in range(k0, n - 1):
        past = np.concatenate([np.asarray(hist[k - na + 1:k + 1][::-1]).ravel(),
                               u[k - nb + 1:k + 1][::-1].ravel(), [1.0]])
        nxt = past @ theta
        out[k + 1] = nxt
        if free:
            hist[k + 1] = nxt
    return out, k0


# --------------------------------------------------------------------------- LSTM
class LSTMModel(nn.Module):
    """A recurrent black box with the same interface as the NODE: state and input in, next state out."""

    def __init__(self, z_dim, hidden=32):
        super().__init__()
        self.cell = nn.LSTMCell(z_dim + 2, hidden)
        self.head = nn.Linear(hidden, z_dim)
        self.hidden = hidden

    def step(self, z, u, state):
        h, c = self.cell(torch.cat([z, u], dim=-1), state)
        return z + self.head(h), (h, c)

    def rollout(self, z0, u, z_true=None):
        h = torch.zeros(1, self.hidden)
        state, z = (h, h.clone()), z0.unsqueeze(0)
        out = [z]
        for k in range(u.shape[0] - 1):
            z_in = z if z_true is None else z_true[k].unsqueeze(0)
            z, state = self.step(z_in, u[k].unsqueeze(0), state)
            out.append(z)
        return torch.cat(out, dim=0)


# --------------------------------------------------------------------------- Transformer
class TransformerModel(nn.Module):
    """Attention over the last `window` frames of input and state, predicting the next state."""

    def __init__(self, z_dim, d_model=32, window=32, layers=2, heads=4):
        super().__init__()
        self.window, self.z_dim = window, z_dim
        self.proj = nn.Linear(z_dim + 2, d_model)
        self.pos = nn.Parameter(torch.zeros(window, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, z_dim)

    def step(self, seq):                       # seq: (batch, window, z_dim + 2)
        h = self.enc(self.proj(seq) + self.pos)
        return seq[:, -1, :self.z_dim] + self.head(h[:, -1])

    def rollout(self, z, u, starts=None, steps=None):
        """Free rollout from the first `window` frames, or K-step rollouts from given starts."""
        w = self.window
        if starts is None:
            buf = torch.cat([z[:w], u[:w]], dim=-1).unsqueeze(0)
            out = [z[:w]]
            for k in range(w, u.shape[0]):
                nxt = self.step(buf)
                out.append(nxt)
                buf = torch.cat([buf[:, 1:], torch.cat([nxt, u[k].unsqueeze(0)], dim=-1).unsqueeze(1)], dim=1)
            return torch.cat(out, dim=0)
        seq = torch.stack([torch.cat([z[s - w:s], u[s - w:s]], dim=-1) for s in starts])
        preds = []
        for j in range(steps):
            nxt = self.step(seq)
            preds.append(nxt)
            uk = torch.stack([u[s + j] for s in starts])
            seq = torch.cat([seq[:, 1:], torch.cat([nxt, uk], dim=-1).unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)


# --------------------------------------------------------------------------- fitting
def fit(case, model_name, size=None, seed=42, epochs=EPOCHS, order=None, window=32, tag="",
        train_frac=1.0):
    """One structure on one record. `order` fixes the ARX orders (na, nb), `size` the LSTM hidden
    width or the Transformer model width, `window` the Transformer context; `tag` names the output."""
    t0 = time.time()
    I.seed_everything(seed)
    z_raw, u_np, p_max, v_max = record(case)
    # Within-record held-out protocol: identify on the first train_frac of the record, then predict
    # the remainder recursively with no state correction at the boundary.
    n_all = len(z_raw)
    n_tr = n_all if train_frac >= 1.0 else int(round(train_frac * n_all))
    zn, z_mean, z_std = standardise(z_raw[:n_tr])
    zn = ((z_raw - z_mean) / z_std).astype(np.float32)
    z, u = torch.tensor(zn), torch.tensor(u_np)
    d = zn.shape[1]
    info = dict(case=case, model=model_name, seed=seed, frames=int(len(zn)), window_s=list(window_for(case)),
                train_frac=float(train_frac), n_train=int(n_tr), n_heldout=int(n_all - n_tr))

    def rmse(a, b):
        """RMSE in units of the identification-part standard deviation (z is already standardised)."""
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    def scores(pred, k0):
        """Free-running accuracy over the whole record, the identified part and the held-out part."""
        out = dict(r2_free=r2(zn[k0:], pred[k0:]), r2_train=r2(zn[k0:n_tr], pred[k0:n_tr]),
                   nrmse_std=rmse(zn[k0:], pred[k0:]))
        if n_tr < n_all:
            out["r2_heldout"] = r2(zn[n_tr:], pred[n_tr:])
            out["nrmse_std_heldout"] = rmse(zn[n_tr:], pred[n_tr:])
        return out

    if model_name == "arx":
        best = None
        for na, nb in ([tuple(order)] if order else ARX_ORDERS):
            theta, k0 = arx_fit(zn, u_np, na, nb, n_tr)
            free, _ = arx_predict(theta, zn, u_np, na, nb, True)
            one, _ = arx_predict(theta, zn, u_np, na, nb, False)
            sc = scores(free, k0)
            key = sc.get("r2_heldout", sc["r2_free"])
            if best is None or key > best["_key"]:
                best = dict(na=na, nb=nb, **sc, _key=key,
                            r2_one=r2(zn[k0:], one[k0:]),
                            r2_one_heldout=r2(zn[n_tr:], one[n_tr:]) if n_tr < n_all else float("nan"),
                            theta=theta, k0=k0, n_params=int(theta.size))
        info.update({k: v for k, v in best.items() if k not in ("theta", "_key")})
        np.savez_compressed(OUT / f"{case}_arx{tag}.npz", theta=best["theta"], z_mean=z_mean, z_std=z_std,
                            na=best["na"], nb=best["nb"], p_max=p_max, v_max=v_max)
    else:
        if model_name == "lstm":
            model = LSTMModel(d, hidden=size or 32)
        else:
            model = TransformerModel(d, d_model=size or 32, window=window)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.05 * LR)
        best_loss, best_state = float("inf"), None
        for ep in range(epochs):
            opt.zero_grad(set_to_none=True)
            if model_name == "lstm":
                loss = F.mse_loss(model.rollout(z[0], u[:n_tr]), z[:n_tr])
            else:
                w = model.window
                starts = torch.randint(w, n_tr - ROLL - 1, (48,))
                pred = model.rollout(z, u, starts=starts, steps=ROLL)
                tgt = torch.stack([z[s:s + ROLL] for s in starts])
                loss = F.mse_loss(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            if float(loss) < best_loss:
                best_loss = float(loss)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ep % 50 == 0:
                print(f"{case} {model_name} epoch {ep:3d} loss {float(loss):.5f}", flush=True)
        model.load_state_dict(best_state)
        with torch.no_grad():
            if model_name == "lstm":
                free = model.rollout(z[0], u).numpy()
                one = model.rollout(z[0], u, z_true=z).numpy()
                k0 = 0
            else:
                free = model.rollout(z, u).numpy()
                k0 = model.window
                seq = torch.stack([torch.cat([z[t - k0:t], u[t - k0:t]], dim=-1)
                                   for t in range(k0, len(z))])
                one = np.concatenate([zn[:k0], model.step(seq).numpy()], axis=0)
        info.update(**scores(free, k0), r2_one=r2(zn[k0:], one[k0:]),
                    r2_one_heldout=r2(zn[n_tr:], one[n_tr:]) if n_tr < n_all else float("nan"),
                    n_params=sum(p.numel() for p in model.parameters()), size=size or 32,
                    window=window if model_name == "transformer" else None)
        torch.save({"model": best_state, "z_mean": z_mean, "z_std": z_std, "size": size or 32,
                    "window": window, "p_max": p_max, "v_max": v_max}, OUT / f"{case}_{model_name}{tag}.pt")

    info["wall_s"] = round(time.time() - t0, 1)
    info["tag"] = tag
    (OUT / f"{case}_{model_name}{tag}.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: info[k] for k in ("case", "model", "r2_free", "r2_heldout", "r2_one", "n_params",
                                           "wall_s") if k in info}), flush=True)
    return info


# --------------------------------------------------------------------------- adaptation
def arx_rollout(theta, na, nb, zn, u, n, g, b):
    """Free rollout of the ARX model with a per-coordinate affine output correction."""
    d = zn.shape[1]
    k0 = max(na, nb)
    nz = na * d
    theta_z, theta_u = theta[:nz], theta[nz:]
    ulag = np.zeros((n, theta_u.shape[0]))
    ulag[:, -1] = 1.0                            # the constant term is the last regressor
    for k in range(n):
        seg = u[max(k - nb + 1, 0):k + 1][::-1].ravel()
        ulag[k, :len(seg)] = seg
    u_part = ulag @ theta_u
    buf = zn[k0 - na + 1:k0 + 1][::-1].astype(np.float64).copy()
    out = np.array(zn[:n], dtype=np.float64, copy=True)
    for k in range(k0, n - 1):
        nxt = g * (buf.ravel() @ theta_z + u_part[k]) + b
        out[k + 1] = nxt
        buf = np.roll(buf, 1, axis=0)
        buf[0] = nxt
    return out


def adapt(source, target, model_name="arx"):
    """Freeze the identified model; re-estimate one gain and one offset per pool-state
    coordinate on the first 20 % of the target record, then predict the rest."""
    from scipy.optimize import minimize
    t0 = time.time()
    ck = np.load(OUT / f"{source}_arx.npz")
    z_mean_s, z_std_s = ck["z_mean"], ck["z_std"]
    na, nb, theta = int(ck["na"]), int(ck["nb"]), ck["theta"]
    z_raw, u_np, _, _ = record(target, float(ck["p_max"]), float(ck["v_max"]))
    zn_s = ((z_raw - z_mean_s) / z_std_s).astype(np.float64)
    n_pre = max(40, int(PREFIX * len(zn_s)))
    mean_t, std_t = z_raw.mean(axis=0), np.maximum(z_raw.std(axis=0), 1e-5)
    d = zn_s.shape[1]

    def cost(x):
        pred = arx_rollout(theta, na, nb, zn_s, u_np, n_pre, x[:d], x[d:])
        k0 = max(na, nb)
        e = pred[k0:] - zn_s[k0:n_pre]
        return float(np.mean(e ** 2)) if np.isfinite(e).all() else 1e6

    x0 = np.concatenate([np.ones(d), np.zeros(d)])
    res = minimize(cost, x0, method="Powell", options=dict(maxfev=4000, xtol=1e-3, ftol=1e-4))
    g, b = res.x[:d], res.x[d:]
    n = len(zn_s)
    none = arx_rollout(theta, na, nb, zn_s, u_np, n, np.ones(d), np.zeros(d))
    adapted = arx_rollout(theta, na, nb, zn_s, u_np, n, g, b)
    held = slice(n_pre, n)
    to_raw = lambda a: a * z_std_s + z_mean_s
    def score(pred):
        o = (z_raw[held] - mean_t) / std_t
        q = (to_raw(pred)[held] - mean_t) / std_t
        ok = np.isfinite(q).all(axis=1)
        return float(I.metrics(o[ok], q[ok])["latent_r2_global"]) if ok.any() else float("nan")
    own = json.loads((OUT / f"{target}_arx.json").read_text())["r2_free"]
    out = dict(source=source, target=target, model="arx", prefix_s=round(n_pre * I.DT, 1),
               r2_none=round(score(none), 3), r2_adapt=round(score(adapted), 3),
               r2_own_record=round(own, 3), gain=[round(float(v), 3) for v in g],
               offset=[round(float(v), 3) for v in b], wall_s=round(time.time() - t0, 1))
    (OUT / f"adapt_{source}_to_{target}_arx_p{int(100 * PREFIX)}.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("source", "target", "r2_none", "r2_adapt", "wall_s")}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("case")
    f.add_argument("model", choices=["arx", "lstm", "transformer"])
    f.add_argument("--size", type=int, default=None)
    f.add_argument("--order", type=int, nargs=2, default=None, metavar=("NA", "NB"))
    f.add_argument("--window", type=int, default=32)
    f.add_argument("--tag", default="")
    f.add_argument("--epochs", type=int, default=EPOCHS)
    f.add_argument("--threads", type=int, default=2)
    f.add_argument("--train-frac", type=float, default=1.0, dest="train_frac",
                   help="identify on the first fraction of the record and score the rest as held out")
    a_ = sub.add_parser("adapt")
    a_.add_argument("source")
    a_.add_argument("target")
    a_.add_argument("model", choices=["arx", "lstm", "transformer"])
    a_.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "fit":
        fit(a.case, a.model, a.size, epochs=a.epochs, order=a.order, window=a.window, tag=a.tag,
            train_frac=a.train_frac)
    else:
        adapt(a.source, a.target, a.model)


if __name__ == "__main__":
    main()
