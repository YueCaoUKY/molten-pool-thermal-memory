"""Shared training library used by GMAW and WAAM ViD runner scripts.

Mirrors the training step described in the original GTAW
`02_training.ipynb`: VAE encoder + control-affine NODE + PCGrad on the
encoder + NODE inner loop. Caller supplies a configured Dataset (already
in `mode='feature'`) and the loss/learning hyper-parameters via the
Config object.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from encoder import Encoder
from prior import ControlAffineNODE
from decoder import Decoder
from losses import multi_frame_recon, temporal_kl_loss, pcgrad_surgery


def make_models(cfg, device):
    enc = Encoder(cfg.dinov2_dim, cfg.encoder_hidden, cfg.d_s).to(device)
    node = ControlAffineNODE(cfg.d_s, cfg.prior_hidden, cfg.dt).to(device)
    dec = Decoder(cfg.d_s, cfg.dec_init_ch, tuple(cfg.dec_block_ch)).to(device)
    return enc, node, dec


def lr_schedule(step, cfg):
    if step < 300:
        return cfg.lr * step / 300
    p = (step - 300) / max(1, cfg.train_steps - 300)
    return 1e-5 + (cfg.lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def collect_zs(enc, ds, device, batch_size, num_workers=0):
    from torch.utils.data import DataLoader
    enc.eval()
    ts, mus, us = [], [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    with torch.no_grad():
        for b in loader:
            mu, _ = enc(b["cls_window"].to(device))
            mus.append(mu[:, 50].cpu().numpy())
            us.append(b["u_center"].numpy())
            ts.append(b["t"].numpy())
    ts = np.concatenate(ts); mus = np.concatenate(mus); us = np.concatenate(us)
    o = np.argsort(ts)
    return ts[o], mus[o], us[o]


def save_zs_plot(out_dir, enc, ds, cfg, device, step, dim_kl=None,
                 use_u_corr=True, batch_size=None):
    ts, mus, us = collect_zs(enc, ds, device, batch_size or cfg.batch_size)
    fig, axes = plt.subplots(cfg.d_s, 1, figsize=(14, 1.8 * cfg.d_s), sharex=True)
    if cfg.d_s == 1:
        axes = [axes]
    label = "u" if use_u_corr else "t"
    ref = us if use_u_corr else ts.astype(np.float32)
    for k in range(cfg.d_s):
        ax = axes[k]; mk = mus[:, k]
        mk_n = (mk - mk.mean()) / (mk.std() + 1e-8)
        r = np.corrcoef(mk, ref)[0, 1] if ref.std() > 0 else float("nan")
        kl_val = dim_kl[k].item() if dim_kl is not None else 0.0
        ax.plot(ts, mk_n, "tab:blue", lw=0.7)
        ax.set_ylabel(f"z[{k}]"); ax.set_ylim(-4, 4)
        ax.text(0.01, 0.9, f"r({label})={r:+.3f}  KL={kl_val:.2f}",
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round", fc="w", alpha=.8))
        if use_u_corr:
            ax2 = ax.twinx(); ax2.plot(ts, us, "tab:red", lw=.8, alpha=.5); ax2.set_ylim(-.1, 1.1)
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"z_s (encoder μ) vs {label} - step {step}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / f"{step:06d}.png", dpi=140, bbox_inches="tight")
    plt.close()


def save_recon_grid(out_dir, enc, dec, ds, cfg, device, step):
    enc.eval(); dec.eval()
    idx = np.linspace(0, len(ds) - 1, 8).astype(int)
    reals, recs, labels = [], [], []
    with torch.no_grad():
        for i in idx:
            s = ds[int(i)]
            cls = s["cls_window"].unsqueeze(0).to(device)
            mu, _ = enc(cls)
            xs = torch.sigmoid(dec(mu[:, 50])).cpu().numpy()[0, 0]
            recs.append(xs); reals.append(s["x_center"].numpy()[0])
            labels.append(float(s["u_center"]))
    fig, ax = plt.subplots(2, 8, figsize=(18, 5))
    for c in range(8):
        ax[0, c].imshow(reals[c], cmap="gray", vmin=0, vmax=1)
        ax[0, c].set_title(f"u={labels[c]:.2f}", fontsize=8)
        ax[1, c].imshow(recs[c], cmap="gray", vmin=0, vmax=1)
        for r in range(2):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[0, 0].set_ylabel("real"); ax[1, 0].set_ylabel("x_s")
    fig.suptitle(f"Step {step}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / f"{step:06d}.png", dpi=150, bbox_inches="tight")
    plt.close()


def train(cfg, ds_factory, use_u_corr=True):
    """Train an encoder+NODE+decoder bundle.

    Parameters
    ----------
    cfg : Config
        Dataset config; cfg.root_dir, cfg.features_cache_dir, cfg.batch_size,
        cfg.train_steps, etc. are all read from here.
    ds_factory : callable() -> torch.utils.data.Dataset
        Function returning the training-mode dataset (mode='feature',
        recon_indices set).
    use_u_corr : bool
        Plot r(z, u). For WAAM ViD pass False so we plot r(z, t) instead
        (u is identically zero there).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.root_dir) / "results"
    zs_dir = out / "zs_plots"
    recon_dir = out / "recon_grids"
    for d in [out, zs_dir, recon_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ds = ds_factory()
    print(f"[train] windows: {len(ds)}", flush=True)
    enc, node, dec = make_models(cfg, device)
    n = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[train] enc:{n(enc):,} node:{n(node):,} dec:{n(dec):,} "
          f"total:{n(enc)+n(node)+n(dec):,}", flush=True)

    save_recon_grid(recon_dir, enc, dec, ds, cfg, device, 0)

    opt_main = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                                 lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_node = torch.optim.AdamW(node.parameters(),
                                 lr=cfg.node_lr, weight_decay=cfg.weight_decay)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    it = iter(loader)
    log = []
    last_dim_kl = None
    t0 = time.time()

    for step in range(1, cfg.train_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)

        cls = batch["cls_window"].to(device)
        u_w = batch["u_window"].to(device)
        x_recon = batch["x_recon"].to(device)

        lr = lr_schedule(step, cfg)
        for g in opt_main.param_groups:
            g["lr"] = lr

        enc.train(); node.train(); dec.train()

        with torch.no_grad():
            mu_frozen, logvar_frozen = enc(cls)
        mu_post, logvar_post = enc(cls)

        z0_frozen = mu_frozen[:, 0]
        for _ in range(cfg.node_inner_steps):
            z_prior_inner = node.rollout(z0_frozen, u_w)
            pv_inner = node.get_prior_logvar(z_prior_inner, u_w, cfg.kl_indices)
            kl_inner, _ = temporal_kl_loss(mu_frozen, logvar_frozen, z_prior_inner,
                                           pv_inner, cfg.kl_indices, cfg.free_nats)
            opt_node.zero_grad(); kl_inner.backward()
            torch.nn.utils.clip_grad_norm_(node.parameters(), cfg.grad_clip)
            opt_node.step()

        z_prior = node.rollout(mu_post[:, 0].detach(), u_w)
        prior_logvars = node.get_prior_logvar(z_prior, u_w, cfg.kl_indices)
        kl_raw, kl_s = temporal_kl_loss(mu_post, logvar_post, z_prior,
                                        prior_logvars, cfg.kl_indices, cfg.free_nats)
        L_rec, rec_s = multi_frame_recon(x_recon, mu_post, logvar_post, dec, cfg.recon_indices)
        L_kl_w = cfg.beta * kl_raw

        opt_main.zero_grad()
        conflict = pcgrad_surgery(enc, L_rec, L_kl_w, retain_graph=True)
        enc_grads = {nn: p.grad.clone() for nn, p in enc.named_parameters() if p.grad is not None}
        opt_main.zero_grad()
        (L_rec + L_kl_w).backward()
        for nn, p in enc.named_parameters():
            if nn in enc_grads:
                p.grad = enc_grads[nn]
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), cfg.grad_clip)
        opt_main.step()

        last_dim_kl = kl_s["dim_kl"]

        if step % cfg.log_every == 0:
            mu_std = mu_post.std(dim=1).mean().item()
            with torch.no_grad():
                rep_t = cfg.kl_indices[len(cfg.kl_indices) // 2]
                sig_p = (0.5 * prior_logvars[rep_t]).exp().mean().item()
            dk = last_dim_kl.cpu().numpy()
            log.append(dict(step=step, l1=rec_s["l1"].item(), kl=kl_s["kl"].item(),
                            mu_std=mu_std, lr=lr, dim_kl=dk.tolist(),
                            sig_prior=sig_p, conflict=bool(conflict),
                            kl_inner=float(kl_inner.item())))
            c_str = "!" if conflict else "."
            elapsed = time.time() - t0
            eta_s = elapsed / step * (cfg.train_steps - step)
            print(f"[{step:5d}/{cfg.train_steps}] {c_str} l1={rec_s['l1'].item():.4f} "
                  f"kl={kl_s['kl'].item():.1f} mu_std={mu_std:.4f} sigp={sig_p:.3f} "
                  f"({elapsed:.0f}s elapsed, ETA {eta_s:.0f}s)", flush=True)

        if step % cfg.val_every == 0:
            save_zs_plot(zs_dir, enc, ds, cfg, device, step, last_dim_kl, use_u_corr=use_u_corr)
            save_recon_grid(recon_dir, enc, dec, ds, cfg, device, step)
            print("  >>> snapshot saved", flush=True)

    torch.save({"enc": enc.state_dict(), "node": node.state_dict(), "dec": dec.state_dict(),
                "step": cfg.train_steps, "log": log, "cfg": cfg.__dict__},
               out / "final.pt")
    save_zs_plot(zs_dir, enc, ds, cfg, device, cfg.train_steps, last_dim_kl, use_u_corr=use_u_corr)
    save_recon_grid(recon_dir, enc, dec, ds, cfg, device, cfg.train_steps)
    print(f"[train] done in {time.time()-t0:.0f}s -> {out/'final.pt'}", flush=True)

    plot_loss_curves(log, out, cfg)
    return enc, node, dec, log


def plot_loss_curves(log, out, cfg):
    steps = [e["step"] for e in log]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes[0, 0].plot(steps, [e["l1"] for e in log], lw=1); axes[0, 0].set_title("L_recon"); axes[0, 0].set_yscale("log")
    axes[0, 1].plot(steps, [e["kl"] for e in log], lw=1, color="tab:red", label="outer KL")
    axes[0, 1].plot(steps, [e["kl_inner"] for e in log], lw=1, color="tab:pink", alpha=0.7, label="inner KL")
    axes[0, 1].set_title("KL"); axes[0, 1].legend(fontsize=8)
    axes[0, 2].plot(steps, [e["sig_prior"] for e in log], lw=1, color="tab:orange"); axes[0, 2].set_title("Prior sigma")
    axes[1, 0].plot(steps, [e["mu_std"] for e in log], lw=1, color="tab:green"); axes[1, 0].set_title("mu within-window std")
    win = 10; conflicts = [e["conflict"] for e in log]
    rates = [sum(conflicts[max(0, i-win):i+1]) / max(min(i+1, win), 1) for i in range(len(conflicts))]
    axes[1, 1].plot(steps, rates, lw=1, color="tab:purple"); axes[1, 1].set_title("Conflict rate"); axes[1, 1].set_ylim(0, 1)
    for k in range(cfg.d_s):
        axes[1, 2].plot(steps, [e["dim_kl"][k] for e in log], lw=0.8, label=f"d{k}")
    axes[1, 2].set_title("Per-dim KL"); axes[1, 2].legend(fontsize=7)
    for a in axes.flat:
        a.set_xlabel("step"); a.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(out / "loss_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    with open(out / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
