"""Losses + gradient surgery for temporal VAE with control-affine NODE.

L_recon: L1 at multiple frames.
L_KL:    KL(posterior || NODE prior) with learned σ_prior.
pcgrad_surgery: project conflicting gradients on encoder so recon and KL
                find common ground instead of fighting.
"""
import torch
import torch.nn.functional as F


def multi_frame_recon(x_recon, mu, logvar, decoder, recon_indices):
    total = 0.0
    for i, ri in enumerate(recon_indices):
        z = mu[:, ri] + (0.5 * logvar[:, ri]).exp() * torch.randn_like(mu[:, ri])
        x_pred = torch.sigmoid(decoder(z))
        total = total + F.l1_loss(x_pred, x_recon[:, i])
    loss = total / len(recon_indices)
    return loss, {"l1": loss.detach()}


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    return 0.5 * (logvar_p - logvar_q + var_q / var_p + (mu_q - mu_p).pow(2) / var_p - 1)


def temporal_kl_loss(mu_post, logvar_post, z_prior, prior_logvars, kl_indices, free_nats=0.1):
    d_s = mu_post.shape[2]
    total = 0.0
    dim_kl_accum = torch.zeros(d_s, device=mu_post.device)
    for t in kl_indices:
        kl = gaussian_kl(mu_post[:, t], logvar_post[:, t], z_prior[:, t], prior_logvars[t])
        dim_kl_accum = dim_kl_accum + kl.mean(dim=0).detach()
        total = total + kl.clamp(min=free_nats).sum(dim=1).mean()
    n = max(len(kl_indices), 1)
    return total / n, {"kl": (total / n).detach(), "dim_kl": dim_kl_accum / n}


def pcgrad_surgery(enc, L_rec, L_kl, retain_graph=False):
    """PCGrad on encoder: project recon gradient if it conflicts with KL gradient.

    Returns combined gradient direction that both losses agree on.
    This is applied ONLY to encoder parameters. Decoder and NODE get normal gradients.

    After calling this, encoder.parameters() have .grad set to the surgered gradient.
    Caller should then backward L_rec through decoder (for decoder grads) and
    L_kl through NODE (for NODE grads) separately, or just use the combined loss
    for non-encoder parameters.
    """
    enc_params = [p for p in enc.parameters() if p.requires_grad]

    # Gradient from L_rec on encoder
    g_rec_list = torch.autograd.grad(L_rec, enc_params, retain_graph=True, allow_unused=True)
    g_rec = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device)
                       for g, p in zip(g_rec_list, enc_params)])

    # Gradient from L_kl on encoder
    g_kl_list = torch.autograd.grad(L_kl, enc_params, retain_graph=retain_graph, allow_unused=True)
    g_kl = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device)
                      for g, p in zip(g_kl_list, enc_params)])

    # PCGrad: if conflicting, project g_rec onto plane ⊥ g_kl
    dot = (g_rec * g_kl).sum()
    conflict = dot < 0
    if conflict:
        g_rec = g_rec - (dot / (g_kl.norm().pow(2) + 1e-8)) * g_kl

    # Combined
    g_combined = g_rec + g_kl

    # Assign to encoder params
    offset = 0
    for p in enc_params:
        numel = p.numel()
        p.grad = g_combined[offset:offset + numel].reshape(p.shape)
        offset += numel

    return conflict.item()
