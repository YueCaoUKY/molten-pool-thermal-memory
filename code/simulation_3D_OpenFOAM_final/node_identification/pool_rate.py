"""Identified pool relaxation time along each record, and how close it sits to its upper limit."""
import sys, numpy as np, torch
import identify_3d as I
import delta_variants as V

LIM = 1.0 / 0.25                      # gamma is floored at 0.35 1/s, so tau_pool <= 2.86 s


def pool_tau(case, variant):
    s, z_raw, u_np, *_ , fit, resets = I.load_record(case, I.STATE_NAMES, V.window_for(case), False)
    t = s["t_s"].to_numpy()
    train = fit & (t <= t[0] + 0.5 * (t[-1] - t[0]))
    mu, sd = np.nanmean(z_raw[train], 0), np.maximum(np.nanstd(z_raw[train], 0), 1e-5)
    zn = np.nan_to_num((z_raw - mu) / sd).astype(np.float32)
    z, u = torch.tensor(zn), torch.tensor(u_np)

    m = V.VariantNODE3D(z.shape[1], **V.VARIANTS[variant], bounds="min_only")
    ck = torch.load(f"results/{case}_v_{variant}_simpleD_p50/checkpoint.pt",
                    map_location="cpu", weights_only=False)
    sd_ck = ck.get("model", ck)
    sd_ck["raw_lam"] = sd_ck["raw_lam"].reshape(m.raw_lam.shape)
    m.load_state_dict(sd_ck)
    f = np.load(f"results/{case}_v_{variant}_simpleD_p50/fit.npz", allow_pickle=True)
    z_hat = torch.tensor(((f["z_pred"] - mu) / sd).astype(np.float32))
    d_hat = torch.tensor(f["delta_hat"].astype(np.float32))
    if d_hat.ndim == 2 and d_hat.shape[1] == 1:
        d_hat = d_hat[:, 0]
    with torch.no_grad():
        _, gamma, _ = m._terms(z_hat, d_hat, u)
    tau = (1.0 / gamma).numpy()[fit]                       # frames x pool coordinates
    return tau, [str(x) for x in I.STATE_NAMES]


CASES = [("PP_steel_speed_free_v2", "钢 仅速度·自由"), ("PP_steel_speed_cooled_v2", "钢 仅速度·水冷"),
         ("PP_steel_4_slow_cycles", "钢 仅电流·自由"), ("PP_ss304_both_free", "304L 两者·自由")]
var = sys.argv[1] if len(sys.argv) > 1 else "poolb4"
print(f"变体 {var}；熔池弛豫时间上限 {LIM:.2f} s\n")
print(f"{'记录':>16} {'中位 τ':>9} {'最大 τ':>9} {'贴上限的时间占比':>16}")
for case, label in CASES:
    tau, names = pool_tau(case, var)
    frac = float((tau > 0.98 * LIM).mean())
    print(f"{label:>16} {np.median(tau):9.2f} {tau.max():9.2f} {frac:15.0%}")
print()
tau, names = pool_tau(CASES[0][0], var)
print(f"{CASES[0][1]} 逐个熔池坐标的中位 τ (s)：")
for k, n in enumerate(names):
    print(f"   {n:<22} {np.median(tau[:, k]):5.2f}   贴上限 {float((tau[:, k] > 0.98*LIM).mean()):4.0%}")
