#!/usr/bin/env python3
"""Held-out accuracy of every model on every record, at two identification shares.

One table per share. Accuracy is R^2 on the held-out interval, recomputed from the saved fits so
that every model is scored on exactly the same samples and with the seven coordinates scaled by the
whole-record standard deviation.

    python split_tables.py
"""
import json
from pathlib import Path

import numpy as np

import baselines as B
import identify_3d as I
from delta_variants import window_for

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OLD = [c for c in (OUT / "paper_cases.txt").read_text().split() if c]
NEW = [c for c in (OUT / "new_cases.txt").read_text().split() if c]
LABEL = {"node": "NODE with $\\delta$", "nodelta": "NODE without $\\delta$", "arx": "ARX (2,2)",
         "lstm": "LSTM (64)", "transformer": "Transformer (16,64)"}
SHORT = {"node": "带 δ", "nodelta": "无 δ", "arx": "ARX", "lstm": "LSTM", "transformer": "Transf."}


def r2(z_true, z_pred, held, sigma):
    t, p = z_true[held] / sigma, z_pred[held] / sigma
    sse = np.sum((p - t) ** 2, axis=0)
    sst = np.sum((t - t.mean(axis=0)) ** 2, axis=0)
    return float(1.0 - sse.sum() / max(sst.sum(), 1e-12))


def row(case, tag):
    _, z_raw, u_np, *_ = I.load_record(case, I.STATE_NAMES, window_for(case), False)
    sigma = np.maximum(z_raw.std(axis=0), 1e-9)
    d = np.load(OUT / f"{case}_v_pool_simpleD{tag}" / "fit.npz", allow_pickle=True)
    n = int(d["n_train"])
    held = np.zeros(len(z_raw), bool); held[n:] = True
    out = {"node": r2(d["z_true"], d["z_pred"], held, sigma)}
    dn = np.load(OUT / f"{case}_nodelta{tag}" / "fit.npz", allow_pickle=True)
    out["nodelta"] = r2(dn["z_true"], dn["z_pred"], held, sigma)
    a = np.load(OUT / "baselines" / f"{case}_arx{tag}.npz")
    zn = ((z_raw - a["z_mean"]) / a["z_std"]).astype(np.float64)
    pred, _ = B.arx_predict(a["theta"], zn, u_np, int(a["na"]), int(a["nb"]), True)
    out["arx"] = r2(z_raw, pred * a["z_std"] + a["z_mean"], held, sigma)
    for m, cls in (("lstm", B.LSTMModel), ("transformer", B.TransformerModel)):
        import torch
        ck = torch.load(OUT / "baselines" / f"{case}_{m}{tag}.pt", weights_only=False)
        model = (cls(z_raw.shape[1], hidden=ck["size"]) if m == "lstm"
                 else cls(z_raw.shape[1], d_model=ck["size"], window=ck["window"]))
        model.load_state_dict(ck["model"])
        z = torch.tensor(((z_raw - ck["z_mean"]) / ck["z_std"]).astype(np.float32))
        u = torch.tensor(u_np)
        with torch.no_grad():
            zh = model.rollout(z[0], u) if m == "lstm" else model.rollout(z, u)
        out[m] = r2(z_raw, zh.numpy() * ck["z_std"] + ck["z_mean"], held, sigma)
    out["split_s"] = float(d["t_split"]) if "t_split" in d.files else float("nan")
    return out


def name(case):
    c = case.replace("PP_", "")
    mat = "钢" if c.startswith("steel") else "304L"
    rest = c.split("_", 1)[1].replace("_v2", "")
    pretty = {"1_normal_var": "1 快速变化", "2_normal_const": "2 快速恒定", "3_mid_var": "3 中速变化",
              "4_slow_cycles": "4 仅电流·自由", "5_slow_cooled": "5 仅电流·水冷",
              "speed_free": "仅速度·自由", "speed_cooled": "仅速度·水冷",
              "both_free": "电流+速度·自由", "both_cooled": "电流+速度·水冷"}
    return f"{mat} {pretty.get(rest, rest)}"


def table(tag, cases):
    keys = ["node", "nodelta", "arx", "lstm", "transformer"]
    rows, ok = [], []
    for c in cases:
        try:
            rows.append((c, row(c, tag))); ok.append(c)
        except (FileNotFoundError, KeyError):
            pass
    head = f"| 记录 | 切分 s | " + " | ".join(SHORT[k] for k in keys) + " |"
    sep = "|---|---:|" + "---:|" * len(keys)
    lines = [head, sep]
    for c, r in rows:
        best = max(keys, key=lambda k: r[k])
        cells = [f"**{r[k]:.3f}**" if k == best else f"{r[k]:.3f}" for k in keys]
        mark = "$^\\dagger$" if "const" in c else ""
        lines.append(f"| {name(c)}{mark} | {r['split_s']:.0f} | " + " | ".join(cells) + " |")
    # the constant-input records have almost no held-out variance, so R^2 is not informative there
    summ = [(c, r) for c, r in rows if "const" not in c]
    if summ:
        arr = {k: np.array([r[k] for _, r in summ]) for k in keys}
        lines.append("| **均值** | | " + " | ".join(f"{arr[k].mean():.3f}" for k in keys) + " |")
        lines.append("| **中位** | | " + " | ".join(f"{np.median(arr[k]):.3f}" for k in keys) + " |")
        lines.append("| **最差** | | " + " | ".join(f"{arr[k].min():.3f}" for k in keys) + " |")
    if len(summ) < len(rows):
        lines.append("")
        lines.append("$^\\dagger$ 恒定输入，留出段几乎无方差，$R^2$ 不可用；不计入汇总。")
    return "\n".join(lines), ok


if __name__ == "__main__":
    # 2 materials x 2 plates x 3 input channels; the current-only pair is designs 4 and 5
    V2 = {"PP_steel_speed_free": "PP_steel_speed_free_v2",
          "PP_steel_speed_cooled": "PP_steel_speed_cooled_v2"}
    order = [V2.get(c, c) for c in
             (f"PP_{m}_{d}" for m in ("steel", "ss304")
              for d in ("4_slow_cycles", "5_slow_cooled", "speed_free", "speed_cooled",
                        "both_free", "both_cooled"))]
    for tag, share in (("_p60", "60 %"), ("_p50", "50 %")):
        t, ok = table(tag, order)
        print(f"\n### 辨识占比 {share}（留出 {100 - int(share.split()[0])} %）\n")
        print(t)
