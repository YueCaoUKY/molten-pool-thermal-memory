"""Table for the band-constrained thermal rates: held-out R^2 against ARX and the scalar delta."""
import json, pathlib, numpy as np, torch

R = pathlib.Path("results"); B = R / "baselines"
CASES = [("PP_steel_speed_free_v2",  "钢 仅速度·自由"),
         ("PP_steel_speed_cooled_v2", "钢 仅速度·水冷"),
         ("PP_steel_4_slow_cycles",  "钢 仅电流·自由"),
         ("PP_ss304_both_free",      "304L 两者·自由")]
VAR = ["pool", "poolb3", "poolb4", "poolb5"]
NAME = {"pool": "1 维 δ", "poolb3": "3 维 δ", "poolb4": "4 维 δ", "poolb5": "5 维 δ"}


def taus(case, v):
    """Identified thermal relaxation times, from the band parameters in the checkpoint."""
    f = R / f"{case}_v_{v}_simpleD_p50" / "checkpoint.pt"
    if not f.exists():
        return None
    raw = torch.load(f, map_location="cpu", weights_only=False)
    raw = raw.get("model", raw)
    lam = raw["raw_lam"].detach().numpy().ravel()
    if v == "pool":
        return [1.0 / np.log1p(np.exp(lam[0]))]
    m = len(lam)
    edge = ((1 / 0.25)) * (40.0 * 0.25) ** (np.arange(m + 1) / m)
    return list(edge[:-1] * (edge[1:] / edge[:-1]) ** (1 / (1 + np.exp(-lam))))


def r2(case, v):
    f = R / f"{case}_v_{v}_simpleD_p50" / "metrics.json"
    return json.load(f.open())["r2_global_heldout"] if f.exists() else None


print(f"{'记录':>16} {'ARX':>8}" + "".join(f"{NAME[v]:>9}" for v in VAR))
for case, label in CASES:
    a = json.load((B / f"{case}_arx_p50.json").open())["r2_heldout"]
    row = "".join(f"{x:9.3f}" if (x := r2(case, v)) is not None else f"{'-':>9}" for v in VAR)
    print(f"{label:>16} {a:8.3f}" + row)
print()
for case, label in CASES:
    for v in VAR[1:]:
        t = taus(case, v)
        if t:
            print(f"{label:>16} {NAME[v]}  τ = {[round(x, 1) for x in t]} s")
