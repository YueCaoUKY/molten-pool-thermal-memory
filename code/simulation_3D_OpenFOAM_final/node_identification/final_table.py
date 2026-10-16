"""Final table: the two-state NODE against every baseline, within-record held-out, 50 % division."""
import json, pathlib, numpy as np

R, B = pathlib.Path("results"), pathlib.Path("results/baselines")
CASES = [("PP_steel_4_slow_cycles", "钢", "自由", "仅电流"), ("PP_steel_5_slow_cooled", "钢", "水冷", "仅电流"),
         ("PP_steel_speed_free_v2", "钢", "自由", "仅速度"), ("PP_steel_speed_cooled_v2", "钢", "水冷", "仅速度"),
         ("PP_steel_both_free", "钢", "自由", "两者"), ("PP_steel_both_cooled", "钢", "水冷", "两者"),
         ("PP_ss304_4_slow_cycles", "304L", "自由", "仅电流"), ("PP_ss304_5_slow_cooled", "304L", "水冷", "仅电流"),
         ("PP_ss304_speed_free_v2", "304L", "自由", "仅速度"), ("PP_ss304_speed_cooled_v2", "304L", "水冷", "仅速度"),
         ("PP_ss304_both_free", "304L", "自由", "两者"), ("PP_ss304_both_cooled", "304L", "水冷", "两者")]


def node(case, tag):
    f = R / f"{case}{tag}" / "metrics.json"
    return json.load(f.open())["r2_global_heldout"] if f.exists() else np.nan


def base(case, name):
    f = B / f"{case}_{name}_p50.json"
    return json.load(f.open())["r2_heldout"] if f.exists() else np.nan


rows = []
for case, mat, plate, inp in CASES:
    rows.append(dict(记录=f"{mat} {inp}·{plate}",
                     NODE_delta=node(case, "_v_poolb4_simpleD_p50"),
                     NODE_no_delta=node(case, "_nodelta_p50"),
                     ARX=base(case, "arx"), LSTM=base(case, "lstm"), Transformer=base(case, "transformer")))

COLS = ["NODE_delta", "NODE_no_delta", "ARX", "LSTM", "Transformer"]
HEAD = {"NODE_delta": "NODE+δ", "NODE_no_delta": "NODE 无δ", "ARX": "ARX", "LSTM": "LSTM", "Transformer": "Transformer"}
print(f"{'记录':<18}" + "".join(f"{HEAD[c]:>12}" for c in COLS))
for r in rows:
    line = "".join(f"{r[c]:12.3f}" if np.isfinite(r[c]) else f"{'-':>12}" for c in COLS)
    print(f"{r['记录']:<18}" + line)
print("-" * 78)
for stat, fn in [("均值", np.nanmean), ("中位", np.nanmedian), ("最差", np.nanmin)]:
    print(f"{stat:<18}" + "".join(f"{fn([r[c] for r in rows]):12.3f}" for c in COLS))
w = sum(1 for r in rows if r["NODE_delta"] > r["ARX"])
print(f"\nNODE+δ 胜过 ARX 的记录数：{w} / {sum(1 for r in rows if np.isfinite(r['NODE_delta']))}")
for r in rows:
    if np.isfinite(r["NODE_delta"]) and r["NODE_delta"] <= r["ARX"]:
        print(f"   输给 ARX：{r['记录']}  {r['NODE_delta']:.3f} vs {r['ARX']:.3f}")
