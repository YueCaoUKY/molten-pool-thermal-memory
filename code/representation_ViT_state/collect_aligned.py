#!/usr/bin/env python3
"""Collect the aligned Section 6 runs: convergence evidence, Table 6.2, and the old-vs-new check."""
import json, pathlib, re, sys
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
PROC = ["GTAW", "GMAW", "WAAM"]
# the published values, from gtaw/gmaw/waam_final_results.md (1200 epochs, the old model)
OLD = {
    "Iden_09": 0.901, "Iden_11": 0.886, "Iden_01": 0.865, "Iden_07": 0.846, "Iden_06": 0.779,
    "Iden_02": 0.771, "Iden_12": 0.750, "Iden_08": 0.723, "Iden_13": 0.616, "Iden_10": 0.579,
    "exp_05": 0.954, "exp_04": 0.897, "exp_07": 0.887, "exp_03": 0.881, "exp_09": 0.878,
    "exp_01": 0.872, "exp_02": 0.855, "exp_06": 0.797, "exp_10": 0.770,
    "video_1": 0.968, "video_3": 0.904, "video_5": 0.944,
}
OLD_B = {"video_1": 0.444, "video_3": 0.459, "video_5": 0.334}   # WAAM without the thermal state


def load():
    rows = []
    for p in PROC:
        for f in sorted((HERE / f"results_node_aligned_{p}").glob("*.json")):
            rows.append(json.loads(f.read_text()))
    d = pd.DataFrame(rows)
    d["tau1_s"] = d["tau_s"].apply(lambda t: round(float(t[0]), 2))
    return d


def convergence(d):
    """Evidence that 1200 epochs has converged: on the records that also ran to 2000 epochs,
    compare the final trajectory loss and the R2 of the two. No further runs are made for this."""
    a = d[d.variant == "A_m1"]
    ref = a[a.epochs == 2000].set_index("record")
    cur = a[a.epochs == 1200].set_index("record")
    both = ref.index.intersection(cur.index)
    if not len(both):
        return pd.DataFrame()
    out = pd.DataFrame({
        "process": cur.loc[both, "process"],
        "loss_1200": cur.loc[both, "best_loss"].round(5),
        "loss_2000": ref.loc[both, "best_loss"].round(5),
        "loss_excess_pct": ((cur.loc[both, "best_loss"] / ref.loc[both, "best_loss"] - 1) * 100).round(2),
        "R2_1200": cur.loc[both, "r2_smoothed"].round(3),
        "R2_2000": ref.loc[both, "r2_smoothed"].round(3),
    })
    out["R2_diff"] = (out.R2_1200 - out.R2_2000).round(3)
    return out.reset_index().rename(columns={"index": "record"})


def table_6_2(d, epochs):
    rows = []
    for p in PROC:
        s = d[(d.process == p) & (d.epochs == epochs)]
        row = {"process": p, "records": int(s[s.variant == "A_m1"].shape[0])}
        for v, name in (("A_m1", "A"), ("B_no_delta", "B")):
            x = s[s.variant == v]["r2_smoothed"]
            row[f"R2_{name}_median"] = round(float(x.median()), 3) if len(x) else np.nan
            row[f"R2_{name}_min"] = round(float(x.min()), 3) if len(x) else np.nan
            row[f"R2_{name}_max"] = round(float(x.max()), 3) if len(x) else np.nan
        row["predictable_share_median"] = round(
            float(s[s.variant == "A_m1"]["predictable_share"].median()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    d = load()
    conv = convergence(d)
    print("=== 收敛证据：1200 轮 对 2000 轮（仅对两者都跑过的记录）===")
    print(conv.to_string(index=False) if len(conv) else "（还没有可比的记录）")
    if len(conv):
        print(f"损失相对 2000 轮高出：中位 {conv.loss_excess_pct.median():+.2f} %  "
              f"最差 {conv.loss_excess_pct.max():+.2f} %;  "
              f"R² 差：中位 {conv.R2_diff.median():+.3f}  最差 {conv.R2_diff.min():+.3f}")
    epochs = epochs or 1200
    print(f"\n使用轮数: {epochs}\n")
    sel = d[d.epochs == epochs].copy()
    sel["old_R2"] = sel.record.map(OLD)
    sel["delta_vs_old"] = (sel.r2_smoothed - sel.old_R2).round(3)
    cols = ["process", "record", "variant", "frames", "r2_smoothed", "r2_as_read",
            "predictable_share", "tau1_s", "old_R2", "delta_vs_old", "wall_s"]
    sel[cols].to_csv(HERE / "per_record.csv", index=False)
    t = table_6_2(d, epochs)
    t.to_csv(HERE / "table_6_2.csv", index=False)
    lines = ["**Table 6.2. Image-based identification, whole record, 1200 epochs.**", "",
             "| Process | Records | Two-state NODE, median (range) | Without thermal state, "
             "median (range) |", "|---|---:|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['process']} | {int(r['records'])} | "
                     f"{r['R2_A_median']:.3f} ({r['R2_A_min']:.3f}\u2013{r['R2_A_max']:.3f}) | "
                     f"{r['R2_B_median']:.3f} ({r['R2_B_min']:.3f}\u2013{r['R2_B_max']:.3f}) |")
    lines += ["", "$R^2$ of the whole free-rollout trajectory, computed on the state after a 2 s "
              "moving average, in each record's own standardization."]
    (HERE / "table_6_2.md").write_text("\n".join(lines) + "\n")
    print("=== Table 6.2 ===")
    print(t.to_string(index=False))
    print("\n=== 新旧对照（变体 A，平滑状态 R²）===")
    a = sel[sel.variant == "A_m1"].sort_values(["process", "delta_vs_old"])
    print(a[["process", "record", "old_R2", "r2_smoothed", "delta_vs_old", "tau1_s"]]
          .to_string(index=False))
    for p in PROC:
        s = a[a.process == p]
        print(f"{p}: 旧 中位 {s.old_R2.median():.3f}  新 中位 {s.r2_smoothed.median():.3f}  "
              f"变化 {s.delta_vs_old.median():+.3f}  赢 {int((s.delta_vs_old>0).sum())}/{len(s)}")


if __name__ == "__main__":
    main()
