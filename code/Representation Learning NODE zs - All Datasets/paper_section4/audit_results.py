"""Read-only completeness and provenance audit for Section 4 results."""
from __future__ import annotations

import json
import math
from pathlib import Path

from experiment import RESULTS_ROOT, SPECS


def numeric_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from numeric_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from numeric_values(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def main() -> None:
    expected = []
    for dataset in SPECS:
        expected.extend((dataset, prior, 5) for prior in ("none", "rnn", "lstm", "transformer", "node"))
        expected.extend((dataset, "node", latent_dim) for latent_dim in (2, 3, 7))

    failures = []
    for dataset, prior, latent_dim in expected:
        run_dir = RESULTS_ROOT / dataset / f"{prior}_d{latent_dim}"
        paths = [run_dir / name for name in ("checkpoint.pt", "metrics.json", "metrics_per_sequence.csv",
                                             "evaluation_samples.npz")]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            failures.append(f"{run_dir}: missing {', '.join(missing)}")
            continue
        metrics = json.loads(paths[1].read_text())
        config = metrics.get("config", {})
        if (config.get("dataset"), config.get("prior"), config.get("latent_dim"), config.get("steps")) != (
            dataset, prior, latent_dim, 3000
        ):
            failures.append(f"{run_dir}: unexpected configuration {config}")
        rows = metrics.get("per_sequence", [])
        train_ids = [row["sequence_id"] for row in rows if row.get("split") == "train"]
        heldout_ids = [row["sequence_id"] for row in rows if row.get("split") == "heldout"]
        spec = SPECS[dataset]
        if train_ids != [spec.train_id] or tuple(heldout_ids) != spec.heldout_ids:
            failures.append(f"{run_dir}: split mismatch train={train_ids}, heldout={heldout_ids}")
        if not all(math.isfinite(number) for number in numeric_values(metrics)):
            failures.append(f"{run_dir}: non-finite metric")

    if failures:
        print("AUDIT FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"AUDIT PASSED: {len(expected)} paper runs are complete, finite, and split-consistent.")
    for dataset, spec in SPECS.items():
        print(f"- {dataset}: train={spec.train_id}; held out={', '.join(spec.heldout_ids)}")


if __name__ == "__main__":
    main()
