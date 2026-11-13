"""Independently re-score the public MODNet artifact using the Matbench API."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from matbench.task import MatbenchTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.reference, "rt") as stream:
        reference = json.load(stream)
    report = dict(
        source="https://github.com/materialsproject/matbench/tree/main/benchmarks/matbench_v0.1_modnet_v0.1.12",
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        kind="published_prediction_rescoring_only", new_training_completed=False, tasks={})
    for name in ("steels", "phonons", "dielectric"):
        task = MatbenchTask("matbench_" + name)
        rows, seen = [], []
        for fold in task.folds:
            _, target = task.get_test_data(fold, include_target=True)
            record = reference["tasks"][task.dataset_name]["results"][f"fold_{fold}"]
            assert set(record["data"]) == set(target.index)
            predictions = np.array([record["data"][sid] for sid in target.index])
            mae = float(np.abs(target.values - predictions).mean())
            np.testing.assert_allclose(mae, record["scores"]["mae"], rtol=0, atol=1e-10)
            seen.extend(target.index)
            rows.append(dict(fold=int(fold), n_test=len(target), mae=mae))
        assert len(seen) == len(set(seen)) == len(task.df)
        result = dict(folds=rows, mean_mae=float(np.mean([r["mae"] for r in rows])),
                      each_sample_tested_once=True, official_ids_and_scores_match=True)
        report["tasks"][task.dataset_name] = result
        print(f"{task.dataset_name}: published MAE={result['mean_mae']:.10f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
