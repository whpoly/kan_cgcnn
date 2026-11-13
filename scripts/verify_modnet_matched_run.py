"""Audit paired members and summarize only genuinely completed outer folds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(root, allow_smoke=False):
    manifest = read(root / "matched-protocol.json")
    if manifest["smoke"] and not allow_smoke:
        raise ValueError("Debug/smoke results must not be reported as benchmark results.")
    rows = []
    for path in sorted(root.glob("matbench_*/fold-*/comparison.json")):
        fold_dir = path.parent
        result = read(path)
        split = read(fold_dir / "outer-split.json")
        outer_train, outer_test = set(split["train_ids"]), set(split["test_ids"])
        assert outer_train.isdisjoint(outer_test)
        selected = read(fold_dir / "frozen-ensemble.json")["members"]
        frame = pd.read_csv(fold_dir / "predictions.csv")
        assert frame.sample_id.tolist() == split["official_test_ids"]
        # Verify every matched configuration, not just the ensemble's MAE.
        for source in selected:
            member_name = Path(source["directory"]).name
            baseline = read(fold_dir / "modnet" / member_name / "result.json")
            train, val = set(baseline["train_ids"]), set(baseline["validation_ids"])
            assert train.isdisjoint(val)
            assert train | val == outer_train
            assert (train | val).isdisjoint(outer_test)
            for variant in ("kan_head", "kan_full"):
                member = read(fold_dir / variant / member_name / "result.json")
                for key in ("seed", "genes", "train_ids", "validation_ids",
                            "selected_features", "features", "num_neurons", "run_fingerprint"):
                    assert baseline[key] == member[key], (fold_dir, member_name, variant, key)
                assert member["parameters"] >= baseline["parameters"]
        for variant in ("modnet", "kan_head", "kan_full"):
            members = np.load(fold_dir / f"{variant}-member-predictions.npz")["predictions"]
            assert members.shape == (len(selected), len(frame))
            assert np.isfinite(members).all()
            prediction = members.mean(axis=0)
            np.testing.assert_allclose(prediction, frame[variant], rtol=1e-7, atol=1e-7)
            mae = float(np.mean(np.abs(frame.target.values - prediction)))
            np.testing.assert_allclose(mae, result["results"][variant]["mae"], rtol=0, atol=1e-9)
            rows.append(dict(task=result["task"], fold=result["fold"], variant=variant,
                             mae=mae, published_mae=result["published_modnet_mae"],
                             ensemble_size=len(selected),
                             parameters_mean=result["results"][variant]["parameters_mean"]))
    table = pd.DataFrame(rows)
    expected = {(t, f, v) for t in ("matbench_steels", "matbench_phonons", "matbench_dielectric")
                for f in range(5) for v in ("modnet", "kan_head", "kan_full")}
    observed = {(r["task"], r["fold"], r["variant"]) for r in rows}
    full = observed == expected and not manifest["smoke"]
    report = dict(verified_completed_rows=len(rows), complete_full_benchmark=full,
                  smoke=manifest["smoke"], matching_features_splits_genes_and_seeds=True if rows else None,
                  finite_predictions_and_recomputed_scores=True if rows else None)
    (root / "matched-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if rows:
        table.to_csv(root / "matched-fold-results.csv", index=False)
        if full:
            summary = table.groupby(["task", "variant"]).agg(
                mae=("mae", "mean"), fold_std=("mae", "std"),
                published_mae=("published_mae", "mean"), parameters=("parameters_mean", "mean"))
            summary.to_csv(root / "matched-summary.csv")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    verify(args.output_dir.resolve(), args.allow_smoke)
