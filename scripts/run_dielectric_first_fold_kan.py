"""Run the queued dielectric MM+MV+KAN fold 0 early using its frozen cache."""
import gc
import msvcrt
import os
import traceback

import run_native_matbench_expansion as exp


OUT = exp.ROOT / "benchmarks/matbench-expansion-20261101/matbench_dielectric"
MONITOR = OUT / "priority-mm-mv-kan-fold-0"
ARM = "mm_mv_kan"
BASELINE = "mm_mv_modnet"


def progress(state, **fields):
    exp.write_json(MONITOR / "status.json", dict(state=state, pid=os.getpid(),
                   updated_at=exp.e.now(), **fields))


def compare(protocol, split):
    reference_ids = split["official_test_ids"]
    target = None
    scores = {}
    for arm in (BASELINE, ARM):
        folder = OUT / arm / "fold-0"
        result = exp.e.read(folder / "complete.json")
        assert result["fold"] == 0 and result["arm"] == arm
        assert result["protocol_fingerprint"] == exp.digest(protocol)
        assert result["ensemble_size"] == 50 and result["search_fits"] == 1000
        archive = folder / "outer-predictions.npz"
        assert exp.file_hash(archive) == result["prediction_sha256"]
        assert exp.file_hash(folder / "frozen-selection.json") == result["selection_sha256"]
        with exp.np.load(archive, allow_pickle=False) as z:
            exp.np.testing.assert_array_equal(z["sample_ids"], reference_ids)
            assert z["members"].shape == (50, len(reference_ids))
            exp.np.testing.assert_array_equal(z["prediction"], z["members"].mean(axis=0))
            if target is None:
                target = z["target"].copy()
            exp.np.testing.assert_array_equal(target, z["target"])
            scores[arm] = float(exp.np.mean(abs(target - z["prediction"])))
            assert abs(scores[arm] - result["mae"]) < 1e-10
    delta = scores[ARM] - scores[BASELINE]
    exp.write_json(MONITOR / "comparison.json", dict(fold=0, passed=True, mae=scores,
                   test_samples=len(reference_ids), kan_minus_modnet=delta,
                   relative_mae_reduction_percent=-100 * delta / scores[BASELINE]))


def run():
    p = exp.e.read(OUT / "protocol.json")
    assert p["task"] == "matbench_dielectric" and not p["smoke"]
    assert exp.e.read(OUT / "baseline-quality-gate.json")["passed"]
    exp.e.check_sources(p["protocol_path"], exp.digest(p))
    split_path = exp.OLD / p["task"] / "fold-0/outer-split.json"
    split = exp.e.read(split_path)
    assert exp.file_hash(split_path) == p["input_sha256"][str(split_path)]
    cache = OUT / "inputs/mm_mv/fold-0/training.pkl.gz"
    manifest = exp.e.read(cache.parent / "manifest.json")
    assert manifest["bank"] == "mm_mv" and manifest["fold"] == 0
    assert manifest["protocol_fingerprint"] == exp.digest(p)
    assert manifest["sha256"] == exp.file_hash(cache)
    assert manifest["train_ids"] == split["train_ids"]
    assert manifest["outer_test_ids"] == split["test_ids"]
    assert set(split["train_ids"]).isdisjoint(split["test_ids"])
    if not (OUT / ARM / "fold-0/complete.json").exists():
        progress("loading_outer_test_features", cache_sha256=manifest["sha256"])
        for path in (p["original"], p["csv"]):
            assert exp.file_hash(path) == p["input_sha256"][path]
        banks, targets = exp.load_banks(p, MONITOR)
        # The formal trainer reads all training rows/features from the saved cache.
        # Only held-out rows are needed in memory for its subsequent evaluation.
        test_bank = banks["mm_mv"].loc[split["test_ids"]].copy()
        test_targets = targets.loc[split["test_ids"]].copy()
        del banks, targets
        gc.collect()
        exp.e.run_fold(OUT, p, {"mm_mv": test_bank}, test_targets, ARM, 0, 2, progress)
    compare(p, split)
    progress("complete", arm=ARM, fold=0)


if __name__ == "__main__":
    MONITOR.mkdir(parents=True, exist_ok=True)
    with (MONITOR / "run.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            progress("verifying_cached_fold")
            run()
        except BaseException:
            progress("failed", traceback=traceback.format_exc())
            raise
