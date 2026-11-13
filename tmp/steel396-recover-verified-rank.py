"""Repair a confirmed single/batched SHAP endpoint mismatch on inner training data."""
import msvcrt
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_steel396_shap_pair as run


def recover():
    np = run.np
    formal = run.BASE / "formal"
    with (formal / "run.lock").open("r+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        previous = run.read(formal / "status.json")
        trace = previous.get("traceback", "")
        match = re.search(r"assert residual < 1e-3, residual\s+AssertionError: ([0-9.eE+-]+)\s*$", trace)
        if previous["state"] != "failed" or "steels_paper_shap_kernel.py" not in trace or not match:
            raise ValueError("Only a failed SHAP additivity check can be recovered")
        failed_residual = float(match[1])
        protocol = run.read(formal / "protocol.json")
        fp = run.nm.digest(protocol)
        archive = run.INPUT / "inputs.npz"
        assert run.source_hashes() == protocol["sources"]
        assert run.nm.file_hash(archive) == protocol["input_sha256"]
        assert run.read(run.BASE / "smoke/audit.json")["passed"]
        fold = next(f for f in range(5) if not (formal / f"fold-{f}/complete.json").exists())
        folder = formal / f"fold-{fold}"
        assert not (folder/"mass50").exists() and not (folder/"top50").exists()
        k = next(i for i in range(5) if not (folder / f"inner-{i}/shap.json").exists())
        destination = folder / f"inner-{k}/shap.json"
        print(f"Recovering outer {fold+1}, inner {k+1}", flush=True)
        with np.load(archive, allow_pickle=False) as z:
            tr = z[f"train_positions_{fold}"]
            frame = run.pd.DataFrame(z["X"][tr], index=z["sample_ids"][tr], columns=z["columns"])
            train = run.make_data(frame, z["y"][tr])
        train.optimal_features = list(frame)
        inner = run.fresh_pairs(train, 5)[k][0]
        native = run.read(folder / f"inner-{k}/native.json")
        selected = run.read(folder / "native/selection.json")
        assert selected["fp"] == native["fp"] == fp
        assert native["train_ids"] == run.nm.sample_ids(inner)
        original = run.importance

        def stable_importance(x, predict, seed, smoke):
            background = np.asarray(x, dtype=np.float64)[np.random.RandomState(seed).choice(
                len(x), min(len(x), 300), replace=False)]
            reference = predict(background)
            direct = np.concatenate([predict(row[None,:]) for row in background])
            disagreement = float(np.max(np.abs(direct - reference)))
            print("Recorded residual and reproduced model discrepancy:", failed_residual, disagreement, flush=True)
            assert disagreement >= 0.001 and abs(disagreement - failed_residual) < 1e-7

            def stable_predict(z):
                if len(z) == 1:
                    return predict(np.repeat(z, 128, axis=0))[:1]
                return predict(z)

            stable = np.concatenate([stable_predict(row[None,:]) for row in background])
            corrected = float(np.max(np.abs(stable - reference)))
            assert corrected < 0.001
            print("Batch-aligned model discrepancy:", corrected, flush=True)
            scores, rows, settings = original(x, stable_predict, seed, smoke)
            assert settings["max_additivity_residual"] < 0.001
            settings["inference_recovery"] = dict(
                helper_sha256=run.nm.file_hash(Path(__file__)),
                method="Repeat single explanation rows to the model inference batch size of 128. Preserve original multirow predictions, including the additivity reference.",
                original_failure_residual_mpa=failed_residual,
                reproduced_single_vs_batch_mpa=disagreement,
                max_replayed_single_vs_original_batch_mpa=corrected,
                original_check_threshold_mpa=0.001)
            return scores, rows, settings

        run.importance = stable_importance
        try:
            with TemporaryDirectory(dir=formal) as temp:
                seed = 260902 + fold*100000 + k + 524
                ranking = run.get_ranking(Path(temp), k, inner, selected["members"], fp, seed, False)
                assert ranking["train_ids"] == native["train_ids"] and len(ranking["subsets"]["top50"]) == 50
                assert not destination.exists()
                (Path(temp) / f"inner-{k}/shap.json").rename(destination)
                print("Accepted residual:", ranking["settings"]["max_additivity_residual"], flush=True)
        finally:
            run.importance = original


if __name__ == "__main__":
    recover()
