"""Rebuild only the failed training-fold ranking, preserving the original additivity check."""
import msvcrt
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_steel396_shap_pair as run


def recover():
    np = run.np
    out = run.BASE / "formal"
    folder = out / "fold-2"
    destination = folder / "inner-1" / "shap.json"
    protocol = run.read(out / "protocol.json")
    fp = run.nm.digest(protocol)
    archive = run.INPUT / "inputs.npz"
    assert run.source_hashes() == protocol["sources"]
    assert run.nm.file_hash(archive) == protocol["input_sha256"]
    assert run.read(run.BASE / "smoke/audit.json")["passed"]
    original_importance = run.importance

    def consistent_importance(x, predict, seed, smoke):
        # KernelExplainer evaluates each explained point alone. Use the same
        # 128-row inference shape as the unchanged background/coalition path.
        def explain_predict(z):
            if len(z) == 1:
                return predict(np.repeat(z, 128, axis=0))[:1]
            return predict(z)

        background = np.asarray(x, dtype=np.float64)[
            np.random.RandomState(seed).choice(len(x), min(len(x), 300), replace=False)]
        reference = predict(background)
        as_rows = np.concatenate([explain_predict(row[None, :]) for row in background])
        disagreement = float(np.max(np.abs(reference - as_rows)))
        print("Max difference against original batched outputs:", disagreement, flush=True)
        if not np.isfinite(disagreement) or disagreement >= 0.001:
            raise ValueError("Batch-consistent inference preflight failed")
        # For the final check predict(background) is called without alteration.
        scores, rows, settings = original_importance(x, explain_predict, seed, smoke)
        settings["inference_recovery"] = dict(
            helper_sha256=run.nm.file_hash(Path(__file__)),
            method="Repeat a single explanation row to an inference batch of 128; take its first prediction. Multirow predictions unchanged.",
            original_check_threshold_mpa=0.001,
            original_failure_residual_mpa=0.0013427734375,
            max_replayed_single_vs_original_batch_mpa=disagreement)
        return scores, rows, settings

    with (out / "run.lock").open("r+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        assert not destination.exists()
        previous = run.read(out / "status.json")
        assert previous["state"] == "failed" and "0.0013427734375" in previous["traceback"]
        with np.load(archive, allow_pickle=False) as z:
            tr = z["train_positions_2"]
            frame = run.pd.DataFrame(z["X"][tr], index=z["sample_ids"][tr], columns=z["columns"])
            train = run.make_data(frame, z["y"][tr])
        train.optimal_features = list(frame)
        pairs = run.fresh_pairs(train, 5)
        inner = pairs[1][0]
        native = run.read(folder / "inner-1/native.json")
        assert native["fp"] == fp and native["train_ids"] == run.nm.sample_ids(inner)
        selected = run.read(folder / "native/selection.json")
        assert selected["fp"] == fp
        run.importance = consistent_importance
        try:
            with TemporaryDirectory(dir=out) as tmp:
                ranking = run.get_ranking(Path(tmp), 1, inner, selected["members"], fp, 461427, False)
                assert ranking["train_ids"] == native["train_ids"]
                assert ranking["settings"]["max_additivity_residual"] < 0.001
                assert len(ranking["subsets"]["top50"]) == 50
                print("Accepted residual:", ranking["settings"]["max_additivity_residual"], flush=True)
                (Path(tmp) / "inner-1/shap.json").replace(destination)
        finally:
            run.importance = original_importance


if __name__ == "__main__":
    recover()
