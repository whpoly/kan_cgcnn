"""Prepare the steel input matrix and official folds without fitting a model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "benchmarks/steels-paper-reproduction-20261102"
EVIDENCE = PAPER / "source-search-20261107"
OUTPUT = PAPER / "mm-roost396"
CSV = ROOT / "benchmarks/matbench-expansion-20261101/source/matbench_steels_featurizedMM2020Comp_mattervial.csv"
SPLITS = ROOT / "benchmarks/official-aligned-20261021/matbench_steels"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    groups = read_json(EVIDENCE / "feature-bank-candidate.json")
    assert sha256(CSV) == read_json(EVIDENCE / "mm-data-comparison.json")["csv_sha256"]
    frame = pd.read_csv(CSV, index_col="material_id")
    ids = np.array([f"mb-steels-{i:03d}" for i in range(1, 313)])
    assert frame.index.is_unique and set(frame.index) == set(ids)
    frame = frame.loc[ids]
    mm = groups["mm"]
    pools = [f"ROOST_{model}_LayerMaterialPooling_#{i:02d}"
             for model in ("mpgap", "oqmd_eform") for i in range(1, 65)]
    assert len(mm) == 268 and set(pools) == set(groups["pooling"])
    columns = mm + pools
    assert len(set(columns)) == 396 and "target" not in columns
    matrix = frame[columns].to_numpy(dtype=np.float64)
    assert np.isfinite(matrix[:, 268:]).all()
    assert np.all(np.ptp(matrix[:, 268:], axis=0) > 0)
    mm_missing = ~np.isfinite(matrix[:, :268])
    matrix[:, :268] = np.where(mm_missing, 0, matrix[:, :268])
    target = frame["target"].to_numpy(dtype=np.float64)
    assert matrix.shape == (312, 396) and np.isfinite(target).all()
    arrays = dict(X=matrix, y=target, columns=np.array(columns), sample_ids=ids)
    all_test_positions = []
    split_hashes = {}
    for fold in range(5):
        path = SPLITS / f"fold-{fold}/outer-split.json"
        split = read_json(path)
        train, test = split["train_positions"], split["test_positions"]
        assert sorted(train + test) == list(range(len(ids)))
        assert list(ids[test]) == split["official_test_ids"]
        assert [f"id{i}" for i in train] == split["train_ids"]
        assert [f"id{i}" for i in test] == split["test_ids"]
        arrays[f"train_positions_{fold}"] = np.array(train)
        arrays[f"test_positions_{fold}"] = np.array(test)
        all_test_positions.extend(test)
        split_hashes[str(fold)] = sha256(path)
    assert sorted(all_test_positions) == list(range(len(ids)))

    OUTPUT.mkdir(exist_ok=True)
    archive = OUTPUT / "inputs.npz"
    if not archive.exists():
        temporary = OUTPUT / "inputs.pending.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(archive)
    with np.load(archive, allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for name, array in arrays.items():
            np.testing.assert_array_equal(saved[name], array)
    manifest = dict(
        state="inputs_ready_awaiting_steel_shap_rule", model="MODNet", start_fold=0,
        samples=len(ids), feature_count=len(columns), mm_nonfinite_replaced=int(mm_missing.sum()),
        groups={"MatMiner": 268, "ROOST_mpgap_LayerMaterialPooling": 64,
                "ROOST_oqmd_eform_LayerMaterialPooling": 64},
        columns=columns, csv=str(CSV.relative_to(ROOT)), csv_sha256=sha256(CSV),
        archive=archive.name, archive_sha256=sha256(archive), split_sha256=split_hashes,
        thresholds_reported=[30, 50, 70, 90], threshold_definition=None,
        steel_shap_model_and_ranking_scope=None, restart_policy="fresh_five_folds",
    )
    pending = OUTPUT / "manifest.pending.json"
    pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pending.replace(OUTPUT / "manifest.json")
    print(f"{archive}: {matrix.shape}, 5 official folds verified, MM replacements={mm_missing.sum()}")


if __name__ == "__main__":
    main()
