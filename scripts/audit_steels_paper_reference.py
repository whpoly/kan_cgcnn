"""Check the published steel metrics and no-reranking controls; does not train."""
import csv
import gzip
import hashlib
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "benchmarks/steels-paper-reproduction-20261102"
ref = out / "reference"
steel = root / "benchmarks/matbench-expansion-20261101/matbench_steels"
raw = Path.home() / ".conda/envs/kan-cgcnn-cuda/Lib/site-packages/matminer/datasets/matbench_steels.json.gz"
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
with gzip.open(raw, "rt") as stream:
    data = json.load(stream)
assert data["columns"] == ["composition", "yield strength"]
targets = {f"mb-steels-{i+1:03d}": row[1] for i, row in zip(data["index"], data["data"])}
controls = {}
for path in ref.glob("predictions*.csv"):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(targets) == 312 and {x[""] for x in rows} == targets.keys()
    fold_mae = []
    for i in range(5):
        col = f"prediction_f{i}"
        test = [x for x in rows if x[col]]
        manifest = read(steel / f"inputs/mm/fold-{i}/manifest.json")
        ids = {f"mb-steels-{int(x[2:])+1:03d}" for x in manifest["outer_test_ids"]}
        assert {x[""] for x in test} == ids
        assert all(not x["target"] or abs(float(x["target"])-targets[x[""]]) < 1e-5 for x in test)
        fold_mae.append(statistics.mean(abs(float(x[col])-targets[x[""]]) for x in test))
    controls[path.name] = dict(folds=fold_mae, mean=statistics.mean(fold_mae))
files = sorted((ref/"results").glob("*fold*"), key=lambda p:int(p.stem.split("fold_")[1].split("_")[0]))
assert len(files) == 5
folds = [read(p)["mae"] for p in files]
score = statistics.mean(folds)
published = read(next((ref/"results").glob("*average*")))["target"]["average_mae"]
assert abs(score-published) < 1e-8
report = dict(source_commit="ccc65ab78daad3553468d393d4427936a0540b9f", table_mae=85.12,
              fold_files=[p.name for p in files], fold_mae=folds, fold_mean=score,
              published_summary=published, no_reranking_controls=controls,
              status="published metrics verified; independent retraining not verified",
              author_log_config=dict(population=50, generations=20, early_stopping=8, refit=False,
                                     threshold_runs=[30,50,70,90],serialized_version="0.1.13"),
              missing="Exact SHAP reranking implementation, threshold definition, and fold-specific rankings or training/validation selection records.",
              gate="No matched KAN until independently reproduced MODNet. Never choose thresholds from test scores.",
              sha256={str(p.relative_to(ref)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in ref.rglob("*") if p.is_file()})
(out/"reference-audit.json").write_text(json.dumps(report,indent=2), encoding="utf-8")
print(score, controls)
