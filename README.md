# CGCNN PyG + KAN

## MODNet KAN Benchmark Quickstart

This repo includes a descriptor-based benchmark for replacing MODNet's dense
MLP hierarchy with KAN blocks while keeping the Matbench fold protocol. The
runner compares `mlp`, `fastkan`, and `spline`, reports validation/test metrics,
parameter counts before/after optional pruning, and can export sparse layerwise
formula summaries for KAN models.

Create the PyTorch/KAN environment:

```powershell
.\scripts\setup_conda_cuda.ps1 kan-cgcnn-cuda
conda activate kan-cgcnn-cuda
```

Fast smoke test:

```powershell
python scripts\benchmark_modnet_kan.py `
  --dataset matbench_phonons `
  --folds 0 `
  --models mlp fastkan `
  --featurizer-preset pymatgen-composition `
  --n-features 64 `
  --epochs 20 `
  --batch-size 64 `
  --device cuda `
  --output-dir benchmarks\modnet-kan-smoke
```

Formula-export smoke test:

```powershell
python scripts\benchmark_modnet_kan.py `
  --dataset matbench_phonons `
  --folds 0 `
  --models fastkan `
  --featurizer-preset pymatgen-composition `
  --n-features 64 `
  --epochs 20 `
  --batch-size 64 `
  --prune-kan-fraction 0.3 `
  --export-formulas `
  --formula-top-k 20 `
  --device cuda `
  --output-dir benchmarks\modnet-kan-formulas
```

For the stricter benchmark against official MODNet descriptors, create the
separate official MODNet environment and run the combined wrapper:

```powershell
conda env create -f environment-modnet-v012.yml
powershell -ExecutionPolicy Bypass -File scripts\run_modnet_kan_matbench.ps1 `
  -Tasks matbench_phonons `
  -RequireCuda
```

Benchmark outputs are intentionally ignored by Git via `benchmarks/`; upload the
source, environment files, scripts, tests, and README, not generated CSV/JSON or
feature caches.

This repository compares two CGCNN variants in PyTorch Geometric:

- `mlp`: the CGCNN convolution uses a standard MLP interaction block for
  `[x_i, x_j, edge_attr]`.
- `kan`: the same CGCNN convolution replaces that internal MLP with a smaller
  FastKAN block by default.

The ordinary `mlp` baseline keeps the original CGCNN-style Linear readout. The
`kan` variant uses KAN both inside CGCNN message passing and in the graph-level
readout head.

## CUDA Conda Setup

On a CUDA server, create a fresh conda environment:

```bash
bash scripts/setup_conda_cuda.sh kan-cgcnn-cuda
conda activate kan-cgcnn-cuda
```

On Windows PowerShell:

```powershell
.\scripts\setup_conda_cuda.ps1 -EnvName kan-cgcnn-cuda
conda activate kan-cgcnn-cuda
```

The setup uses `environment-cuda.yml`, installs `torch==2.11.0+cu128`, then
installs `matbench==0.6` with `--no-deps` because the published Matbench package
pins old Python-era dependencies that are not compatible with modern
Python/CUDA environments. The `cu128` wheel is important for newer GPUs such as
RTX 50-series cards.

Quick CUDA check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

## Matbench Benchmark

The real-structure benchmark script is
[`scripts/benchmark_matbench.py`](scripts/benchmark_matbench.py). It uses
Matbench official train/test folds and accepts any Matbench v0.1 regression task
whose input type is `structure`.

Recommended full 5-fold CUDA run on a small Matbench structure dataset:

```bash
bash scripts/run_matbench_5fold_cuda.sh
```

That defaults to:

- dataset: `matbench_phonons`
- folds: `0 1 2 3 4`
- epochs: `200`
- batch size: `64`
- atom hidden dim: `64`
- conv layers: `4`
- atom features: Matminer CGCNN atom feature table, dimension `92`
- MLP readout hidden dim: `32`
- KAN readout hidden dim: `8`
- edge Gaussian dim: `41`
- cutoff: `6 Angstrom`
- KAN implementation: `fastkan`
- KAN hidden dim: `16`
- KAN grid size: `3`
- device: `cuda`
- models: `mlp` and `kan`
- no train/test subsampling
- no extra validation split; each fold trains on Matbench's full
  `train_and_val` split and evaluates on Matbench's test split

Matbench itself does not define CGCNN or KAN model hyperparameters. It defines
the official datasets, targets, folds, and scoring protocol. The model settings
above are matched to the CGCNN-style checkpoint structure you provided:
`embedding 92 -> 64`, `4` graph convolutions, `fc_full (2*64+41) -> 2*64`,
`conv_to_fc 64 -> 32`, and `fc_out 32 -> 1`. All settings are recorded in each
result JSON.

There is no maximum-neighbor cap in this implementation. Structure graphs keep
all periodic neighbors within the `6 Angstrom` cutoff.

The default KAN implementation is FastKAN-style Gaussian RBF KAN. The older
in-repo B-spline KAN remains available with `--conv-kan-impl spline`.

### MODNet-Style KAN

MODNet is descriptor-based rather than graph-based: structures/compositions are
converted to tabular material features, a relevance-redundancy style selector
chooses the top descriptors on the training split, and a dense hierarchy maps
`features -> shared trunk -> property group -> property head -> output`.

This repo now includes a PyTorch version of that hierarchy with the hidden dense
blocks swapped for the in-repo KAN layers:

- model: `cgcnn_pyg_kan.modnet.MODNetKAN`
- features/preprocessing: `cgcnn_pyg_kan.modnet_features`
- Matbench runner: `scripts/benchmark_modnet_kan.py`

To reproduce the official Matbench MODNet baseline, use the separate
`modnet==0.1.12` environment. The official package depends on an older
TensorFlow / pymatgen / scikit-learn stack, so do not install it into the
PyTorch CUDA environment:

One-command run for official MODNet plus KAN/MLP on the exported official
descriptors:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_modnet_kan_matbench.ps1
```

The default task set is the Matbench tasks below 20,000 samples. It first runs
official MODNet in `modnet-v012-matbench`, then runs `mlp`, `fastkan`, and
`spline` in `kan-cgcnn-cuda` on the exported fold descriptors.

```powershell
conda env create -f environment-modnet-v012.yml
conda activate modnet-v012-matbench
python -u scripts\run_official_modnet_matbench.py `
  --task-set small `
  --n-jobs 4 `
  --export-feature-folds `
  --export-max-features 512 `
  --output-dir benchmarks\official-modnet-v012-small
```

This mirrors the MODNet v0.1.12 Matbench setup: `DeBreuck2020Featurizer` for
structure tasks, `CompositionOnlyFeaturizer` for composition tasks,
MODNet's own relevance-redundancy feature selection, `EnsembleMODNetModel`,
5-model bootstraps, and 5-fold nested `fit_preset` selection. It writes
`official-modnet-summary-<task>.csv` plus
`official_feature_folds/fold_<n>/train_features.csv.gz` and
`test_features.csv.gz`.

After that, run KAN on the official MODNet-selected descriptors from the normal
PyTorch environment:

```powershell
conda activate kan-cgcnn-cuda
python -u scripts\benchmark_modnet_kan.py `
  --dataset matbench_phonons `
  --folds 0 1 2 3 4 `
  --models mlp fastkan spline `
  --precomputed-feature-dir benchmarks\official-modnet-v012-small\matbench_phonons\official_feature_folds `
  --n-features 512 `
  --common-dims 512 `
  --group-dims 256 `
  --property-dims 64 `
  --target-dims 64 `
  --loss mae `
  --epochs 300 `
  --batch-size 64 `
  --prune-kan-fraction 0.5 `
  --export-formulas `
  --output-dir benchmarks\kan-on-official-modnet-features\matbench_phonons `
  --device cuda
```

Quick smoke run:

```powershell
python scripts/benchmark_modnet_kan.py `
  --dataset matbench_phonons `
  --folds 0 `
  --models kan mlp `
  --featurizer-preset pymatgen-composition `
  --n-features 64 `
  --epochs 20 `
  --batch-size 64 `
  --device cuda
```

Fuller descriptor run:

```powershell
python scripts/benchmark_modnet_kan.py `
  --dataset matbench_phonons `
  --folds 0 1 2 3 4 `
  --models kan mlp `
  --featurizer-preset auto `
  --n-features 256 `
  --common-dims 64 `
  --group-dims 32 `
  --property-dims 16 `
  --kan-impl fastkan `
  --kan-grid-size 5 `
  --epochs 200 `
  --batch-size 128 `
  --val-ratio 0.1 `
  --early-stopping-patience 30 `
  --lr 0.001 `
  --device cuda `
  --require-cuda
```

`--featurizer-preset auto` uses matminer composition features for composition
tasks and a light matminer structure preset for structure tasks. The script
forces matminer featurization to `--featurizer-jobs 1` by default because
Windows multiprocessing can be slow to spawn; increase it on Linux/CUDA servers
if feature generation is the bottleneck. If matminer featurization fails, the
runner falls back to a pure pymatgen descriptor set.

Parameter tuning plus Matbench-aligned final benchmark:

```powershell
python scripts/tune_modnet_kan.py `
  --dataset matbench_phonons `
  --model-families mlp fastkan spline `
  --tune-folds 0 1 `
  --final-folds 0 1 2 3 4 `
  --search-space random `
  --num-random-trials 8 `
  --tune-epochs 80 `
  --final-epochs 300 `
  --tune-train-size 512 `
  --featurizer-preset auto `
  --n-feature-candidates 128 256 280 512 `
  --common-dim-candidates 64 128 512 `
  --group-dim-candidates 32 64 128 `
  --property-dim-candidates 16 32 64 `
  --kan-grid-size-candidates 3 5 `
  --kan-spline-order-candidates 3 `
  --lr-candidates 0.0003 0.001 0.005 `
  --weight-decay-candidates 0 1e-6 1e-5 `
  --dropout-candidates 0 0.05 `
  --loss-candidates mae rmse `
  --prune-kan-fraction-candidates 0 `
  --target-scale none `
  --batch-size 64 `
  --val-ratio 0.1 `
  --early-stopping-patience 60 `
  --device cuda `
  --require-cuda
```

The tuner optimizes MLP, FastKAN, and B-spline KAN separately. `--metric auto`
selects validation MAE for regression tasks and validation ROC-AUC for
classification tasks; you can still force a specific `best_val_*` metric. For
regression, `--loss-candidates mae rmse` tunes whether the training objective is
MAE or RMSE-style MSE. Classification tasks use binary cross entropy and record
probabilities for Matbench scoring. For each family the tuner selects the best
trial on `--tune-folds`, then reruns that configuration on `--final-folds`. The
tuning phase skips the official Matbench test fold by default; use
`--evaluate-tune-test` only for diagnostics, not for model selection. Final
outputs include
`final-summary-<dataset>.csv`, `final-fold-results-<dataset>.csv`, per-family
MatbenchTask records, and `best_config.json`.

By default, KAN final selection enforces the parameter budget: FastKAN and
B-spline KAN candidates are selected only when their `effective_params_mean` is
below the selected MLP's effective parameter count. If no KAN trial satisfies
that budget, that KAN family is skipped in the final benchmark. Add
`--prune-kan-fraction-candidates 0 0.3 0.5` to test post-training global
magnitude pruning; the summaries record both raw `params_mean` and nonzero
`effective_params_mean`. Use `--allow-kan-larger-than-mlp` only for ablations
where parameter fairness is intentionally disabled.

The parameter columns are also written with explicit pruning names:
`params_before_prune_mean`, `params_after_prune_mean`, `params_pruned_mean`, and
`params_pruned_pct_mean`. For final KAN-family benchmarks, the tuner writes
sparse layerwise formula summaries by default under the family final benchmark
directory, for example `formula-matbench_phonons-fold0-fastkan.txt`. Use
`--formula-top-k` to control how many nonzero terms are shown per layer/output,
`--formula-top-k 0` to write all nonzero terms, `--formula-min-abs` to hide tiny
coefficients, or `--no-export-final-formulas` to disable the files.

Matbench-strict all-dataset run:

```powershell
python scripts/tune_modnet_all_matbench.py `
  --model-families mlp fastkan spline `
  --tune-folds 0 1 `
  --final-folds 0 1 2 3 4 `
  --search-space random `
  --num-random-trials 8 `
  --tune-epochs 80 `
  --final-epochs 300 `
  --tune-train-size 512 `
  --featurizer-preset auto `
  --n-feature-candidates 128 256 280 512 `
  --common-dim-candidates 64 128 512 `
  --group-dim-candidates 32 64 128 `
  --property-dim-candidates 16 32 64 `
  --kan-grid-size-candidates 3 5 `
  --kan-spline-order-candidates 3 `
  --lr-candidates 0.0003 0.001 0.005 `
  --weight-decay-candidates 0 1e-6 1e-5 `
  --dropout-candidates 0 0.05 `
  --loss-candidates mae rmse `
  --prune-kan-fraction-candidates 0 0.3 `
  --target-scale none `
  --batch-size 64 `
  --val-ratio 0.1 `
  --early-stopping-patience 60 `
  --device cuda `
  --require-cuda
```

This runner automatically includes Matbench v0.1 structure/composition tasks
with `n_samples <= 20000`, including the small classification tasks
`matbench_expt_is_metal` and `matbench_glass`. `matbench_mp_is_metal` is larger
than 20000 samples and is skipped unless you raise `--max-samples`. During
tuning, it follows the Matbench protocol by using only
`get_train_and_val_data()` plus an internal validation split; the official
holdout fold from `get_test_data()` is used only in the final benchmark stage
and predictions are written with `MatbenchTask.record()`. After each dataset it
prints a compact leaderboard and updates
`all-datasets-summary.csv` / `all-datasets-summary.json`.

To inspect the included datasets without launching the run:

```powershell
python scripts/tune_modnet_all_matbench.py --list-datasets
```

### Fair CGCNN vs KAN Comparison

Use the fair runner for the main conclusion. It keeps the graph representation
fixed across models by default: `--*-atom-features cgcnn` uses Matminer's
CGCNN atom feature table, and `--*-edge-features gaussian` uses the same
41-dimensional distance expansion for both branches.

The default fair profiles are:

- `cgcnn`: ordinary CGCNN, MLP message block, MLP readout.
- `kan-readout`: same input and same message block, but KAN readout.
- `kan-conv`: same input and same MLP readout, but KAN message block.
- `kan-full`: same input, KAN message block, KAN readout.

The earlier compact KAN input (`elemental + distance`) is now an optional
ablation via `--include-simple-kan`, not the main fair comparison.

```powershell
python scripts/benchmark_matbench_fair.py --dataset matbench_phonons --epochs 200 --batch-size 64 --hidden-dim 64 --head-hidden-dims 32 --kan-head-hidden-dims 8 --num-convs 4 --conv-kan-hidden-dim 16 --conv-kan-grid-size 3 --conv-kan-impl fastkan --edge-dim 41 --cutoff 6.0 --kan-lr 0.003 --kan-weight-decay 1e-5 --num-workers 0 --device cuda --require-cuda
```

This writes one subdirectory per profile plus `fair-summary-<dataset>.csv/json`
with MAE, RMSE, parameter count, training time, and forward latency.

### Why MLP Can Beat KAN

In this codebase, `mlp` is the original CGCNN edge gate: a single linear
projection from `[x_i, x_j, edge_attr]` to the gate/core vector. FastKAN expands
edge features into Gaussian RBF bases and applies a bottleneck KAN on every
directed neighbor edge. That makes KAN more expensive, and it is not guaranteed
to be more accurate on a small Matbench fold.

The previous `conv_kan_grid_size=8` default was also not parameter matched. In
the fair profile with shared `cgcnn + gaussian` inputs, the current parameter
counts are close enough for a useful comparison:

| profile | message block | readout | params |
| --- | --- | --- | ---: |
| `cgcnn` | Linear gate | MLP `64 -> 32 -> 1` | 96,641 |
| `kan-readout` | Linear gate | FastKAN `64 -> 8 -> 1` | 96,761 |
| `kan-conv` | FastKAN, grid `3`, hidden `16` | MLP `64 -> 32 -> 1` | 87,689 |
| `kan-full` | FastKAN, grid `3`, hidden `16` | FastKAN `64 -> 8 -> 1` | 87,809 |

If KAN underperforms, first try a parameter-matched run with grid size `3`, then
tune KAN optimization separately. Larger grids can improve flexibility, but they
usually cost much more wall time because the KAN block is evaluated per edge.

### Fast Hyperparameter Tuning

Use `scripts/tune_matbench_fair.py` to avoid exhaustive grid search. The default
strategy is random search plus successive halving across the KAN readout-only,
KAN conv-only, and full KAN profiles:

- sample a small set of KAN candidates for each profile;
- run cheap early rungs on multiple folds, not a single fixed fold;
- keep the best candidates by validation MAE while preserving at least one
  survivor per profile;
- confirm the survivors on all requested folds;
- benchmark the ordinary CGCNN baseline for direct comparison;
- optionally rerun the selected best profile from each family on the full
  Matbench train/test folds with `--run-final-benchmark`.

Recommended CUDA tuning run:

```powershell
python scripts\tune_matbench_fair.py `
  --dataset matbench_phonons `
  --folds 0 1 2 3 4 `
  --epochs 200 `
  --batch-size 64 `
  --hidden-dim 64 `
  --head-hidden-dims 32 `
  --num-convs 4 `
  --edge-dim 41 `
  --cutoff 6 `
  --search-space random `
  --strategy successive-halving `
  --num-random-trials 18 `
  --conv-kan-hidden-dim-candidates 8 16 24 `
  --conv-kan-grid-size-candidates 2 3 4 `
  --head-kan-hidden-dim-candidates 4 8 16 32 `
  --head-kan-grid-size-candidates 2 3 4 `
  --lrs 0.001 0.002 0.003 `
  --weight-decays 0 1e-5 `
  --rung-epochs 20 60 200 `
  --rung-fold-counts 2 3 5 `
  --rung-train-sizes 512 0 0 `
  --run-final-benchmark `
  --device cuda `
  --require-cuda `
  --num-workers 0
```

The tuning script writes `best_config.json`, `tuning-fold-results-*.csv`, and
`tuning-summary-*.csv` under `benchmarks/tune-fair-.../`. With
`--run-final-benchmark`, it also writes `final-benchmark-summary-*.csv/json`
after refitting the selected best `kan-readout`, `kan-conv`, and `kan-full`
profiles on the requested Matbench folds.

For a smaller but still cross-fold tuning pass:

```powershell
python scripts\tune_matbench_fair.py `
  --dataset matbench_phonons `
  --folds 0 1 2 3 4 `
  --epochs 50 `
  --batch-size 32 `
  --num-random-trials 9 `
  --conv-kan-hidden-dim-candidates 8 16 `
  --conv-kan-grid-size-candidates 2 3 `
  --head-kan-hidden-dim-candidates 4 8 16 `
  --head-kan-grid-size-candidates 2 3 `
  --rung-epochs 5 15 50 `
  --rung-fold-counts 2 3 5 `
  --rung-train-sizes 256 512 0 `
  --device cuda `
  --require-cuda
```

On Windows PowerShell:

```powershell
python scripts/benchmark_matbench_fair.py --dataset matbench_phonons --epochs 200 --batch-size 64 --hidden-dim 64 --head-hidden-dims 32 --kan-head-hidden-dims 8 --num-convs 4 --conv-kan-hidden-dim 16 --conv-kan-grid-size 3 --conv-kan-impl fastkan --edge-dim 41 --cutoff 6.0 --kan-lr 0.003 --kan-weight-decay 1e-5 --num-workers 0 --device cuda --require-cuda
```

If the machine is too laggy while the display GPU is training, use a lighter
explicit override:

```powershell
.\scripts\run_matbench_5fold_cuda.ps1 `
  -Dataset matbench_phonons `
  -Epochs 100 `
  -BatchSize 16 `
  -HiddenDim 24 `
  -NumConvs 2 `
  -ConvKanHiddenDim 4 `
  -ConvKanImpl fastkan `
  -EpochPauseSeconds 0.3 `
  -NumWorkers 0
```

Run a larger structure task, for example Materials Project formation energy:

```bash
DATASET=matbench_mp_e_form EPOCHS=100 BATCH_SIZE=128 bash scripts/run_matbench_5fold_cuda.sh
```

PowerShell equivalent:

```powershell
.\scripts\run_matbench_5fold_cuda.ps1 `
  -Dataset matbench_mp_e_form `
  -Epochs 100 `
  -BatchSize 64 `
  -NumWorkers 0
```

If GPU memory or preprocessing time is tight, use explicit real-data subsets:

```bash
python scripts/benchmark_matbench.py \
  --dataset matbench_mp_e_form \
  --fold 0 \
  --conv-nets mlp kan \
  --device cuda \
  --require-cuda \
  --train-size 20000 \
  --test-size 5000 \
  --epochs 50 \
  --batch-size 128 \
  --hidden-dim 64 \
  --head-hidden-dims 32 \
  --kan-head-hidden-dims 8 \
  --num-convs 4 \
  --conv-kan-impl fastkan \
  --conv-kan-hidden-dim 16 \
  --conv-kan-grid-size 3 \
  --kan-lr 0.003 \
  --kan-weight-decay 1e-5 \
  --edge-dim 41 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers
```

Each 5-fold run writes into a fresh directory like
`benchmarks/matbench_phonons-5fold-YYYYMMDD-HHMMSS/`. It contains one CSV/JSON
pair per fold plus:

- `summary-matbench_phonons.csv`
- `summary-matbench_phonons.json`

The summary reports 5-fold mean/std for MAE, RMSE, timing, and optimizer steps.
The per-fold JSON files include the Matbench dataset/fold, graph conversion
settings, target scaling, Torch version, CUDA availability, and CUDA device
name.

For `matbench_phonons` with `batch_size=64`, each fold has about `1012`
Matbench train samples, so the script trains each model for about `1600`
optimizer steps per fold, or about `8000` optimizer steps over all 5 folds.
Larger Matbench datasets scale with their fold train size.

If you want an internal validation split for early model selection, pass
`--val-ratio 0.1` directly to `scripts/benchmark_matbench.py`; the 5-fold runner
uses the Matbench fold split as-is.

## JARVIS-DFT + Frozen MACE Embeddings

For the MACE representation experiment, use a two-stage pipeline:

1. Cache frozen MACE descriptors for JARVIS crystals.
2. Train small readout heads on the cached embedding matrix.

This keeps the expensive MACE forward pass separate from the cheap Linear/MLP/KAN
head ablations.

Install the optional dependencies inside the CUDA environment:

```powershell
conda activate kan-cgcnn-cuda
pip install -r requirements-jarvis-mace.txt
```

Choose the MACE-MP encoder size with `--mace-model small`, `--mace-model
medium`, or `--mace-model large`. Start with `small`; `medium` and `large`
usually give stronger descriptors but make the full JARVIS cache slower and
heavier. The model name is included in the output filename, so caches for
different encoder sizes can coexist.

Run a small local smoke test first. This only checks the pipeline; it is not the
benchmark dataset:

```powershell
python scripts\prepare_jarvis_mace_embeddings.py `
  --dataset dft_3d `
  --target optb88vdw_bandgap `
  --mace-model small `
  --device cuda `
  --max-samples 200 `
  --max-atoms 0 `
  --output data\jarvis_mace\bandgap_200_smoke.npz

python scripts\benchmark_jarvis_mace_heads.py `
  --embeddings data\jarvis_mace\bandgap_200_smoke.npz `
  --heads ridge linear mlp kan rf `
  --split random `
  --epochs 50 `
  --patience 10 `
  --batch-size 64 `
  --kan-hidden-dim 32 `
  --kan-grid-size 3 `
  --device cuda `
  --output-dir benchmarks\jarvis-mace-smoke
```

Then scale up to the full JARVIS-DFT OPTB88-vdW band gap task. The current
JARVIS `dft_3d` release contains `optb88vdw_bandgap` labels for all 93,902
entries, so do not pass `--max-samples`:

```powershell
python scripts\prepare_jarvis_mace_embeddings.py `
  --dataset dft_3d `
  --target optb88vdw_bandgap `
  --mace-model small `
  --device cuda `
  --max-atoms 0 `
  --resume

python scripts\benchmark_jarvis_mace_heads.py `
  --embeddings data\jarvis_mace\dft_3d_optb88vdw_bandgap_mace-small_mean-std_all_allatoms.npz `
  --heads ridge linear mlp kan rf `
  --split formula `
  --epochs 200 `
  --patience 30 `
  --batch-size 256 `
  --kan-hidden-dim 32 `
  --kan-grid-size 3 `
  --device cuda `
  --output-dir benchmarks\jarvis-mace-bandgap
```

The default MLP head is `512 -> 128 -> 64 -> 1` for the MACE small descriptor
cache. The default KAN head is deliberately smaller than the first smoke-test
version: `512 -> 32 -> 1` with `grid_size=3`, which gives a parameter count
close to the MLP head instead of making KAN win by being much larger.

For a tuned comparison, run randomized hyperparameter search over MLP, FastKAN,
B-spline KAN, and RandomForest. Selection uses validation MAE; the test split is
only used for reporting the selected configurations:

```powershell
python scripts\tune_jarvis_mace_heads.py `
  --embeddings data\jarvis_mace\dft_3d_optb88vdw_bandgap_mace-small_mean-std_all_allatoms.npz `
  --heads mlp fastkan spline rf `
  --split formula `
  --num-trials-per-head 12 `
  --epoch-candidates 80 120 180 240 `
  --patience 20 `
  --batch-sizes 128 256 512 `
  --device cuda `
  --output-dir benchmarks\jarvis-mace-bandgap-tuning
```

If you want a cheaper tuning pass before the full run, add for example
`--tune-train-size 20000 --tune-val-size 5000 --epoch-candidates 40 80 120`.
The script will tune on those subsets, then refit the best configuration for
each head on the full train split unless `--no-refit-best` is passed.

Useful JARVIS target keys to start with:

| Property | Target key |
| --- | --- |
| Formation energy per atom | `formation_energy_peratom` |
| OPTB88-vdW band gap | `optb88vdw_bandgap` |
| Energy above hull | `ehull` |
| Magnetic moment | `magmom_oszicar` |
| Bulk modulus | `bulk_modulus_kv` |
| Shear modulus | `shear_modulus_gv` |
| MBJ band gap | `mbj_bandgap` |

The descriptor cache stores `X`, `y`, `ids`, `formulas`, `n_atoms`, and JSON
metadata in one `.npz` file. The head benchmark standardizes features and
targets using only the train split, then reports MAE/RMSE/R2 for each head. Use
`--split formula` for a stricter formula-disjoint split; use `--split random`
when you want a faster diagnostic comparison.

## Model Usage

```python
from cgcnn_pyg_kan import CGCNN

model = CGCNN(
    node_input_dim=92,
    edge_input_dim=41,
    hidden_dim=64,
    num_convs=4,
    head_hidden_dims=(8,),
    conv_net="kan",          # or "mlp"
    head_net="kan",
    conv_kan_impl="fastkan",
    conv_kan_hidden_dim=16,  # smaller KAN bottleneck
    conv_kan_grid_size=3,
)
```
