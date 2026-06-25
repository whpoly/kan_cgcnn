# CGCNN PyG + KAN

This repository compares two CGCNN variants in PyTorch Geometric:

- `mlp`: the CGCNN convolution uses a standard MLP interaction block for
  `[x_i, x_j, edge_attr]`.
- `kan`: the same CGCNN convolution replaces that internal MLP with a smaller
  FastKAN block by default.

The graph-level readout is a normal MLP in both variants. The comparison is
therefore inside CGCNN message passing, not just at the final readout head.

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
- epochs: `100`
- batch size: `64`
- atom hidden dim: `64`
- conv layers: `4`
- readout hidden dim: `32`
- edge Gaussian dim: `41`
- cutoff: `6 Angstrom`
- KAN implementation: `fastkan`
- KAN hidden dim: `16`
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

Quick CUDA sanity benchmark for the current FastKAN configuration:

```powershell
conda run -n kan-cgcnn-cuda python scripts\benchmark_matbench.py `
  --dataset matbench_phonons `
  --fold 0 `
  --conv-nets mlp kan `
  --conv-kan-impl fastkan `
  --epochs 10 `
  --batch-size 64 `
  --hidden-dim 64 `
  --head-hidden-dims 32 `
  --num-convs 4 `
  --conv-kan-hidden-dim 16 `
  --conv-kan-grid-size 8 `
  --edge-dim 41 `
  --cutoff 6 `
  --device cuda `
  --require-cuda
```

| model | params | test MAE | test RMSE | train seconds | forward ms/batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 96,641 | 132.592 | 242.745 | 2.296 | 1.861 |
| FastKAN | 182,729 | 156.701 | 285.474 | 9.004 | 18.124 |

This is a 10-epoch smoke benchmark on fold 0, not the final 5-fold 100-epoch
result.

On Windows PowerShell:

```powershell
.\scripts\run_matbench_5fold_cuda.ps1 `
  -Dataset matbench_phonons `
  -Epochs 100 `
  -BatchSize 64 `
  -NumWorkers 0
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
  --num-convs 4 \
  --conv-kan-impl fastkan \
  --conv-kan-hidden-dim 16 \
  --conv-kan-grid-size 8 \
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

## Model Usage

```python
from cgcnn_pyg_kan import CGCNN

model = CGCNN(
    node_input_dim=92,
    edge_input_dim=41,
    hidden_dim=64,
    num_convs=4,
    head_hidden_dims=(32,),
    conv_net="kan",          # or "mlp"
    conv_kan_impl="fastkan",
    conv_kan_hidden_dim=16,  # smaller KAN bottleneck
    conv_kan_grid_size=8,
)
```
