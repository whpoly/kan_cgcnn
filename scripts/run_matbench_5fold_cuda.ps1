param(
    [string]$Dataset = "matbench_phonons",
    [int]$Epochs = 200,
    [int]$BatchSize = 64,
    [int]$NumWorkers = 0,
    [int]$LogEverySteps = 50,
    [double]$EpochPauseSeconds = 0.0,
    [int]$HiddenDim = 64,
    [int[]]$KanHeadHiddenDims = @(8),
    [string]$MlpHeadNet = "mlp",
    [string]$KanHeadNet = "kan",
    [int]$NumConvs = 4,
    [int]$ConvKanHiddenDim = 16,
    [string]$ConvKanImpl = "fastkan",
    [int]$ConvKanGridSize = 3,
    [int]$EdgeDim = 41,
    [double]$Cutoff = 6.0,
    [string]$AtomFeatures = "",
    [string]$EdgeFeatures = "",
    [string]$MlpAtomFeatures = "cgcnn",
    [string]$MlpEdgeFeatures = "gaussian",
    [string]$KanAtomFeatures = "cgcnn",
    [string]$KanEdgeFeatures = "gaussian",
    [double]$Lr = 3e-3,
    [double]$WeightDecay = 1e-5,
    [string]$MlpLr = "",
    [string]$KanLr = "0.003",
    [string]$MlpWeightDecay = "",
    [string]$KanWeightDecay = "0.0",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $OutputDir) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = "benchmarks/$Dataset-5fold-$stamp"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

for ($fold = 0; $fold -lt 5; $fold++) {
    Write-Host "=== $Dataset fold $fold/4, $Epochs epochs ==="
    $cmd = @(
        "python", "scripts/benchmark_matbench.py",
        "--dataset", $Dataset,
        "--fold", $fold,
        "--conv-nets", "mlp", "kan",
        "--device", "cuda",
        "--require-cuda",
        "--epochs", $Epochs,
        "--batch-size", $BatchSize,
        "--hidden-dim", $HiddenDim,
        "--head-hidden-dims", "32",
        "--kan-head-hidden-dims"
    )
    $cmd += $KanHeadHiddenDims
    $cmd += @(
        "--mlp-head-net", $MlpHeadNet,
        "--kan-head-net", $KanHeadNet,
        "--num-convs", $NumConvs,
        "--conv-kan-impl", $ConvKanImpl,
        "--conv-kan-hidden-dim", $ConvKanHiddenDim,
        "--conv-kan-grid-size", $ConvKanGridSize,
        "--edge-dim", $EdgeDim,
        "--cutoff", $Cutoff,
        "--mlp-atom-features", $MlpAtomFeatures,
        "--mlp-edge-features", $MlpEdgeFeatures,
        "--kan-atom-features", $KanAtomFeatures,
        "--kan-edge-features", $KanEdgeFeatures,
        "--lr", $Lr,
        "--weight-decay", $WeightDecay,
        "--num-workers", $NumWorkers,
        "--log-every-steps", $LogEverySteps,
        "--epoch-pause-seconds", $EpochPauseSeconds,
        "--forward-iters", "10",
        "--warmup-iters", "2",
        "--output-dir", $OutputDir,
        "--pin-memory"
    )
    if ($MlpLr) {
        $cmd += @("--mlp-lr", $MlpLr)
    }
    if ($AtomFeatures) {
        $cmd += @("--atom-features", $AtomFeatures)
    }
    if ($EdgeFeatures) {
        $cmd += @("--edge-features", $EdgeFeatures)
    }
    if ($KanLr) {
        $cmd += @("--kan-lr", $KanLr)
    }
    if ($MlpWeightDecay) {
        $cmd += @("--mlp-weight-decay", $MlpWeightDecay)
    }
    if ($KanWeightDecay) {
        $cmd += @("--kan-weight-decay", $KanWeightDecay)
    }
    $exe = $cmd[0]
    $cmdArgs = $cmd[1..($cmd.Count - 1)]
    & $exe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

python scripts/summarize_matbench_results.py `
    --input-dir $OutputDir `
    --dataset $Dataset `
    --expect-folds 5
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "5-fold outputs: $OutputDir"
