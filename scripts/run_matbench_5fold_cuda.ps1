param(
    [string]$Dataset = "matbench_phonons",
    [int]$Epochs = 100,
    [int]$BatchSize = 64,
    [int]$NumWorkers = 0,
    [int]$LogEverySteps = 50,
    [double]$EpochPauseSeconds = 0.0,
    [int]$HiddenDim = 64,
    [int]$NumConvs = 4,
    [int]$ConvKanHiddenDim = 16,
    [string]$ConvKanImpl = "fastkan",
    [int]$ConvKanGridSize = 8,
    [int]$EdgeDim = 41,
    [string]$EnvName = "kan-cgcnn-cuda",
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
    conda run -n $EnvName python scripts/benchmark_matbench.py `
        --dataset $Dataset `
        --fold $fold `
        --conv-nets mlp kan `
        --device cuda `
        --require-cuda `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --hidden-dim $HiddenDim `
        --head-hidden-dims 32 `
        --num-convs $NumConvs `
        --conv-kan-impl $ConvKanImpl `
        --conv-kan-hidden-dim $ConvKanHiddenDim `
        --conv-kan-grid-size $ConvKanGridSize `
        --edge-dim $EdgeDim `
        --num-workers $NumWorkers `
        --log-every-steps $LogEverySteps `
        --epoch-pause-seconds $EpochPauseSeconds `
        --forward-iters 10 `
        --warmup-iters 2 `
        --output-dir $OutputDir `
        --pin-memory
}

conda run -n $EnvName python scripts/summarize_matbench_results.py `
    --input-dir $OutputDir `
    --dataset $Dataset `
    --expect-folds 5

Write-Host "5-fold outputs: $OutputDir"
