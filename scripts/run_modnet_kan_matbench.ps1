param(
    [ValidateSet("small", "all")]
    [string]$TaskSet = "small",
    [string[]]$Tasks = @(),
    [string]$OfficialEnv = "modnet-v012-matbench",
    [string]$KanEnv = "kan-cgcnn-cuda",
    [string]$OfficialOutputDir = "",
    [string]$KanOutputRoot = "",
    [int]$NJobs = 4,
    [int]$ExportMaxFeatures = 512,
    [int]$NModels = 5,
    [int]$NestedFolds = 5,
    [switch]$FastOfficial,
    [switch]$SkipOfficial,
    [string[]]$Models = @("mlp", "fastkan", "spline"),
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [int]$NFeatures = 512,
    [int[]]$CommonDims = @(512),
    [int[]]$GroupDims = @(256),
    [int[]]$PropertyDims = @(64),
    [int[]]$TargetDims = @(64),
    [ValidateSet("auto", "mae", "rmse", "mse", "bce")]
    [string]$KanLoss = "auto",
    [int]$Epochs = 300,
    [int]$BatchSize = 64,
    [double]$PruneKanFraction = 0.5,
    [string]$Device = "cuda",
    [switch]$RequireCuda,
    [switch]$NoExportFormulas,
    [int]$FormulaTopK = 20,
    [double]$FormulaMinAbs = 0.0,
    [switch]$NoMatbenchRecords,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$allTasks = @(
    "matbench_dielectric",
    "matbench_expt_gap",
    "matbench_glass",
    "matbench_jdft2d",
    "matbench_elastic",
    "matbench_mp_e_form",
    "matbench_mp_gap",
    "matbench_mp_is_metal",
    "matbench_perovskites",
    "matbench_phonons",
    "matbench_steels"
)

$smallTasks = @(
    "matbench_dielectric",
    "matbench_expt_gap",
    "matbench_glass",
    "matbench_jdft2d",
    "matbench_elastic",
    "matbench_perovskites",
    "matbench_phonons",
    "matbench_steels"
)

if ($Tasks.Count -eq 0) {
    if ($TaskSet -eq "all") {
        $Tasks = $allTasks
    } else {
        $Tasks = $smallTasks
    }
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $OfficialOutputDir) {
    $OfficialOutputDir = "benchmarks/official-modnet-v012-$TaskSet-$stamp"
}
if (-not $KanOutputRoot) {
    $KanOutputRoot = "benchmarks/kan-on-official-modnet-features-$TaskSet-$stamp"
}

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$CommandArgs
    )
    Write-Host ""
    Write-Host "=== $Title ==="
    Write-Host ("conda " + ($CommandArgs -join " "))
    if ($DryRun) {
        return
    }
    & conda @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title"
    }
}

New-Item -ItemType Directory -Force -Path $KanOutputRoot | Out-Null

$metadata = [ordered]@{
    task_set = $TaskSet
    tasks = $Tasks
    official_env = $OfficialEnv
    kan_env = $KanEnv
    official_output_dir = $OfficialOutputDir
    kan_output_root = $KanOutputRoot
    official_fast = [bool]$FastOfficial
    official_n_models = $NModels
    official_nested_folds = $(if ($FastOfficial) { 0 } else { $NestedFolds })
    export_max_features = $ExportMaxFeatures
    kan_models = $Models
    folds = $Folds
    kan_n_features = $NFeatures
    kan_loss = $KanLoss
    kan_epochs = $Epochs
    kan_batch_size = $BatchSize
    kan_prune_fraction = $PruneKanFraction
    formula_top_k = $FormulaTopK
    formula_min_abs = $FormulaMinAbs
    started_at = (Get-Date).ToString("s")
}
$metadata | ConvertTo-Json -Depth 8 | Set-Content -Path (Join-Path $KanOutputRoot "combined-run-metadata.json") -Encoding UTF8

if (-not $SkipOfficial) {
    $officialArgs = @(
        "run", "--no-capture-output",
        "-n", $OfficialEnv,
        "python", "-u", "scripts/run_official_modnet_matbench.py",
        "--tasks"
    )
    $officialArgs += $Tasks
    $officialArgs += @(
        "--n-jobs", "$NJobs",
        "--hp-strategy", "fit_preset",
        "--random-state", "7",
        "--nested-folds", "$NestedFolds",
        "--n-models", "$NModels",
        "--skip-existing",
        "--export-feature-folds",
        "--export-max-features", "$ExportMaxFeatures",
        "--output-dir", $OfficialOutputDir
    )
    if ($FastOfficial) {
        $officialArgs += "--fast"
    }
    Invoke-Step -Title "Official MODNet benchmark and feature export" -CommandArgs $officialArgs
} else {
    Write-Host "Skipping official MODNet. Using existing features under: $OfficialOutputDir"
}

foreach ($task in $Tasks) {
    $featureDir = Join-Path $OfficialOutputDir "$task\official_feature_folds"
    if (-not $DryRun -and -not (Test-Path $featureDir)) {
        throw "Missing official feature directory: $featureDir"
    }

    $taskOut = Join-Path $KanOutputRoot $task
    New-Item -ItemType Directory -Force -Path $taskOut | Out-Null

    $kanArgs = @(
        "run", "--no-capture-output",
        "-n", $KanEnv,
        "python", "-u", "scripts/benchmark_modnet_kan.py",
        "--dataset", $task,
        "--folds"
    )
    $kanArgs += ($Folds | ForEach-Object { "$_" })
    $kanArgs += "--models"
    $kanArgs += $Models
    $kanArgs += @(
        "--precomputed-feature-dir", $featureDir,
        "--n-features", "$NFeatures",
        "--common-dims"
    )
    $kanArgs += ($CommonDims | ForEach-Object { "$_" })
    $kanArgs += "--group-dims"
    $kanArgs += ($GroupDims | ForEach-Object { "$_" })
    $kanArgs += "--property-dims"
    $kanArgs += ($PropertyDims | ForEach-Object { "$_" })
    if ($TargetDims.Count -gt 0) {
        $kanArgs += "--target-dims"
        $kanArgs += ($TargetDims | ForEach-Object { "$_" })
    }
    $kanArgs += @(
        "--loss", $KanLoss,
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--prune-kan-fraction", "$PruneKanFraction",
        "--output-dir", $taskOut,
        "--device", $Device
    )
    if ($RequireCuda) {
        $kanArgs += "--require-cuda"
    }
    if (-not $NoExportFormulas) {
        $kanArgs += "--export-formulas"
        $kanArgs += @("--formula-top-k", "$FormulaTopK")
        $kanArgs += @("--formula-min-abs", "$FormulaMinAbs")
    }
    if ($NoMatbenchRecords) {
        $kanArgs += "--no-matbench-records"
    }

    Invoke-Step -Title "KAN/MLP on official MODNet features: $task" -CommandArgs $kanArgs
}

Write-Host ""
Write-Host "Combined run complete."
Write-Host "Official MODNet outputs: $OfficialOutputDir"
Write-Host "KAN outputs: $KanOutputRoot"
