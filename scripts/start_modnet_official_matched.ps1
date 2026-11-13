param(
    [string]$Python = "$env:USERPROFILE\.conda\envs\modnet-v012-matbench\python.exe",
    [string]$OutputDir = "benchmarks/official-aligned-20261021",
    [int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$outputPath = [IO.Path]::GetFullPath((Join-Path $workspace $OutputDir))
$cachePath = Join-Path $outputPath 'precomputed'
$referencePath = Join-Path $outputPath 'reference/results.json.gz'
$runner = Join-Path $PSScriptRoot 'run_modnet_official_matched.py'
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python: $Python" }
if (-not (Test-Path -LiteralPath $referencePath)) { throw 'Prepare the official reference first.' }
$existingStatus = Join-Path $outputPath 'matched-status.json'
if (Test-Path -LiteralPath $existingStatus) {
    $state = Get-Content -LiteralPath $existingStatus -Raw | ConvertFrom-Json
    if ($state.state -eq 'running' -and $state.pid) {
        $active = Get-CimInstance Win32_Process -Filter "ProcessId=$($state.pid)"
        if ($active -and $active.CommandLine -like '*run_modnet_official_matched.py*') {
            throw "This run is already active with PID $($state.pid)."
        }
    }
}
$arguments = @('-u', ('"{0}"' -f $runner), '--output-dir', ('"{0}"' -f $outputPath),
    '--cache-dir', ('"{0}"' -f $cachePath), '--reference', ('"{0}"' -f $referencePath),
    '--n-jobs', $Jobs)
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $workspace `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputPath 'matched.stdout.log') `
    -RedirectStandardError (Join-Path $outputPath 'matched.stderr.log')
@{pid=$process.Id; started_at=(Get-Date -Format o); python=$Python; arguments=$arguments;
  output_dir=$outputPath; jobs=$Jobs} | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $outputPath 'launch.json') -Encoding UTF8
Write-Output "Started official MODNet / matched KAN workflow. PID=$($process.Id)"
Write-Output "State: $existingStatus"
