param(
    [string]$Python = "$env:USERPROFILE\.conda\envs\modnet-v012-matbench\python.exe",
    [string]$BaselineDir = 'benchmarks/official-aligned-20261021',
    [string]$OutputDir = 'benchmarks/hybrid-nested-20261022',
    [int]$Jobs = 6
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$outputPath = [IO.Path]::GetFullPath((Join-Path $workspace $OutputDir))
$baselinePath = [IO.Path]::GetFullPath((Join-Path $workspace $BaselineDir))
$launcher = Join-Path $PSScriptRoot 'launch_modnet_hybrid_detached.py'
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $baselinePath 'matched-verification.json'))) {
    throw 'The completed official baseline is required.'
}
$statePath = Join-Path $outputPath 'hybrid-status.json'
if (Test-Path -LiteralPath $statePath) {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.state -eq 'complete') { throw 'This experiment has already completed.' }
    if ($state.pid) {
        $active = Get-CimInstance Win32_Process -Filter "ProcessId=$($state.pid)"
        if ($active -and $active.CommandLine -like '*run_modnet_hybrid_nested.py*') {
            throw "Already active with PID $($state.pid)."
        }
    }
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$launchPath = Join-Path $outputPath "launch-$stamp.json"
$commandLine = ('"{0}" -u "{1}" --baseline-dir "{2}" --output-dir "{3}" --launch-record "{4}" --n-jobs {5}' -f `
    $Python, $launcher, $baselinePath, $outputPath, $launchPath, $Jobs)
# Windows owns the short launcher; the training job has its own direct log-file
# handles and no console. It does not depend on this PowerShell session's pipes.
# https://learn.microsoft.com/windows/win32/procthread/job-objects
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly `
    -Property @{ShowWindow=[uint16]0}
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine=$commandLine; CurrentDirectory=$workspace; ProcessStartupInformation=$startup
}
if ($created.ReturnValue -ne 0) { throw "Windows process creation failed: $($created.ReturnValue)" }
for ($attempt=0; $attempt -lt 50 -and -not (Test-Path -LiteralPath $launchPath); $attempt++) {
    Start-Sleep -Milliseconds 200
}
if (-not (Test-Path -LiteralPath $launchPath)) { throw "Launcher produced no record; launcher PID=$($created.ProcessId)" }
$launch = Get-Content -LiteralPath $launchPath -Raw | ConvertFrom-Json
if ($launch.state -eq 'launcher_failed') { throw $launch.traceback }
if (-not (Get-Process -Id $launch.pid -ErrorAction SilentlyContinue)) { throw 'Training process exited during launch; inspect its stderr log.' }
Write-Output "Started round-two inner-validation search. PID=$($launch.pid)"
Write-Output "Launch record: $launchPath"
Write-Output "State: $statePath"
