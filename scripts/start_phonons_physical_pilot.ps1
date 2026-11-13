param(
    [string]$Python = "$env:USERPROFILE\.conda\envs\modnet-v012-matbench\python.exe",
    [string]$Baseline = 'benchmarks/official-aligned-20261021',
    [string]$Output = 'benchmarks/phonons-physical-pilot-20261023',
    [int]$Jobs = 2
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$outputPath = [IO.Path]::GetFullPath((Join-Path $workspace $Output))
$baselinePath = [IO.Path]::GetFullPath((Join-Path $workspace $Baseline))
$launcher = Join-Path $PSScriptRoot 'launch_phonons_physical_pilot.py'
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $baselinePath 'matched-verification.json'))) {
    throw 'Completed official baseline required.'
}
$statePath = Join-Path $outputPath 'pilot-status.json'
if (Test-Path -LiteralPath $statePath) {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.state -eq 'complete') { throw 'Pilot already complete.' }
    if ($state.pid) {
        $active = Get-CimInstance Win32_Process -Filter "ProcessId=$($state.pid)"
        if ($active -and $active.CommandLine -like '*run_phonons_physical_pilot.py*') {
            throw "Pilot already active: PID=$($state.pid)"
        }
    }
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$recordPath = Join-Path $outputPath "launch-$stamp.json"
$commandLine = ('"{0}" -u "{1}" --baseline "{2}" --output "{3}" --record "{4}" --n-jobs {5}' -f `
    $Python, $launcher, $baselinePath, $outputPath, $recordPath, $Jobs)
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine=$commandLine; CurrentDirectory=$workspace; ProcessStartupInformation=$startup
}
if ($created.ReturnValue -ne 0) { throw "Windows process creation failed: $($created.ReturnValue)" }
for ($attempt=0; $attempt -lt 50 -and -not (Test-Path -LiteralPath $recordPath); $attempt++) {
    Start-Sleep -Milliseconds 200
}
if (-not (Test-Path -LiteralPath $recordPath)) { throw "Launcher produced no record: PID=$($created.ProcessId)" }
$launch = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
if (-not (Get-Process -Id $launch.pid -ErrorAction SilentlyContinue)) { throw 'Pilot exited; inspect launch logs.' }
Write-Output "Started phonons training-only pilot. PID=$($launch.pid)"
Write-Output "Launch record: $recordPath"
Write-Output "State: $statePath"
