param(
    [string]$Python = "$env:USERPROFILE\.conda\envs\modnet-v013-mattervial\python.exe",
    [string]$Output = 'benchmarks/mattervial-phonons-20261023/run',
    [int]$Jobs = 2,
    [switch]$Smoke
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$outputPath = [IO.Path]::GetFullPath((Join-Path $workspace $Output))
$launcher = Join-Path $PSScriptRoot 'launch_mattervial_phonons.py'
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python: $Python" }
$statePath = Join-Path $outputPath 'status.json'
if (Test-Path -LiteralPath $statePath) {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.state -eq 'complete') { throw 'Study already complete.' }
}
$active = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -like '*run_mattervial_phonons.py*' -and $_.CommandLine.Contains($outputPath)
}
if ($active) { throw "This output already has a running process: $($active.ProcessId)" }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$recordPath = Join-Path $outputPath "launch-$stamp.json"
$commandLine = ('"{0}" -u "{1}" --output "{2}" --record "{3}" --n-jobs {4}' -f `
    $Python, $launcher, $outputPath, $recordPath, $Jobs)
if ($Smoke) { $commandLine += ' --smoke' }
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
if (-not (Get-Process -Id $launch.pid -ErrorAction SilentlyContinue)) { throw 'Runner exited; inspect logs.' }
Write-Output "Started MatterVial phonons. PID=$($launch.pid)"
Write-Output "Launch record: $recordPath"
Write-Output "State: $statePath"
