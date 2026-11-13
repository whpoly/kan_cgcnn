param([string]$Output = 'benchmarks/standalone-kan-confirmation-20261028', [int]$Jobs = 4,
      [ValidateSet('fold0','defer')][string]$OuterAction = 'fold0')
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $env:USERPROFILE '.conda/envs/modnet-v013-mattervial/python.exe'
$outputPath = [IO.Path]::GetFullPath((Join-Path $workspace $Output))
$launcher = Join-Path $PSScriptRoot 'launch_standalone_kan_confirmation.py'
$active = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -like '*run_standalone_kan_confirmation.py*' -and $_.CommandLine.Contains($outputPath)
}
if ($active) { throw "This study is already running: $($active.ProcessId)" }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$recordPath = Join-Path $outputPath "launch-$stamp.json"
$commandLine = ('"{0}" -u "{1}" --output "{2}" --record "{3}" --n-jobs {4} --outer-action {5}' -f $pythonPath,$launcher,$outputPath,$recordPath,$Jobs,$OuterAction)
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine=$commandLine;CurrentDirectory=$workspace;ProcessStartupInformation=$startup
}
if ($created.ReturnValue -ne 0) { throw "Process creation failed: $($created.ReturnValue)" }
for ($attempt=0; $attempt -lt 50 -and -not (Test-Path -LiteralPath $recordPath); $attempt++) { Start-Sleep -Milliseconds 200 }
if (-not (Test-Path -LiteralPath $recordPath)) { throw 'No launch record was produced.' }
$launch = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
if (-not (Get-Process -Id $launch.pid -ErrorAction SilentlyContinue)) { throw 'Runner exited; inspect logs.' }
Write-Output "Started frozen standalone KAN confirmation. PID=$($launch.pid)"
Write-Output "Launch record: $recordPath"
