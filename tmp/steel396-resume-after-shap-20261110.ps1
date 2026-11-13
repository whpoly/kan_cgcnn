$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$result = Join-Path $repo 'benchmarks/steel396-shap-pair-20261108'
$python = 'C:\Users\User\.conda\envs\modnet-v013-mattervial\python.exe'
if (Test-Path "$result/recovery-launch-20261110.json") { throw 'Recovery already launched' }
if (Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and ($_.CommandLine -match 'run_steel396_shap_pair|steel396-shap-recover' -or $_.ParentProcessId -eq 24796) }) { throw 'Steel process already exists' }
if (-not (Test-Path "$result/formal/fold-2/inner-1/shap.json")) {
    $repair = Start-Process -FilePath $python -ArgumentList @('-u','tmp/steel396-shap-recover-fold2-inner1.py') -WorkingDirectory $repo -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput "$result/recovery-shap.out" -RedirectStandardError "$result/recovery-shap.err"
    if ($repair.ExitCode -ne 0) { throw 'Ranking repair failed' }
}
$rank = Get-Content "$result/formal/fold-2/inner-1/shap.json" -Raw | ConvertFrom-Json
if ($null -eq $rank.settings.inference_recovery -or $rank.settings.max_additivity_residual -ge 0.001) { throw 'Ranking verification failed' }
$runner = Start-Process -FilePath $python -ArgumentList @('-u','scripts/run_steel396_shap_pair.py','sequence') -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput "$result/runner-resumed-20261110.out" -RedirectStandardError "$result/runner-resumed-20261110.err"
@{pid=$runner.Id; started_at=(Get-Date).ToString('o'); previous_pid=24796; entrypoint='scripts/run_steel396_shap_pair.py sequence'; stdout='runner-resumed-20261110.out'; stderr='runner-resumed-20261110.err'; repaired_rank='formal/fold-2/inner-1/shap.json'; residual_mpa=$rank.settings.max_additivity_residual} | ConvertTo-Json | Set-Content "$result/recovery-launch-20261110.json" -Encoding utf8
