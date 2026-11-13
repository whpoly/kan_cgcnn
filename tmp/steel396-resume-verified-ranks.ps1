$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$result = Join-Path $repo 'benchmarks/steel396-shap-pair-20261108'
$python = 'C:\Users\User\.conda\envs\modnet-v013-mattervial\python.exe'
$guard = [IO.File]::Open("$result/rank-recovery-monitor.lock",'OpenOrCreate','ReadWrite','None')
try {
    while ($true) {
        $status = Get-Content "$result/formal/status.json" -Raw | ConvertFrom-Json
        if ($status.state -ne 'failed' -or $status.traceback -notmatch 'assert residual < 1e-3, residual') { break }
        $oldPid = [int]$status.pid
        $live = Get-CimInstance Win32_Process | Where-Object {
            $_.ProcessId -eq $oldPid -or $_.ParentProcessId -eq $oldPid -or
            ($_.Name -match '^python' -and $_.CommandLine -match 'run_steel396_shap_pair|steel396-recover-verified-rank|steel396-shap-recover-fold')
        }
        if ($live) { throw "Found active process associated with the failed runner or its checkpoints" }
        $attempt = "recovery-$oldPid-$((Get-Date).ToString('yyyyMMdd-HHmmss'))"
        $folder = New-Item -Path "$result/$attempt" -ItemType Directory
        $record = [ordered]@{previous_pid=$oldPid; manager_pid=$PID; attempt=$attempt; phase='repairing'; started_at=(Get-Date).ToString('o')}
        $status | ConvertTo-Json | Set-Content "$folder/failed-status.json" -Encoding utf8
        $record | ConvertTo-Json | Set-Content "$result/rank-recovery-current.json" -Encoding utf8
        $repair = Start-Process $python -ArgumentList @('-u','tmp/steel396-recover-verified-rank.py') -WindowStyle Hidden -WorkingDirectory $repo -RedirectStandardOutput "$folder/repair.out" -RedirectStandardError "$folder/repair.err" -PassThru -Wait
        if ($repair.ExitCode -ne 0) { throw "Inner-fold numerical validation failed; see $attempt/repair.err" }
        $train = Start-Process $python -ArgumentList @('-u','scripts/run_steel396_shap_pair.py','sequence') -WindowStyle Hidden -WorkingDirectory $repo -RedirectStandardOutput "$folder/runner.out" -RedirectStandardError "$folder/runner.err" -PassThru
        $record.phase='running'
        $record.runner_pid=$train.Id
        $record.resumed_at=(Get-Date).ToString('o')
        $record | ConvertTo-Json | Set-Content "$folder/resume.json" -Encoding utf8
        $record | ConvertTo-Json | Set-Content "$result/rank-recovery-current.json" -Encoding utf8
        $train.WaitForExit()
        $latest = Get-Content "$result/formal/status.json" -Raw | ConvertFrom-Json
        if ($latest.pid -ne $train.Id) { throw 'New formal runner never took ownership' }
    }
} finally {
    $guard.Dispose()
}
