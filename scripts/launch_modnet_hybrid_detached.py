"""Launch the authorized training job with direct log files and no console.

Called through Windows process management by start_modnet_hybrid_nested.ps1.
This launcher performs no training and does not change the frozen protocol.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--launch-record", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = args.launch_record.resolve()
    if record.parent != output or record.exists():
        raise ValueError("Use a fresh launch record inside the output directory.")
    stamp = record.stem.removeprefix("launch-") if sys.version_info >= (3, 9) else record.stem[len("launch-"):]
    stdout_path, stderr_path = output / f"run-{stamp}.stdout.log", output / f"run-{stamp}.stderr.log"
    argv = [sys.executable, "-u", str(root / "scripts" / "run_modnet_hybrid_nested.py"),
            "--baseline-dir", str(args.baseline_dir.resolve()), "--output-dir", str(output),
            "--n-jobs", str(args.n_jobs)]
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        with stdout_path.open("xb", buffering=0) as stdout, stderr_path.open("xb", buffering=0) as stderr:
            process = subprocess.Popen(
                argv, cwd=str(root), env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, close_fds=True,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        result = dict(pid=process.pid, started_at=datetime.now(timezone.utc).isoformat(),
                      python=sys.executable, arguments=argv[1:], output_dir=str(output), jobs=args.n_jobs,
                      mode="windows_managed_launcher_no_console_direct_log_files",
                      stdout=str(stdout_path), stderr=str(stderr_path))
    except Exception:
        result = dict(state="launcher_failed", traceback=traceback.format_exc())
        record.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise
    temporary = record.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(record)


if __name__ == "__main__":
    main()
