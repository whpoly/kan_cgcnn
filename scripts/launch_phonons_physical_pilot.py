"""Hidden Windows launcher with direct log files for the training-only pilot."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output, record = args.output.resolve(), args.record.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if record.parent != output or record.exists():
        raise ValueError("Use a fresh launch record within the pilot output directory.")
    stdout_path = output / (record.stem + ".stdout.log")
    stderr_path = output / (record.stem + ".stderr.log")
    argv = [sys.executable, "-u", str(root / "scripts" / "run_phonons_physical_pilot.py"),
            "--baseline", str(args.baseline.resolve()), "--output", str(output), "--n-jobs", str(args.n_jobs)]
    with stdout_path.open("xb", buffering=0) as stdout, stderr_path.open("xb", buffering=0) as stderr:
        process = subprocess.Popen(argv, cwd=str(root), env=dict(os.environ, PYTHONIOENCODING="utf-8"),
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    data = dict(pid=process.pid, started_at=datetime.now(timezone.utc).isoformat(),
                arguments=argv, stdout=str(stdout_path), stderr=str(stderr_path),
                mode="windows_managed_launcher_no_console_direct_log_files", jobs=args.n_jobs)
    tmp = record.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(record)


if __name__ == "__main__":
    main()
