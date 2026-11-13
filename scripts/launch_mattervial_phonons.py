"""Detach the audited phonons run with direct log files on Windows."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output, record = args.output.resolve(), args.record.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if record.parent != output or record.exists():
        raise ValueError("A fresh launch record inside the output directory is required.")
    stdout_path = output / (record.stem + ".stdout.log")
    stderr_path = output / (record.stem + ".stderr.log")
    argv = [sys.executable, "-u", str(root / "scripts/run_mattervial_phonons.py"),
            "--output", str(output), "--n-jobs", str(args.n_jobs)]
    if args.smoke:
        argv.append("--smoke")
    with stdout_path.open("xb", buffering=0) as stdout, stderr_path.open("xb", buffering=0) as stderr:
        process = subprocess.Popen(argv, cwd=str(root), env=dict(os.environ, PYTHONIOENCODING="utf-8"),
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    record.write_text(json.dumps(dict(pid=process.pid, started_at=datetime.now(timezone.utc).isoformat(),
        arguments=argv, stdout=str(stdout_path), stderr=str(stderr_path), jobs=args.n_jobs,
        mode="hidden Windows process with direct log files"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
