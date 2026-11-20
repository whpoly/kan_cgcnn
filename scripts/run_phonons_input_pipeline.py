"""Run the input-budget smoke check and scan after the existing baseline completes."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def write(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(.05 * (attempt + 1), .5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-output', type=Path, required=True)
    parser.add_argument('--pipeline-output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    folder = args.pipeline_output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / 'run.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    script = Path(__file__).with_name('run_phonons_standardized_input_budget.py').resolve()
    base = [sys.executable, '-u', str(script), '--reference', str(args.reference.resolve())]
    stages = [
        ('waiting_for_baseline_then_smoke', base + ['--output', str(args.smoke_output.resolve()),
            '--workers', str(min(2, args.workers)), '--smoke', '--wait-for-baseline']),
        ('input_dimension_scan', base + ['--output', str(args.output.resolve()),
            '--workers', str(args.workers), '--smoke-audit', str(args.smoke_output.resolve())]),
    ]
    state = dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                 baseline=str(args.reference.resolve()), smoke=str(args.smoke_output.resolve()),
                 output=str(args.output.resolve()), workers=args.workers)
    try:
        for phase, command in stages:
            stdout, stderr = folder / (phase + '.stdout.log'), folder / (phase + '.stderr.log')
            with stdout.open('a', encoding='utf-8') as log, stderr.open('a', encoding='utf-8') as error:
                child = subprocess.Popen(command, stdout=log, stderr=error, cwd=str(script.parent.parent),
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                state.update(state='running', phase=phase, child_pid=child.pid, command=command,
                             stdout=str(stdout), stderr=str(stderr), updated_at=datetime.now(timezone.utc).isoformat())
                write(folder / 'status.json', state)
                code = child.wait()
            if code:
                raise RuntimeError(phase + ' failed with exit code ' + str(code))
        state.update(state='complete', complete=True, updated_at=datetime.now(timezone.utc).isoformat())
        write(folder / 'status.json', state)
    except BaseException as error:
        state.update(state='failed', complete=False, error=repr(error), updated_at=datetime.now(timezone.utc).isoformat())
        write(folder / 'status.json', state)
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
