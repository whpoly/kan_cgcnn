"""Run standardized steels first, then dielectric, with audited stage handoffs."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('matbench_steels', 'matbench_dielectric')
DATE = '20261116'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    for attempt in range(20):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(.05 * (attempt + 1), .5))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(.05 * (attempt + 1), .5))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stages(workers):
    pair = ROOT / 'scripts/run_standardized_matbench_pair.py'
    scan = ROOT / 'scripts/run_standardized_matbench_input_budget.py'
    result = []
    for task in TASKS:
        short = task[len('matbench_'):]
        baseline = ROOT / 'benchmarks' / f'{short}-standardized-modnet012-{DATE}'
        smoke = ROOT / 'benchmarks' / f'{short}-standardized-modnet012-smoke-{DATE}'
        reduced = ROOT / 'benchmarks' / f'{short}-standardized-input-budget-{DATE}'
        reduced_smoke = ROOT / 'benchmarks' / f'{short}-standardized-input-smoke-{DATE}'
        pair_cmd = [sys.executable, '-u', str(pair), '--task', task]
        scan_cmd = [sys.executable, '-u', str(scan), '--reference', str(baseline)]
        specifications = [
            ('baseline_smoke', smoke, True, pair_cmd + ['--output', str(smoke), '--smoke', '--n-jobs', str(min(2, workers))]),
            ('baseline', baseline, False, pair_cmd + ['--output', str(baseline), '--smoke-audit', str(smoke), '--n-jobs', str(workers)]),
            ('input_smoke', reduced_smoke, True, scan_cmd + ['--output', str(reduced_smoke), '--smoke', '--workers', str(min(2, workers))]),
            ('input_scan', reduced, False, scan_cmd + ['--output', str(reduced), '--smoke-audit', str(reduced_smoke), '--workers', str(workers)]),
        ]
        result.extend(dict(task=task, phase=name, output=str(output), smoke=is_smoke, command=command)
                      for name, output, is_smoke, command in specifications)
    return result


def verify_completed(stage):
    out = Path(stage['output'])
    status_path, audit_path = out / 'status.json', out / 'artifact-audit.json'
    if not status_path.exists() or read(status_path).get('state') != 'complete':
        return False
    assert audit_path.exists(), 'Completed stage has no audit: ' + str(out)
    audit, protocol, summary = read(audit_path), read(out / 'protocol.json'), read(out / 'summary.json')
    assert audit['passed'] and audit['complete'] and audit['smoke'] == stage['smoke']
    assert summary['complete'] and protocol['task'] == stage['task']
    assert protocol['smoke'] == stage['smoke']
    for source, fingerprint in protocol['source_sha256'].items():
        assert sha(source) == fingerprint, 'Changed source: ' + source
    for key in ('input_sha256', 'reference_sha256'):
        for path, fingerprint in protocol.get(key, {}).items():
            assert sha(path) == fingerprint, 'Changed input/reference: ' + path
    if not stage['smoke']:
        expected = 312 if stage['task'] == 'matbench_steels' else 4764
        assert audit['outer_samples'] == expected
    return True


def make_plan(workers):
    scripts = ['run_standardized_matbench_sequence.py', 'run_standardized_matbench_pair.py',
               'run_standardized_matbench_input_budget.py']
    return dict(task_order=list(TASKS), workers=workers, stages=stages(workers),
        source_sha256={str(ROOT / 'scripts' / name): sha(ROOT / 'scripts' / name) for name in scripts},
        authorization='User requested steels rerun first, then dielectric; both follow completed standardized phonons.',
        features='Original MM only; same outer-training RR ranking for both arms.',
        target_scaling='Both arms: inner-training z-score, ddof=0; score in original target units.',
        ga=dict(size_pop=20, num_generations=10, nested=5, early_stopping=4, refit=False),
        training=dict(epochs=800, monitor='loss', patience=30, min_delta_physical=.001,
                      min_delta_normalized='.001 / inner_training_target_std', restore_best_weights=False),
        ensemble_members_per_arm_fold=50,
        stage_policy='A complete, passed artifact audit is required before each handoff; failure stops the sequence.',
        dimension_caps=dict(matbench_steels=[112, 96, 80, 64, 48, 32, 16],
                            matbench_dielectric=[192, 160, 128, 112, 96, 80, 64, 48, 32, 16]),
        dimension_policy='Freeze selected KAN hidden widths and training genes; cap input prefixes and retrain changed models. '
                         'Use inner validation for step-4 then step-1 refinement and strict/+5% choices; '
                         'record outer curves as descriptive comparisons only.',
        parameter_reporting='Per-member mean and full 50-member total; no promise of equal final independently selected ensembles.',
        historical_reuse='Verified data/split/ranking caches only; no historical fitted models.',
        limitations=['Modified MODNet with target scaling, not literal unstandardized published-score reproduction.',
                     'Outer-training feature ranking is shared across inner folds, matching the phonons/MODNet protocol.',
                     'Parameter savings against full-input MLP do not isolate architecture from input reduction.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert args.workers > 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = make_plan(args.workers)
    if (out / 'plan.json').exists():
        assert read(out / 'plan.json') == plan, 'Frozen sequence plan changed'
    else:
        write(out / 'plan.json', plan)
    if args.prepare_only:
        print(json.dumps(dict(plan=str(out / 'plan.json'), tasks=list(TASKS), formal_training_started=False)))
        return
    lock = out / 'run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    state = dict(pid=os.getpid(), started_at=now(), completed_stages=[], workers=args.workers)
    child = None
    try:
        for stage in plan['stages']:
            for source, fingerprint in plan['source_sha256'].items():
                assert sha(source) == fingerprint, 'Frozen sequence source changed'
            if not verify_completed(stage):
                prefix = stage['task'] + '--' + stage['phase']
                stdout, stderr = out / (prefix + '.stdout.log'), out / (prefix + '.stderr.log')
                with stdout.open('a', encoding='utf-8') as log, stderr.open('a', encoding='utf-8') as error:
                    child = subprocess.Popen(stage['command'], cwd=ROOT, stdout=log, stderr=error,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                    state.update(state='running', task=stage['task'], phase=stage['phase'],
                                 stage_output=stage['output'], child_pid=child.pid, command=stage['command'],
                                 stdout=str(stdout), stderr=str(stderr), updated_at=now())
                    write(out / 'status.json', state)
                    while True:
                        try:
                            code = child.wait(timeout=15)
                            break
                        except subprocess.TimeoutExpired:
                            state['updated_at'] = now()
                            write(out / 'status.json', state)
                    if code != 0:
                        raise RuntimeError(prefix + ' failed with exit code ' + str(code))
                assert verify_completed(stage), 'Child exited without a complete audited result'
            result = dict(task=stage['task'], phase=stage['phase'], output=stage['output'], verified_at=now())
            state['completed_stages'].append(result)
            if stage['phase'] in ('baseline', 'input_scan'):
                summary = read(Path(stage['output']) / 'summary.json')
                write(out / (stage['task'] + '--' + stage['phase'] + '--results.json'), summary)
            state.update(state='stage_complete', updated_at=now())
            write(out / 'status.json', state)
        state.update(state='complete', complete=True, updated_at=now())
        write(out / 'status.json', state)
    except BaseException:
        if child is not None and child.poll() is None:
            child.terminate()
        state.update(state='failed', complete=False, traceback=traceback.format_exc(), updated_at=now())
        write(out / 'status.json', state)
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
