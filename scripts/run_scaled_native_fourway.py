"""Versioned target-scale correction; native MODNet search/training unchanged."""
import run_native_fourway as engine
from scaled_native_spline import fit_spline
from run_native_fourway import (ROOT, read, digest, file_hash, write_json, Path,
    argparse, time, os, now, traceback, shutil, pickle)

PRIOR = ROOT/'benchmarks/native-fourway-20261029'
ORIGINAL_RUN_FIT = engine.run_fit


def run_fit(job):
    result = ORIGINAL_RUN_FIT(job)
    if job['variant'] == 'bspline' and 'target_scaling' not in result:
        with (Path(job['directory'])/'model.pkl').open('rb') as handle:
            model = pickle.load(handle)
        result['target_scaling'] = model.target_scaling
        assert result['target_scaling']['fit_ids'] == job['train_ids']
        write_json(Path(job['directory'])/'result.json', result)
    return result


# Applied at import in the parent and Windows spawned workers. Existing engine
# source and all prior protocol hashes remain unchanged.
engine.fit_spline = fit_spline
engine.run_fit = run_fit


def freeze(out, smoke):
    reference = out/'native-protocol-reference'
    reference.mkdir(parents=True, exist_ok=True)
    p = engine.freeze(reference, smoke)
    p['protocol_path'] = str(out/'protocol.json')
    p['kind'] = 'Native-protocol four-arm phonons with train-only B-spline target scaling'
    p['kan'] = p['kan'].replace('No target scaling.',
        'Train-only target mean/std; fixed nontrainable inverse transform in saved inference graph. '
        'Train-loss stopping delta 0.001 in physical units (0.001/std while fitting).')
    p['training_by_family'] = dict(modnet='Unmodified native raw-target training',
        bspline='Same native rules/ranges/budget, with train-only target standardization; see kan')
    p['reuse_baseline'] = not smoke
    p['new_fits'] = 32 if smoke else 15000
    p['reused_fits'] = 0 if smoke else 5000
    p['prior_run'] = str(PRIOR)
    p['prior_protocol_sha256'] = file_hash(PRIOR/'protocol.json')
    p['limitations'].append('Target-scale correction chosen after inner-only diagnostics on this dataset; '
        'not an independent confirmation. Aborted raw-target KAN and diagnostic fits excluded from '
        'the formal equal budget; their prior exploratory cost remains recorded separately.')
    p['limitations'] = [s.replace('full four-arm baseline is recomputed, not replaced by old34.013.',
        'completed equal-budget baseline33.9276 is reused with artifact verification.') for s in p['limitations']]
    p['reused_artifact_sha256'] = {}
    if not smoke:
        assert read(PRIOR/'baseline-quality-gate.json')['passed']
        for fold in range(5):
            for name in ('complete.json', 'outer-predictions.npz', 'frozen-selection.json', 'predictions.csv'):
                path = PRIOR/'mm_modnet'/f'fold-{fold}'/name
                p['reused_artifact_sha256'][str(path)] = file_hash(path)
    for name in ('run_scaled_native_fourway.py', 'scaled_native_spline.py', 'audit_scaled_native_fourway.py'):
        path = ROOT/'scripts'/name
        p['source_sha256'][str(path)] = file_hash(path)
    path = out/'protocol.json'
    if path.exists():
        assert digest(read(path)) == digest(p), 'Frozen protocol changed; use a new output'
    write_json(path, p)
    for source, sha in p['source_sha256'].items():
        dest = out/'source-snapshot'/(Path(source).parent.name+'--'+Path(source).name)
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            shutil.copyfile(source, dest)
        assert file_hash(dest) == sha
    return p


def import_baseline(out, p):
    if not p['reuse_baseline']:
        return
    from audit_scaled_native_fourway import verify_prior
    verify_prior(p)
    for fold in p['folds']:
        src = PRIOR/'mm_modnet'/f'fold-{fold}'
        dst = out/'mm_modnet'/f'fold-{fold}'
        dst.mkdir(parents=True, exist_ok=True)
        for name in ('outer-predictions.npz', 'frozen-selection.json', 'predictions.csv'):
            if not (dst/name).exists():
                shutil.copyfile(src/name, dst/name)
            assert file_hash(dst/name) == file_hash(src/name)
        result = read(src/'complete.json')
        result.update(source_protocol_fingerprint=result['protocol_fingerprint'],
            protocol_fingerprint=digest(p), reused_from=str(src))
        write_json(dst/'complete.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    def progress(state, **kw):
        write_json(out/'status.json', dict(state=state, pid=os.getpid(), updated_at=now(),
            wall_seconds=time.perf_counter()-start, **kw))
    try:
        progress('freezing'); p = freeze(out, args.smoke)
        progress('verifying_reused_baseline'); import_baseline(out, p)
        banks, y = engine.load_banks(out); engine.report(out, p)
        for arm in engine.ARMS:
            for fold in p['folds']:
                engine.run_fold(out, p, banks, y, arm, fold, args.n_jobs, progress)
                summary = engine.report(out, p)
            if arm == 'mm_modnet':
                mean = summary['means'][arm]
                passed = args.smoke or mean <= p['published_mae']*1.02
                write_json(out/'baseline-quality-gate.json', dict(passed=passed, mean_mae=mean,
                    reused=p['reuse_baseline'], maximum_mae=p['published_mae']*1.02))
                if not passed:
                    progress('baseline_quality_requires_investigation', complete=False); return
        from audit_scaled_native_fourway import audit
        audit(out, require_complete=True)
        progress('complete', complete=True, smoke=args.smoke)
    except BaseException:
        progress('failed', complete=False, traceback=traceback.format_exc()); raise


if __name__ == '__main__':
    main()
