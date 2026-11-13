"""Resume the scale-corrected run with an explicit, separately frozen amendment."""
import run_scaled_native_fourway as scaled
import native_fourway_models as models
import relaxed_spline_policy as policy
from run_scaled_native_fourway import engine
from run_native_fourway import (ROOT, read, digest, file_hash, write_json, Path,
    argparse, time, os, now, traceback, shutil)

BASE_RUN_FIT = scaled.run_fit
CHECKED_AMENDMENTS = {}
models.matched_widths = policy.matched_widths
engine.matched_widths = policy.matched_widths


def check_amendment(out):
    path = Path(out)/'parameter-matching-amendment.json'
    key = str(path)
    if key not in CHECKED_AMENDMENTS:
        a = read(path)
        assert file_hash(Path(out)/'protocol.json') == a['base_protocol_sha256']
        for source, sha in a['source_sha256'].items():
            assert file_hash(source) == sha, source
        CHECKED_AMENDMENTS[key] = a
    return CHECKED_AMENDMENTS[key]


def run_fit(job):
    amendment = check_amendment(Path(job['protocol_path']).parent)
    path = Path(job['directory'])/'result.json'
    legacy = amendment['preserved_fit_results'].get(str(path))
    if legacy:
        assert file_hash(path) == legacy['result_sha256']
    result = BASE_RUN_FIT(job)
    if not legacy:
        fingerprint = digest(amendment)
        if 'amendment_fingerprint' in result:
            assert result['amendment_fingerprint'] == fingerprint
        else:
            result['amendment_fingerprint'] = fingerprint
            if job['variant'] == 'bspline':
                widths = tuple(x[0] for x in engine.neurons_from_genes(job['genes']))
                result['parameter_mapping'] = policy.describe(job['genes']['n_feat'], widths)
                assert result['parameters'] == result['parameter_mapping']['spline_parameters']
            write_json(path, result)
    return result


engine.run_fit = run_fit


def freeze_amendment(out):
    path = out/'parameter-matching-amendment.json'
    if path.exists():
        return check_amendment(out)
    p = read(out/'protocol.json')
    engine.check_sources(str(out/'protocol.json'), digest(p))
    sources = {str(ROOT/'scripts'/name):file_hash(ROOT/'scripts'/name) for name in
        ('relaxed_spline_policy.py','resume_relaxed_native_fourway.py','audit_relaxed_native_fourway.py')}
    preserved = {}
    for arm in p['arms']:
        for file in sorted((out/arm).glob('fold-*/fits/*/result.json')):
            r = read(file)
            assert file_hash(file.parent/'model.pkl') == r['model_sha256']
            if r['variant'] == 'bspline':
                widths = tuple(x[0] for x in engine.neurons_from_genes(r['genes']))
                desc = policy.describe(r['genes']['n_feat'], widths)
                assert not desc['fallback_used']
                assert r['spline_widths'] == desc['widths'] and r['parameters'] == desc['spline_parameters']
            preserved[str(file)] = dict(result_sha256=file_hash(file),model_sha256=r['model_sha256'])
    a = dict(kind='User-authorized symmetric parameter matching relaxation',created_at=now(),
        base_protocol_sha256=file_hash(out/'protocol.json'),base_protocol_fingerprint=digest(p),
        authorization='User accepts widening the parameter-count tolerance and both increases and decreases.',
        rule='Keep the exact original mapping whenever within5%. Otherwise select minimum absolute relative '
             'parameter error from the same integer-width candidate set; ties use ratio error then widths. '
             'Both increases and decreases are allowed; deviation alone never aborts the run.',
        invariants='Same splits, seeds, genes/search ranges, feature ranking, noSISSO, train-only target scaling, '
                   'training/early-stopping rules and1000fits per arm/fold. No score-dependent architecture choice.',
        source_sha256=sources,preserved_fit_results=preserved,
        interpretation='Equal search budget with approximate per-candidate parameter matching, not exact equal '
                       'parameter counts. Report actual parameter deviations and selected ensemble model sizes.',
        limitations='Protocol amended after three outer folds were observed, to fix an architecture feasibility '
                    'failure. Completed models are unchanged; all future candidates follow this same rule.')
    write_json(path,a)
    for source, sha in sources.items():
        dest=out/'amendment-source-snapshot'/Path(source).name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,dest)
        assert file_hash(dest)==sha
    CHECKED_AMENDMENTS[str(path)]=a
    return a


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n-jobs',type=int,default=4);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    if (out/'status.json').exists():
        previous=read(out/'status.json')
        write_json(out/f'resume-prior-status-{time.time_ns()}.json',previous)
    def progress(state,**kw):
        write_json(out/'status.json',dict(state=state,pid=os.getpid(),updated_at=now(),
            wall_seconds=time.perf_counter()-start,entrypoint='resume_relaxed_native_fourway.py',**kw))
    try:
        progress('freezing_amendment')
        p=scaled.freeze(out,args.smoke)
        amendment=freeze_amendment(out)
        progress('loading_data',preserved_fits=len(amendment['preserved_fit_results']))
        if not (out/'mm_modnet/fold-0/complete.json').exists():scaled.import_baseline(out,p)
        banks,y=engine.load_banks(out)
        for arm in engine.ARMS:
            for fold in p['folds']:
                progress('resuming_fold',arm=arm,fold=fold)
                engine.run_fold(out,p,banks,y,arm,fold,args.n_jobs,progress)
                summary=engine.report(out,p)
            if arm=='mm_modnet':
                assert args.smoke or summary['means'][arm]<=p['published_mae']*1.02
        from audit_relaxed_native_fourway import audit
        audit(out,require_complete=True)
        progress('complete',complete=True,smoke=args.smoke)
    except BaseException:
        progress('failed',complete=False,traceback=traceback.format_exc());raise


if __name__=='__main__':main()
