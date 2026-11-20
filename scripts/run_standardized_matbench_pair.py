"""Task adapter for the audited, target-standardized phonons comparison.

Original phonons sources and artifacts are immutable. Only original MM inputs
are used. New task searches start from scratch, with the same native GA and
train-only target transforms in both families.
"""
import argparse
import gc
import importlib.metadata
import os
import pickle
import shutil
import time
import traceback
from pathlib import Path

import run_phonons_standardized_matbench as base
from run_modnet_official_matched import (
    np, pd, MODData, digest, file_hash, write_json, sample_ids,
    matbench_kfold_splits, official_ids,
)

ROOT, OLD = base.ROOT, base.OLD
REFERENCE = ROOT / 'benchmarks/phonons-standardized-modnet012-20261115'
ARMS, GA = base.ARMS, base.GA
TASKS = {
    'matbench_steels': dict(samples=312, target='yield_strength', units='MPa'),
    'matbench_dielectric': dict(samples=4764, target='n', units='unitless'),
}


def read(path):
    for attempt in range(20):
        try:
            return base.read(path)
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(.05 * (attempt + 1), .5))


def run_fit(job):
    """Use the original worker, then verify its serialized target transform."""
    result = dict(base.BASE_RUN_FIT(job), reused=False)
    directory = Path(job['directory'])
    with (directory / 'model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    result['target_scaling'] = model.target_scaling
    del model
    data = base.training_for(job)
    base.check_scale(result['target_scaling'], job, data)
    p = read(job['protocol_path'])
    result.update(task=p['task'], protocol_fingerprint=digest(p),
        history_units=dict(loss=p['units'], val_loss=p['units'],
                           mae='standardized', val_mae='standardized'))
    if job['variant'] == 'bspline':
        widths = tuple(x[0] for x in base.engine.neurons_from_genes(job['genes']))
        result['parameter_mapping'] = base.mapping.describe(job['genes']['n_feat'], widths)
        assert result['parameters'] == result['parameter_mapping']['spline_parameters']
    write_json(directory / 'result.json', result)
    return result


base.engine.run_fit = run_fit


def audit_inputs(task, out):
    original_path = OLD / 'precomputed' / (task + '_moddata.pkl.gz')
    identity = read(OLD / task / 'dataset-identity.json')
    assert file_hash(original_path) == identity['sha256']
    original = MODData.load(original_path)
    assert len(original.df_targets) == TASKS[task]['samples']
    assert list(original.target_names) == [TASKS[task]['target']]
    assert original.df_targets.index.is_unique and original.df_featurized.columns.is_unique
    assert np.isfinite(original.df_targets.values).all()
    published = read(OLD / task / 'published-reference-verified.json')
    all_ids = sample_ids(original)
    seen, rows, hashes = [], [], {str(original_path): file_hash(original_path)}
    hashes[str(OLD / task / 'dataset-identity.json')] = file_hash(OLD / task / 'dataset-identity.json')
    hashes[str(OLD / task / 'published-reference-verified.json')] = file_hash(OLD / task / 'published-reference-verified.json')
    for fold, (tr, te) in enumerate(matbench_kfold_splits(original)):
        folder = OLD / task / f'fold-{fold}'
        split_path, cache = folder / 'outer-split.json', folder / 'official-train-moddata.pkl.gz'
        split = read(split_path)
        sha = file_hash(cache)
        assert sha == split['training_data_sha256']
        training = MODData.load(cache)
        assert sample_ids(training) == split['train_ids'] == [all_ids[i] for i in tr]
        assert split['test_ids'] == [all_ids[i] for i in te]
        assert split['official_test_ids'] == official_ids(task, te, len(all_ids))
        assert set(split['official_test_ids']) == set(published['folds'][fold]['test_ids'])
        assert set(split['train_ids']).isdisjoint(split['test_ids'])
        assert set(split['train_ids'] + split['test_ids']) == set(all_ids)
        pd.testing.assert_frame_equal(training.df_targets, original.df_targets.loc[split['train_ids']])
        pd.testing.assert_frame_equal(training.df_featurized, original.df_featurized.loc[split['train_ids']])
        ranking = list(training.get_optimal_descriptors())
        assert ranking == split['selected_features'] and len(ranking) == len(set(ranking))
        assert set(ranking).issubset(original.df_featurized.columns)
        assert len(training.df_targets) < 5000, 'This adapter requires the complete native GA sample set.'
        seen.extend(split['official_test_ids'])
        hashes[str(cache)], hashes[str(split_path)] = sha, file_hash(split_path)
        rows.append(dict(fold=fold, train=len(tr), test=len(te), ranked_features=len(ranking),
                         training_cache=str(cache), training_sha256=sha))
        del training
        gc.collect()
    assert len(seen) == len(set(seen)) == len(all_ids)
    result = dict(passed=True, task=task, samples=len(all_ids), units=TASKS[task]['units'],
        original=str(original_path), mm_features=len(original.df_featurized.columns), folds=rows,
        input_sha256=hashes, original_values_verified=True, native_and_published_outer_ids_verified=True,
        feature_scope='Native outer-training RR ranking reused across inner folds; not fully nested feature selection.',
        model_reuse=False, verified_at=base.engine.now())
    write_json(out / 'input-audit.json', result)
    return original, result


def freeze(out, task, smoke, audit):
    ref = read(REFERENCE / 'protocol.json')
    sources = dict(ref['source_sha256'])
    for path, sha in sources.items():
        assert file_hash(path) == sha, path
    sources[str(Path(__file__).resolve())] = file_hash(__file__)
    packages = {name: importlib.metadata.version(name) for name in ref['packages']}
    assert packages == ref['packages']
    p = dict(kind='MM-MLP versus MM-KAN; task extension of standardized phonons',
        task=task, **TASKS[task], original=audit['original'], protocol_path=str(out / 'protocol.json'),
        smoke=smoke, arms=list(ARMS), folds=[0] if smoke else list(range(5)), seed=ref['seed'],
        ga=GA, fits_per_arm_fold=8 if smoke else 1000,
        fit_budget_meaning='Upper bound; upstream GA early stopping remains enabled for both arms',
        packages=packages, source_sha256=sources, input_sha256=audit['input_sha256'],
        reference_protocol=str(REFERENCE / 'protocol.json'),
        reference_protocol_sha256=file_hash(REFERENCE / 'protocol.json'),
        target_scaling=ref['target_scaling'], training=ref['training'],
        inference='Zero-parameter inverse target transform; report native and unrepaired MAE in target units',
        features=ref['features'], feature_selection_scope=audit['feature_scope'],
        architecture=ref['architecture'], search=ref['search'], selection=ref['selection'],
        model_reuse=False, task_order=['matbench_steels', 'matbench_dielectric'])
    path = out / 'protocol.json'
    if path.exists():
        assert digest(read(path)) == digest(p), 'Frozen protocol changed'
    else:
        write_json(path, p)
    for name, sha in sources.items():
        source = Path(name)
        dest = out / 'source-snapshot' / (source.parent.name + '--' + source.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(source, dest)
        assert file_hash(dest) == sha
    return p


def prepare(out, p, original, fold):
    folder = OLD / p['task'] / f'fold-{fold}'
    split, cache = read(folder / 'outer-split.json'), folder / 'official-train-moddata.pkl.gz'
    assert file_hash(cache) == p['input_sha256'][str(cache)]
    data = MODData.load(cache)
    assert sample_ids(data) == split['train_ids']
    assert list(data.get_optimal_descriptors()) == split['selected_features']
    if p['smoke']:
        data.optimal_features = data.optimal_features[:24]
        cache = out / 'smoke-training.pkl.gz'
        if not cache.exists():
            data.save(cache)
        data = MODData.load(cache)
    write_json(out / 'inputs' / f'fold-{fold}.json', dict(training_cache=str(cache),
        training_sha256=file_hash(cache), split=split, ranked_features=list(data.get_optimal_descriptors()),
        raw_target_min=float(data.df_targets.values.min()), raw_target_max=float(data.df_targets.values.max()),
        raw_targets_verified=True))
    return data, cache, split


base.common.prepare = prepare


def report(out, p):
    rows = [read(out / arm / f'fold-{fold}' / 'complete.json')
            for arm in ARMS for fold in p['folds'] if (out / arm / f'fold-{fold}' / 'complete.json').exists()]
    means = {arm: {key: float(np.mean([r[key] for r in rows if r['arm'] == arm]))
                  for key in ('mae', 'raw_mae', 'parameters_mean', 'parameters_sum')}
             for arm in ARMS if sum(r['arm'] == arm for r in rows) == len(p['folds'])}
    summary = dict(task=p['task'], units=p['units'], smoke=p['smoke'],
        complete=len(rows) == 2 * len(p['folds']), rows=rows, means=means,
        ga=p['ga'], target_scaling=p['target_scaling'], protocol_fingerprint=digest(p))
    if summary['complete']:
        expected = [sid for f in p['folds']
                    for sid in read(out / 'inputs' / f'fold-{f}.json')['split']['official_test_ids']]
        for arm in ARMS:
            ids = []
            for fold in p['folds']:
                folder = out / arm / f'fold-{fold}'
                rec = read(folder / 'complete.json')
                assert rec['protocol_fingerprint'] == digest(p)
                assert file_hash(folder / 'outer-predictions.npz') == rec['prediction_sha256']
                assert file_hash(folder / 'frozen-selection.json') == rec['selection_sha256']
                selection = read(folder / 'frozen-selection.json')
                assert selection['test_labels_used'] is False
                assert len(selection['members']) == (8 if p['smoke'] else 50)
                for member in selection['members']:
                    saved = read(Path(member['directory']) / 'result.json')
                    assert file_hash(Path(member['directory']) / 'model.pkl') == saved['model_sha256']
                with np.load(folder / 'outer-predictions.npz') as z:
                    ids.extend(z['sample_ids'].tolist())
                    for prefix in ('', 'raw_'):
                        np.testing.assert_array_equal(z[prefix + 'prediction'], z[prefix + 'members'].mean(axis=0))
                        np.testing.assert_allclose(float(abs(z[prefix + 'prediction'] - z['target']).mean()),
                                                   rec[prefix + 'mae'], atol=1e-10, rtol=0)
            assert ids == expected and len(ids) == len(set(ids))
            if not p['smoke']:
                assert len(ids) == p['samples']
        for fold in p['folds']:
            left, right = [read(out / arm / f'fold-{fold}' / 'generation-00.json') for arm in ARMS]
            assert left['genes'] == right['genes']
            for inner in range(2 if p['smoke'] else 5):
                name = f'generation-00-candidate-00-inner-{inner}'
                a, b = [read(out / arm / f'fold-{fold}' / 'fits' / name / 'result.json') for arm in ARMS]
                for key in ('train_ids', 'validation_ids', 'seed', 'target_scaling'):
                    assert a[key] == b[key], key
        summary['kan_relative_mae_change'] = means[ARMS[1]]['mae'] / means[ARMS[0]]['mae'] - 1
        summary['kan_relative_parameter_change'] = means[ARMS[1]]['parameters_sum'] / means[ARMS[0]]['parameters_sum'] - 1
        summary['fold_sample_sd'] = {arm: float(np.std([r['mae'] for r in rows if r['arm'] == arm], ddof=1))
                                     for arm in ARMS} if not p['smoke'] else {}
        write_json(out / 'artifact-audit.json', dict(passed=True, complete=True, smoke=p['smoke'],
            task=p['task'], outer_samples=len(expected), protocol_fingerprint=digest(p),
            input_audit_passed=True, identical_initial_population=True, identical_inner_splits_and_scaling=True,
            native_early_stopping=True, physical_mae_and_prediction_hashes_verified=True,
            selected_model_hashes_verified=True, inverse_transform_parameters=0))
    write_json(out / 'summary.json', summary)
    if rows:
        pd.DataFrame(rows).to_csv(out / 'fold-results.csv', index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-audit', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error('--n-jobs must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out / 'run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    started = time.perf_counter()
    def progress(state, **values):
        write_json(out / 'status.json', dict(state=state, task=args.task, pid=os.getpid(),
            updated_at=base.engine.now(), elapsed_seconds=time.perf_counter() - started,
            n_jobs=args.n_jobs, total_is_upper_bound=True, ga_early_stopping=4, **values))
    try:
        progress('preflight')
        original, audit = audit_inputs(args.task, out)
        if args.preflight_only:
            progress('preflight_passed', training_started=False)
            return
        if not args.smoke:
            assert args.smoke_audit is not None, 'A completed task-specific smoke audit is required.'
            smoke = read(args.smoke_audit / 'artifact-audit.json')
            smoke_protocol = read(args.smoke_audit / 'protocol.json')
            assert smoke['passed'] and smoke['complete'] and smoke['smoke'] and smoke['task'] == args.task
            for source, sha in smoke_protocol['source_sha256'].items():
                assert file_hash(source) == sha, 'Source changed after smoke: ' + source
            assert smoke_protocol['input_sha256'] == audit['input_sha256']
        p = freeze(out, args.task, args.smoke, audit)
        probe = MODData.load(audit['folds'][0]['training_cache'])
        write_json(out / 'early-stop-selfcheck.json', base.common.verify_early_stop(probe))
        del probe
        # Paired per-fold progress makes an initial MLP/KAN comparison available early.
        for fold in p['folds']:
            for arm in ARMS:
                progress('starting_fold', fold=fold, arm=arm)
                base.run_fold(out, p, original, arm, fold, args.n_jobs, progress)
                report(out, p)
        summary = report(out, p)
        assert summary['complete']
        progress('complete', complete=True, smoke=args.smoke, means=summary['means'])
    except BaseException:
        progress('failed', complete=False, traceback=traceback.format_exc())
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
