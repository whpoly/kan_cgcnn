"""MM-MLP and MM-KAN with identical train-only target standardization and GA stopping."""
import argparse
import contextlib
import gc
import importlib.metadata
import io
import os
import pickle
import shutil
import time
import traceback
from pathlib import Path

import run_phonons_rawtarget_matbench as common
import run_native_fourway as engine
import native_fourway_models as spline_models
import relaxed_spline_policy as mapping
from scaled_native_mlp import evaluate_standardized
from scaled_native_spline import fit_spline
from run_modnet_official_matched import (
    np, pd, tf, MODData, digest, file_hash, write_json, set_seed,
)
from native_fourway_features import ROOT, OLD, ORIGINAL, read, make_data

PRIOR = ROOT / 'benchmarks/native-fourway-scaled-kan-20261029'
ARMS = ('mm_scaled_modnet', 'mm_scaled_kan')
GA = dict(size_pop=20, num_generations=10, nested=5, early_stopping=4, refit=False)
BASE_RUN_FIT = common.BASE_RUN_FIT
engine.NativeIndividual.evaluate = evaluate_standardized
engine.fit_spline = fit_spline
engine.matched_widths = mapping.matched_widths
spline_models.matched_widths = mapping.matched_widths
DATA = {}
CHECKED = set()


def check_sources(p):
    fingerprint = digest(p)
    if fingerprint not in CHECKED:
        engine.check_sources(p['protocol_path'], fingerprint)
        for path, sha in p['reuse_protocol_sha256'].items():
            assert file_hash(path) == sha, path
        CHECKED.add(fingerprint)


def training_for(job):
    key = job['training_cache']
    if key not in DATA:
        assert file_hash(key) == job['training_sha256']
        DATA[key] = MODData.load(key)
    return DATA[key]


def check_scale(scale, job, data):
    values = data.df_targets.loc[job['train_ids']].values
    assert scale['fit_ids'] == job['train_ids']
    assert scale['mean'] == float(values.mean()) and scale['std'] == float(values.std())
    assert scale['ddof'] == 0 and scale['prediction_units'] == 'physical'
    assert scale['early_stopping_delta_physical'] == .001
    assert scale['early_stopping_delta_normalized'] == .001/scale['std']
    return values


def run_fit(job):
    p = read(job['protocol_path'])
    check_sources(p)
    data = training_for(job)
    directory = Path(job['directory'])
    result_file = directory / 'result.json'
    if result_file.exists():
        result = read(result_file)
        assert result['job_signature'] == digest(job)
        assert file_hash(directory / 'model.pkl') == result['model_sha256']
        if result.get('target_scaling') and 'reused' in result:
            check_scale(result['target_scaling'], job, data)
            return result
    else:
        result = None
    source_dir = PRIOR / 'mm_kan' / f"fold-{job['fold']}" / 'fits' / directory.name
    if result is None and job['variant'] == 'bspline' and not job['smoke']:
        source_result = source_dir / 'result.json'
        assert source_result.exists(), 'The standardized KAN replay must match the completed historical search'
        result = read(source_result)
        for key in ('variant', 'genes', 'seed', 'train_ids', 'validation_ids'):
            assert result[key] == job[key], (str(source_result), key)
        check_scale(result['target_scaling'], job, data)
        assert result['features'] == list(data.get_optimal_descriptors())
        assert result['selected_features'] == list(data.get_optimal_descriptors())[:job['genes']['n_feat']]
        widths = tuple(x[0] for x in engine.neurons_from_genes(job['genes']))
        desc = mapping.describe(job['genes']['n_feat'], widths)
        assert result['parameters'] == desc['spline_parameters'] and result['spline_widths'] == desc['widths']
        source_model = source_dir / 'model.pkl'
        assert file_hash(source_model) == result['model_sha256']
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / 'model.pkl').exists():
            try:
                os.link(source_model, directory / 'model.pkl')
            except OSError:
                shutil.copy2(source_model, directory / 'model.pkl')
        assert file_hash(directory / 'model.pkl') == result['model_sha256']
        result = dict(result, directory=str(directory), job_signature=digest(job),
            run_fingerprint=job['run_fingerprint'], reused=True,
            provenance=dict(source=str(source_dir), result_sha256=file_hash(source_result),
                            model_sha256=result['model_sha256']))
        for key in ('arm', 'fold', 'generation', 'candidate', 'inner'):
            result[key] = job[key]
    else:
        result = dict(BASE_RUN_FIT(job), reused=False)
        with (directory / 'model.pkl').open('rb') as handle:
            model = pickle.load(handle)
        result['target_scaling'] = model.target_scaling
        del model
    check_scale(result['target_scaling'], job, data)
    result['protocol_fingerprint'] = digest(p)
    result['history_units'] = dict(loss='cm^-1', val_loss='cm^-1', mae='standardized', val_mae='standardized')
    if job['variant'] == 'bspline':
        widths = tuple(x[0] for x in engine.neurons_from_genes(job['genes']))
        result['parameter_mapping'] = mapping.describe(job['genes']['n_feat'], widths)
    write_json(result_file, result)
    return result


engine.run_fit = run_fit


def freeze(out, smoke):
    old = read(PRIOR / 'protocol.json')
    sources = dict(old['source_sha256'])
    sources.update(read(PRIOR / 'parameter-matching-amendment.json')['source_sha256'])
    for path, sha in sources.items():
        assert file_hash(path) == sha, path
    for name in (Path(__file__).name, 'scaled_native_mlp.py', 'run_phonons_rawtarget_matbench.py'):
        path = ROOT / 'scripts' / name
        sources[str(path)] = file_hash(path)
    assert importlib.metadata.version('modnet') == '0.1.12'
    inputs = {str(ORIGINAL): file_hash(ORIGINAL)}
    for fold in range(5):
        for name in ('official-train-moddata.pkl.gz', 'outer-split.json'):
            path = OLD / 'matbench_phonons' / f'fold-{fold}' / name
            inputs[str(path)] = file_hash(path)
    p = dict(kind='MM-MLP versus MM-KAN with matched target standardization',
        protocol_path=str(out / 'protocol.json'), smoke=smoke, arms=ARMS,
        folds=[0] if smoke else list(range(5)), seed=old['seed'], ga=GA,
        fits_per_arm_fold=8 if smoke else 1000, fit_budget_meaning='Maximum, native early stopping enabled',
        packages={n: importlib.metadata.version(n) for n in ('modnet', 'tensorflow', 'numpy', 'pandas', 'scikit-learn')},
        source_sha256=sources, input_sha256=inputs,
        reuse_protocol_sha256={str(path): file_hash(path) for path in
            (PRIOR / 'protocol.json', PRIOR / 'parameter-matching-amendment.json')},
        target_scaling={arm: 'z-score, ddof=0, fitted exclusively on each inner training set' for arm in ARMS},
        training=dict(optimizer='TensorFlow Adam', loss='MAE on standardized targets', epochs=800,
            monitor='loss', patience=30, min_delta_physical=.001,
            min_delta_normalized='.001 / training_target_std', restore_best_weights=False),
        inference='Fixed zero-parameter inverse transform stored in both networks; all scoring in cm^-1',
        features='Same original MM values and native outer-training RR ranking, shared across both arms',
        architecture='Native MODNet MLP; existing pure cubic B-spline KAN and accepted width-matching policy',
        search='Independent upstream GA per family under standardized training; native ranges and initial paired population',
        selection='Top 10 candidate configurations x 5 inner members, refit=False; select before test evaluation',
        reuse='Only matched historical standardized KAN fits are reused, replaying GA early stopping; MLP is trained anew',
        comparison='Original raw-target MODNet retained as reference, not called a standardized baseline',
        scoring='Primary native MODNet postprocessing; also preserve raw predictions and replacement counts')
    path = out / 'protocol.json'
    if path.exists():
        assert digest(read(path)) == digest(p), 'Frozen protocol changed; use a new output'
    else:
        write_json(path, p)
    for source, sha in sources.items():
        dest = out / 'source-snapshot' / (Path(source).parent.name + '--' + Path(source).name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(source, dest)
        assert file_hash(dest) == sha
    return p


def score_members(jobs, test, training, directory, split):
    predictions, raw_predictions, counts, parameters = [], [], [], []
    for job in jobs:
        path = Path(job['directory'])
        rec = read(path / 'result.json')
        y = check_scale(rec['target_scaling'], job, training)
        assert set(job['train_ids']).isdisjoint(job['validation_ids'])
        assert set(job['train_ids'] + job['validation_ids']) == set(split['train_ids'])
        assert set(split['train_ids']).isdisjoint(split['test_ids'])
        assert file_hash(path / 'model.pkl') == rec['model_sha256']
        with (path / 'model.pkl').open('rb') as stream:
            model = pickle.load(stream)
        assert model.target_scaling == rec['target_scaling']
        np.testing.assert_array_equal(model.min_y, y.min(axis=0))
        np.testing.assert_array_equal(model.max_y, y.max(axis=0))
        model._restore_model()
        assert model.model.layers[-1].name == 'physical_target_units'
        transform = model.model.layers[-1].get_config()
        assert transform['scale'] == rec['target_scaling']['std']
        assert transform['offset'] == rec['target_scaling']['mean']
        assert model.model.layers[-1].count_params() == 0
        assert model.model.count_params() == rec['parameters']
        assert model.xscale == rec['genes']['xscale']
        set_seed(job['seed'] + 100000000)
        x = test.get_featurized_df().replace([np.inf, -np.inf, np.nan], 0)[
            model.optimal_descriptors[:model.n_feat]].values
        x = np.nan_to_num(model._scaler.transform(np.nan_to_num(x)), nan=-1)
        raw = np.asarray(model.model.predict(x, verbose=0)).reshape(-1)
        with contextlib.redirect_stdout(io.StringIO()):
            pred = model.predict(test).values[:, 0]
        assert np.isfinite(raw).all() and np.isfinite(pred).all()
        predictions.append(pred)
        raw_predictions.append(raw)
        counts.append(int(np.sum(raw != pred)))
        parameters.append(rec['parameters'])
        del model
        tf.keras.backend.clear_session()
    matrix, raw_matrix = np.asarray(predictions), np.asarray(raw_predictions)
    pred, raw = matrix.mean(axis=0), raw_matrix.mean(axis=0)
    target = test.df_targets.values[:, 0]
    np.savez_compressed(directory / 'outer-predictions.npz', sample_ids=split['official_test_ids'],
        target=target, prediction=pred, raw_prediction=raw, members=matrix,
        raw_members=raw_matrix, replaced_counts=counts)
    pd.DataFrame(dict(sample_id=split['official_test_ids'], target=target,
        prediction=pred, raw_prediction=raw)).to_csv(directory / 'predictions.csv', index=False, float_format='%.17g')
    return dict(mae=float(abs(pred-target).mean()), raw_mae=float(abs(raw-target).mean()),
        ensemble_size=len(jobs), parameters_mean=float(np.mean(parameters)), parameters_sum=sum(parameters),
        replaced_member_predictions=sum(counts), standardized_models_verified=len(jobs))


def run_fold(out, p, original, arm, fold, n_jobs, progress):
    directory = out / arm / f'fold-{fold}'
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / 'complete.json'
    if done.exists():
        result = read(done)
        assert result['protocol_fingerprint'] == digest(p)
        assert file_hash(directory / 'outer-predictions.npz') == result['prediction_sha256']
        return result
    training, cache, split = common.prepare(out, p, original, fold)
    seed = p['seed'] + 1000000 + fold*100000
    set_seed(seed)
    ga = engine.EqualBudgetGA(training, directory, cache, p, arm, fold, seed, progress, n_jobs)
    try:
        ensemble = ga.run(n_jobs=n_jobs, fast=p['smoke'], **p['ga'])
    except BaseException:
        if getattr(ga, 'pool', None):
            ga.pool.terminate()
        raise
    jobs = [ga.job_by_path[ref.path] for ref in ensemble.model]
    stopping = common.stopping_audit(directory, jobs, p)
    assert len(ga.records) == stopping['evaluated_fits']
    assert len(jobs) == (8 if p['smoke'] else 50)
    write_json(directory / 'frozen-selection.json', dict(members=jobs, test_labels_used=False,
        protocol_fingerprint=digest(p), selection=p['selection'],
        model_sha256={j['directory']: file_hash(Path(j['directory']) / 'model.pkl') for j in jobs}))
    progress('outer_evaluation', arm=arm, fold=fold, generations=stopping['generations'])
    test = make_data(original.df_featurized.loc[split['test_ids']], original.df_targets.loc[split['test_ids']])
    result = score_members(jobs, test, training, directory, split)
    result.update(arm=arm, fold=fold, generations=stopping['generations'],
        stopping_reason=stopping['stopping_reason'], search_fits=len(ga.records),
        reused_fits=sum(bool(r['reused']) for r in ga.records),
        new_fits=sum(not r['reused'] for r in ga.records),
        search_training_seconds=sum(r['seconds'] for r in ga.records),
        protocol_fingerprint=digest(p), prediction_sha256=file_hash(directory / 'outer-predictions.npz'),
        selection_sha256=file_hash(directory / 'frozen-selection.json'))
    write_json(done, result)
    print('FOLD_COMPLETE', arm, fold, result, flush=True)
    del ga, ensemble, training, test
    gc.collect()
    return result


def report(out, p):
    rows = [read(out / arm / f'fold-{fold}' / 'complete.json') for arm in ARMS for fold in p['folds']
            if (out / arm / f'fold-{fold}' / 'complete.json').exists()]
    means = {arm: {metric: float(np.mean([r[metric] for r in rows if r['arm'] == arm]))
                  for metric in ('mae', 'raw_mae', 'parameters_mean', 'parameters_sum')}
             for arm in ARMS if sum(r['arm'] == arm for r in rows) == len(p['folds'])}
    result = dict(complete=len(rows) == 2*len(p['folds']), smoke=p['smoke'], rows=rows, means=means,
        ga=GA, target_scaling=p['target_scaling'], raw_target_mlp_reference_mae=34.013377285371305,
        published_modnet_reference_mae=34.27510615406341)
    if result['complete']:
        for arm in ARMS:
            ids = []
            for fold in p['folds']:
                directory = out / arm / f'fold-{fold}'
                record = read(directory / 'complete.json')
                assert file_hash(directory / 'outer-predictions.npz') == record['prediction_sha256']
                with np.load(directory / 'outer-predictions.npz') as z:
                    ids.extend(z['sample_ids'].tolist())
                    np.testing.assert_array_equal(z['prediction'], z['members'].mean(axis=0))
                    np.testing.assert_allclose(float(abs(z['prediction']-z['target']).mean()), record['mae'], atol=1e-10, rtol=0)
            assert len(ids) == len(set(ids)) == (253 if p['smoke'] else 1265)
        for fold in p['folds']:
            left = read(out / ARMS[0] / f'fold-{fold}' / 'generation-00.json')
            right = read(out / ARMS[1] / f'fold-{fold}' / 'generation-00.json')
            assert left['genes'] == right['genes']
            for inner in range(2 if p['smoke'] else 5):
                name = f'generation-00-candidate-00-inner-{inner}'
                a, b = [read(out / arm / f'fold-{fold}' / 'fits' / name / 'result.json') for arm in ARMS]
                for key in ('train_ids', 'validation_ids', 'seed', 'target_scaling'):
                    assert a[key] == b[key], key
        result['kan_relative_mae_change'] = means[ARMS[1]]['mae']/means[ARMS[0]]['mae']-1
        result['kan_relative_parameter_change'] = means[ARMS[1]]['parameters_sum']/means[ARMS[0]]['parameters_sum']-1
        write_json(out / 'artifact-audit.json', dict(passed=True, complete=True, smoke=p['smoke'],
            checks=['Complete official OOF IDs', 'Native early stopping and top-10 selection',
                    'Identical initial candidates, inner partitions and target statistics',
                    'Saved inverse transforms have no trainable parameters',
                    'Raw physical target bounds and reproducible prediction seeds', 'Prediction hashes and physical MAE'], rows=rows))
    write_json(out / 'summary.json', result)
    if rows:
        pd.DataFrame(rows).to_csv(out / 'fold-results.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out / 'run.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    started = time.perf_counter()
    def progress(state, **kwargs):
        write_json(out / 'status.json', dict(state=state, pid=os.getpid(), updated_at=engine.now(),
            elapsed_seconds=time.perf_counter()-started, n_jobs=args.n_jobs,
            total_is_upper_bound=True, ga_early_stopping=4, **kwargs))
    try:
        progress('preflight')
        probe = MODData.load(OLD / 'matbench_phonons/fold-0/official-train-moddata.pkl.gz')
        write_json(out / 'early-stop-selfcheck.json', common.verify_early_stop(probe))
        del probe
        p = freeze(out, args.smoke)
        original = MODData.load(ORIGINAL)
        # Quickly recover the standardized KAN reference before new MLP training.
        for arm in reversed(ARMS):
            for fold in p['folds']:
                progress('starting_fold', arm=arm, fold=fold)
                run_fold(out, p, original, arm, fold, args.n_jobs, progress)
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
