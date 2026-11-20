"""MM-only MODNet 0.1.12 GA versus pure B-spline KAN, both with raw y.

The upstream GA owns evolution, stopping and final selection. Historical fits
are reused only after checking their data, genes, seeds and model provenance.
No source or artifact belonging to an earlier experiment is modified.
"""
import argparse
import contextlib
import copy
import gc
import importlib.metadata
import io
import os
import pickle
import shutil
import time
import traceback
from pathlib import Path

import run_native_fourway as engine
import native_fourway_models as spline_models
import relaxed_spline_policy as mapping
from run_modnet_official_matched import (
    np, pd, tf, MODData, FitGenetic, EnsembleMODNetModel, ModelReference,
    digest, file_hash, write_json, set_seed, sample_ids, matbench_kfold_splits,
)
from native_fourway_features import ROOT, OLD, ORIGINAL, read, make_data

RAW_PRIOR = ROOT / 'benchmarks/native-fourway-20261029'
ARMS = ('mm_modnet', 'mm_kan')
GA = dict(size_pop=20, num_generations=10, nested=5, early_stopping=4, refit=False)
BASE_RUN_FIT = engine.run_fit
SOURCE_CHECKED = set()
FEATURES = {}
spline_models.matched_widths = mapping.matched_widths
engine.matched_widths = mapping.matched_widths


def now():
    return engine.now()


def verify_early_stop(data):
    """Exercise the actual upstream GA loop without fitting neural networks."""
    class Probe(FitGenetic):
        def __init__(self, source, improving):
            super().__init__(source)
            self.improving = improving
            self.calls = 0

        def _init_run(self, n_jobs=None):
            pass

        def _end_run(self):
            pass

        def function_fitness(self, pop, **kwargs):
            generation = self.calls
            self.calls += 1
            losses, models, individuals = [], [], []
            for i, source in enumerate(pop):
                loss = (1.0 / (generation + 1) if self.improving else
                        (1.0 if generation == 0 else 2.0)) if i == 0 else 10.0 + i
                ref = ModelReference('probe', source.targets, source.weights, source.num_classes)
                model = EnsembleMODNetModel(modnet_models=[ref])
                individual = copy.copy(source)
                individual.model, individual.val_loss = ref, loss
                losses.append(loss)
                models.append(model)
                individuals.append(individual)
            return np.array(losses), np.array(models), np.array(individuals)

    result = {}
    for name, improving, expected in [('unchanged_best', False, 4), ('improving_best', True, 10)]:
        set_seed(20261115)
        probe = Probe(data, improving)
        probe.run(n_jobs=1, **GA)
        assert probe.calls == expected, (name, probe.calls, expected)
        result[name] = dict(actual_generations=probe.calls, expected_generations=expected)
    return dict(passed=True, actual_upstream_loop=True, settings=GA, cases=result)


def check_reuse_sources(p):
    fp = digest(p)
    if fp in SOURCE_CHECKED:
        return
    engine.check_sources(p['protocol_path'], fp)
    for path, sha in p['reuse_protocols'].items():
        assert file_hash(path) == sha, path
        old = read(path)
        assert old['training']['target_scaling'] is None
        for source, old_sha in old['source_sha256'].items():
            assert file_hash(source) == old_sha, source
    SOURCE_CHECKED.add(fp)


def run_fit(job):
    p = read(job['protocol_path'])
    check_reuse_sources(p)
    directory = Path(job['directory'])
    result_path = directory / 'result.json'
    if result_path.exists():
        result = read(result_path)
        assert result['job_signature'] == digest(job)
        assert result['target_scaling'] is None
        assert file_hash(directory / 'model.pkl') == result['model_sha256']
        return result

    source_dir = ((OLD / 'matbench_phonons' / f"fold-{job['fold']}" / 'modnet')
                  if job['arm'] == 'mm_modnet' else
                  (RAW_PRIOR / 'mm_kan' / f"fold-{job['fold']}" / 'fits')) / directory.name
    source_result = source_dir / 'result.json'
    if not job['smoke'] and source_result.exists():
        result = read(source_result)
        for key in ('variant', 'genes', 'seed', 'train_ids', 'validation_ids'):
            assert result[key] == job[key], (str(source_result), key)
        assert not result['fast'] and result.get('target_scaling') is None
        cache = job['training_cache']
        if cache not in FEATURES:
            FEATURES[cache] = list(MODData.load(cache).get_optimal_descriptors())
        assert result['features'] == FEATURES[cache]
        assert result['selected_features'] == FEATURES[cache][:job['genes']['n_feat']]
        if job['variant'] == 'bspline':
            widths = tuple(x[0] for x in engine.neurons_from_genes(job['genes']))
            desc = mapping.describe(job['genes']['n_feat'], widths)
            assert result['parameters'] == desc['spline_parameters']
            assert result['spline_widths'] == desc['widths'] and not desc['fallback_used']
        source_model = source_dir / 'model.pkl'
        model_sha = file_hash(source_model)
        if 'model_sha256' in result:
            assert result['model_sha256'] == model_sha
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / 'model.pkl'
        if not destination.exists():
            try:
                os.link(source_model, destination)
            except OSError:
                shutil.copy2(source_model, destination)
        assert file_hash(destination) == model_sha
        result = dict(result, directory=str(directory), model_sha256=model_sha,
                      job_signature=digest(job), run_fingerprint=job['run_fingerprint'],
                      reused=True, provenance=dict(source=str(source_dir),
                      result_sha256=file_hash(source_result), model_sha256=model_sha))
        for key in ('arm', 'fold', 'generation', 'candidate', 'inner'):
            result[key] = job[key]
    else:
        result = dict(BASE_RUN_FIT(job), reused=False)

    result['target_scaling'] = None
    result['target_units'] = 'cm^-1'
    result['protocol_fingerprint'] = digest(p)
    if job['variant'] == 'bspline':
        widths = tuple(x[0] for x in engine.neurons_from_genes(job['genes']))
        result['parameter_mapping'] = mapping.describe(job['genes']['n_feat'], widths)
        assert result['parameters'] == result['parameter_mapping']['spline_parameters']
    write_json(result_path, result)
    return result


engine.run_fit = run_fit


def freeze(out, smoke):
    old = read(OLD / 'matched-protocol.json')
    assert importlib.metadata.version('modnet') == '0.1.12'
    sources = dict(read(RAW_PRIOR / 'protocol.json')['source_sha256'])
    for source, sha in sources.items():
        assert file_hash(source) == sha, source
    for name in (Path(__file__).name, 'relaxed_spline_policy.py'):
        path = ROOT / 'scripts' / name
        sources[str(path)] = file_hash(path)
    inputs = {str(ORIGINAL): file_hash(ORIGINAL)}
    for fold in range(5):
        for name in ('official-train-moddata.pkl.gz', 'outer-split.json'):
            path = OLD / 'matbench_phonons' / f'fold-{fold}' / name
            inputs[str(path)] = file_hash(path)
    p = dict(kind='Matbench MODNet 0.1.12 GA versus raw-target pure B-spline KAN',
             protocol_path=str(out / 'protocol.json'), smoke=smoke, arms=ARMS,
             folds=[0] if smoke else list(range(5)), seed=old['seed'], ga=GA,
             fits_per_arm_fold=8 if smoke else 1000,
             fit_budget_meaning='Upper bound; stop after 4 generations with unchanged best candidate',
             training=old['training'], source_sha256=sources, input_sha256=inputs,
             packages={n: importlib.metadata.version(n) for n in old['packages']},
             reuse_protocols={str(path): file_hash(path) for path in
                              (OLD / 'matched-protocol.json', RAW_PRIOR / 'protocol.json')},
             features='MM only; official outer-training RR ranking shared by both families and all inner folds',
             target_scaling={'mm_modnet': None, 'mm_kan': None},
             input_scaling='Native GA minmax [-0.5,0.5] or standard; fitted on each inner training set',
             kan='Existing pure 3-layer cubic B-spline, 8 coefficients per edge, same initialization '
                 'and bounded hidden coordinates; original parameter mapping where feasible, '
                 'otherwise closest integer widths as in the already accepted KAN experiment',
             selection='Independent native GA per family; top 10 candidates x 5 inner models, refit=False',
             scoring='Primary: native MODNet prediction including historical outlier replacement. '
                     'Also save raw predictions, raw MAE, replacement counts and prediction seeds.',
             historical_seed_limit='Declared local seed, not the unpublished historical submission seed',
             reuse='Replay GA from initial seed; exact matching historical raw-target fits only; '
                   'never reuse standardized-target KAN fits or later generations beyond native stopping')
    path = out / 'protocol.json'
    if path.exists():
        assert digest(read(path)) == digest(p), 'Protocol changed; use a new output directory'
    else:
        write_json(path, p)
    for source, sha in sources.items():
        dest = out / 'source-snapshot' / (Path(source).parent.name + '--' + Path(source).name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(source, dest)
        assert file_hash(dest) == sha
    return p


def prepare(out, p, original, fold):
    source = OLD / 'matbench_phonons' / f'fold-{fold}'
    split = read(source / 'outer-split.json')
    cache = source / 'official-train-moddata.pkl.gz'
    assert file_hash(cache) == p['input_sha256'][str(cache)]
    data = MODData.load(cache)
    tr, te = list(matbench_kfold_splits(original))[fold]
    original_ids = sample_ids(original)
    assert sample_ids(data) == split['train_ids'] == [original_ids[i] for i in tr]
    assert split['test_ids'] == [original_ids[i] for i in te]
    pd.testing.assert_frame_equal(data.df_targets, original.df_targets.loc[split['train_ids']])
    pd.testing.assert_frame_equal(data.df_featurized, original.df_featurized.loc[split['train_ids']])
    assert list(data.get_optimal_descriptors()) == split['selected_features']
    if p['smoke']:
        data.optimal_features = data.optimal_features[:24]
        cache = out / 'smoke-training.pkl.gz'
        if not cache.exists():
            data.save(cache)
        data = MODData.load(cache)
    write_json(out / 'inputs' / f'fold-{fold}.json', dict(
        training_cache=str(cache), training_sha256=file_hash(cache), split=split,
        raw_target_min=float(data.df_targets.values.min()),
        raw_target_max=float(data.df_targets.values.max()),
        ranked_features=list(data.get_optimal_descriptors()), raw_targets_verified=True))
    return data, cache, split


def stopping_audit(directory, selected, p):
    files = sorted(directory.glob('generation-*.json'))
    pop, nested, maximum = (2, 2, 2) if p['smoke'] else (20, 5, 10)
    losses, winners, history = [], [], []
    first_stop = None
    for g, path in enumerate(files):
        assert path.name == f'generation-{g:02d}.json'
        record = read(path)
        values = np.asarray(record['validation_mae'])
        assert values.shape == (pop, nested) and np.isfinite(values).all()
        np.testing.assert_allclose(values.mean(axis=1), record['mean_validation_mae'], rtol=0, atol=1e-10)
        losses.extend(values.mean(axis=1))
        winner = int(np.asarray(losses).argsort()[0])
        winners.append(winner)
        history.append(dict(generation=g + 1, best_candidate_index=winner,
                            best_validation_mae=float(losses[winner])))
        if g >= 3 and winners[g - 3] == winner and first_stop is None:
            first_stop = g + 1
    actual = len(files)
    assert actual == (first_stop if first_stop is not None else maximum)
    order = np.asarray(losses).argsort()[:10]
    expected = [f'generation-{int(k)//pop:02d}-candidate-{int(k)%pop:02d}-inner-{i}'
                for k in order for i in range(nested)]
    assert [Path(j['directory']).name for j in selected] == expected
    result = dict(passed=True, generations=actual, early_stopping=4, maximum_generations=maximum,
                  stopping_reason='unchanged_best_4_generations' if first_stop else 'maximum_generations',
                  history=history, selected_members=len(selected), evaluated_fits=actual * pop * nested)
    write_json(directory / 'ga-stopping-audit.json', result)
    return result


def score_members(jobs, test, training, directory, split):
    corrected, raw, counts, params, records = [], [], [], [], []
    for job in jobs:
        path = Path(job['directory'])
        rec = read(path / 'result.json')
        assert rec['target_scaling'] is None
        assert set(rec['train_ids']).isdisjoint(rec['validation_ids'])
        assert set(rec['train_ids'] + rec['validation_ids']) == set(split['train_ids'])
        assert set(split['train_ids']).isdisjoint(split['test_ids'])
        assert file_hash(path / 'model.pkl') == rec['model_sha256']
        with (path / 'model.pkl').open('rb') as stream:
            model = pickle.load(stream)
        assert getattr(model, 'target_scaling', None) is None
        y = training.df_targets.loc[rec['train_ids']].values
        np.testing.assert_array_equal(model.min_y, y.min(axis=0))
        np.testing.assert_array_equal(model.max_y, y.max(axis=0))
        model._restore_model()
        assert all(layer.name != 'physical_target_units' for layer in model.model.layers)
        assert model.model.count_params() == rec['parameters']
        assert model.xscale == rec['genes']['xscale']
        set_seed(job['seed'] + 100000000)
        x = test.get_featurized_df().replace([np.inf, -np.inf, np.nan], 0)[
            model.optimal_descriptors[:model.n_feat]].values
        x = np.nan_to_num(model._scaler.transform(np.nan_to_num(x)), nan=-1)
        before = np.asarray(model.model.predict(x, verbose=0)).reshape(-1)
        with contextlib.redirect_stdout(io.StringIO()):
            after = model.predict(test).values[:, 0]
        assert np.isfinite(before).all() and np.isfinite(after).all()
        raw.append(before)
        corrected.append(after)
        counts.append(int(np.sum(before != after)))
        params.append(rec['parameters'])
        records.append(rec)
        del model
        tf.keras.backend.clear_session()
    raw, corrected = np.asarray(raw), np.asarray(corrected)
    target = test.df_targets.values[:, 0]
    prediction, raw_prediction = corrected.mean(axis=0), raw.mean(axis=0)
    np.savez_compressed(directory / 'outer-predictions.npz', sample_ids=split['official_test_ids'],
                        target=target, prediction=prediction, raw_prediction=raw_prediction,
                        members=corrected, raw_members=raw, replaced_counts=counts)
    pd.DataFrame(dict(sample_id=split['official_test_ids'], target=target,
                      prediction=prediction, raw_prediction=raw_prediction)).to_csv(
                          directory / 'predictions.csv', index=False, float_format='%.17g')
    return dict(mae=float(abs(prediction - target).mean()),
                raw_mae=float(abs(raw_prediction - target).mean()),
                replaced_member_predictions=sum(counts), ensemble_size=len(jobs),
                parameters_mean=float(np.mean(params)), parameters_sum=sum(params),
                selected_raw_target_models_verified=len(records),
                parameter_mapping_fallback_members=sum(
                    r.get('parameter_mapping', {}).get('fallback_used', False) for r in records))


def run_fold(out, p, original, arm, fold, n_jobs, progress):
    directory = out / arm / f'fold-{fold}'
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / 'complete.json'
    if done.exists():
        result = read(done)
        assert result['protocol_fingerprint'] == digest(p)
        assert file_hash(directory / 'outer-predictions.npz') == result['prediction_sha256']
        return result
    training, cache, split = prepare(out, p, original, fold)
    seed = p['seed'] + 1000000 + fold * 100000
    set_seed(seed)
    ga = engine.EqualBudgetGA(training, directory, cache, p, arm, fold, seed, progress, n_jobs)
    try:
        ensemble = ga.run(n_jobs=n_jobs, fast=p['smoke'], **p['ga'])
    except BaseException:
        if getattr(ga, 'pool', None):
            ga.pool.terminate()
        raise
    jobs = [ga.job_by_path[ref.path] for ref in ensemble.model]
    stop = stopping_audit(directory, jobs, p)
    assert len(ga.records) == stop['evaluated_fits']
    assert len(jobs) == (8 if p['smoke'] else 50)
    write_json(directory / 'frozen-selection.json', dict(members=jobs,
               protocol_fingerprint=digest(p), test_labels_used=False,
               selection='Native GA validation ranking, after native early stopping',
               model_sha256={j['directory']: file_hash(Path(j['directory']) / 'model.pkl') for j in jobs}))
    progress('outer_evaluation', arm=arm, fold=fold, generations=stop['generations'])
    test = make_data(original.df_featurized.loc[split['test_ids']],
                     original.df_targets.loc[split['test_ids']])
    result = score_members(jobs, test, training, directory, split)
    if arm == 'mm_modnet' and not p['smoke']:
        old_dir = OLD / 'matbench_phonons' / f'fold-{fold}'
        old_members = read(old_dir / 'frozen-ensemble.json')['members']
        assert [Path(j['directory']).name for j in jobs] == [Path(j['directory']).name for j in old_members]
        with np.load(old_dir / 'modnet-member-predictions.npz') as previous, np.load(directory / 'outer-predictions.npz') as current:
            np.testing.assert_allclose(current['members'], previous['predictions'], rtol=0, atol=1e-6)
        np.testing.assert_allclose(result['mae'], read(old_dir / 'modnet-score.json')['mae'], rtol=0, atol=1e-7)
        result['historical_baseline_replay_verified'] = True
    result.update(arm=arm, fold=fold, generations=stop['generations'],
                  stopping_reason=stop['stopping_reason'], search_fits=len(ga.records),
                  reused_fits=sum(bool(r.get('reused')) for r in ga.records),
                  new_fits=sum(not r.get('reused') for r in ga.records),
                  search_training_seconds=sum(r['seconds'] for r in ga.records),
                  protocol_fingerprint=digest(p), prediction_sha256=file_hash(directory / 'outer-predictions.npz'),
                  selection_sha256=file_hash(directory / 'frozen-selection.json'))
    write_json(done, result)
    print('FOLD_COMPLETE', arm, fold, result, flush=True)
    del ga, ensemble, training, test
    gc.collect()
    return result


def report(out, p):
    rows = [read(out / a / f'fold-{f}' / 'complete.json') for a in ARMS for f in p['folds']
            if (out / a / f'fold-{f}' / 'complete.json').exists()]
    means = {a: {metric: float(np.mean([r[metric] for r in rows if r['arm'] == a]))
                 for metric in ('mae', 'raw_mae', 'parameters_mean', 'parameters_sum')}
             for a in ARMS if sum(r['arm'] == a for r in rows) == len(p['folds'])}
    complete = len(rows) == 2 * len(p['folds'])
    result = dict(complete=complete, smoke=p['smoke'], published_modnet_mae=34.27510615406341,
                  rows=rows, means=means, target_scaling=p['target_scaling'], ga=GA)
    if complete:
        result['kan_relative_mae_change'] = means['mm_kan']['mae'] / means['mm_modnet']['mae'] - 1
        result['kan_relative_parameter_change'] = means['mm_kan']['parameters_sum'] / means['mm_modnet']['parameters_sum'] - 1
        for arm in ARMS:
            ids = []
            for fold in p['folds']:
                directory = out / arm / f'fold-{fold}'
                with np.load(directory / 'outer-predictions.npz') as z:
                    ids.extend(z['sample_ids'].tolist())
                    np.testing.assert_array_equal(z['prediction'], z['members'].mean(axis=0))
                    np.testing.assert_allclose(float(abs(z['prediction'] - z['target']).mean()),
                                               read(directory / 'complete.json')['mae'], atol=1e-10, rtol=0)
            assert len(ids) == len(set(ids)) == (253 if p['smoke'] else 1265)
        write_json(out / 'artifact-audit.json', dict(passed=True, complete=True, smoke=p['smoke'],
                   checks=['Official outer IDs and complete OOF coverage', 'Upstream GA early stopping',
                           'Native top-10 selection before test evaluation', 'Raw targets in both families',
                           'Selected model hashes and physical target bounds', 'OOF MAE reconstruction',
                           'Historical MLP selection and predictions replayed'], rows=rows))
    write_json(out / 'summary.json', result)
    if rows:
        pd.DataFrame(rows).to_csv(out / 'fold-results.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify-early-stop-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out / 'run.lock'
    start = time.perf_counter()
    def progress(state, **kwargs):
        write_json(out / 'status.json', dict(state=state, pid=os.getpid(), updated_at=now(),
                   elapsed_seconds=time.perf_counter()-start, n_jobs=args.n_jobs,
                   total_is_upper_bound=True, ga_early_stopping=4, **kwargs))
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    try:
        progress('preflight')
        original = MODData.load(ORIGINAL)
        probe_data = MODData.load(OLD / 'matbench_phonons/fold-0/official-train-moddata.pkl.gz')
        write_json(out / 'early-stop-selfcheck.json', verify_early_stop(probe_data))
        del probe_data
        if args.verify_early_stop_only:
            progress('early_stop_verified', complete=True)
            return
        p = freeze(out, args.smoke)
        for arm in ARMS:
            for fold in p['folds']:
                progress('starting_fold', arm=arm, fold=fold)
                run_fold(out, p, original, arm, fold, args.n_jobs, progress)
                report(out, p)
        result = report(out, p)
        assert result['complete']
        progress('complete', complete=True, smoke=args.smoke, means=result['means'])
    except BaseException:
        progress('failed', complete=False, traceback=traceback.format_exc())
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
