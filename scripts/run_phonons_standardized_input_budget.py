"""Reduce KAN input prefixes after the standardized five-fold benchmark finishes.

Keep all 50 selected KAN members and their hidden widths. Retrain only changed
inputs, select dimensions using inner validation, then score outer folds.
"""
import argparse
import gc
import os
import pickle
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_modnet_official_matched as native
import native_fourway_models as spline
from native_fourway_features import ROOT, ORIGINAL, read as read_once
from scaled_native_spline import fit_spline

np, tf = native.np, native.tf
ARMS = ('mm_scaled_modnet', 'mm_scaled_kan')
COARSE = (192, 160, 128, 112, 96, 80, 64, 48, 32, 16)
MARGINS = (0., .05)


def read(path):
    """Windows can briefly deny readers while an atomic checkpoint is replaced."""
    for attempt in range(20):
        try:
            return read_once(path)
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(.05 * (attempt + 1), .5))


def now():
    return datetime.now(timezone.utc).isoformat()


def status(out, state, **values):
    native.write_json(Path(out) / 'status.json', dict(
        state=state, pid=os.getpid(), updated_at=now(), **values))


def wait_for_baseline(reference, out, wait):
    while True:
        summary = read(reference / 'summary.json')
        audit_path = reference / 'artifact-audit.json'
        if summary['complete'] and audit_path.exists():
            audit = read(audit_path)
            assert audit['passed'] and audit['complete'] and not audit['smoke']
            return
        progress = read(reference / 'status.json')
        if progress['state'] in ('failed', 'paused'):
            raise RuntimeError('Baseline is ' + progress['state'])
        if not wait:
            raise RuntimeError('The standardized five-fold baseline is not complete')
        # Query only the recorded PID; no optional process-management dependency.
        query = ("Get-CimInstance Win32_Process -Filter 'ProcessId = " + str(int(progress['pid'])) +
                 "' | Select-Object -ExpandProperty CommandLine")
        result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', query],
            capture_output=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
        command = result.stdout.decode('utf-8', errors='replace').strip()
        if not command and read(reference / 'summary.json')['complete']:
            continue
        assert 'run_phonons_standardized_matbench.py' in command, 'Baseline process is no longer running'
        assert str(reference).replace('\\', '/').lower() in command.replace('\\', '/').lower()
        status(out, 'waiting_for_standardized_five_folds', baseline=progress)
        time.sleep(15)


@lru_cache(maxsize=1)
def training_data(cache, sha):
    assert native.file_hash(cache) == sha
    data = native.MODData.load(cache)
    return data, {str(sid): i for i, sid in enumerate(data.df_targets.index)}


def prepare_protocol(reference, out, smoke):
    base = read(reference / 'protocol.json')
    sources = dict(base['source_sha256'])
    sources[str(Path(__file__).resolve())] = native.file_hash(__file__)
    for filename, sha in sources.items():
        assert native.file_hash(filename) == sha, filename
    references = {str(reference / name): native.file_hash(reference / name)
                  for name in ('protocol.json', 'summary.json', 'artifact-audit.json')}
    folds = []
    for number in ([0] if smoke else range(5)):
        inp = read(reference / 'inputs' / f'fold-{number}.json')
        split = inp['split']
        data, positions = training_data(inp['training_cache'], inp['training_sha256'])
        ranking = inp['ranked_features']
        assert list(data.get_optimal_descriptors()) == ranking
        assert list(positions) == split['train_ids']
        fold = dict(fold=number, cache=inp['training_cache'], cache_sha256=inp['training_sha256'],
                    ranking=ranking, train_ids=split['train_ids'], test_ids=split['test_ids'],
                    official_test_ids=split['official_test_ids'], arms={}, splits={})
        assert len(split['test_ids']) == 253 and set(split['train_ids']).isdisjoint(split['test_ids'])
        for arm in ARMS:
            folder = reference / arm / f'fold-{number}'
            complete = read(folder / 'complete.json')
            selection = read(folder / 'frozen-selection.json')
            assert len(selection['members']) == complete['ensemble_size'] == 50
            assert selection['test_labels_used'] is False
            members = []
            for index, job in enumerate(selection['members']):
                directory = Path(job['directory'])
                rec = read(directory / 'result.json')
                assert rec['selected_features'] == ranking[:rec['genes']['n_feat']]
                assert native.file_hash(directory / 'model.pkl') == rec['model_sha256']
                inner = str(job['inner'])
                partition = dict(train_ids=job['train_ids'], validation_ids=job['validation_ids'])
                if inner in fold['splits']:
                    assert fold['splits'][inner] == partition
                else:
                    fold['splits'][inner] = partition
                values = data.df_targets.loc[partition['train_ids']].values
                scale = rec['target_scaling']
                assert scale['fit_ids'] == partition['train_ids']
                assert scale['mean'] == float(values.mean()) and scale['std'] == float(values.std())
                assert set(partition['train_ids']).isdisjoint(partition['validation_ids'])
                assert set(partition['train_ids'] + partition['validation_ids']) == set(split['train_ids'])
                member = dict(index=index, inner=job['inner'], directory=str(directory),
                              seed=job['seed'], genes=job['genes'], parameters=rec['parameters'],
                              model_sha256=rec['model_sha256'], result_sha256=native.file_hash(directory / 'result.json'),
                              target_scaling=scale)
                if arm.endswith('kan'):
                    member['hidden_widths'] = rec['spline_widths'][:2]
                members.append(member)
            assert [sum(m['inner'] == i for m in members) for i in range(5)] == [10] * 5
            assert sum(m['parameters'] for m in members) == complete['parameters_sum']
            fold['arms'][arm] = dict(members=members, complete=complete,
                                    input_union=max(m['genes']['n_feat'] for m in members))
            for name in ('complete.json', 'frozen-selection.json', 'outer-predictions.npz'):
                references[str(folder / name)] = native.file_hash(folder / name)
            assert references[str(folder / 'outer-predictions.npz')] == complete['prediction_sha256']
            assert references[str(folder / 'frozen-selection.json')] == complete['selection_sha256']
        assert sorted(sid for part in fold['splits'].values() for sid in part['validation_ids']) == sorted(split['train_ids'])
        folds.append(fold)
    protocol = dict(task='matbench_phonons', reference=str(reference), output=str(out), smoke=smoke,
        source_sha256=sources, reference_sha256=references, original_sha256=native.file_hash(ORIGINAL),
        coarse_budgets=[16, 8] if smoke else list(COARSE), margins=list(MARGINS), folds=folds,
        method='Fixed 50-member ensemble; cap each original feature prefix at k; freeze hidden widths, '
               'seeds and training hyperparameters; initialize and retrain every changed input model.',
        preprocessing='Original outer-training RR ranking; target z-score fitted on each inner training split only.',
        validation='Mean of five inner MAEs, averaging the ten held-out members within each split; raw physical predictions.',
        refinement='Per outer fold and margin, refine the interval below the smallest qualifying measured input cap '
                   'in steps of four, then one. If 16 qualifies, also evaluate 8,4,2,1. No outer scores used.',
        selection='Smallest measured input cap meeting each inner-validation threshold; if none qualifies, '
                  'label unmet and use the best inner-MAE cap as fallback.',
        reporting='Outer fixed-grid minima are descriptive; do not claim a global minimum over unmeasured dimensions.',
        training=base['training'], prediction='Native MODNet postprocessing plus raw MAE; deterministic per-member seeds.')
    path = out / 'protocol.json'
    if path.exists():
        assert native.digest(read(path)) == native.digest(protocol), 'Frozen protocol changed'
    else:
        native.write_json(path, protocol)
    for filename, sha in sources.items():
        source = Path(filename)
        target = out / 'source-snapshot' / (source.parent.name + '--' + source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        assert native.file_hash(target) == sha
    return protocol


@lru_cache(maxsize=1)
def load_protocol(filename, fingerprint):
    p = read(filename)
    assert native.digest(p) == fingerprint
    for source, sha in p['source_sha256'].items():
        assert native.file_hash(source) == sha, source
    return p


def load_network(directory, expected_hash, expected_features, expected_scale, expected_parameters):
    assert native.file_hash(Path(directory) / 'model.pkl') == expected_hash
    with (Path(directory) / 'model.pkl').open('rb') as handle:
        model = pickle.load(handle)
    assert model.target_scaling == expected_scale
    model._restore_model()
    assert model.model.count_params() == expected_parameters
    assert model.model.input_shape[-1] == len(expected_features) == model.n_feat
    assert list(model.optimal_descriptors[:model.n_feat]) == expected_features
    inverse = model.model.layers[-1]
    assert inverse.name == 'physical_target_units' and inverse.count_params() == 0
    config = inverse.get_config()
    assert config['scale'] == expected_scale['std'] and config['offset'] == expected_scale['mean']
    return model


def raw_prediction(model, frame):
    x = frame.replace([np.inf, -np.inf, np.nan], 0)[model.optimal_descriptors[:model.n_feat]].values
    x = np.nan_to_num(model._scaler.transform(np.nan_to_num(x)), nan=-1)
    prediction = np.asarray(model.model.predict(x, verbose=0)).reshape(-1)
    assert np.isfinite(prediction).all()
    return prediction


def native_prediction(raw, model, seed):
    """Exact single-target MODNet repair; also retain the unrepaired values."""
    native.set_seed(seed + 100000000)
    lower, upper = float(model.min_y[0]), float(model.max_y[0])
    outside = (raw < lower - .25 * (upper - lower)) | (raw > upper + .25 * (upper - lower))
    prediction = raw.copy()
    prediction[outside] = np.random.uniform(0, 1, int(outside.sum())) * (upper - lower) + lower
    return prediction, int(outside.sum())


def release(model):
    del model
    tf.keras.backend.clear_session()
    gc.collect()


def dimension_folder(out, fold, budget):
    return Path(out) / f'fold-{fold}' / f'k-{budget:03d}'


def fit_member(job):
    protocol_path, fingerprint, number, budget, index = job
    p = load_protocol(protocol_path, fingerprint)
    fold = next(f for f in p['folds'] if f['fold'] == number)
    source = fold['arms'][ARMS[1]]['members'][index]
    split = fold['splits'][str(source['inner'])]
    count = min(budget, source['genes']['n_feat'])
    names = fold['ranking'][:count]
    directory = dimension_folder(p['output'], number, budget) / f'member-{index:02d}'
    metadata_path = directory / 'evaluation.json'
    if metadata_path.exists():
        metadata = read(metadata_path)
        assert metadata['protocol_fingerprint'] == fingerprint
        assert native.file_hash(Path(metadata['model_directory']) / 'model.pkl') == metadata['model_sha256']
        assert native.file_hash(directory / 'validation.npz') == metadata['validation_sha256']
        return metadata
    directory.mkdir(parents=True, exist_ok=True)
    data, positions = training_data(fold['cache'], fold['cache_sha256'])
    assert list(data.get_optimal_descriptors()) == fold['ranking']
    train, val = data.split(tuple([positions[sid] for sid in split[key]] for key in ('train_ids', 'validation_ids')))
    train.optimal_features, val.optimal_features = names, names
    genes = dict(source['genes'], n_feat=count)
    reused = count == source['genes']['n_feat'] and not p['smoke']
    if reused:
        model_directory = Path(source['directory'])
        rec = read(model_directory / 'result.json')
    else:
        targets = list(train.target_names)
        individual = SimpleNamespace(genes=genes, targets=[[targets]], weights={name: 1 for name in targets},
                                     num_classes=train.num_classes)
        model_directory = directory
        with patch.object(spline, 'matched_widths', return_value=tuple(source['hidden_widths'])), \
             patch.object(native, 'fit_kan', partial(fit_spline, seed=source['seed'])):
            rec = native.evaluate_job(dict(individual=individual, train_data=train, val_data=val,
                directory=str(directory), seed=source['seed'], fast=p['smoke'], variant='bspline', run_fingerprint=fingerprint))
    expected_parameters = spline.spline_count(count, *source['hidden_widths'])
    assert rec['parameters'] == expected_parameters
    assert rec['genes'] == genes and rec['seed'] == source['seed']
    assert rec['train_ids'] == split['train_ids'] and rec['validation_ids'] == split['validation_ids']
    assert rec['selected_features'] == names
    sha = native.file_hash(model_directory / 'model.pkl')
    model = load_network(model_directory, sha, names, source['target_scaling'], expected_parameters)
    actual_widths = [int(layer.coefficients.shape[1]) for layer in model.model.layers if hasattr(layer, 'coefficients')]
    assert actual_widths == source['hidden_widths'] + [1]
    values = train.df_targets.values
    assert model.target_scaling['mean'] == float(values.mean()) and model.target_scaling['std'] == float(values.std())
    np.testing.assert_array_equal(model.min_y, values.min(axis=0))
    np.testing.assert_array_equal(model.max_y, values.max(axis=0))
    prediction = raw_prediction(model, val.df_featurized)
    target = val.df_targets.values[:, 0]
    mae = float(np.abs(prediction - target).mean())
    np.testing.assert_allclose(mae, rec['validation_mae'], rtol=1e-5, atol=.003)
    np.savez_compressed(directory / 'validation.npz', sample_ids=split['validation_ids'], target=target, prediction=prediction)
    metadata = dict(protocol_fingerprint=fingerprint, fold=number, budget=budget, index=index,
        inner=source['inner'], input_features=count, parameters=expected_parameters, seed=source['seed'],
        hidden_widths=source['hidden_widths'], model_directory=str(model_directory), model_sha256=sha,
        validation_sha256=native.file_hash(directory / 'validation.npz'), validation_mae=mae,
        reused=reused, epochs=rec['epochs'], seconds=rec['seconds'], target_scaling=model.target_scaling)
    native.write_json(metadata_path, metadata)
    release(model)
    return metadata


def inner_reference(p, fold):
    out = Path(p['output']) / f"fold-{fold['fold']}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'mlp-inner-reference.json'
    fingerprint = native.digest(p)
    if path.exists():
        saved = read(path)
        assert saved['protocol_fingerprint'] == fingerprint
        return saved
    data, positions = training_data(fold['cache'], fold['cache_sha256'])
    matrices = [[] for _ in range(5)]
    for member in fold['arms'][ARMS[0]]['members']:
        split = fold['splits'][str(member['inner'])]
        names = fold['ranking'][:member['genes']['n_feat']]
        model = load_network(member['directory'], member['model_sha256'], names,
                             member['target_scaling'], member['parameters'])
        prediction = raw_prediction(model, data.df_featurized.loc[split['validation_ids']])
        rec = read(Path(member['directory']) / 'result.json')
        truth = data.df_targets.loc[split['validation_ids']].values[:, 0]
        np.testing.assert_allclose(float(np.abs(prediction - truth).mean()), rec['validation_mae'], rtol=1e-5, atol=.003)
        matrices[member['inner']].append(prediction)
        release(model)
    losses = []
    for inner, matrix in enumerate(matrices):
        assert len(matrix) == 10
        ids = fold['splits'][str(inner)]['validation_ids']
        prediction = np.mean(matrix, axis=0)
        target = data.df_targets.loc[ids].values[:, 0]
        losses.append(float(np.abs(prediction - target).mean()))
        np.savez_compressed(out / f'mlp-inner-{inner}.npz', sample_ids=ids, target=target, prediction=prediction)
    saved = dict(protocol_fingerprint=fingerprint, inner_mae=losses, mean_inner_mae=float(np.mean(losses)))
    native.write_json(path, saved)
    return saved


def aggregate_validation(p, fold, budget):
    folder = dimension_folder(p['output'], fold['fold'], budget)
    rows = [read(folder / f'member-{i:02d}' / 'evaluation.json') for i in range(50)]
    losses = []
    for inner in range(5):
        predictions = []
        ids = fold['splits'][str(inner)]['validation_ids']
        target = None
        for row in rows:
            if row['inner'] != inner:
                continue
            archive = folder / f"member-{row['index']:02d}" / 'validation.npz'
            assert native.file_hash(archive) == row['validation_sha256']
            with np.load(archive) as z:
                assert z['sample_ids'].tolist() == ids
                if target is None:
                    target = z['target']
                else:
                    np.testing.assert_array_equal(z['target'], target)
                predictions.append(z['prediction'])
        assert len(predictions) == 10
        losses.append(float(np.abs(np.mean(predictions, axis=0) - target).mean()))
    result = dict(protocol_fingerprint=native.digest(p), budget=budget, fold=fold['fold'],
        inner_mae=losses, mean_inner_mae=float(np.mean(losses)), ensemble_size=50,
        input_union=max(row['input_features'] for row in rows), input_mean=float(np.mean([r['input_features'] for r in rows])),
        parameters_sum=sum(r['parameters'] for r in rows), parameters_mean=float(np.mean([r['parameters'] for r in rows])),
        reused_fits=sum(r['reused'] for r in rows), new_fits=sum(not r['reused'] for r in rows))
    native.write_json(folder / 'validation-summary.json', result)
    return result


def measured(p, number):
    folder = Path(p['output']) / f'fold-{number}'
    return {int(f.parent.name[2:]): read(f) for f in folder.glob('k-*/validation-summary.json')}


def choose(rows, reference, margin):
    limit = reference * (1 + margin)
    eligible = [k for k, r in rows.items() if r['mean_inner_mae'] <= limit]
    budget = min(eligible) if eligible else min(rows, key=lambda k: (rows[k]['mean_inner_mae'], k))
    return dict(budget=budget, qualifies=bool(eligible), threshold=limit,
                mean_inner_mae=rows[budget]['mean_inner_mae'], relative_margin=margin)


def refinement(rows, reference, step):
    requests = set()
    for margin in MARGINS:
        choice = choose(rows, reference, margin)
        if not choice['qualifies']:
            continue
        upper = choice['budget']
        lower = max([k for k in rows if k < upper] or [0])
        requests.update(range(lower + step, upper, step))
    return sorted(k for k in requests if k >= 1 and k not in rows)


def run_budgets(p, requests, workers, phase):
    fingerprint = native.digest(p)
    jobs = [(str(Path(p['output']) / 'protocol.json'), fingerprint, number, k, i)
            for number, budgets in requests.items() for k in sorted(set(budgets), reverse=True) for i in range(50)]
    if not jobs:
        return
    finished = 0
    with native.SafePool(workers) as pool:
        for rec in pool.imap_unordered(fit_member, jobs):
            finished += 1
            status(p['output'], 'training', phase=phase, completed=finished, total=len(jobs),
                   fold=rec['fold'], input_cap=rec['budget'], member=rec['index'], workers=workers)
            print(f"{phase} {finished}/{len(jobs)} fold={rec['fold']} k={rec['budget']} member={rec['index']}", flush=True)
    for number, budgets in requests.items():
        fold = next(f for f in p['folds'] if f['fold'] == number)
        for budget in sorted(set(budgets)):
            aggregate_validation(p, fold, budget)


def frozen_choices(p):
    choices = []
    for fold in p['folds']:
        rows = measured(p, fold['fold'])
        reference = inner_reference(p, fold)['mean_inner_mae']
        choices.append(dict(fold=fold['fold'], mlp_inner_mae=reference, measured_budgets=sorted(rows),
                            strict=choose(rows, reference, 0.), within_5_percent=choose(rows, reference, .05)))
    result = dict(protocol_fingerprint=native.digest(p), test_labels_used=False, choices=choices)
    path = Path(p['output']) / 'frozen-dimension-selection.json'
    if path.exists():
        assert read(path) == result
    else:
        native.write_json(path, result)
    return result


def score(p, selections):
    out = Path(p['output'])
    assert native.file_hash(ORIGINAL) == p['original_sha256']
    original = native.MODData.load(ORIGINAL)
    rows, ids_all = [], []
    for fold in p['folds']:
        number = fold['fold']
        frame = original.df_featurized.loc[fold['test_ids']]
        target = original.df_targets.loc[fold['test_ids']].values[:, 0]
        ids_all.extend(fold['official_test_ids'])
        for arm in ARMS:
            folder = Path(p['reference']) / arm / f'fold-{number}'
            with np.load(folder / 'outer-predictions.npz') as z:
                np.testing.assert_array_equal(z['target'], target)
                assert z['sample_ids'].tolist() == fold['official_test_ids']
                np.testing.assert_allclose(np.abs(z['prediction'] - target).mean(), fold['arms'][arm]['complete']['mae'], atol=1e-10, rtol=0)
        for budget, validation in sorted(measured(p, number).items(), reverse=True):
            folder = dimension_folder(out, number, budget)
            done = folder / 'complete.json'
            if done.exists():
                result = read(done)
                assert result['protocol_fingerprint'] == native.digest(p)
                assert native.file_hash(folder / 'outer-predictions.npz') == result['prediction_sha256']
                rows.append(result)
                continue
            predictions, raw_predictions, repairs = [], [], []
            for index in range(50):
                rec = read(folder / f'member-{index:02d}' / 'evaluation.json')
                model = load_network(rec['model_directory'], rec['model_sha256'],
                    fold['ranking'][:rec['input_features']], rec['target_scaling'], rec['parameters'])
                raw = raw_prediction(model, frame)
                prediction, count = native_prediction(raw, model, rec['seed'])
                if rec['reused']:
                    archive = Path(p['reference']) / ARMS[1] / f'fold-{number}' / 'outer-predictions.npz'
                    with np.load(archive) as z:
                        np.testing.assert_allclose(raw, z['raw_members'][index], atol=1e-7, rtol=1e-7)
                        np.testing.assert_allclose(prediction, z['members'][index], atol=1e-7, rtol=1e-7)
                predictions.append(prediction); raw_predictions.append(raw); repairs.append(count)
                release(model)
            prediction, raw_prediction_mean = np.mean(predictions, axis=0), np.mean(raw_predictions, axis=0)
            np.savez_compressed(folder / 'outer-predictions.npz', sample_ids=fold['official_test_ids'], target=target,
                prediction=prediction, raw_prediction=raw_prediction_mean, members=predictions, raw_members=raw_predictions)
            baseline = fold['arms'][ARMS[0]]['complete']
            result = dict(validation, mae=float(np.abs(prediction - target).mean()),
                raw_mae=float(np.abs(raw_prediction_mean - target).mean()), repaired_member_predictions=sum(repairs),
                mlp_mae=baseline['mae'], mlp_parameters_sum=baseline['parameters_sum'],
                parameter_reduction_vs_mlp=1-validation['parameters_sum']/baseline['parameters_sum'],
                prediction_sha256=native.file_hash(folder / 'outer-predictions.npz'))
            native.write_json(done, result); rows.append(result)
            status(out, 'scoring', fold=number, input_cap=budget)
    assert len(ids_all) == len(set(ids_all)) == 253 * len(p['folds'])
    groups = {}
    for row in rows:
        groups.setdefault(row['budget'], []).append(row)
    curve = []
    mlp_mae = float(np.mean([f['arms'][ARMS[0]]['complete']['mae'] for f in p['folds']]))
    mlp_parameters = float(np.mean([f['arms'][ARMS[0]]['complete']['parameters_sum'] for f in p['folds']]))
    mlp_inputs = float(np.mean([f['arms'][ARMS[0]]['input_union'] for f in p['folds']]))
    kan_parameters = float(np.mean([f['arms'][ARMS[1]]['complete']['parameters_sum'] for f in p['folds']]))
    kan_inputs = float(np.mean([f['arms'][ARMS[1]]['input_union'] for f in p['folds']]))
    kan_mae = float(np.mean([f['arms'][ARMS[1]]['complete']['mae'] for f in p['folds']]))
    for budget, group in sorted(groups.items()):
        if len(group) != len(p['folds']):
            continue
        item = dict(budget=budget, mae=float(np.mean([r['mae'] for r in group])),
            raw_mae=float(np.mean([r['raw_mae'] for r in group])),
            parameters_mean=float(np.mean([r['parameters_mean'] for r in group])),
            parameters_sum=float(np.mean([r['parameters_sum'] for r in group])),
            input_union_mean=float(np.mean([r['input_union'] for r in group])))
        item.update(relative_mae_change=item['mae']/mlp_mae-1,
                    parameter_reduction_vs_mlp=1-item['parameters_sum']/mlp_parameters,
                    input_reduction_vs_mlp=1-item['input_union_mean']/mlp_inputs,
                    parameter_reduction_vs_full_kan=1-item['parameters_sum']/kan_parameters,
                    input_reduction_vs_full_kan=1-item['input_union_mean']/kan_inputs)
        curve.append(item)
    adaptive = {}
    for name in ('strict', 'within_5_percent'):
        selected = [next(r for r in rows if r['fold'] == c['fold'] and r['budget'] == c[name]['budget']) for c in selections['choices']]
        adaptive[name] = dict(fold_budgets=[r['budget'] for r in selected],
            all_inner_thresholds_met=all(c[name]['qualifies'] for c in selections['choices']),
            mae=float(np.mean([r['mae'] for r in selected])),
            parameters_mean=float(np.mean([r['parameters_mean'] for r in selected])),
            parameters_sum=float(np.mean([r['parameters_sum'] for r in selected])),
            input_union_mean=float(np.mean([r['input_union'] for r in selected])))
        adaptive[name]['parameter_reduction_vs_mlp'] = 1-adaptive[name]['parameters_sum']/mlp_parameters
        adaptive[name]['parameter_reduction_vs_full_kan'] = 1-adaptive[name]['parameters_sum']/kan_parameters
        adaptive[name]['relative_mae_change'] = adaptive[name]['mae']/mlp_mae-1
        adaptive[name]['input_reduction_vs_mlp'] = 1-adaptive[name]['input_union_mean']/mlp_inputs
        adaptive[name]['input_reduction_vs_full_kan'] = 1-adaptive[name]['input_union_mean']/kan_inputs
    summary = dict(complete=True, smoke=p['smoke'], protocol_fingerprint=native.digest(p),
        baseline=dict(mae=mlp_mae, parameters_mean=mlp_parameters/50, parameters_sum=mlp_parameters, input_union_mean=mlp_inputs),
        full_kan_reference=dict(mae=kan_mae, parameters_mean=kan_parameters/50, parameters_sum=kan_parameters, input_union_mean=kan_inputs),
        fixed_input_curve=curve, inner_selected=adaptive, rows=rows,
        minimum_measured_fixed_input_strict=min([r['budget'] for r in curve if r['mae'] <= mlp_mae] or [None]),
        minimum_measured_fixed_input_within_5_percent=min([r['budget'] for r in curve if r['mae'] <= 1.05*mlp_mae] or [None]),
        limitation='Fixed-grid outer minima are descriptive; no claim of global minimum or statistical equivalence. '
                   'Adaptive choices use only their own outer-training fold; the RR ranking follows the MODNet protocol.')
    native.write_json(out / 'summary.json', summary)
    draw_summary(out, summary)
    native.write_json(out / 'artifact-audit.json', dict(passed=True, complete=True, smoke=p['smoke'],
        outer_samples=len(ids_all), same_50_member_count=True, hidden_widths_frozen=True,
        target_scaling_train_only=True, inverse_transform_parameters=0,
        selections_frozen_before_outer_scoring=True, prediction_hashes_verified=True))
    return summary


def draw_summary(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = summary['fixed_input_curve']
    x = [r['budget'] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(x, [r['mae'] for r in rows], 'o-', color='#1565c0', label='KAN, fixed 50 members')
    axes[0].axhline(summary['baseline']['mae'], color='#c62828', label='Standardized MM-MLP')
    axes[0].axhline(1.05*summary['baseline']['mae'], color='#c62828', linestyle=':', label='MLP +5% tolerance')
    axes[0].set_ylabel('Outer-test MAE (cm$^{-1}$)')
    axes[0].set_title('Phonons: KAN input reduction' + (' (one-epoch smoke check)' if summary['smoke'] else ' (five folds)'))
    axes[0].legend(fontsize=9)
    axes[1].plot(x, [r['parameters_sum']/1e6 for r in rows], 'o-', color='#1565c0', label='KAN')
    axes[1].axhline(summary['baseline']['parameters_sum']/1e6, color='#c62828', label='MM-MLP')
    axes[1].axhline(summary['full_kan_reference']['parameters_sum']/1e6, color='#2e7d32', linestyle='--', label='Original full-input KAN')
    axes[1].set_ylabel('50-member parameters (millions)')
    axes[1].set_xlabel('Input cap per member (descriptors)')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(Path(out) / 'input-count-mae-parameters.png', dpi=180)
    plt.close(fig)


def self_check():
    with patch(__name__ + '.read_once', side_effect=[PermissionError('busy'), PermissionError('busy'), {'complete': True}]) as mocked, \
         patch.object(time, 'sleep'):
        assert read('test-status.json') == {'complete': True}
        assert mocked.call_count == 3
    with patch(__name__ + '.read_once', side_effect=PermissionError('persistent')) as mocked, \
         patch.object(time, 'sleep'):
        try:
            read('test-status.json')
        except PermissionError:
            assert mocked.call_count == 20
        else:
            raise AssertionError('Persistent read failures must remain visible')
    rows = {16: {'mean_inner_mae': 42.}, 32: {'mean_inner_mae': 39.}, 48: {'mean_inner_mae': 40.}, 64: {'mean_inner_mae': 38.}}
    assert choose(rows, 40., 0.)['budget'] == 32
    assert choose(rows, 30., 0.)['qualifies'] is False
    assert refinement(rows, 40., 4) == [4, 8, 12, 20, 24, 28]
    assert spline.spline_count(64, 34, 16) == 21939
    assert min(192, 190) == 190
    print('Checkpoint read retries, selection, refinement and fixed-width parameter checks passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / 'benchmarks/phonons-standardized-modnet012-20261115')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--wait-for-baseline', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-audit', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    if args.output is None or args.workers < 1:
        parser.error('--output and positive --workers are required')
    out, reference = args.output.resolve(), args.reference.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out / 'run.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    try:
        wait_for_baseline(reference, out, args.wait_for_baseline)
        if not args.smoke:
            if args.smoke_audit is None:
                raise RuntimeError('A completed --smoke-audit directory is required for the full scan')
            smoke_audit = read(args.smoke_audit / 'artifact-audit.json')
            smoke_protocol = read(args.smoke_audit / 'protocol.json')
            assert smoke_audit['passed'] and smoke_audit['complete'] and smoke_audit['smoke']
            assert smoke_protocol['reference'] == str(reference)
            for source, sha in smoke_protocol['source_sha256'].items():
                assert native.file_hash(source) == sha, 'Code changed after the smoke check: ' + source
        status(out, 'preflight')
        p = prepare_protocol(reference, out, args.smoke)
        for fold in p['folds']:
            inner_reference(p, fold)
        # Include an unchanged full-input control for each fold.
        full = {f['fold']: [f['arms'][ARMS[1]]['input_union']] for f in p['folds']}
        run_budgets(p, full, args.workers, 'full_input_control')
        for budget in p['coarse_budgets']:
            run_budgets(p, {f['fold']: [budget] for f in p['folds']}, args.workers, 'coarse')
        if not p['smoke']:
            low = {f['fold']: [8, 4, 2, 1] for f in p['folds']
                   if any(choose(measured(p, f['fold']), inner_reference(p, f)['mean_inner_mae'], m)['budget'] <= 16
                          and choose(measured(p, f['fold']), inner_reference(p, f)['mean_inner_mae'], m)['qualifies'] for m in MARGINS)}
            run_budgets(p, low, args.workers, 'lower_dimensions')
            for step in (4, 1):
                requests = {f['fold']: refinement(measured(p, f['fold']), inner_reference(p, f)['mean_inner_mae'], step) for f in p['folds']}
                run_budgets(p, requests, args.workers, f'refine_step_{step}')
        selections = frozen_choices(p)
        summary = score(p, selections)
        status(out, 'complete', complete=True, smoke=p['smoke'], inner_selected=summary['inner_selected'])
    except BaseException:
        status(out, 'failed', complete=False, traceback=traceback.format_exc())
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
