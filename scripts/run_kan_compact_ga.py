"""Fixed-input KAN GA preferring fewer parameters within 1% MAE bands.

Training, splits, search budgets and 50-member ensembles retain the existing
protocol. Parent selection, final ranking and stopping use the new preference.
"""
import argparse
import copy
import gc
import importlib.metadata
import os
import pickle
import random
import shutil
import time
import traceback
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_phonons_standardized_matbench as base
import run_standardized_matbench_input_budget as helpers
import native_fourway_models as spline
import kan_accuracy_size_policy as preference
from run_modnet_official_matched import (
    np, tf, MODData, digest, file_hash, write_json, set_seed, sample_ids, matbench_kfold_splits,
)

ROOT, native, engine = base.ROOT, helpers.native, base.engine
read = helpers.read
GA = dict(size_pop=20, num_generations=10, nested=5, early_stopping=4, refit=False)
TASKS = {
    'matbench_steels': dict(samples=312, units='MPa',
        reference='steels-standardized-input-budget-20261116', budgets=[16, 48, 32, 64, 80, 96, 112]),
    'matbench_phonons': dict(samples=1265, units='cm^-1',
        reference='phonons-standardized-input-budget-20261115', budgets=[16, 48, 32, 64, 80, 96, 112, 128, 160, 192]),
}
A_INITIAL = (1, 2, 4, 8, 16, 32, 64, 128, 256)
B_INITIAL = (1, 2, 4, 8, 16, 32, 64, 128)
LR = (.1, .01, .005, .001)
BATCH = (32, 64, 128, 256)
VARIABLES = ('kan_a', 'kan_b', 'xscale', 'lr', 'batch_size')


def verify_genes(g, n):
    assert g['n_feat'] == n and g['loss'] == 'mae' and g['act'] == 'elu'
    assert type(g['kan_a']) is int and 1 <= g['kan_a'] <= 256
    assert type(g['kan_b']) is int and 1 <= g['kan_b'] <= 128
    assert g['lr'] in LR and g['batch_size'] in BATCH and g['xscale'] in ('minmax', 'standard')


class DirectIndividual:
    """Five variable genes; exact feature count is immutable for this GA."""
    def __init__(self, max_feat, num_classes, multi_label=False, targets=None, weights=None):
        self.max_feat, self.num_classes = max_feat, num_classes
        self.multi_label, self.targets, self.weights = multi_label, targets, weights
        self.genes = dict(act='elu', loss='mae', n_feat=max_feat,
            kan_a=random.choice(A_INITIAL), kan_b=random.choice(B_INITIAL),
            xscale=random.choice(('minmax', 'standard')), lr=random.choice(LR), batch_size=random.choice(BATCH))
        self.model = None

    def crossover(self, partner):
        assert self.max_feat == partner.max_feat
        child = copy.copy(self)
        maternal = set(random.sample(VARIABLES, random.choice((2, 3))))
        child.genes = dict(self.genes)
        for key in VARIABLES:
            child.genes[key] = self.genes[key] if key in maternal else partner.genes[key]
        child.model = None
        verify_genes(child.genes, self.max_feat)
        return child

    def mutation(self, prob_mut):
        if np.random.rand() < prob_mut:
            for key, maximum in (('kan_a', 256), ('kan_b', 128)):
                delta = max(1, int(round(self.genes[key] / 4))) * random.randint(-2, 2)
                self.genes[key] = int(max(1, min(maximum, self.genes[key] + delta)))
            key = random.choice(('xscale', 'lr', 'batch_size'))
            values = {'xscale': ('minmax', 'standard'), 'lr': LR, 'batch_size': BATCH}[key]
            self.genes[key] = random.choice(values)
        verify_genes(self.genes, self.max_feat)


class DirectGA(engine.EqualBudgetGA):
    def initialization_population(self, size_pop, multi_label):
        self.pop = [DirectIndividual(len(self.train_data.get_optimal_descriptors()),
                    self.num_classes, multi_label, self.targets, self.weights) for _ in range(size_pop)]

    def run(self, size_pop=20, num_generations=10, prob_mut=None, nested=5,
            multi_label=False, n_jobs=None, early_stopping=4, refit=False, fast=False):
        assert not refit and not multi_label
        if fast:
            size_pop, num_generations, nested = 2, 2, 2
        if prob_mut is None:
            prob_mut = 1 / size_pop
        self._init_run(n_jobs=n_jobs)
        self.initialization_population(size_pop, multi_label=False)
        losses, parameters, models, individuals, states = [], [], [], [], []
        population = self.pop
        try:
            for generation in range(num_generations):
                print('COMPACT_GENERATION', generation, flush=True)
                current_losses, current_models, current_individuals = self.function_fitness(
                    population, n_jobs=n_jobs, nested=nested, multi_label=False, fast=fast)
                for candidate, individual in enumerate(current_individuals):
                    g = individual.genes
                    expected = spline.spline_count(g['n_feat'], g['kan_a'], g['kan_b'])
                    # Read saved actual counts, not only the architecture estimate.
                    actual = [read(self.directory/'fits'/
                        f'generation-{generation:02d}-candidate-{candidate:02d}-inner-{i}'/'result.json')['parameters']
                        for i in range(nested)]
                    assert actual == [expected] * nested
                    parameters.append(actual[0])
                losses.extend(map(float, current_losses))
                models.extend(current_models)
                individuals.extend(current_individuals)
                order, bands = preference.accuracy_size_order(losses, parameters)
                states.append(preference.champion_state(losses, order))
                selected = order[:10]
                mae_only = sorted(range(len(losses)), key=lambda i: (losses[i], i))[:10]
                selected_mae = float(np.mean([losses[i] for i in selected]))
                mae_only_mae = float(np.mean([losses[i] for i in mae_only]))
                assert sum(parameters[i] for i in selected) <= sum(parameters[i] for i in mae_only)
                assert selected_mae <= (1 + preference.TOLERANCE)*mae_only_mae + 1e-10
                write_json(self.directory/f'preference-{generation:02d}.json', dict(
                    generation=generation, relative_mae_tolerance=preference.TOLERANCE,
                    mean_validation_mae=losses, actual_parameters=parameters,
                    ranking=order, bands=bands, champion_state=states[-1],
                    pareto_indices=preference.pareto_indices(losses, parameters),
                    parent_probabilities=(np.asarray(preference.parent_weights(len(order))) /
                                          sum(preference.parent_weights(len(order)))).tolist(),
                    selected_candidate_indices=selected, mae_only_candidate_indices=mae_only,
                    selected_mean_candidate_validation_mae=selected_mae,
                    mae_only_mean_candidate_validation_mae=mae_only_mae,
                    selected_parameters_sum=nested*sum(parameters[i] for i in selected),
                    mae_only_parameters_sum=nested*sum(parameters[i] for i in mae_only),
                    first_band_selected=sum(i in bands[0]['candidate_indices'] for i in selected),
                    test_labels_used=False))
                self.best_individual = individuals[order[0]]
                stop = preference.should_stop(states, early_stopping)
                if stop or generation == num_generations - 1:
                    break
                weights = preference.parent_weights(len(order))
                parents = [individuals[i] for i in order]
                mothers = random.choices(parents, weights=weights, k=size_pop)
                fathers = random.choices(parents, weights=weights, k=size_pop)
                population = [a.crossover(b) for a, b in zip(mothers, fathers)]
                for child in population:
                    child.mutation(prob_mut)
        except BaseException:
            self.pool.terminate()
            raise
        else:
            self._end_run()
        self.best_model = engine.EnsembleMODNetModel(
            modnet_models=[member for i in order[:10] for member in models[i].model])
        self.results = self.best_individual.genes
        return self.best_model


def direct_fit(individual, train, val, variant, fast=False, seed=0):
    g = individual.genes
    verify_genes(g, len(train.get_optimal_descriptors()))
    y = train.df_targets.values.copy()
    mean, std = float(y.mean()), float(y.std())
    assert np.isfinite(std) and std > 0
    normalized = []
    for data in (train, val):
        clone = copy.copy(data)
        clone.df_targets = (data.df_targets.copy() - mean) / std
        normalized.append(clone)
    # MODNet's four-block argument is only interface metadata. Actual KAN
    # construction uses exactly the two explicitly searched widths below.
    with patch.object(spline, 'matched_widths', return_value=(g['kan_a'], g['kan_b'])):
        model = spline.NativeSplineModel(targets=individual.targets, weights=individual.weights,
            num_classes=individual.num_classes, multi_label=False, n_feat=g['n_feat'],
            num_neurons=[[g['kan_a']], [g['kan_b']], [1], [1]], act=g['act'],
            spline_seed=seed, input_scale=g['xscale'])
    model.reference_dense_parameters = None
    model.direct_kan_widths = [g['kan_a'], g['kan_b']]
    model.fit(normalized[0], val_data=normalized[1], loss=g['loss'], lr=g['lr'],
        epochs=1 if fast else 800, batch_size=g['batch_size'], xscale=g['xscale'], verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30,
            min_delta=.001/std, restore_best_weights=False)])
    parameters = model.model.count_params()
    assert parameters == spline.spline_count(g['n_feat'], g['kan_a'], g['kan_b'])
    output = tf.keras.layers.Rescaling(std, mean, dtype='float64', name='physical_target_units')(model.model.output)
    model.model = tf.keras.Model(model.model.input, output)
    assert model.model.count_params() == parameters
    model.min_y, model.max_y = y.min(axis=0), y.max(axis=0)
    model.target_scaling = dict(mean=mean, std=std, ddof=0, fit_ids=list(train.df_targets.index),
        prediction_units='physical', early_stopping_delta_physical=.001, early_stopping_delta_normalized=.001/std)
    model.normalized_history = copy.deepcopy(model.history)
    for key in ('loss', 'val_loss', 'mean_absolute_error', 'val_mean_absolute_error'):
        if key in model.history:
            model.history[key] = [float(v)*std for v in model.history[key]]
    individual.val_loss, individual.model = model.evaluate(val), model


# Only this process and its own spawned workers use these adapters.
native.neurons_from_genes = lambda g: [[g['kan_a']], [g['kan_b']], [1]]


def run_fit(job):
    p = helpers.load_protocol(job['protocol_path'], job['protocol_fingerprint'])
    directory = Path(job['directory'])
    path = directory / 'result.json'
    if path.exists():
        saved = read(path)
        if saved.get('job_signature') == digest(job):
            assert file_hash(directory / 'model.pkl') == saved['model_sha256']
            return saved
    data, positions = helpers.training_data(job['training_cache'], job['training_sha256'])
    verify_genes(job['genes'], len(data.get_optimal_descriptors()))
    assert set(job['train_ids']).isdisjoint(job['validation_ids'])
    assert set(job['train_ids'] + job['validation_ids']) == set(positions)
    train, val = data.split(tuple([positions[s] for s in job[key]] for key in ('train_ids', 'validation_ids')))
    names = list(data.get_optimal_descriptors())
    targets = list(data.target_names)
    individual = SimpleNamespace(genes=dict(job['genes']), targets=[[targets]], weights={n: 1 for n in targets},
                                 num_classes=data.num_classes, multi_label=False)
    with patch.object(native, 'fit_kan', partial(direct_fit, seed=job['seed'])):
        result = native.evaluate_job(dict(individual=individual, train_data=train, val_data=val,
            directory=str(directory), seed=job['seed'], fast=job['smoke'], variant='bspline',
            run_fingerprint=job['run_fingerprint']))
    model_path = directory / 'model.pkl'
    with model_path.open('rb') as stream:
        model = pickle.load(stream)
    scale = model.target_scaling
    assert scale['fit_ids'] == job['train_ids']
    assert scale['mean'] == float(train.df_targets.values.mean()) and scale['std'] == float(train.df_targets.values.std())
    assert model.direct_kan_widths == [job['genes']['kan_a'], job['genes']['kan_b']]
    assert model.reference_dense_parameters is None
    del model
    assert result['selected_features'] == names and result['parameters'] == spline.spline_count(len(names), job['genes']['kan_a'], job['genes']['kan_b'])
    result.update(job_signature=digest(job), model_sha256=file_hash(model_path), task=p['task'],
        target_scaling=scale, spline_widths=[job['genes']['kan_a'], job['genes']['kan_b'], 1],
        parameter_mapping='Direct KAN width genes; no virtual MLP or fixed ratio',
        reused=False, arm=job['arm'], fold=job['fold'], generation=job['generation'],
        candidate=job['candidate'], inner=job['inner'], protocol_fingerprint=digest(p))
    write_json(path, result)
    return result


engine.run_fit = run_fit


def freeze(out, task, smoke):
    config = TASKS[task]
    reference = ROOT / 'benchmarks' / config['reference']
    old, summary, audit = [read(reference / name) for name in ('protocol.json', 'summary.json', 'artifact-audit.json')]
    assert summary['complete'] and audit['passed'] and not audit['smoke']
    baseline = read(Path(old['reference']) / 'protocol.json')
    sources = dict(old['source_sha256'])
    sources[str(Path(helpers.__file__).resolve())] = file_hash(helpers.__file__)
    sources[str(Path(__file__).resolve())] = file_hash(__file__)
    sources[str(Path(preference.__file__).resolve())] = file_hash(preference.__file__)
    for path, expected in sources.items():
        assert file_hash(path) == expected, path
    packages = {name: importlib.metadata.version(name) for name in baseline['packages']}
    assert packages == baseline['packages']
    references = {str(reference / name): file_hash(reference / name)
                  for name in ('protocol.json', 'summary.json', 'artifact-audit.json')}
    folds = []
    for f in old['folds']:
        assert file_hash(f['cache']) == f['cache_sha256']
        assert set(f['train_ids']).isdisjoint(f['test_ids'])
        refs = {}
        for n in config['budgets']:
            folder = reference / f"fold-{f['fold']}" / f'k-{n:03d}'
            refs[str(n)] = read(folder / 'complete.json')
            for name in ('complete.json', 'outer-predictions.npz'):
                references[str(folder / name)] = file_hash(folder / name)
            assert references[str(folder / 'outer-predictions.npz')] == refs[str(n)]['prediction_sha256']
        folds.append(dict(fold=f['fold'], cache=f['cache'], cache_sha256=f['cache_sha256'], ranking=f['ranking'],
            train_ids=f['train_ids'], test_ids=f['test_ids'], official_test_ids=f['official_test_ids'],
            expected_inner_splits=f['splits'], mlp_reference=f['arms']['mm_scaled_modnet']['complete'],
            fixed_width_references=refs))
    original = base.OLD / 'precomputed' / (task + '_moddata.pkl.gz')
    assert file_hash(original) == old['original_sha256']
    budgets = [min(config['budgets']), max(config['budgets'])] if smoke else config['budgets']
    p = dict(task=task, units=config['units'], samples=config['samples'], protocol_path=str(out/'protocol.json'),
        output=str(out), reference=str(reference), baseline=old['reference'], original=str(original),
        original_sha256=file_hash(original), source_sha256=sources, reference_sha256=references, packages=packages,
        smoke=smoke, folds=folds[:1] if smoke else folds, budgets=budgets, seed=baseline['seed'], ga=GA,
        fits_per_arm_fold=8 if smoke else 1000, training=baseline['training'],
        search_space=dict(kan_a=dict(bounds=[1,256],initial=list(A_INITIAL)),
            kan_b=dict(bounds=[1,128],initial=list(B_INITIAL)), fixed_depth=2,
            learning_rates=list(LR), batch_sizes=list(BATCH), input_scalers=['minmax','standard'],
            width_ratio=None, parameter_matching=False, scalar_parameter_penalty=False),
        preference=dict(relative_mae_tolerance=preference.TOLERANCE,
            ranking='Anchor a band at the lowest remaining mean inner-validation MAE; include candidates up to 1% worse; sort by actual parameters, then MAE, then stable candidate index; repeat for subsequent bands.',
            parent_selection='Sample all archived candidates with weights 1/(preference rank+1), using this same accuracy/size ordering.',
            final_selection='First 10 ranked candidates, each with its 5 inner members. Keep 50 unique trained members; if the first band has fewer than 10 candidates, fill from subsequent bands and report this explicitly.',
            stopping='Stop when both preferred compact candidate and minimum-MAE candidate are unchanged for 4 consecutive generations, or after 10 generations.',
            accuracy_limit='The 1% band applies to candidate mean inner-validation MAE. It is not a guarantee on ensemble or outer-test MAE.',
            parameter_limit='No hard parameter ceiling; optimize size within accuracy bands and report reduction relative to standardized MM-MLP.'),
        operators=dict(crossover='Balanced uniform crossover of five variable genes.',
            mutation='Native probability 1/population; bounded +/-0,1,2 quarter-width steps on a,b and resample one training gene.',
            unchanged='Population size, maximum generations, training implementation, splits, direct width operators and ensemble size.'),
        inputs='Exactly n top-ranked MM descriptors in every candidate; old capped-input reference members may use fewer.',
        fitness='Mean of five inner validation MAEs in physical units and actual parameter count; no outer scoring in GA.',
        selection='Top 10 under anchored 1%-MAE-band / parameter-count ranking x five inner members; refit=False; freeze before test scoring.',
        target_scaling='Each inner-training subset only; z-score ddof=0 and zero-parameter inverse transform.',
        restart='Resume only identically signed fits from this experiment; no reuse of historical trained models.',
        interpretation='Exploratory follow-up after observing prior outer results. Added GA search budget and user-approved 1% accuracy/size preference; not a pure width-only ablation or unchanged native MODNet selection.')
    if (out/'protocol.json').exists():
        assert digest(read(out/'protocol.json')) == digest(p), 'Frozen protocol changed'
    else:
        write_json(out/'protocol.json', p)
    for path, expected in sources.items():
        source=Path(path); dest=out/'source-snapshot'/(source.parent.name+'--'+source.name)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copy2(source,dest)
        assert file_hash(dest)==expected
    return p


def prepare(out,p,f,n):
    source=MODData.load(f['cache'])
    assert file_hash(f['cache'])==f['cache_sha256'] and sample_ids(source)==f['train_ids']
    assert list(source.get_optimal_descriptors())==f['ranking']
    source.optimal_features=f['ranking'][:n]
    assert len(source.optimal_features)==n
    directory=out/'inputs'/f"fold-{f['fold']}"/f'k-{n:03d}'
    cache=directory/'training.pkl.gz'; manifest=directory/'manifest.json'
    directory.mkdir(parents=True,exist_ok=True)
    if not cache.exists(): source.save(cache)
    data=MODData.load(cache)
    assert sample_ids(data)==f['train_ids'] and list(data.get_optimal_descriptors())==f['ranking'][:n]
    np.testing.assert_array_equal(data.df_targets.values, source.df_targets.values)
    record=dict(protocol_fingerprint=digest(p),sha256=file_hash(cache),source_sha256=f['cache_sha256'],features=f['ranking'][:n])
    if manifest.exists(): assert read(manifest)==record
    else: write_json(manifest,record)
    return data,cache


def stopping_audit(directory, selected, p):
    files = sorted(directory.glob('generation-*.json'))
    pop, nested, maximum = (2, 2, 2) if p['smoke'] else (20, 5, 10)
    losses, parameters, states, history = [], [], [], []
    first_stop = None
    for generation, path in enumerate(files):
        assert path.name == f'generation-{generation:02d}.json'
        data = read(path)
        values = np.asarray(data['validation_mae'])
        assert values.shape == (pop, nested) and np.isfinite(values).all()
        np.testing.assert_allclose(values.mean(axis=1), data['mean_validation_mae'], atol=1e-10, rtol=0)
        for candidate, g in enumerate(data['genes']):
            expected = spline.spline_count(g['n_feat'], g['kan_a'], g['kan_b'])
            for inner in range(nested):
                rec = read(directory/'fits'/f'generation-{generation:02d}-candidate-{candidate:02d}-inner-{inner}'/'result.json')
                assert rec['parameters'] == expected and rec['genes'] == g
                assert rec['validation_mae'] == values[candidate, inner]
            parameters.append(expected)
        losses.extend(values.mean(axis=1).tolist())
        order, bands = preference.accuracy_size_order(losses, parameters)
        state = preference.champion_state(losses, order)
        states.append(state)
        recorded = read(directory/f'preference-{generation:02d}.json')
        assert recorded['ranking'] == order and recorded['bands'] == bands
        assert recorded['actual_parameters'] == parameters
        assert recorded['mean_validation_mae'] == losses
        assert recorded['champion_state'] == list(state)
        assert recorded['selected_candidate_indices'] == order[:10]
        assert not recorded['test_labels_used']
        mae_only = sorted(range(len(losses)), key=lambda i: (losses[i], i))[:10]
        chosen = order[:10]
        assert sum(parameters[i] for i in chosen) <= sum(parameters[i] for i in mae_only)
        assert np.mean([losses[i] for i in chosen]) <= (1 + preference.TOLERANCE)*np.mean([losses[i] for i in mae_only]) + 1e-10
        np.testing.assert_allclose(recorded['parent_probabilities'],
            np.asarray(preference.parent_weights(len(order)))/sum(preference.parent_weights(len(order))), atol=1e-15, rtol=0)
        history.append(dict(generation=generation+1, preferred_candidate=order[0],
            best_validation_mae=min(losses), preferred_validation_mae=losses[order[0]],
            preferred_parameters=parameters[order[0]]))
        if preference.should_stop(states, 4) and first_stop is None:
            first_stop = generation + 1
    assert len(files) == (first_stop if first_stop is not None else maximum)
    expected = [f'generation-{index//pop:02d}-candidate-{index%pop:02d}-inner-{i}'
                for index in order[:10] for i in range(nested)]
    assert [Path(job['directory']).name for job in selected] == expected
    assert len({job['directory'] for job in selected}) == len(selected)
    result = dict(passed=True, generations=len(files), early_stopping=4, maximum_generations=maximum,
        stopping_reason='unchanged_compact_and_accuracy_champions_4_generations' if first_stop else 'maximum_generations',
        history=history, selected_members=len(selected), evaluated_fits=len(files)*pop*nested,
        relative_mae_tolerance=preference.TOLERANCE,
        first_band_candidate_count=len(bands[0]['candidate_indices']),
        first_band_selected=sum(i in bands[0]['candidate_indices'] for i in order[:10]),
        selected_candidates=len(order[:10]), preference_sha256=file_hash(directory/f'preference-{len(files)-1:02d}.json'))
    write_json(directory/'ga-stopping-audit.json', result)
    return result


def run_fold(out,p,f,n,workers,progress):
    directory=out/f'k-{n:03d}'/f"fold-{f['fold']}"; directory.mkdir(parents=True,exist_ok=True)
    done=directory/'complete.json'
    if done.exists():
        rec=read(done); assert rec['protocol_fingerprint']==digest(p)
        assert file_hash(directory/'outer-predictions.npz')==rec['prediction_sha256']
        return rec
    training,cache=prepare(out,p,f,n)
    seed=p['seed']+1000000+f['fold']*100000
    set_seed(seed)
    ga=DirectGA(training,directory,cache,p,f'kan_direct_k{n:03d}',f['fold'],seed,progress,workers)
    if not p['smoke']:
        ids=np.asarray(sample_ids(ga.train_data))
        for inner,(tr,va) in enumerate(matbench_kfold_splits(ga.train_data,n_splits=5)):
            assert ids[tr].tolist()==f['expected_inner_splits'][str(inner)]['train_ids']
            assert ids[va].tolist()==f['expected_inner_splits'][str(inner)]['validation_ids']
    try:
        ensemble=ga.run(n_jobs=workers,fast=p['smoke'],**GA)
    except BaseException:
        if getattr(ga,'pool',None): ga.pool.terminate()
        raise
    jobs=[ga.job_by_path[ref.path] for ref in ensemble.model]
    stopping=stopping_audit(directory,jobs,p)
    assert len(jobs)==(8 if p['smoke'] else 50)
    selection=dict(members=jobs,protocol_fingerprint=digest(p),test_labels_used=False,
                   selection=p['selection'],fixed_input_count=n, preference=p['preference'],
                   preference_audit=stopping)
    write_json(directory/'frozen-selection.json',selection)
    original=MODData.load(p['original']); assert file_hash(p['original'])==p['original_sha256']
    test=base.make_data(original.df_featurized.loc[f['test_ids']],original.df_targets.loc[f['test_ids']])
    progress('outer_evaluation',budget=n,fold=f['fold'])
    result=base.score_members(jobs,test,training,directory,f)
    shapes=[]
    for j in jobs:
        rec=read(Path(j['directory'])/'result.json');g=rec['genes']
        assert rec['parameters']==spline.spline_count(n,g['kan_a'],g['kan_b'])
        verify_genes(g,n)
        shapes.append([n,g['kan_a'],g['kan_b'],1])
    result.update(task=p['task'],budget=n,fold=f['fold'],generations=stopping['generations'],
        stopping_reason=stopping['stopping_reason'],search_fits=len(ga.records),selected_shapes=shapes,
        preference_audit=stopping, search_training_seconds=sum(r['seconds'] for r in ga.records),
        protocol_fingerprint=digest(p),prediction_sha256=file_hash(directory/'outer-predictions.npz'),
        selection_sha256=file_hash(directory/'frozen-selection.json'),mlp_reference=f['mlp_reference'],
        fixed_width_reference=f['fixed_width_references'][str(n)])
    write_json(done,result)
    print('FOLD_COMPLETE',p['task'],n,f['fold'],result['mae'],result['parameters_sum'],flush=True)
    del ga,ensemble,training,test,original
    gc.collect()
    return result


def report(out,p):
    rows=[read(out/f'k-{n:03d}'/f"fold-{f['fold']}"/'complete.json') for n in p['budgets'] for f in p['folds']
          if (out/f'k-{n:03d}'/f"fold-{f['fold']}"/'complete.json').exists()]
    curve=[]
    for n in p['budgets']:
        group=[r for r in rows if r['budget']==n]
        if len(group)!=len(p['folds']): continue
        item=dict(budget=n,**{key:float(np.mean([r[key] for r in group])) for key in ('mae','raw_mae','parameters_mean','parameters_sum')})
        for name,field in (('mlp','mlp_reference'),('fixed_width_kan','fixed_width_reference')):
            mae=float(np.mean([r[field]['mae'] for r in group])); params=float(np.mean([r[field]['parameters_sum'] for r in group]))
            item[name]=dict(mae=mae,parameters_sum=params)
            item['relative_mae_change_vs_'+name]=item['mae']/mae-1
            item['parameter_reduction_vs_'+name]=1-item['parameters_sum']/params
        curve.append(item)
    complete=len(rows)==len(p['folds'])*len(p['budgets'])
    summary=dict(task=p['task'],units=p['units'],complete=complete,smoke=p['smoke'],
        protocol_fingerprint=digest(p),rows=rows,curve=curve,
        total_search_fits=sum(r['search_fits'] for r in rows),budgets=p['budgets'],
        minimum_measured_input_strict=min([r['budget'] for r in curve if r['mae']<=r['mlp']['mae']] or [None]),
        limitation=p['interpretation'])
    if complete:
        for n in p['budgets']:
            ids=[]
            for f in p['folds']:
                folder=out/f'k-{n:03d}'/f"fold-{f['fold']}"; rec=read(folder/'complete.json')
                assert file_hash(folder/'outer-predictions.npz')==rec['prediction_sha256']
                assert file_hash(folder/'frozen-selection.json')==rec['selection_sha256']
                with np.load(folder/'outer-predictions.npz') as z:
                    assert z['sample_ids'].tolist()==f['official_test_ids']
                    ids.extend(z['sample_ids'].tolist())
                    for prefix in ('','raw_'):
                        np.testing.assert_array_equal(z[prefix+'prediction'],z[prefix+'members'].mean(axis=0))
                        np.testing.assert_allclose(abs(z[prefix+'prediction']-z['target']).mean(),rec[prefix+'mae'],atol=1e-10,rtol=0)
            assert len(ids)==len(set(ids))==sum(len(f['test_ids']) for f in p['folds'])
            if not p['smoke']: assert len(ids)==p['samples']
        write_json(out/'artifact-audit.json',dict(passed=True,complete=True,smoke=p['smoke'],task=p['task'],
            outer_samples=len(ids),budgets=p['budgets'],direct_width_genes=True,target_scaling_train_only=True,
            exact_input_counts=True,accuracy_size_selection_and_stopping_verified=True,
            relative_mae_tolerance=preference.TOLERANCE,zero_inverse_transform_parameters=True,
            selections_frozen_before_test=True,prediction_hashes_and_mae_verified=True))
    write_json(out/'summary.json',summary)
    return summary


def self_check():
    set_seed(20261117)
    parent=DirectIndividual(16,{'y':0},targets=[[['y']]],weights={'y':1})
    partner=DirectIndividual(16,{'y':0},targets=[[['y']]],weights={'y':1})
    seen=set()
    for _ in range(1000):
        before=copy.deepcopy(parent.genes)
        child=parent.crossover(partner)
        assert all(child.genes[k] in (parent.genes[k],partner.genes[k]) for k in VARIABLES)
        child.mutation(1)
        assert parent.genes==before
        verify_genes(child.genes,16)
        seen.add((child.genes['kan_a'],child.genes['kan_b']))
        parent,partner=child,parent
    assert len(seen)>100
    assert spline.spline_count(16,32,16)==8369
    return dict(passed=True,mutation_crossover_trials=1000,unique_width_pairs=len(seen),fixed_input_verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=TASKS,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-audit',type=Path)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();assert args.workers>0
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=out/'run.lock'
    with lock.open('x',encoding='utf-8') as h:h.write(str(os.getpid()))
    started=time.perf_counter()
    def progress(state,**kw):
        write_json(out/'status.json',dict(state=state,pid=os.getpid(),task=args.task,updated_at=engine.now(),
            workers=args.workers,elapsed_seconds=time.perf_counter()-started,**kw))
    try:
        progress('preflight')
        write_json(out/'operator-selfcheck.json',self_check())
        write_json(out/'preference-selfcheck.json',preference.self_check())
        p=freeze(out,args.task,args.smoke)
        if args.preflight_only:
            progress('preflight_passed',budgets=p['budgets'],training_started=False);return
        if not args.smoke:
            assert args.smoke_audit is not None,'A task-specific passed smoke is required'
            a=read(args.smoke_audit/'artifact-audit.json');sp=read(args.smoke_audit/'protocol.json')
            assert a['passed'] and a['complete'] and a['smoke'] and a['task']==args.task
            assert sp['search_space']==p['search_space'] and sp['training']==p['training']
            assert sp['preference']==p['preference']
            for name,sha in sp['source_sha256'].items():assert file_hash(name)==sha
        for n in p['budgets']:
            for f in p['folds']:
                progress('starting_fold',budget=n,fold=f['fold'])
                run_fold(out,p,f,n,args.workers,progress)
                report(out,p)
        s=report(out,p);assert s['complete']
        progress('complete',complete=True,smoke=p['smoke'],curve=s['curve'])
    except BaseException:
        progress('failed',complete=False,traceback=traceback.format_exc());raise
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
