"""Four independent native-GA searches, equal budgets, original training rules."""
import run_modnet_official_matched as native
from run_modnet_official_matched import (np, pd, tf, MODData, MODNetModel, FitGenetic,
    EnsembleMODNetModel, SafePool, digest, file_hash, write_json, set_seed, sample_ids,
    ModelReference, matbench_kfold_splits, predict_members)
from native_fourway_models import NativeSplineModel, matched_widths, dense_count
from native_fourway_features import ROOT, OLD, CSV, CSV_SHA, ORIGINAL, read, load_banks, make_data, extend_ranking
from official_kan_models import neurons_from_genes
from fourway_phonons_policy import factorial_effects
from modnet.hyper_opt import fit_genetic as genetic
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
import argparse
import copy
import gc
import importlib.metadata
import inspect
import os
import pickle
import shutil
import time
import traceback

ARMS = ('mm_modnet', 'mm_kan', 'mm_mv_modnet', 'mm_mv_kan')
CHECKED = set()
DATA_CACHE = {}
NativeIndividual = genetic.Individual


class IntegerBoundIndividual(NativeIndividual):
    """Fix upstream sqrt(max_feat) float randint bound for >=2000 features.

    Exact native code and RNG draws otherwise; intended square-feature range
    is 1^2 .. floor(sqrt(max_feat))^2. Pure-MM path is entirely unchanged.
    """
    def __init__(self, max_feat, *args, **kwargs):
        randint = genetic.random.randint
        if max_feat >= 2000:
            with patch.object(genetic.random, 'randint', lambda a, b: randint(a, int(b))):
                super().__init__(max_feat, *args, **kwargs)
        else:
            super().__init__(max_feat, *args, **kwargs)


# Native crossover refers to its module's Individual class. This is a local
# process adapter, never a modification of the installed source or old runs.
genetic.Individual = IntegerBoundIndividual


def now(): return datetime.now(timezone.utc).isoformat()


def check_sources(protocol_path, fp):
    if fp in CHECKED: return
    p = read(protocol_path)
    assert digest(p) == fp
    for path, sha in p['source_sha256'].items():
        if file_hash(path) != sha: raise ValueError('Frozen source changed: '+path)
    CHECKED.add(fp)


def fit_spline(individual, train, val, variant, fast=False, seed=0):
    g = individual.genes
    model = NativeSplineModel(targets=individual.targets, weights=individual.weights,
        num_classes=individual.num_classes, multi_label=False, n_feat=g['n_feat'],
        num_neurons=neurons_from_genes(g), act=g['act'], spline_seed=seed, input_scale=g['xscale'])
    model.fit(train, val_data=val, loss=g['loss'], lr=g['lr'], epochs=1 if fast else 800,
        batch_size=g['batch_size'], xscale=g['xscale'], verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30,
            min_delta=.001, restore_best_weights=False)])
    individual.val_loss = model.evaluate(val); individual.model = model


def run_fit(job):
    check_sources(job['protocol_path'], job['protocol_fingerprint'])
    directory = Path(job['directory']); path = directory/'result.json'; signature = digest(job)
    if path.exists():
        saved = read(path)
        if 'job_signature' in saved:
            assert saved['job_signature'] == signature
            assert file_hash(directory/'model.pkl') == saved['model_sha256']
            return saved
    key = (job['training_cache'], job['training_sha256'])
    if key not in DATA_CACHE:
        assert file_hash(key[0]) == key[1]
        DATA_CACHE[key] = MODData.load(key[0])
    data = DATA_CACHE[key]
    lookup = {sid:i for i,sid in enumerate(data.df_targets.index)}
    assert not set(job['train_ids']) & set(job['validation_ids'])
    assert set(job['train_ids']+job['validation_ids']) == set(lookup)
    train, val = data.split(([lookup[x] for x in job['train_ids']], [lookup[x] for x in job['validation_ids']]))
    individual = IntegerBoundIndividual(max_feat=len(data.optimal_features), num_classes=data.num_classes,
        multi_label=False, targets=[[list(data.target_names)]], weights={n:1 for n in data.target_names})
    individual.genes = dict(job['genes'])
    fit_job = dict(individual=individual, train_data=train, val_data=val, directory=str(directory),
        seed=job['seed'], fast=job['smoke'], variant=job['variant'], run_fingerprint=job['run_fingerprint'])
    original = native.fit_kan
    try:
        native.fit_kan = lambda i,t,v,variant,fast=False: fit_spline(i,t,v,variant,fast,seed=job['seed'])
        result = native.evaluate_job(fit_job)
    finally:
        native.fit_kan = original
    result.update(job_signature=signature, model_sha256=file_hash(directory/'model.pkl'),
        arm=job['arm'], fold=job['fold'], generation=job['generation'], candidate=job['candidate'], inner=job['inner'])
    if job['variant'] == 'bspline':
        widths = tuple(x[0] for x in neurons_from_genes(job['genes']))
        result.update(spline_widths=list(matched_widths(job['genes']['n_feat'], widths))+[1],
            paired_modnet_parameters=dense_count(job['genes']['n_feat'], widths))
    # Same old seeds, splits, GA and exact training implementation: the entire
    # common prefix must reproduce, not just one hand-picked configuration.
    if job['arm'] == 'mm_modnet' and not job['smoke']:
        historical = OLD/'matbench_phonons'/f"fold-{job['fold']}"/'modnet'/directory.name/'result.json'
        if historical.exists():
            old = read(historical)
            for name in ('genes', 'seed', 'train_ids', 'validation_ids', 'selected_features'):
                if result[name] != old[name]: raise ValueError('Native prefix changed: '+name)
            delta = result['validation_mae']-old['validation_mae']
            if abs(delta) > .01: raise ValueError('Native prefix training no longer reproduces: '+str(delta))
            result['historical_prefix_check'] = dict(passed=True, mae_difference=delta, source=str(historical))
    write_json(path, result)
    return result


class EqualBudgetGA(FitGenetic):
    def __init__(self, data, directory, cache, p, arm, fold, seed, progress, n_jobs):
        super().__init__(data)
        self.directory=directory; self.cache=cache; self.training_sha=file_hash(cache)
        self.p=p; self.arm=arm; self.fold=fold; self.seed=seed; self.progress=progress
        self.generation=0; self.job_by_path={}; self.records=[]; self.n_jobs=n_jobs

    def _init_run(self, n_jobs=None): self.pool=SafePool(n_jobs or self.n_jobs)

    def function_fitness(self, pop, n_jobs, nested=5, val_fraction=.1, multi_label=False, fast=False):
        assert not multi_label and nested
        splits=list(matbench_kfold_splits(self.train_data, n_splits=nested))
        jobs=[]; fp=digest(self.p)
        for candidate, individual in enumerate(pop):
            for inner,(tr,va) in enumerate(splits):
                ids=np.array(sample_ids(self.train_data))
                directory=self.directory/'fits'/f'generation-{self.generation:02d}-candidate-{candidate:02d}-inner-{inner}'
                job=dict(directory=str(directory), genes=individual.genes.copy(), seed=self.seed+self.generation*10000+candidate*100+inner,
                    arm=self.arm, variant='modnet' if self.arm.endswith('modnet') else 'bspline',
                    fold=self.fold, generation=self.generation, candidate=candidate, inner=inner,
                    train_ids=ids[tr].tolist(), validation_ids=ids[va].tolist(), smoke=self.p['smoke'],
                    training_cache=str(self.cache), training_sha256=self.training_sha,
                    protocol_path=self.p['protocol_path'], protocol_fingerprint=fp,
                    run_fingerprint=digest(dict(protocol=fp,training_data=self.training_sha)))
                jobs.append(job); self.job_by_path[str(directory/'model.pkl')]=job
        losses=np.empty((len(pop),nested)); positions={j['directory']:i for i,j in enumerate(jobs)}
        for done,result in enumerate(self.pool.imap_unordered(run_fit,jobs),1):
            candidate,inner=divmod(positions[result['directory']],nested)
            losses[candidate,inner]=result['validation_mae']; self.records.append(result)
            self.progress('training', arm=self.arm, fold=self.fold, generation=self.generation,
                generation_completed=done, generation_total=len(jobs), completed=len(self.records),
                total=self.p['fits_per_arm_fold'], last_mae=result['validation_mae'])
            print('FIT',self.arm,self.fold,self.generation,done,'/',len(jobs),result['validation_mae'],flush=True)
        models=[]; individuals=[]
        for candidate,source in enumerate(pop):
            refs=[ModelReference(str(Path(jobs[candidate*nested+i]['directory'])/'model.pkl'),
                source.targets,source.weights,source.num_classes) for i in range(nested)]
            models.append(EnsembleMODNetModel(modnet_models=refs))
            ind=copy.copy(source); ind.model=refs[-1]; ind.val_loss=losses[candidate,-1]; individuals.append(ind)
        write_json(self.directory/f'generation-{self.generation:02d}.json',dict(
            genes=[i.genes for i in pop],validation_mae=losses,mean_validation_mae=losses.mean(axis=1)))
        self.generation+=1
        return losses.mean(axis=1),np.array(models),np.array(individuals)


def freeze(out, smoke):
    old=read(OLD/'matched-protocol.json')
    assert importlib.metadata.version('modnet')=='0.1.12'
    for path,sha in old['source_sha256'].items():
        if file_hash(path)!=sha: raise ValueError('Original baseline source changed: '+path)
    verified=read(ROOT/'benchmarks/modnet-baseline-diagnosis-20261029/replay-summary.json')
    if not verified['complete'] or not verified['verified']: raise ValueError('Baseline replay not verified')
    files=[ROOT/'scripts'/name for name in ('run_native_fourway.py','native_fourway_models.py',
        'native_fourway_features.py','standalone_kan_models.py','physical_feature_modules.py',
        'fourway_phonons_policy.py','audit_native_fourway.py')]
    sources=dict(old['source_sha256']); sources.update({str(f):file_hash(f) for f in files})
    inputs={str(ORIGINAL):file_hash(ORIGINAL),str(CSV):file_hash(CSV)}
    for fold in range(5):
        for name in ('official-train-moddata.pkl.gz','outer-split.json'):
            path=OLD/'matbench_phonons'/f'fold-{fold}'/name; inputs[str(path)]=file_hash(path)
    p=dict(kind='Native-protocol no-SISSO four-arm phonons benchmark',protocol_path=str(out/'protocol.json'),
        smoke=smoke,arms=ARMS,folds=[0] if smoke else list(range(5)),source_sha256=sources,input_sha256=inputs,
        packages={n:importlib.metadata.version(n) for n in ('modnet','tensorflow','numpy','pandas','scikit-learn')},
        seed=old['seed'],ga=dict(size_pop=20,num_generations=10,nested=5,early_stopping=11,refit=False),
        fits_per_arm_fold=8 if smoke else 1000,total_fits=32 if smoke else 20000,
        training=old['training'],native_replayed_mae=verified['mae'],published_mae=verified['published_mae'],
        search='Unmodified native initialization/crossover/mutation/fitness-selection ranges, independent GA per arm. Fixed10 generations; no GA early termination. Floor sqrt(max_feat) randint bound for>=2000 feature bug only.',
        features='Original MM values+native cached OUTER TRAIN ranking; no RFE. Enhanced retains MM-MM precomputed cross-NMI and adds native entropy/MI pairs on each outer train, then native RR formula. Both networks share ranking within bank. Exclude every SISSO column.',
        kan='Latest pure3-layer B-spline (8 coefficients, cubic fixed grid, near-linear Greville init); width ratio17:8, paired native candidate trainable parameter count within5%. No Dense trunk/bypass. Standard-scaled first coordinates use0.5*tanh; minmax coordinates retain latest direct grid input. No target scaling.',
        ensemble='Native top10 candidates across all generations, all5 inner members each, no refit; smoke uses native fast2x2x2 and keeps8 members.',
        predictions='Primary raw50-member mean, never random outlier repair; native repaired predictions and count additionally saved. Original phonons baseline had zero repairs.',
        baseline_gate='All5 pure MODNet folds first. MAE must be<=published*1.02 or stop for diagnosis without test-driven retuning.',
        limitations=['Outer-training supervised ranking includes inner-validation rows, exactly as original reproduction; outer tests remain isolated.',
            'Fixed search budget extends original early-stopped GA; full four-arm baseline is recomputed, not replaced by old34.013.',
            'Previously explored dataset and KAN design; not a pristine external confirmation.',
            'Equal fits and epoch cap, not equal compute time; selected model sizes can differ after independent searches.',
            'SISSO excluded; external pretraining overlap not established.'])
    path=out/'protocol.json'
    if path.exists() and digest(read(path))!=digest(p): raise ValueError('Changed frozen protocol; use a new output')
    write_json(path,p)
    for source,sha in sources.items():
        dest=out/'source-snapshot'/(Path(source).parent.name+'--'+Path(source).name)
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists(): shutil.copyfile(source,dest)
        assert file_hash(dest)==sha
    return p


def prepare(out,p,banks,y,bank,fold,n_jobs,progress):
    source=OLD/'matbench_phonons'/f'fold-{fold}'
    cache=out/'inputs'/bank/f'fold-{fold}'/'training.pkl.gz'
    manifest=cache.parent/'manifest.json';cache.parent.mkdir(parents=True,exist_ok=True)
    split=read(source/'outer-split.json')
    if manifest.exists():
        m=read(manifest); assert m['protocol_fingerprint']==digest(p) and file_hash(cache)==m['sha256']
        return cache,split
    data=MODData.load(source/'official-train-moddata.pkl.gz')
    pd.testing.assert_frame_equal(data.df_featurized,banks['mm'].loc[split['train_ids']],check_names=False)
    np.testing.assert_array_equal(data.df_targets.values,y.loc[split['train_ids']].values)
    if bank=='mm_mv':
        enhanced=make_data(banks[bank].loc[split['train_ids']],y.loc[split['train_ids']])
        if p['smoke']:
            # Exercise the genuine cross-NMI extension, on a small training-only
            # feature bank. Formal rankings never take this branch.
            cols=list(data.optimal_features[:16])+[c for c in banks[bank] if c not in banks['mm']][:8]
            enhanced.df_featurized=enhanced.df_featurized[cols]
        def feature_status(state,**kw): progress(state,bank=bank,fold=fold,**kw)
        data=extend_ranking(enhanced,data,cache.parent/'feature-selection',p['seed']+1000000+fold*100000,n_jobs,feature_status)
    if p['smoke']:
        data.optimal_features=data.optimal_features[:24]
    if any('SISSO' in c.upper() for c in data.df_featurized): raise ValueError('SISSO contamination')
    data.save(cache)
    write_json(manifest,dict(protocol_fingerprint=digest(p),sha256=file_hash(cache),train_ids=list(data.df_targets.index),
        features=data.optimal_features,bank=bank,fold=fold,outer_test_ids=split['test_ids']))
    return cache,split


def run_fold(out,p,banks,y,arm,fold,n_jobs,progress):
    directory=out/arm/f'fold-{fold}';directory.mkdir(parents=True,exist_ok=True)
    done=directory/'complete.json'
    if done.exists():
        r=read(done); assert r['protocol_fingerprint']==digest(p)
        assert file_hash(directory/'outer-predictions.npz')==r['prediction_sha256']
        return r
    bank=arm.rsplit('_',1)[0]
    cache,split=prepare(out,p,banks,y,bank,fold,n_jobs,progress)
    data=MODData.load(cache);seed=p['seed']+1000000+fold*100000;set_seed(seed)
    ga=EqualBudgetGA(data,directory,cache,p,arm,fold,seed,progress,n_jobs)
    try:
        ensemble=ga.run(n_jobs=n_jobs,fast=p['smoke'],**p['ga'])
    except BaseException:
        if getattr(ga,'pool',None): ga.pool.terminate()
        raise
    jobs=[ga.job_by_path[ref.path] for ref in ensemble.model]
    assert len(ga.records)==p['fits_per_arm_fold']
    assert len(jobs)==(8 if p['smoke'] else 50)
    selection=dict(members=jobs,selection='Native mean-inner-MAE ranking; all budget consumed',
        protocol_fingerprint=digest(p),test_labels_used=False,
        model_sha256={j['directory']:file_hash(Path(j['directory'])/'model.pkl') for j in jobs})
    write_json(directory/'frozen-selection.json',selection)
    progress('outer_evaluation',arm=arm,fold=fold)
    test=make_data(banks[bank].loc[split['test_ids']],y.loc[split['test_ids']])
    corrected,_,replaced=predict_members(jobs,test,directory,'native')
    with np.load(directory/'native-member-predictions.npz') as z: matrix=z['raw_predictions'].copy()
    if not np.isfinite(matrix).all(): raise ValueError('Nonfinite outer prediction')
    prediction=matrix.mean(axis=0);target=test.df_targets.values[:,0]
    np.savez_compressed(directory/'outer-predictions.npz',sample_ids=split['official_test_ids'],
        target=target,prediction=prediction,members=matrix)
    selected=[read(Path(j['directory'])/'result.json') for j in jobs]
    r=dict(arm=arm,fold=fold,mae=float(np.mean(abs(prediction-target))),
        native_repaired_mae=float(np.mean(abs(corrected-target))),native_replaced_predictions=replaced,
        ensemble_size=len(jobs),search_fits=len(ga.records),search_epochs=sum(m['epochs'] for m in ga.records),
        search_seconds=sum(m['seconds'] for m in ga.records),parameters_mean=float(np.mean([m['parameters'] for m in selected])),
        prefix_members_verified=sum('historical_prefix_check' in m for m in ga.records),
        protocol_fingerprint=digest(p),prediction_sha256=file_hash(directory/'outer-predictions.npz'),
        selection_sha256=file_hash(directory/'frozen-selection.json'))
    pd.DataFrame(dict(sample_id=split['official_test_ids'],target=target,prediction=prediction)).to_csv(directory/'predictions.csv',index=False,float_format='%.17g')
    write_json(done,r);print('OUTER',arm,fold,r['mae'],flush=True)
    del ga,ensemble,data,test;gc.collect();return r


def report(out,p):
    rows=[read(out/arm/f'fold-{fold}'/'complete.json') for arm in ARMS for fold in p['folds']
          if (out/arm/f'fold-{fold}'/'complete.json').exists()]
    means={a:float(np.mean([r['mae'] for r in rows if r['arm']==a])) for a in ARMS
           if sum(r['arm']==a for r in rows)==len(p['folds'])}
    complete=len(rows)==4*len(p['folds'])
    comparisons={}
    for name,a,b in (('KAN_on_MM','mm_modnet','mm_kan'),('KAN_on_MM_MV','mm_mv_modnet','mm_mv_kan'),
                     ('MV_on_MODNet','mm_modnet','mm_mv_modnet'),('MV_on_KAN','mm_kan','mm_mv_kan')):
        if a in means and b in means:
            delta=[next(r['mae'] for r in rows if r['arm']==a and r['fold']==f)-
                   next(r['mae'] for r in rows if r['arm']==b and r['fold']==f) for f in p['folds']]
            comparisons[name]=dict(relative_improvement=1-means[b]/means[a],wins=sum(v>0 for v in delta),per_fold_improvement=delta)
    analysis=dict(complete=complete,smoke=p['smoke'],rows=rows,means=means,comparisons=comparisons,
        effects=factorial_effects(means) if complete else None,
        fold_sample_sd={a:float(np.std([r['mae'] for r in rows if r['arm']==a],ddof=1)) for a in means} if not p['smoke'] else {})
    write_json(out/'analysis.json',analysis)
    lines=['# Original-protocol four-way phonons comparison','',
        'Only complete five-fold arms receive a mean. Smoke is execution testing only.','',
        '| Arm | Complete folds | MAE |','| --- | ---: | ---: |']
    for a in ARMS: lines.append(f"| {a} | {sum(r['arm']==a for r in rows)}/{len(p['folds'])} | {means.get(a,'pending')} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return analysis


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n-jobs',type=int,default=4);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    def progress(state,**kw):write_json(out/'status.json',dict(state=state,pid=os.getpid(),updated_at=now(),wall_seconds=time.perf_counter()-start,**kw))
    try:
        progress('freezing');p=freeze(out,args.smoke);banks,y=load_banks(out);report(out,p)
        for arm in ARMS:
            for fold in p['folds']:
                run_fold(out,p,banks,y,arm,fold,args.n_jobs,progress);summary=report(out,p)
            if arm=='mm_modnet':
                mean=summary['means'][arm];passed=args.smoke or mean<=p['published_mae']*1.02
                write_json(out/'baseline-quality-gate.json',dict(passed=passed,smoke=args.smoke,mean_mae=mean,
                    maximum_mae=p['published_mae']*1.02,prefix_members_verified=sum(r['prefix_members_verified'] for r in summary['rows'] if r['arm']==arm)))
                if not passed:
                    progress('baseline_quality_requires_investigation',complete=False);return
        from audit_native_fourway import audit
        audit(out,require_complete=True)
        progress('complete',complete=True,smoke=args.smoke)
    except BaseException:
        progress('failed',complete=False,traceback=traceback.format_exc());raise


if __name__=='__main__':main()
