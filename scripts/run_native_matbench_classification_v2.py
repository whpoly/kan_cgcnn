"""Version 2: four-arm classification; reference probability-AUC schema corrected before formal runs. Worker fitting remains frozen v1."""
import argparse
import contextlib
import gc
import os
import pickle
import shutil
import time
import traceback
from pathlib import Path
import run_native_matbench_regression_expansion_v2 as r
import native_classification_features as features
import native_classification_spline as spline
from native_classification_ga import ClassificationGA
from run_modnet_official_matched import np,pd,tf,MODData,MODNetModel,set_seed,write_json,file_hash,digest
from sklearn.metrics import roc_auc_score

e=r.e;ROOT=r.ROOT;PREP=r.OLD;DATA={}
SUPPORTED={'matbench_expt_is_metal':6,'matbench_glass':7}
EPS=1e-6

def make_data(frame,target):
    return MODData(materials=[None]*len(frame),targets=target.values,target_names=list(target.columns),
        num_classes={target.columns[0]:2},structure_ids=list(frame.index),df_featurized=frame)

def run_fit(job):
    e.check_sources(job['protocol_path'],job['protocol_fingerprint'])
    d=Path(job['directory']);d.mkdir(parents=True,exist_ok=True);signature=digest(job)
    if (d/'result.json').exists():
        result=e.read(d/'result.json');assert result['job_signature']==signature
        assert file_hash(d/'model.pkl')==result['model_sha256'];return result
    start=time.monotonic();tf.keras.backend.clear_session();set_seed(job['seed'])
    key=(job['training_cache'],job['training_sha256'])
    if key not in DATA:
        assert file_hash(key[0])==key[1];DATA[key]=MODData.load(key[0])
    data=DATA[key];lookup={sid:i for i,sid in enumerate(data.df_targets.index)}
    assert set(job['train_ids']).isdisjoint(job['validation_ids'])
    assert set(job['train_ids']+job['validation_ids'])==set(lookup)
    train,val=data.split(([lookup[s] for s in job['train_ids']],[lookup[s] for s in job['validation_ids']]))
    g=job['genes'];kwargs=dict(targets=[[list(data.target_names)]],weights={n:1 for n in data.target_names},
        num_classes=data.num_classes,multi_label=False,n_feat=g['n_feat'],num_neurons=e.neurons_from_genes(g),act=g['act'])
    if job['variant']=='bspline':
        model=spline.NativeClassificationSpline(**kwargs,spline_seed=job['seed'],input_scale=g['xscale'])
    else:model=MODNetModel(**kwargs)
    set_seed(job['seed'])
    with (d/'training.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        model.fit(train,val_data=val,loss='categorical_crossentropy',lr=g['lr'],epochs=1 if job['smoke'] else 800,
            batch_size=g['batch_size'],xscale=g['xscale'],verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=.001,patience=30,restore_best_weights=False)])
    assert model.model.loss=='categorical_crossentropy' and not hasattr(model,'target_scaling')
    prediction=spline.probabilities(model,val.df_featurized)[:,1]
    target=val.df_targets.values[:,0].astype(int);auc=float(roc_auc_score(target,prediction))
    np.savez_compressed(d/'validation-predictions.npz',prediction=prediction,target=target,sample_ids=np.array(job['validation_ids']))
    result=dict(job,job_signature=signature,validation_auc=auc,selection_loss=1-auc+EPS,
        parameters=int(model.model.count_params()),epochs=len(model.history['loss']),seconds=time.monotonic()-start,
        features=list(data.optimal_features),selected_features=list(model.optimal_descriptors[:model.n_feat]),
        loss='categorical_crossentropy',target_standardization=False,
        validation_prediction_sha256=file_hash(d/'validation-predictions.npz'))
    if job['variant']=='bspline':result['parameter_mapping']=model.parameter_mapping
    model._make_picklable()
    with (d/'model.pkl.tmp').open('wb') as f:pickle.dump(model,f,protocol=4)
    (d/'model.pkl.tmp').replace(d/'model.pkl');result['model_sha256']=file_hash(d/'model.pkl')
    write_json(d/'result.json',result);del model;tf.keras.backend.clear_session();gc.collect();return result

def prepare(out,p,banks,y,bank,fold,n_jobs,progress):
    source=PREP/p['task']/f'fold-{fold}';split=e.read(source/'outer-split.json')
    cache=out/'inputs'/bank/f'fold-{fold}'/'training.pkl.gz';manifest=cache.parent/'manifest.json'
    cache.parent.mkdir(parents=True,exist_ok=True)
    if manifest.exists():
        m=e.read(manifest);assert m['protocol_fingerprint']==digest(p) and file_hash(cache)==m['sha256'];return cache,split
    data=MODData.load(source/'official-train-moddata.pkl.gz');assert list(data.num_classes.values())==[2]
    pd.testing.assert_frame_equal(data.df_featurized,banks['mm'].loc[split['train_ids']],check_names=False)
    np.testing.assert_array_equal(data.df_targets.values,y.loc[split['train_ids']].values)
    if bank=='mm_mv':
        enhanced=make_data(banks[bank].loc[split['train_ids']],y.loc[split['train_ids']])
        if p['smoke']:
            cols=list(data.optimal_features[:16])+[c for c in banks[bank] if c not in banks['mm']][:8]
            enhanced.df_featurized=enhanced.df_featurized[cols]
        data=features.extend_ranking(enhanced,data,cache.parent/'feature-selection',p['seed']+1000000+fold*100000,
            n_jobs,lambda state,**kw:progress(state,bank=bank,fold=fold,**kw))
    if p['smoke']:data.optimal_features=data.optimal_features[:24]
    data.save(cache)
    write_json(manifest,dict(protocol_fingerprint=digest(p),sha256=file_hash(cache),features=data.optimal_features,
        train_ids=split['train_ids'],outer_test_ids=split['test_ids'],classification=True))
    return cache,split

def freeze(out,task,csv,smoke):
    base=e.read(ROOT/'benchmarks/native-fourway-scaled-kan-20261029/protocol.json')
    sources=dict(base['source_sha256'])
    for name in ['run_native_matbench_classification_v2.py','run_native_matbench_classification_v1.py','native_classification_ga.py','native_classification_features.py',
                 'native_classification_spline.py','run_native_matbench_regression_expansion_v2.py','relaxed_spline_policy.py','scaled_native_spline.py']:
        f=ROOT/'scripts'/name;sources[str(f)]=file_hash(f)
    original=PREP/'precomputed'/f'{task}_moddata.pkl.gz';reference=PREP/task/'published-reference-verification.json'
    amendment=ROOT/'benchmarks/matbench-expansion-20261101/classification-metric-amendment.json'
    inputs={str(f):file_hash(f) for f in [original,csv,reference,amendment]}
    folds=[0] if smoke else list(range(5))
    for fold in folds:
        for name in ['outer-split.json','official-train-moddata.pkl.gz']:
            f=PREP/task/f'fold-{fold}'/name;inputs[str(f)]=file_hash(f)
    p=dict(task=task,metric='probability_roc_auc',protocol_path=str(out/'protocol.json'),source_sha256=sources,input_sha256=inputs,
        original=str(original),csv=str(csv),seed=20261021+SUPPORTED[task]*1000000-1000000,
        arms=list(e.ARMS),folds=folds,smoke=smoke,ga=base['ga'],fits_per_arm_fold=8 if smoke else 1000,
        training=dict(epochs=800,early_stopping_monitor='loss',patience=30,min_delta=.001,restore_best_weights=False,
                      loss='categorical_crossentropy',output='two-class softmax',target_standardization=False),
        amendments=['Explicit stratified outer/inner splits and classification target NMI.',
            'Use selection loss 1 - positive-class probability ROC-AUC + 1e-6 in all arms. Native roulette on negative AUC reverses preference; positive transform fixes this while preserving the native GA search and ranking.',
            'Recompute parameter matching for two output logits; no regression target scaling for binary labels.',
            'Keep published stored threshold-based AUC distinct from probability AUC; reference probabilities are recomputed.'],
        ensemble='Mean positive-class probability of top10 candidates by mean inner AUC, all5 members, no refit.',
        baseline_gate='Investigate if five-fold probability AUC is more than 0.02 below recomputed published probability AUC; never tune to outer labels.',
        limitations=['Outer-training supervised feature ranking is shared across inner folds; outer test remains isolated.',
            'Classification GA selection adaptation differs from published native negative-AUC roulette; shared identically by all four arms.',
            'Equal fits, not equal wall time; independent searches may select different parameter counts.',
            'No SISSO; external representation pretraining overlap not established.'])
    if (out/'protocol.json').exists():assert e.read(out/'protocol.json')==p
    write_json(out/'protocol.json',p)
    for f,h in sources.items():
        dest=out/'source-snapshot'/(Path(f).parent.name+'--'+Path(f).name);dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copyfile(f,dest)
        assert file_hash(dest)==h
    return p

def run_fold(out,p,banks,y,arm,fold,n_jobs,progress):
    d=out/arm/f'fold-{fold}';d.mkdir(parents=True,exist_ok=True)
    if (d/'complete.json').exists():
        result=e.read(d/'complete.json');assert result['protocol_fingerprint']==digest(p)
        assert file_hash(d/'outer-predictions.npz')==result['prediction_sha256'];return result
    bank=arm.rsplit('_',1)[0];cache,split=prepare(out,p,banks,y,bank,fold,n_jobs,progress)
    data=MODData.load(cache);seed=p['seed']+1000000+fold*100000;set_seed(seed)
    ga=ClassificationGA(data,d,cache,p,arm,fold,seed,progress,n_jobs)
    try:ensemble=ga.run(n_jobs=n_jobs,fast=p['smoke'],**p['ga'])
    except BaseException:
        if getattr(ga,'pool',None):ga.pool.terminate()
        raise
    jobs=[ga.job_by_path[ref.path] for ref in ensemble.model]
    assert len(ga.records)==p['fits_per_arm_fold'] and len(jobs)==(8 if p['smoke'] else 50)
    selection=dict(members=jobs,test_labels_used=False,protocol_fingerprint=digest(p),
        model_sha256={j['directory']:file_hash(Path(j['directory'])/'model.pkl') for j in jobs})
    write_json(d/'frozen-selection.json',selection);progress('outer_evaluation',arm=arm,fold=fold)
    matrix=[];test=banks[bank].loc[split['test_ids']]
    for job in jobs:
        with (Path(job['directory'])/'model.pkl').open('rb') as f:model=pickle.load(f)
        model._restore_model();matrix.append(spline.probabilities(model,test)[:,1])
        del model;tf.keras.backend.clear_session()
    matrix=np.asarray(matrix);prediction=matrix.mean(axis=0);target=y.loc[split['test_ids']].values[:,0].astype(int)
    np.savez_compressed(d/'outer-predictions.npz',prediction=prediction,target=target,members=matrix,sample_ids=np.array(split['official_test_ids']))
    result=dict(arm=arm,fold=fold,roc_auc=float(roc_auc_score(target,prediction)),ensemble_size=len(jobs),
        search_fits=len(ga.records),protocol_fingerprint=digest(p),prediction_sha256=file_hash(d/'outer-predictions.npz'),
        selection_sha256=file_hash(d/'frozen-selection.json'))
    write_json(d/'complete.json',result);del ga,ensemble,data;gc.collect();return result

def report(out,p):
    rows=[e.read(path) for arm in e.ARMS for fold in p['folds'] if (path:=out/arm/f'fold-{fold}'/'complete.json').exists()]
    means={a:float(np.mean([x['roc_auc'] for x in rows if x['arm']==a])) for a in e.ARMS if not p['smoke'] and sum(x['arm']==a for x in rows)==5}
    result=dict(task=p['task'],metric='probability_roc_auc',smoke=p['smoke'],complete=len(rows)==4*len(p['folds']),rows=rows,means=means)
    write_json(out/'analysis.json',result);return result

def audit(out,p):
    partitions={};initial={};rows=[]
    for path,sha in {**p['source_sha256'],**p['input_sha256']}.items():assert file_hash(path)==sha
    original=MODData.load(p['original'])
    official_splits=list(e.matbench_kfold_splits(original,classification=True))
    original_ids=np.array(list(original.df_targets.index))
    for arm in e.ARMS:
        for fold in p['folds']:
            d=out/arm/f'fold-{fold}';complete=e.read(d/'complete.json');split=e.read(PREP/p['task']/f'fold-{fold}'/'outer-split.json')
            otr,ote=official_splits[fold]
            assert split['train_ids']==list(original_ids[otr]) and split['test_ids']==list(original_ids[ote])
            assert complete['protocol_fingerprint']==digest(p)
            assert file_hash(d/'outer-predictions.npz')==complete['prediction_sha256']
            assert file_hash(d/'frozen-selection.json')==complete['selection_sha256']
            manifest=e.read(out/'inputs'/arm.rsplit('_',1)[0]/f'fold-{fold}'/'manifest.json')
            cache=out/'inputs'/arm.rsplit('_',1)[0]/f'fold-{fold}'/'training.pkl.gz';assert file_hash(cache)==manifest['sha256']
            data=MODData.load(cache);nested=2 if p['smoke'] else 5
            # Native FitGenetic first permutes (and caps at 5000) its training data.
            positions=np.random.RandomState(p['seed']+1000000+fold*100000).permutation(len(data.df_targets))[:5000]
            ga_data,_=data.split((positions,[]))
            expected_splits=list(e.matbench_kfold_splits(ga_data,n_splits=nested,classification=True));ids=np.array(list(ga_data.df_targets.index))
            records=[e.read(f) for f in (d/'fits').glob('*/result.json')];assert len(records)==p['fits_per_arm_fold']
            pop,gens=(2,2) if p['smoke'] else (20,10)
            assert {(m['generation'],m['candidate'],m['inner']) for m in records}=={(g,c,i) for g in range(gens) for c in range(pop) for i in range(nested)}
            groups={}
            for m in records:
                md=Path(m['directory']);assert file_hash(md/'model.pkl')==m['model_sha256']
                job_keys=['directory','genes','seed','arm','variant','fold','generation','candidate','inner',
                    'train_ids','validation_ids','smoke','training_cache','training_sha256','protocol_path',
                    'protocol_fingerprint','run_fingerprint']
                assert digest({key:m[key] for key in job_keys})==m['job_signature']
                assert file_hash(md/'validation-predictions.npz')==m['validation_prediction_sha256']
                assert m['selected_features']==manifest['features'][:m['genes']['n_feat']]
                assert not any('SISSO' in x.upper() for x in m['features'])
                tr,va=expected_splits[m['inner']];assert m['train_ids']==list(ids[tr]) and m['validation_ids']==list(ids[va])
                assert not set(m['train_ids']+m['validation_ids'])&set(split['test_ids'])
                with np.load(md/'validation-predictions.npz') as z:
                    assert list(z['sample_ids'])==m['validation_ids']
                    np.testing.assert_array_equal(z['target'],data.df_targets.loc[m['validation_ids']].values[:,0].astype(int))
                    auc=float(roc_auc_score(z['target'],z['prediction']));assert auc==m['validation_auc'] and m['selection_loss']==1-auc+EPS
                assert m['loss']=='categorical_crossentropy' and not m['target_standardization']
                if arm.endswith('kan'):
                    mapping=spline.mapping(m['genes']['n_feat'],tuple(x[0] for x in e.neurons_from_genes(m['genes'])))
                    assert mapping==m['parameter_mapping'] and mapping['spline_parameters']==m['parameters']
                if m['generation']==0:
                    key=(arm.rsplit('_',1)[0],fold,m['candidate'],m['inner'])
                    if key in initial:assert initial[key]==m['genes']
                    initial[key]=m['genes']
                groups.setdefault((m['generation'],m['candidate']),[]).append(m)
            groups=[sorted(groups[k],key=lambda x:x['inner']) for k in sorted(groups)]
            order=np.argsort([np.mean([x['selection_loss'] for x in g]) for g in groups])[:10]
            expected=[x['directory'] for i in order for x in groups[i]]
            selection=e.read(d/'frozen-selection.json');assert expected==[x['directory'] for x in selection['members']]
            assert not selection['test_labels_used'] and selection['protocol_fingerprint']==digest(p)
            for path,sha in selection['model_sha256'].items():assert file_hash(Path(path)/'model.pkl')==sha
            with np.load(d/'outer-predictions.npz') as z:
                assert list(z['sample_ids'])==split['official_test_ids']
                np.testing.assert_array_equal(z['target'],original.df_targets.loc[split['test_ids']].values[:,0].astype(int))
                assert np.isfinite(z['members']).all() and (z['members']>=0).all() and (z['members']<=1).all()
                np.testing.assert_array_equal(z['prediction'],z['members'].mean(axis=0))
                assert float(roc_auc_score(z['target'],z['prediction']))==complete['roc_auc']
            rows.append(dict(arm=arm,fold=fold,fits=len(records)))
    write_json(out/'artifact-audit.json',dict(passed=True,complete=True,smoke=p['smoke'],metric='probability_roc_auc',rows=rows))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=SUPPORTED,required=True);ap.add_argument('--csv',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--n-jobs',type=int,default=3);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    def progress(state,**kw):write_json(out/'status.json',dict(state=state,pid=os.getpid(),task=args.task,updated_at=e.now(),**kw))
    try:
        progress('freezing');p=freeze(out,args.task,args.csv.resolve(),args.smoke);banks,y=r.load_banks(p,out)
        for arm in e.ARMS:
            for fold in p['folds']:run_fold(out,p,banks,y,arm,fold,args.n_jobs,progress);summary=report(out,p)
            if arm=='mm_modnet' and not args.smoke:
                reference=e.read(PREP/args.task/'published-reference-verification.json')
                published=float(np.mean([x['recomputed_probability_rocauc'] for x in reference['folds']]))
                passed=summary['means'][arm]>=published-.02
                write_json(out/'baseline-quality-gate.json',dict(passed=passed,mean_auc=summary['means'][arm],reference_probability_auc=published))
                if not passed:progress('baseline_quality_requires_investigation');return
        progress('auditing');audit(out,p);progress('complete',complete=True,smoke=args.smoke)
    except BaseException:progress('failed',traceback=traceback.format_exc());raise

if __name__=='__main__':main()
