"""Frozen standalone KAN/MLP nested confirmation and conditional outer fold 0."""
from run_modnet_official_matched import SafePool,digest,file_hash,set_seed,write_json
from run_standalone_kan_linear_init import linear_initialized_model
from standalone_kan_models import build_model as original_builder,COUNTS,PHYSICAL_FEATURES,export_network,np,tf
import run_standalone_kan_screen as engine
from run_mattervial_quick_pilot import select_once,load_training_rows,CSV,CSV_SHA
from run_mattervial_kan_refinement import target_statistics
from physical_feature_modules import fit_minmax
from standalone_kan_confirmation_policy import BANKS,VARIANTS,LRS,TUNE_SEEDS,CONFIRM_SEEDS,OUTER_SEEDS,SPLIT_SEED,SUBSPLIT_SEED,GATE,make_splits,choose,evaluate_gate
from pathlib import Path
from datetime import datetime,timezone
import argparse
import gc
import importlib.metadata
import json
import os
import pickle
import time
import traceback
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'benchmarks/standalone-kan-linear-init-20261027'
DATA=ROOT/'benchmarks/mattervial-quick-pilot-20261026'
OUTER_SPLIT=ROOT/'benchmarks/mattervial-phonons-20261023/run/paper_features_sisso_unverified/fold-0/outer-split.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()


def build_candidate(bank,variant,seed):
    return linear_initialized_model(bank,variant,seed) if variant=='kan' else original_builder(bank,variant,seed)


engine.build_model=build_candidate


def freeze(out,outer_action):
    old=read(OLD/'protocol.json')
    for name,value in old['source_sha256'].items():
        if file_hash(ROOT/'scripts'/name)!=value:raise ValueError('Changed frozen dependency: '+name)
    input_audit=read(DATA/'input-audit.json')
    cache=DATA/'training-only-no-sisso.pkl.gz'
    if file_hash(cache)!=input_audit['training_cache_sha256']:raise ValueError('Training cache changed')
    # Only split IDs are used; old feature lists do not enter the experiment.
    outer=read(OUTER_SPLIT);ids=sorted(outer['train_ids']);splits=make_splits(ids)
    if len(ids)!=1012 or len(outer['test_ids'])!=253 or set(ids)&set(outer['test_ids']):raise ValueError('Outer split')
    prior=old['parent_protocol']['source_inputs']
    if set(ids)!=set(prior[0]['source']['train_ids']+prior[0]['source']['validation_ids']):raise ValueError('Universe changed')
    for split in splits:
        if any(set(split['validation_ids'])==set(i['source']['validation_ids']) for i in prior):raise ValueError('Reused split')
    dependencies=dict(old['source_sha256'])
    for name in ('run_standalone_kan_confirmation.py','standalone_kan_confirmation_policy.py',
                 'run_mattervial_quick_pilot.py','run_mattervial_phonons.py'):
        dependencies[name]=file_hash(ROOT/'scripts'/name)
    p=dict(kind='Nested resampling confirmation of frozen standalone KAN vs parameter-matched MLP',
        frozen_architecture_source_sha256=file_hash(OLD/'protocol.json'),source_sha256=dependencies,
        source_cache=str(cache),source_cache_sha256=file_hash(cache),source_input_audit_sha256=file_hash(DATA/'input-audit.json'),
        csv=str(CSV),csv_sha256=CSV_SHA,outer_fold=0,outer_train_ids=ids,outer_test_ids=outer['test_ids'],
        outer_split_sha256=file_hash(OUTER_SPLIT),splits=splits,split_seed=SPLIT_SEED,subsplit_seed=SUBSPLIT_SEED,
        banks=BANKS,variants=VARIANTS,learning_rates=LRS,tuning_seeds=TUNE_SEEDS,confirmation_seeds=CONFIRM_SEEDS,outer_seeds=OUTER_SEEDS,
        topology=dict(enhanced_kan=[256,17,8,1],physical_kan=[16,17,8,1],enhanced_mlp=[256,96,72,36,45,1],physical_mlp=[16,75,27,1]),
        initializer=old['initializer'],expected_parameters={b:{v:COUNTS[b][v] for v in VARIANTS} for b in BANKS},
        optimizer='Adam',loss='train-standardized target MAE',max_epochs=400,batch_size=32,
        tuning_early_stop=dict(patience=40,min_delta_raw=.05,restore='exact lowest inner tuning validation MAE'),
        tuning_budget=dict(lr_candidates_per_model=3,seeds_per_candidate=2,confirmation_folds=5,banks=2,models=2,total_fits=120),
        confirmation_budget=dict(seeds=2,folds=5,banks=2,models=2,total_fits=40),
        selection='Per confirmation fold/bank/model: select LR by mean tuning-validation MAE across 2 seeds; ties choose smaller LR. Fixed refit epochs = rounded median best epoch of those two tuning fits.',
        confirmation='Refit on full confirmation-training partition with new seeds and fixed selected epochs. No confirmation validation_data, early stopping, checkpoint selection, or hyperparameter changes.',
        preprocessing='Recompute XGB gain ranking (100 trees), fixed 16 physical + top 240 on each fitting partition only; no reuse of previously selected features. Fixed zero imputation; train-only minmax; train-only target mean/std.',
        selector_seeds=dict(tuning=[780281+i for i in range(5)],confirmation=[780381+i for i in range(5)],outer=780481),
        no_sisso=True,gate=GATE,outer_action=outer_action,
        outer_rule='Only after enhanced confirmation gate passes: choose each LR from all 10 tuning fits per LR (5 subpartitions x 2 seeds); fixed epochs = rounded median selected-LR best epoch; recompute preprocessing on all 1012 outer-training samples; refit two new seeds and evaluate official fold 0 once. Report member scores and fixed two-member ensemble for both models and both banks.',
        packages={n:importlib.metadata.version(n) for n in ('modnet','tensorflow','numpy','pandas','scikit-learn','xgboost')},
        limitations=['New partitions/seeds, same previously explored 1012 samples: resampling stability, not a new independent dataset.',
            'Candidate architecture was chosen from earlier exploratory results; equal prospective budget does not erase past asymmetric search.',
            'Single tuning holdout within each confirmation-training fold to bound compute; not exhaustive tuning.',
            'Frozen embedding pretraining provenance remains inherited; removing SISSO alone does not establish absence of pretraining overlap.',
            'Gate is a compute-allocation/stability heuristic, not a significance test; ten fits are correlated.',
            'Only official outer fold 0 is conditionally evaluated; not a complete five-fold benchmark.'])
    path=out/'protocol.json'
    if path.exists() and digest(read(path))!=digest(p):raise ValueError('Protocol changed; use new directory')
    write_json(path,p)
    return p


def prepare_input(job):
    out=Path(job['directory']);out.mkdir(parents=True,exist_ok=True);signature=digest(job)
    if (out/'manifest.json').exists():
        saved=read(out/'manifest.json')
        if saved['signature']!=signature:raise ValueError('Changed preprocessing job')
        for name,value in saved['artifacts'].items():
            if file_hash(out/name)!=value:raise ValueError('Changed preprocessing artifact')
        return saved
    start=time.perf_counter()
    if file_hash(job['cache'])!=job['cache_sha256']:raise ValueError('Changed cache')
    frame=pd.read_pickle(job['cache'])
    if sorted(frame.index.tolist())!=sorted(job['allowed_cache_ids']):raise ValueError('Unexpected cache scope')
    columns=[c for c in frame if c!='target']
    if any('SISSO' in c.upper() for c in columns):raise ValueError('SISSO present')
    tr=frame.loc[job['train_ids'],columns];y=frame.loc[job['train_ids'],'target'].values
    if set(job['train_ids'])&set(job['validation_ids']):raise ValueError('Preprocessing overlap')
    selected,importance=select_once(tr,y,256,job['selector_seed'])
    if job['phase']=='outer':
        gate=read(job['gate_path'])
        if file_hash(job['gate_path'])!=job['gate_sha256'] or not gate['passed']:raise ValueError('Outer gate not passed')
        if file_hash(CSV)!=CSV_SHA:raise ValueError('Changed CSV')
        frame,_=load_training_rows(CSV,job['validation_ids'])
    va=frame.loc[job['validation_ids'],selected];yv=frame.loc[job['validation_ids'],'target'].values
    scaler,x,xv=fit_minmax(tr[selected].values,va.values)
    mean,std=target_statistics(y)
    np.savez_compressed(out/'arrays.npz',x_train=x,y_train=y,x_validation=xv,y_validation=yv)
    with (out/'scaler.pkl').open('wb') as f:pickle.dump(scaler,f,protocol=4)
    saved=dict(signature=signature,run_fingerprint=job['run_fingerprint'],directory=str(out),phase=job['phase'],inner=job['inner'],
        selected_features=selected,train_ids=job['train_ids'],validation_ids=job['validation_ids'],
        feature_selection_fit_ids=job['train_ids'],scaling_fit_ids=job['train_ids'],target_transform_fit_ids=job['train_ids'],
        selector_seed=job['selector_seed'],target_mean=mean,target_std=std,candidate_feature_count=len(columns),
        importance=importance,seconds=time.perf_counter()-start,
        artifacts={name:file_hash(out/name) for name in ('arrays.npz','scaler.pkl')})
    write_json(out/'manifest.json',saved)
    return saved


def as_input(info):
    manifest=Path(info['directory'])/'manifest.json'
    return dict(source_manifest=str(manifest),source_manifest_sha256=file_hash(manifest),source=info,
                target_mean=info['target_mean'],target_std=info['target_std'],target_transform_fit_ids=info['train_ids'])


def fixed_fit(job):
    """Evaluation labels and features are first opened AFTER all weight updates."""
    out=Path(job['directory']);out.mkdir(parents=True,exist_ok=True);signature=digest(job)
    if (out/'result.json').exists():
        r=read(out/'result.json')
        if r['signature']!=signature:raise ValueError('Changed fixed-fit job')
        for name,value in r['artifacts'].items():
            if file_hash(out/name)!=value:raise ValueError('Changed fixed-fit artifact')
        return r
    start=time.perf_counter();inp=job['input'];info=inp['source']
    if file_hash(inp['source_manifest'])!=inp['source_manifest_sha256']:raise ValueError('Changed input manifest')
    array=Path(info['directory'])/'arrays.npz'
    if file_hash(array)!=info['artifacts']['arrays.npz']:raise ValueError('Changed arrays')
    features=info['selected_features'] if job['bank']=='enhanced' else PHYSICAL_FEATURES
    indices=[info['selected_features'].index(f) for f in features]
    with np.load(array) as a:x=a['x_train'][:,indices];y=a['y_train']
    mean,std=target_statistics(y)
    if (mean,std)!=(inp['target_mean'],inp['target_std']):raise ValueError('Target scaling mismatch')
    model=build_candidate(job['bank'],job['variant'],job['seed'])
    model.compile(optimizer=tf.keras.optimizers.Adam(job['lr']),loss='mae',metrics=['mae'])
    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            if not np.isfinite(logs['loss']):raise ValueError('Nonfinite training loss')
            if (epoch+1)%50==0:write_json(out/'status.json',dict(state='training',epoch=epoch+1,fixed_epochs=job['fixed_epochs'],updated_at=now()))
    set_seed(job['seed'])
    history=model.fit(x,((y-mean)/std).astype('float32'),epochs=job['fixed_epochs'],batch_size=32,
                      verbose=0,callbacks=[Progress()])
    training_finished=now()
    with (out/'model.pkl').open('wb') as f:
        pickle.dump(dict(model_json=model.to_json(),weights=model.get_weights(),target_mean=mean,target_std=std),f,protocol=4)
    # This cannot feed validation data/labels back into model.fit or checkpoint choice.
    with np.load(array) as a:xv=a['x_validation'][:,indices]
    prediction=model.predict(xv,verbose=0).reshape(-1).astype(float)*std+mean
    train_prediction=model.predict(x,verbose=0).reshape(-1).astype(float)*std+mean
    with np.load(array) as a:yv=a['y_validation']
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite prediction')
    np.savez_compressed(out/'predictions.npz',prediction=prediction,target=yv,sample_ids=info['validation_ids'],
                        train_prediction=train_prediction,train_target=y,train_ids=info['train_ids'])
    artifacts=['model.pkl','predictions.npz']
    if job['variant']=='kan':
        network=export_network(model,features,x);network.update(target_mean=mean,target_std=std)
        write_json(out/'network.json',network);artifacts.append('network.json')
    r=dict(signature=signature,run_fingerprint=job['run_fingerprint'],directory=str(out),phase=job['phase'],bank=job['bank'],variant=job['variant'],
        repeat=job['repeat'],inner=job['inner'],seed=job['seed'],lr=job['lr'],selection_sha256=job['selection_sha256'],
        input_digest=digest(inp),features=features,train_ids=info['train_ids'],validation_ids=info['validation_ids'],
        target_transform_fit_ids=info['train_ids'],target_mean=mean,target_std=std,parameters=model.count_params(),
        epochs=len(history.history['loss']),fixed_epochs=job['fixed_epochs'],training_finished_at=training_finished,
        evaluation_labels_used_during_training=False,validation_data_passed_to_fit=False,history=history.history,
        validation_mae=float(np.mean(abs(prediction-yv))),train_mae=float(np.mean(abs(train_prediction-y))),seconds=time.perf_counter()-start,
        artifacts={name:file_hash(out/name) for name in artifacts})
    if r['parameters']!=job['expected_parameters'] or r['epochs']!=job['fixed_epochs']:raise ValueError('Fit budget or capacity')
    write_json(out/'result.json',r);write_json(out/'status.json',dict(state='complete',updated_at=now()))
    del model,history;tf.keras.backend.clear_session();gc.collect()
    return r


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n-jobs',type=int,default=4);parser.add_argument('--outer-action',choices=['fold0','defer'],default='fold0')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    def status(state,**kwargs):write_json(out/'status.json',dict(state=state,pid=os.getpid(),updated_at=now(),wall_seconds=time.perf_counter()-start,**kwargs))
    try:
        status('freezing',completed=0,total=160,outer_test_evaluated=False)
        p=freeze(out,args.outer_action);fp=digest(p)
        def prep_job(phase,inner,tr,va,seed):
            return dict(directory=str(out/'inputs'/phase/f'inner-{inner}'),phase=phase,inner=inner,train_ids=tr,validation_ids=va,
                cache=p['source_cache'],cache_sha256=p['source_cache_sha256'],allowed_cache_ids=p['outer_train_ids'],selector_seed=seed,run_fingerprint=fp)
        jobs=[]
        for split in p['splits']:
            i=split['inner']
            jobs += [prep_job('tuning',i,split['tuning_train_ids'],split['tuning_validation_ids'],p['selector_seeds']['tuning'][i]),
                     prep_job('confirmation',i,split['train_ids'],split['validation_ids'],p['selector_seeds']['confirmation'][i])]
        inputs={}
        with SafePool(min(2,args.n_jobs)) as pool:
            for info in pool.imap_unordered(prepare_input,jobs):
                inputs[(info['phase'],info['inner'])]=as_input(info)
                status('preparing_inputs',prepared=len(inputs),total_inputs=10,completed=0,total=160,outer_test_evaluated=False)
                print('INPUT',len(inputs),'/10',info['phase'],info['inner'],flush=True)
        def fit_job(phase,bank,variant,repeat,inner,seed,lr,inp):
            return dict(directory=str(out/'fits'/phase/bank/variant/f'lr-{lr:g}'/f'seed-{repeat}-inner-{inner}'),
                bank=bank,variant=variant,repeat=repeat,inner=inner,seed=seed,lr=lr,max_epochs=400,
                expected_parameters=COUNTS[bank][variant],input=inp,run_fingerprint=fp)
        tuning=[]
        jobs=[fit_job('tuning',b,v,s,i,seed+i,lr,inputs[('tuning',i)])
              for i in range(5) for lr in LRS for s,seed in enumerate(TUNE_SEEDS) for b in BANKS for v in VARIANTS]
        with SafePool(args.n_jobs) as pool:
            for r in pool.imap_unordered(engine.train_job,jobs):
                tuning.append(r);status('tuning',completed=len(tuning),total=160,phase_completed=len(tuning),phase_total=120,outer_test_evaluated=False)
                print('TUNE',len(tuning),'/120',r['bank'],r['variant'],r['inner'],r['lr'],r['repeat'],r['validation_mae'],flush=True)
        selected=[]
        for i in range(5):
            for b in BANKS:
                for v in VARIANTS:
                    rows=[r for r in tuning if r['inner']==i and r['bank']==b and r['variant']==v]
                    selected.append(dict(inner=i,bank=b,variant=v,**choose(rows,2)))
        selection=dict(run_fingerprint=fp,selected_at=now(),choices=selected,tuning_results_sha256={r['directory']:file_hash(Path(r['directory'])/'result.json') for r in sorted(tuning,key=lambda r:r['directory'])})
        path=out/'selection.json'
        if path.exists():selection['selected_at']=read(path)['selected_at']
        write_json(path,selection);selection_sha=file_hash(path)
        jobs=[]
        for c in selected:
            for s,seed in enumerate(CONFIRM_SEEDS):
                j=fit_job('confirmation',c['bank'],c['variant'],s,c['inner'],seed+c['inner'],c['lr'],inputs[('confirmation',c['inner'])])
                j.update(phase='confirmation',fixed_epochs=c['epochs'],selection_sha256=selection_sha);jobs.append(j)
        confirmation=[]
        with SafePool(args.n_jobs) as pool:
            for r in pool.imap_unordered(fixed_fit,jobs):
                confirmation.append(r);status('confirmation',completed=120+len(confirmation),total=160,phase_completed=len(confirmation),phase_total=40,outer_test_evaluated=False)
                print('CONFIRM',len(confirmation),'/40',r['bank'],r['variant'],r['inner'],r['repeat'],r['validation_mae'],flush=True)
        gate=evaluate_gate(confirmation);gate.update(run_fingerprint=fp,selection_sha256=selection_sha,decided_at=now(),
            confirmation_results_sha256={r['directory']:file_hash(Path(r['directory'])/'result.json') for r in sorted(confirmation,key=lambda r:r['directory'])})
        gatepath=out/'gate.json'
        if gatepath.exists():gate['decided_at']=read(gatepath)['decided_at']
        write_json(gatepath,gate);print('GATE',json.dumps(gate['rows']),flush=True)
        outer=[]
        if gate['passed'] and args.outer_action=='fold0':
            choices=[dict(bank=b,variant=v,**choose([r for r in tuning if r['bank']==b and r['variant']==v],10)) for b in BANKS for v in VARIANTS]
            op=out/'outer-selection.json';write_json(op,dict(run_fingerprint=fp,choices=choices,gate_sha256=file_hash(gatepath)))
            job=prep_job('outer',0,p['outer_train_ids'],p['outer_test_ids'],p['selector_seeds']['outer'])
            job.update(gate_path=str(gatepath),gate_sha256=file_hash(gatepath))
            status('preparing_outer',completed=160,total=168,gate_passed=True,outer_test_evaluated=False)
            inp=as_input(prepare_input(job));jobs=[]
            for c in choices:
                for s,seed in enumerate(OUTER_SEEDS):
                    j=fit_job('outer',c['bank'],c['variant'],s,0,seed,c['lr'],inp)
                    j.update(phase='outer',fixed_epochs=c['epochs'],selection_sha256=file_hash(op));jobs.append(j)
            with SafePool(args.n_jobs) as pool:
                for r in pool.imap_unordered(fixed_fit,jobs):
                    outer.append(r);status('outer',completed=160+len(outer),total=168,gate_passed=True,outer_test_evaluated=True)
                    print('OUTER',len(outer),'/8',r['bank'],r['variant'],r['repeat'],r['validation_mae'],flush=True)
            outer_summary=[]
            for b in BANKS:
                for v in VARIANTS:
                    rs=sorted([r for r in outer if r['bank']==b and r['variant']==v],key=lambda r:r['repeat'])
                    ps=[]
                    for r in rs:
                        with np.load(Path(r['directory'])/'predictions.npz') as a:ps.append(a['prediction']);target=a['target'];ids=a['sample_ids']
                    pred=np.mean(ps,axis=0)
                    np.savez_compressed(out/f'outer-{b}-{v}-ensemble.npz',sample_ids=ids,prediction=pred,target=target)
                    outer_summary.append(dict(bank=b,variant=v,member_mae=[r['validation_mae'] for r in rs],ensemble_mae=float(np.mean(abs(pred-target)))))
            write_json(out/'outer-summary.json',dict(run_fingerprint=fp,outer_fold=0,rows=outer_summary,full_benchmark=False))
        status('complete',completed=160+len(outer),total=160+len(outer),gate_passed=gate['passed'],outer_test_evaluated=bool(outer),
            outer_deferred=bool(gate['passed'] and args.outer_action=='defer'))
    except Exception:status('failed',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
