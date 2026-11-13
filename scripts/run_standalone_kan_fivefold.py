"""Extend the frozen enhanced KAN/MLP protocol to the remaining official folds."""
import run_standalone_kan_confirmation as base
from run_modnet_official_matched import SafePool,write_json,file_hash,digest
from standalone_kan_confirmation_policy import make_splits,choose
from standalone_kan_fivefold_policy import official_ids,validate_fivefold
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import time
import traceback

ROOT=base.ROOT
SOURCE=ROOT/'benchmarks/standalone-kan-confirmation-20261028'
OFFICIAL=ROOT/'benchmarks/official-aligned-20261021/matbench_phonons'
read,now,np=base.read,base.now,base.np


def snapshot_fold0():
    files=[SOURCE/n for n in ('protocol.json','gate.json','outer-selection.json','outer-summary.json','analysis.json','status.json')]
    for phase in ('tuning','outer'):
        results=sorted((SOURCE/'fits'/phase/'enhanced').glob('*/*/*/result.json'))
        if len(results)!=(60 if phase=='tuning' else 4):raise ValueError('Incomplete preserved enhanced fits')
        for result in results:
            files.append(result)
            for name,value in read(result)['artifacts'].items():
                if file_hash(result.parent/name)!=value:raise ValueError('Changed preserved fit artifact')
                files.append(result.parent/name)
        for manifest in sorted((SOURCE/'inputs'/phase).glob('*/manifest.json')):
            files.append(manifest)
            for name,value in read(manifest)['artifacts'].items():
                if file_hash(manifest.parent/name)!=value:raise ValueError('Changed preserved input')
                files.append(manifest.parent/name)
    return {str(path):file_hash(path) for path in sorted(set(files))}


def freeze(out):
    prior=read(SOURCE/'protocol.json');analysis=read(SOURCE/'analysis.json')
    if not analysis['complete'] or analysis['verified_fits']!=168 or not read(SOURCE/'gate.json')['passed']:
        raise ValueError('Prior verification/gate incomplete')
    for name,value in prior['source_sha256'].items():
        if file_hash(ROOT/'scripts'/name)!=value:raise ValueError('Frozen source changed: '+name)
    if file_hash(base.CSV)!=base.CSV_SHA:raise ValueError('Published source changed')
    folds=[]
    for f in range(5):
        path=OFFICIAL/f'fold-{f}'/'outer-split.json';tr,te=official_ids(read(path))
        folds.append(dict(fold=f,train_ids=tr,test_ids=te,source=str(path),source_sha256=file_hash(path)))
    validate_fivefold(folds)
    if folds[0]['train_ids']!=prior['outer_train_ids'] or folds[0]['test_ids']!=prior['outer_test_ids']:
        raise ValueError('First fold does not match preserved run')
    dependencies=dict(prior['source_sha256'])
    for name in ('run_standalone_kan_fivefold.py','standalone_kan_fivefold_policy.py'):
        dependencies[name]=file_hash(ROOT/'scripts'/name)
    p=dict(kind='Frozen enhanced standalone KAN vs matched MLP, complete official five-fold evaluation',
        prior_protocol_sha256=file_hash(SOURCE/'protocol.json'),fold0_sources=snapshot_fold0(),source_sha256=dependencies,
        data_source=str(base.CSV),csv_sha256=base.CSV_SHA,folds=folds,additional_folds=[1,2,3,4],bank='enhanced',
        topology={k:prior['topology'][k] for k in ('enhanced_kan','enhanced_mlp')},initializer=prior['initializer'],
        learning_rates=prior['learning_rates'],tuning_seeds=prior['tuning_seeds'],outer_seeds=prior['outer_seeds'],
        max_epochs=prior['max_epochs'],batch_size=prior['batch_size'],optimizer=prior['optimizer'],loss=prior['loss'],
        tuning_early_stop=prior['tuning_early_stop'],expected_parameters={'kan':35994,'matched_mlp':35995},
        split_seed=prior['split_seed'],subsplit_seed=prior['subsplit_seed'],selector_seeds=prior['selector_seeds'],
        selection='Same as fold 0: within each outer training set, make five inner partitions; in the training part of each, make an 80/20 tuning holdout. Choose LR using mean of 5 holdouts x 2 seeds; ties select smaller LR. Fixed full-training epochs = rounded median selected-LR best epoch across those 10 fits.',
        preprocessing=prior['preprocessing'],no_sisso=True,split_rule='Reuse frozen make_splits and seed lists on each distinct outer training universe',
        per_fold_budget=dict(tuning_per_model=30,refits_per_model=2,total=64),additional_fits=256,
        core_fivefold_fits=320,prior_confirmation_and_other_feature_bank_fits_excluded_from_core_timing=True,
        evaluate_all_folds=True,per_fold_gate=False,
        metrics=dict(primary='Mean of five outer-fold MAEs of fixed two-seed mean prediction',
                     secondary='Each seed separately and mean single-member MAE; never select a winning seed',
                     dispersion='Sample SD across five fold MAEs (ddof=1), not confidence interval',
                     improvement='MLP MAE minus KAN MAE; positive favors KAN',
                     timing='Sum per-fit worker elapsed seconds for 30 tuning + 2 outer refit jobs per model/fold; input preparation shared and reported separately; elapsed wall time reported separately'),
        original_gate_path=str(SOURCE/'gate.json'),original_gate_sha256=file_hash(SOURCE/'gate.json'),
        packages={n:importlib.metadata.version(n) for n in ('modnet','tensorflow','numpy','pandas','scikit-learn','xgboost')},
        limitations=['Architecture was developed using fold-0 training samples, which overlap other outer test folds. This is a frozen-procedure five-fold evaluation, not a prospectively untouched external confirmation; per-fold fitting and selection remain isolated.',
            'Expansion was decided after the first outer fold result. All remaining folds are reported without conditional exclusion or adapting hyperparameter search.',
            'Pretrained embedding provenance remains inherited; absence of SISSO does not establish no pretraining overlap.',
            'The inner tuning subsets use about 64% of each outer training universe, exactly as preserved fold 0; not an exhaustive hyperparameter search.',
            'Parallel CPU task durations include construction, prediction and serialization, not isolated model.fit time; unrelated formal replication remains running.'])
    path=out/'protocol.json'
    if path.exists() and digest(read(path))!=digest(p):raise ValueError('Frozen extension protocol changed')
    write_json(path,p);return p,prior


def fold_protocol(root,global_protocol,prior,fold):
    spec=global_protocol['folds'][fold]
    p={key:prior[key] for key in ('source_sha256','learning_rates','tuning_seeds','outer_seeds','max_epochs','batch_size',
        'expected_parameters','selector_seeds','optimizer','loss','tuning_early_stop','initializer')}
    p.update(outer_fold=fold,bank='enhanced',outer_train_ids=spec['train_ids'],outer_test_ids=spec['test_ids'],
        splits=make_splits(spec['train_ids']),source_sha256=global_protocol['source_sha256'],
        parent_protocol_sha256=file_hash(root/'protocol.json'),source_cache=str(root/f'fold-{fold}'/'training-only-no-sisso.pkl.gz'),
        csv_sha256=global_protocol['csv_sha256'],selection=global_protocol['selection'])
    return p


def prepare_fold(root,global_protocol,prior,fold):
    directory=root/f'fold-{fold}';directory.mkdir(exist_ok=True);p=fold_protocol(root,global_protocol,prior,fold)
    path=directory/'data-audit.json';cache=Path(p['source_cache'])
    identity=digest(p)
    if path.exists():
        audit=read(path)
        if audit['identity']!=identity or file_hash(cache)!=audit['source_cache_sha256']:raise ValueError('Changed fold training cache')
    else:
        frame,removed=base.load_training_rows(base.CSV,p['outer_train_ids'])
        if len(frame)!=1012 or set(frame.index)&set(p['outer_test_ids']):raise ValueError('Outer test leaked into fold cache')
        frame.to_pickle(cache,compression='gzip',protocol=4)
        audit=dict(identity=identity,source_cache_sha256=file_hash(cache),train_ids=p['outer_train_ids'],
            excluded_sisso_count=len(removed),candidate_features=len(frame.columns)-1,outer_targets_converted_or_stored=False)
        write_json(path,audit)
    p['source_cache_sha256']=audit['source_cache_sha256'];protocol=directory/'protocol.json'
    if protocol.exists() and digest(read(protocol))!=digest(p):raise ValueError('Changed fold protocol')
    write_json(protocol,p);return p


def input_job(directory,p,phase,inner,tr,va):
    selector=p['selector_seeds']['tuning'][inner] if phase=='tuning' else p['selector_seeds']['outer']
    job=dict(directory=str(directory/'inputs'/phase/f'inner-{inner}'),phase=phase,inner=inner,train_ids=tr,validation_ids=va,
        cache=p['source_cache'],cache_sha256=p['source_cache_sha256'],allowed_cache_ids=p['outer_train_ids'],selector_seed=selector,run_fingerprint=digest(p))
    if phase=='outer':job.update(gate_path=str(SOURCE/'gate.json'),gate_sha256=file_hash(SOURCE/'gate.json'))
    return job


def fit_job(directory,p,phase,variant,repeat,inner,lr,inp,choice=None):
    seed=p['tuning_seeds'][repeat]+inner if phase=='tuning' else p['outer_seeds'][repeat]
    job=dict(directory=str(directory/'fits'/phase/'enhanced'/variant/f'lr-{lr:g}'/f'seed-{repeat}-inner-{inner}'),
        bank='enhanced',variant=variant,repeat=repeat,inner=inner,seed=seed,lr=lr,max_epochs=p['max_epochs'],
        expected_parameters=p['expected_parameters']['enhanced'][variant],input=inp,run_fingerprint=digest(p))
    if phase=='outer':job.update(phase='outer',fixed_epochs=choice['epochs'],selection_sha256=file_hash(directory/'outer-selection.json'))
    return job


def summarize_outer(directory,p,records):
    rows=[]
    for variant in ('kan','matched_mlp'):
        rs=sorted([r for r in records if r['variant']==variant],key=lambda r:r['repeat'])
        if len(rs)!=2:raise ValueError('Two members required')
        predictions=[];target=None;ids=None
        for r in rs:
            with np.load(Path(r['directory'])/'predictions.npz') as a:
                if ids is not None:
                    np.testing.assert_array_equal(ids,a['sample_ids']);np.testing.assert_array_equal(target,a['target'])
                predictions.append(a['prediction']);ids=a['sample_ids'];target=a['target']
        pred=np.mean(predictions,axis=0)
        np.savez_compressed(directory/f'outer-enhanced-{variant}-ensemble.npz',sample_ids=ids,prediction=pred,target=target)
        rows.append(dict(bank='enhanced',variant=variant,member_mae=[r['validation_mae'] for r in rs],
                         ensemble_mae=float(np.mean(abs(pred-target)))))
    write_json(directory/'outer-summary.json',dict(run_fingerprint=digest(p),outer_fold=p['outer_fold'],rows=rows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--n-jobs',type=int,default=4)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    completed=0;complete_folds=[]
    def status(state,**kwargs):write_json(out/'status.json',dict(state=state,pid=os.getpid(),updated_at=now(),wall_seconds=time.perf_counter()-start,
        completed=completed,total=256,completed_additional_folds=complete_folds,preserved_outer_fold=0,**kwargs))
    try:
        status('freezing');global_protocol,prior=freeze(out)
        for fold in (1,2,3,4):
            directory=out/f'fold-{fold}';foldstart=time.perf_counter();status('preparing_fold_cache',current_outer_fold=fold)
            p=prepare_fold(out,global_protocol,prior,fold);inputs={}
            jobs=[input_job(directory,p,'tuning',s['inner'],s['tuning_train_ids'],s['tuning_validation_ids']) for s in p['splits']]
            with SafePool(min(2,args.n_jobs)) as pool:
                for info in pool.imap_unordered(base.prepare_input,jobs):
                    inputs[info['inner']]=base.as_input(info);status('preparing_inputs',current_outer_fold=fold,prepared=len(inputs),total_inputs=5)
                    print('INPUT',fold,len(inputs),'/5',flush=True)
            jobs=[fit_job(directory,p,'tuning',v,s,i,lr,inputs[i]) for i in range(5) for lr in p['learning_rates']
                  for s in range(2) for v in ('kan','matched_mlp')]
            tuning=[]
            with SafePool(args.n_jobs) as pool:
                for r in pool.imap_unordered(base.engine.train_job,jobs):
                    tuning.append(r);completed+=1;status('tuning',current_outer_fold=fold,phase_completed=len(tuning),phase_total=60)
                    print('TUNE',fold,len(tuning),'/60',r['variant'],r['inner'],r['repeat'],r['lr'],r['validation_mae'],flush=True)
            choices=[dict(bank='enhanced',variant=v,**choose([r for r in tuning if r['variant']==v],10)) for v in ('kan','matched_mlp')]
            selection=dict(run_fingerprint=digest(p),choices=choices,
                tuning_results_sha256={r['directory']:file_hash(Path(r['directory'])/'result.json') for r in sorted(tuning,key=lambda r:r['directory'])})
            path=directory/'outer-selection.json'
            if path.exists() and digest(read(path))!=digest(selection):raise ValueError('Changed selection on resume')
            write_json(path,selection)
            # Outer data are converted only after LR and epoch selection has been frozen.
            status('preparing_outer',current_outer_fold=fold)
            info=base.prepare_input(input_job(directory,p,'outer',0,p['outer_train_ids'],p['outer_test_ids']));inp=base.as_input(info)
            jobs=[fit_job(directory,p,'outer',c['variant'],s,0,c['lr'],inp,c) for c in choices for s in range(2)]
            outer=[]
            with SafePool(args.n_jobs) as pool:
                for r in pool.imap_unordered(base.fixed_fit,jobs):
                    outer.append(r);completed+=1;status('outer',current_outer_fold=fold,phase_completed=len(outer),phase_total=4)
                    print('OUTER',fold,r['variant'],r['repeat'],r['validation_mae'],flush=True)
            summarize_outer(directory,p,outer)
            timing=directory/'timing.json'
            if not timing.exists():write_json(timing,dict(wall_seconds=time.perf_counter()-foldstart,
                shared_input_task_seconds=sum(x['source']['seconds'] for x in inputs.values())+info['seconds']))
            complete_folds.append(fold);status('fold_complete',current_outer_fold=fold)
        status('complete')
    except Exception:status('failed',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
