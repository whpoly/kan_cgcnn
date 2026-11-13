"""Paired inner-only ablations, followed by fixed-config native baseline replay.

No hyperparameter or checkpoint is selected from outer test results. The replay
is new training of historically inner-selected configs, NOT a fresh GA search.
"""
from run_modnet_official_matched import (SafePool,write_json,digest,file_hash,set_seed,
    evaluate_job,predict_members,np,pd,tf,MODData,MODNetModel)
from official_kan_models import neurons_from_genes
from modnet.hyper_opt.fit_genetic import Individual
from pathlib import Path
from datetime import datetime,timezone
import argparse
import contextlib
import gc
import importlib.metadata
import json
import os
import pickle
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'benchmarks/official-aligned-20261021'
NEW=ROOT/'benchmarks/fourway-phonons-20261029'
MODES=('native','target_only','checkpoint_only','target_checkpoint',
       'new_split_legacy','new_features_only','new_features_training')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def legacy_id(x):return 'id'+str(int(x.rsplit('-',1)[-1])-1)


def freeze(out,smoke):
    if importlib.metadata.version('modnet')!='0.1.12':raise ValueError('Use original modnet-v012-matbench environment')
    sources=[Path(__file__),ROOT/'scripts/run_modnet_official_matched.py',ROOT/'scripts/official_kan_models.py',ROOT/'scripts/modnet_runtime.py']
    oldp=read(OLD/'matched-protocol.json')
    # Native model/training implementation must match the actual old run.
    for path,value in oldp['source_sha256'].items():
        if file_hash(path)!=value:raise ValueError('Historical reproduction source changed: '+path)
    files={str(s):file_hash(s) for s in sources}
    replay_sources={}
    for fold in range(5):
        directory=OLD/'matbench_phonons'/f'fold-{fold}'
        for name in ('official-train-moddata.pkl.gz','frozen-ensemble.json','outer-split.json','baseline.json'):
            path=directory/name;replay_sources[str(path)]=file_hash(path)
        for m in read(directory/'frozen-ensemble.json')['members']:
            path=Path(m['directory'])/'result.json';replay_sources[str(path)]=file_hash(path)
    inputs={}
    for i in range(5):
        path=NEW/'inputs/mm/fold-0'/f'inner-{i}'/'manifest.json'
        inputs[str(path)]=file_hash(path)
    p=dict(kind='Native MODNet reliability replay and inner-only protocol-change diagnostics',smoke=smoke,
        source_sha256=files,historical_source_sha256=oldp['source_sha256'],replay_sources=replay_sources,
        diagnostic_input_manifests=inputs,diagnostic_outer_fold=0,diagnostic_modes=MODES,
        diagnostic_configs='The FIRST candidate in the frozen original fold0 ensemble (selected by original inner validation), all five original/new inner splits. No outer-driven choice.',
        diagnostics_fits=7 if smoke else 35,replay_fits=0 if smoke else 250,
        replay='Original50 selected members per outer fold; original feature ordering, per-member genes/splits/seeds and native fit; all weights newly initialized, no old weights loaded.',
        verification='Native diagnostic/replay validation scores must match saved original within0.01cm^-1. Replay fivefold mean must match original34.013377 within0.05cm^-1 and published34.275106 within2%.',
        feature_diagnostic='New inner IDs: legacy outer-training ranking vs frozen new inner-training RFE/RR ranking at the SAME feature count and architecture.',
        paired_training='Raw/native train-loss stopping vs target-standardization-only (train-loss min_delta kept at0.001 physical units) vs validation-best-checkpoint-only vs both. Same features, rows, seed, optimizer, LR and batch.',
        boundaries=['No KAN/MatterVial training is launched by this script.',
            'Ablations inspect inner validation only; validation-best scores have selection optimism and are not an independent accuracy claim.',
            'Original feature ranking included inner validation rows. Replay is paper-protocol reproduction, not strict nested validation.',
            'First-fold one-config diagnosis is conditional evidence, not a quantified explanation of the full fivefold ensemble difference.',
            'The final outer replay is a reproduction audit of frozen old choices, not a new independent test. No new choices are made from its scores.'])
    path=out/'protocol.json'
    if path.exists() and digest(read(path))!=digest(p):raise ValueError('Changed diagnostic protocol')
    write_json(path,p)
    for source,value in files.items():
        dest=out/'source-snapshot'/Path(source).name;dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copyfile(source,dest)
        if file_hash(dest)!=value:raise ValueError('Changed source snapshot')
    return p


def make_job(out,p,mode,fold,member_index):
    source=read(OLD/'matbench_phonons'/f'fold-{fold}'/'frozen-ensemble.json')['members'][member_index]
    result_path=Path(source['directory'])/'result.json'
    return dict(directory=str(out/('replay' if mode=='replay' else 'diagnostics')/f'fold-{fold}'/mode/f'member-{member_index:02d}'),
        mode=mode,fold=fold,member_index=member_index,source_result=str(result_path),source_sha256=file_hash(result_path),
        run_fingerprint=digest(p),smoke=p['smoke'],protocol=str(out/'protocol.json'))


CHECKED=set()
def job_inputs(job):
    if job['run_fingerprint'] not in CHECKED:
        p=read(job['protocol'])
        if digest(p)!=job['run_fingerprint']:raise ValueError('Changed protocol')
        for path,value in {**p['source_sha256'],**p['historical_source_sha256']}.items():
            if file_hash(path)!=value:raise ValueError('Changed executing source')
        CHECKED.add(job['run_fingerprint'])
    if file_hash(job['source_result'])!=job['source_sha256']:raise ValueError('Changed historical config')
    r=read(job['source_result']);fold=job['fold'];mode=job['mode']
    cache=OLD/'matbench_phonons'/f'fold-{fold}'/'official-train-moddata.pkl.gz'
    p=read(job['protocol'])
    if file_hash(cache)!=p['replay_sources'][str(cache)]:raise ValueError('Changed original feature cache')
    data=MODData.load(cache)
    trids,vaids=r['train_ids'],r['validation_ids'];features=r['features']
    if mode.startswith('new_'):
        path=NEW/'inputs/mm/fold-0'/f"inner-{job['member_index']}"/'manifest.json'
        if file_hash(path)!=p['diagnostic_input_manifests'][str(path)]:raise ValueError('Changed diagnostic feature input')
        inp=read(path);trids=list(map(legacy_id,inp['train_ids']));vaids=list(map(legacy_id,inp['validation_ids']))
        if mode!='new_split_legacy':features=inp['selected_features']
    if set(trids)&set(vaids) or set(trids+vaids)!=set(data.df_targets.index):raise ValueError('Wrong nested training universe')
    lookup={sid:i for i,sid in enumerate(data.df_targets.index)}
    tr,va=data.split(([lookup[x] for x in trids],[lookup[x] for x in vaids]))
    tr.optimal_features=list(features);va.optimal_features=list(features)
    if any('SISSO' in x.upper() for x in features):raise ValueError('SISSO')
    return r,tr,va


def run_job(job):
    out=Path(job['directory']);out.mkdir(parents=True,exist_ok=True)
    signature=digest(job);done=out/'diagnostic.json'
    if done.exists():
        r=read(done)
        if r['signature']!=signature:raise ValueError('Changed diagnostic job')
        for name,value in r['artifacts'].items():
            if file_hash(out/name)!=value:raise ValueError('Changed diagnostic output')
        return r
    started=time.perf_counter();source,train,val=job_inputs(job)
    g=source['genes'];seed=source['seed'];mode=job['mode']
    normalized=mode in ('target_only','target_checkpoint','new_features_training')
    validation_stop=mode in ('checkpoint_only','target_checkpoint','new_features_training')
    y=train.df_targets.values[:,0].copy();yv=val.df_targets.values[:,0].copy()
    mean,std=(float(y.mean()),float(y.std())) if normalized else (0.,1.)
    try:
        if mode in ('native','replay'):
            individual=Individual(max_feat=len(train.optimal_features),num_classes=train.num_classes,multi_label=False,
                targets=[[list(train.target_names)]],weights={n:1 for n in train.target_names})
            individual.genes=dict(g)
            fitjob=dict(directory=str(out),variant='modnet',seed=seed,individual=individual,train_data=train,val_data=val,
                fast=job['smoke'],run_fingerprint=job['run_fingerprint'])
            result=evaluate_job(fitjob)
            if not job['smoke'] and abs(result['validation_mae']-source['validation_mae'])>.01:
                raise ValueError('Native replay failed to reproduce original inner score: '+str((result['validation_mae'],source['validation_mae'])))
            with (out/'model.pkl').open('rb') as f:model=pickle.load(f)
            model._restore_model()
            history=result['history'];epochs=result['epochs'];best_epoch=epochs
        else:
            tf.keras.backend.clear_session();set_seed(seed)
            model=MODNetModel(targets=[[list(train.target_names)]],weights={n:1 for n in train.target_names},
                n_feat=g['n_feat'],num_neurons=neurons_from_genes(g),act=g['act'],num_classes=train.num_classes,multi_label=False)
            if normalized:
                train.df_targets=train.df_targets.copy();val.df_targets=val.df_targets.copy()
                train.df_targets.iloc[:,0]=(y-mean)/std;val.df_targets.iloc[:,0]=(yv-mean)/std
            class Best(tf.keras.callbacks.Callback):
                def __init__(self):super().__init__();self.value=float('inf');self.weights=None;self.epoch=0
                def on_epoch_end(self,epoch,logs=None):
                    value=float(logs['val_mae'])
                    if not np.isfinite(value):raise ValueError('Nonfinite validation')
                    if value<self.value:self.value=value;self.weights=self.model.get_weights();self.epoch=epoch+1
                    if (epoch+1)%50==0:write_json(out/'status.json',dict(state='training',epoch=epoch+1,mode=mode))
            best=Best()
            callbacks=[best,tf.keras.callbacks.EarlyStopping(monitor='val_mae' if validation_stop else 'loss',
                patience=40 if validation_stop else 30,min_delta=(.05 if validation_stop else .001)/std,
                restore_best_weights=False)]
            set_seed(seed)
            with (out/'training.log').open('w',encoding='utf-8') as f:
                with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
                    model.fit(train,val_data=val,loss=g['loss'],lr=g['lr'],epochs=1 if job['smoke'] else 800,
                        batch_size=g['batch_size'],xscale=g['xscale'],callbacks=callbacks,verbose=0)
            history=model.history;epochs=len(history['loss']);best_epoch=best.epoch if validation_stop else epochs
            if validation_stop:model.model.set_weights(best.weights)
        features=list(model.optimal_descriptors[:model.n_feat])
        xv=val.df_featurized[features].replace([np.inf,-np.inf,np.nan],0).values
        xv=np.nan_to_num(model._scaler.transform(np.nan_to_num(xv)),nan=-1)
        prediction=model.model.predict(xv,verbose=0).reshape(-1).astype(float)*std+mean
        mae=float(np.mean(abs(prediction-yv)))
        if not np.isfinite(prediction).all():raise ValueError('Nonfinite predictions')
        np.savez_compressed(out/'validation.npz',sample_ids=list(val.df_targets.index),prediction=prediction,target=yv)
        if mode not in ('native','replay'):
            model._make_picklable()
            with (out/'model.pkl').open('wb') as f:pickle.dump(model,f,protocol=4)
        r=dict(signature=signature,mode=mode,fold=job['fold'],member_index=job['member_index'],seed=seed,genes=g,
            train_ids=list(train.df_targets.index),validation_ids=list(val.df_targets.index),features=features,
            target_mean=mean,target_std=std,target_fit_ids=list(train.df_targets.index),validation_mae=mae,
            original_validation_mae=source['validation_mae'],epochs=epochs,best_epoch=best_epoch,history=history,
            seconds=time.perf_counter()-started,directory=str(out),run_fingerprint=job['run_fingerprint'],
            artifacts={n:file_hash(out/n) for n in ('model.pkl','validation.npz')})
        write_json(done,r);write_json(out/'status.json',dict(state='complete',mode=mode))
        del model;tf.keras.backend.clear_session();gc.collect()
        return r
    except Exception:
        write_json(out/'status.json',dict(state='failed',traceback=traceback.format_exc()))
        raise


def diagnostic_report(out,p):
    rows=[read(f) for f in (out/'diagnostics').glob('fold-*/*/*/diagnostic.json')]
    groups=[]
    for mode in MODES:
        rs=sorted([r for r in rows if r['mode']==mode],key=lambda r:r['member_index'])
        if len(rs)!=(1 if p['smoke'] else 5):continue
        groups.append(dict(mode=mode,mean_validation_mae=float(np.mean([r['validation_mae'] for r in rs])),
            mean_epochs=float(np.mean([r['epochs'] for r in rs])),mean_best_epoch=float(np.mean([r['best_epoch'] for r in rs])),
            per_fold_mae=[r['validation_mae'] for r in rs]))
    write_json(out/'diagnostic-summary.json',dict(complete=len(groups)==7,smoke=p['smoke'],groups=groups,
        scope='One historically inner-selected configuration, fold0 training universe only; no outer predictions/labels read for diagnostics'))
    lines=['# MODNet配对消融：仅内层诊断','',
        '固定原外层第1折内层选出的首个配置、配对种子；只在1012个外层训练样本内训练和计分。没有读取外层测试标签。',
        '验证集早停组使用验证集选checkpoint，其较低验证误差存在选择偏差，不能据此声称泛化更好。','',
        '| 变体 | 内层MAE均值 | 平均训练轮数 | 平均使用轮数 |','| --- | ---: | ---: | ---: |']
    for r in groups:lines.append(f"| {r['mode']} | {r['mean_validation_mae']:.6f} | {r['mean_epochs']:.1f} | {r['mean_best_epoch']:.1f} |")
    lines+=['','native=原协议；target_only=只标准化目标；checkpoint_only=只改验证早停并恢复最低点；target_checkpoint=同时改变两项。',
        'new_split_legacy=只换为新内层划分，保留旧排名和训练规则；new_features_only=再换为新内层特征排名；new_features_training=再用新目标缩放与早停规则。',
        '前三种new_变体之间严格配对样本、参数、种子及特征数。它们与原native组划分不同，不能直接归因为特征或训练改动。',
        '单配置诊断不等同于全部GA/网格集成的因果分解；不从这里选择最终四组配置。']
    (out/'diagnostic-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return groups


def replay_outer(out,p,fold,records):
    directory=out/'replay'/f'fold-{fold}';old=OLD/'matbench_phonons'/f'fold-{fold}'
    split=read(old/'outer-split.json')
    original=MODData.load(OLD/'precomputed/matbench_phonons_moddata.pkl.gz')
    test=original.from_indices(split['test_positions'])
    jobs=[dict(directory=r['directory'],seed=r['seed']) for r in sorted(records,key=lambda r:r['member_index'])]
    pred,_,replaced=predict_members(jobs,test,directory,'modnet')
    mae=float(np.mean(abs(pred-test.df_targets.values[:,0])))
    expected=read(old/'baseline.json')['results']['modnet']['mae']
    with np.load(old/'modnet-member-predictions.npz') as z:
        with np.load(directory/'modnet-member-predictions.npz') as fresh:
            difference=float(np.max(abs(z['predictions']-fresh['predictions'])))
    score=dict(fold=fold,mae=mae,old_mae=expected,absolute_difference=mae-expected,
        maximum_member_prediction_difference=difference,ensemble_size=len(records),replaced=replaced,
        new_training_fits=len(records),selection='Replayed frozen old inner-selected ensemble, no new search',run_fingerprint=digest(p))
    pd.DataFrame(dict(sample_id=split['official_test_ids'],target=test.df_targets.values[:,0],prediction=pred)).to_csv(directory/'predictions.csv',index=False,float_format='%.17g')
    write_json(directory/'complete.json',score)
    return score


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n-jobs',type=int,default=4);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--phase',choices=('all','diagnostics','replay'),default='all')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    def status(state,**kw):write_json(out/'status.json',dict(state=state,pid=os.getpid(),updated_at=now(),wall_seconds=time.perf_counter()-start,**kw))
    try:
        status('freezing');p=freeze(out,args.smoke)
        if args.phase in ('all','diagnostics'):
            count=1 if args.smoke else 5
            jobs=[make_job(out,p,mode,0,i) for mode in MODES for i in range(count)]
            records=[];status('diagnostics',completed=0,total=len(jobs))
            with SafePool(args.n_jobs) as pool:
                for r in pool.imap_unordered(run_job,jobs):
                    records.append(r);status('diagnostics',completed=len(records),total=len(jobs),last_mode=r['mode'])
                    diagnostic_report(out,p)
                    print('DIAGNOSTIC',len(records),'/',len(jobs),r['mode'],r['member_index'],r['validation_mae'],flush=True)
        if args.phase in ('all','replay') and not args.smoke:
            scores=[]
            for fold in range(5):
                records=[];jobs=[make_job(out,p,'replay',fold,i) for i in range(50)]
                status('native_replay_training',fold=fold,completed=0,total=50)
                with SafePool(args.n_jobs) as pool:
                    for r in pool.imap_unordered(run_job,jobs):
                        records.append(r);status('native_replay_training',fold=fold,completed=len(records),total=50)
                        print('REPLAY',fold,len(records),'/50',r['validation_mae'],flush=True)
                score=replay_outer(out,p,fold,records);scores.append(score)
                write_json(out/'replay-summary.json',dict(complete=False,rows=scores))
                print('REPLAY OUTER',fold,score['mae'],flush=True)
            mean=float(np.mean([r['mae'] for r in scores]));oldmean=float(np.mean([r['old_mae'] for r in scores]))
            verified=abs(mean-oldmean)<=.05 and abs(mean/34.27510615406341-1)<=.02
            summary=dict(complete=True,verified=verified,rows=scores,mae=mean,old_mae=oldmean,published_mae=34.27510615406341,
                new_training_fits=250,scope='Fresh weights with historically frozen inner-selected configurations; not a new GA search')
            write_json(out/'replay-summary.json',summary)
            if not verified:raise ValueError('Fresh native baseline replay failed numerical consistency check')
        status('complete',smoke=args.smoke,phase=args.phase,fourway_comparison_resumed=False)
    except Exception:
        status('failed',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
