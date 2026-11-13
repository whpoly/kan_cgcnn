"""Inner-only 60-fit KAN refinement and conditional frozen-config seed check."""
from __future__ import annotations

from run_modnet_official_matched import SafePool, digest, file_hash, set_seed, write_json
from mattervial_kan_refinement_models import VARIANTS, KAN_CANDIDATES, MLP_CANDIDATES, PAIRS, build_model
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'benchmarks/mattervial-quick-pilot-20261026'
LRS = (.001,.0005)
SEEDS = (20261027,20271027)
LABELS = dict(modnet='增强 MODNet',group_mlp='物理分组 MLP',group_kan='物理分组 KAN（无正则）',
    group_sparse_diagnostic='物理分组 KAN（原稀疏正则，诊断）',latent_mlp='隐藏表示残差 MLP',latent_kan='隐藏表示残差 KAN')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def now():
    return datetime.now(timezone.utc).isoformat()


def target_statistics(y_train):
    y = np.asarray(y_train,dtype='float64')
    if not np.isfinite(y).all():
        raise ValueError('Invalid training targets.')
    mean, scale = float(y.mean()),float(y.std())
    if scale <= 0:
        raise ValueError('Constant target.')
    return mean,scale


def prepare(output):
    source_protocol = read(SOURCE/'protocol.json')
    for name,sha in source_protocol['source_sha256'].items():
        if file_hash(ROOT/'scripts'/name) != sha:
            raise ValueError('Source protocol code changed: '+name)
    infos=[]
    for inner in range(5):
        path=SOURCE/'inputs'/f'inner-{inner}'/'mattervial'/'manifest.json'
        info=read(path)
        if info['run_fingerprint'] != digest(source_protocol):
            raise ValueError('Wrong source input provenance.')
        if len(info['selected_features']) != 256 or any('SISSO' in c.upper() for c in info['selected_features']):
            raise ValueError('Wrong feature bank.')
        if info['train_ids'] != info['feature_selection_fit_ids'] or info['train_ids'] != info['scaling_fit_ids']:
            raise ValueError('Feature preprocessing fit scope mismatch.')
        if set(info['train_ids']) & set(info['validation_ids']):
            raise ValueError('Overlapping partitions.')
        for name,sha in info['artifacts'].items():
            if file_hash(Path(info['directory'])/name) != sha:
                raise ValueError('Source input changed.')
        with np.load(Path(info['directory'])/'arrays.npz') as arrays:
            mean,scale=target_statistics(arrays['y_train'])
        infos.append(dict(source_manifest=str(path),source_manifest_sha256=file_hash(path),source=info,
            target_mean=mean,target_std=scale,target_transform_fit_ids=info['train_ids']))
    covered=[x for i in infos for x in i['source']['validation_ids']]
    if len(covered) != len(set(covered)) or len(covered) != 1012:
        raise ValueError('Wrong training-only universe.')
    for info in infos:
        if set(info['source']['train_ids']+info['source']['validation_ids']) != set(covered):
            raise ValueError('Different inner-fold training universe.')
    dependencies=('run_mattervial_kan_refinement.py','mattervial_kan_refinement_models.py',
        'run_modnet_official_matched.py','modnet_runtime.py','official_kan_models.py','physical_feature_modules.py')
    protocol=dict(kind='KAN optimization and small latent correction, exploratory inner-only',
        source_inputs=infos,source_protocol_sha256=file_hash(SOURCE/'protocol.json'),
        source_sha256={name:file_hash(ROOT/'scripts'/name) for name in dependencies},
        packages={name:importlib.metadata.version(name) for name in ('modnet','tensorflow','numpy','scikit-learn')},
        variants=VARIANTS,learning_rates=LRS,seeds=SEEDS,max_epochs=400,batch_size=32,
        target_scaling='mean/std from each INNER TRAIN only; predictions returned to physical units',
        early_stopping=dict(monitor='val_mae',patience=40,min_delta_physical=.05,restore='exact minimum inner val_mae weights'),
        optimizer='Adam, fixed starting LR, no scheduler',loss='normalized-target MAE plus declared diagnostic penalties',
        input_features='identical no-SISSO 256-feature matrices for every model, unchanged existing train-only preprocessing',
        source_models_or_outer_scores_loaded=False,outer_test_evaluated=False,
        trunk=[96,72,36,36],group_module='16 physical -> 4; 148 params; no penalties except sparse diagnostic',
        latent_module='36 -> learned Dense16 -> non-affine LayerNorm -> 0.5*tanh -> scalar KAN or MLP; zero-initialized scalar gate',
        latent_parameters=dict(projection=592,core=145,gate=1,total_correction=738),
        model_parameters=dict(modnet=35653,group_mlp=36185,group_kan=36185,group_sparse_diagnostic=36185,
                              latent_mlp=36391,latent_kan=36391),
        search_budget=dict(screening_fits=60,eligible_kan_candidates=4,eligible_mlp_candidates=4,
            fits_per_eligible_family=20,baseline_fits=10,sparse_diagnostic_fits=10,
            diagnostic_excluded_from_candidate_selection=True,per_fit_same_epoch_cap_and_stopping=True),
        selection='Each LR scored on all five inner folds; independently select KAN family, MLP family, and MODNet. Keep matched MLP too.',
        confirmation_gate=dict(min_relative_improvement=.01,min_improved_folds=4,
            comparisons=['best MLP family','matched-architecture MLP','enhanced MODNet'],
            all_required=True,action='freeze selected LR/architectures; evaluate second seed only, 15 or 20 added fits'),
        claims='No outer-test claim; old raw-target fixed-last-epoch scores are not a matched comparison to this new protocol',
        references=['https://github.com/KindXiaoming/pykan#Advice-on-hyperparameter-tuning'])
    fp=digest(protocol)
    path=output/'protocol.json'
    if path.exists() and digest(read(path)) != fp:
        raise ValueError('Protocol/source changed: use a new output directory.')
    write_json(path,protocol)
    return protocol,fp,infos


def train_job(job):
    directory=Path(job['directory'])
    directory.mkdir(parents=True,exist_ok=True)
    signature=digest(job)
    result_path=directory/'result.json'
    if result_path.exists():
        saved=read(result_path)
        if saved['signature'] != signature:
            raise ValueError('Changed fit identity.')
        for name,sha in saved['artifacts'].items():
            if file_hash(directory/name) != sha:
                raise ValueError('Changed saved fit artifact.')
        return saved
    try:
        inp=job['input']
        if file_hash(inp['source_manifest']) != inp['source_manifest_sha256']:
            raise ValueError('Changed source manifest.')
        info=inp['source']
        arrays_path=Path(info['directory'])/'arrays.npz'
        if file_hash(arrays_path) != info['artifacts']['arrays.npz']:
            raise ValueError('Changed feature arrays.')
        with np.load(arrays_path) as arrays:
            x,xv,y,yv=[arrays[k] for k in ('x_train','x_validation','y_train','y_validation')]
        mean,scale=target_statistics(y)
        if (mean,scale) != (inp['target_mean'],inp['target_std']):
            raise ValueError('Target-transform scope changed.')
        yn,yvn=((y-mean)/scale).astype('float32'),((yv-mean)/scale).astype('float32')
        started=time.perf_counter()
        model=build_model(job['variant'],info['selected_features'],job['seed'])
        if model.count_params() != job['expected_parameters']:
            raise ValueError('Model capacity changed.')
        set_seed(job['seed'])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=job['lr']),loss='mae',metrics=['mae'])
        class Checkpoint(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.value=float('inf');self.weights=None;self.epoch=None
            def on_epoch_end(self,epoch,logs=None):
                value=float(logs['val_mae'])
                if not np.isfinite(value):
                    raise ValueError('Nonfinite inner validation.')
                if value<self.value:
                    self.value=value;self.weights=self.model.get_weights();self.epoch=epoch+1
                if (epoch+1)%25==0:
                    write_json(directory/'status.json',dict(state='training',epoch=epoch+1,
                        best_epoch=self.epoch,best_validation_mae=self.value*scale,
                        seconds=time.perf_counter()-started,updated_at=now()))
        checkpoint=Checkpoint()
        history=model.fit(x,yn,validation_data=(xv,yvn),batch_size=32,epochs=job['max_epochs'],verbose=0,
            callbacks=[checkpoint,tf.keras.callbacks.EarlyStopping(monitor='val_mae',patience=40,
                min_delta=.05/scale,restore_best_weights=False)])
        model.set_weights(checkpoint.weights)
        pred=model.predict(xv,verbose=0).reshape(-1).astype('float64')*scale+mean
        train_pred=model.predict(x,verbose=0).reshape(-1).astype('float64')*scale+mean
        if not np.isfinite(pred).all() or not np.isfinite(train_pred).all():
            raise ValueError('Nonfinite predictions.')
        diagnostics={}
        names={l.name for l in model.layers}
        if 'latent_correction' in names:
            layer=model.get_layer('latent_correction')
            diagnostics['gate']=float(layer.gate.numpy())
            correction=tf.keras.Model(model.inputs,layer.output).predict(x,verbose=0).reshape(-1)*scale
            diagnostics['correction_train_std']=float(correction.std())
            diagnostics['correction_train_abs_mean']=float(np.mean(abs(correction)))
        if 'physical_module' in names:
            layer=model.get_layer('physical_module')
            values=tf.keras.Model(model.inputs,layer.output).predict(x,verbose=0)
            diagnostics['module_train_std']=values.std(axis=0).tolist()
            model(tf.convert_to_tensor(x[:32]),training=False)
            diagnostics['regularization_physical_units']=float(sum(model.losses))*scale
        np.savez_compressed(directory/'validation-predictions.npz',sample_ids=info['validation_ids'],target=yv,prediction=pred)
        with (directory/'model.pkl').open('wb') as stream:
            pickle.dump(dict(model_json=model.to_json(),weights=model.get_weights(),target_mean=mean,target_std=scale,
                input_manifest=inp['source_manifest']),stream,protocol=4)
        record=dict(signature=signature,run_fingerprint=job['run_fingerprint'],variant=job['variant'],lr=job['lr'],
            inner=job['inner'],repeat=job['repeat'],seed=job['seed'],directory=str(directory),
            input_digest=digest(inp),selected_features=info['selected_features'],train_ids=info['train_ids'],validation_ids=info['validation_ids'],
            target_transform_fit_ids=info['train_ids'],target_mean=mean,target_std=scale,parameters=model.count_params(),
            validation_mae=float(np.mean(abs(pred-yv))),train_mae=float(np.mean(abs(train_pred-y))),
            best_epoch=checkpoint.epoch,epochs=len(history.history['loss']),hit_epoch_cap=len(history.history['loss'])==job['max_epochs'],
            checkpoint_metric_mae=checkpoint.value*scale,history=history.history,diagnostics=diagnostics,
            seconds=time.perf_counter()-started,
            artifacts={name:file_hash(directory/name) for name in ('model.pkl','validation-predictions.npz')})
        if abs(record['checkpoint_metric_mae']-record['validation_mae'])>.003:
            raise ValueError('Restored checkpoint does not reproduce its validation score.')
        write_json(result_path,record)
        write_json(directory/'status.json',dict(state='complete',finished_at=now(),seconds=record['seconds']))
        del model,history,checkpoint
        tf.keras.backend.clear_session();gc.collect()
        return record
    except Exception:
        write_json(directory/'status.json',dict(state='failed',updated_at=now(),traceback=traceback.format_exc()))
        raise


def delta(candidate,reference):
    candidate,reference=np.asarray(candidate),np.asarray(reference)
    if candidate.shape!=(5,) or reference.shape!=(5,) or not np.isfinite(candidate+reference).all():
        raise ValueError('Five finite paired folds required.')
    diff=candidate-reference
    fraction=float((reference.mean()-candidate.mean())/reference.mean())
    wins=int((diff<0).sum())
    return dict(delta_by_fold=diff.tolist(),mean_delta=float(diff.mean()),relative_improvement=fraction,
                improved_folds=wins,passes=wins>=4 and fraction>=.01)


def choose(records):
    keys=[(r['variant'],r['lr'],r['inner']) for r in records]
    expected={(v,lr,i) for v in VARIANTS for lr in LRS for i in range(5)}
    if len(keys)!=60 or set(keys)!=expected or any(r['repeat']!=0 for r in records):
        raise ValueError('Complete first-seed factorial grid required.')
    options={}
    for variant in VARIANTS:
        items=[]
        for lr in LRS:
            rows=sorted([r for r in records if r['variant']==variant and r['lr']==lr],key=lambda r:r['inner'])
            scores=[r['validation_mae'] for r in rows]
            items.append(dict(variant=variant,lr=lr,mean_mae=float(np.mean(scores)),fold_mae=scores,
                members=[r['directory'] for r in rows]))
        options[variant]=dict(candidates=items,winner=min(items,key=lambda r:(r['mean_mae'],r['lr'])))
    kan=min([options[v]['winner'] for v in KAN_CANDIDATES],key=lambda r:(r['mean_mae'],r['variant']))
    mlp=min([options[v]['winner'] for v in MLP_CANDIDATES],key=lambda r:(r['mean_mae'],r['variant']))
    base=options['modnet']['winner'];matched=options[PAIRS[kan['variant']]]['winner']
    comparisons={name:delta(kan['fold_mae'],c['fold_mae']) for name,c in
        (('best_mlp',mlp),('matched_mlp',matched),('modnet',base))}
    confirm=all(c['passes'] for c in comparisons.values())
    configs={c['variant']:c for c in (base,kan,mlp,matched)}
    return dict(options=options,best_kan=kan,best_mlp=mlp,matched_mlp=matched,baseline=base,
        comparisons=comparisons,confirm=confirm,confirmation_configs=list(configs.values()),
        selection_scope='first seed, all five INNER validation folds only; diagnostic cannot enter KAN candidate selection')


def verify(records,protocol,infos):
    for record in records:
        inp=infos[record['inner']];info=inp['source']
        if record['input_digest']!=digest(inp) or record['parameters']!=protocol['model_parameters'][record['variant']]:
            raise ValueError('Unpaired inputs or model capacity.')
        for key in ('train_ids','validation_ids','selected_features'):
            if record[key]!=info[key]:
                raise ValueError('Wrong '+key)
        if record['train_ids']!=record['target_transform_fit_ids']:
            raise ValueError('Wrong target scaling scope.')
        if record['seed']!=protocol['seeds'][record['repeat']]+record['inner']:
            raise ValueError('Unpaired seeds.')
        if record['best_epoch']<1 or record['best_epoch']>record['epochs'] or record['epochs']>protocol['max_epochs']:
            raise ValueError('Invalid epoch checkpoint.')
        directory=Path(record['directory'])
        for name,sha in record['artifacts'].items():
            if file_hash(directory/name)!=sha:
                raise ValueError('Changed result artifact.')
        with np.load(directory/'validation-predictions.npz') as p:
            if p['sample_ids'].tolist()!=info['validation_ids']:
                raise ValueError('Wrong prediction IDs.')
            with np.load(Path(info['directory'])/'arrays.npz') as source:
                np.testing.assert_array_equal(p['target'],source['y_validation'])
            np.testing.assert_allclose(np.mean(abs(p['prediction']-p['target'])),record['validation_mae'],rtol=0,atol=1e-12)


def report(output,records,selection,protocol,infos,wall_seconds,confirmation=False):
    verify(records,protocol,infos)
    first=[r for r in records if r['repeat']==0]
    rows=[]
    for variant in VARIANTS:
        choice=selection['options'][variant]['winner']
        members=[r for r in first if r['variant']==variant and r['lr']==choice['lr']]
        rows.append(dict(**choice,parameters=members[0]['parameters'],
            train_mae_mean=float(np.mean([r['train_mae'] for r in members])),
            best_epochs=[r['best_epoch'] for r in sorted(members,key=lambda r:r['inner'])],
            fit_seconds=sum(r['seconds'] for r in first if r['variant']==variant),
            cap_hits=sum(r['hit_epoch_cap'] for r in first if r['variant']==variant)))
    confirmed=None
    if confirmation:
        scores={}
        for config in selection['confirmation_configs']:
            group=sorted([r for r in records if r['repeat']==1 and r['variant']==config['variant']],key=lambda r:r['inner'])
            if len(group)!=5 or any(r['lr']!=config['lr'] for r in group):
                raise ValueError('Incomplete frozen-config confirmation.')
            scores[config['variant']]=dict(lr=config['lr'],fold_mae=[r['validation_mae'] for r in group],
                mean_mae=float(np.mean([r['validation_mae'] for r in group])))
        kan=scores[selection['best_kan']['variant']]['fold_mae']
        comps={name:delta(kan,scores[selection[key]['variant']]['fold_mae']) for name,key in
            (('best_mlp','best_mlp'),('matched_mlp','matched_mlp'),('modnet','baseline'))}
        confirmed=dict(scores=scores,comparisons=comps,signal_replicated=all(c['passes'] for c in comps.values()),
            no_retuning_on_confirmation=True,scope='second seed on the SAME inner folds, not new held-out data')
    summary=dict(completed_fits=len(records),first_seed_fits=60,outer_test_evaluated=False,no_sisso=True,
        rows=rows,selection=selection,confirmation=confirmed,wall_seconds=wall_seconds,
        verification=dict(predictions_recomputed=True,paired_ids_features_targets_and_seeds=True,
            parameters_matched=True,artifact_hashes_verified=True,target_scaling_fit_on_inner_train=True),
        limitation='Exploratory: inner validation chooses epoch, LR, and architecture; no independent final benchmark')
    stem='confirmed' if confirmation else 'screening'
    write_json(output/(stem+'-summary.json'),summary)
    lines=['# MatterVial：KAN 改进试验', '',
        f"已完成 {len(records)} 次训练；墙钟时间 {wall_seconds/60:.2f} 分钟。仅使用原有第一外层折的 1012 个训练样本及五个内层划分。",
        '全部 SISSO 排除；所有模型使用相同的 256 个增强输入。每个内层训练集独立确定目标均值/标准差。',
        '统一最多 400 轮、batch 32、Adam、内层验证早停 patience 40，并恢复验证 MAE 最低的权重。所有 MAE 已还原为 cm^-1。',
        '本轮协议与旧快筛不同；只有本轮同步重跑的基线是公平对照。', '',
        '| 模型 | 内层选定学习率 | 五折验证 MAE | 五折训练 MAE | 参数量 | 两学习率累计训练分钟 | 达到轮数上限 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {LABELS[r['variant']]} | {r['lr']} | {r['mean_mae']:.6f} | {r['train_mae_mean']:.6f} | {r['parameters']} | {r['fit_seconds']/60:.2f} | {r['cap_hits']}/10 |")
    lines += ['', '| 内层折 | 增强 MODNet | 分组 MLP | 无正则分组 KAN | 稀疏 KAN 诊断 | 残差 MLP | 残差 KAN |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for i in range(5):
        lines.append(f'| {i+1} | '+' | '.join(f"{r['fold_mae'][i]:.6f}" for r in rows)+' |')
    lines += ['', '第一种子所有候选（不隐藏未选中的学习率）：', '',
        '| 模型 | 学习率 | 五折 MAE |', '| --- | ---: | ---: |']
    for variant in VARIANTS:
        for option in selection['options'][variant]['candidates']:
            lines.append(f"| {LABELS[variant]} | {option['lr']} | {option['mean_mae']:.6f} |")
    lines += ['', f"初筛最佳 KAN：{LABELS[selection['best_kan']['variant']]}；最佳 MLP：{LABELS[selection['best_mlp']['variant']]}。",
        '主比较的 KAN/MLP 各有两种结构、两个学习率，即各 20 次训练。原稀疏 KAN 的 10 次训练仅作消融，不参与候选选择。',
        f"是否达到追加种子条件：{selection['confirm']}。要求 KAN 相对最佳 MLP、对应等参数 MLP、增强 MODNet 均平均改善至少 1%，且各有至少 4/5 折改善。", '',
        '| 初筛比较 | MAE 差值（KAN−对照） | 改善百分比 | 改善折数 |', '| --- | ---: | ---: | ---: |']
    for name,c in selection['comparisons'].items():
        lines.append(f"| {name} | {c['mean_delta']:.6f} | {c['relative_improvement']*100:.3f}% | {c['improved_folds']}/5 |")
    if confirmed:
        lines += ['', '追加种子保持结构和学习率冻结；仍是同一批内层划分，不是独立测试集。', '',
                  '| 模型 | 固定学习率 | 第二种子五折 MAE |', '| --- | ---: | ---: |']
        for v,r in confirmed['scores'].items():
            lines.append(f"| {LABELS[v]} | {r['lr']} | {r['mean_mae']:.6f} |")
        lines += ['',f"第二种子是否再次达到全部改善条件：{confirmed['signal_replicated']}。"]
    lines += ['', '所有选择使用内层验证，结果有选择偏差；不能声称超过论文外层成绩。两个种子也不能代替独立外层测试。',
        '无正则分组 KAN 与原稀疏 KAN 的对照同时移除了稀疏项和平滑项，不能单独归因于其中某一项。',
        '参数量相等并不意味着表达能力相同。新增隐藏表示模块是联合训练的残差支路，不是对旧模型做后验外层校准。',
        '训练分钟为 worker 累计耗时；墙钟时间另列。所有失败/未改善结果保留，不以外层结果决定试验配置。', '',
        '设计参考：[pykan 作者调参建议](https://github.com/KindXiaoming/pykan#Advice-on-hyperparameter-tuning)：先从小模型、无正则设定检查精度，再考虑稀疏化；这并不保证在本任务有收益。']
    (output/(stem+'-report.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n-jobs',type=int,default=2)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    def status(state,**fields):
        write_json(output/'status.json',dict(state=state,pid=os.getpid(),updated_at=now(),**fields))
    try:
        status('preparing',completed=0,total=60)
        protocol,fp,infos=prepare(output)
        def jobs_for(configs,repeat):
            return [dict(directory=str(output/'fits'/variant/f'lr-{lr}-seed-{repeat}-inner-{inner}'),
                variant=variant,lr=lr,inner=inner,repeat=repeat,seed=SEEDS[repeat]+inner,input=infos[inner],
                max_epochs=protocol['max_epochs'],expected_parameters=protocol['model_parameters'][variant],
                run_fingerprint=fp) for inner in range(5) for variant,lr in configs]
        records=[]
        def execute(jobs,total):
            status('training',completed=len(records),total=total)
            with SafePool(args.n_jobs) as pool:
                for r in pool.imap_unordered(train_job,jobs):
                    records.append(r)
                    status('training',completed=len(records),total=total,last_variant=r['variant'],last_inner=r['inner'],
                        last_mae=r['validation_mae'],last_epochs=r['epochs'],last_best_epoch=r['best_epoch'])
                    print('FIT',len(records),'/',total,r['variant'],'lr',r['lr'],'inner',r['inner'],'mae',r['validation_mae'],
                          'epoch',r['best_epoch'],'/',r['epochs'],'seconds',r['seconds'],flush=True)
        execute(jobs_for([(v,lr) for lr in LRS for v in VARIANTS],0),60)
        selection=choose(records)
        write_json(output/'frozen-stage1-selection.json',dict(run_fingerprint=fp,decided_at=now(),**selection))
        report(output,records,selection,protocol,infos,time.perf_counter()-started)
        if selection['confirm']:
            configs=[(c['variant'],c['lr']) for c in selection['confirmation_configs']]
            execute(jobs_for(configs,1),60+5*len(configs))
            report(output,records,selection,protocol,infos,time.perf_counter()-started,confirmation=True)
        status('complete',completed=len(records),total=len(records),confirmation_run=selection['confirm'],
            outer_test_evaluated=False,wall_seconds=time.perf_counter()-started)
        print('COMPLETE',output,flush=True)
    except Exception:
        status('failed',traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
