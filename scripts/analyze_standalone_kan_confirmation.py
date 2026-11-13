"""Independent NumPy audit of nested splits, fixed fits, decisions and scores."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import pickle
import pandas as pd
from analyze_standalone_kan_screen import np,plt,read,sha,digest,check,forward,ScalerStateReader
from standalone_kan_confirmation_policy import choose,evaluate_gate,BANKS,VARIANTS

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'benchmarks/standalone-kan-confirmation-20261028'


def audit_input(info,p,cache):
    phase,i=info['phase'],info['inner']
    if phase=='outer':
        tr,va=p['outer_train_ids'],p['outer_test_ids']
        check(read(RUN/'gate.json')['passed'],'Outer data before gate')
    else:
        split=p['splits'][i]
        tr=split['tuning_train_ids'] if phase=='tuning' else split['train_ids']
        va=split['tuning_validation_ids'] if phase=='tuning' else split['validation_ids']
    check(info['train_ids']==tr==info['feature_selection_fit_ids']==info['scaling_fit_ids']==info['target_transform_fit_ids'],'Preprocessing fit scope')
    check(info['validation_ids']==va and not set(tr)&set(va),'Validation scope')
    check(not any('SISSO' in f.upper() for f in info['selected_features']),'SISSO')
    check(len(info['selected_features'])==len(set(info['selected_features']))==256,'Feature count')
    directory=Path(info['directory'])
    for name,value in info['artifacts'].items():check(sha(directory/name)==value,'Changed input artifact')
    with (directory/'scaler.pkl').open('rb') as f:scaler=ScalerStateReader(f).load()
    raw=cache.loc[tr,info['selected_features']].values
    np.testing.assert_allclose(scaler.data_min_,np.min(raw,axis=0),atol=0,rtol=0)
    np.testing.assert_allclose(scaler.data_max_,np.max(raw,axis=0),atol=0,rtol=0)
    with np.load(directory/'arrays.npz') as a:
        expected=(raw*scaler.scale_+scaler.min_).astype('float32')
        np.testing.assert_array_equal(a['x_train'],expected)
        np.testing.assert_array_equal(a['y_train'],cache.loc[tr,'target'].values)
        if phase!='outer':
            expected=(cache.loc[va,info['selected_features']].values*scaler.scale_+scaler.min_).astype('float32')
            np.testing.assert_array_equal(a['x_validation'],expected)
            np.testing.assert_array_equal(a['y_validation'],cache.loc[va,'target'].values)
        np.testing.assert_allclose([info['target_mean'],info['target_std']],
            [a['y_train'].astype(float).mean(),a['y_train'].astype(float).std()],atol=1e-10,rtol=0)
    manifest=directory/'manifest.json'
    return dict(source_manifest=str(manifest),source_manifest_sha256=sha(manifest),source=info,
                target_mean=info['target_mean'],target_std=info['target_std'],target_transform_fit_ids=tr)


def audit_fit(path,p,inputs):
    r=read(path);phase=path.relative_to(RUN/'fits').parts[0]
    info=inputs[(phase,r['inner'])];source=info['source'];fp=digest(p)
    check(Path(r['directory']).resolve()==path.parent.resolve(),'Fit directory')
    job=dict(directory=r['directory'],bank=r['bank'],variant=r['variant'],repeat=r['repeat'],inner=r['inner'],
        seed=r['seed'],lr=r['lr'],max_epochs=p['max_epochs'],expected_parameters=p['expected_parameters'][r['bank']][r['variant']],
        input=info,run_fingerprint=fp)
    if phase!='tuning':
        selection=RUN/('selection.json' if phase=='confirmation' else 'outer-selection.json')
        choices=read(selection)['choices']
        c=next(c for c in choices if c['bank']==r['bank'] and c['variant']==r['variant'] and (phase=='outer' or c['inner']==r['inner']))
        check(r['lr']==c['lr'] and r['fixed_epochs']==c['epochs']==r['epochs'],'Fixed configuration mismatch')
        check(r['selection_sha256']==sha(selection),'Selection changed')
        job.update(phase=phase,fixed_epochs=c['epochs'],selection_sha256=sha(selection))
        check(not r['evaluation_labels_used_during_training'] and not r['validation_data_passed_to_fit'],'Evaluation used in fit')
        check(not any('val_' in k for k in r['history']),'Validation history in fixed fit')
    else:
        check(r['lr'] in p['learning_rates'],'Unknown learning rate')
        check(r['best_epoch']==int(np.argmin(r['history']['val_mae']))+1,'Wrong checkpoint')
        check(1<=r['best_epoch']<=r['epochs']<=400,'Epoch bounds')
    seeds=p[{'tuning':'tuning_seeds','confirmation':'confirmation_seeds','outer':'outer_seeds'}[phase]]
    check(r['seed']==seeds[r['repeat']]+(0 if phase=='outer' else r['inner']),'Wrong seed')
    check(r['signature']==digest(job) and r['run_fingerprint']==fp,'Job signature')
    check(r['input_digest']==digest(info),'Input digest')
    check(r['train_ids']==source['train_ids']==r['target_transform_fit_ids'],'Training IDs')
    check(r['validation_ids']==source['validation_ids'],'Validation IDs')
    features=source['selected_features'] if r['bank']=='enhanced' else source['selected_features'][:16]
    check(features==r['features'],'Features mismatch')
    for name,value in r['artifacts'].items():check(sha(path.parent/name)==value,'Changed model artifact')
    with np.load(path.parent/'predictions.npz') as a:
        check(a['sample_ids'].tolist()==source['validation_ids'],'Prediction IDs')
        check(a['train_ids'].tolist()==source['train_ids'],'Train prediction IDs')
        mae=float(np.mean(abs(a['prediction']-a['target'])))
        train_mae=float(np.mean(abs(a['train_prediction']-a['train_target'])))
        with np.load(Path(source['directory'])/'arrays.npz') as original:
            np.testing.assert_array_equal(a['target'],original['y_validation'])
            np.testing.assert_array_equal(a['train_target'],original['y_train'])
    check(abs(mae-r['validation_mae'])<1e-10 and abs(train_mae-r['train_mae'])<1e-10,'Saved score mismatch')
    with (path.parent/'model.pkl').open('rb') as f:model=pickle.load(f)
    check(sum(w.size for w in model['weights'])==r['parameters']==job['expected_parameters'],'Capacity')
    np.testing.assert_allclose([r['target_mean'],r['target_std']],[source['target_mean'],source['target_std']],atol=0,rtol=0)
    error=None
    if r['variant']=='kan':
        classes=[l['class_name'] for l in json.loads(model['model_json'])['config']['layers']]
        check(classes==['InputLayer']+['StandaloneKAN>SplineEdgeLayer']*3,'Not frozen KAN graph')
        with np.load(Path(source['directory'])/'arrays.npz') as a:
            indices=[source['selected_features'].index(f) for f in features]
            reconstructed=forward(read(path.parent/'network.json'),a['x_validation'][:,indices])
        with np.load(path.parent/'predictions.npz') as a:error=float(np.max(abs(reconstructed-a['prediction'])))
        check(error<.01,'Independent KAN reconstruction')
    return dict(r,phase=phase,numpy_max_prediction_difference=error)


def plot(rows,gate):
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for bank,ax in zip(BANKS,axes):
        row=gate['rows'][bank]
        for s in range(2):
            xs=np.arange(1,6)+(s-.5)*.12
            ax.plot(xs,row['delta'][s],marker='o',label=f'Seed {s+1}')
        ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(1,6));ax.set_xlabel('New inner confirmation fold')
        ax.set_ylabel('KAN MAE - matched MLP MAE (cm$^{-1}$)')
        ax.set_title('256 enhanced features' if bank=='enhanced' else '16 physical descriptors');ax.legend()
    fig.suptitle('Nested confirmation | negative favors KAN\nLearning rate and epoch count selected without each confirmation holdout')
    fig.tight_layout(rect=(0,0,1,.89));(RUN/'figures').mkdir(exist_ok=True)
    fig.savefig(RUN/'figures/paired-confirmation.png',dpi=180);fig.savefig(RUN/'figures/paired-confirmation.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    p=read(RUN/'protocol.json');status=read(RUN/'status.json')
    if not args.partial:check(status['state']=='complete','Study not complete')
    for name,value in p['source_sha256'].items():check(sha(ROOT/'scripts'/name)==value,'Frozen source changed: '+name)
    check(sha(p['source_cache'])==p['source_cache_sha256'],'Changed training-only cache')
    cache=pd.read_pickle(p['source_cache'])
    check(sorted(cache.index.tolist())==sorted(p['outer_train_ids']),'Cache universe')
    covered=[]
    for split in p['splits']:
        tr,va,a,b=[set(split[k]) for k in ('train_ids','validation_ids','tuning_train_ids','tuning_validation_ids')]
        check(not(tr&va or a&b or a&va or b&va),'Nested partition overlap')
        check(a|b==tr and tr|va==set(p['outer_train_ids']),'Nested partition coverage')
        check(not (tr|va)&set(p['outer_test_ids']),'Outer test inside inner experiment')
        covered+=list(va)
    check(len(covered)==len(set(covered))==1012,'Confirmation OOF coverage')
    inputs={}
    for path in sorted((RUN/'inputs').glob('*/*/manifest.json')):
        info=read(path);inputs[(info['phase'],info['inner'])]=audit_input(info,p,cache)
    records=[audit_fit(path,p,inputs) for path in sorted((RUN/'fits').glob('*/*/*/*/*/result.json'))]
    stages={stage:[r for r in records if r['phase']==stage] for stage in ('tuning','confirmation','outer')}
    keys=[(r['phase'],r['bank'],r['variant'],r['repeat'],r['inner'],r['lr']) for r in records]
    check(len(keys)==len(set(keys)),'Duplicate results')
    if not args.partial:check(len(stages['tuning'])==120 and len(stages['confirmation'])==40,'Missing completed fits')
    if (RUN/'selection.json').exists():
        selected=read(RUN/'selection.json');check(selected['run_fingerprint']==digest(p),'Selection provenance')
        for directory,value in selected['tuning_results_sha256'].items():check(sha(Path(directory)/'result.json')==value,'Changed selection input')
        for c in selected['choices']:
            group=[r for r in stages['tuning'] if r['inner']==c['inner'] and r['bank']==c['bank'] and r['variant']==c['variant']]
            check(c==dict(inner=c['inner'],bank=c['bank'],variant=c['variant'],**choose(group,2)),'Selection not reproducible')
    gate=None
    if len(stages['confirmation'])==40:
        gate=evaluate_gate(stages['confirmation']);saved=read(RUN/'gate.json')
        for key,value in gate.items():check(saved[key]==value,'Gate differs: '+key)
        for directory,value in saved['confirmation_results_sha256'].items():check(sha(Path(directory)/'result.json')==value,'Changed gate input')
        if not gate['passed']:check(not stages['outer'] and ('outer',0) not in inputs,'Outer evaluation despite failed gate')
    rows=[]
    for stage,rs in stages.items():
        for bank in BANKS:
            for variant in VARIANTS:
                group=[r for r in rs if r['bank']==bank and r['variant']==variant]
                if not group:continue
                rows.append(dict(phase=stage,bank=bank,variant=variant,completed=len(group),parameters=group[0]['parameters'],
                    mean_mae=float(np.mean([r['validation_mae'] for r in group])),
                    seed_mae=[float(np.mean([r['validation_mae'] for r in group if r['repeat']==s])) if any(r['repeat']==s for r in group) else None for s in range(2)],
                    mean_seconds=float(np.mean([r['seconds'] for r in group])),mean_epochs=float(np.mean([r['epochs'] for r in group])),
                    max_epoch_hits=sum(r['epochs']==400 for r in group)))
    outer=None
    if (RUN/'outer-summary.json').exists():
        check(gate and gate['passed'],'Outer without gate')
        outer=read(RUN/'outer-summary.json')
        op=read(RUN/'outer-selection.json')
        for c in op['choices']:
            group=[r for r in stages['tuning'] if r['bank']==c['bank'] and r['variant']==c['variant']]
            check(c==dict(bank=c['bank'],variant=c['variant'],**choose(group,10)),'Outer selection mismatch')
        for c in outer['rows']:
            with np.load(RUN/f"outer-{c['bank']}-{c['variant']}-ensemble.npz") as a:
                group=sorted([r for r in stages['outer'] if r['bank']==c['bank'] and r['variant']==c['variant']],key=lambda r:r['repeat'])
                ps=[]
                for r in group:
                    with np.load(Path(r['directory'])/'predictions.npz') as member:ps.append(member['prediction'])
                np.testing.assert_array_equal(a['prediction'],np.mean(ps,axis=0))
                check(abs(float(np.mean(abs(a['prediction']-a['target'])))-c['ensemble_mae'])<1e-10,'Outer ensemble MAE')
                check(a['sample_ids'].tolist()==p['outer_test_ids'],'Outer prediction IDs')
    result=dict(analyzed_at=datetime.now(timezone.utc).isoformat(),analyzer_sha256=sha(__file__),run_fingerprint=digest(p),
        complete=status['state']=='complete',verified_fits=len(records),verified_inputs=len(inputs),status=status,rows=rows,gate=gate,outer=outer,
        max_numpy_prediction_difference=max([r['numpy_max_prediction_difference'] for r in records if r['numpy_max_prediction_difference'] is not None]+[0]),
        outer_test_evaluated=bool(stages['outer']),limitations=p['limitations'])
    filename='partial-analysis.json' if args.partial else 'analysis.json'
    (RUN/filename).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if gate:plot(rows,gate)
    if not args.partial:report(result,p)
    print(json.dumps(dict(verified_fits=len(records),verified_inputs=len(inputs),state=status['state'],gate=gate,outer=outer),ensure_ascii=False,indent=2))


def report(result,p):
    gate=result['gate'];rows=result['rows']
    lines=['# 冻结完整 KAN：新划分、新种子的嵌套确认','',
        f"共独立核验 {result['verified_fits']} 次训练、{result['verified_inputs']} 份输入。主要稳定性门槛：{'通过' if gate['passed'] else '未通过'}。",'',
        '三学习率 × 两调参种子，每个确认训练部分内部单独选择学习率与轮数；再用两个不同种子固定轮数重训。确认集不参与早停或 checkpoint 选择。此协议比上一轮直接以内层确认 MAE 早停更严格，不能将新旧误差变化全部归因于架构。','',
        '| 输入 | 模型 | 确认种子1 MAE | 确认种子2 MAE | 参数量 | 平均每次秒数 | 平均轮数 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        if r['phase']=='confirmation':lines.append(f"| {r['bank']} | {r['variant']} | {r['seed_mae'][0]:.3f} | {r['seed_mae'][1]:.3f} | {r['parameters']:,} | {r['mean_seconds']:.1f} | {r['mean_epochs']:.1f} |")
    lines+=['','MAE 单位 cm⁻¹；数值为五个确认折的平均单模型误差，不是集成结果。时间来自并行 CPU 运行。','',
        '![逐折配对差值](figures/paired-confirmation.png)']
    for bank,r in gate['rows'].items():
        lines+=['',f'## {bank} 的逐折确认','',
            f"平均相对改善 {r['relative_improvement']:.2%}；胜出 {r['total_wins']}/10，两个种子分别 {r['seed_wins'][0]}/5、{r['seed_wins'][1]}/5，种子平均后的折胜数 {r['fold_mean_wins']}/5。",'',
            '| 折 | 种子1 KAN | MLP | 差值 | 种子2 KAN | MLP | 差值 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for i in range(5):lines.append(f"| {i+1} | {r['kan_mae'][0][i]:.3f} | {r['mlp_mae'][0][i]:.3f} | {r['delta'][0][i]:+.3f} | {r['kan_mae'][1][i]:.3f} | {r['mlp_mae'][1][i]:.3f} | {r['delta'][1][i]:+.3f} |")
        lines+=['', '预先固定的门槛逐项判定：'+json.dumps(r['criteria'],ensure_ascii=False)+'。']
    lines+=['','## 内层选择记录','',
        '| 输入 | 模型 | 确认折 | 学习率 | 固定重训轮数 |','| --- | --- | ---: | ---: | ---: |']
    for c in read(RUN/'selection.json')['choices']:lines.append(f"| {c['bank']} | {c['variant']} | {c['inner']+1} | {c['lr']} | {c['epochs']} |")
    lines+=['','## 外层测试','']
    if result['outer']:
        lines+=['仅官方第1个外层折、253 个测试样本。不是完整五折 benchmark；集成是预先固定的两个种子等权平均。','',
            '| 输入 | 模型 | 外层种子1 | 外层种子2 | 两种子集成 MAE |','| --- | --- | ---: | ---: | ---: |']
        for r in result['outer']['rows']:lines.append(f"| {r['bank']} | {r['variant']} | {r['member_mae'][0]:.3f} | {r['member_mae'][1]:.3f} | {r['ensemble_mae']:.3f} |")
    else:lines+=['没有评估外层测试。'+('主要门槛未通过，按预先协议停止。' if not gate['passed'] else '主要门槛通过，但外层动作设为暂缓。')]
    lines+=['','## 核验与解释范围','',
        f"源码、输入、模型哈希、嵌套 ID 隔离、原始缓存到缩放矩阵、训练目标均值/标准差、参数量、最佳轮次与固定轮数、学习率选择及门槛均已核验。独立 NumPy 重建 KAN 的最大预测差为 {result['max_numpy_prediction_difference']:.6f} cm⁻¹。",'',
        '本轮双方调参预算完全对称，模型结构与初始化冻结；但不能消除过去的探索和架构选择偏差。数据仍是此前用过的 1,012 个样本，新划分与种子检验的是重新划分后的稳定性，不等于未见过的新数据集。十次拟合互相关联，门槛不是显著性检验。增强特征仍含预训练嵌入，SISSO 已去除但预训练来源未因此得到全面排除。','',
        f"本次运行墙钟 {result['status']['wall_seconds']/60:.1f} 分钟。完整数值见 analysis.json，选择依据见 selection.json，事前规则见 protocol.json。"]
    (RUN/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
