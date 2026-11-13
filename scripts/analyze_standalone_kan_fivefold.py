"""Audit all five frozen enhanced folds, OOF predictions, capacity and timing."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import pickle
import pandas as pd
import analyze_standalone_kan_confirmation as verify
from analyze_standalone_kan_screen import np,plt,read,sha,digest,check,ScalerStateReader
from standalone_kan_confirmation_policy import choose
from standalone_kan_fivefold_policy import official_ids,validate_fivefold,metrics

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'benchmarks/standalone-kan-fivefold-20261028'
SOURCE=ROOT/'benchmarks/standalone-kan-confirmation-20261028'


def audit_input(info,p,cache):
    phase,i=info['phase'],info['inner']
    if phase=='outer':tr,va=p['outer_train_ids'],p['outer_test_ids']
    else:
        split=p['splits'][i];tr,va=split['tuning_train_ids'],split['tuning_validation_ids']
        check(not (set(tr)|set(va))&set(p['outer_test_ids']),'Outer IDs entered tuning')
    check(info['train_ids']==tr==info['feature_selection_fit_ids']==info['scaling_fit_ids']==info['target_transform_fit_ids'],'Preprocessing scope')
    check(info['validation_ids']==va and not set(tr)&set(va),'Evaluation overlap')
    check(info['run_fingerprint']==digest(p),'Input provenance')
    directory=Path(info['directory'])
    for name,value in info['artifacts'].items():check(sha(directory/name)==value,'Changed input artifact')
    features=info['selected_features']
    check(len(features)==len(set(features))==256 and not any('SISSO' in f.upper() for f in features),'Feature scope')
    with (directory/'scaler.pkl').open('rb') as f:scaler=ScalerStateReader(f).load()
    raw=cache.loc[tr,features].values
    np.testing.assert_array_equal(scaler.data_min_,raw.min(axis=0));np.testing.assert_array_equal(scaler.data_max_,raw.max(axis=0))
    with np.load(directory/'arrays.npz') as a:
        np.testing.assert_array_equal(a['x_train'],(raw*scaler.scale_+scaler.min_).astype('float32'))
        np.testing.assert_array_equal(a['y_train'],cache.loc[tr,'target'].values)
        np.testing.assert_allclose([info['target_mean'],info['target_std']],
            [a['y_train'].astype(float).mean(),a['y_train'].astype(float).std()],atol=1e-10,rtol=0)
        if phase=='tuning':
            np.testing.assert_array_equal(a['x_validation'],(cache.loc[va,features].values*scaler.scale_+scaler.min_).astype('float32'))
            np.testing.assert_array_equal(a['y_validation'],cache.loc[va,'target'].values)
    manifest=directory/'manifest.json'
    return dict(source_manifest=str(manifest),source_manifest_sha256=sha(manifest),source=info,
        target_mean=info['target_mean'],target_std=info['target_std'],target_transform_fit_ids=tr)


def audit_fold(directory,p,partial):
    check(sha(p['source_cache'])==p['source_cache_sha256'],'Changed training cache')
    cache=pd.read_pickle(p['source_cache'])
    check(sorted(cache.index.tolist())==p['outer_train_ids'],'Cache train scope')
    check(not set(cache.index)&set(p['outer_test_ids']),'Test sample inside training cache')
    for s in p['splits']:
        a,b,c=[set(s[k]) for k in ('tuning_train_ids','tuning_validation_ids','validation_ids')]
        check(not(a&b or a&c or b&c),'Nested overlap')
        check(a|b==set(s['train_ids']) and a|b|c==set(p['outer_train_ids']),'Nested coverage')
    inputs={}
    for phase in ('tuning','outer'):
        for path in sorted((directory/'inputs'/phase).glob('*/manifest.json')):
            info=read(path);inputs[(phase,info['inner'])]=audit_input(info,p,cache)
    # Reuse the independent fixed-fit auditor with this fold's own protocol and root.
    verify.RUN=directory
    records=[]
    for phase in ('tuning','outer'):
        for path in sorted((directory/'fits'/phase/'enhanced').glob('*/*/*/result.json')):
            records.append(verify.audit_fit(path,p,inputs))
    tuning=[r for r in records if r['phase']=='tuning'];outer=[r for r in records if r['phase']=='outer']
    expected={(v,s,i,lr) for v in ('kan','matched_mlp') for s in range(2) for i in range(5) for lr in p['learning_rates']}
    keys=[(r['variant'],r['repeat'],r['inner'],r['lr']) for r in tuning]
    check(len(keys)==len(set(keys)) and set(keys)<=expected,'Tuning grid')
    if not partial:check(len(tuning)==60 and len(outer)==4,'Incomplete fold')
    if (directory/'outer-selection.json').exists():
        check(len(tuning)==60,'Selection before full budget')
        saved=read(directory/'outer-selection.json')
        check(saved['run_fingerprint']==digest(p),'Selection provenance')
        for c in saved['choices']:
            if c['bank']!='enhanced':continue
            check(c==dict(bank='enhanced',variant=c['variant'],**choose([r for r in tuning if r['variant']==c['variant']],10)),'Selection reproducibility')
        if 'tuning_results_sha256' in saved:
            for name,value in saved['tuning_results_sha256'].items():check(sha(Path(name)/'result.json')==value,'Changed selection result')
    rows=[]
    if (directory/'outer-summary.json').exists():
        check(len(outer)==4,'Summary before outer completion')
        summary=read(directory/'outer-summary.json')
        for variant in ('kan','matched_mlp'):
            rs=sorted([r for r in outer if r['variant']==variant],key=lambda r:r['repeat'])
            check([r['repeat'] for r in rs]==[0,1],'Outer seed grid')
            ps=[]
            for r in rs:
                with np.load(Path(r['directory'])/'predictions.npz') as a:ps.append(a['prediction']);target=a['target'];ids=a['sample_ids']
            with np.load(directory/f'outer-enhanced-{variant}-ensemble.npz') as a:
                np.testing.assert_array_equal(a['prediction'],np.mean(ps,axis=0));np.testing.assert_array_equal(a['target'],target)
                check(a['sample_ids'].tolist()==p['outer_test_ids']==ids.tolist(),'Outer IDs')
                ensemble_mae=float(np.mean(abs(a['prediction']-a['target'])))
            saved=next(r for r in summary['rows'] if r['bank']=='enhanced' and r['variant']==variant)
            check(saved['member_mae']==[r['validation_mae'] for r in rs] and abs(saved['ensemble_mae']-ensemble_mae)<1e-10,'Outer metrics')
            tune=[r for r in tuning if r['variant']==variant]
            rows.append(dict(outer_fold=p['outer_fold'],variant=variant,member_mae=saved['member_mae'],
                ensemble_mae=ensemble_mae,parameters=rs[0]['parameters'],ensemble_parameters=sum(r['parameters'] for r in rs),
                tuning_task_seconds=sum(r['seconds'] for r in tune),refit_task_seconds=sum(r['seconds'] for r in rs),
                total_task_seconds=sum(r['seconds'] for r in tune+rs),mean_refit_seconds=float(np.mean([r['seconds'] for r in rs])),
                tuning_epoch_cap_hits=sum(r['epochs']==400 for r in tune),
                prediction_path=str(directory/f'outer-enhanced-{variant}-ensemble.npz')))
    return dict(records=records,rows=rows,inputs=len(inputs),cache=cache,
                shared_input_task_seconds=sum(inp['source']['seconds'] for inp in inputs.values()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    p=read(RUN/'protocol.json');status=read(RUN/'status.json')
    if not args.partial:check(status['state']=='complete','Extension incomplete')
    for name,value in p['source_sha256'].items():check(sha(ROOT/'scripts'/name)==value,'Frozen source changed')
    for name,value in p['fold0_sources'].items():check(sha(name)==value,'Preserved fold 0 changed')
    validate_fivefold(p['folds'])
    for f in p['folds']:
        check(sha(f['source'])==f['source_sha256'],'Official split changed')
        check(official_ids(read(f['source']))==(f['train_ids'],f['test_ids']),'Official split mapping')
    records=[];rows=[];preparation=[];all_targets={};completed=[]
    for fold in range(5):
        directory=SOURCE if fold==0 else RUN/f'fold-{fold}'
        if not (directory/'protocol.json').exists():
            check(args.partial,'Missing fold protocol');continue
        fp=read(directory/'protocol.json')
        check(fp['outer_fold']==fold and fp['outer_train_ids']==p['folds'][fold]['train_ids'] and fp['outer_test_ids']==p['folds'][fold]['test_ids'],'Wrong fold protocol')
        if fold:check(fp['parent_protocol_sha256']==sha(RUN/'protocol.json'),'Changed parent')
        audit=audit_fold(directory,fp,args.partial)
        records+=audit['records'];rows+=audit['rows'];preparation.append(dict(fold=fold,task_seconds=audit['shared_input_task_seconds']))
        if len(audit['rows'])==2:completed.append(fold)
        for key,value in audit['cache']['target'].items():
            if key in all_targets:check(all_targets[key]==float(value),'Inconsistent raw target across caches')
            all_targets[key]=float(value)
    result=dict(analyzed_at=datetime.now(timezone.utc).isoformat(),analyzer_sha256=sha(__file__),
        inherited_auditor_sha256=sha(verify.__file__),run_fingerprint=digest(p),complete=len(completed)==5,
        verified_core_fits=len(records),completed_folds=completed,fold_rows=rows,shared_preparation=preparation,
        max_numpy_prediction_difference=max([r['numpy_max_prediction_difference'] for r in records if r['numpy_max_prediction_difference'] is not None]+[0]),
        extension_wall_seconds=status['wall_seconds'],limitations=p['limitations'])
    if result['complete']:
        check(len(records)==320,'Core budget should be 320')
        comparisons={};all_oof={}
        for variant in ('kan','matched_mlp'):
            rs=sorted([r for r in rows if r['variant']==variant],key=lambda r:r['outer_fold'])
            ids=[];prediction=[];target=[];fold_index=[]
            for r in rs:
                with np.load(r['prediction_path']) as a:
                    ids+=a['sample_ids'].tolist();prediction+=a['prediction'].tolist();target+=a['target'].tolist()
                    fold_index += [r['outer_fold']]*len(a['target'])
            check(len(ids)==len(set(ids))==1265 and set(ids)==set(all_targets),'OOF coverage')
            np.testing.assert_array_equal(target,[all_targets[x] for x in ids])
            order=np.argsort(ids);ids=np.array(ids)[order];prediction=np.array(prediction)[order];target=np.array(target)[order];fold_index=np.array(fold_index)[order]
            np.savez_compressed(RUN/f'oof-{variant}.npz',sample_ids=ids,prediction=prediction,target=target,outer_fold=fold_index)
            all_oof[variant]=dict(prediction=prediction,target=target)
            check(abs(float(np.mean(abs(prediction-target)))-np.mean([r['ensemble_mae'] for r in rs]))<1e-10,'OOF/mean fold MAE mismatch')
        k=sorted([r for r in rows if r['variant']=='kan'],key=lambda r:r['outer_fold'])
        m=sorted([r for r in rows if r['variant']=='matched_mlp'],key=lambda r:r['outer_fold'])
        comparisons['ensemble']=metrics([r['ensemble_mae'] for r in k],[r['ensemble_mae'] for r in m])
        for s in range(2):comparisons[f'seed_{s+1}']=metrics([r['member_mae'][s] for r in k],[r['member_mae'][s] for r in m])
        comparisons['mean_single_member']=metrics([np.mean(r['member_mae']) for r in k],[np.mean(r['member_mae']) for r in m])
        result['comparisons']=comparisons
        result['timing']={v:{key:sum(r[key] for r in rows if r['variant']==v) for key in ('tuning_task_seconds','refit_task_seconds','total_task_seconds')} for v in ('kan','matched_mlp')}
        plot(result);report(result)
    filename='partial-analysis.json' if args.partial else 'analysis.json'
    (RUN/filename).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('complete','verified_core_fits','completed_folds','max_numpy_prediction_difference')},ensure_ascii=False))
    if 'comparisons' in result:print(json.dumps(result['comparisons'],ensure_ascii=False,indent=2))


def plot(a):
    rows=a['fold_rows'];fig,axes=plt.subplots(1,2,figsize=(11,4.5));x=np.arange(1,6)
    for v,label in [('kan','Standalone KAN'),('matched_mlp','Matched MLP')]:
        r=sorted([r for r in rows if r['variant']==v],key=lambda r:r['outer_fold'])
        axes[0].plot(x,[z['ensemble_mae'] for z in r],'-o',label=label)
    axes[0].set_ylabel('Outer MAE (cm$^{-1}$)');axes[0].set_xlabel('Official outer fold');axes[0].set_xticks(x);axes[0].legend()
    delta=a['comparisons']['ensemble']['fold_absolute_improvement']
    bars=axes[1].bar(x,delta,color=['#00838f' if d>0 else '#c65d3b' for d in delta]);axes[1].bar_label(bars,fmt='%.2f',padding=3)
    axes[1].axhline(0,color='gray',lw=.8);axes[1].set_xlabel('Official outer fold');axes[1].set_xticks(x)
    axes[1].set_ylabel('MLP MAE - KAN MAE; positive favors KAN')
    fig.suptitle('Frozen enhanced-feature comparison | equal tuning budgets\nBoth models: fixed two-seed mean prediction; all five folds included')
    fig.tight_layout(rect=(0,0,1,.88));out=RUN/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'fivefold-comparison.png',dpi=180);fig.savefig(out/'fivefold-comparison.pdf');plt.close(fig)


def report(a):
    rows=a['fold_rows'];c=a['comparisons']['ensemble']
    lines=['# 增强特征完整 KAN 与等参数 MLP：官方五折汇总','',
        f"五折核心比较共核验 {a['verified_core_fits']} 次训练；第1折沿用已冻结结果，剩余四折全部纳入。每折两种子等权平均预测的 MAE：KAN {c['kan_mean']:.3f} ± {c['kan_fold_sample_sd']:.3f}，MLP {c['mlp_mean']:.3f} ± {c['mlp_fold_sample_sd']:.3f} cm⁻¹；± 为折间样本标准差，不是置信区间。",'',
        f"总体相对改善 {c['relative_improvement']:.2%}，胜出 {c['wins']}/5 折。参数量：KAN 单模型35,994、双模型集成71,988；MLP 单模型35,995、双模型集成71,990。",'',
        '| 外层折 | KAN 集成 MAE | MLP 集成 MAE | 绝对改善 | 相对改善 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for f in range(5):
        k=next(r for r in rows if r['outer_fold']==f and r['variant']=='kan');m=next(r for r in rows if r['outer_fold']==f and r['variant']=='matched_mlp')
        lines.append(f"| {f+1} | {k['ensemble_mae']:.3f} | {m['ensemble_mae']:.3f} | {m['ensemble_mae']-k['ensemble_mae']:+.3f} | {1-k['ensemble_mae']/m['ensemble_mae']:+.2%} |")
    lines+=['','![五折比较](figures/fivefold-comparison.png)','','## 单种子结果','','| 外层折 | KAN 种子1 | MLP 种子1 | KAN 种子2 | MLP 种子2 |','| --- | ---: | ---: | ---: | ---: |']
    for f in range(5):
        k=next(r for r in rows if r['outer_fold']==f and r['variant']=='kan');m=next(r for r in rows if r['outer_fold']==f and r['variant']=='matched_mlp')
        lines.append(f"| {f+1} | {k['member_mae'][0]:.3f} | {m['member_mae'][0]:.3f} | {k['member_mae'][1]:.3f} | {m['member_mae'][1]:.3f} |")
    for key,label in [('seed_1','种子1'),('seed_2','种子2'),('mean_single_member','单模型均值')]:
        r=a['comparisons'][key];lines+=['',f"{label}：五折平均 KAN {r['kan_mean']:.3f}、MLP {r['mlp_mean']:.3f}；相对改善 {r['relative_improvement']:.2%}。"]
    lines+=['','## 参数与耗时','','每个模型每折30次调参＋2次外层重训。下表为worker任务累计秒数，包含模型构建、预测和保存；并发执行下不能相加当作等待时间。','','| 外层折 | 模型 | 单模型参数 | 调参累计秒 | 重训累计秒 | 单次重训均秒 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in sorted(rows,key=lambda r:(r['outer_fold'],r['variant'])):
        lines.append(f"| {r['outer_fold']+1} | {r['variant']} | {r['parameters']:,} | {r['tuning_task_seconds']:.1f} | {r['refit_task_seconds']:.1f} | {r['mean_refit_seconds']:.1f} |")
    for v,t in a['timing'].items():lines+=['',f"{v} 五折累计任务时间：调参 {t['tuning_task_seconds']/60:.2f} 分钟、外层重训 {t['refit_task_seconds']/60:.2f} 分钟，合计 {t['total_task_seconds']/60:.2f} 分钟。"]
    lines+=['',f"共享输入准备累计任务时间：{sum(r['task_seconds'] for r in a['shared_preparation'])/60:.2f} 分钟；本轮新增四折实际墙钟：{a['extension_wall_seconds']/60:.2f} 分钟。第1折原始运行包含其他实验，不把其28.5分钟直接加到本轮墙钟。",'',
        '## 协议与核验','',
        '结构、近线性初始化、三个学习率候选、种子列表、早停和固定重训轮数选择规则均冻结。每折只用该折训练范围内的五组调参验证集选学习率，完整外层训练按选定轮数重训；外层标签不参与选择。所有SISSO排除，每个训练范围独立筛特征和拟合缩放。双方预算相同，没有择优种子或丢弃不利折。','',
        f"原第1折文件哈希、全部320次任务签名、各折样本隔离、缩放矩阵、目标标准化、参数量、预测MAE、学习率选择、固定轮数和完整1,265样本折外覆盖均已核验。独立NumPy重建KAN最大预测差 {a['max_numpy_prediction_difference']:.6f} cm⁻¹。",'',
        '前期架构选择用过第1外层训练样本，其中含其余折的测试样本；且扩展决定是在看到第1折结果之后作出。因此这是冻结方法后的五折评估，不能称为全开发过程完全未见过测试数据的独立外部确认。当前每折拟合与调参保持隔离。预训练嵌入的来源限制仍保留。','',
        '完整数值：analysis.json；两模型全体折外预测：oof-kan.npz、oof-matched_mlp.npz；逐折配置和训练结果在fold-1至fold-4，第一折来源由protocol.json固定。']
    (RUN/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
