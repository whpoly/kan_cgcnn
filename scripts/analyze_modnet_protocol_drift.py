"""Compare frozen protocols and saved inner results without fitting/selecting on tests."""
from pathlib import Path
import ast
import collections
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'benchmarks/official-aligned-20261021/matbench_phonons'
NEW=ROOT/'benchmarks/fourway-phonons-20261029'
OUT=ROOT/'benchmarks/modnet-baseline-diagnosis-20261029'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def count(rows,key,nested=None):
    return dict(collections.Counter(str((r[nested] if nested else r)[key]) for r in rows))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p=read(NEW/'protocol.json');rows=[];source_checks=[]
    for name in ('models/vanilla.py','models/ensemble.py','preprocessing.py','hyper_opt/fit_genetic.py'):
        a=Path('C:/Users/User/.conda/envs/modnet-v012-matbench/Lib/site-packages/modnet')/name
        b=Path('C:/Users/User/.conda/envs/modnet-v013-mattervial/Lib/site-packages/modnet')/name
        source_checks.append(dict(file=name,identical_bytes=a.read_bytes()==b.read_bytes(),
            identical_python_ast=ast.dump(ast.parse(a.read_text()))==ast.dump(ast.parse(b.read_text()))))
    for fold in range(5):
        d=OLD/f'fold-{fold}';frozen=read(d/'frozen-ensemble.json')
        old=[read(Path(m['directory'])/'result.json') for m in frozen['members']]
        newdir=NEW/'mm_modnet'/f'fold-{fold}'
        old_features=read(d/'outer-split.json')['selected_features']
        oldval=float(np.mean([r['validation_mae'] for r in old]))
        row=dict(fold=fold,old_search_fits=len(list((d/'modnet').glob('*/result.json'))),
            old_mae=read(d/'baseline.json')['results']['modnet']['mae'],old_mean_validation_mae=oldval,
            old_median_epochs=float(np.median([r['epochs'] for r in old])),
            old_mean_parameters=float(np.mean([r['parameters'] for r in old])),
            old_selected_genes={k:count(old,k,'genes') for k in ('n_feat','n_neurons_first_layer','fraction1','fraction2','fraction3','lr','batch_size','xscale')},
            old_first_candidate=old[0]['genes'],old_ranking_length=len(old_features),old_selected_configurations_supported_by_new_grid=0)
        oldconfigs=[old[i]['genes'] for i in range(0,len(old),5)]
        for g in oldconfigs:
            if any(g['n_feat']==c['n_feat'] and g['n_neurons_first_layer']==c['width'] and g['lr']==c['lr']
                and g['batch_size']==c['batch_size'] and g['xscale']=='minmax'
                and [g['fraction1'],g['fraction2'],g['fraction3']]==[.75,.5,1] for c in p['candidates']):
                row['old_selected_configurations_supported_by_new_grid']+=1
        if (newdir/'complete.json').exists():
            newscore=read(newdir/'complete.json');selection=read(newdir/'frozen-selection.json')
            new=[read(Path(m)/'result.json') for m in selection['members']]
            with np.load(newdir/'outer-predictions.npz') as z:
                mae=float(np.mean(abs(z['prediction']-z['target'])))
                np.testing.assert_allclose(mae,newscore['mae'],atol=1e-9,rtol=0)
            overlaps=[]
            for i in range(5):
                info=read(NEW/'inputs/mm'/f'fold-{fold}'/f'inner-{i}'/'manifest.json')
                features=info['selected_features']
                overlaps.append(dict(inner=i,old_top100_surviving_rfe=sum(f in features for f in old_features[:100]),
                    old_top100_in_new_top100=len(set(old_features[:100])&set(features[:100])),
                    old_first_candidate_features_surviving_rfe=sum(f in features for f in old[0]['selected_features']),
                    appended_low_entropy_features=len(info.get('entropy_filtered_appended',[]))))
            allnew=[read(x) for x in (newdir/'fits').glob('*/result.json')]
            row.update(new_mae=mae,new_search_fits=newscore['search_fits'],
                new_mean_validation_mae=float(np.mean([r['validation_mae'] for r in new])),
                new_median_epochs=float(np.median([r['epochs'] for r in new])),
                new_median_best_epoch=float(np.median([r['best_epoch'] for r in new])),
                new_mean_parameters=newscore['parameters_mean'],feature_overlap=overlaps,
                new_selected_configs={k:count(new,k,'config') for k in ('n_feat','width','lr','batch_size')},
                new_epoch_cap_hits=sum(r['epochs']==800 for r in allnew),new_all_fit_count=len(allnew))
        rows.append(row)
    matched=[r for r in rows if 'new_mae' in r]
    result=dict(kind='Protocol drift audit; descriptive, not causal ablation',rows=rows,source_checks=source_checks,
        old_fivefold_mae=float(np.mean([r['old_mae'] for r in rows])),
        completed_new_folds=len(matched),new_partial_mean=float(np.mean([r['new_mae'] for r in matched])),
        old_same_folds_mean=float(np.mean([r['old_mae'] for r in matched])),
        old_selected_candidates_in_new_grid=sum(r['old_selected_configurations_supported_by_new_grid'] for r in rows),
        old_selected_candidates_total=50,
        interpretation=['Multiple changes were introduced together. Original 34.013 reproduction cannot establish the strength of this changed baseline.',
            'All reported partial scores are descriptive; no hyperparameters are chosen by outer results.',
            'Lower selected inner scores are not directly comparable across changed splits, fitting scope and validation checkpoint selection.',
            'Strict inner feature isolation should not be removed to chase published performance; paper-style reproduction must be labeled separately.'])
    (OUT/'protocol-drift.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# MODNet基线退化：冻结协议与已保存产物对照','',
        '已暂停新四组流程；保留全部3个完成外折以及第4折部分预处理。不将部分结果称为完整五折，不用它们给KAN报提升。','',
        '| 项目 | 原复现协议 | 新统一协议 |','| --- | --- | --- |',
        '| 外层划分、原始MatMiner | 官方五折、2059列 | 相同 |',
        '| 特征筛选 | 外层训练排序，内层复用；外部预计算cross-NMI | 每个内层重新去常量、XGB RFE800、训练内cross-NMI和RR |',
        '| 训练目标 | 原始cm⁻¹ | 训练集均值/标准差标准化 |',
        '| 早停 | 训练loss，patience30、min_delta0.001，不恢复最佳权重 | 验证MAE，patience40、min_delta0.05cm⁻¹，恢复最低验证误差权重 |',
        '| 搜索 | 原生GA，宽度及三层比例可变，特征数细粒度变化 | 192点固定网格，四档特征数、四档宽度、固定隐藏比例 |',
        '| 输入缩放候选 | minmax和standard | 只有minmax |',
        '| 学习率候选 | 初始0.1/0.01/0.005/0.001 | 0.01/0.003/0.001/0.0003 |',
        '| batch候选 | 初始32/64/128/256，GA还可变异 | 32/64/128 |',
        '| 内层划分、种子 | 原GA随机排列后内折和原种子 | 新内折和新种子 |',
        '| 集成 | 前10配置×5内层模型=50 | 形式相同，选中的模型、训练时长和多样性不同 |','',
        '| 外折 | 原MAE | 新MAE | 原入选模型epoch中位数 | 新停止epoch中位数 | 新最佳epoch中位数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in matched:
        lines.append(f"| {r['fold']+1} | {r['old_mae']:.6f} | {r['new_mae']:.6f} | {r['old_median_epochs']:.1f} | {r['new_median_epochs']:.1f} | {r['new_median_best_epoch']:.1f} |")
    lines+=['',f"原五折MAE {result['old_fivefold_mae']:.6f}。新协议仅完成{len(matched)}折，均值{result['new_partial_mean']:.6f}；原协议相同折均值{result['old_same_folds_mean']:.6f}。",'',
        f"原五折共50个入选配置（每个5模型），新网格能够原样表达{result['old_selected_candidates_in_new_grid']}个。增加训练次数没有保留原来的搜索能力。",'',
        '网络vanilla.py字节完全相同；版本间其他相关源码的AST一致性见JSON。版本号变化本身不能解释当前差距。',
        '新模型训练更早结束、搜索空间限制、特征排名改变及验证集兼作checkpoint与超参选择，都是有证据支持的排查方向；尚不能仅凭这些观测量化单项因果贡献。','',
        '原复现的外层测试仍隔离，但特征监督排序使用了全部外层训练（包含内层验证），与严格嵌套的含义不同。不能为了追34.27把这一差异隐藏掉。','',
        '下一步：原协议固定选中配置从头重训验证运行一致性；只在原外层第1折训练范围内做配对训练/特征消融。后者不读取外层测试标签，不据外层成绩挑变体。']
    (OUT/'protocol-drift.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','interpretation')},indent=2))


if __name__=='__main__':main()
