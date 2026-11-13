"""Audit the initialization follow-up and compare all standalone candidates."""
from datetime import datetime, timezone
from pathlib import Path
import json
import pickle

import analyze_standalone_kan_screen as audit
from analyze_standalone_kan_screen import np, plt, read, sha, digest, check

ROOT = audit.ROOT
PARENT = audit.RUN
RUN = ROOT / 'benchmarks/standalone-kan-linear-init-20261027'


def verify_fit(path, protocol):
    r = read(path)
    parent = protocol['parent_protocol']
    fp = digest(protocol)
    inp = parent['source_inputs'][r['inner']]
    info = inp['source']
    job = dict(directory=r['directory'], bank=r['bank'], variant=r['variant'],
               repeat=r['repeat'], inner=r['inner'], seed=r['seed'],
               lr=protocol['learning_rate'], max_epochs=protocol['max_epochs'],
               expected_parameters=parent['expected_parameters'][r['bank']]['kan'],
               input=inp, run_fingerprint=fp)
    check(r['signature'] == digest(job) and r['run_fingerprint'] == fp, 'Job fingerprint')
    check(Path(r['directory']).resolve() == path.parent.resolve(), 'Wrong directory')
    check(r['input_digest'] == digest(inp), 'Input digest')
    check(r['variant'] == 'kan' and r['lr'] == protocol['learning_rate'], 'Wrong candidate')
    check(r['seed'] == protocol['seeds'][r['repeat']] + r['inner'], 'Unpaired seed')
    check(r['train_ids'] == info['train_ids'] == r['target_transform_fit_ids'], 'Training scope')
    check(info['train_ids'] == info['feature_selection_fit_ids'] == info['scaling_fit_ids'],
          'Feature selection or scaling used another scope')
    check(r['validation_ids'] == info['validation_ids'], 'Validation scope')
    check(not set(r['train_ids']) & set(r['validation_ids']), 'Train/validation overlap')
    check(sha(inp['source_manifest']) == inp['source_manifest_sha256'], 'Changed manifest')
    for name, value in info['artifacts'].items():
        check(sha(Path(info['directory']) / name) == value, 'Changed input artifact')
    for name, value in r['artifacts'].items():
        check(sha(path.parent / name) == value, 'Changed fit artifact')
    features = info['selected_features'] if r['bank'] == 'enhanced' else info['selected_features'][:16]
    check(r['features'] == features, 'Feature mismatch')
    check(not any('SISSO' in f.upper() for f in features), 'SISSO found')
    with np.load(path.parent / 'predictions.npz') as saved:
        check(saved['sample_ids'].tolist() == info['validation_ids'], 'Prediction IDs')
        check(saved['train_ids'].tolist() == info['train_ids'], 'Training prediction IDs')
        mae = float(np.mean(abs(saved['prediction'] - saved['target'])))
        train_mae = float(np.mean(abs(saved['train_prediction'] - saved['train_target'])))
        with np.load(Path(info['directory']) / 'arrays.npz') as source:
            np.testing.assert_array_equal(saved['target'], source['y_validation'])
            np.testing.assert_array_equal(saved['train_target'], source['y_train'])
            np.testing.assert_allclose([r['target_mean'], r['target_std']],
                [source['y_train'].astype(float).mean(), source['y_train'].astype(float).std()],
                atol=1e-10, rtol=0)
    check(abs(mae - r['validation_mae']) < 1e-10, 'Validation MAE mismatch')
    check(abs(train_mae - r['train_mae']) < 1e-10, 'Training MAE mismatch')
    check(1 <= r['best_epoch'] <= r['epochs'] <= protocol['max_epochs'], 'Epoch bounds')
    check(r['best_epoch'] == int(np.argmin(r['history']['val_mae'])) + 1, 'Best checkpoint')
    check(abs(mae - r['checkpoint_mae']) < .003, 'Checkpoint units')
    with (path.parent / 'model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    check(sum(w.size for w in model['weights']) == r['parameters'] == job['expected_parameters'], 'Capacity')
    classes = [layer['class_name'].removeprefix('StandaloneKAN>')
               for layer in json.loads(model['model_json'])['config']['layers']]
    check(classes.count('SplineEdgeLayer') == 3 and set(classes) <= {'InputLayer', 'SplineEdgeLayer'},
          'Not a standalone three-layer KAN: ' + repr(classes))
    network = read(path.parent / 'network.json')
    check(network['features'] == features, 'Export feature mismatch')
    return r, audit.audit_network(path.parent, r, inp)


def paired(fits, controls):
    refs = {(r['repeat'], r['inner']): r for r in controls}
    pairs = []
    for r in sorted(fits, key=lambda x: (x['repeat'], x['inner'])):
        ref = refs[(r['repeat'], r['inner'])]
        check(r['input_digest'] == ref['input_digest'] and r['features'] == ref['features'], 'Unpaired inputs')
        check(r['train_ids'] == ref['train_ids'] and r['validation_ids'] == ref['validation_ids'], 'Unpaired IDs')
        check(r['seed'] == ref['seed'] and r['lr'] == ref['lr'], 'Unpaired training')
        pairs.append(dict(repeat=r['repeat'], inner=r['inner'], kan_mae=r['validation_mae'],
                          reference_mae=ref['validation_mae'], delta=r['validation_mae']-ref['validation_mae']))
    return dict(wins=sum(p['delta'] < 0 for p in pairs), pairs=pairs,
                seed_wins=[sum(p['delta'] < 0 for p in pairs if p['repeat'] == s) for s in range(2)],
                seed_delta=[float(np.mean([p['delta'] for p in pairs if p['repeat'] == s])) for s in range(2)],
                seed_improvement_percent=[100*(1-np.mean([p['kan_mae'] for p in pairs if p['repeat']==s]) /
                    np.mean([p['reference_mae'] for p in pairs if p['repeat']==s])) for s in range(2)])


def plots(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for bank, ax in zip(('enhanced', 'physical'), axes):
        candidates = [r for r in rows if r['bank'] == bank and r['variant'] in ('modnet', 'matched_mlp', 'linear_init_kan')]
        labels = {'modnet':'MODNet', 'matched_mlp':'Matched MLP', 'linear_init_kan':'Standalone KAN\nnear-linear init'}
        x = np.arange(len(candidates))
        for s in range(2):
            bars = ax.bar(x+(s-.5)*.34, [r['seed_mae'][s] for r in candidates], .34, label=f'Seed {s+1}')
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        ax.set_xticks(x, [labels[r['variant']] for r in candidates])
        ax.set_ylim(0, max(max(r['seed_mae']) for r in candidates)*1.2)
        ax.set_ylabel('Inner-validation MAE (cm$^{-1}$); lower is better')
        ax.set_title('256 enhanced features' if bank == 'enhanced' else '16 physical descriptors')
        ax.legend(fontsize=8)
    fig.suptitle('Standalone KAN replaces the predictor | 5 inner folds per seed\nExploratory follow-up; same per-fit budget; no outer test', fontsize=11)
    fig.tight_layout(rect=(0,0,1,.9))
    fig.savefig(output/'comparison.png', dpi=180)
    fig.savefig(output/'comparison.pdf')
    plt.close(fig)


def main():
    protocol, status = read(RUN/'protocol.json'), read(RUN/'status.json')
    check(status['state'] == 'complete' and status['completed'] == 20, 'Incomplete follow-up')
    check(sha(PARENT/'protocol.json') == protocol['parent_protocol_sha256'], 'Parent protocol changed')
    check(read(PARENT/'protocol.json') == protocol['parent_protocol'], 'Embedded protocol differs')
    for name, value in protocol['source_sha256'].items():
        check(sha(ROOT/'scripts'/name) == value, 'Changed frozen source: '+name)
    previous = read(PARENT/'analysis.json')
    check(previous['audited_fits'] == 70 and previous['run_fingerprint'] == digest(protocol['parent_protocol']), 'Parent audit missing')
    check(previous['analyzer_sha256'] == sha(audit.__file__), 'Stale parent audit')
    references = [read(path) for path in sorted((PARENT/'fits').glob('*/*/*/result.json'))]
    records, details = [], []
    for path in sorted((RUN/'fits').glob('*/*/*/result.json')):
        r, detail = verify_fit(path, protocol)
        records.append(r)
        details.append(detail)
        print('AUDIT', r['bank'], r['repeat'], r['inner'], flush=True)
    expected = {(b, s, i) for b in protocol['banks'] for s in range(2) for i in range(5)}
    check(len(records) == 20 and {(r['bank'],r['repeat'],r['inner']) for r in records} == expected, 'Incomplete grid')
    rows = []
    for bank in protocol['banks']:
        fits = [r for r in records if r['bank'] == bank]
        ds = [r for r in details if r['bank'] == bank]
        values = [[r['validation_mae'] for r in sorted(fits, key=lambda r:r['inner']) if r['repeat']==s] for s in range(2)]
        item = dict(bank=bank, variant='linear_init_kan', seed_mae=[float(np.mean(x)) for x in values],
            fold_mae=values, mean_mae=float(np.mean(values)), parameters=fits[0]['parameters'],
            train_mae=float(np.mean([r['train_mae'] for r in fits])),
            mean_seconds=float(np.mean([r['seconds'] for r in fits])),
            mean_best_epoch=float(np.mean([r['best_epoch'] for r in fits])),
            epoch_cap_hits=sum(r['epochs'] == 400 for r in fits),
            comparisons={v:paired(fits,[r for r in references if r['bank']==bank and r['variant']==v])
                         for v in (['modnet','matched_mlp','kan'] if bank=='enhanced' else ['matched_mlp','kan'])},
            pruned_mae=float(np.mean([d['pruned_mae'] for d in ds])),
            pruned_prediction_shift=float(np.mean([d['pruned_prediction_shift'] for d in ds])),
            pruned_mean_abs_mae_change=float(np.mean([abs(d['pruned_mae']-d['original_mae']) for d in ds])),
            pruned_max_mae_increase=float(max(d['pruned_mae']-d['original_mae'] for d in ds)),
            symbolic_mae=float(np.mean([d['symbolic_mae'] for d in ds])),
            symbolic_prediction_shift=float(np.mean([d['symbolic_prediction_shift'] for d in ds])),
            symbolic_max_mae_increase=float(max(d['symbolic_mae']-d['original_mae'] for d in ds)),
            mean_kept_edges=float(np.mean([d['kept_edges'] for d in ds])), total_edges=ds[0]['edges'],
            symbolic_edge_fraction=sum(d['symbolic_edges'] for d in ds)/sum(d['edges'] for d in ds),
            simplified_degree_counts={str(k):sum(d['layers'][l]['degree_counts'][str(k)] for d in ds for l in range(3)) for k in range(-1,4)})
        rows.append(item)
    all_rows = previous['rows'] + rows
    figures = RUN/'figures'
    figures.mkdir(exist_ok=True)
    ale = audit.plot_ale(details, figures)
    plots(all_rows, figures)
    result = dict(analyzed_at=datetime.now(timezone.utc).isoformat(), analyzer_sha256=sha(__file__),
        network_analyzer_sha256=sha(audit.__file__), parent_analysis_sha256=sha(PARENT/'analysis.json'),
        run_fingerprint=digest(protocol), audited_fits=20, combined_audited_fits=90,
        max_numpy_prediction_difference=max(d['numpy_max_prediction_difference'] for d in details),
        rows=rows, all_rows=all_rows, network_audits=details, physical_ale_stability=ale,
        followup_wall_minutes=status['wall_seconds']/60, parent_wall_minutes=previous['wall_minutes'],
        outer_test_evaluated=False, adaptation=protocol['adaptation'], search_budget_limitation=protocol['search_budget_limitation'])
    (RUN/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report(result)
    print(json.dumps({k:result[k] for k in ('combined_audited_fits','max_numpy_prediction_difference','rows','physical_ale_stability')}, ensure_ascii=False, indent=2))


def report(result):
    lines = ['# 完整 KAN 替代 MODNet：90 次训练汇总', '',
        '2026-10-28 核验。70 次初筛与 20 次初始化追加实验均已完成；全部为 phonons 第一外层训练折中的五折内层验证，两个训练种子。MAE 单位 cm⁻¹，越低越好。', '',
        '近线性初始化的完整 KAN 在两个种子的均值上都优于同输入的等参数 MLP；增强输入组同时优于 MODNet。该 KAN 为 256→17→8→1，物理输入版为 16→17→8→1；每条边是可学习三次 B 样条，隐藏坐标逐维经过 0.5*tanh。没有 MODNet 主干、Dense 层或残差支路。', '',
        '追加实验只改变初始边函数：用 Greville 节点初始化成接近直线，再允许训练学习非线性。参数量、网络、特征、数据划分、优化器、学习率、最大轮次、早停规则完全沿用初筛。', '',
        '| 输入 | 模型 | 种子 1 | 种子 2 | 参数量 | 平均每次秒数 | 达 400 轮上限 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    names = {'modnet':'MODNet', 'matched_mlp':'等参数 MLP', 'kan':'KAN，随机曲线初始化',
             'sparse_kan':'稀疏＋平滑 KAN，随机曲线初始化', 'linear_init_kan':'KAN，近线性初始化'}
    for r in result['all_rows']:
        lines.append(f"| {r['bank']} | {names[r['variant']]} | {r['seed_mae'][0]:.3f} | {r['seed_mae'][1]:.3f} | {r['parameters']:,} | {r['mean_seconds']:.1f} | {r['epoch_cap_hits']}/10 |")
    lines += ['', '![结果对比](figures/comparison.png)', '',
        '同组同折、同种子的配对比较如下。差值为 KAN − 对照，负值代表 KAN 更好；两种子使用相同的五个划分，因此十次拟合不是十个独立数据集。']
    for r in result['rows']:
        for v in ('modnet','matched_mlp'):
            if v not in r['comparisons']: continue
            c=r['comparisons'][v]
            lines += ['', f"## {r['bank']}：近线性初始化 KAN 对 {names[v]}", '',
                f"胜出 {c['wins']}/10 次；按种子分别为 {c['seed_wins'][0]}/5、{c['seed_wins'][1]}/5。种子均值相对改善 {c['seed_improvement_percent'][0]:.2f}%、{c['seed_improvement_percent'][1]:.2f}%。", '',
                '| 内层折 | 种子 1 KAN | 对照 | 差值 | 种子 2 KAN | 对照 | 差值 |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
            for i in range(5):
                a,b=[next(p for p in c['pairs'] if p['repeat']==s and p['inner']==i) for s in range(2)]
                lines.append(f"| {i+1} | {a['kan_mae']:.3f} | {a['reference_mae']:.3f} | {a['delta']:+.3f} | {b['kan_mae']:.3f} | {b['reference_mae']:.3f} | {b['delta']:+.3f} |")
    lines += ['', '## 可解释性审计', '',
        '对全部 60 个 KAN（初筛 40、追加 20）从导出的边函数独立重建完整预测；追加组最大 NumPy/原训练框架预测差为 '
        f"{result['max_numpy_prediction_difference']:.6f} cm⁻¹。训练来源、模型文件哈希、样本 ID、目标标准化、参数量、保存预测 MAE 和最佳轮次均通过检查。", '',
        '| 输入 | 原始 MAE | 剪枝后 | 简式替换后 | 保留边数 | 简式边比例 | 简式预测平均偏移 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in result['rows']:
        lines.append(f"| {r['bank']} | {r['mean_mae']:.3f} | {r['pruned_mae']:.3f} | {r['symbolic_mae']:.3f} | {r['mean_kept_edges']:.1f}/{r['total_edges']} | {r['symbolic_edge_fraction']:.1%} | {r['symbolic_prediction_shift']:.3f} |")
    lines += ['',
        '剪枝阈值为各层训练边响应标准差最大值的 5%，删除边的训练均值补入节点偏置；无剪后微调，且未压缩参数张量。简式库固定为常数、一至三次多项式，训练拟合 RMSE ≤ 边响应标准差的 5% 才接受，其余保留样条。剪枝与简式是两个各自从原网络出发的消融，没有合并。所有选择只用训练激活，报告的是从原输入运行整个修改网络的验证误差。', '',
        '简式边比例可能包含贡献较小的边，不能代表同等比例的预测机制已被解释。当前还是多层函数网络，不能直接称为简洁物理定律。', '',
        '本轮剪枝仅去掉增强组约 3.3%、物理组约 1.3% 的边，压缩程度很小。增强组简式替换后 MAE 从 32.567 升至 34.304，已基本失去相对配对 MLP 均值 34.272 的优势；暂不推荐以该简式网络替代原模型。物理组简式替换后从 78.478 升至 78.893，但单样本预测平均变化仍达 7.749，不能把均值 MAE 接近解释为预测完全一致。', '',
        '![物理量对最终预测的累积局部效应](figures/physical-ale-kan.png)', '',
        '上图使用每次拟合的训练数据计算完整输出的一阶 ALE，比较同名物理输入而非任意编号的隐藏节点。阴影为十次拟合的标准差，并非置信区间。相关描述符、隐藏交互和训练样本支持范围限制解释；ALE 不是因果效应，也不是精确加性分解。', '',
        '| 物理描述符 | 跨拟合曲线相关系数中位数 | 端点整体上升次数 / 10 |',
        '| --- | ---: | ---: |']
    for r in result['physical_ale_stability'].get('kan',[]):
        corr=r['median_curve_correlation']
        feature=r['feature'].replace('|', '\\|')
        lines.append(f"| {feature} | {corr:.3f} | {r['positive_endpoint_trends']} |" if corr is not None else f"| {feature} | 无定义 | {r['positive_endpoint_trends']} |")
    lines += ['', '## 结论的适用范围与下一步', '',
        '这轮支持继续采用完整 KAN 替换预测器，并固定近线性初始化作为待确认候选。随机曲线初始化明显更差，说明该实现的初始化选择会显著影响结果；这不证明所有 KAN 实现必然同样表现。', '',
        '256 输入组保留无 SISSO 的 MatterVial 增强特征，包含预训练嵌入，不能把每条边都赋予物理含义。16 个物理描述符版更容易提出可检验假设，但绝对预测误差明显较高。物理可解释性与增强输入精度应分别报告。', '',
        '物理版中，平均原子质量、平均共价半径、平均键长对应的整网 ALE 曲线在十次拟合中均呈端点下降，跨拟合曲线相关中位数分别约 0.992、0.997、0.998；每原子体积的曲线一致性较弱（0.580）。这些是模型内部响应的稳定性证据，可以用于提出后续物理检验问题，不能当作因果规律。ALE 本身也适用于 MLP，并非 KAN 特有优势；KAN 特有的边函数导出、训练剪枝和逐边简式替换已单独审计。', '',
        '这是观察初筛第一种子后追加的探索，KAN 多试了一种初始化而 MLP 没有追加搜索；每次训练预算相同，不等于模型家族总搜索预算相同。内层划分和种子此前已反复使用，不能当作独立确认，更不能据此声称超过论文外层成绩。未读取或评估外层测试。预训练嵌入的数据来源风险不因去除 SISSO 自动消失。', '',
        '下一步应先冻结网络与初始化，在未使用的验证划分/新种子上做小规模配对确认，并给 MLP 对称的调参预算；新种子只能检查优化稳定性，不能替代新数据划分或外层评估。确认后再按预先固定的协议跑外层测试。解释性方面优先检查训练剪枝、剪后微调以及整网简化的精度保持和跨折稳定性，仍只以内层数据做选择。', '',
        f"初筛墙钟 {result['parent_wall_minutes']:.2f} 分钟；追加实验墙钟 {result['followup_wall_minutes']:.2f} 分钟，两者时间重叠，不能相加当作总等待时间。各模型每次耗时来自并行 CPU worker，受到并发负载影响，不是隔离的运行速度基准。", '',
        '可复查产物：本目录 `analysis.json`；每个 `fits/<bank>/kan/seed-<s>-inner-<i>/` 下的 `network.json`、`pruned-network.json`、`symbolic-network.json`、`interpretation.json`。训练源码保持冻结。']
    (RUN/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
