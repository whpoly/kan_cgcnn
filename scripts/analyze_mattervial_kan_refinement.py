"""Independently audit saved refinement artifacts without importing training code."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'benchmarks/mattervial-kan-refinement-20261026'
LABELS = dict(modnet='MatterVial + MODNet', group_mlp='物理分组 MLP',
              group_kan='无正则物理分组 KAN', group_sparse_diagnostic='原正则物理分组 KAN',
              latent_mlp='隐藏表示残差 MLP', latent_kan='隐藏表示残差 KAN')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, tolerance=1e-10):
    check(np.allclose(actual, expected, atol=tolerance, rtol=0), message)


def main():
    protocol = read(RUN / 'protocol.json')
    frozen = read(RUN / 'frozen-stage1-selection.json')
    summary = read(RUN / 'confirmed-summary.json')
    status = read(RUN / 'status.json')
    fingerprint = digest(protocol)
    check(frozen['run_fingerprint'] == fingerprint, 'Frozen protocol mismatch')
    check(status['state'] == 'complete' and status['completed'] == 80, 'Incomplete run')
    check(not protocol['outer_test_evaluated'], 'Outer-test evaluation present')
    check(not protocol['source_models_or_outer_scores_loaded'], 'Unexpected source models/scores')
    for name, expected in protocol['source_sha256'].items():
        check(sha(ROOT / 'scripts' / name) == expected, 'Source changed: ' + name)
    source_protocol_path = ROOT / 'benchmarks/mattervial-quick-pilot-20261026/protocol.json'
    check(sha(source_protocol_path) == protocol['source_protocol_sha256'], 'Source protocol changed')
    source_protocol = read(source_protocol_path)
    for name, expected in source_protocol['source_sha256'].items():
        check(sha(ROOT / 'scripts' / name) == expected, 'Upstream source changed: ' + name)
    inputs = protocol['source_inputs']
    cached = {}
    universe = []
    for inner, inp in enumerate(inputs):
        check(sha(inp['source_manifest']) == inp['source_manifest_sha256'], 'Manifest changed')
        src = read(inp['source_manifest'])
        check(src == inp['source'], 'Embedded manifest differs')
        check(src['run_fingerprint'] == digest(source_protocol), 'Upstream provenance mismatch')
        check(src['train_ids'] == src['feature_selection_fit_ids'] == src['scaling_fit_ids']
              == inp['target_transform_fit_ids'], 'Preprocessing scope mismatch')
        check(not set(src['train_ids']) & set(src['validation_ids']), 'Overlapping partitions')
        check(len(src['selected_features']) == 256 and
              not any('SISSO' in x.upper() for x in src['selected_features']), 'Wrong features')
        for name, expected in src['artifacts'].items():
            check(sha(Path(src['directory']) / name) == expected, 'Input artifact changed')
        with np.load(Path(src['directory']) / 'arrays.npz') as data:
            cached[inner] = {k: data[k].copy() for k in data.files}
        train_y = cached[inner]['y_train'].astype('float64')
        close([train_y.mean(), train_y.std()], [inp['target_mean'], inp['target_std']], 'Target statistics')
        universe.extend(src['validation_ids'])
    check(len(universe) == len(set(universe)) == 1012, 'Wrong sample universe')
    for inp in inputs:
        src = inp['source']
        check(set(src['train_ids'] + src['validation_ids']) == set(universe), 'Wrong fold universe')

    records = []
    max_mae_difference = 0.
    max_checkpoint_difference = 0.
    frozen_time = datetime.fromisoformat(frozen['decided_at'])
    for path in sorted((RUN / 'fits').glob('*/*/result.json')):
        r = read(path)
        inp = inputs[r['inner']]
        src = inp['source']
        check(Path(r['directory']) == path.parent, 'Fit directory mismatch')
        job = dict(directory=r['directory'], variant=r['variant'], lr=r['lr'], inner=r['inner'],
                   repeat=r['repeat'], seed=r['seed'], input=inp, max_epochs=protocol['max_epochs'],
                   expected_parameters=protocol['model_parameters'][r['variant']], run_fingerprint=fingerprint)
        check(digest(job) == r['signature'] and r['run_fingerprint'] == fingerprint, 'Job signature')
        check(r['input_digest'] == digest(inp), 'Input identity')
        for key in ('selected_features', 'train_ids', 'validation_ids'):
            check(r[key] == src[key], 'Unpaired ' + key)
        check(r['target_transform_fit_ids'] == src['train_ids'], 'Target fit IDs')
        close([r['target_mean'], r['target_std']], [inp['target_mean'], inp['target_std']], 'Target transform')
        check(r['seed'] == protocol['seeds'][r['repeat']] + r['inner'], 'Unpaired seed')
        check(1 <= r['best_epoch'] <= r['epochs'] <= protocol['max_epochs'], 'Epoch range')
        hist = r['history']['val_mae']
        check(len(hist) == r['epochs'] and int(np.argmin(hist)) + 1 == r['best_epoch'], 'Checkpoint epoch')
        close(min(hist) * r['target_std'], r['checkpoint_metric_mae'], 'Checkpoint metric')
        for name, expected in r['artifacts'].items():
            check(sha(path.parent / name) == expected, 'Fit artifact changed')
        with np.load(path.parent / 'validation-predictions.npz') as saved:
            check(saved['sample_ids'].tolist() == src['validation_ids'], 'Prediction IDs')
            check(np.array_equal(saved['target'], cached[r['inner']]['y_validation']), 'Prediction targets')
            check(saved['prediction'].shape == saved['target'].shape, 'Prediction shape')
            check(np.isfinite(saved['prediction']).all(), 'Nonfinite predictions')
            mae = float(np.mean(abs(saved['target'] - saved['prediction'])))
        max_mae_difference = max(max_mae_difference, abs(mae-r['validation_mae']))
        close(mae, r['validation_mae'], 'Saved MAE')
        max_checkpoint_difference = max(max_checkpoint_difference, abs(mae-r['checkpoint_metric_mae']))
        check(abs(mae-r['checkpoint_metric_mae']) < .003, 'Restored checkpoint metric')
        # Only locally generated, hash-verified model artifacts are unpickled.
        with (path.parent / 'model.pkl').open('rb') as stream:
            model = pickle.load(stream)
        count = sum(w.size for w in model['weights'])
        check(count == r['parameters'] == protocol['model_parameters'][r['variant']], 'Serialized capacity')
        close([model['target_mean'], model['target_std']], [inp['target_mean'], inp['target_std']], 'Model scaling')
        fit_status = read(path.parent / 'status.json')
        check(fit_status['state'] == 'complete', 'Fit incomplete')
        finish = datetime.fromisoformat(fit_status['finished_at'])
        check((finish < frozen_time) if r['repeat'] == 0 else (finish > frozen_time), 'Stage chronology')
        records.append(r)

    configs = [(c['variant'], c['lr']) for c in frozen['confirmation_configs']]
    expected = {(v, lr, 0, i) for v in protocol['variants'] for lr in protocol['learning_rates'] for i in range(5)}
    expected |= {(v, lr, 1, i) for v, lr in configs for i in range(5)}
    keys = [(r['variant'], r['lr'], r['repeat'], r['inner']) for r in records]
    check(len(keys) == len(set(keys)) == 80 and set(keys) == expected, 'Training grid mismatch')
    for key, value in summary['selection'].items():
        check(frozen[key] == value, 'Selection changed after confirmation: ' + key)

    def rows(v, lr, repeat):
        return sorted([r for r in records if (r['variant'], r['lr'], r['repeat']) == (v, lr, repeat)],
                      key=lambda r: r['inner'])

    def scores(v, lr, repeat):
        return np.array([r['validation_mae'] for r in rows(v, lr, repeat)])

    for v, choice in frozen['options'].items():
        candidates = []
        for c in choice['candidates']:
            values = scores(v, c['lr'], 0)
            close(values, c['fold_mae'], 'Screen fold scores')
            close(values.mean(), c['mean_mae'], 'Screen mean')
            candidates.append(c)
        check(min(candidates, key=lambda c: (c['mean_mae'], c['lr'])) == choice['winner'], 'LR selection')
    for key, variants in [('best_kan', ('group_kan', 'latent_kan')), ('best_mlp', ('group_mlp', 'latent_mlp'))]:
        winner = min([frozen['options'][v]['winner'] for v in variants], key=lambda c: (c['mean_mae'], c['variant']))
        check(winner == frozen[key], 'Family selection')
    kan = frozen['best_kan']
    comparisons = {}
    for repeat in (0, 1):
        comparisons[repeat] = {}
        target = scores(kan['variant'], kan['lr'], repeat)
        for label, selection_key in [('modnet', 'baseline'), ('best_mlp', 'best_mlp'), ('matched_mlp', 'matched_mlp')]:
            c = frozen[selection_key]
            ref = scores(c['variant'], c['lr'], repeat)
            delta = target-ref
            improvement = float((ref.mean()-target.mean()) / ref.mean())
            wins = int((delta < 0).sum())
            calc = dict(delta_by_fold=delta.tolist(), mean_delta=float(delta.mean()),
                        relative_improvement=improvement, improved_folds=wins, passes=improvement >= .01 and wins >= 4)
            old = (frozen if repeat == 0 else summary['confirmation'])['comparisons'][label]
            for key in calc:
                close(calc[key], old[key], 'Comparison ' + key)
            comparisons[repeat][label] = calc
    check(all(c['passes'] for c in comparisons[0].values()) == frozen['confirm'], 'Trigger decision')
    replicated = all(c['passes'] for c in comparisons[1].values())
    check(replicated == summary['confirmation']['signal_replicated'], 'Confirmation decision')
    table = []
    for v, lr in configs:
        a, b = scores(v, lr, 0), scores(v, lr, 1)
        close(b, summary['confirmation']['scores'][v]['fold_mae'], 'Confirmation folds')
        close(b.mean(), summary['confirmation']['scores'][v]['mean_mae'], 'Confirmation mean')
        fits = rows(v, lr, 0) + rows(v, lr, 1)
        table.append(dict(variant=v, lr=lr, seed0_mean=float(a.mean()), seed1_mean=float(b.mean()),
                          seed0_folds=a.tolist(), seed1_folds=b.tolist(), parameters=fits[0]['parameters'],
                          mean_seconds_per_fit=float(np.mean([r['seconds'] for r in fits])),
                          mean_best_epoch=float(np.mean([r['best_epoch'] for r in fits]))))
    result = dict(verified_at=datetime.now(timezone.utc).isoformat(), run_fingerprint=fingerprint,
                  analyzer_sha256=sha(__file__), completed_fits=len(records), input_samples=1012,
                  all_source_input_artifact_hashes_pass=True, signatures_and_grid_pass=True,
                  serialized_parameter_counts_pass=True, train_only_target_scaling_pass=True,
                  paired_ids_features_seeds_pass=True, checkpoint_history_pass=True,
                  frozen_selection_unchanged=True, stage_finish_chronology_pass=True,
                  max_prediction_mae_difference=max_mae_difference, max_checkpoint_mae_difference=max_checkpoint_difference,
                  confirmation_signal_replicated=replicated, comparisons=comparisons, table=table,
                  wall_minutes=status['wall_seconds']/60, worker_minutes=sum(r['seconds'] for r in records)/60,
                  outer_test_evaluated=False,
                  limitation='Same inner folds; seed 0 selected architecture/LR; neither seed is independent held-out data. Hashes establish artifact consistency, not independent feature-pretraining provenance.')
    (RUN / 'independent-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    lines = ['# KAN 改进实验：结果与下一步决定', '',
             '已完成 60 次初筛和 20 次冻结配置换种子复核。低维隐藏表示残差 KAN 是本轮最值得保留的候选，但第二种子未复现其相对等参数 MLP 的优势。', '',
             'MAE 单位 cm⁻¹，越低越好。以下均使用无 SISSO 的相同增强输入、相同五个内层划分、固定学习率 0.001。第一种子参与选择结构和学习率；第二种子只更换初始化，仍不是独立测试数据。', '',
             '| 模型 | 第一种子：筛选 | 第二种子：固定配置 | 参数量 | 平均每次训练秒数 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for r in table:
        lines.append(f"| {LABELS[r['variant']]} | {r['seed0_mean']:.6f} | {r['seed1_mean']:.6f} | {r['parameters']:,} | {r['mean_seconds_per_fit']:.2f} |")
    lines += ['', '耗时按上述固定学习率下两个种子共 10 次训练平均；包含构建、训练和导出，两个 CPU worker 并发，不能视为隔离的速度测试。',
              f"整轮实际墙钟时间 {result['wall_minutes']:.2f} 分钟；累计 worker 时间 {result['worker_minutes']:.2f} 分钟。", '',
              '| KAN 对照 | 第一种子 MAE 降幅 / 胜出折数 | 第二种子 MAE 降幅 / 胜出折数 |',
              '| --- | ---: | ---: |']
    for key, label in [('modnet', '增强 MODNet'), ('best_mlp', '物理分组 MLP'), ('matched_mlp', '等参数残差 MLP')]:
        a, b = comparisons[0][key], comparisons[1][key]
        lines.append(f"| {label} | {100*a['relative_improvement']:.3f}% / {a['improved_folds']}/5 | {100*b['relative_improvement']:.3f}% / {b['improved_folds']}/5 |")
    lines += ['', '负降幅表示 KAN 误差更高。预设门槛为相对三个对照均改善至少 1%，且各胜出至少 4/5 折；第二种子三个比较均未达标。第一种子 KAN 对残差 MLP 的 4.63% 改善不能表述为稳定收益，也不应只合并两个种子的均值来掩盖此结果。', '',
              '第二种子逐折结果：', '',
              '| 内层折 | 增强 MODNet | 残差 KAN | 分组 MLP | 等参数残差 MLP |',
              '| --- | ---: | ---: | ---: | ---: |']
    for i in range(5):
        lines.append(f"| {i+1} | " + ' | '.join(f"{r['seed1_folds'][i]:.6f}" for r in table) + ' |')
    lines += ['', '## 已测试、可复现的候选', '',
              '保留 MODNet 96/72/36/36 主干。在最后 36 维隐藏表示上加一个分支：Dense 16 → 无仿射参数 LayerNorm → 0.5 × tanh → 16→1 三次样条 KAN → 零初始化标量门控，再加到原线性输出。总共只增加 738 个参数；对应 MLP 分支同样增加 738 个有效参数。所有支路和主干联合训练。', '',
              '目标只按内层训练集均值和标准差缩放。Adam 0.001、batch 32、最多 400 轮，内层验证早停 patience 40，并恢复验证 MAE 最低的权重。暂不加稀疏和平滑惩罚。该方案已经实跑，收益仍需复核。', '',
              '单独去掉物理分组 KAN 的稀疏和平滑项：第一种子均值从 35.569771 降到 35.022268，但仍高于增强 MODNet 的 34.731972。它说明这两项约束的合并消融有一定收益，不能支持物理分组 KAN 已优于基线。', '',
              '## 下一步建议（尚未启动）', '',
              '先固定上述候选及学习率，不增加架构搜索。用三个预先登记的新基础种子 20281027、20291027、20301027（每折加折编号），比较增强 MODNet、残差 KAN、等参数残差 MLP：3 模型 × 3 种子 × 5 内层折，共 45 次训练。独立保存为新实验，不覆盖这轮协议或结果。', '',
              '决定规则预先固定：新增三个种子中，KAN 相对两个对照的五折均值都必须更好；新增 15 个种子×折配对总体均值均至少降低 1%，且各至少 10/15 个配对改善。任一项失败，就不继续围绕这一结构搜索来宣称精度优势；这仍是探索性停止规则，不是统计显著性检验。', '',
              '三个新种子仍然不能提供独立泛化证据，也不能修复已有内层验证的反复使用。如果通过，再按冻结方案在各外层训练集内部重做特征筛选、缩放和 epoch 选择，等预算训练后一次性汇总外层结果。已经看过的外层成绩必须明确披露，不能称为全新盲测。', '',
              '目前最合理的结论是：存在值得复核的低维残差 KAN 信号，但尚无稳定优于等参数 MLP 的证据；若新增复核也失败，应保留 MatterVial + MODNet/MLP 作为精度方案。', '',
              '## 独立核验范围', '',
              f"已直接重算 80 份保存预测的 MAE，最大记录差为 {max_mae_difference:.3g}；检查全部模型权重数组参数量、最佳 epoch、输入与源码哈希、训练集目标统计、种子与样本配对以及第二阶段冻结配置。未运行新的训练，也未读取外层测试。", '',
              '这是对已保存产物的独立分析，不是从零重复训练。输入文件一致性不能证明 MatterVial 所用预训练模型不存在其他来源的数据重叠。', '',
              '本轮目标标准化、早停和最佳权重恢复规则都与上一轮 80 次快筛不同，只能与本轮同步重跑的对照比较。全部学习率候选和初筛逐折结果保存在 `confirmed-report.md`，审计细节保存在 `independent-verification.json`。']
    (RUN / 'decision-report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(verified_fits=len(records), max_mae_difference=max_mae_difference,
                         signal_replicated=replicated, wall_minutes=result['wall_minutes'], table=table), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
