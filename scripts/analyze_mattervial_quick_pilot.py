"""Independently verify 80 inner-only fits and report every LR/seed pairing."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


VARIANTS = ('matminer_modnet', 'mattervial_modnet', 'mattervial_mlp', 'mattervial_kan')
LABELS = ('A MatMiner MODNet', 'B MatterVial MODNet', 'C MatterVial MLP', 'D MatterVial KAN')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def differences(candidate, reference):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    delta = candidate-reference
    return dict(delta_by_fold=delta.tolist(), mean_delta=float(delta.mean()),
        improvement_percent=float(100*(reference.mean()-candidate.mean())/reference.mean()),
        improved_folds=int((delta < 0).sum()))


def analyze(root):
    protocol, state = read(root/'protocol.json'), read(root/'status.json')
    if state['state'] != 'complete' or state['completed'] != 80 or protocol['smoke']:
        raise ValueError('A completed, non-smoke 80-fit run is required.')
    fingerprint = digest(protocol)
    repo = Path(__file__).resolve().parents[1]
    for name, expected in protocol['source_sha256'].items():
        assert sha(repo/'scripts'/name) == expected, name
    audit = read(root/'input-audit.json')
    allowed = set(audit['training_ids'])
    assert len(allowed) == 1012 and audit['removed_sisso_count'] == 283
    assert sha(root/'training-only-no-sisso.pkl.gz') == audit['training_cache_sha256']
    inputs, arrays = {}, {}
    for inner in range(5):
        for bank in ('matminer', 'mattervial'):
            d = root/'inputs'/f'inner-{inner}'/bank
            info = read(d/'manifest.json')
            assert info['run_fingerprint'] == fingerprint
            assert len(info['selected_features']) == len(set(info['selected_features'])) == 256
            assert not any('SISSO' in c.upper() for c in info['selected_features'])
            assert info['train_ids'] == info['feature_selection_fit_ids'] == info['scaling_fit_ids']
            assert info['train_ids'] == protocol['splits'][inner]['train_ids']
            assert info['validation_ids'] == protocol['splits'][inner]['validation_ids']
            assert set(info['train_ids']).isdisjoint(info['validation_ids'])
            assert set(info['train_ids']+info['validation_ids']) == allowed
            for name, expected in info['artifacts'].items():
                assert sha(d/name) == expected
            with np.load(d/'arrays.npz', allow_pickle=False) as data:
                arrays[(inner,bank)] = {k:data[k] for k in data.files}
            inputs[(inner,bank)] = info
    records = [read(p) for p in (root/'fits').glob('*/*/result.json')]
    keys = [(r['variant'],r['inner'],r['lr'],r['repeat']) for r in records]
    expected = {(v,i,lr,s) for v in VARIANTS for i in range(5) for lr in protocol['learning_rates'] for s in range(2)}
    assert len(keys) == len(set(keys)) == 80 and set(keys) == expected
    for record in records:
        inner, variant = record['inner'], record['variant']
        bank = 'matminer' if variant == 'matminer_modnet' else 'mattervial'
        info, data = inputs[(inner,bank)], arrays[(inner,bank)]
        directory = Path(record['directory'])
        assert record['run_fingerprint'] == fingerprint
        assert record['input_manifest_digest'] == digest(info)
        assert record['train_ids'] == info['train_ids'] and record['validation_ids'] == info['validation_ids']
        assert record['selected_features'] == info['selected_features']
        assert record['seed'] == protocol['seeds'][record['repeat']]+inner
        assert record['epochs'] <= protocol['max_epochs'] == 300
        assert record['epochs'] == len(record['history']['mae'])
        for name, expected_hash in record['artifacts'].items():
            assert sha(directory/name) == expected_hash
        job = dict(directory=str(directory), variant=variant, inner=inner, lr=record['lr'], repeat=record['repeat'],
            seed=record['seed'], max_epochs=protocol['max_epochs'], run_fingerprint=fingerprint,
            manifest=str(Path(info['directory'])/'manifest.json'), manifest_digest=digest(info))
        assert digest(job) == record['signature']
        with np.load(directory/'validation-predictions.npz', allow_pickle=False) as prediction:
            assert prediction['sample_ids'].tolist() == info['validation_ids']
            np.testing.assert_array_equal(prediction['target'], data['y_validation'])
            np.testing.assert_allclose(np.mean(abs(prediction['prediction']-prediction['target'])),
                record['validation_mae'], rtol=0, atol=1e-12)
        expected_params = 35653 if variant in VARIANTS[:2] else 36185
        assert record['parameters'] == expected_params
        assert record['module_parameters'] == (0 if variant in VARIANTS[:2] else 148)
    for inner in range(5):
        for part in ('y_train','y_validation'):
            np.testing.assert_array_equal(arrays[(inner,'matminer')][part], arrays[(inner,'mattervial')][part])
    matrix = {(r['variant'],r['inner'],r['lr'],r['repeat']):r['validation_mae'] for r in records}
    combos = []
    for lr in protocol['learning_rates']:
        for repeat, seed in enumerate(protocol['seeds']):
            folds = {v:[matrix[(v,i,lr,repeat)] for i in range(5)] for v in VARIANTS}
            combos.append(dict(lr=lr, repeat=repeat, base_seed=seed,
                means={v:float(np.mean(folds[v])) for v in VARIANTS}, fold_mae=folds,
                mattervial_vs_matminer=differences(folds[VARIANTS[1]],folds[VARIANTS[0]]),
                kan_vs_mlp=differences(folds[VARIANTS[3]],folds[VARIANTS[2]]),
                kan_vs_modnet=differences(folds[VARIANTS[3]],folds[VARIANTS[1]])))
    summary = read(root/'stage-80-summary.json')
    for row in summary['results']:
        for candidate in row['candidates']:
            cs = [c for c in combos if c['lr'] == candidate['lr']]
            recomputed = np.mean([c['fold_mae'][row['variant']] for c in cs],axis=0)
            np.testing.assert_allclose(recomputed,candidate['fold_mae'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(recomputed.mean(),candidate['mean_mae'],atol=1e-12,rtol=0)
    costs = {v:dict(parameters=next(r['parameters'] for r in records if r['variant']==v),
        fit_seconds=sum(r['seconds'] for r in records if r['variant']==v),
        epoch_cap_hits=sum(r['hit_epoch_cap'] for r in records if r['variant']==v)) for v in VARIANTS}
    result = dict(verified_fits=80, run_fingerprint=fingerprint, complete_factorial_grid=True,
        predictions_recomputed=True, source_and_artifact_hashes_verified=True, job_signatures_verified=True,
        paired_ids_targets_inputs_and_seeds=True, parameter_matching_verified=True,
        train_only_selection_and_scaling=True, all_sisso_excluded=True, outer_test_evaluated=False,
        combinations=combos, costs=costs, total_wall_seconds=state['wall_seconds'],
        mattervial_improved_pairs=sum(c['mattervial_vs_matminer']['improved_folds'] for c in combos),
        kan_improved_pairs=sum(c['kan_vs_mlp']['improved_folds'] for c in combos),
        comparison_pairs=20, independent_replicates=False,
        note='Repeated folds reuse samples; paired counts are descriptive, not 20 independent experiments.')
    (root/'stage-80-independent-analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines = ['# MatterVial / KAN：80 次内层快筛复核', '',
        f"80/80 已完成，总墙钟时间 {state['wall_seconds']/60:.2f} 分钟。全部 283 个 SISSO 特征排除；未评估外层测试。",
        '四组共用同一组五个内层划分、配对种子、最大轮数及早停协议。B/C/D 使用完全相同的筛选输入和缩放结果。', '',
        '下表每个数字是五折 MAE 的等权平均，单位 cm^-1；不是两个种子的预测集成分数。', '',
        '| 学习率 | 基础种子 | A MatMiner MODNet | B 增强 MODNet | C 等参数 MLP | D 分组 KAN |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in combos:
        lines.append(f"| {c['lr']} | {c['base_seed']} | "+' | '.join(f"{c['means'][v]:.6f}" for v in VARIANTS)+' |')
    lines += ['', '| 学习率 | 基础种子 | MatterVial 改善折数 | MatterVial 平均降幅 | KAN 优于 MLP 折数 | KAN 平均降幅 |',
              '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in combos:
        mv,kan=c['mattervial_vs_matminer'],c['kan_vs_mlp']
        lines.append(f"| {c['lr']} | {c['base_seed']} | {mv['improved_folds']}/5 | {mv['improvement_percent']:.3f}% | "
                     f"{kan['improved_folds']}/5 | {kan['improvement_percent']:.3f}% |")
    lines += ['', '降幅为负代表变差。四种学习率/种子配置中，MatterVial 的平均 MAE 均低于 MatMiner 对照；KAN 均高于等参数 MLP。',
        f"逐折配对中，MatterVial 改善 {result['mattervial_improved_pairs']}/20 次，KAN 优于 MLP {result['kan_improved_pairs']}/20 次。这 20 个配对不是独立实验。", '',
        '四组通过全部内层折、两个种子的平均分均选择学习率 0.001。以下逐折成绩是两个种子的成员 MAE 平均：', '',
        '| 内层折 | A MatMiner MODNet | B 增强 MODNet | C 等参数 MLP | D 分组 KAN |',
        '| ---: | ---: | ---: | ---: | ---: |']
    selected = {r['variant']:r for r in summary['results']}
    for inner in range(5):
        lines.append(f'| {inner+1} | '+' | '.join(f"{selected[v]['fold_mae'][inner]:.6f}" for v in VARIANTS)+' |')
    lines += ['', '| 模型 | 参数量 | 20 次训练累计分钟 | 跑满 300 轮次数 |', '| --- | ---: | ---: | ---: |']
    for variant,label in zip(VARIANTS,LABELS):
        cost=costs[variant]
        lines.append(f"| {label} | {cost['parameters']} | {cost['fit_seconds']/60:.3f} | {cost['epoch_cap_hits']}/20 |")
    lines += ['', '训练累计时间是各 worker 耗时相加，不能与整轮墙钟时间混淆。', '',
        '复核了 80 个任务的完整组合、任务签名、源码与模型/预测/输入文件哈希、预测 MAE、配对样本和目标、输入特征、种子以及参数量。',
        '所有特征筛选及缩放的拟合 ID 均为各自内层训练集；只扫描源文件中的非训练行，不转换或保存其目标值。', '',
        '限制：78/80 次训练达到 300 轮上限，因此结论只适用于当前小预算，不能证明已充分收敛。两种子不是广泛随机重复；'
        '内层数据也参与了学习率选择及此前探索，不能当作独立外层测试，不能直接与论文 30.08 比较。'
        '排除 SISSO 不等于已经核实所有冻结外部预训练特征的标签来源。']
    (root/'stage-80-stability-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified_fits=80,combinations=combos,costs=costs,
        mattervial_improved_pairs=result['mattervial_improved_pairs'],kan_improved_pairs=result['kan_improved_pairs'],
        report=str(root/'stage-80-stability-report.md')),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/mattervial-quick-pilot-20261026'))
    analyze(parser.parse_args().output.resolve())
