"""Descriptive audit of saved outer predictions; never selects a new model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def balanced_order(rng, members=50, groups=10):
    if members != 50 or groups != 10:
        raise ValueError("Expected ten configurations with five inner members each.")
    within = [rng.permutation(5) + 5 * g for g in range(groups)]
    return np.array([within[g][round_] for round_ in range(5) for g in rng.permutation(groups)])


def analyze(round_one, round_two_steels, output, repeats=100):
    output.mkdir(parents=True, exist_ok=True)
    cohorts = [("round1", task, round_one) for task in
               ("matbench_steels", "matbench_phonons", "matbench_dielectric")]
    cohorts.append(("round2", "matbench_steels", round_two_steels))
    hashes, metrics, pairs, examples, curves, bins, mixtures = {}, [], [], [], [], [], []
    for round_name, task, root in cohorts:
        fold_data = []
        for fold in range(5):
            directory = root / task / f"fold-{fold}"
            frame_path = directory / "predictions.csv"
            frame = pd.read_csv(frame_path)
            hashes[str(frame_path.resolve())] = hashlib.sha256(frame_path.read_bytes()).hexdigest()
            names = [n for n in frame.columns if n not in ("sample_id", "target") and not n.endswith("_std")]
            stored = {}
            for name in names:
                path = directory / f"{name}-member-predictions.npz"
                hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
                arrays = np.load(path)
                pred = arrays["predictions"]
                assert pred.shape == (50, len(frame)) and np.isfinite(pred).all()
                np.testing.assert_allclose(pred.mean(axis=0), frame[name], atol=1e-7, rtol=1e-7)
                stored[name] = pred
                y = frame.target.values
                err = pred.mean(axis=0) - y
                raw_err = arrays["raw_predictions"].mean(axis=0) - y
                worst = np.sort(np.abs(err))[::-1][:max(1, int(np.ceil(.1 * len(err))))]
                member_corr = np.corrcoef(pred - y[None, :])
                metrics.append(dict(round=round_name, task=task, fold=fold, variant=name,
                    mae=float(np.abs(err).mean()), rmse=float(np.sqrt(np.mean(err ** 2))), bias=float(err.mean()),
                    raw_mae=float(np.abs(raw_err).mean()), repaired_predictions=int(arrays["replaced_counts"].sum()),
                    worst_decile_absolute_error_share=float(worst.sum() / np.abs(err).sum()),
                    member_residual_correlation=float(member_corr[np.triu_indices(50, 1)].mean())))
                if name != "modnet":
                    base_err = frame.modnet.values - y
                    blend_err = .5 * (frame.modnet.values + frame[name].values) - y
                    pairs.append(dict(round=round_name, task=task, fold=fold, variant=name,
                        residual_correlation=float(np.corrcoef(err, base_err)[0, 1]),
                        absolute_error_correlation=float(np.corrcoef(np.abs(err), np.abs(base_err))[0, 1]),
                        sample_win_fraction=float((np.abs(err) < np.abs(base_err)).mean()),
                        fixed_half_blend_mae=float(np.abs(blend_err).mean()),
                        blend_delta_vs_modnet=float(np.abs(blend_err).mean() - np.abs(base_err).mean()),
                        mae_delta_vs_modnet=float(np.abs(err).mean() - np.abs(base_err).mean())))
            copy = frame.copy()
            copy["round"], copy["task"], copy["fold"] = round_name, task, fold
            for name in names:
                copy[name + "_abs_error"] = abs(copy[name] - copy.target)
            examples.append(copy)
            fold_data.append((frame, stored))
        combined = pd.concat([item[0] for item in fold_data], ignore_index=True)
        assert not combined.sample_id.duplicated().any()
        quartile = pd.qcut(combined.target, 4, labels=False, duplicates="drop")
        for group in sorted(quartile.unique()):
            part = combined[quartile == group]
            for name in names:
                bins.append(dict(round=round_name, task=task, target_quartile=int(group + 1),
                    variant=name, count=len(part), target_min=float(part.target.min()),
                    target_max=float(part.target.max()), mae=float(abs(part[name] - part.target).mean())))
        for repeat in range(repeats):
            rng = np.random.default_rng(20261023 + repeat)
            orders = [balanced_order(rng) for _ in range(5)]
            for name in names:
                if name == "modnet":
                    continue
                values = [np.abs(.5 * (pred["modnet"][order[:25]].mean(axis=0) +
                                        pred[name][order[:25]].mean(axis=0)) - frame.target.values).mean()
                          for (frame, pred), order in zip(fold_data, orders)]
                mixtures.append(dict(round=round_name, task=task, variant=name, total_members=50,
                                     members_per_variant=25, repeat=repeat, mae=float(np.mean(values))))
            for size in (1, 5, 10, 25, 50):
                if size == 50 and repeat:
                    continue
                for name in names:
                    values = [np.abs(pred[name][order[:size]].mean(axis=0) - frame.target.values).mean()
                              for (frame, pred), order in zip(fold_data, orders)]
                    curves.append(dict(round=round_name, task=task, variant=name, members=size,
                                       repeat=repeat, mae=float(np.mean(values))))
    tables = {"fold-errors": pd.DataFrame(metrics), "fold-complementarity": pd.DataFrame(pairs),
              "sample-errors": pd.concat(examples, ignore_index=True), "target-quartiles": pd.DataFrame(bins),
              "ensemble-repeats": pd.DataFrame(curves), "equal-member-mixture-repeats": pd.DataFrame(mixtures)}
    for name, table in tables.items():
        table.to_csv(output / (name + ".csv"), index=False)
    summary = tables["fold-errors"].groupby(["round", "task", "variant"]).mean(numeric_only=True).drop(columns="fold")
    pair_summary = tables["fold-complementarity"].groupby(["round", "task", "variant"]).mean(numeric_only=True).drop(columns="fold")
    curve_summary = tables["ensemble-repeats"].groupby(["round", "task", "variant", "members"]).mae.agg(["mean", "std", "min", "max"])
    summary.to_csv(output / "error-summary.csv")
    pair_summary.to_csv(output / "complementarity-summary.csv")
    curve_summary.to_csv(output / "ensemble-summary.csv")
    mixture_summary = tables["equal-member-mixture-repeats"].groupby(["round", "task", "variant"]).mae.agg(["mean", "std", "min", "max"])
    mixture_summary.to_csv(output / "equal-member-mixture-summary.csv")
    audit = dict(descriptive_outer_test_analysis_only=True, no_blend_weight_or_architecture_selected=True,
                 fixed_blend_weight=.5, full_blend_total_members=100, equal_member_control_total_members=50,
                 mixture_compute_or_parameter_matched=False, ensemble_repetitions=repeats, complete_cohorts=len(cohorts),
                 verified_prediction_rows=len(metrics), input_sha256=hashes,
                 subset_rule="Round-robin over randomized configurations; randomized inner members, paired across variants.",
                 uncertainty_scope="Subset variation is not uncertainty across independently trained experiments.")
    (output / "analysis-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# 已有外层预测的误差与互补性诊断", "",
        "本分析仅描述已看过的外层测试结果。50:50 融合权重预先固定，没有扫描权重或据此选择新模型。"
        "任何后续融合或门控选择必须重新使用外层训练集内的折外预测。", "",
        "下表完整融合使用 50 个 MODNet 成员加 50 个另一模型成员，共 100 个；单模型只有 50 个成员。"
        "后面另列 25+25 成员对照。成员数相等仍不代表参数量、训练时间相等，尤其此前全 KAN 单成员参数量约为 MODNet 四倍。", "",
        "| 轮次 | 数据集 | 模型 | MAE | 残差相关性 vs MODNet | 50:50 融合 MAE | 融合 MAE 变化 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for key, row in summary.iterrows():
        pair = pair_summary.loc[key] if key in pair_summary.index else None
        extra = "— | — | —" if pair is None else f"{pair.residual_correlation:.4f} | {pair.fixed_half_blend_mae:.6f} | {pair.blend_delta_vs_modnet:+.6f}"
        lines.append(f"| {' / '.join(key).replace(' / ', ' | ')} | {row.mae:.6f} | {extra} |")
    lines += ["", "## 总成员数相同的融合诊断", "",
        "固定等权融合，各取 25 个成员，共 50 个。按同一组配置/内层成员索引配对抽样 100 次；"
        "标准差描述成员子集敏感性，不是泛化置信区间。", "",
        "| 轮次 | 数据集 | 混合模型 | 25+25 MAE 均值 | 子集标准差 | 相比 50 成员 MODNet |",
        "| --- | --- | --- | ---: | ---: | ---: |"]
    for key, row in mixture_summary.iterrows():
        baseline = summary.loc[(key[0], key[1], "modnet"), "mae"]
        lines.append(f"| {' | '.join(key)} | {row['mean']:.6f} | {row['std']:.6f} | {row['mean']-baseline:+.6f} |")
    lines += ["", "## 误差集中与集成规模", "",
        "worst_decile_absolute_error_share 是每折误差最大的约 10% 样本占总绝对误差的比例。"
        "目标四分位仅用于描述困难样本，不能据此删除或重加权测试样本。", "",
        "1/5/10/25/50 成员曲线在十组配置之间平衡抽样，避免只取保存顺序前 N 个导致偏差。"
        "每个规模最多 100 次配对子集抽样；这些重复不是独立训练种子。", "",
        "![Ensemble size](ensemble-curves.png)", "",
        "详细逐折、逐样本、目标区间及原始/后处理误差见同目录 CSV。"]
    (output / "analysis-report.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (round_name, task, _) in zip(axes.ravel(), cohorts):
        part = curve_summary.loc[(round_name, task)]
        for name in part.index.get_level_values(0).unique():
            row = part.loc[name]
            ax.plot(row.index, row["mean"], marker="o", label=name)
            ax.fill_between(row.index, row["mean"] - row["std"].fillna(0), row["mean"] + row["std"].fillna(0), alpha=.12)
        ax.set(title=round_name + " / " + task.removeprefix("matbench_"), xlabel="Ensemble members", ylabel="Outer MAE")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(output / "ensemble-curves.png", dpi=160)
    plt.close(fig)
    print(summary.to_string())
    print(pair_summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-one", type=Path, required=True)
    parser.add_argument("--round-two-steels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.round_one, args.round_two_steels, args.output)
