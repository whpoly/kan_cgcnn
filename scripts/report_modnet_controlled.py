"""Verify complete controlled runs against Matbench and write a report/figure."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_modnet_controlled import summarize

OFFICIAL_REFERENCE_URL = (
    "https://github.com/materialsproject/matbench/blob/main/docs_src/"
    "Full%20Benchmark%20Data/matbench_v0.1_modnet_v0.1.12.md"
)
# Published mean fold MAE, read from the official Matbench v0.1.12 entry.
# These are reference numbers, NOT a locally reproduced baseline.
OFFICIAL_REFERENCE_MAE = {
    "matbench_steels": 87.7627,
    "matbench_phonons": 34.2751,
    "matbench_dielectric": 0.2711,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def merge_shards(out, shards, manifest):
    """Merge disjoint executions of the same predeclared experiment."""
    excluded = {"datasets", "folds", "output_dir"}
    reference_args = {k: v for k, v in manifest["args"].items() if k not in excluded}
    provenance = []
    for source in shards:
        source = Path(source).resolve()
        shard_manifest = read_json(source / "manifest.json")
        shard_args = {k: v for k, v in shard_manifest["args"].items() if k not in excluded}
        if shard_args != reference_args:
            raise ValueError(f"Shard training settings differ: {source}")
        for key in ("source_sha256", "packages", "architectures", "optimizer", "scaling"):
            if shard_manifest[key] != manifest[key]:
                raise ValueError(f"Shard {key} differs: {source}")
        if not set(shard_manifest["args"]["datasets"]) <= set(manifest["args"]["datasets"]):
            raise ValueError("Shard contains unexpected datasets.")
        if not set(shard_manifest["args"]["folds"]) <= set(manifest["args"]["folds"]):
            raise ValueError("Shard contains unexpected folds.")
        # Status may remain "running" when an execution was stopped deliberately
        # at a completed fold boundary. Individual atomically-written result
        # files and subsequent full verification establish completion.
        for path in source.glob("matbench_*"):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            destination = out / path.name
            if destination.exists():
                if path.suffix == ".json" and read_json(destination) != read_json(path):
                    raise ValueError(f"Conflicting shard result: {path.name}")
            else:
                shutil.copyfile(path, destination)
        provenance.append({"directory": str(source), "manifest": shard_manifest,
                           "original_status": read_json(source / "status.json")})
    (out / "execution-shards.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")


def verify(out, manifest):
    from matbench.task import MatbenchTask

    settings = manifest["args"]
    if sorted(settings["folds"]) != list(range(5)):
        raise ValueError("The final report requires all five official folds.")
    rows = []
    for dataset in settings["datasets"]:
        task = MatbenchTask(dataset)
        all_test_ids = []
        for fold in settings["folds"]:
            x_train, _ = task.get_train_and_val_data(fold)
            x_test, y_test = task.get_test_data(fold, include_target=True)
            split = read_json(out / f"{dataset}-fold{fold}-split.json")
            training, validation, testing = (
                set(split[key]) for key in ("train_ids", "validation_ids", "test_ids")
            )
            assert training.isdisjoint(validation)
            assert (training | validation).isdisjoint(testing)
            assert training | validation == set(x_train.index)
            assert split["test_ids"] == x_test.index.tolist()
            all_test_ids.extend(split["test_ids"])
            for seed in settings["seeds"]:
                for variant in settings["variants"]:
                    path = out / f"{dataset}-f{fold}-s{seed}-{variant}.json"
                    payload = read_json(path)
                    row = payload["row"]
                    assert row["seed"] == seed and row["fold"] == fold
                    assert row["variant"] == variant and row["dataset"] == dataset
                    assert payload["test_ids"] == x_test.index.tolist()
                    prediction = np.asarray(payload["prediction"])
                    target = np.asarray(payload["target"])
                    assert prediction.shape == target.shape == (len(x_test),)
                    assert np.isfinite(prediction).all()
                    np.testing.assert_allclose(
                        target, y_test.to_numpy(), rtol=1e-6, atol=1e-6)
                    recomputed = np.abs(prediction - target).mean()
                    np.testing.assert_allclose(
                        recomputed, row["test_mae"], rtol=2e-6, atol=1e-5)
                    rows.append(row)
        assert len(all_test_ids) == len(set(all_test_ids)) == len(task.df)
        assert set(all_test_ids) == set(task.df.index)
    return rows


def make_figure(summary, out, datasets, variants):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "mlp": "MLP",
        "hybrid-fastkan": "MLP + KAN head",
        "fastkan": "Compact FastKAN",
        "fastkan-no-layernorm": "FastKAN, no LayerNorm",
        "spline": "Compact spline KAN",
    }
    colors = ["#46556b", "#1b9e77", "#1976d2", "#e69523", "#9b59b6"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(13, 4.8), layout="constrained")
    for ax, dataset in zip(np.atleast_1d(axes), datasets):
        part = summary[summary.dataset == dataset].set_index("variant").loc[variants]
        x = np.arange(len(variants))
        ax.bar(x, part.mae_mean, color=colors, alpha=0.9, width=0.7)
        ax.errorbar(x, part.mae_mean, yerr=part.mae_fold_std,
                    fmt="none", ecolor="#333333", capsize=3, linewidth=1.2)
        ax.set_title(dataset.replace("matbench_", ""), fontsize=13)
        ax.set_xticks(x, [labels[v] for v in variants], rotation=38, ha="right", fontsize=9)
        ax.set_ylabel("Test MAE (native target units)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("KAN / MODNet-style architecture audit | lower is better\n"
                 "5 official folds x 3 seeds; bars: seed-averaged fold mean, "
                 "whiskers: fold SD", fontsize=13)
    fig.savefig(out / "comparison.png", dpi=190, bbox_inches="tight")
    fig.savefig(out / "comparison.svg", bbox_inches="tight")
    plt.close(fig)


def write_report(summary, rows, manifest, destination, out):
    labels = {"mlp": "MLP", "hybrid-fastkan": "MLP + FastKAN 输出头",
              "fastkan": "紧凑 FastKAN", "fastkan-no-layernorm": "FastKAN 无 LayerNorm",
              "spline": "紧凑 B-spline KAN"}
    lines = [
        "# KAN + MODNet 架构诊断：2026-10-20",
        "",
        "本报告记录实际完成的 225 次训练：3 个任务 × 官方 5 折 × "
        "3 个初始化种子 × 5 种模型。所有测试集完整使用，没有抽样。",
        "",
        "**范围：这是轻量描述符上的受控架构实验，不是官方 MODNet 复现，"
        "也不是完整的嵌套超参数搜索。** 不应把下面的 MLP 称为官方 MODNet 分数。",
        "",
        "## 与公开 MODNet benchmark 是否对齐",
        "",
        "**尚未对齐，且没有在本轮重新训练原版 MODNet。** "
        "以下公开参考值来自 Matbench 官方仓库的 MODNet v0.1.12 条目，"
        "不是本机测得的结果。特征、调参和训练协议不同，"
        "这张表用于检查差距，不能作为隔离 KAN 效果的公平比较。",
        "",
        "| 任务 | 公开 MODNet v0.1.12 MAE | 本轮 PyTorch MLP MAE | 本轮紧凑 FastKAN MAE |",
        "|---|---:|---:|---:|",
    ]
    reference_rows = []
    for dataset in manifest["args"]["datasets"]:
        mlp = summary[(summary.dataset == dataset) & (summary.variant == "mlp")].iloc[0]
        kan = summary[(summary.dataset == dataset) & (summary.variant == "fastkan")].iloc[0]
        reference = OFFICIAL_REFERENCE_MAE[dataset]
        lines.append(
            f"| {dataset.removeprefix('matbench_')} | {reference:.4f} | "
            f"{mlp.mae_mean:.4f} | {kan.mae_mean:.4f} |")
        reference_rows.append({
            "dataset": dataset, "published_modnet_v012_mae": reference,
            "local_mlp_mae": mlp.mae_mean, "local_fastkan_mae": kan.mae_mean,
            "reference_source": OFFICIAL_REFERENCE_URL,
            "local_official_modnet_reproduction": False,
        })
    pd.DataFrame(reference_rows).to_csv(out / "official-reference-comparison.csv", index=False)
    lines += [
        "",
        f"来源：[Matbench 官方 MODNet v0.1.12 分数]({OFFICIAL_REFERENCE_URL})。"
        "公开均值按页面保留四位小数。"
        "本轮 FastKAN 在 steels/phonons 上比本地 MLP 好，但三个任务均未达到"
        "该公开 MODNet 参考值；这不能分辨描述符、调参、集成和架构各自的影响。",
        "",
        "## 协议",
        "",
        "- 任务：steels（312 条）、phonons（1265 条）、dielectric（4764 条）。",
        "- 每个外层训练折留出 20% 验证集；训练/验证划分种子为 1729 + fold。"
        "不同初始化种子共享划分与描述符列表。",
        "- 在剩余训练子集上独立完成 MI/Pearson 近似相关性-冗余筛选、"
        "中位数填补、输入 minmax[-0.5,0.5] 和目标标准化。"
        "测试和验证标签均不参与这些拟合。",
        "- 全部模型使用相同 64 个特征。steel 使用 matminer-composition，"
        "结构任务使用 matminer-structure-lite；没有静默回退到其它特征集。",
        "- 固定架构：MLP/Hybrid 隐藏宽度 64→32→16，FastKAN 16→8→4，"
        "B-spline KAN 10→6→4；KAN grid=3、spline order=3。",
        "- 所有模型采用 Adam，lr=0.003，batch=64，MAE，最多 400 epoch，"
        "验证集早停 patience=60，恢复最低验证 MAE 的权重；初始化种子为 7/19/43。",
        "- 不做剪枝、符号回归、测试集调参或根据测试结果追加配置。"
        "不在全外层训练集上重训；这是有保留验证集的固定架构诊断。",
        "- 每个训练进程使用一个 CPU 线程；独立任务/折并行执行。"
        "时间是在当时并发负载下测得的耗时，不能当作严格的硬件速度基准或推断 GPU 加速效果。",
        "- 先在每折平均三个种子的 MAE，再报告五折均值与样本标准差。"
        "标准差不是置信区间；15 次训练不能当作 15 个独立测试集。",
        "",
        "## 实测结果",
        "",
        "| 任务 | 模型 | MAE 均值 ± 折间 SD | 相对 MLP 的 MAE 变化 | 胜出折数 | 参数 | 平均训练秒数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in manifest["args"]["datasets"]:
        for variant in manifest["args"]["variants"]:
            r = summary[(summary.dataset == dataset) & (summary.variant == variant)].iloc[0]
            lines.append(
                f"| {dataset.removeprefix('matbench_')} | {labels[variant]} | "
                f"{r.mae_mean:.4f} ± {r.mae_fold_std:.4f} | {r.mae_change_pct:+.2f}% | "
                f"{int(r.wins_vs_mlp)}/5 | {int(r.params):,} | {r.train_seconds:.2f} |")
    lines += [
        "",
        "MAE 变化为负表示改善；steels 的原始目标单位为 MPa，phonons 为 cm⁻¹，"
        "dielectric 为折射率（无量纲）。这些任务数值不可直接横向平均。",
        "",
        "## 对已有实现的核查",
        "",
        "1. **FastKAN 的 LayerNorm 开关失效（已修复）。** "
        "`make_kan_mlp` 没有把 `use_layernorm` 传给 FastKANMLP，"
        "后者又使用 FastKANLinear 的默认归一化。旧的关闭 LayerNorm 实验需要重跑。"
        "保留 LayerNorm 时，各边函数依赖归一化后的多个输入，不能直接宣称"
        "是原始描述符上的独立单变量函数。",
        "2. **验证早停忽略 min_delta（已修复）。** "
        "原实现只有 training-loss 分支使用阈值；现在 validation 分支也按"
        "相对监控最优值的累计改善重置 patience，并仍保存真正最佳的验证权重。",
        "3. **预计算特征有内层选择泄漏（已标注，尚未实现官方逐内折重排）。** "
        "官方导出先用整个 outer-train 标签排名，再把这个顺序用于 inner validation。"
        "这不等于测试集泄漏，但内层验证标签已进入特征筛选，"
        "因此不能叫严格嵌套的特征选择。代码增加警告，调参输出记录筛选范围；"
        "本次实验直接在 inner-train 重新拟合本地近似筛选器，避开该路径。",
        "4. **已有结果规模不足。** 检查到的 25 个旧结果 JSON 中，"
        "绝大多数是 1–2 epoch 的 smoke test。此前 dielectric 对照只用 fold 0/1/2，"
        "train_size=1200、test_size=300、120 epoch，不是完整官方五折评测。",
        "5. **本地筛选不是官方 MODNet NMI-RR。** 本地实现用未归一化 MI"
        "表示相关性、Pearson 相关表示冗余；官方实现用归一化互信息。"
        "只有官方完整描述符、官方筛选及原版模型的运行才能称为官方复现。",
        "6. **服务器入口不一致（已修复）。** 两个 Slurm 脚本传入 80/300 epoch，"
        "与调参器的 1000-epoch 模式检查冲突，会在训练前失败。"
        "已改为 1000/1000。1000 是该复现模式的设置，并非 Matbench 的普遍要求。"
        "全任务列表还漏了 `matbench_expt_is_metal`，现已补齐，数组扩展至 12 项。",
        "",
        "## 建议的 KAN + MODNet 路线",
        "",
        "1. 先固定材料输入、官方描述符与逐内折 NMI-RR 排名。"
        "以同特征 MLP 为基线，分别比较“只替换输出头”和“缩窄的全 KAN”。"
        "架构消融与减少特征数量的消融分开做。",
        "2. 正式比较保留原版 TensorFlow MODNet 一列，同时增加同训练代码的"
        "PyTorch MLP，区分框架/训练协议差异和 KAN 层本身的收益。"
        "单模型对单模型；集成模型单独报告，不能拿单个 KAN 对 MODNet 集成作等价比较。",
        "3. 每个 outer fold 独立做 inner CV：缺失值处理、特征筛选和缩放都在"
        "inner-train 拟合；MLP、Hybrid、FastKAN、spline 分配相同搜索预算。"
        "选择架构/超参后在整个 outer-train 重训，再评估完整 outer-test。",
        "4. KAN 从浅层、小 grid 开始。更少参数不意味着更少运算；同时报告"
        "参数量、训练与推理耗时、种子波动及逐折结果。"
        "本轮固定预算结果只能判断这些配置，不能证明一个模型族普遍更强。",
        "5. 预测性能可靠之后再独立做剪枝、公式提取和保真度评测。"
        "若需要原始特征上的可解释单变量边函数，应使用真正无 LayerNorm 的模型；"
        "不要把关闭归一化当作不会改变性能的导出选项。",
        "",
        "## 产物与复现",
        "",
        "运行命令：",
        "",
        "```powershell",
        "python scripts/benchmark_modnet_controlled.py --output-dir benchmarks/controlled-audit-20261020",
        "python scripts/report_modnet_controlled.py --input-dir benchmarks/controlled-audit-20261020",
        "```",
        "",
        f"原始目录：`{out.as_posix()}`。`manifest.json` 记录源代码 SHA256、"
        "依赖版本与预先固定的设置；`*-split.json` 保存样本 ID/描述符/目标缩放；"
        "每个训练 JSON 含原始测试目标与预测；`summary.csv`、"
        "`paired-fold-deltas.csv` 和 `comparison.png/.svg` 给出汇总。"
        "`verification.json` 记录逐样本核对及 MAE 重算结果。",
        "",
        "资料：[MODNet 官方实现](https://github.com/ppdebreuck/modnet)、"
        "[MODNet 原始 Matbench 基准仓库](https://github.com/modl-uclouvain/modnet-matbench)、"
        "[Matbench 官方实现](https://github.com/materialsproject/matbench)。",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--report", default="docs/kan_modnet_audit_20261020.md")
    parser.add_argument("--shards", nargs="*", default=[])
    args = parser.parse_args()
    out = Path(args.input_dir).resolve()
    manifest = read_json(out / "manifest.json")
    if args.shards:
        merge_shards(out, args.shards, manifest)
    rows = verify(out, manifest)
    summary = summarize(rows, out)
    make_figure(summary, out, manifest["args"]["datasets"], manifest["args"]["variants"])
    write_report(summary, rows, manifest, Path(args.report), out)
    verification = {"verified_runs": len(rows),
                    "official_test_ids_match": True, "official_targets_match": True,
                    "train_validation_test_disjoint": True,
                    "each_sample_tested_once_per_seed_model": True,
                    "all_predictions_finite": True, "all_mae_recomputed": True}
    (out / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    (out / "status.json").write_text(
        json.dumps({"state": "complete", "completed_fits": len(rows),
                    "verified": True}, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
