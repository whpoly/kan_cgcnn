"""Report completed matched MODNet/KAN runs without changing training artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

TASKS = ("matbench_steels", "matbench_phonons", "matbench_dielectric")
VARIANTS = ("modnet", "kan_head", "kan_full")
LABELS = dict(modnet="MODNet", kan_head="输出头 KAN", kan_full="全 KAN")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f"Inconsistent recorded statistics: {a} != {b}")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(root):
    verification = read(root / "matched-verification.json")
    status = read(root / "matched-status.json")
    if status["state"] != "complete" or not verification["complete_full_benchmark"]:
        raise ValueError("The final report requires a completed, verified full benchmark.")
    if verification["smoke"] or verification["verified_completed_rows"] != 45:
        raise ValueError("Expected 45 verified production rows.")
    output = root / "final-report"
    output.mkdir(exist_ok=True)
    alignment = read(root / "baseline-alignment.json")
    fold_rows, search_rows, hashes = [], [], {}
    for task in TASKS:
        for fold in range(5):
            directory = root / task / f"fold-{fold}"
            comparison_path = directory / "comparison.json"
            hashes[str(comparison_path.relative_to(root))] = hashlib.sha256(
                comparison_path.read_bytes()).hexdigest()
            result = read(comparison_path)
            selected = read(directory / "frozen-ensemble.json")["members"]
            names = [Path(member["directory"]).name for member in selected]
            if len(names) != 50 or len(set(names)) != 50:
                raise ValueError("Expected 50 distinct ensemble members.")
            baseline = result["results"]["modnet"]
            for variant in VARIANTS:
                records = [read(directory / variant / name / "result.json") for name in names]
                parameters = [record["parameters"] for record in records]
                seconds = [record["seconds"] for record in records]
                if not all(math.isfinite(s) and s > 0 for s in seconds):
                    raise ValueError("Invalid member duration.")
                stats = result["results"][variant]
                close(sum(parameters), stats["parameters_sum"])
                close(statistics.mean(parameters), stats["parameters_mean"])
                close(sum(seconds), stats["training_seconds_sum"])
                fold_rows.append(dict(
                    task=task, fold=fold, variant=variant, mae=stats["mae"],
                    official_fold_mae=result["published_modnet_mae"],
                    delta_vs_retrained_pct=(stats["mae"] / baseline["mae"] - 1) * 100,
                    members=50, member_parameters_mean=statistics.mean(parameters),
                    ensemble_parameters=sum(parameters), member_seconds_sum=sum(seconds),
                    epochs_mean=statistics.mean(r["epochs"] for r in records),
                    postprocess_replacements=stats["postprocess_replaced_member_predictions"]))
            # Tiny status files avoid reparsing all the GA training histories.
            ga_statuses = list((directory / "modnet").glob("*/status.json"))
            ga_records = [read(path) for path in ga_statuses]
            if not all(record["state"] == "complete" for record in ga_records):
                raise ValueError("Incomplete GA worker status.")
            if any(not (path.parent / "result.json").exists() for path in ga_statuses):
                raise ValueError("Missing GA result.")
            search_rows.append(dict(task=task, fold=fold,
                                    generations=len(list(directory.glob("generation-*.json"))),
                                    ga_fits=len(ga_records),
                                    ga_member_seconds_sum=sum(r["seconds"] for r in ga_records)))
        print(f"Audited parameter and timing records: {task}", flush=True)

    summary = []
    for task in TASKS:
        baseline_rows = [r for r in fold_rows if r["task"] == task and r["variant"] == "modnet"]
        base_mae = statistics.mean(r["mae"] for r in baseline_rows)
        base_params = statistics.mean(r["member_parameters_mean"] for r in baseline_rows)
        base_seconds = sum(r["member_seconds_sum"] for r in baseline_rows)
        close(base_mae, alignment["tasks"][task]["retrained_mae"])
        for variant in VARIANTS:
            rows = [r for r in fold_rows if r["task"] == task and r["variant"] == variant]
            mae = statistics.mean(r["mae"] for r in rows)
            seconds = sum(r["member_seconds_sum"] for r in rows)
            params = statistics.mean(r["member_parameters_mean"] for r in rows)
            summary.append(dict(task=task, variant=variant, folds=5, total_members=250,
                                mae_mean=mae, mae_fold_std=statistics.stdev(r["mae"] for r in rows),
                                official_mae=alignment["tasks"][task]["published_mae"],
                                delta_vs_retrained_pct=(mae / base_mae - 1) * 100,
                                improved_folds=sum(r["mae"] < b["mae"] for r, b in zip(rows, baseline_rows)),
                                member_parameters_mean=params,
                                ensemble_parameters_mean=statistics.mean(r["ensemble_parameters"] for r in rows),
                                parameter_ratio=params / base_params,
                                selected_member_seconds_sum=seconds,
                                selected_member_minutes_sum=seconds / 60,
                                selected_member_seconds_mean=seconds / 250,
                                selected_member_time_ratio=seconds / base_seconds,
                                epochs_mean=statistics.mean(r["epochs_mean"] for r in rows),
                                postprocess_replacements=sum(r["postprocess_replacements"] for r in rows)))
    write_csv(output / "summary.csv", summary)
    write_csv(output / "fold-results.csv", fold_rows)
    write_csv(output / "ga-search-times.csv", search_rows)
    hk = timezone(timedelta(hours=8))
    launch = read(root / "launch.json")
    started = datetime.fromisoformat(launch["started_at"]).astimezone(hk)
    finished = datetime.fromtimestamp((root / "matched-status.json").stat().st_mtime, hk)
    wall_seconds = (finished - started).total_seconds()
    audit = dict(
        generated_at=datetime.now(hk).isoformat(), complete=True, rows=45,
        selected_member_records_checked=2250, verification=verification,
        started_at=started.isoformat(), finished_at_from_status_mtime=finished.isoformat(),
        approximate_wall_seconds=wall_seconds, parallel_workers=launch["jobs"],
        timing_definition="Member elapsed time includes construction, fit and validation; "
                          "excludes model serialization, outer-test prediction and featurization. "
                          "Sums across concurrent workers are not wall-clock durations.",
        search_accounting="GA time already includes selected MODNet members; do not add twice. "
                          "Both KAN variants reuse this search and have no independent search.",
        comparison_sha256=hashes, summary=summary, search_by_fold=search_rows)
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# MODNet 与 KAN 完整实验结果", "",
             f"正式训练结束于 {finished:%Y-%m-%d %H:%M:%S}（香港时间，结束时刻取完成状态文件的修改时间）。",
             "三个任务 × 五个官方外折 × 三种模型，共 45 行正式结果；全部通过配对及预测复算核验。", "",
             "## 官方成绩复现", "",
             "MAE 为五个外折的算术平均，越低越好。", "",
             "| 数据集 | 官方发布 MAE | 本次 MODNet MAE | 相对官方偏差 |",
             "| --- | ---: | ---: | ---: |"]
    for task in TASKS:
        item = alignment["tasks"][task]
        lines.append(f"| {task.removeprefix('matbench_')} | {item['published_mae']:.6f} | "
                     f"{item['retrained_mae']:.6f} | {item['relative_deviation'] * 100:+.2f}% |")
    lines += ["", "三项偏差均小于 2%，通过预设的 10% 偏差排查线。该门槛不等于统计等价或逐位复现；"
              "公开记录缺少原始种子和逐折 GA 配置，本轮使用 MODNet 0.1.12 自带 GA 默认协议。", "",
              "## 精度、参数量与训练耗时", "",
              "参数量是单成员的跨折平均；每个外折实际使用 50 个成员。耗时是五折合计 250 个选中成员的累计用时，"
              "单位分钟，不是并行运行的实际等待时间，也不含 GA 搜索的其余候选。", "",
              "| 数据集 | 模型 | MAE ± 折间标准差 | 相对本次 MODNet | 平均单成员参数 | 累计成员耗时（分钟） | 耗时倍数 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['task'].removeprefix('matbench_')} | {LABELS[row['variant']]} | "
                     f"{row['mae_mean']:.6f} ± {row['mae_fold_std']:.6f} | "
                     f"{row['delta_vs_retrained_pct']:+.2f}% | {row['member_parameters_mean']:,.0f} | "
                     f"{row['selected_member_minutes_sum']:.2f} | {row['selected_member_time_ratio']:.2f}× |")
    lines += ["", "正的 MAE 变化表示变差。折间标准差描述不同测试折的差异，不是多次独立实验的置信区间。", "",
              "## GA 搜索与总运行成本", "",
              "| 数据集 | GA 训练次数 | GA 累计成员耗时（小时） |",
              "| --- | ---: | ---: |"]
    for task in TASKS:
        rows = [r for r in search_rows if r["task"] == task]
        lines.append(f"| {task.removeprefix('matbench_')} | {sum(r['ga_fits'] for r in rows):,} | "
                     f"{sum(r['ga_member_seconds_sum'] for r in rows) / 3600:.2f} |")
    lines += ["", f"主流程启动于 {started:%Y-%m-%d %H:%M:%S}，到完成约 {wall_seconds / 3600:.2f} 小时，"
              f"使用 {launch['jobs']} 个 CPU 工作进程。此前生成描述符的时间不在这个区间内；"
              "该区间包含主流程中的搜索、训练、数据处理、模型读写、外折预测和核验。", "",
              "GA 累计耗时已经包含选入最终集成的 MODNet 成员，不能再重复相加。两种 KAN 复用同一套 MODNet GA 配置，"
              "没有自己的搜索开销，因此不能把 KAN 的增量训练时间解释成从零开发 KAN 的完整成本。", "",
              "成员计时包含模型构建、训练和内层验证，截止于保存模型之前；不含模型序列化、外层测试预测和描述符生成。"
              "不同成员并行运行，累计成员耗时不等于实际等待时间，也不是严格的 CPU 核时。计时来自这一次实际运行，"
              "未做独占硬件、多次重复的速度测试。", "",
              "## 结论与边界", "",
              "- 仅替换输出头在三个数据集上均未改善平均 MAE。",
              "- 全 KAN 在 phonons 上改善 2.92%，五折中四折优于 MODNet；steels 和 dielectric 的平均 MAE 分别变差约 0.70% 和 0.84%。",
              "- 全 KAN 参数量约为 MODNet 的四倍；其选中成员累计耗时约为 MODNet 的 1.51 倍、1.92 倍、3.69 倍（按 steels、phonons、dielectric 顺序）。",
              "- 这是同宽度替换实验；特征、划分、成员种子、优化器、超参数和集成成员均配对。KAN 使用 Gaussian RBF FastKAN，"
              "并非原始 B-spline KAN。未做参数量匹配或 KAN 独立调参。",
              "- 保留官方版本的早停和预测后处理规则；steels 的越界成员预测替换数为 MODNet 28、输出头 66、全 KAN 7，"
              "其他任务的准确计数见 summary.csv。原始及后处理预测均保留。",
              "- 当前证据不足以支持全面替换 MODNet，也不足以宣称统计显著。下一轮候选仍可讨论残差输出头、局部隐藏层替换及参数量匹配对照，尚未启动任何新实验。", "",
              "## 交付文件", "",
              "- [九组完整汇总](summary.csv)：包含平均单成员及 50 成员集成参数量、耗时、相对变化、胜出折数。",
              "- [45 行外折结果](fold-results.csv)。",
              "- [逐折 GA 搜索成本](ga-search-times.csv)。",
              "- [精度与成本对照图](accuracy-cost.png)，另有可编辑的 SVG 版本。",
              "- [核验与来源记录](audit.json)：含 15 份 comparison.json 的 SHA-256。",
              "- [实验协议](../../../docs/modnet_official_matched_protocol_20261021.md)。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(report=str(output / 'report.md'), summary=summary,
                          search_by_fold=search_rows, wall_hours=wall_seconds / 3600), ensure_ascii=False))
    return output, summary


def plot_summary(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")
    x = np.arange(3)
    colors = ("#3378B8", "#DE7A32")
    fields = ("delta_vs_retrained_pct", "parameter_ratio", "selected_member_time_ratio")
    titles = ("MAE change vs. retrained MODNet (%)", "Parameters / MODNet", "Selected-member time / MODNet")
    for ax, field, title in zip(axes, fields, titles):
        for i, variant in enumerate(("kan_head", "kan_full")):
            vals = [next(r[field] for r in summary if r["task"] == task and r["variant"] == variant) for task in TASKS]
            bars = ax.bar(x + (i - 0.5) * 0.34, vals, 0.32, color=colors[i],
                          label=("KAN head" if i == 0 else "Full KAN"))
            labels = [f"{v:+.2f}%" if field == fields[0] else f"{v:.2f}x" for v in vals]
            ax.bar_label(bars, labels=labels, padding=4, fontsize=9)
        ax.axhline(0 if field == fields[0] else 1, color="#444444", lw=0.8, ls="--")
        ax.set_xticks(x, [t.removeprefix("matbench_") for t in TASKS])
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.margins(y=0.22)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Official-protocol matched comparison | all 5 outer folds completed", fontsize=13)
    fig.supxlabel("Time: sum over 250 selected members per task and model; excludes GA search. Lower is better.", fontsize=9)
    fig.savefig(output / "accuracy-cost.png", dpi=180)
    fig.savefig(output / "accuracy-cost.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    directory, rows = build_report(args.output_dir.resolve())
    plot_summary(directory, rows)
