"""Verify completed MatterVial folds and write an explicitly partial/full report."""
from pathlib import Path
import argparse
from run_mattervial_phonons import (read, write_json, digest, np, pd, PHASES, VARIANTS, STUDY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "run")
    args = parser.parse_args()
    root = args.output.resolve()
    protocol = read(root / "protocol.json")
    fingerprint = digest(protocol)
    nested = 2 if protocol["smoke"] else 5
    folds, comparisons = [], []
    verified_members = 0
    for phase in PHASES:
        for directory in sorted((root / phase).glob("fold-*")):
            if not (directory / "complete.json").exists():
                continue
            if read(directory / "complete.json")["run_fingerprint"] != fingerprint:
                raise ValueError("Completed fold identity changed.")
            split = read(directory / "outer-split.json")
            test_ids, outer_ids = set(split["test_ids"]), set(split["train_ids"])
            if test_ids & outer_ids:
                raise ValueError("Outer contamination.")
            validation_cover = []
            manifest_by_ids = {}
            for i in range(nested):
                info = read(directory / "inputs" / f"inner-{i}" / "manifest.json")
                train_ids, val_ids = set(info["train_ids"]), set(info["validation_ids"])
                if train_ids & val_ids or (train_ids | val_ids) != outer_ids:
                    raise ValueError("Invalid inner fold.")
                for key in ("feature_selection_fit_ids", "cross_nmi_fit_ids"):
                    if info[key] != info["train_ids"]:
                        raise ValueError("Feature fitting scope changed.")
                if phase == "no_sisso" and any(c.startswith("SISSO") for c in info["rfe_features"]):
                    raise ValueError("SISSO found in primary control.")
                validation_cover += info["validation_ids"]
                manifest_by_ids[digest(info["validation_ids"])] = info
            if sorted(validation_cover) != sorted(split["train_ids"]):
                raise ValueError("Incomplete inner coverage.")
            for p in (directory / "modnet").glob("*/result.json"):
                record = read(p)
                info = manifest_by_ids[digest(record["validation_ids"])]
                if record["train_ids"] != info["train_ids"] or record["features"] != info["selected_features"]:
                    raise ValueError("GA model trained on unexpected partition/features.")
                verified_members += 1
            score = read(directory / "ga-score.json")
            frame = pd.read_csv(directory / "ga-predictions.csv")
            if frame.sample_id.tolist() != split["test_ids"]:
                raise ValueError("Test prediction order changed.")
            # NPZ preserves float32 predictions exactly. Default pandas CSV
            # rendering can round those values, so audit the binary source too.
            binary = np.load(directory / "ga_modnet-member-predictions.npz")
            for key, metric in (("predictions", "mae"), ("raw_predictions", "raw_mae")):
                np.testing.assert_allclose(np.mean(abs(binary[key].mean(axis=0) - frame.target.values)),
                                           score[metric], rtol=0, atol=1e-8)
            np.testing.assert_allclose(np.mean(abs(frame.prediction - frame.target)), score["mae"], rtol=0, atol=1e-5)
            np.testing.assert_allclose(np.mean(abs(frame.raw_prediction - frame.target)), score["raw_mae"], rtol=0, atol=1e-5)
            folds.append(score)
            if phase != "no_sisso":
                continue
            frozen = read(directory / "frozen-controls.json")["selections"]
            comparison = read(directory / "comparison.json")
            records = [read(p) for p in (directory / "controls").glob("*/*/result.json")]
            for record in records:
                info = manifest_by_ids[digest(record["validation_ids"])]
                if record["train_ids"] != info["train_ids"] or record["scaling_fit_ids"] != info["train_ids"]:
                    raise ValueError("Control train/scaler fitting scope changed.")
                val = np.load(Path(record["directory"]) / "validation.npz")
                np.testing.assert_allclose(np.mean(abs(val["predictions"] - val["target"])), record["validation_mae"], rtol=0, atol=1e-8)
                verified_members += 1
            for variant in VARIANTS:
                rows = [read(Path(d) / "result.json") for d in frozen[variant]["members"]]
                saved = np.load(directory / (variant + "-member-predictions.npz"))
                np.testing.assert_array_equal(saved["sample_ids"], split["test_ids"])
                np.testing.assert_allclose(np.mean(abs(saved["predictions"].mean(axis=0) - saved["target"])),
                                           comparison["results"][variant]["mae"], rtol=0, atol=1e-8)
                if sum(r["parameters"] for r in rows) != comparison["results"][variant]["parameters_sum"]:
                    raise ValueError("Ensemble parameter count mismatch.")
            comparisons.append(dict(fold=score["fold"], **comparison))
    state = read(root / "status.json")
    complete = state["state"] == "complete"
    expected_folds = 2 if protocol["smoke"] else 10
    if complete and len(folds) != expected_folds:
        raise ValueError("Completion status does not match fold artifacts.")
    write_json(root / "verification.json", dict(completed_folds=len(folds), expected_folds=expected_folds,
        verified_training_members=verified_members, completed_folds_verified=True,
        all_training_complete=complete, smoke=protocol["smoke"],
        scope="manifest/sample identity, fit scope, inner and outer MAEs, selected ensemble parameters"))
    lines = ["# MatterVial phonons 进度与结果", "",
        f"状态：{state['state']}；完成 {len(folds)}/{expected_folds} 个阶段×外层折；smoke={protocol['smoke']}。", "",
        "作者公开预测重算为 30.08006109 cm⁻¹；这不是本地重新训练成绩。含 SISSO 的结果仅为来源仍有疑问的诊断。", "",
        "| 阶段 | 外层折 | GA MAE | 原始 MAE | 平均参数量 | 选中训练分钟 | 全搜索分钟 | 搜索训练数 |", 
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in folds:
        lines.append(f"| {r['phase']} | {r['fold'] + 1} | {r['mae']:.6f} | {r['raw_mae']:.6f} | {r['parameters_mean']:.1f} | "
                     f"{r['selected_training_seconds']/60:.2f} | {r['search_seconds']/60:.2f} | {r['search_fits']} |")
    if not folds:
        lines += ["", "尚无完成的外层测试成绩。"]
    lines += ["", "| 无 SISSO 对照 | 外层折 | MAE | 平均参数量 | 集成总参数量 | 成员数 | 搜索分钟 | 训练次数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for comparison in comparisons:
        for variant, r in comparison["results"].items():
            lines.append(f"| {variant} | {comparison['fold']+1} | {r['mae']:.6f} | {r['parameters_mean']:.1f} | "
                f"{r['parameters_sum']} | {r['ensemble_size']} | {r['search_seconds']/60:.2f} | {r['search_fits']} |")
    if complete and not protocol["smoke"]:
        summary = read(root / "summary.json")
        lines += ["", "完整五折均值：", "", *[f"- {k}: {v:.6f}" for k, v in summary["ga_mean_mae"].items()],
                  *[f"- 无 SISSO {k}: {v:.6f}" for k, v in summary["controlled_mean_mae"].items()]]
    lines += ["", "训练时间为各成员累计时间，不等于并行墙钟时间。共同的 MODNet GA 主干搜索成本单列；相等预算仅指受控阶段。预训练表示提取成本未知。",
        "", "全部超参数选择使用内层验证。主比较全部去掉 SISSO；这不等于证明所有外部预训练数据均无任务重叠。",
        "", "SMOKE 成绩只用于代码验证，不得用于模型性能结论。" if protocol["smoke"] else ""]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "report.md")


if __name__ == "__main__":
    main()
