"""Independent final audit and accuracy/parameter/time reports for round two."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ("residual_kan", "last_hidden_kan", "residual_mlp", "last_hidden_mlp")
PAIRS = {"residual_kan": "residual_mlp", "last_hidden_kan": "last_hidden_mlp"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    spec, lock = read(root / "hybrid-protocol.json"), read(root / "selections-locked.json")
    expected_paths = {str(Path(task) / f"fold-{fold}" / "selection.json")
                      for task in spec["tasks"] for fold in spec["folds"]}
    if set(lock["selection_sha256"]) != expected_paths:
        raise ValueError("The global selection lock does not cover every requested outer fold.")
    rows, checks, tested_ids = [], 0, {}
    for task in spec["tasks"]:
        tested_ids[task] = []
        for fold in spec["folds"]:
            directory = root / task / f"fold-{fold}"
            path = directory / "selection.json"
            assert sha(path) == lock["selection_sha256"][str(path.relative_to(root))]
            selected, inputs, comparison = read(path), read(directory / "inputs.json"), read(directory / "comparison.json")
            assert selected["outer_test_accessed"] is False
            assert comparison["selection_sha256"] == sha(path)
            assert selected["run_fingerprint"] == comparison["run_fingerprint"] == lock["run_fingerprint"]
            split = inputs["outer_split"]
            outer_train, outer_test = set(split["train_ids"]), set(split["test_ids"])
            assert outer_train.isdisjoint(outer_test)
            frame = pd.read_csv(directory / "predictions.csv")
            assert frame.sample_id.tolist() == split["official_test_ids"]
            tested_ids[task].extend(frame.sample_id.tolist())
            baseline_csv = Path(spec["baseline_dir"]) / task / f"fold-{fold}" / "predictions.csv"
            baseline_frame = pd.read_csv(baseline_csv)
            assert baseline_frame.sample_id.tolist() == frame.sample_id.tolist()
            np.testing.assert_array_equal(baseline_frame.target, frame.target)
            records_by_variant = {}
            for variant in VARIANTS:
                search = selected["variants"][variant]
                budget = 4 if spec["smoke"] else 6
                expected_members = 5 if spec["smoke"] else 50
                all_records = []
                for candidate in search["candidates"]:
                    members = [read(Path(m["directory"]) / "result.json") for m in candidate["members"]]
                    assert len(members) == 5 and sorted(m["inner"] for m in members) == list(range(5))
                    np.testing.assert_allclose(np.mean([m["validation_mae"] for m in members]),
                                               candidate["mean_validation_mae"], atol=1e-12, rtol=0)
                    for member in members:
                        assert member["run_fingerprint"] == lock["run_fingerprint"]
                        assert member["variant"] == variant and member["smoke"] == spec["smoke"]
                        assert (member["base"], member["trial"], member["grids"], member["lr_multiplier"]) == (
                            candidate["base"], candidate["trial"], candidate["grids"], candidate["lr_multiplier"])
                        source = read(Path(member["source_member"]) / "result.json")
                        for key in ("genes", "seed", "train_ids", "validation_ids", "selected_features", "num_neurons"):
                            assert member[key] == source[key], (task, fold, variant, key)
                        train, val = set(member["train_ids"]), set(member["validation_ids"])
                        assert train.isdisjoint(val) and train | val == outer_train
                        assert (train | val).isdisjoint(outer_test)
                        assert abs(member["parameter_error"]) <= 0.01
                        np.testing.assert_allclose(member["effective_lr"], member["genes"]["lr"] * member["lr_multiplier"])
                        checks += 1
                    all_records.extend(members)
                assert len(all_records) == budget * expected_members == search["candidate_evaluations"]
                np.testing.assert_allclose(sum(m["seconds"] for m in all_records), search["search_member_seconds_sum"], atol=1e-6, rtol=1e-10)
                winners = []
                for winner in search["winners"]:
                    candidates = [c for c in search["candidates"] if c["base"] == winner["base"]]
                    assert len(candidates) == budget
                    assert min(candidates, key=lambda c: (c["mean_validation_mae"], c["trial"])) == winner
                    if variant.endswith("kan"):
                        actual = {(c["grids"], c["lr_multiplier"]) for c in candidates}
                        assert actual == {(g, lr) for g in spec["kan_grid"] for lr in spec["kan_lr_multipliers"]}
                    else:
                        kan = next(k for k, m in PAIRS.items() if m == variant)
                        paired = next(w for w in selected["variants"][kan]["winners"] if w["base"] == winner["base"])
                        assert {c["grids"] for c in candidates} == {paired["grids"]}
                        assert {c["lr_multiplier"] for c in candidates} == set(spec["mlp_lr_multipliers"])
                    winners += [read(Path(m["directory"]) / "result.json") for m in winner["members"]]
                assert len(winners) == expected_members
                records_by_variant[variant] = winners
            for kan, mlp in PAIRS.items():
                for k, m in zip(records_by_variant[kan], records_by_variant[mlp]):
                    assert (k["source_member"], k["grids"]) == (m["source_member"], m["grids"])
                    assert abs(m["parameters"] / k["parameters"] - 1) <= 0.01
            for variant in ("modnet", *VARIANTS):
                predictions = np.load(directory / f"{variant}-member-predictions.npz")
                ensemble = predictions["predictions"]
                assert ensemble.shape == (spec["ensemble_members"], len(frame))
                assert np.isfinite(ensemble).all()
                np.testing.assert_allclose(ensemble.mean(axis=0), frame[variant], rtol=1e-7, atol=1e-7)
                mae = float(np.mean(np.abs(frame.target.values - ensemble.mean(axis=0))))
                stats = comparison["results"][variant]
                np.testing.assert_allclose(mae, stats["mae"], rtol=0, atol=1e-9)
                if variant != "modnet":
                    members = records_by_variant[variant]
                    np.testing.assert_allclose(np.mean([m["parameters"] for m in members]), stats["parameters_mean"])
                    np.testing.assert_allclose(sum(m["seconds"] for m in members), stats["selected_member_seconds_sum"])
                elif not spec["smoke"]:
                    np.testing.assert_allclose(frame[variant], baseline_frame[variant], rtol=0, atol=1e-7)
                rows.append(dict(task=task, fold=fold, variant=variant, mae=mae,
                                 parameters=stats["parameters_mean"], ensemble_parameters=stats["parameters_sum"],
                                 selected_member_seconds=stats["selected_member_seconds_sum"],
                                 search_member_seconds=stats["search_member_seconds_sum"],
                                 postprocess_replacements=stats["postprocess_replacements"]))
        assert len(tested_ids[task]) == len(set(tested_ids[task]))
    table = pd.DataFrame(rows)
    table.to_csv(root / "hybrid-fold-results.csv", index=False)
    summary = table.groupby(["task", "variant"]).agg(mae=("mae", "mean"), fold_std=("mae", "std"),
                 parameters=("parameters", "mean"), ensemble_parameters=("ensemble_parameters", "mean"),
                 selected_member_seconds=("selected_member_seconds", "sum"),
                 search_member_seconds=("search_member_seconds", lambda x: x.sum(min_count=1)))
    summary.to_csv(root / "hybrid-summary.csv")
    full = len(rows) == 75 and not spec["smoke"]
    report = dict(complete_full_benchmark=full, smoke=spec["smoke"], verified_rows=len(rows),
                  verified_search_members=checks, matching_features_splits_and_seeds=True,
                  parameter_matched_controls_within_one_percent=True, only_inner_scores_used_for_selection=True,
                  global_selection_lock_verified=True, finite_predictions_and_recomputed_scores=True)
    (root / "hybrid-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# 第二轮 KAN 与等参数量 MLP 实验", "", f"正式完整结果：{full}；smoke：{spec['smoke']}。", "",
             "MAE 为五折平均；参数量为单成员平均。耗时为成员累计构建、训练、验证时间（分钟），不是墙钟时间。"
             "搜索时间已经包含选中成员，不能重复相加。MODNet 沿用第一轮，不记为零成本重新训练。", "",
             "| 数据集 | 模型 | MAE | 折间标准差 | 参数量 | 选中成员分钟 | 搜索成员分钟 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (task, variant), row in summary.iterrows():
        search = "第一轮复用" if pd.isna(row.search_member_seconds) else f"{row.search_member_seconds / 60:.2f}"
        lines.append(f"| {task} | {variant} | {row.mae:.6f} | {row.fold_std:.6f} | {row.parameters:.0f} | "
                     f"{row.selected_member_seconds / 60:.2f} | {search} |")
    lines += ["", "MLP 分别匹配两个 KAN 的内层选定参数量（误差不超过 1%）。特征排序沿用第一轮外层训练集，"
              "内层不重新筛选；这是同特征对照，不是严格嵌套的特征筛选。所有新选择先冻结，再评估外层测试。"
              "本轮假设提出前已看过第一轮外层成绩，应作为探索性后续实验，不能称为从未接触的独立测试集。", ""]
    (root / "hybrid-report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    verify(parser.parse_args().output_dir)
