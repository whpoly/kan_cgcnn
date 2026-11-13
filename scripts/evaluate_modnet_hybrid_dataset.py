"""Evaluate an already selected five-fold task without changing a live search.

This is a user-requested amendment to the original all-three-task release gate.
Every selection and selected checkpoint for this task is hashed before testing.
The original training protocol and files remain untouched.
"""
from __future__ import annotations

from run_modnet_hybrid_nested import (
    TASKS, VARIANTS, digest, evaluate_fold, file_hash, now, read, write_json,
)
import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import time
import traceback


def copy_exact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise ValueError("Existing evaluation snapshot differs: " + str(destination))
    else:
        shutil.copyfile(source, destination)


def check_hashes(paths):
    for path, expected in paths.items():
        if file_hash(Path(path)) != expected:
            raise ValueError("Frozen file changed: " + path)


def prepare(args):
    source_protocol = args.source_dir / "hybrid-protocol.json"
    original = read(source_protocol)
    fingerprint = digest(original)
    if original["smoke"] or original["folds"] != list(range(5)):
        raise ValueError("This evaluator requires a production five-fold run.")
    if args.task not in original["tasks"]:
        raise ValueError("Task is outside the original protocol.")
    if {name: importlib.metadata.version(name) for name in original["packages"]} != original["packages"]:
        raise ValueError("Prediction environment differs from the training environment.")
    check_hashes(original["source_sha256"])
    task_inputs = {p: h for p, h in original["input_sha256"].items()
                   if args.task in Path(p).parts or Path(p).name == args.task + "_moddata.pkl.gz"}
    check_hashes(task_inputs)
    args.baseline_dir = Path(original["baseline_dir"])
    if file_hash(args.baseline_dir / "matched-protocol.json") != original["baseline_protocol_sha256"]:
        raise ValueError("Baseline protocol changed.")
    if args.output_dir in (args.source_dir, args.baseline_dir):
        raise ValueError("Use a separate evaluation output directory.")

    hashes = {}
    source_selections = {}
    for fold in range(5):
        source = args.source_dir / args.task / f"fold-{fold}"
        selected = read(source / "selection.json")
        if (selected["run_fingerprint"] != fingerprint or selected["smoke"]
                or selected["outer_test_accessed"] is not False):
            raise ValueError("Selection is not an untouched compatible production selection.")
        inputs = read(source / "inputs.json")
        source_selections[str(source / "selection.json")] = file_hash(source / "selection.json")
        for variant in VARIANTS:
            search = selected["variants"][variant]
            if len(search["winners"]) != 10 or search["candidate_evaluations"] != 300:
                raise ValueError("Task does not have a complete inner search.")
            for winner in search["winners"]:
                candidates = [c for c in search["candidates"] if c["base"] == winner["base"]]
                if (len(candidates) != 6 or len(winner["members"]) != 5
                        or min(candidates, key=lambda c: (c["mean_validation_mae"], c["trial"])) != winner):
                    raise ValueError("Selection is not the complete inner-validation winner.")
                for member in winner["members"]:
                    directory = Path(member["directory"])
                    for filename in ("model.pkl", "result.json"):
                        path = directory / filename
                        hashes[str(path)] = file_hash(path)
        for base in inputs["bases"]:
            for member in base["sources"]:
                for filename in ("model.pkl", "result.json"):
                    path = Path(member["directory"]) / filename
                    hashes[str(path)] = file_hash(path)
        for filename in ("selection.json", "inputs.json"):
            copy_exact(source / filename, args.output_dir / args.task / f"fold-{fold}" / filename)

    copy_exact(source_protocol, args.output_dir / "source-hybrid-protocol.json")
    spec = dict(original, tasks=[args.task], evaluation_only=True,
                source_run_fingerprint=fingerprint, source_dir=str(args.source_dir),
                original_tasks=original["tasks"],
                gate="All five selections for this task frozen before its test; user requested early task evaluation.")
    spec_path = args.output_dir / "hybrid-protocol.json"
    if spec_path.exists() and read(spec_path) != spec:
        raise ValueError("Evaluation scope changed.")
    write_json(spec_path, spec)
    lock_path = args.output_dir / "selections-locked.json"
    lock = dict(run_fingerprint=fingerprint, scope="all_five_folds_of_" + args.task, smoke=False,
                selection_sha256={str(p.relative_to(args.output_dir)): file_hash(p)
                                  for p in sorted((args.output_dir / args.task).glob("fold-*/selection.json"))})
    if lock_path.exists():
        previous = read(lock_path)
        if {k: previous[k] for k in lock} != lock:
            raise ValueError("Task selection changed after the lock.")
        lock = previous
    else:
        lock["locked_at"] = now()
        write_json(lock_path, lock)
    checkpoint_lock = args.output_dir / "selected-checkpoints-locked.json"
    frozen = dict(source_selection_sha256=source_selections, selected_files_sha256=hashes)
    if checkpoint_lock.exists():
        if read(checkpoint_lock) != frozen:
            raise ValueError("Checkpoint or source selection changed after the lock.")
    else:
        write_json(checkpoint_lock, frozen)
    write_json(args.output_dir / "evaluation-amendment.json", dict(
        source_dir=str(args.source_dir), task=args.task, folds=list(range(5)),
        authorization="User requested outer-test evaluation now after steels completed inner selection.",
        original_three_task_gate_satisfied=False,
        task_five_fold_gate_satisfied=True, locked_at=lock["locked_at"],
        training_and_selection_unchanged=True,
        script_sha256=file_hash(Path(__file__)),
        prediction_function="unchanged run_modnet_hybrid_nested.evaluate_fold",
        source_protocol_sha256=file_hash(source_protocol)))
    return fingerprint, lock, frozen


def report(args, audit, elapsed):
    import numpy as np
    import pandas as pd
    table = pd.read_csv(args.output_dir / "hybrid-fold-results.csv")
    wide = table.pivot(index="fold", columns="variant", values="mae")
    rows = []
    for variant in ("modnet", *VARIANTS):
        part = table[table.variant == variant]
        paired = {"residual_kan": "residual_mlp", "last_hidden_kan": "last_hidden_mlp"}.get(variant)
        rows.append(dict(
            variant=variant, outer_mae=float(part.mae.mean()),
            fold_std_population=float(np.std(part.mae, ddof=0)),
            change_vs_modnet_pct=float((part.mae.mean() / wide.modnet.mean() - 1) * 100),
            improved_folds_vs_modnet=int((wide[variant] < wide.modnet).sum()),
            paired_mlp=paired,
            change_vs_paired_mlp_pct=(float((part.mae.mean() / wide[paired].mean() - 1) * 100) if paired else None),
            improved_folds_vs_paired_mlp=(int((wide[variant] < wide[paired]).sum()) if paired else None),
            member_parameters_mean=float(part.parameters.mean()),
            ensemble_parameters_mean=float(part.ensemble_parameters.mean()),
            selected_member_minutes=float(part.selected_member_seconds.sum() / 60),
            search_member_minutes=(float(part.search_member_seconds.sum() / 60) if variant != "modnet" else None)))
    result = dict(task=args.task, folds=5, ensemble_members=50,
                  complete_task_outer_evaluation=True, complete_three_task_benchmark=False,
                  evaluation_wall_seconds=elapsed, completed_at=now(), results=rows,
                  verified_search_members=audit["verified_search_members"],
                  original_modnet_predictions_reproduced=True,
                  selection_lock_scope="all_five_folds_of_" + args.task)
    write_json(args.output_dir / "dataset-outer-summary.json", result)
    pd.DataFrame(rows).to_csv(args.output_dir / "dataset-outer-summary.csv", index=False)
    labels = {"modnet": "原 MODNet", "residual_kan": "残差 KAN", "last_hidden_kan": "末隐藏层 KAN",
              "residual_mlp": "残差等参数 MLP", "last_hidden_mlp": "末隐藏层等参数 MLP"}
    lines = ["# steels 第二轮完整五折外层测试", "",
             "按用户要求提前评估已经完成内层选择的 steels。测试前冻结全部五折选型及所选模型；"
             "原三数据集训练继续进行，训练代码和候选选择未修改。此目录仅报告 steels，其他数据集尚未完成。", "",
             "MAE 为五个官方外层测试折的算术平均，每折使用 50 成员集成，采用与第一轮相同的预测后处理。", "",
             "| 模型 | 外层 MAE | 相对原 MODNet 变化 | 单成员平均参数量 | 选中成员累计分钟 | 搜索累计分钟 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        search = "复用第一轮" if row["search_member_minutes"] is None else f"{row['search_member_minutes']:.2f}"
        lines.append(f"| {labels[row['variant']]} | {row['outer_mae']:.6f} | {row['change_vs_modnet_pct']:+.3f}% | "
                     f"{row['member_parameters_mean']:.2f} | {row['selected_member_minutes']:.2f} | {search} |")
    lines += ["", "| 外层折 | 原 MODNet | 残差 KAN | 残差等参数 MLP | 末隐藏层 KAN | 末隐藏层等参数 MLP |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for fold, row in wide.iterrows():
        vals = [f"{row[v]:.6f}" for v in ("modnet", "residual_kan", "residual_mlp", "last_hidden_kan", "last_hidden_mlp")]
        lines.append("| " + str(fold + 1) + " | " + " | ".join(vals) + " |")
    lines += ["", "参数量为单成员平均，平均集成参数量为其 50 倍。训练时间是成员累计构建、训练与验证时间，"
              "不是并行墙钟时间；搜索时间已包含选中成员，不应相加。原 MODNet 沿用第一轮训练时间。", "",
              f"独立核验 {audit['verified_rows']} 行外层结果及 {audit['verified_search_members']} 个候选训练记录；"
              "原 MODNet 逐样本预测与第一轮一致，所选 KAN/MLP 成员参数量匹配误差不超过 1%。", "",
              "本轮是观察过第一轮 benchmark 后提出的探索性后续实验。其他数据集仍按预先固定的候选和内层选择规则完成，"
              "不根据此次外层成绩改变本轮模型或训练方案。", ""]
    (args.output_dir / "dataset-outer-report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", default="matbench_steels", choices=TASKS)
    args = parser.parse_args()
    args.source_dir, args.output_dir = args.source_dir.resolve(), args.output_dir.resolve()
    args.smoke = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = args.output_dir / "evaluation-status.json"
    started = time.perf_counter()
    try:
        write_json(status, dict(state="running", phase="freeze_completed_task", pid=os.getpid(), started_at=now()))
        fingerprint, lock, frozen = prepare(args)
        print("LOCKED all five task selections and selected checkpoints before test evaluation.", flush=True)
        for fold in range(5):
            write_json(status, dict(state="running", phase="outer_evaluation", pid=os.getpid(), task=args.task,
                                    fold=fold, completed_folds=fold, updated_at=now()))
            result = evaluate_fold(args, args.task, fold, fingerprint, lock)
            print("COMPLETED fold", fold + 1, {k: v["mae"] for k, v in result["results"].items()}, flush=True)
        write_json(status, dict(state="running", phase="independent_audit", pid=os.getpid(), updated_at=now()))
        from verify_modnet_hybrid_nested import verify
        audit = verify(args.output_dir)
        if audit["verified_rows"] != 25 or audit["verified_search_members"] != 6000:
            raise ValueError("Incomplete five-fold task audit.")
        check_hashes(frozen["source_selection_sha256"])
        check_hashes(frozen["selected_files_sha256"])
        result = report(args, audit, time.perf_counter() - started)
        write_json(status, dict(state="complete", completed_at=now(), task=args.task,
                                completed_folds=5, evaluation_wall_seconds=result["evaluation_wall_seconds"]))
        print("COMPLETE", args.output_dir / "dataset-outer-summary.json", flush=True)
    except Exception:
        write_json(status, dict(state="failed", pid=os.getpid(), failed_at=now(), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
