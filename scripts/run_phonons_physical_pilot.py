"""Training-only phonons pilot: 16 physical inputs, four features, five controls."""
from __future__ import annotations

# Sets native DLL paths, CPU-only TensorFlow, and one thread per worker.
from run_modnet_official_matched import SafePool, digest, file_hash, set_seed, write_json, sample_ids
from run_modnet_hybrid_nested import recorded_inner_positions
from physical_feature_modules import PHYSICAL_FEATURES, PHYSICAL_GROUPS, VARIANTS, WIDTHS, build_predictor, fit_minmax
from modnet.preprocessing import MODData
import argparse
import contextlib
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(args):
    source = args.baseline / "matbench_phonons" / "fold-0"
    path = source / "official-train-moddata.pkl.gz"
    source_protocol = read(args.baseline / "matched-protocol.json")
    source_hashes = {str(Path(__file__).resolve()): file_hash(Path(__file__)),
                     str(Path(__file__).with_name("physical_feature_modules.py").resolve()):
                     file_hash(Path(__file__).with_name("physical_feature_modules.py"))}
    for dependency in ("run_modnet_official_matched.py", "run_modnet_hybrid_nested.py", "modnet_runtime.py"):
        p = Path(__file__).with_name(dependency).resolve()
        source_hashes[str(p)] = file_hash(p)
    protocol = dict(kind="inner_only_physical_features_pilot", task="matbench_phonons", outer_fold=0,
        source_train_cache=str(path.resolve()), source_train_sha256=file_hash(path),
        source_frozen_sha256=file_hash(source / "frozen-ensemble.json"), source_sha256=source_hashes,
        packages={name: importlib.metadata.version(name) for name in source_protocol["packages"]},
        physical_groups=PHYSICAL_GROUPS, physical_outputs=4, variants=VARIANTS,
        trunk_widths=WIDTHS, base_feature_count=190, inner_folds=5,
        learning_rates=[.0005, .001], seeds=[20261023] if args.smoke else [20261023, 20271023],
        max_epochs=1 if args.smoke else 800, batch_size=32,
        early_stopping=dict(monitor="mae", patience=30, min_delta=.001, restore_best_weights=False),
        loss="raw-target MAE plus declared module penalties", scaling="train-only minmax [-0.5,0.5]",
        feature_selection="Recompute native MODNet relevance-redundancy on each inner training partition; reuse frozen target-independent cross-NMI.",
        spline=dict(degree=3, intervals=5, basis_count=8, group_shape=[4, 4], module_parameters=148,
                    smooth_strength=.001, sparse_strength=.01, penalty_scale="inner-training target standard deviation"),
        spline_kan_distinction="Identical grouped additive B-spline function class and smoothness; KAN additionally has soft edge/group sparsity.",
        selection="Equal-weight mean validation MAE over all five inner folds and both paired seeds; learning rate only.",
        inherited_design="Core widths and feature count fixed using the earlier phonons inner-selected configuration, never its outer-test score.",
        outer_test_accessed=False, smoke=args.smoke, original_training=source_protocol["training"])
    fingerprint = digest(protocol)
    p = args.output / "pilot-protocol.json"
    if p.exists() and digest(read(p)) != fingerprint:
        raise ValueError("Pilot protocol changed; use another output directory.")
    write_json(p, protocol)
    data = MODData.load(path)
    if len(data.df_targets) != 1012:
        raise ValueError("Expected only outer-fold-0 training samples.")
    if set(PHYSICAL_FEATURES) - set(data.df_featurized):
        raise ValueError("Missing predeclared physical descriptors.")
    if data.cross_nmi is None:
        raise ValueError("Missing frozen target-independent cross-NMI.")
    original_members = read(source / "frozen-ensemble.json")["members"]
    group = Path(original_members[0]["directory"]).name.rsplit("-inner-", 1)[0]
    members = sorted([m for m in original_members if Path(m["directory"]).name.rsplit("-inner-", 1)[0] == group],
                     key=lambda m: int(Path(m["directory"]).name.rsplit("-inner-", 1)[1]))
    assert len(members) == 5
    outer_ids = sample_ids(data)
    manifests, covered = [], []
    for inner, member in enumerate(members):
        directory = args.output / "inputs" / f"inner-{inner}"
        directory.mkdir(parents=True, exist_ok=True)
        saved = directory / "manifest.json"
        if saved.exists():
            item = read(saved)
            if (item["run_fingerprint"] != fingerprint or file_hash(directory / "arrays.npz") != item["arrays_sha256"]
                    or file_hash(directory / "scaler.pkl") != item["scaler_sha256"]):
                raise ValueError("Incompatible cached pilot inputs.")
            manifests.append(item)
            covered += item["validation_ids"]
            continue
        source_result = read(Path(member["directory"]) / "result.json")
        train_pos, val_pos = recorded_inner_positions(outer_ids, source_result)
        train, validation = data.split((train_pos, val_pos))
        set_seed(20261023 + inner)
        # Recomputes target NMI from train.df_targets; validation labels are never passed.
        with (directory / "feature-selection.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                train.feature_selection(n=190, cross_nmi=data.cross_nmi, n_jobs=1)
        core = list(train.get_optimal_descriptors())[:190]
        if len(core) != 190:
            raise ValueError("Insufficient native MODNet features.")
        augmented = core + [f for f in PHYSICAL_FEATURES if f not in core]
        raw_train = train.df_featurized[augmented].replace([np.inf, -np.inf], np.nan).fillna(0).values
        raw_val = validation.df_featurized[augmented].replace([np.inf, -np.inf], np.nan).fillna(0).values
        scaler, x_train, x_val = fit_minmax(raw_train, raw_val)
        y_train, y_val = train.df_targets.iloc[:, 0].values, validation.df_targets.iloc[:, 0].values
        np.savez_compressed(directory / "arrays.npz", x_train=x_train, x_validation=x_val,
                            y_train=y_train, y_validation=y_val)
        with (directory / "scaler.pkl").open("wb") as stream:
            pickle.dump(scaler, stream, protocol=4)
        item = dict(inner=inner, run_fingerprint=fingerprint, core_features=core, input_features=augmented,
            physical_features=PHYSICAL_FEATURES, train_ids=sample_ids(train), validation_ids=sample_ids(validation),
            target_name=str(train.df_targets.columns[0]), target_scale=float(np.std(y_train)),
            arrays=str((directory / "arrays.npz").resolve()), arrays_sha256=file_hash(directory / "arrays.npz"),
            scaler_sha256=file_hash(directory / "scaler.pkl"), feature_selection_fit_ids=sample_ids(train),
            scaling_fit_ids=sample_ids(train), scaled_validation_outside_fraction=float(np.mean((x_val < -.5) | (x_val > .5))))
        write_json(saved, item)
        manifests.append(item)
        covered += item["validation_ids"]
        print("PREPARED inner", inner, "core", len(core), "union", len(augmented), flush=True)
    if sorted(covered) != sorted(outer_ids) or len(covered) != len(set(covered)):
        raise ValueError("Inner validation partitions do not cover the outer training data exactly once.")
    write_json(args.output / "input-audit.json", dict(outer_train_ids=outer_ids,
               all_five_inner_splits_verified=True, features_selected_on_inner_train_only=True,
               scaling_fitted_on_inner_train_only=True, outer_test_accessed=False,
               input_manifests={str((args.output / "inputs" / f"inner-{i}" / "manifest.json").resolve()):
                                file_hash(args.output / "inputs" / f"inner-{i}" / "manifest.json") for i in range(5)}))
    return protocol, fingerprint, manifests


def train_job(job):
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    identity = digest(job)
    if (directory / "result.json").exists():
        cached = read(directory / "result.json")
        if cached["signature"] != identity or not (directory / "model.pkl").exists():
            raise ValueError("Cached pilot member mismatch.")
        return cached
    try:
        info = job["inputs"]
        if file_hash(Path(info["arrays"])) != info["arrays_sha256"]:
            raise ValueError("Training-only arrays changed.")
        arrays = np.load(info["arrays"])
        n = len(info["core_features"]) if job["variant"] == "modnet" else len(info["input_features"])
        x, xv = arrays["x_train"][:, :n], arrays["x_validation"][:, :n]
        y, yv = arrays["y_train"], arrays["y_validation"]
        started = time.perf_counter()
        model = build_predictor(job["variant"], info["core_features"], info["input_features"][:n],
                                info["target_name"], job["seed"], info["target_scale"])
        module = next((layer for layer in model.layers if layer.name == "physical_module"), None)
        if module is not None and module.count_params() != 148:
            raise ValueError("Physical modules must have exactly 148 active parameters.")
        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 25 == 0:
                    write_json(directory / "status.json", dict(state="training", epoch=epoch + 1,
                               seconds=time.perf_counter() - started, metrics=logs or {}))
        set_seed(job["seed"])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=job["lr"]), loss="mae", metrics=["mae"])
        with (directory / "training.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                history = model.fit(x, y, validation_data=(xv, yv), batch_size=32, epochs=job["max_epochs"], verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="mae", min_delta=.001, patience=30,
                                                                 restore_best_weights=False), Progress()])
        prediction = model.predict(xv, verbose=0).reshape(-1).astype(float)
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite inner validation predictions.")
        np.savez_compressed(directory / "validation-predictions.npz", sample_ids=info["validation_ids"],
                            target=yv, predictions=prediction)
        diagnostics = {}
        if module is not None:
            features = tf.keras.Model(model.input, module.output).predict(x, verbose=0)
            diagnostics["feature_train_std"] = features.std(axis=0).tolist()
            model(tf.convert_to_tensor(x[:32]), training=False)
            diagnostics["regularization_loss"] = float(sum(model.losses))
            if job["variant"] != "group_mlp":
                diagnostics["edge_coefficient_norms"] = np.sqrt(np.mean(module.coefficients.numpy() ** 2, axis=-1) + module.base.numpy() ** 2).tolist()
        record = dict(signature=identity, run_fingerprint=job["run_fingerprint"], variant=job["variant"],
            lr=job["lr"], inner=info["inner"], seed=job["seed"], seed_repeat=job["seed_repeat"],
            validation_mae=float(np.mean(abs(prediction - yv))), validation_rmse=float(np.sqrt(np.mean((prediction - yv) ** 2))),
            parameters=model.count_params(), module_parameters=0 if module is None else module.count_params(),
            epochs=len(history.history["loss"]), seconds=time.perf_counter() - started,
            train_ids=info["train_ids"], validation_ids=info["validation_ids"], input_features=info["input_features"][:n],
            diagnostics=diagnostics, history=history.history, directory=str(directory))
        with (directory / "model.pkl").open("wb") as stream:
            pickle.dump(dict(model_json=model.to_json(), weights=model.get_weights(), input_features=record["input_features"],
                             input_manifest=info, variant=job["variant"]), stream, protocol=4)
        write_json(directory / "result.json", record)
        write_json(directory / "status.json", dict(state="complete", seconds=record["seconds"], finished_at=now()))
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        return record
    except Exception:
        write_json(directory / "status.json", dict(state="failed", traceback=traceback.format_exc()))
        raise


def choose(records, seeds):
    selections = {}
    for variant in VARIANTS:
        candidates = []
        for lr in (.0005, .001):
            members = sorted([r for r in records if r["variant"] == variant and r["lr"] == lr],
                             key=lambda r: (r["inner"], r["seed_repeat"]))
            if {(r["inner"], r["seed_repeat"]) for r in members} != {(i, s) for i in range(5) for s in range(seeds)}:
                raise ValueError("Incomplete paired validation candidate.")
            candidates.append(dict(lr=lr, mean_validation_mae=float(np.mean([r["validation_mae"] for r in members])),
                                   members=[r["directory"] for r in members]))
        selections[variant] = dict(candidates=candidates, winner=min(candidates, key=lambda c: (c["mean_validation_mae"], c["lr"])))
    return selections


def finish(args, protocol, records, manifests):
    choices = choose(records, len(protocol["seeds"]))
    write_json(args.output / "inner-selections.json", choices)
    rows, oof = [], {}
    for variant in VARIANTS:
        chosen = [read(Path(d) / "result.json") for d in choices[variant]["winner"]["members"]]
        fold_scores = []
        for inner in range(5):
            members = [r for r in chosen if r["inner"] == inner]
            predictions = [np.load(Path(r["directory"]) / "validation-predictions.npz") for r in members]
            for p in predictions:
                assert p["sample_ids"].tolist() == manifests[inner]["validation_ids"]
            target = predictions[0]["target"]
            mean = np.mean([p["predictions"] for p in predictions], axis=0)
            fold_scores.append(float(np.mean(abs(mean - target))))
            for sample, y, pred in zip(predictions[0]["sample_ids"], target, mean):
                oof.setdefault(str(sample), {"sample_id":str(sample), "inner":inner, "target":float(y)})[variant] = float(pred)
        row = dict(variant=variant, selected_lr=choices[variant]["winner"]["lr"],
            selected_member_validation_mae=choices[variant]["winner"]["mean_validation_mae"],
            two_seed_ensemble_validation_mae=float(np.mean(fold_scores)), fold_ensemble_mae=fold_scores,
            parameters_mean=float(np.mean([r["parameters"] for r in chosen])),
            module_parameters=chosen[0]["module_parameters"], selected_member_minutes=sum(r["seconds"] for r in chosen)/60,
            search_member_minutes=sum(r["seconds"] for r in records if r["variant"] == variant)/60)
        rows.append(row)
    for record in records:
        info = manifests[record["inner"]]
        assert record["train_ids"] == info["train_ids"] and record["validation_ids"] == info["validation_ids"]
        assert set(record["train_ids"]).isdisjoint(record["validation_ids"])
        p = np.load(Path(record["directory"]) / "validation-predictions.npz")
        np.testing.assert_allclose(np.mean(abs(p["predictions"]-p["target"])), record["validation_mae"], atol=1e-12, rtol=0)
    for inner in range(5):
        for seed in protocol["seeds"]:
            for lr in protocol["learning_rates"]:
                matched = [r["parameters"] for r in records if r["inner"] == inner and r["seed"] == seed + inner
                           and r["lr"] == lr and r["variant"] in ("group_mlp", "spline", "group_sparse_kan")]
                assert len(matched) == 3 and len(set(matched)) == 1
    summary = dict(smoke=args.smoke, outer_test_accessed=False, inner_only=True, selected_on_reported_validation=True,
                   completed_fits=len(records), outer_fold=0, results=rows)
    write_json(args.output / "pilot-summary.json", summary)
    pd.DataFrame(rows).to_csv(args.output / "pilot-summary.csv", index=False)
    pd.DataFrame(list(oof.values())).to_csv(args.output / "selected-inner-oof-predictions.csv", index=False)
    write_json(args.output / "pilot-verification.json", dict(verified_members=len(records),
        validation_predictions_recomputed=True, paired_splits_and_seeds=True,
        all_three_modules_exactly_148_parameters=True, module_models_equal_total_parameters=True,
        selections_use_only_inner_validation=True, outer_test_accessed=False,
        feature_and_scaler_fit_scope="inner_train", results_are_selection_validation_not_unbiased_outer_test=True))
    labels = dict(modnet="MODNet 架构对照", physical_input="相同物理输入", group_mlp="等参数分组 MLP",
                  spline="分组样条", group_sparse_kan="分组稀疏 KAN")
    lines = ["# phonons：16 个物理描述符、4 个输出特征的内层试验", "",
        f"smoke={args.smoke}；完成 {len(records)} 次训练；仅使用外层第 1 折的 1012 个训练样本及其中五个内层划分。",
        "每模型两个学习率、两个配对种子（smoke 仅一个）。每个内层折重新按原 MODNet 方法筛选 190 个特征；"
        "16 个物理量按先验固定，新增物理输入去重；缩放仅拟合内层训练数据。", "",
        "所有模型使用同一 MODNet 96/72/36/36 主干、原尺度 MAE、Adam、最多 800 epochs，监视训练 MAE 早停。"
        "这是重新训练的试验对照，不是复用此前 50 成员官方 benchmark 分数。", "",
        "| 模型 | 学习率 | 选中成员验证 MAE | 两种子集成验证 MAE | 平均参数量 | 模块参数量 | 搜索累计分钟 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {labels[row['variant']]} | {row['selected_lr']} | {row['selected_member_validation_mae']:.6f} | "
                     f"{row['two_seed_ensemble_validation_mae']:.6f} | {row['parameters_mean']:.0f} | {row['module_parameters']} | {row['search_member_minutes']:.2f} |")
    lines += ["", "样条与 KAN 具有相同的分组加性函数类和二阶系数差分平滑项；KAN 额外增加软边/组稀疏惩罚。"
        "因此二者差异检验的是该稀疏约束，不证明 KAN 独有的交互表达能力。四组各输出一个特征；组间交互由后续主干完成。", "",
        "这是用于选择学习率的内层验证结果，不是独立外层测试，不能直接与正式五折 benchmark 相减。"
        "两个种子只能初步检查初始化稳定性，也不构成大量独立重复。软稀疏惩罚不保证精确剪枝。", "",
        "| 内层折 | MODNet | 相同物理输入 | 分组 MLP | 样条 | 分组稀疏 KAN |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for inner in range(5):
        lines.append("| " + str(inner+1) + " | " + " | ".join(f"{r['fold_ensemble_mae'][inner]:.6f}" for r in rows) + " |")
    (args.output / "pilot-report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-jobs", default=2, type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.baseline, args.output = args.baseline.resolve(), args.output.resolve()
    if args.output == args.baseline:
        raise ValueError("Use a separate pilot directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output / "pilot-status.json"
    try:
        write_json(status, dict(state="running", phase="prepare_inner_training_inputs", pid=os.getpid(), started_at=now()))
        protocol, fingerprint, manifests = prepare(args)
        jobs = [dict(variant=variant, lr=lr, seed_repeat=repeat, seed=seed+info["inner"], inputs=info,
                max_epochs=protocol["max_epochs"], run_fingerprint=fingerprint,
                directory=str(args.output / variant / f"lr-{lr}-seed-{repeat}-inner-{info['inner']}"))
                for variant in VARIANTS for lr in protocol["learning_rates"]
                for repeat, seed in enumerate(protocol["seeds"]) for info in manifests]
        records=[]
        with SafePool(args.n_jobs) as pool:
            for done, record in enumerate(pool.imap_unordered(train_job, jobs), 1):
                records.append(record)
                write_json(status, dict(state="running", phase="inner_pilot", pid=os.getpid(), completed=done,
                    total=len(jobs), last_variant=record["variant"], last_validation_mae=record["validation_mae"], updated_at=now()))
                print("FIT",done,"/",len(jobs),record["variant"],record["lr"],record["seed_repeat"],record["inner"],record["validation_mae"],flush=True)
        summary = finish(args, protocol, records, manifests)
        write_json(status, dict(state="complete", completed_at=now(), completed=len(records), total=len(jobs),
                                outer_test_accessed=False, smoke=args.smoke))
        print("COMPLETE", args.output / "pilot-summary.json", flush=True)
    except Exception:
        write_json(status, dict(state="failed", pid=os.getpid(), failed_at=now(), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
