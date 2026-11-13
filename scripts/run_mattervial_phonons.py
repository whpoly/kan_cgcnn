"""Audited MatterVial phonons reproduction and strictly nested paired controls.

Run with the isolated MODNet 0.1.13 environment. Published SISSO features are
diagnostic only; the primary comparison excludes the entire SISSO feature bank.
"""
from __future__ import annotations

from run_modnet_official_matched import (CheckpointGA, SafePool, digest, file_hash,
    set_seed, sample_ids, write_json, predict_members)
from official_kan_models import neurons_from_genes
from physical_feature_modules import PHYSICAL_FEATURES, PHYSICAL_GROUPS, GroupFeatures
from modnet.preprocessing import MODData
from modnet.matbench.benchmark import matbench_kfold_splits
from modnet.models import MODNetModel
import argparse
import contextlib
import copy
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import time
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "benchmarks/mattervial-phonons-20261023"
CSV = STUDY / "source/extracted/featurized_datasets/matbench_phonons_featurizedMM2020Struct_mattervial.csv"
CSV_SHA = "6369c2322436adfaa29834b3e456192c73eab3d7da159f14c0a6102d91d0b535"
ORIGINAL = ROOT / "benchmarks/official-aligned-20261021/precomputed/matbench_phonons_moddata.pkl.gz"
VARIANTS = ("modnet", "group_mlp", "group_sparse_kan")
PHASES = ("paper_features_sisso_unverified", "no_sisso")
LR_FACTORS = (.5, 1., 2.)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def now():
    return datetime.now(timezone.utc).isoformat()


def feature_group(name):
    if "|" in name and not name.startswith("SISSO"):
        return "matminer"
    for prefix, group in (("SISSO", "sisso"), ("MEGNet_OFMEncoded_v1_", "megnet_ofm"),
            ("MVL16_", "mvl_all"), ("MVL32_", "mvl_all"),
            ("ROOST_mpgap_", "roost_mpgap"), ("ORB_v3_", "orb_v3")):
        if name.startswith(prefix):
            return group
    return None


def make_data(frame, targets, name="target"):
    # No featurization is done here. Structures are validated separately against
    # our trusted Matbench cache, never reconstructed from a downloaded pickle.
    return MODData(materials=[None] * len(frame), targets=np.asarray(targets),
                   target_names=[name], structure_ids=list(frame.index), df_featurized=frame)


def load_data(audit_dir):
    if file_hash(CSV) != CSV_SHA:
        raise ValueError("Published phonons CSV changed.")
    frame = pd.read_csv(CSV).set_index("material_id").sort_index()
    if len(frame) != 1265 or not frame.index.is_unique:
        raise ValueError("Unexpected phonons sample IDs.")
    original = MODData.load(ORIGINAL)
    # Our original precomputation used MODData's positional id0..id1264.
    # The official benchmark maps position i to mb-phonons-(i+1), zero padded.
    if list(original.df_targets.index) == [f"id{i}" for i in range(1265)]:
        official_ids = [f"mb-phonons-{i + 1:04d}" for i in range(1265)]
        original.df_targets.index = official_ids
        original.df_structure.index = official_ids
    if set(frame.index) != set(original.df_targets.index):
        raise ValueError("Original sample IDs do not match the author CSV.")
    reference_y = original.df_targets.loc[frame.index].iloc[:, 0].values
    np.testing.assert_allclose(frame.target.values, reference_y, rtol=0, atol=1e-8)
    if not np.isfinite(frame.target).all():
        raise ValueError("Invalid targets.")
    audit_path = audit_dir / "dataset-audit.json"
    if not audit_path.exists():
        from pymatgen.core import Structure
        for key, row in frame.iterrows():
            published = Structure.from_dict(json.loads(row.structure))
            trusted = original.df_structure.loc[key].iloc[0]
            if list(published.species) != list(trusted.species):
                raise ValueError("Species mismatch: " + key)
            np.testing.assert_allclose(published.lattice.matrix, trusted.lattice.matrix, rtol=0, atol=1e-8)
            np.testing.assert_allclose(published.frac_coords, trusted.frac_coords, rtol=0, atol=1e-8)
        raw = frame.drop(columns=["target", "structure"])
        groups = {g: sorted(c for c in raw if feature_group(c) == g)
                  for g in ("matminer", "megnet_ofm", "mvl_all", "roost_mpgap", "sisso", "orb_v3")}
        if set(PHYSICAL_FEATURES) - set(raw.columns):
            raise ValueError("Missing fixed physical features.")
        write_json(audit_path, dict(csv=str(CSV), csv_sha256=CSV_SHA,
            trusted_reference_sha256=file_hash(ORIGINAL), samples=len(frame), total_features=len(raw.columns),
            ids_targets_and_all_structures_verified=True, target_unit="cm^-1; argmax phonon DOS",
            feature_group_counts={g: len(v) for g, v in groups.items()}, feature_groups=groups,
            excluded_columns=sorted(c for c in raw if feature_group(c) is None),
            nan_values=int(raw.isna().sum().sum()), infinity_values=int(np.isinf(raw.values).sum()),
            missing_value_policy="fixed zero replacement, no data-dependent fitting",
            frozen_pretrained_feature_cost="unknown; not counted as zero"))
    elif read(audit_path)["trusted_reference_sha256"] != file_hash(ORIGINAL):
        raise ValueError("Reference audit changed.")
    return frame


def xgb_rfe(train_x, train_y, count, log_path, threads=1):
    """Only training rows/targets are accepted; same 10% RFE rule as author code."""
    columns = list(train_x.columns)
    records = []
    started = time.perf_counter()
    while len(columns) > count:
        model = XGBRegressor(n_jobs=threads, random_state=1, objective="reg:squarederror")
        model.fit(train_x[columns].values, np.asarray(train_y).reshape(-1))
        # Matches the author's np.argsort tie behavior; package versions pinned.
        drop = min(max(1, int(len(columns) * .1)), len(columns) - count)
        removed = set(np.asarray(columns)[np.argsort(model.feature_importances_)[:drop]])
        columns = [c for c in columns if c not in removed]
        records.append(dict(remaining=len(columns), removed=sorted(removed)))
        write_json(log_path, dict(state="rfe", fit_ids=list(train_x.index),
            iterations=records, seconds=time.perf_counter() - started))
    return columns


def prepare_inner(train, validation, directory, fingerprint, seed, count, threads):
    """RFE, feature-feature MI and target MI are all fitted inside inner training."""
    directory.mkdir(parents=True, exist_ok=True)
    expected = dict(run_fingerprint=fingerprint, train_ids=sample_ids(train),
                    validation_ids=sample_ids(validation), count=count, seed=seed)
    train_path, val_path = directory / "train.pkl.gz", directory / "validation.pkl.gz"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["identity"] != digest(expected):
            raise ValueError("Changed inner selection identity.")
        for p, key in ((train_path, "train_sha256"), (val_path, "validation_sha256")):
            if file_hash(p) != manifest[key]:
                raise ValueError("Changed inner selection data: " + str(p))
        return MODData.load(train_path), MODData.load(val_path)
    if set(expected["train_ids"]) & set(expected["validation_ids"]):
        raise ValueError("Inner train/validation overlap.")
    started = time.perf_counter()
    set_seed(seed)
    raw = train.df_featurized.copy()
    rfe_path = directory / "rfe-selected.json"
    if rfe_path.exists():
        saved = read(rfe_path)
        if saved["identity"] != digest(expected):
            raise ValueError("Changed RFE scope.")
        selected = saved["features"]
    else:
        selected = xgb_rfe(raw, train.df_targets.values, count, directory / "rfe-progress.json", threads)
        write_json(rfe_path, dict(identity=digest(expected), features=selected))
    train.df_featurized = raw[selected].copy()
    train.cross_nmi = None
    with (directory / "feature-selection.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            train.feature_selection(n=-1, use_precomputed_cross_nmi=False, n_jobs=threads)
    features = list(train.get_optimal_descriptors())
    if not features or set(features) - set(selected):
        raise ValueError("Invalid selected features.")
    # Preserve all raw inputs for the fixed physical-feature union in controls.
    train.df_featurized = raw
    validation.optimal_features = features
    train.save(train_path)
    validation.save(val_path)
    write_json(manifest_path, dict(**expected, identity=digest(expected),
        train_path=str(train_path), validation_path=str(val_path),
        train_sha256=file_hash(train_path), validation_sha256=file_hash(val_path),
        feature_selection_fit_ids=sample_ids(train), cross_nmi_fit_ids=sample_ids(train),
        selected_features=features, rfe_features=selected, seconds=time.perf_counter() - started))
    return train, validation


def build_control(variant, features, genes, seed, target_scale):
    """Native MODNet Dense trunk plus an optional exactly 148-parameter module."""
    tf.keras.backend.clear_session()
    set_seed(seed)
    original = MODNetModel(targets=[[["target"]]], weights={"target": 1}, n_feat=len(features),
                          num_neurons=neurons_from_genes(genes), act=genes["act"])
    if variant == "modnet":
        return original.model
    if variant not in VARIANTS:
        raise ValueError(variant)
    inputs = tf.keras.Input(shape=(len(features),), name="shared_inputs")
    extra = GroupFeatures([features.index(c) for c in PHYSICAL_FEATURES], variant,
                          target_scale=target_scale, name="physical_module")(inputs)
    x = tf.keras.layers.Concatenate()([inputs, extra])
    for index, old in enumerate(l for l in original.model.layers if isinstance(l, tf.keras.layers.Dense)):
        new = tf.keras.layers.Dense.from_config(old.get_config())
        x = new(x)
        if index == 0:
            kernel, bias = old.get_weights()
            extended = tf.keras.initializers.GlorotUniform(seed=seed + 109)(new.kernel.shape).numpy()
            extended[:len(features)] = kernel
            new.set_weights([extended, bias])
        else:
            new.set_weights(old.get_weights())
    model = tf.keras.Model(inputs, x)
    if model.get_layer("physical_module").count_params() != 148:
        raise ValueError("Module capacity mismatch.")
    return model


def train_control(job):
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    signature = digest(job)
    result_path = directory / "result.json"
    if result_path.exists():
        saved = read(result_path)
        if saved["signature"] != signature or file_hash(directory / "model.pkl") != saved["model_sha256"]:
            raise ValueError("Changed control checkpoint.")
        return saved
    try:
        inp = read(job["manifest"])
        if digest(inp) != job["manifest_digest"]:
            raise ValueError("Changed shared input manifest.")
        for key in ("train", "validation"):
            if file_hash(inp[key + "_path"]) != inp[key + "_sha256"]:
                raise ValueError("Changed cached training partition.")
        train, val = MODData.load(inp["train_path"]), MODData.load(inp["validation_path"])
        features = inp["selected_features"][:job["genes"]["n_feat"]]
        features += [c for c in PHYSICAL_FEATURES if c not in features]
        scaler = MinMaxScaler(feature_range=(-.5, .5))
        x = scaler.fit_transform(train.df_featurized[features]).astype("float32")
        xv = scaler.transform(val.df_featurized[features]).astype("float32")
        y, yv = train.df_targets.values[:, 0], val.df_targets.values[:, 0]
        started = time.perf_counter()
        model = build_control(job["variant"], features, job["genes"], job["seed"], float(np.std(y)))
        set_seed(job["seed"])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=job["lr"]), loss="mae", metrics=["mae"])
        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 25 == 0:
                    write_json(directory / "status.json", dict(state="training", epoch=epoch + 1,
                        seconds=time.perf_counter() - started, metrics=logs))
        with (directory / "training.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                history = model.fit(x, y, validation_data=(xv, yv), epochs=job["epochs"],
                    batch_size=job["genes"]["batch_size"], verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="mae", min_delta=.001,
                        patience=30, restore_best_weights=False), Progress()])
        prediction = model.predict(xv, verbose=0).reshape(-1)
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite control prediction.")
        with (directory / "model.pkl").open("wb") as stream:
            pickle.dump(dict(json=model.to_json(), weights=model.get_weights(), scaler=scaler,
                             features=features), stream, protocol=4)
        np.savez_compressed(directory / "validation.npz", sample_ids=sample_ids(val), target=yv, predictions=prediction)
        record = dict(**job, signature=signature, parameters=model.count_params(),
            module_parameters=0 if job["variant"] == "modnet" else 148,
            selected_features=features, train_ids=sample_ids(train), validation_ids=sample_ids(val),
            scaling_fit_ids=sample_ids(train), validation_mae=float(np.mean(abs(prediction - yv))),
            epochs_run=len(history.history["loss"]), seconds=time.perf_counter() - started,
            history=history.history, model_sha256=file_hash(directory / "model.pkl"))
        write_json(result_path, record)
        write_json(directory / "status.json", dict(state="complete", seconds=record["seconds"]))
        tf.keras.backend.clear_session()
        gc.collect()
        return record
    except Exception:
        write_json(directory / "status.json", dict(state="failed", traceback=traceback.format_exc()))
        raise


def choose_controls(records, templates, inner_count):
    """Pick one LR for each shared template, using all inner folds, for each model."""
    chosen = {}
    for variant in VARIANTS:
        members, candidates = [], []
        for template in range(templates):
            options = []
            for factor in LR_FACTORS:
                rows = sorted((r for r in records if r["variant"] == variant
                    and r["template"] == template and r["lr_factor"] == factor), key=lambda r: r["inner"])
                if [r["inner"] for r in rows] != list(range(inner_count)):
                    raise ValueError("Cannot select from incomplete inner validation.")
                options.append(dict(template=template, lr_factor=factor,
                    mean_validation_mae=float(np.mean([r["validation_mae"] for r in rows])),
                    members=[r["directory"] for r in rows]))
            winner = min(options, key=lambda r: (r["mean_validation_mae"], r["lr_factor"]))
            members += winner["members"]
            candidates.append(dict(options=options, selected=winner))
        chosen[variant] = dict(candidates=candidates, members=members)
    return chosen


def evaluate_controls(selections, test, directory):
    results = {}
    predictions = pd.DataFrame(dict(sample_id=sample_ids(test), target=test.df_targets.values[:, 0]))
    for variant in VARIANTS:
        vectors, rows = [], []
        for d in selections[variant]["members"]:
            record = read(Path(d) / "result.json")
            if file_hash(Path(d) / "model.pkl") != record["model_sha256"]:
                raise ValueError("Changed frozen model.")
            with (Path(d) / "model.pkl").open("rb") as stream:
                saved = pickle.load(stream)
            model = tf.keras.models.model_from_json(saved["json"])
            model.set_weights(saved["weights"])
            x = saved["scaler"].transform(test.df_featurized[saved["features"]]).astype("float32")
            pred = model.predict(x, verbose=0).reshape(-1)
            if not np.isfinite(pred).all():
                raise ValueError("Nonfinite frozen-model test predictions.")
            vectors.append(pred)
            rows.append(record)
            tf.keras.backend.clear_session()
        matrix = np.asarray(vectors)
        np.savez_compressed(directory / (variant + "-member-predictions.npz"), predictions=matrix,
                            sample_ids=sample_ids(test), target=predictions.target.values)
        predictions[variant] = matrix.mean(axis=0)
        results[variant] = dict(mae=float(np.mean(abs(predictions[variant] - predictions.target))),
            ensemble_size=len(rows), parameters_mean=float(np.mean([r["parameters"] for r in rows])),
            parameters_sum=sum(r["parameters"] for r in rows), selected_training_seconds=sum(r["seconds"] for r in rows))
    if results["group_mlp"]["parameters_sum"] != results["group_sparse_kan"]["parameters_sum"]:
        raise ValueError("Final KAN/MLP ensembles must have exactly equal parameters.")
    predictions.to_csv(directory / "predictions.csv", index=False, float_format="%.17g")
    return results


def controlled_stage(ga, ensemble, test, directory, fingerprint, args, seed):
    selection_path = directory / "frozen-controls.json"
    nested = 2 if args.smoke else 5
    groups = [ensemble.model[i:i + nested] for i in range(0, len(ensemble.model), nested)]
    jobs = []
    for template, refs in enumerate(groups):
        genes = ga.job_by_path[refs[0].path]["individual"].genes
        for inner in range(nested):
            manifest = directory / "inputs" / f"inner-{inner}" / "manifest.json"
            for factor in LR_FACTORS:
                for variant in VARIANTS:
                    jobs.append(dict(directory=str(directory / "controls" / variant /
                        f"template-{template:02d}-factor-{factor:g}-inner-{inner}"),
                        manifest=str(manifest), manifest_digest=digest(read(manifest)), variant=variant,
                        genes=genes, seed=seed + 5000000 + template * 100 + inner, lr=genes["lr"] * factor,
                        lr_factor=factor, inner=inner, template=template, epochs=1 if args.smoke else 800,
                        run_fingerprint=fingerprint))
    records = []
    with SafePool(args.n_jobs) as pool:
        for done, result in enumerate(pool.imap_unordered(train_control, jobs), 1):
            records.append(result)
            write_json(directory / "control-progress.json", dict(completed=done, total=len(jobs),
                       last_member=result["directory"]))
            print("CONTROL", directory.name, done, "/", len(jobs), result["variant"], result["validation_mae"], flush=True)
    # Check exact paired training inputs and model capacity before any test call.
    for template in range(len(groups)):
        for inner in range(nested):
            subset = [r for r in records if r["template"] == template and r["inner"] == inner]
            for key in ("selected_features", "train_ids", "validation_ids", "scaling_fit_ids", "seed"):
                if len({digest(r[key]) for r in subset}) != 1:
                    raise ValueError("Unpaired comparison: " + key)
            module_rows = [r for r in subset if r["variant"] != "modnet"]
            if len({r["parameters"] for r in module_rows}) != 1:
                raise ValueError("Unequal KAN/MLP capacity.")
    selections = choose_controls(records, len(groups), nested)
    write_json(selection_path, dict(run_fingerprint=fingerprint, selections=selections,
        selection_uses="inner validation only; all selections frozen before test evaluation",
        test_labels_used_for_selection=False, paired_features_splits_seeds_and_capacity_verified=True))
    results = evaluate_controls(selections, test, directory)
    for variant in VARIANTS:
        subset = [r for r in records if r["variant"] == variant]
        results[variant].update(search_fits=len(subset), search_seconds=sum(r["seconds"] for r in subset))
    write_json(directory / "comparison.json", dict(run_fingerprint=fingerprint, results=results,
        shared_architecture_discovery="clean MODNet GA; separate shared cost, not equal end-to-end optimization",
        primary_prediction_rule="raw neural output; no random outlier replacement"))


def train_fold(frame, phase, fold, split, args, protocol, fingerprint):
    directory = args.output / phase / f"fold-{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    done_path = directory / "complete.json"
    if done_path.exists():
        if read(done_path)["run_fingerprint"] != fingerprint:
            raise ValueError("Completed fold fingerprint changed.")
        return
    columns = sorted(c for c in frame if feature_group(c) is not None
        and (phase != "no_sisso" or feature_group(c) != "sisso"))
    if args.smoke:
        # A tiny, target-independent mixed feature bank for code-path testing;
        # never report these scores as benchmark results.
        columns = sorted(set(PHYSICAL_FEATURES) | {c for g in set(map(feature_group, columns))
            for c in [f for f in columns if feature_group(f) == g][:8]})
    x = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    data = make_data(x, frame.target)
    train, test = data.split(split)
    outer_ids, test_ids = sample_ids(train), sample_ids(test)
    # Test labels are held out from both feature fitting and all training jobs.
    write_json(directory / "outer-split.json", dict(train_ids=outer_ids, test_ids=test_ids,
        train_positions=split[0], test_positions=split[1], features=columns,
        sisso_status="excluded" if phase == "no_sisso" else "target-derived; outer-exclusion provenance unverified"))
    seed = args.seed + fold * 100000  # Same outer/inner partition and GA seed across feature banks.
    train.optimal_features = columns[:protocol["rfe_count"]]
    set_seed(seed)
    ga = CheckpointGA(train, directory, seed, fingerprint)
    nested = 2 if args.smoke else 5
    inner_splits = list(matbench_kfold_splits(ga.train_data, n_splits=nested))
    rng_python, rng_numpy = random.getstate(), np.random.get_state()
    inner_data, covered = [], []
    for inner, inner_split in enumerate(inner_splits):
        inner_train, validation = ga.train_data.split(inner_split)
        if (set(sample_ids(inner_train)) | set(sample_ids(validation))) != set(outer_ids):
            raise ValueError("Inner rows are not exactly the outer training set.")
        if (set(sample_ids(inner_train)) | set(sample_ids(validation))) & set(test_ids):
            raise ValueError("Outer test contamination.")
        write_json(args.output / "status.json", dict(state="feature_selection", phase=phase, fold=fold,
                   inner=inner, pid=os.getpid(), updated_at=now(), smoke=args.smoke))
        pair = prepare_inner(inner_train, validation, directory / "inputs" / f"inner-{inner}",
            fingerprint, seed + inner, protocol["rfe_count"], args.n_jobs)
        inner_data.append(pair)
        covered += sample_ids(validation)
    if sorted(covered) != sorted(outer_ids):
        raise ValueError("Incomplete inner validation coverage.")
    # Upstream GA consumes a random row permutation in its constructor. Preserve
    # that exact stream across expensive feature preparation and resumed runs.
    random.setstate(rng_python)
    np.random.set_state(rng_numpy)
    ga.inner_data[nested] = inner_data
    max_feat = min(len(pair[0].get_optimal_descriptors()) for pair in inner_data)
    ga.train_data.optimal_features = columns[:max_feat]
    write_json(args.output / "status.json", dict(state="ga_training", phase=phase, fold=fold,
               pid=os.getpid(), updated_at=now(), smoke=args.smoke))
    try:
        ensemble = ga.run(n_jobs=args.n_jobs, fast=args.smoke, **protocol["ga"])
    except BaseException:
        if getattr(ga, "pool", None):
            ga.pool.terminate()
        raise
    selected_jobs = [ga.job_by_path[ref.path] for ref in ensemble.model]
    expected = 8 if args.smoke else 50
    if len(selected_jobs) != expected:
        raise ValueError("Unexpected selected ensemble size.")
    write_json(directory / "frozen-ensemble.json", dict(run_fingerprint=fingerprint,
        members=[dict(directory=j["directory"], seed=j["seed"], genes=j["individual"].genes)
                 for j in selected_jobs], selection="top 10 configurations by mean inner validation; all 5 inner models"))
    # Controls are selected without seeing either the GA baseline or control test
    # scores. The diagnostic paper phase is separate from this clean comparison.
    if phase == "no_sisso":
        write_json(args.output / "status.json", dict(state="controlled_training", phase=phase, fold=fold,
                   pid=os.getpid(), updated_at=now(), smoke=args.smoke))
        controlled_stage(ga, ensemble, test, directory, fingerprint, args, seed)
    prediction, std, replaced = predict_members(selected_jobs, test, directory, "ga_modnet")
    raw = np.load(directory / "ga_modnet-member-predictions.npz")["raw_predictions"].mean(axis=0)
    pd.DataFrame(dict(sample_id=test_ids, target=test.df_targets.values[:, 0],
        prediction=prediction, raw_prediction=raw, std=std)).to_csv(directory / "ga-predictions.csv", index=False)
    rows = [read(Path(j["directory"]) / "result.json") for j in selected_jobs]
    all_rows = [read(p) for p in (directory / "modnet").glob("*/result.json")]
    score = dict(run_fingerprint=fingerprint, phase=phase, fold=fold,
        mae=float(np.mean(abs(prediction - test.df_targets.values[:, 0]))),
        raw_mae=float(np.mean(abs(raw - test.df_targets.values[:, 0]))),
        random_outlier_replacements=replaced, ensemble_size=len(rows),
        parameters_mean=float(np.mean([r["parameters"] for r in rows])),
        parameters_sum=sum(r["parameters"] for r in rows),
        selected_training_seconds=sum(r["seconds"] for r in rows),
        search_fits=len(all_rows), search_seconds=sum(r["seconds"] for r in all_rows),
        feature_selection_seconds=sum(read(directory / "inputs" / f"inner-{i}" / "manifest.json")["seconds"] for i in range(nested)))
    write_json(directory / "ga-score.json", score)
    write_json(done_path, dict(run_fingerprint=fingerprint, completed_at=now()))


def make_protocol(args):
    sources = [Path(__file__), *[Path(__file__).with_name(f) for f in
        ("run_modnet_official_matched.py", "official_kan_models.py", "modnet_runtime.py", "physical_feature_modules.py")]]
    return dict(kind="MatterVial phonons audited reproduction and nested controls", seed=args.seed,
        smoke=args.smoke, csv_sha256=CSV_SHA, outer_split=dict(n_splits=5, shuffle=True, random_state=18012019),
        phases=PHASES, source_sha256={str(p): file_hash(p) for p in sources},
        packages={p: importlib.metadata.version(p) for p in ("modnet", "tensorflow", "xgboost", "numpy", "pandas", "scipy", "scikit-learn")},
        ga=dict(size_pop=50, num_generations=20, nested=5, early_stopping=8, refit=False),
        rfe_count=24 if args.smoke else 800, max_epochs=1 if args.smoke else 800,
        feature_selection_scope="every inner training partition: XGB RFE + cross NMI + target NMI; no global MI cache",
        sisso="included only in the paper-feature diagnostic; entire bank excluded in primary KAN comparison",
        reproduction_limits=["Exact phonons training entrypoint and author RNG seeds unavailable",
            "Inner-fold feature selection is stricter than the available related author script",
            "SISSO formula construction sample IDs/outer exclusion unverified",
            "Frozen external GNN training overlap and versions reported separately; extraction time unknown"],
        controls=dict(variants=VARIANTS, physical_groups=PHYSICAL_GROUPS, physical_outputs=4, module_parameters=148,
            template_source="top 10 clean MODNet GA candidates; shared architecture discovery cost",
            candidates_per_model=150, learning_rate_factors=LR_FACTORS,
            feature_input="same selected core union fixed 16 physical features for every variant",
            scaler="inner-training minmax [-0.5,0.5], fixed for all controlled candidates",
            early_stopping=dict(monitor="mae", patience=30, min_delta=.001, restore_best_weights=False),
            seeds="paired across all variants and LR factors; template-specific, inner-fold-specific",
            selection="per shared template choose LR on mean of all 5 inner validation MAEs; retain 50 members",
            budget_scope="equal controlled-stage fit/epoch caps; shared MODNet architecture search reported separately",
            prediction="raw network output, no random outlier repair",
            sparse_strength=.01, smooth_strength=.001, spline_intervals=5, spline_degree=3,
            penalty_scale="inner-training target standard deviation"),
        numerical_closeness=dict(reference_mae=30.08006079904005, relative_tolerance=.02,
            meaning="descriptive numerical proximity, not equivalence or proof of leakage-free reproduction"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "run")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = make_protocol(args)
    if protocol["packages"]["modnet"] != "0.1.13":
        raise ValueError("Use the isolated MODNet 0.1.13 environment.")
    fingerprint = digest(protocol)
    path = args.output / "protocol.json"
    if path.exists() and digest(read(path)) != fingerprint:
        raise ValueError("Protocol or source code changed; use a new output directory.")
    write_json(path, protocol)
    frame = load_data(STUDY / "audit")
    if args.audit_only:
        return
    splits = list(KFold(n_splits=5, shuffle=True, random_state=18012019).split(frame))
    # Confirm split identity against the published predictions, without using
    # their errors to choose model settings.
    published = pd.read_csv(STUDY / "audit/published_fold_predictions.csv").set_index("material_id")
    for fold, (_, test) in enumerate(splits):
        ids = set(published.index[published[f"predictions_f{fold + 1}"].notna()])
        if ids != set(frame.index[test]):
            raise ValueError("Published outer split mismatch.")
    started = time.perf_counter()
    try:
        for phase in PHASES:
            for fold, split in enumerate(splits[:1] if args.smoke else splits):
                train_fold(frame, phase, fold, split, args, protocol, fingerprint)
        scores = [read(p) for p in args.output.glob("*/fold-*/ga-score.json")]
        comparisons = [read(p) for p in args.output.glob("no_sisso/fold-*/comparison.json")]
        summary = dict(smoke=args.smoke, ga_folds=scores, controlled_folds=comparisons)
        if not args.smoke:
            summary["ga_mean_mae"] = {phase: float(np.mean([r["mae"] for r in scores if r["phase"] == phase])) for phase in PHASES}
            summary["controlled_mean_mae"] = {v: float(np.mean([r["results"][v]["mae"] for r in comparisons])) for v in VARIANTS}
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "status.json", dict(state="complete", smoke=args.smoke, completed_at=now(),
                   current_invocation_wall_seconds=time.perf_counter() - started))
    except BaseException:
        write_json(args.output / "status.json", dict(state="failed", smoke=args.smoke,
                   failed_at=now(), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
