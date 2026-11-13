"""Resumable MODNet 0.1.12 GA reproduction and paired TensorFlow KAN ablations.

Run inside environment-modnet-v012.yml. The unmodified upstream GA implements
population evolution/selection; this adapter checkpoints individual evaluations
and stores lazy model references instead of thousands of live TF graphs.
Public v0.4.0 does not publish seeds/fold hyperparameters. We use v0.1.12's
matbench_benchmark GA defaults, record that limitation, and never tune on tests.
"""
from __future__ import annotations

import os
from modnet_runtime import configure_official_runtime
configure_official_runtime()
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import dataclass
import gc
import gzip
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
from pathlib import Path
import pickle
import random
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf
from modnet.hyper_opt.fit_genetic import FitGenetic
from modnet.matbench.benchmark import matbench_kfold_splits
from modnet.models import EnsembleMODNetModel, MODNetModel
from modnet.preprocessing import MODData

from official_kan_models import fit_kan, neurons_from_genes

TASKS = ("matbench_steels", "matbench_phonons", "matbench_dielectric")
VARIANTS = ("modnet", "kan_head", "kan_full")
STATUS_PATH = None
GA_SETTINGS = dict(size_pop=20, num_generations=10, nested=5,
                   early_stopping=4, refit=False)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, default=jsonable, indent=2,
                                    allow_nan=False), encoding="utf-8")
    replace_with_retry(temporary, path)


def replace_with_retry(source, destination):
    # Windows readers/antivirus can briefly deny delete-sharing during rename.
    # Retrying an atomic replace preserves the last complete checkpoint.
    for attempt in range(20):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.02 * (attempt + 1), 0.2))


def digest(value):
    return hashlib.sha256(json.dumps(value, default=jsonable,
                                     sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def sample_ids(data):
    return list(map(str, data.df_targets.index))


@dataclass
class ModelReference:
    """Metadata-only stand-in; upstream GA never predicts during selection."""
    path: str
    targets: list
    weights: dict
    num_classes: dict
    out_act: str = "linear"


def reference_from_job(job):
    ind = job["individual"]
    return ModelReference(str(Path(job["directory"]) / "model.pkl"),
                          ind.targets, ind.weights, ind.num_classes)


def _evaluate_job(job):
    """No outer test samples/targets enter a training worker."""
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    train, val = job["train_data"], job["val_data"]
    identity = dict(variant=job["variant"], seed=job["seed"],
                    genes=job["individual"].genes,
                    train_ids=sample_ids(train), validation_ids=sample_ids(val),
                    features=list(train.get_optimal_descriptors()),
                    run_fingerprint=job["run_fingerprint"], fast=job["fast"])
    signature = digest(identity)
    result_path = directory / "result.json"
    if result_path.exists() and (directory / "model.pkl").exists():
        result = json.loads(result_path.read_text())
        if result["signature"] != signature:
            raise RuntimeError(f"Incompatible cached member: {directory}")
        return result

    tf.keras.backend.clear_session()
    set_seed(job["seed"])
    individual = copy.deepcopy(job["individual"])
    # Parents carry a lazy best-model pointer, which must not affect a new fit.
    individual.model = None
    started = time.perf_counter()
    write_json(directory / "status.json", dict(state="training", **identity))
    original_fit = MODNetModel.fit

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 25 == 0:
                write_json(directory / "status.json", dict(
                    state="training", epoch=epoch + 1, logs=logs or {},
                    elapsed_seconds=time.perf_counter() - started))

    def seeded_fit(model, *args, **kwargs):
        # KAN initialization consumes additional random draws. Reset before fit
        # so matched members use the same minibatch shuffle seed.
        set_seed(job["seed"])
        kwargs["callbacks"] = list(kwargs.get("callbacks") or []) + [Progress()]
        return original_fit(model, *args, **kwargs)

    with (directory / "training.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            MODNetModel.fit = seeded_fit
            try:
                if job["variant"] == "modnet":
                    individual.evaluate(train, val, fast=job["fast"])
                else:
                    fit_kan(individual, train, val, job["variant"], fast=job["fast"])
            finally:
                MODNetModel.fit = original_fit
    model = individual.model
    if not np.isfinite(individual.val_loss):
        raise RuntimeError(f"Nonfinite validation MAE: {directory}")
    result = dict(
        **identity, signature=signature, validation_mae=float(individual.val_loss),
        parameters=int(model.model.count_params()),
        num_neurons=neurons_from_genes(individual.genes),
        epochs=len(model.history["loss"]), seconds=time.perf_counter() - started,
        selected_features=list(model.optimal_descriptors[:model.n_feat]),
        history=model.history, directory=str(directory))
    model._make_picklable()
    temporary = directory / "model.pkl.tmp"
    with temporary.open("wb") as stream:
        pickle.dump(model, stream, protocol=4)
    replace_with_retry(temporary, directory / "model.pkl")
    write_json(result_path, result)
    write_json(directory / "status.json", dict(state="complete", seconds=result["seconds"]))
    del model, individual
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def evaluate_job(job):
    try:
        return _evaluate_job(job)
    except Exception:
        write_json(Path(job["directory"]) / "status.json",
                   dict(state="failed", traceback=traceback.format_exc()))
        raise


class SafePool:
    """Surface native worker exits instead of silently waiting forever."""

    def __init__(self, n_jobs):
        self.executor = ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn"))

    def imap_unordered(self, function, tasks):
        futures = [self.executor.submit(function, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()

    def close(self):
        self.executor.shutdown(wait=True)

    def join(self):
        pass

    def terminate(self):
        # Python 3.8 has no public terminate_workers(). These are exclusively
        # this executor's child processes; never touch unrelated Python jobs.
        for process in list((getattr(self.executor, "_processes", None) or {}).values()):
            if process.is_alive():
                process.terminate()
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.terminate() if exc_type else self.close()


class CheckpointGA(FitGenetic):
    """Same official GA; stable worker seeds and resumable evaluations."""

    def __init__(self, data, directory, seed, run_fingerprint, **kwargs):
        super().__init__(data, **kwargs)
        self.directory = Path(directory)
        self.seed = seed
        self.run_fingerprint = run_fingerprint
        self.generation = 0
        self.job_by_path = {}
        self.inner_data = {}

    def _init_run(self, n_jobs=None):
        self.pool = SafePool(n_jobs or 4)

    def function_fitness(self, pop, n_jobs, nested=5, val_fraction=0.1,
                         multi_label=False, fast=False):
        if multi_label or not nested:
            raise ValueError("This benchmark requires five-fold regression CV.")
        splits = list(matbench_kfold_splits(self.train_data, n_splits=nested))
        if nested not in self.inner_data:
            self.inner_data[nested] = [self.train_data.split(split) for split in splits]
        jobs = []
        for candidate, individual in enumerate(pop):
            for inner, split in enumerate(splits):
                train, val = self.inner_data[nested][inner]
                directory = self.directory / "modnet" / (
                    f"generation-{self.generation:02d}-candidate-{candidate:02d}-inner-{inner}")
                # Does not consume the parent's GA random stream.
                seed = self.seed + self.generation * 10000 + candidate * 100 + inner
                job = dict(individual=individual, train_data=train, val_data=val,
                           directory=str(directory), seed=seed, fast=fast,
                           variant="modnet", run_fingerprint=self.run_fingerprint)
                jobs.append(job)
                self.job_by_path[str(directory / "model.pkl")] = job
        losses = np.empty((len(pop), nested))
        by_directory = {job["directory"]: i for i, job in enumerate(jobs)}
        for done, result in enumerate(self.pool.imap_unordered(evaluate_job, jobs), 1):
            index = by_directory[result["directory"]]
            candidate, inner = divmod(index, nested)
            losses[candidate, inner] = result["validation_mae"]
            print(f"GA generation {self.generation}: {done}/{len(jobs)}; "
                  f"candidate={candidate}, inner={inner}, "
                  f"MAE={result['validation_mae']:.5g}, epochs={result['epochs']}, "
                  f"seconds={result['seconds']:.1f}", flush=True)
            write_json(self.directory / "progress.json", dict(
                stage="official_ga", generation=self.generation, completed=done,
                total=len(jobs), last_member=result["directory"]))
        models, individuals = [], []
        for candidate, source in enumerate(pop):
            refs = [reference_from_job(jobs[candidate * nested + inner])
                    for inner in range(nested)]
            models.append(EnsembleMODNetModel(modnet_models=refs))
            individual = copy.copy(source)
            individual.model = refs[-1]
            individual.val_loss = losses[candidate, -1]
            individuals.append(individual)
        write_json(self.directory / f"generation-{self.generation:02d}.json", dict(
            genes=[individual.genes for individual in pop],
            validation_mae=losses, mean_validation_mae=losses.mean(axis=1)))
        self.generation += 1
        return losses.mean(axis=1), np.array(models), np.array(individuals)


def official_ids(task, positions, n_samples):
    prefix = task.replace("matbench_", "mb-")
    return [f"{prefix}-{int(i) + 1:0{len(str(n_samples))}d}" for i in positions]


def verify_reference(task, data, reference, task_dir):
    records = reference["tasks"][task]["results"]
    rows = []
    for fold, (_, test) in enumerate(matbench_kfold_splits(data)):
        ids = official_ids(task, test, len(data.df_targets))
        record = records[f"fold_{fold}"]
        if set(ids) != set(record["data"]):
            raise RuntimeError(f"Official outer split mismatch: {task}/{fold}")
        y = data.df_targets.iloc[test, 0].values
        predictions = np.array([record["data"][sid] for sid in ids])
        mae = float(np.mean(np.abs(y - predictions)))
        np.testing.assert_allclose(mae, record["scores"]["mae"], rtol=0, atol=1e-10)
        rows.append(dict(fold=fold, mae=mae, test_ids=ids, test_positions=test))
    write_json(task_dir / "published-reference-verified.json", dict(
        task=task, folds=rows, mean_mae=np.mean([r["mae"] for r in rows]),
        meaning="Recomputed published predictions; not a new trained reproduction."))
    return rows


def predict_members(jobs, test_data, directory, variant):
    predictions, raw_predictions = [], []
    for member, job in enumerate(jobs):
        model_path = Path(job["directory"]) / "model.pkl"
        with model_path.open("rb") as stream:
            model = pickle.load(stream)
        model._restore_model()
        set_seed(job["seed"] + 100000000)
        # Diagnostic only: reproduce the native prediction input transform to
        # retain predictions before MODNet's historical random outlier repair.
        x = test_data.get_featurized_df().replace([np.inf, -np.inf, np.nan], 0)[
            model.optimal_descriptors[:model.n_feat]].values
        x = np.nan_to_num(x)
        if model._scaler is not None:
            x = np.nan_to_num(model._scaler.transform(x), nan=-1)
        raw = np.asarray(model.model.predict(x, verbose=0)).reshape(-1)
        raw_predictions.append(raw)
        with contextlib.redirect_stdout(io.StringIO()):
            pred = model.predict(test_data).values[:, 0]
        if not np.isfinite(pred).all():
            raise RuntimeError(f"Nonfinite outer predictions: {model_path}")
        predictions.append(pred)
        del model
        tf.keras.backend.clear_session()
    predictions = np.asarray(predictions)
    raw_predictions = np.asarray(raw_predictions)
    replaced = np.sum(predictions != raw_predictions, axis=1)
    np.savez_compressed(directory / f"{variant}-member-predictions.npz",
                        predictions=predictions, raw_predictions=raw_predictions,
                        replaced_counts=replaced)
    return predictions.mean(axis=0), predictions.std(axis=0), int(replaced.sum())


def run_fold(data, task, fold, split, args, fingerprint, reference_row, phase):
    directory = args.output_dir / task / f"fold-{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    completion = directory / ("baseline.json" if phase == "baseline" else "comparison.json")
    if completion.exists():
        previous = json.loads(completion.read_text())
        if previous["run_fingerprint"] != fingerprint:
            raise RuntimeError(f"Changed protocol in {directory}")
        return previous
    fold_seed = args.seed + TASKS.index(task) * 1000000 + fold * 100000
    set_seed(fold_seed)
    train, test = data.split(split)
    cache = directory / "official-train-moddata.pkl.gz"
    if cache.exists():
        selected_train = MODData.load(cache)
        assert sample_ids(selected_train) == sample_ids(train)
        np.testing.assert_array_equal(selected_train.df_targets.values, train.df_targets.values)
        pd.testing.assert_frame_equal(selected_train.df_featurized, train.df_featurized)
        train = selected_train
    else:
        train.feature_selection(n=-1, use_precomputed_cross_nmi=True, n_jobs=args.n_jobs)
        train.save(cache)
    write_json(directory / "outer-split.json", dict(
        train_ids=sample_ids(train), test_ids=sample_ids(test),
        train_positions=split[0], test_positions=split[1],
        official_test_ids=reference_row["test_ids"],
        selected_features=list(train.get_optimal_descriptors()),
        feature_selection_scope="outer_train; reused in inner CV, as in MODNet 0.1.12",
        training_data_sha256=file_hash(cache)))
    # Reset after feature selection so resume and first run have the same GA RNG.
    set_seed(fold_seed)
    member_fingerprint = digest(dict(protocol=fingerprint, training_data=file_hash(cache)))
    ga = CheckpointGA(train, directory, fold_seed, member_fingerprint)
    try:
        ensemble = ga.run(n_jobs=args.n_jobs, fast=args.smoke, **GA_SETTINGS)
    except BaseException:
        if getattr(ga, "pool", None):
            ga.pool.terminate()
            ga.pool.join()
        raise
    selected_jobs = [ga.job_by_path[ref.path] for ref in ensemble.model]
    expected = 8 if args.smoke else 50  # fast: 2 candidates x 2 generations x 2 folds
    if len(selected_jobs) != expected:
        raise RuntimeError(f"Unexpected ensemble size {len(selected_jobs)} != {expected}")
    write_json(directory / "frozen-ensemble.json", dict(
        ensemble_size=len(selected_jobs), best_genes=ga.results,
        members=[dict(directory=j["directory"], seed=j["seed"],
                      genes=j["individual"].genes,
                      train_ids=sample_ids(j["train_data"]),
                      validation_ids=sample_ids(j["val_data"])) for j in selected_jobs],
        selection="top 10 candidates by mean inner-fold validation MAE, all 5 inner models"))
    results = {}
    frame = pd.DataFrame(dict(sample_id=reference_row["test_ids"],
                              target=test.df_targets.iloc[:, 0].values))
    for variant in (("modnet",) if phase == "baseline" else VARIANTS):
        if variant == "modnet":
            jobs = selected_jobs
        else:
            jobs = []
            for source in selected_jobs:
                job = dict(source)
                job["variant"] = variant
                job["directory"] = str(directory / variant / Path(source["directory"]).name)
                jobs.append(job)
            with SafePool(args.n_jobs) as pool:
                for done, result in enumerate(pool.imap_unordered(evaluate_job, jobs), 1):
                    print(f"{task} fold={fold} {variant}: {done}/{len(jobs)}, "
                          f"MAE(val)={result['validation_mae']:.5g}", flush=True)
                    write_json(directory / "progress.json", dict(
                        stage=variant, completed=done, total=len(jobs)))
        pred, std, replaced = predict_members(jobs, test, directory, variant)
        frame[variant] = pred
        frame[variant + "_std"] = std
        mae = float(np.abs(frame.target.values - pred).mean())
        member_stats = [json.loads((Path(j["directory"]) / "result.json").read_text())
                        for j in jobs]
        results[variant] = dict(mae=mae, ensemble_size=len(jobs),
            postprocess_replaced_member_predictions=replaced,
            parameters_sum=sum(m["parameters"] for m in member_stats),
            parameters_mean=np.mean([m["parameters"] for m in member_stats]),
            training_seconds_sum=sum(m["seconds"] for m in member_stats))
        frame.to_csv(directory / "predictions.csv", index=False, float_format="%.17g")
        write_json(directory / f"{variant}-score.json", results[variant])
    result = dict(task=task, fold=fold, run_fingerprint=fingerprint, smoke=args.smoke, phase=phase,
                  published_modnet_mae=reference_row["mae"], results=results)
    write_json(completion, result)
    return result


def check_alignment(args):
    rows = {}
    for task in TASKS:
        baseline = []
        for fold in range(5):
            path = args.output_dir / task / f"fold-{fold}" / "baseline.json"
            if not path.exists():
                return dict(passed=False, reason="All 15 baseline folds are required first.")
            record = json.loads(path.read_text())
            if record["smoke"]:
                raise RuntimeError("Smoke results cannot establish baseline alignment.")
            baseline.append(record)
        published = float(np.mean([r["published_modnet_mae"] for r in baseline]))
        trained = float(np.mean([r["results"]["modnet"]["mae"] for r in baseline]))
        deviation = (trained - published) / published
        rows[task] = dict(published_mae=published, retrained_mae=trained,
                          relative_deviation=deviation,
                          within_tolerance=abs(deviation) <= args.alignment_tolerance)
    return dict(tasks=rows, passed=all(r["within_tolerance"] for r in rows.values()),
                tolerance=args.alignment_tolerance,
                meaning="Predeclared diagnostic gate, not statistical equivalence or exact reproduction.")


def protocol(args):
    packages = {name: importlib.metadata.version(name) for name in (
        "modnet", "tensorflow", "numpy", "pandas", "scikit-learn", "matminer", "pymatgen")}
    if packages["modnet"] != "0.1.12":
        raise RuntimeError("Requires modnet==0.1.12")
    import inspect
    import modnet.models.vanilla as vanilla
    import modnet.hyper_opt.fit_genetic as genetic
    import modnet.preprocessing as preprocessing
    import modnet.models.ensemble as ensemble_module
    source_files = [Path(__file__), Path(__file__).with_name("official_kan_models.py"),
                    Path(__file__).with_name("modnet_runtime.py"),
                    Path(inspect.getfile(vanilla)), Path(inspect.getfile(genetic)),
                    Path(inspect.getfile(preprocessing)), Path(inspect.getfile(ensemble_module)),
                    Path(__file__).with_name("verify_modnet_matched_run.py")]
    return dict(
        packages=packages, seed=args.seed, smoke=args.smoke,
        source_sha256={str(p): file_hash(p) for p in source_files},
        ga=GA_SETTINGS, outer_folds=5, outer_split_seed=18012019,
        training=dict(epochs=800, early_stopping_monitor="loss", patience=30,
                      min_delta=0.001, restore_best_weights=False,
                      loss="mae", optimizer="tf.keras.optimizers.Adam", target_scaling=None,
                      seeds="same member seed before initialization and again before fit"),
        kan=dict(implementation="TensorFlow FastKAN", grids=3, grid_range=[-2, 2],
                 layernorm_epsilon=1e-5, base="Dense(SiLU(x)), GlorotUniform",
                 rbf_weights="Normal(0, 0.1)", same_widths=True,
                 head="replace last Dense only; same initial trunk weights",
                 full="replace all five Dense layers; no extra ELU",
                 tuning="frozen MODNet-selected configurations, no KAN-specific search"),
        published_reproduction_limits=[
            "v0.4.0 release says all tasks use GA, but its run script still uses old grid defaults.",
            "Published fold hyperparameters, random seeds and trained weights are not provided.",
            "Use installed v0.1.12 matbench GA defaults; numerical alignment must be measured.",
            "Published-score recomputation and new model training are reported separately."],
        sources=[
            "https://github.com/modl-uclouvain/modnet-matbench/releases/tag/v0.4.0",
            "https://github.com/materialsproject/matbench/tree/main/benchmarks/matbench_v0.1_modnet_v0.1.12"])


def main():
    global STATUS_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--phase", choices=("all", "baseline", "kan"), default="all")
    parser.add_argument("--alignment-tolerance", type=float, default=0.10,
                        help="Absolute relative MAE deviation gate before KAN; not an equivalence test.")
    parser.add_argument("--smoke", action="store_true", help="Separate directory; never report as benchmark.")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    STATUS_PATH = args.output_dir / "matched-status.json"
    args.cache_dir = args.cache_dir.resolve()
    args.tasks = [t if t.startswith("matbench_") else "matbench_" + t for t in args.tasks]
    if set(args.tasks) - set(TASKS) or set(args.folds) - set(range(5)):
        parser.error("Only steels, phonons, dielectric and folds 0..4 are supported.")
    spec = protocol(args)
    spec["baseline_alignment_tolerance"] = args.alignment_tolerance
    spec["reference_sha256"] = file_hash(args.reference)
    fingerprint = digest(spec)
    manifest = args.output_dir / "matched-protocol.json"
    if manifest.exists() and digest(json.loads(manifest.read_text())) != fingerprint:
        raise RuntimeError("Protocol changed; use a new output directory.")
    write_json(manifest, spec)
    for source in spec["source_sha256"]:
        destination = args.output_dir / "source-snapshot" / Path(source).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        if file_hash(destination) != spec["source_sha256"][source]:
            raise RuntimeError(f"Source snapshot mismatch: {destination}")
    with gzip.open(args.reference, "rt") as stream:
        reference = json.load(stream)
    completed = []
    phases = ("baseline", "kan") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "kan" and not args.smoke:
            alignment = check_alignment(args)
            write_json(args.output_dir / "baseline-alignment.json", alignment)
            if not alignment["passed"]:
                write_json(args.output_dir / "matched-status.json", dict(
                    state="baseline_alignment_requires_investigation", **alignment))
                print("Baseline alignment gate failed; investigate before KAN comparison.", flush=True)
                raise SystemExit(2)
        for task in args.tasks:
            data_path = args.cache_dir / f"{task}_moddata.pkl.gz"
            identity_path = args.output_dir / task / "dataset-identity.json"
            identity = dict(path=str(data_path), sha256=file_hash(data_path))
            if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
                raise RuntimeError(f"Dataset cache changed: {data_path}")
            write_json(identity_path, identity)
            data = MODData.load(data_path)
            references = verify_reference(task, data, reference, args.output_dir / task)
            splits = list(matbench_kfold_splits(data))
            for fold in args.folds:
                write_json(args.output_dir / "matched-status.json", dict(
                    state="running", pid=os.getpid(), task=task, fold=fold, phase=phase, smoke=args.smoke))
                completed.append(run_fold(data, task, fold, splits[fold], args,
                                          fingerprint, references[fold], phase))
                write_json(args.output_dir / f"completed-{phase}-folds.json", completed)
                gc.collect()
    if "kan" in phases:
        from verify_modnet_matched_run import verify
        verify(args.output_dir, allow_smoke=args.smoke)
    write_json(args.output_dir / "matched-status.json", dict(
        state="complete", tasks=args.tasks, folds=args.folds, smoke=args.smoke,
        phases=phases, completed_fold_count=len(completed)))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if STATUS_PATH is not None:
            write_json(STATUS_PATH, dict(state="failed", pid=os.getpid(),
                                        traceback=traceback.format_exc()))
        traceback.print_exc()
        raise
