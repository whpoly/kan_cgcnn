"""Read frozen benchmark members and export exact KAN internal contributions.

No fitting, model selection, or outer-test labels. Use the first recorded member
of each complete fold, a reproducible illustrative subset of the full ensemble.
Run with the existing modnet-v013-mattervial Python environment.
"""
from pathlib import Path
import argparse
import gc
import json
import pickle

import run_native_fourway as engine
from run_modnet_official_matched import np, tf, MODData, file_hash, write_json
from standalone_kan_models import SplineEdgeLayer
from physical_feature_modules import cubic_basis

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "phonons": ROOT / "benchmarks/native-fourway-scaled-kan-20261029",
    "steels": ROOT / "benchmarks/matbench-expansion-20261101/matbench_steels",
}
PHYSICAL = [
    "ElementProperty|MagpieData mean AtomicWeight",
    "ElementProperty|MagpieData mean CovalentRadius",
    "AverageBondLength|mean Average bond length",
]


def scaled(model, raw):
    return np.nan_to_num(model._scaler.transform(np.nan_to_num(raw)), nan=-1).astype("float32")


def trajectory(model, raw):
    values = scaled(model, raw)
    layers = []
    for layer in model.model.layers:
        if not isinstance(layer, SplineEdgeLayer):
            continue
        coordinate = .5 * np.tanh(values) if layer.hidden_coordinates else values
        basis = cubic_basis(tf.convert_to_tensor(coordinate)).numpy()
        coefficients = layer.coefficients.numpy()
        edges = np.einsum("nik,iok->nio", basis, coefficients)
        actual = layer(values).numpy()
        reconstructed = edges.sum(axis=1) + layer.bias.numpy()
        np.testing.assert_allclose(reconstructed, actual, atol=3e-5, rtol=2e-5)
        layers.append(dict(coordinate=coordinate, edges=edges, output=actual,
                           coefficients=coefficients, bias=layer.bias.numpy(),
                           hidden=layer.hidden_coordinates, name=layer.name))
        values = actual
    return layers


def inspect_member(task, run, fold, out):
    directory = run / "mm_kan" / f"fold-{fold}"
    selection = engine.read(directory / "frozen-selection.json")
    complete = engine.read(directory / "complete.json")
    assert file_hash(directory / "frozen-selection.json") == complete["selection_sha256"]
    job = selection["members"][0]
    path = Path(job["directory"]) / "model.pkl"
    assert file_hash(path) == selection["model_sha256"][job["directory"]]
    assert file_hash(job["training_cache"]) == job["training_sha256"]
    data = MODData.load(job["training_cache"])
    with path.open("rb") as f:
        model = pickle.load(f)
    model._restore_model()
    columns = list(model.optimal_descriptors[:model.n_feat])
    train = data.df_featurized.loc[job["train_ids"], columns].values
    val = data.df_featurized.loc[job["validation_ids"], columns].values
    assert not set(job["train_ids"]) & set(job["validation_ids"])
    tr = trajectory(model, train)
    va = trajectory(model, val)
    assert len(tr) == 3
    sigma = model.target_scaling["std"]
    mu = model.target_scaling["mean"]
    terminal = tr[-1]["edges"][:, :, 0].astype(float)
    center = terminal.mean(axis=0)
    contributions = sigma * (terminal - center)
    val_contributions = sigma * (va[-1]["edges"][:, :, 0] - center)
    baseline = mu + sigma * (float(tr[-1]["bias"][0]) + center.sum())
    actual = np.asarray(model.model(scaled(model, val), training=False)).reshape(-1)
    reconstructed = baseline + val_contributions.sum(axis=1)
    error = float(np.max(abs(actual - reconstructed)))
    np.testing.assert_allclose(actual, reconstructed, atol=.005, rtol=0)
    importance = np.mean(abs(contributions), axis=0)
    order = np.argsort(-importance)
    share = np.cumsum(importance[order]) / importance.sum()

    # Fit only an affine summary of each first-layer edge on its training
    # support. This diagnostic never replaces the edge or changes predictions.
    first = tr[0]
    edge_sd = first["edges"].std(axis=0)
    affine_relative_error = np.zeros(edge_sd.shape)
    for i in range(len(columns)):
        design = np.column_stack([np.ones(len(train)), first["coordinate"][:, i]])
        weights = np.linalg.lstsq(design, first["edges"][:, i, :], rcond=None)[0]
        residual = first["edges"][:, i, :] - design @ weights
        affine_relative_error[i] = np.sqrt(np.mean(residual ** 2, axis=0)) / np.maximum(edge_sd[i], 1e-10)
    important = edge_sd >= np.quantile(edge_sd, .90)
    feature_energy = np.sqrt(np.mean(edge_sd ** 2, axis=1))
    candidates = [columns.index(f) for f in PHYSICAL if f in columns]
    if task == "steels":
        candidates = list(np.argsort(-feature_energy)[:3])
    examples = []
    for i in candidates:
        j = int(np.argmax(edge_sd[i]))
        domain = np.quantile(np.nan_to_num(train[:, i]), [.05, .95])
        grid = np.linspace(*domain, 120)
        probes = np.tile(np.nanmedian(train, axis=0), (len(grid), 1))
        probes[:, i] = grid
        transformed = scaled(model, probes)[:, i]
        coordinate = .5 * np.tanh(transformed) if first["hidden"] else transformed
        curve = cubic_basis(tf.convert_to_tensor(coordinate)).numpy() @ first["coefficients"][i, j]
        edge_center = float(first["edges"][:, i, j].mean())
        examples.append(dict(feature=columns[i], input_index=i, output_node=j,
            coefficients=first["coefficients"][i, j].tolist(), training_edge_mean=edge_center,
            training_edge_std=float(edge_sd[i, j]), raw_grid=grid.tolist(),
            spline_coordinate=coordinate.tolist(), centered_response=(curve-edge_center).tolist(),
            affine_relative_rmse=float(affine_relative_error[i, j]),
            response_unit="first hidden preactivation; no identified physical unit",
            selection="largest training edge standard deviation for this input"))

    sample = int(np.argsort(np.asarray(job["validation_ids"]))[0])
    sample_record = dict(id=job["validation_ids"][sample],
        selection="lexicographically first inner-validation ID, independent of labels and errors",
        prediction=float(actual[sample]), baseline=float(baseline),
        contributions=val_contributions[sample].tolist(),
        reconstruction_error=float(abs(actual[sample]-reconstructed[sample])))
    destination = out / task / f"fold-{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    arrays = dict(train_last_contributions=contributions,
        validation_last_contributions=val_contributions,
        train_ids=np.asarray(job["train_ids"]), validation_ids=np.asarray(job["validation_ids"]),
        first_layer_edge_sd=edge_sd, first_layer_affine_relative_rmse=affine_relative_error,
        first_hidden_train=tr[0]["output"], second_hidden_train=tr[1]["output"],
        last_coordinates_train=tr[-1]["coordinate"])
    for i, layer in enumerate(tr):
        arrays[f"layer_{i}_coefficients"] = layer["coefficients"]
        arrays[f"layer_{i}_bias"] = layer["bias"]
    np.savez_compressed(destination / "internals.npz", **arrays)
    record = dict(task=task, arm="mm_kan", fold=fold, member_index=0,
        total_members_in_fold=len(selection["members"]), model_path=str(path),
        model_sha256=file_hash(path), selection_sha256=file_hash(directory/"frozen-selection.json"),
        training_sha256=job["training_sha256"], input_features=columns,
        widths=[len(columns)]+[int(x["output"].shape[1]) for x in tr],
        scaler=str(model._scaler), bounded_coordinates=[x["hidden"] for x in tr],
        target_mean=mu, target_std=sigma, contribution_baseline=float(baseline),
        max_validation_reconstruction_error=error,
        mean_abs_last_contribution=importance.tolist(), last_node_training_centers=center.tolist(),
        n_last_terms_for_90pct_abs_magnitude=int(np.searchsorted(share, .9)+1),
        n_last_terms=int(len(importance)),
        first_layer_top_decile_nonlinear_fraction=float(np.mean(affine_relative_error[important]>.1)),
        first_layer_top_decile_median_affine_relative_rmse=float(np.median(affine_relative_error[important])),
        first_layer_examples=examples, sample=sample_record,
        limitations=["A fixed first member per fold, not the whole 50-member ensemble.",
            "Hidden nodes are not aligned between folds; their index has no invariant physical meaning.",
            "Exact terminal contributions are not independent physical or causal contributions.",
            "90 percent absolute contribution magnitude is not explained variance or accuracy retained.",
            "Affine edge approximation diagnoses nonlinearity, not superiority over an MLP.",
            "No outer-test rows or labels were read; inner validation was previously used by the benchmark."],
        internals_sha256=file_hash(destination/"internals.npz"))
    write_json(destination / "summary.json", record)
    print(json.dumps({k:record[k] for k in ["task","fold","widths", "max_validation_reconstruction_error",
        "n_last_terms", "n_last_terms_for_90pct_abs_magnitude", "first_layer_top_decile_nonlinear_fraction"]}), flush=True)
    del model, data, tr, va
    tf.keras.backend.clear_session()
    gc.collect()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"benchmarks/interpretability-review-20261106")
    args = parser.parse_args()
    out = args.output.resolve()
    records = []
    for task, run in RUNS.items():
        audit = engine.read(run/"artifact-audit.json")
        assert audit["passed"] and audit["complete"] and not audit["smoke"]
        for fold in range(5):
            records.append(inspect_member(task, run, fold, out))
    write_json(out/"native-internals.json", dict(complete=True,
        scope="first recorded MM KAN member per outer fold, phonons and steels",
        training_performed=False, outer_test_loaded=False,
        extractor_sha256=file_hash(__file__), records=records))


if __name__ == "__main__":
    main()
