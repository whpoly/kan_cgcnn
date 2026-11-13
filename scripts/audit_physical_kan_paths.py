"""Replay frozen three-input explanations on crossed materials and trace their output paths."""
import argparse
import pickle
from pathlib import Path

import discover_physical_kan_responses as discovery
from scipy.interpolate import BSpline
from run_modnet_official_matched import file_hash, np, tf, write_json


def terminal_values(layers, hidden, model, arm):
    """Return uncentered native output terms and the independent Keras prediction."""
    value = tf.convert_to_tensor(hidden)
    if arm == "kan":
        last_index = next(i for i in reversed(range(len(layers))) if hasattr(layers[i], "coefficients"))
        for layer in layers[1:last_index]:
            value = layer(value, training=False)
        last = layers[last_index]
        spline = discovery.basis(value.numpy(), last.hidden_coordinates)
        terms = np.einsum("nik,ik->ni", spline, last.coefficients.numpy()[:, 0, :]).astype(float)
        terms *= model.target_scaling["std"]
        rest = layers[last_index:]
        offset = model.target_scaling["mean"] + last.bias.numpy()[0] * model.target_scaling["std"]
    else:
        late = max(i for i, layer in enumerate(layers[:-1]) if layer.activation.__name__ != "linear")
        for layer in layers[1:late + 1]:
            value = layer(value, training=False)
        rest = layers[late + 1:]
        effective = np.ones(1)
        offset = 0.
        for layer in reversed(rest):
            assert layer.activation.__name__ == "linear"
            weight, bias = layer.get_weights()
            offset += float(bias @ effective)
            effective = weight @ effective
        terms = value.numpy().astype(float) * effective
    for layer in rest:
        value = layer(value, training=False)
    prediction = value.numpy().reshape(-1).astype(float)
    np.testing.assert_allclose(terms.sum(1) + offset, prediction, atol=.006, rtol=1e-6)
    return terms, prediction


def crossed_check(first, xt, cols, anchor, node, record, recipients, donors, raw_test, h_ref, arm):
    mixed = xt[recipients].copy()
    mixed[:, cols] = xt[donors][:, cols]
    observed = first(mixed, training=False).numpy()[:, node] - h_ref[recipients, node]
    recipe = record["source_recipe"]
    if arm == "kan":
        def evaluate(raw):
            scaled = np.asarray(raw)*recipe["input_multiplier"] + recipe["input_intercept"]
            coordinate = .5*np.tanh(scaled) if recipe["bounded"] else scaled
            at = np.clip(coordinate, *recipe["clipping"])
            return np.stack([BSpline(recipe["spline_knots"], c, 3)(t)
                             for c, t in zip(recipe["coefficients"], at.T)], axis=-1)
        predicted = (evaluate(raw_test[donors]) - evaluate(record["anchor"])).sum(axis=1)
    else:
        weights, bias = first.get_weights()
        shift = (xt[donors][:, cols] - anchor) @ recipe["weights"]
        ref = xt[recipients].copy()
        ref[:, cols] = anchor
        base = (ref @ weights[:, node] + bias[node]).astype(float)
        predicted = discovery.activation(first, base + shift) - discovery.activation(first, base)
    np.testing.assert_allclose(predicted, observed, atol=.002, rtol=1e-5)
    return observed, predicted


def greedy_path(terms, target):
    residual = target.copy()
    selected = []
    for _ in range(min(3, terms.shape[1])):
        gain = 2 * (terms.T @ residual) - np.sum(terms**2, axis=0)
        gain[selected] = -np.inf
        node = int(np.argmax(gain))
        selected.append(node)
        residual -= terms[:, node]
    return selected


def edge_contrasts(layers, record, arm, cols, raw_train, xt, hidden):
    """Propagate each selected edge from its training q25 to q75 on each actual outer background."""
    node = record["node"]
    first = layers[0]
    levels = np.quantile(raw_train, [.25, .75], axis=0)
    recipe = record["source_recipe"]
    scaled_levels = (levels * recipe["input_multiplier"] + recipe["input_intercept"]).astype("float32")
    if arm == "kan":
        weights = first.coefficients.numpy()[cols, node, :]
        def values(s):
            return np.einsum("nib,ib->ni", discovery.basis(s, first.hidden_coordinates), weights)
        changes = values(scaled_levels)[:, None, :] - values(xt[:, cols])[None, :, :]
    else:
        weights, bias = first.get_weights()
        pre = (xt @ weights[:, node] + bias[node]).astype(float)
        shift = (scaled_levels[:, None, :] - xt[:, cols]) * weights[cols, node]
        changes = discovery.activation(first, pre[None, :, None]+shift) - discovery.activation(first, pre)[None, :, None]
    # Axis order is level, input, material, hidden dimension.
    alternatives = np.tile(hidden, (2, len(cols), 1, 1))
    alternatives[:, :, :, node] += changes.transpose(0, 2, 1)
    response = tf.convert_to_tensor(alternatives.reshape(-1, hidden.shape[1]))
    for layer in layers[1:]:
        response = layer(response, training=False)
    output = response.numpy().reshape(2, len(cols), len(xt)).astype(float)
    target_change = (output[1]-output[0]).T
    assert np.isfinite(target_change).all()
    if np.any(levels[0] == levels[1]):
        np.testing.assert_allclose(target_change[:,levels[0] == levels[1]], 0, atol=.002)
    return levels, target_change, changes[1]-changes[0]


def trace(source, record, archive, arm, frame, train_ids, test_ids, names, anchor_id, recipients, donors):
    path, _, sha, _ = source
    assert file_hash(path) == sha
    with path.open("rb") as stream:
        model = pickle.load(stream)
    model._restore_model()
    layers = [layer for layer in model.model.layers if not isinstance(layer, tf.keras.layers.InputLayer)]
    columns = list(model.optimal_descriptors[:model.n_feat])
    cols = [columns.index(name) for name in names]
    raw = frame.loc[train_ids + test_ids, columns].values
    x = discovery.p.scaled(model, raw)
    n = len(train_ids)
    first, node = layers[0], record["node"]
    anchor = x[train_ids.index(anchor_id), cols]
    x_ref = x.copy()
    x_ref[:, cols] = anchor
    h = first(x, training=False).numpy()
    h_ref = first(x_ref, training=False).numpy()
    observed = h[:, node].astype(float) - h_ref[:, node]
    np.testing.assert_allclose(observed[n:], archive[arm+"_component"], atol=.002, rtol=1e-5)
    edited = h.copy()
    edited[:, node] = h_ref[:, node]
    terms, outputs = terminal_values(layers, np.concatenate((h, edited)), model, arm)
    original, altered = np.split(outputs, 2)
    np.testing.assert_allclose(original[n:], archive[arm+"_raw_reference"], atol=.01, rtol=1e-6)
    np.testing.assert_allclose(altered[n:], archive[arm+"_remove"], atol=.01, rtol=1e-6)
    before, after = np.split(terms, 2)
    changes = before - after
    delta = original - altered
    reconstruction = np.max(abs(changes.sum(1)-delta))
    np.testing.assert_allclose(changes.sum(1), delta, atol=.012, rtol=1e-6)
    select = greedy_path(changes[:n], delta[:n])
    explained = changes[n:, select]
    portable = archive[arm+"_portable_component"]
    if arm == "kan":
        portable_train = observed[:n]
    else:
        weights, bias = first.get_weights()
        shift = (x[:n, cols]-anchor) @ np.array(record["source_recipe"]["weights"])
        context = (x[:n] @ weights[:, node] + bias[node]).astype(float) - shift
        portable_train = (discovery.activation(first, context[:, None]+shift)
                          - discovery.activation(first, context)[:, None]).mean(0)
    beta = float(portable_train @ delta[:n] / max(portable_train @ portable_train, 1e-24))
    measured_cross, context_recipe = crossed_check(first, x[n:], cols, anchor, node, record, recipients, donors,
                                                   raw[n:, cols], h_ref[n:], arm)
    levels, edge_output, edge_hidden = edge_contrasts(layers, record, arm, cols, raw[:n, cols], x[n:], h[n:])
    remainder = delta[n:] - explained.sum(1)
    metrics = dict(three_path_error=discovery.rms_ratio(remainder, delta[n:]),
                   single_scale_error=discovery.rms_ratio(beta*portable-delta[n:], delta[n:]),
                   single_scale_mae=np.abs(beta*portable-delta[n:]).mean(),
                   cross_numeric_max=np.max(abs(measured_cross-context_recipe)),
                   cross_portable_error=discovery.rms_ratio(portable[donors]-measured_cross, measured_cross),
                   output_reconstruction_max=reconstruction)
    arrays = dict(output_change=delta[n:], terminal_changes=explained, remainder=remainder,
                  crossed_observed=measured_cross, crossed_recipe=context_recipe,
                  edge_iqr_output_change=edge_output, edge_iqr_hidden_change=edge_hidden)
    return dict(terminal_nodes=select, beta=beta, metrics=metrics, contrast_levels=levels,
                contrast_median=np.median(edge_output, axis=0), contrast_mean_abs=abs(edge_output).mean(0)), arrays


def run(task, folds, out, limit):
    data = discovery.load_data(task, out / task / f"data-check-{folds[0]}")
    source_hash = file_hash(__file__)
    main_hash = file_hash(discovery.__file__)
    for fold in folds:
        sources, _, _, test_ids = discovery.load_fold(task, fold, data)
        folder = out / task / f"fold-{fold}"
        (folder / "paths").mkdir(exist_ok=True)
        for member in range(limit):
            tag = f"member-{member:02d}"
            previous = folder / f"{tag}.json"
            result = folder / "paths" / f"{tag}.json"
            output = result.with_suffix(".npz")
            meta = discovery.steel.read(previous)
            assert meta["analysis_sha256"] == main_hash
            assert meta["arrays_sha256"] == file_hash(previous.with_suffix(".npz"))
            assert meta["model_sha256"] == [s[member][2] for s in sources]
            parent_hash = file_hash(previous)
            if result.exists():
                saved = discovery.steel.read(result)
                if (saved["parent_sha256"] == parent_hash and saved["audit_sha256"] == source_hash
                        and saved["arrays_sha256"] == file_hash(output)):
                    continue
            train_ids = sources[0][member][3]
            order = np.random.RandomState(121126 + 1000*fold + member).permutation(len(test_ids))
            recipients, donors = order[:32], np.roll(order, 1)[:32]
            arrays = dict(ids=test_ids, recipients=recipients, donors=donors)
            result_rows = {}
            with np.load(previous.with_suffix(".npz")) as archive:
                np.testing.assert_array_equal(archive["ids"], test_ids)
                for arm, branch in zip(discovery.ARMS, sources):
                    result_rows[arm], values = trace(branch[member], meta[arm], archive, arm, data[0], train_ids,
                        test_ids, meta["inputs"], meta["selection"]["anchor_id"], recipients, donors)
                    arrays.update({f"{arm}_{key}": v for key, v in values.items()})
                    tf.keras.backend.clear_session()
            np.savez_compressed(output, **arrays)
            write_json(result, dict(task=task, fold=fold, member=member, audit_sha256=source_hash,
                       parent_sha256=parent_hash, arrays_sha256=file_hash(output), **result_rows))
            if (member+1) % 25 == 0 or limit == 1:
                print(task, fold, member+1, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("phonons", "steel"), required=True)
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    parser.add_argument("--members", type=int, choices=range(1,51), default=50)
    parser.add_argument("--output", type=Path, default=discovery.OUT)
    args = parser.parse_args()
    run(args.task, args.folds, args.output, args.members)
