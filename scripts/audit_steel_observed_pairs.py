"""Trace prediction differences through frozen models using only observed steel records."""
import argparse
import pickle

import discover_physical_kan_responses as exp
from audit_steel_equal_nodes import chemistry_pairs
from audit_steel_physical_strength import OUT as MATCH_ARCHIVE, match_real
from run_modnet_official_matched import file_hash, np, tf, write_json
from scipy.interpolate import BSpline


OUT = exp.ROOT / "benchmarks/steel-observed-pairs-20261112"
CR = "ElementFraction|Cr"
KNOTS = np.r_[[-.5] * 3, np.linspace(-.5, .5, 6), [.5] * 3]
SPLINE = BSpline(KNOTS, np.eye(8), 3)
SPLINE_DERIVATIVE = SPLINE.derivative()


def basis(values, bounded, derivative=False):
    coord = .5 * np.tanh(values) if bounded else values
    clipped = np.clip(coord, -.5, .5 - 1e-6)
    if not derivative:
        return SPLINE(clipped)
    slope = .5 * (1 - np.tanh(values) ** 2) if bounded else np.ones_like(coord)
    slope *= (coord >= -.5) & (coord <= .5 - 1e-6)
    return SPLINE_DERIVATIVE(clipped) * slope[..., None]


def divided_change(change, step, derivative):
    denominator = step.reshape(step.shape + (1,) * (change.ndim - step.ndim))
    return np.divide(change, denominator, out=derivative.copy(), where=np.abs(denominator) > 1e-9)


def elu(x):
    return np.maximum(x, 0) + np.expm1(np.minimum(x, 0))


def forward(layers, x, kan, scale, offset):
    """Log states of original data rows, including the inputs to each nonlinearity."""
    states = []
    for layer in layers:
        if kan:
            w = layer.coefficients.numpy().astype(float)
            bounded = layer.hidden_coordinates
            b = basis(x, bounded)
            states.append((x, b, w, bounded))
            x = b.reshape(len(x), -1) @ w.transpose(0, 2, 1).reshape(-1, w.shape[1])
            x += layer.bias.numpy()
        else:
            w, bias = layer.get_weights()
            u = x @ w.astype(float) + bias
            name = layer.activation.__name__
            assert name in ("linear", "elu"), name
            x = u if name == "linear" else elu(u)
            states.append((u, x, w.astype(float), name))
        if len(states) == 1:
            first = x
    assert x.shape[1] == 1
    return states, first, x[:, 0] * scale + offset


def spline_slopes(state, lo, hi):
    x, b, _, bounded = state
    return divided_change(b[hi] - b[lo], x[hi] - x[lo], basis(x[lo], bounded, derivative=True))


def activation_slopes(state, lo, hi):
    u, h, _, name = state
    if name == "linear":
        return np.ones_like(u[lo])
    return divided_change(h[hi] - h[lo], u[hi] - u[lo], np.exp(np.minimum(u[lo], 0)))


def first_contributions(states, first, lo, hi, kan, scale):
    # A scalar output multiplier per real pair. Biases cancel in differences.
    back = np.full((len(lo), 1), scale, dtype=float)
    for state in reversed(states[1:]):
        w = state[2]
        if kan:
            slopes = spline_slopes(state, lo, hi)
            back = np.einsum("pib,ijb,pj->pi", slopes, w, back, optimize=True)
        else:
            back = (back * activation_slopes(state, lo, hi)) @ w.T
    return (first[hi] - first[lo]) * back, back


def rank_strength(terms, total, count):
    """Greedy train-pair least squares with original signed MPa contributions."""
    order = []
    residual = total.copy()
    for _ in range(count):
        error = ((residual[:, None] - terms) ** 2).sum(axis=0)
        error[order] = np.inf
        index = int(error.argmin())
        order.append(index)
        residual -= terms[:, index]
    return np.array(order)


def feature_terms(state, inputs, lo, hi, nodes, back, kan):
    """Distribute the selected nodes' MPa to their original descriptor inputs."""
    if kan:
        delta = state[1][hi] - state[1][lo]
        edges = np.einsum("pib,ijb->pij", delta, state[2][:, nodes], optimize=True)
    else:
        dx = inputs[hi] - inputs[lo]
        slope = activation_slopes(state, lo, hi)[:, nodes]
        edges = dx[:, :, None] * state[2][:, nodes] * slope[:, None, :]
    return edges, edges * back[:, None, nodes]


def audit_member(arm, source, frame, test_ids, q, train_pairs, test_pairs, raw, official, example):
    path, _, digest, train_ids = source
    assert file_hash(path) == digest
    with path.open("rb") as stream:
        model = pickle.load(stream)
    model._restore_model()
    names = list(model.optimal_descriptors[:model.n_feat])
    all_ids = train_ids + test_ids
    positions = {name: i for i, name in enumerate(all_ids)}
    assert len(positions) == len(all_ids)
    x = exp.p.scaled(model, frame.loc[all_ids, names].values).astype(float)
    layers = [l for l in model.model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    kan = arm == "kan"
    scale, offset = 1., 0.
    if kan:
        assert model.target_scaling["fit_ids"] == train_ids
        assert len(layers) == 4 and isinstance(layers[-1], tf.keras.layers.Rescaling)
        scale, offset = model.target_scaling["std"], model.target_scaling["mean"]
        layers = layers[:-1]
        assert all(hasattr(l, "coefficients") for l in layers)
    else:
        assert all(isinstance(l, tf.keras.layers.Dense) for l in layers)
        assert any(l.activation.__name__ != "linear" for l in layers[1:])
    states, first, pred = forward(layers, x, kan, scale, offset)
    n = len(train_ids)
    raw_error = float(np.max(np.abs(pred[n:] - raw)))
    np.testing.assert_allclose(pred[n:], raw, atol=.02, rtol=1e-6)
    pairs = train_pairs + test_pairs
    lo = np.array([positions[p["low"]] for p in pairs])
    hi = np.array([positions[p["high"]] for p in pairs])
    terms, back = first_contributions(states, first, lo, hi, kan, scale)
    total = pred[hi] - pred[lo]
    conserv = float(np.abs(terms.sum(axis=1) - total).max())
    np.testing.assert_allclose(terms.sum(axis=1), total, atol=1e-5, rtol=1e-10)
    ntrain = len(train_pairs)
    assert ntrain and ntrain < len(lo)
    fraction_count = int(np.ceil(.1 * first.shape[1]))
    order = rank_strength(terms[:ntrain], total[:ntrain], max(5, fraction_count))
    q_train = q.loc[train_ids].values[:, None]
    nodes, _, _, q_estimate = exp.steel.three_node_probe(first[:n], first[n:], q_train, .01)
    nodes = nodes[0]
    tr = first[:n, order[:3]]
    mean = tr.mean(axis=0)
    s = tr - mean
    # Use the same layer-wide ridge scale as the three-node concept probe.
    ridge = .01 * np.square(first[:n] - first[:n].mean(axis=0)).sum(axis=0).mean()
    beta = np.linalg.solve(s.T @ s + ridge * np.eye(3), s.T @ (q_train[:, 0] - q_train.mean()))
    strength_readout = (first[n:, order[:3]] - mean) @ beta + q_train.mean()
    lt, ht = lo[ntrain:], hi[ntrain:]
    t = terms[ntrain:]
    edges, paths = feature_terms(states[0], x, lt, ht, nodes, back[ntrain:], kan)
    node_delta = first[ht][:, nodes] - first[lt][:, nodes]
    np.testing.assert_allclose(edges.sum(axis=1), node_delta, atol=1e-7, rtol=1e-10)
    path_error = float(np.max(np.abs(paths.sum(axis=1) - t[:, nodes])))
    np.testing.assert_allclose(paths.sum(axis=1), t[:, nodes], atol=1e-5, rtol=1e-10)
    delta_raw = raw[ht - n] - raw[lt - n]
    delta_official = official[ht - n] - official[lt - n]
    delta_raw = delta_raw.astype(float)
    records = dict(sha256=digest, first_width=first.shape[1], physical_nodes=nodes,
                   strength_order=order, replay_max_mpa=raw_error, conservation_max_mpa=conserv,
                   descriptor_sum_max_mpa=path_error, n_train_pairs=ntrain)
    if example is not None:
        i = names.index(CR)
        j = 0  # Leading physical node, training order. Same illustration rule for both models.
        records["example"] = dict(node=int(nodes[j]), input_name=CR,
            scaled_values=x[[lt[example], ht[example]], i],
            node_values=first[[lt[example], ht[example]], nodes[j]],
            input_to_node=edges[example, i, j], node_to_output_mpa=t[example, nodes[j]],
            input_via_node_mpa=paths[example, i, j],
            input_via_three_mpa=paths[example, i].sum(),
            three_node_mpa=t[example, nodes])
        if kan:
            records["example"]["edge_values"] = states[0][1][[lt[example], ht[example]], i] @ states[0][2][i, nodes[j]]
        else:
            records["example"]["input_weight"] = states[0][2][i, nodes[j]]
            records["example"]["preactivation"] = states[0][0][[lt[example], ht[example]], nodes[j]]
    data = dict(raw=delta_raw, official=delta_official,
                corrected=(np.abs(raw - official) > .01)[lt - n] | (np.abs(raw - official) > .01)[ht - n],
                physical3=t[:, nodes].sum(axis=1), strength1=t[:, order[0]],
                strength3=t[:, order[:3]].sum(axis=1), strength5=t[:, order[:5]].sum(axis=1),
                strength10pct=t[:, order[:fraction_count]].sum(axis=1),
                q_physical3=q_estimate[:, 0], q_strength3=strength_readout)
    return records, data, names


def fold_pairs(frame, y, q, coords, split, fold):
    tr, te = split
    ids = list(frame.index)
    held = list(frame.index[te])
    recalc = match_real(q.values, coords, tr, te, ids)["pairs"]
    archive = exp.steel.read(MATCH_ARCHIVE / f"fold-{fold}/design.json")["pairs"]["ni_equivalent"]["pairs"]
    assert [(p["low"], p["high"]) for p in recalc] == [(p["low"], p["high"]) for p in archive]
    for p, a in zip(recalc, archive):
        np.testing.assert_allclose([p[k] for k in ("q_low", "q_high", "distance")],
                                   [a[k] for k in ("q_low", "q_high", "distance")], rtol=0, atol=1e-10)
    records = [dict(**p, group="matched") for p in recalc]
    if fold == 0:
        records.append(dict(low="mb-steels-020", high="mb-steels-005", group="requested"))
        # Both specified steels can serve as the higher endpoint of a near-unit observed difference.
        # Match the sizes of the two differences before using any labels or model predictions.
        targets = ("mb-steels-005", "mb-steels-020")
        choices = []
        for high in targets:
            gap = q[high] - q.loc[held]
            choices.append([(low, gap[low]) for low in gap.index[(gap >= .75) & (gap <= 1.25)]])
        candidates = [(abs(d1 - d2), abs(d1 - 1) + abs(d2 - 1), low1, low2)
                      for low1, d1 in choices[0] for low2, d2 in choices[1]
                      if len({low1, low2, *targets}) == 4]
        if candidates:
            _, _, low1, low2 = min(candidates)
            for low, high in zip((low1, low2), targets):
                records.append(dict(low=low, high=high, group="requested_near_unit"))
    strict = chemistry_pairs(frame, held, q.loc[held].values)
    records.extend(dict(**p, group="near_unit") for p in strict)
    for p in records:
        l, h = p["low"], p["high"]
        assert l in held and h in held
        p.update(q_low=q[l], q_high=q[h], q_gap=q[h]-q[l],
                 y_low=y[l], y_high=y[h], measured=y[h]-y[l])
    return records


def run(folds, limit):
    data = exp.load_data("steel", OUT)
    frame, y, splits = data
    q = exp.steel.concept_frame(frame)["ni_equivalent"]
    fractions = frame.filter(regex=r"^ElementFraction\|")
    coords = np.sqrt(fractions.loc[:, fractions.ne(0).any()].values) / np.sqrt(2)
    for fold in folds:
        sources, refs, official, test_ids = exp.load_fold("steel", fold, data)
        pairs = fold_pairs(frame, y, q, coords, splits[fold], fold)
        folder = OUT / f"fold-{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        counts = {}
        all_counts = {}
        if fold == 0:
            for name in ("mb-steels-005", "mb-steels-020"):
                gap = q.loc[test_ids] - q[name]
                counts[name] = list(gap.index[(gap >= .75) & (gap <= 1.25)])
                gap = q - q[name]
                all_counts[name] = list(gap.index[(gap >= .75) & (gap <= 1.25)])
        write_json(folder / "pairs.json", dict(fold=fold, pairs=pairs, test_ids=test_ids,
                   test_q=q.loc[test_ids].values, requested_higher_near_unit_candidates=counts,
                   all_rows_higher_near_unit_candidates=all_counts))
        script_hash = file_hash(__file__)
        for i in range(limit):
            meta = folder / f"member-{i:02d}.json"
            archive = meta.with_suffix(".npz")
            hashes = [s[i][2] for s in sources]
            if meta.exists() and archive.exists():
                old = exp.steel.read(meta)
                if (old["code_sha256"] == script_hash and old["model_sha256"] == hashes and
                        old["arrays_sha256"] == file_hash(archive)):
                    continue
            train_ids = sources[0][i][3]
            pos = frame.index.get_indexer(train_ids)
            assert np.min(pos) >= 0
            train_pairs = match_real(q.values, coords, pos, pos, list(frame.index))["pairs"]
            example = next((j for j,p in enumerate(pairs) if p["group"] == "requested"), None) if i == 0 else None
            records, arrays, columns = {}, {}, []
            for a, arm in enumerate(exp.ARMS):
                rec, arr, names = audit_member(arm, sources[a][i], frame, test_ids, q, train_pairs, pairs,
                                               refs[a][i], official[a][i], example)
                records[arm] = rec
                arrays.update({arm + "_" + key: value for key, value in arr.items()})
                columns.append(names)
            assert columns[0] == columns[1]
            np.savez_compressed(archive, **arrays)
            write_json(meta, dict(member=i, code_sha256=script_hash, model_sha256=hashes,
                                  arrays_sha256=file_hash(archive), **records))
            tf.keras.backend.clear_session()
            if (i + 1) % 10 == 0:
                print(f"fold {fold}: {i+1} paired members verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    run(args.folds, args.limit)
