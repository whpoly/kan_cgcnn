"""Discover directly editable physical-input responses in saved KAN and MLP pairs."""
import argparse
import pickle
from pathlib import Path

import analyze_steel396_concepts as steel
from physical_feature_modules import cubic_basis
from run_modnet_official_matched import np, pd, tf, file_hash, write_json


p = steel.phonons
ROOT = steel.ROOT
OUT = ROOT / "benchmarks/physical-edge-discovery-20261112"
ARMS = ("kan", "mlp")
BUDGETS = ("one", "three", "five", "ten", "quarter", "train90", "all")
FAMILIES = ("ElementFraction", "AverageBondLength", "AverageBondAngle", "DensityFeatures",
            "ElectronegativityDiff", "CoordinationNumber", "StructuralHeterogeneity", "ValenceOrbital",
            "YangSolidSolution", "Miedema", "BandCenter", "AtomicPackingEfficiency", "Stoichiometry")
MAGPIE = {"AtomicWeight", "CovalentRadius", "Electronegativity", "MeltingT", "GSvolume_pa",
          "GSbandgap", "GSmagmom", "NValence", "NUnfilled", *(f"N{o}{k}" for o in "spdf" for k in ("Valence", "Unfilled"))}
DESIGN = dict(
    search="Shared source-selected scalar descriptors from listed physical families, >=5 distinct central quantile values. No concept targets.",
    family_allowlist=FAMILIES, magpie_properties=sorted(MAGPIE),
    selection="96 fixed actual training rows. Mean absolute downstream gradient times native one-edge change from the training median. First input by mean within-arm ranks of each feature's best node. At each arm's winning node select up to two further common inputs by mean rank, |training r| < .95 with each preceding input. Ties by name.",
    reference="Actual training row nearest the componentwise medians in units of the q90-q10 range of the selected features.",
    transplant="All outer group donor vectors x all outer recipient backgrounds. Fixed portable MLP response averages its exact joint nonlinear change over training backgrounds; outer-background mean is an oracle bound only. Natural rows also assessed.",
    compact="For each selected edge, anchored affine and continuous two-slope fit on 17 training quantiles q10..q90. Hinge chosen by training loss at q20/35/50/65/80. MLP slices hold the other group inputs at reference and average over training backgrounds. Primary errors include all outer donors.",
    terminal="Both architectures use exact train-centered contributions in target units. Greedily retain the next native term minimizing training squared error with its original weight. Fixed budgets and first training R2 >= .90.",
    random="Four other first-layer nodes, same selected group, same member and outer samples.",
    examples="Fold 0, member 0 in both tasks; first outer ID in lexical order. Indices zero-based.",
    comparisons="Per-fold medians over all 50 matched pairs, then equal fold mean. Ensembles also scored against their own raw teachers and original target convention.")


def physical(name):
    family, _, label = name.partition("|")
    if family == "ElementProperty":
        return label.startswith("MagpieData ") and label.split()[-1] in MAGPIE
    return family in FAMILIES


def basis(scaled, bounded):
    x = tf.convert_to_tensor(scaled, dtype=tf.float32)
    return cubic_basis(.5 * tf.tanh(x) if bounded else x).numpy()


def activation(layer, value):
    if layer.activation.__name__ == "elu":
        return np.maximum(value, 0) + np.expm1(np.minimum(value, 0))
    return layer.activation(tf.convert_to_tensor(value)).numpy()


def rms_ratio(errors, signal):
    return float(np.sqrt(np.mean(np.square(errors, dtype=float)) / max(np.mean(np.square(signal, dtype=float)), 1e-24)))


def ranks(scores):
    # Average tied ranks so that descriptor naming is only the final tie breaker.
    from scipy.stats import rankdata
    return rankdata(-np.asarray(scores), method="average")


def terminal_terms(train, test, train_pred):
    base = train_pred.mean()
    center = train.mean(axis=0)
    c = train - center
    gram = c.T @ c
    gain = 2 * (c.T @ (train_pred - base)) - gram.diagonal()
    order = []
    for _ in range(len(gain)):
        node = int(np.argmax(gain))
        order.append(node)
        gain -= 2 * gram[:, node]
        gain[order] = -np.inf
    train_prefix = base + np.cumsum(c[:, order], axis=1)
    error = np.mean((train_prefix - train_pred[:, None])**2, axis=0)
    k90 = int(np.flatnonzero(error <= .1 * np.var(train_pred))[0]) + 1
    caps = [1, 3, 5, 10, int(np.ceil(len(order) / 4)), k90, len(order)]
    prefix = base + np.cumsum((test - center)[:, order], axis=1)
    return prefix[:, np.minimum(caps, len(order)) - 1], dict(order=order, k90=k90, width=len(order), base=base)


class Member:
    def __init__(self, source, frame, train_ids, test_ids, raw_ref, arm):
        path, self.seed, sha = source
        assert file_hash(path) == sha
        with path.open("rb") as stream:
            self.model = pickle.load(stream)
        self.model._restore_model()
        self.kan = arm == "kan"
        if self.kan:
            assert self.model.target_scaling["fit_ids"] == train_ids
        self.names = list(self.model.optimal_descriptors[:self.model.n_feat])
        self.index = {name: i for i, name in enumerate(self.names)}
        self.raw = frame.loc[train_ids, self.names].values
        self.test_raw = frame.loc[test_ids, self.names].values
        self.x = p.scaled(self.model, self.raw)
        self.xt = p.scaled(self.model, self.test_raw)
        tr, self.y, terms, layers, first, e1 = p.forward(self.model, self.x, self.kan)
        te, self.yt, test_terms, _, _, e2 = p.forward(self.model, self.xt, self.kan)
        assert first == 0
        np.testing.assert_allclose(self.yt, raw_ref, atol=.01, rtol=1e-6)
        self.replay = float(np.max(abs(self.yt - raw_ref)))
        self.decomposition = max(e1, e2)
        self.h, self.ht = tr["first"], te["first"]
        self.first, self.tail = layers[0], layers[1:]
        self.ledger_pred, self.ledger = terminal_terms(terms, test_terms, self.y)
        np.testing.assert_allclose(self.ledger_pred[:, -1], self.yt, atol=.012, rtol=1e-6)
        if self.kan:
            self.weights = self.first.coefficients.numpy()
            self.bounded = self.first.hidden_coordinates
        else:
            self.weights, self.bias = self.first.get_weights()
        self.sha = sha

    def downstream(self, h):
        for layer in self.tail:
            h = layer(h, training=False)
        return h

    def scores(self, common, rows):
        h = tf.convert_to_tensor(self.h[rows])
        with tf.GradientTape() as tape:
            tape.watch(h)
            output = self.downstream(h)
        gradient = tape.gradient(output, h).numpy()
        indices = np.array([self.index[n] for n in common])
        median = np.median(self.x[:, indices], axis=0)
        x = self.x[rows]
        if not self.kan:
            pre = x @ self.weights + self.bias
        blocks = []
        for start in range(0, len(common), 32):
            idx = indices[start:start+32]
            anchor = median[start:start+32]
            if self.kan:
                diff = basis(x[:, idx], self.bounded) - basis(anchor, self.bounded)
                change = np.einsum("sib,ijb->sij", diff, self.weights[idx])
            else:
                shift = (x[:, idx] - anchor)[:, :, None] * self.weights[idx]
                change = activation(self.first, pre)[:, None, :] - activation(self.first, pre[:, None, :] - shift)
            blocks.append(np.mean(np.abs(change * gradient[:, None, :]), axis=0))
        return np.vstack(blocks)

    def after_edit(self, node, delta):
        changed = self.ht.copy()
        changed[:, node] += np.asarray(delta, dtype=changed.dtype)
        return np.asarray(self.downstream(tf.convert_to_tensor(changed))).reshape(-1)

    def process(self, predictions, task, dtype):
        if task == "steel":
            return steel.native_postprocess(predictions, self.model, self.seed, dtype)
        return predictions


def pick_inputs(pair, train_frame, rows):
    common = sorted(n for n in set(pair[0].names).intersection(pair[1].names) if physical(n))
    values = train_frame[common].values
    grid = np.quantile(values, np.linspace(.1, .9, 17), axis=0)
    span = np.ptp(grid, axis=0)
    resolvable = span > 1e-9 * np.maximum(1, np.max(abs(grid), axis=0))
    names = [n for i, n in enumerate(common) if resolvable[i] and
             len(np.unique(np.round((grid[:, i]-grid[0, i])/span[i], 6))) >= 5]
    assert names
    scores = [m.scores(names, rows) for m in pair]
    rank_sum = sum(ranks(s.max(axis=1)) for s in scores)
    lead = int(np.argmin(rank_sum))
    nodes = [int(np.argmax(s[lead])) for s in scores]
    joint = sum(ranks(s[:, node]) for s, node in zip(scores, nodes))
    selected = [lead]
    corr = p.corr_matrix(train_frame[names].values, train_frame[names].values)
    for i in np.argsort(joint, kind="stable"):
        if i not in selected and np.all(np.abs(corr[i, selected]) < .95):
            selected.append(int(i))
        if len(selected) == 3:
            break
    chosen = [names[i] for i in selected]
    x = train_frame[chosen].values
    low, mid, high = np.quantile(x, [.1, .5, .9], axis=0)
    anchor = int(np.argmin(np.sum(((x - mid) / (high - low))**2, axis=1)))
    return chosen, nodes, anchor, dict(eligible=len(names), names=names, anchor_id=train_frame.index[anchor],
            gradients_times_changes=[s[selected, node] for s, node in zip(scores, nodes)],
            rank_of_lead=[float(ranks(s.max(axis=1))[lead]) for s in scores])


def fit_curve(grid, curve, ref, knots):
    shift = grid - ref
    linear = np.linalg.lstsq(shift[:, None], curve, rcond=None)[0][0]
    best = None
    for knot in np.unique(knots):
        kink = np.maximum(grid - knot, 0) - max(ref - knot, 0)
        design = np.column_stack((shift, kink))
        if np.linalg.matrix_rank(design) < 2:
            continue
        slopes = np.linalg.lstsq(design, curve, rcond=None)[0]
        error = np.mean((design @ slopes - curve)**2)
        if best is None or error < best[0]:
            best = error, knot, slopes[0], slopes[1]
    _, knot, left, extra = best if best is not None else (0, ref, linear, 0)
    return dict(anchor=ref, affine_slope=linear, knot=knot, left=left, right=left+extra)


def curve_approx(t, fit):
    shift = t - fit["anchor"]
    lin = fit["affine_slope"] * shift
    kink = np.maximum(t - fit["knot"], 0) - max(fit["anchor"] - fit["knot"], 0)
    return lin, fit["left"] * shift + (fit["right"] - fit["left"]) * kink


def group_response(m, names, node, anchor_pos, rng):
    idx = [m.index[n] for n in names]
    raw, raw_test = m.raw[:, idx], m.test_raw[:, idx]
    base = raw[anchor_pos]
    x, xt, anchor = m.x[:, idx], m.xt[:, idx], m.x[anchor_pos, idx]
    scaler = m.model._scaler
    if hasattr(scaler, "min_"):
        scale, offset = scaler.scale_[idx], scaler.min_[idx]
    else:
        scale, offset = 1 / scaler.scale_[idx], -scaler.mean_[idx] / scaler.scale_[idx]
    np.testing.assert_allclose(raw*scale+offset, x, atol=3e-4, rtol=1e-6)
    grid = np.quantile(raw, np.linspace(.1, .9, 17), axis=0)
    gs = grid * scale + offset
    knots = np.quantile(raw, [.2, .35, .5, .65, .8], axis=0)
    # Compute a joint group transplant in the first layer, keeping each recipient's
    # remaining descriptors. Diagonal entries are the edits of the natural outer rows.
    if m.kan:
        weights = m.weights[idx, node, :]
        def edges(s):
            return np.einsum("nib,ib->ni", basis(s, m.bounded) - basis(anchor, m.bounded), weights).astype(float)
        etrain, etest, on_grid = edges(x), edges(xt), edges(gs)
        portable, train_contribution = etest.sum(1), etrain.sum(1)
        true_grid = np.broadcast_to(portable, (len(xt), len(xt)))
        support_curves = np.stack((on_grid, on_grid))
        recipe = dict(coefficients=weights, bounded=bool(m.bounded), input_multiplier=scale, input_intercept=offset,
                      spline_knots=np.r_[[-.5]*3, np.linspace(-.5, .5, 6), [.5]*3], clipping=[-.5, .5-1e-6])
    else:
        weights = m.weights[idx, node].astype(float)
        def contexts(data, selected):
            return (data @ m.weights[:, node] + m.bias[node]).astype(float) - (selected - anchor) @ weights
        btrain, btest = contexts(m.x, x), contexts(m.xt, xt)
        def response(b, shift):
            return activation(m.first, b[:, None] + shift) - activation(m.first, b)[:, None]
        shifts = (gs - anchor) * weights
        on_grid = np.column_stack([response(btrain, s).mean(0) for s in shifts.T])
        support_curves = np.stack([np.quantile(response(btest, s), [.1, .9], axis=0) for s in shifts.T], axis=-1)
        etest = np.column_stack([response(btrain, s).mean(0) for s in ((xt - anchor) * weights).T])
        true_grid = response(btest, (xt - anchor) @ weights)
        portable = response(btrain, (xt - anchor) @ weights).mean(0)
        train_contribution = response(btrain, (x - anchor) @ weights).mean(0)
        recipe = dict(weights=weights, activation=m.first.activation.__name__, input_multiplier=scale, input_intercept=offset)
    fits = [fit_curve(grid[:, j], on_grid[:, j], base[j], knots[:, j]) for j in range(len(names))]
    approx = [curve_approx(raw_test[:, i], fit) for i, fit in enumerate(fits)]
    linear = np.array([v[0] for v in approx]).T
    hinge = np.array([v[1] for v in approx]).T
    diag = np.diag(true_grid)
    # Check the algebra against the actual first-layer implementation at the anchor.
    x_ref = m.xt.copy()
    x_ref[:, idx] = anchor
    h_ref = m.first(tf.convert_to_tensor(x_ref), training=False).numpy()
    observed = (m.ht[:, node] - h_ref[:, node]).astype(float)
    np.testing.assert_allclose(diag, observed, atol=.002, rtol=1e-5)
    h_mean = (m.h[:, node] - train_contribution).mean()
    h_test = m.ht[:, node]
    node_r2 = 1 - np.mean((h_mean + portable - h_test)**2) / max(float(np.var(h_test)), 1e-14)
    other = np.delete(np.arange(m.ht.shape[1]), node)
    controls = rng.choice(other, min(4, len(other)), replace=False)
    assert len(controls) == 4
    predictions = dict(remove=m.after_edit(node, -observed),
                       portable=m.after_edit(node, portable-observed),
                       hinge=m.after_edit(node, hinge.sum(1)-observed),
                       random=np.column_stack([m.after_edit(j, h_ref[:, j]-m.ht[:, j]) for j in controls]))
    curve_amp = np.sqrt(np.mean(on_grid**2, axis=0))
    metrics = dict(transfer_all=rms_ratio(true_grid-portable, true_grid),
                   oracle_all=rms_ratio(true_grid-true_grid.mean(0), true_grid),
                   transfer_natural=rms_ratio(portable-diag, diag),
                   hinge_all=rms_ratio(true_grid-hinge.sum(1), true_grid),
                   hinge_natural=rms_ratio(hinge.sum(1)-diag, diag),
                   affine_all=rms_ratio(true_grid-linear.sum(1), true_grid),
                   affine_natural=rms_ratio(linear.sum(1)-diag, diag),
                   whole_node_r2=float(node_r2), curve_rms=curve_amp,
                   single_edge_bend=rms_ratio(etest-linear, etest),
                   mean_abs_output_edit=np.mean(abs(predictions["remove"]-m.yt)),
                   mean_abs_random_edit=np.mean(abs(predictions["random"]-m.yt[:, None])),
                   mean_abs_hinge_output_error=np.mean(abs(predictions["hinge"]-m.yt)),
                   mean_abs_portable_output_error=np.mean(abs(predictions["portable"]-m.yt)),
                   first_edit_replay_max=float(np.max(abs(diag-observed))))
    if m.kan:
        def coordinate(s):
            return np.clip(.5*np.tanh(s) if m.bounded else s, -.5, .5-1e-6)
        coord = coordinate(gs)-coordinate(anchor)
        affine_slope = (coord*on_grid).sum(0) / np.maximum((coord**2).sum(0), 1e-20)
        affine_coord = (coordinate(xt)-coordinate(anchor))*affine_slope
        metrics["bend_beyond_coordinate"] = rms_ratio(etest-affine_coord, etest)
        metrics["coordinate_affine_output_error"] = np.mean(np.abs(m.after_edit(node, affine_coord.sum(1)-observed)-m.yt))
        recipe["effective_coordinate_affine_slopes"] = affine_slope
    record = dict(node=node, first_width=m.ht.shape[1], anchor=base, curves=fits, grid=grid, response=on_grid,
                  test_background_p10_p90=support_curves, source_recipe=recipe, metrics=metrics, random_nodes=controls)
    return record, predictions, dict(component=diag, portable_component=portable, hinge_component=hinge.sum(1),
                                     slice_components=etest, hidden=h_test, source_values=raw_test)


def load_fold(task, fold, data):
    frame, targets, splits = data
    sources, refs, official = [], [], []
    if task == "phonons":
        split = steel.read(p.engine.OLD / "matbench_phonons" / f"fold-{fold}/outer-split.json")
        test_ids = split["test_ids"]
        outer_train = set(split["train_ids"])
        for suffix in ("kan", "modnet"):
            directory = p.RUN / f"mm_mv_{suffix}/fold-{fold}"
            done = steel.read(directory / "complete.json")
            for filename, key in (("frozen-selection.json","selection_sha256"), ("outer-predictions.npz","prediction_sha256")):
                assert file_hash(directory / filename) == done[key]
            selection = steel.read(directory / "frozen-selection.json")
            sources.append([(Path(m["directory"])/"model.pkl", m["seed"], selection["model_sha256"][m["directory"]], m["train_ids"])
                            for m in selection["members"]])
            with np.load(directory / "outer-predictions.npz") as z:
                np.testing.assert_array_equal(z["sample_ids"], split["official_test_ids"])
                np.testing.assert_allclose(z["target"], targets.loc[test_ids], rtol=0, atol=1e-8)
                refs.append(z["members"])
            official.append(refs[-1])
    else:
        outer_train = set(frame.index[splits[fold][0]])
        test_ids = frame.index[splits[fold][1]].tolist()
        for arm, directory in zip(ARMS, (steel.KAN / f"fold-{fold}", steel.MLP / f"fold-{fold}/top50")):
            audit = steel.read(steel.OUT / f"fold-{fold}/{arm}.json")
            pred_path = directory / "outer-member-predictions.npz"
            assert file_hash(pred_path) == audit["member_predictions_sha256"]
            members = []
            for entry in audit["members"]:
                path = ROOT / entry["path"]
                assert file_hash(path.with_name("result.json")) == entry["fit_sha256"]
                fit = steel.read(path.with_name("result.json"))
                members.append((path, fit["seed"], entry["sha256"], fit["train_ids"]))
            sources.append(members)
            with np.load(pred_path) as z, np.load(directory / "test.npz") as test:
                np.testing.assert_array_equal(test["ids"], test_ids)
                np.testing.assert_array_equal(test["target"], targets.loc[test_ids])
                refs.append(z["raw_predictions"])
                official.append(z["predictions"])
                np.testing.assert_array_equal(official[-1].mean(0), test["prediction"])
    assert len(sources[0]) == len(sources[1]) == 50 and not outer_train.intersection(test_ids)
    assert all(k[3] == m[3] and set(k[3]) <= outer_train for k, m in zip(*sources))
    return sources, refs, official, test_ids


def load_data(task, out):
    if task == "phonons":
        banks, targets = p.engine.load_banks(out)
        return banks["mm_mv"], targets.iloc[:, 0], None
    archive = steel.INPUT / "inputs.npz"
    assert file_hash(archive) == steel.read(steel.KAN / "protocol.json")["input_sha256"]
    with np.load(archive) as z:
        frame = pd.DataFrame(z["X"], index=z["sample_ids"], columns=z["columns"])
        return frame, pd.Series(z["y"], index=frame.index), [(z[f"train_positions_{f}"], z[f"test_positions_{f}"]) for f in range(5)]


def run(task, folds, out, limit):
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(task, out / task / f"data-check-{folds[0]}")
    frame, target, _ = data
    source_hash = file_hash(__file__)
    for fold in folds:
        sources, refs, official, test_ids = load_fold(task, fold, data)
        folder = out / task / f"fold-{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        for member in range(limit):
            meta = folder / f"member-{member:02d}.json"
            archive = meta.with_suffix(".npz")
            model_hashes = [s[member][2] for s in sources]
            if meta.exists():
                saved = steel.read(meta)
                if saved["analysis_sha256"] == source_hash and saved["model_sha256"] == model_hashes and saved["arrays_sha256"] == file_hash(archive):
                    continue
            train_ids = sources[0][member][3]
            pair = [Member(s[member][:3], frame, train_ids, test_ids, ref[member], arm) for s, ref, arm in zip(sources, refs, ARMS)]
            rng = np.random.RandomState(261112 + fold*1000 + member)
            rows = np.sort(rng.choice(len(train_ids), min(96, len(train_ids)), replace=False))
            names, nodes, anchor, selection = pick_inputs(pair, frame.loc[train_ids], rows)
            arrays = dict(ids=test_ids, target=target.loc[test_ids].values)
            records = {}
            for a, (arm, model, node) in enumerate(zip(ARMS, pair, nodes)):
                result, preds, physical_values = group_response(model, names, node, anchor,
                                            np.random.RandomState(901112 + fold*1000 + member))
                official_ref = official[a][member]
                reproduced = model.process(model.yt[:, None], task, official_ref.dtype)[:, 0]
                np.testing.assert_allclose(reproduced, official_ref, rtol=1e-6, atol=.01)
                records[arm] = dict(**result, model_path=str(sources[a][member][0].relative_to(ROOT)),
                    terminal=model.ledger, raw_replay_max=model.replay, decomposition_max=model.decomposition)
                for key, value in dict(**physical_values, **preds, raw_reference=refs[a][member],
                                       official_reference=official_ref, terminal=model.ledger_pred).items():
                    arrays[f"{arm}_{key}"] = value
                for key, value in dict(**preds, terminal=model.ledger_pred).items():
                    v = value.reshape(len(test_ids), -1)
                    arrays[f"{arm}_{key}_official"] = model.process(v, task, official_ref.dtype)
            np.savez_compressed(archive, **arrays)
            write_json(meta, dict(task=task, fold=fold, member=member, inputs=names, selection=selection,
                       analysis_sha256=source_hash, arrays_sha256=file_hash(archive), model_sha256=model_hashes, **records))
            del pair, model
            tf.keras.backend.clear_session()
            if (member+1) % 10 == 0 or limit == 1:
                print(task, fold, member+1, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--task", choices=("steel", "phonons"), required=True)
    cli.add_argument("--folds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    cli.add_argument("--members", type=int, choices=range(1, 51), default=50)
    cli.add_argument("--output", type=Path, default=OUT)
    args = cli.parse_args()
    run(args.task, args.folds, args.output, args.members)
