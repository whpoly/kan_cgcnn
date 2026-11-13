"""Read physical quantities from the frozen, paired steel SHAP Top-50 networks."""
import argparse
import json
import pickle
from pathlib import Path

import analyze_formal_phonons_concepts as phonons
from audit_formal_phonons_readouts import fit_readouts
from physical_feature_modules import cubic_basis
from pymatgen.core import Element
from run_modnet_official_matched import file_hash, np, pd, tf, write_json


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "benchmarks/steels-paper-reproduction-20261102/mm-roost396"
KAN = ROOT / "benchmarks/steel396-shap-top50-kan-20261112"
MLP = ROOT / "benchmarks/steel396-shap-pair-20261108/formal"
OUT = ROOT / "benchmarks/steel396-physical-concepts-20261112"
MAGPIE = "ElementProperty|MagpieData "
# (family, physical unit, exact feature or affine-equivalent feature, definition)
CONCEPTS = {
    "carbon": ("interstitial", "at.%", "ElementFraction|C", "100 x_C"),
    "carbon_nitrogen": ("interstitial", "at.%", None, "100 (x_C + x_N)"),
    "chromium": ("chromium", "at.%", "ElementFraction|Cr", "100 x_Cr"),
    "nickel": ("nickel", "at.%", "ElementFraction|Ni", "100 x_Ni"),
    "molybdenum": ("molybdenum", "at.%", "ElementFraction|Mo", "100 x_Mo"),
    "cobalt": ("cobalt", "at.%", "ElementFraction|Co", "100 x_Co"),
    "titanium": ("titanium", "at.%", "ElementFraction|Ti", "100 x_Ti"),
    "non_iron": ("mixing", "at.%", "ElementFraction|Fe", "100 (1 - x_Fe)"),
    "ideal_mixing": ("mixing", "dimensionless", None, "-sum x_i ln x_i, nominal ideal mixing entropy / R"),
    "size_mismatch": ("size", "%", "YangSolidSolution|Yang delta", "100 sqrt(sum x_i (1-r_i/mean_r)^2), Miracle radii"),
    "mean_radius": ("size", "pm", MAGPIE + "mean CovalentRadius", "atomic-fraction-weighted covalent radius"),
    "mean_mass": ("mass", "u", MAGPIE + "mean AtomicWeight", "atomic-fraction-weighted atomic mass"),
    "mean_valence": ("valence", "electrons/atom", MAGPIE + "mean NValence", "sum x_i v_i, Magpie valence counts"),
    "valence_deviation": ("valence", "electrons/atom", MAGPIE + "avg_dev NValence", "sum x_i |v_i-mean_v|"),
    "cr_equivalent": ("chromium", "wt.% equivalent", None, "Schaeffler w_Cr + w_Mo + 1.5 w_Si + 0.5 w_Nb"),
    "ni_equivalent": ("nickel", "wt.% equivalent", None, "Schaeffler w_Ni + 30 w_C + 0.5 w_Mn"),
}
NAMES = list(CONCEPTS)
PARENTS = {"carbon_nitrogen": "carbon", "ideal_mixing": "non_iron",
           "cr_equivalent": "chromium", "ni_equivalent": "nickel"}
RANDOM_DRAWS = 8


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def concept_frame(frame):
    fractions = frame.filter(regex=r"^ElementFraction\|").copy()
    fractions.columns = [c.split("|")[1] for c in fractions]
    fractions = fractions.loc[:, fractions.ne(0).any()]
    assert (fractions.values >= 0).all()
    np.testing.assert_allclose(fractions.sum(axis=1), 1, rtol=0, atol=1e-12)
    masses = fractions * [float(Element(s).atomic_mass) for s in fractions]
    weight = 100 * masses.div(masses.sum(axis=1), axis=0)
    np.testing.assert_allclose(weight.sum(axis=1), 100, rtol=0, atol=1e-10)
    derived = {
        "carbon_nitrogen": 100 * (fractions.C + fractions.N),
        "ideal_mixing": -(fractions * np.log(fractions.mask(fractions == 0, 1))).sum(axis=1),
        "cr_equivalent": weight.Cr + weight.Mo + 1.5 * weight.Si + 0.5 * weight.Nb,
        "ni_equivalent": weight.Ni + 30 * weight.C + 0.5 * weight.Mn,
    }
    values = {}
    for name, (_, unit, column, _) in CONCEPTS.items():
        if name in derived:
            values[name] = derived[name]
        elif name == "non_iron":
            values[name] = 100 * (1 - fractions.Fe)
        else:
            values[name] = frame[column] * (100 if unit in ("at.%", "%") else 1)
    q = pd.DataFrame(values, index=frame.index)
    assert np.isfinite(q.values).all() and (q.std() > 0).all()
    return q


def affine_probe(h, ht, q):
    """Select and calibrate with training rows only. Held-out q is not an argument."""
    correlations = phonons.corr_matrix(h, q)
    nodes = np.argmax(np.abs(correlations), axis=0)
    chosen = h[:, nodes].astype(float)
    centered = chosen - chosen.mean(axis=0)
    slopes = (centered * (q - q.mean(axis=0))).sum(axis=0) / np.maximum((centered**2).sum(axis=0), 1e-15)
    offsets = q.mean(axis=0) - slopes * chosen.mean(axis=0)
    return nodes, slopes, offsets, ht[:, nodes] * slopes + offsets


def partial_by_family(h, ht, q, qt):
    result = []
    for i, (family, _, _, _) in enumerate(CONCEPTS.values()):
        other = [j for j, (group, _, column, _) in enumerate(CONCEPTS.values())
                 if group != family and column is not None]
        result.append(phonons.residual_control(h[:, i:i+1], ht[:, i:i+1], q[:, i:i+1],
                                               qt[:, i:i+1], q[:, other], qt[:, other])[0])
    return np.array(result)


def three_node_probe(h, ht, q, penalty):
    """Forward regression with a fixed three-node budget and one scale for the whole layer."""
    centered = h - h.mean(axis=0)
    gram = centered.T @ centered
    # The shared RMS scaling and ridge objective are the same as fit_readouts.
    ridge = penalty * gram.diagonal().mean()
    gram += ridge * np.eye(gram.shape[0])
    cross = centered.T @ (q - q.mean(axis=0))
    floor = gram.diagonal().max() * 1e-12
    nodes = np.empty((q.shape[1], 3), dtype=int)
    slopes = np.empty_like(nodes, dtype=float)
    for i, sequence in enumerate(nodes):
        residual_gram, residual_cross = gram.copy(), cross[:, i].copy()
        for step in range(3):
            variance = residual_gram.diagonal()
            eligible = variance > floor
            eligible[sequence[:step]] = False
            assert np.any(eligible), "Fewer than three independent native nodes"
            improvement = np.divide(residual_cross**2, variance, where=eligible,
                                    out=np.full_like(variance, -np.inf))
            node = sequence[step] = np.argmax(improvement)
            cov = residual_gram[:, node].copy()
            residual_cross -= cov * (residual_cross[node] / variance[node])
            residual_gram -= np.outer(cov, cov) / variance[node]
        if penalty:
            slopes[i] = np.linalg.solve(gram[np.ix_(sequence, sequence)], cross[sequence, i])
        else:
            slopes[i] = np.linalg.lstsq(centered[:, sequence], q[:, i] - q[:, i].mean(), rcond=None)[0]
    offsets = q.mean(axis=0) - np.sum(h.mean(axis=0)[nodes] * slopes, axis=1)
    return nodes, slopes, offsets, (ht[:, nodes] * slopes).sum(axis=2) + offsets


def late_checks(h, ht, q, qt, centered_terms, pt, model, seed, dtype, rng):
    ols_nodes, ols_slopes, ols_offsets, ols_decoded = three_node_probe(h, ht, q, penalty=0.)
    nodes, slopes, offsets, decoded = three_node_probe(h, ht, q, penalty=.01)
    random_groups = np.array([rng.choice(h.shape[1], size=3, replace=False) for _ in range(RANDOM_DRAWS)])
    all_groups = np.vstack((nodes, random_groups))
    ablated = pt[:, None] - centered_terms[:, all_groups].sum(axis=2)
    # Can the layer recover the part of an equivalent that a linear calibration
    # of its principal element (or of non-Fe for entropy) misses?
    bases, derived = [NAMES.index(v) for v in PARENTS.values()], [NAMES.index(v) for v in PARENTS]
    parent_train, parent_test = [], []
    for j, i in zip(bases, derived):
        _, slope, offset, parent = affine_probe(q[:, [j]], qt[:, [j]], q[:, [i]])
        parent_train.append(q[:, j] * slope[0] + offset[0])
        parent_test.append(parent[:, 0])
    parent_test = np.column_stack(parent_test)
    residual = fit_readouts(h, ht, q[:, derived] - np.column_stack(parent_train), qt[:, derived] - parent_test)
    return dict(three_nodes=nodes, three_slopes=slopes, three_offsets=offsets, three=decoded,
                ols_nodes=ols_nodes, ols_slopes=ols_slopes, ols_offsets=ols_offsets, ols=ols_decoded,
                three_ablated_raw=ablated, three_ablated_official=native_postprocess(ablated, model, seed, dtype),
                three_random_nodes=random_groups, parent_only=parent_test, parent_residual_decoded=residual["decoded"])


def native_postprocess(values, model, seed, dtype):
    """Replay MODNet 0.1.13 regression repair with the original member seed per intervention."""
    result = np.asarray(values, dtype=dtype).copy()
    low, high = float(model.min_y[0]), float(model.max_y[0])
    span = high - low
    draws = np.random.RandomState(seed + 100000000).uniform(size=len(result)) * span + low
    for column in result.T:
        outside = (column < low - .25 * span) | (column > high + .25 * span)
        column[outside] = draws[:outside.sum()]
    return result


def case_weights(out, model, layers, first_index, x, h, ht, train_ids, test_ids, cols):
    layer = layers[first_index]
    coords = .5 * tf.tanh(x) if layer.hidden_coordinates else tf.convert_to_tensor(x)
    basis = cubic_basis(coords).numpy()
    coefficients = layer.coefficients.numpy()
    edges = np.einsum("nik,iok->nio", basis, coefficients)
    rebuilt = edges.sum(axis=1) + layer.bias.numpy()
    error = float(np.max(np.abs(rebuilt - h)))
    np.testing.assert_allclose(rebuilt, h, atol=1e-4, rtol=0)
    scaler = model._scaler
    scaling = {k: getattr(scaler, k) for k in ("mean_", "min_", "scale_") if hasattr(scaler, k)}
    np.savez_compressed(out / "first-kan-case.npz", train_ids=train_ids, test_ids=test_ids, columns=cols,
                        train_activation=h, test_activation=ht, coefficients=coefficients, bias=layer.bias.numpy(),
                        mean_abs_centered_edge=np.abs(edges - edges.mean(axis=0)).mean(axis=0),
                        knots=np.r_[[-.5]*3, np.linspace(-.5, .5, 6), [.5]*3],
                        bounded=layer.hidden_coordinates, scaler=type(scaler).__name__, replay_error=error, **scaling)


def inspect_member(path, fit, frame, qframe, test_ids, raw_ref, official_ref, arm, fold, member, out):
    train_ids = fit["train_ids"]
    cols = fit["selected_features"]
    with path.open("rb") as stream:
        model = pickle.load(stream)
    model._restore_model()
    assert list(model.optimal_descriptors[:model.n_feat]) == cols
    is_kan = arm == "kan"
    if is_kan:
        assert model.target_scaling["fit_ids"] == train_ids
    raw = frame.loc[train_ids, cols].values
    raw_test = frame.loc[test_ids, cols].values
    x = phonons.scaled(model, raw)
    x_test = phonons.scaled(model, raw_test)
    q, qt = qframe.loc[train_ids].values, qframe.loc[test_ids].values
    train, pred, terms, layers, first_index, err_tr = phonons.forward(model, x, is_kan)
    test, pt, terms_test, _, _, err_te = phonons.forward(model, x_test, is_kan)
    np.testing.assert_allclose(pt, raw_ref, atol=.01, rtol=1e-6)
    repaired = native_postprocess(pt[:, None], model, fit["seed"], official_ref.dtype)[:, 0]
    np.testing.assert_allclose(repaired, official_ref, atol=.01, rtol=1e-6)
    perm = np.random.RandomState(20261112 + 1000 * fold + member).permutation(len(q))
    rng = np.random.RandomState(301112 + 1000 * fold + member)
    input_nodes, _, _, input_decoded = affine_probe(raw, raw_test, q)
    arrays = {"output_only": affine_probe(pred[:, None], pt[:, None], q)[3],
              "input_r2": phonons.r2(qt, input_decoded),
              "direct": [v[2] in cols for v in CONCEPTS.values()]}
    if fold == member == 0 and is_kan:
        case_weights(out, model, layers, first_index, x, train["first"], test["first"], train_ids, test_ids, cols)
    for layer in ("first", "late"):
        h, ht = train[layer].astype(float), test[layer].astype(float)
        nodes, slopes, offsets, decoded = affine_probe(h, ht, q)
        chosen, chosen_test = h[:, nodes], ht[:, nodes]
        probe = fit_readouts(h, ht, q, qt)
        selected = np.r_[nodes, rng.randint(h.shape[1], size=RANDOM_DRAWS)]
        if layer == "first":
            shifts = phonons.first_node_shifts(layers, first_index, test[layer], train[layer].mean(axis=0), selected, pt)
        else:
            shifts = (terms_test - terms.mean(axis=0))[:, selected]
        ablated = pt[:, None] - shifts
        native_ablated = native_postprocess(ablated, model, fit["seed"], official_ref.dtype)
        rho = phonons.paired_corr(chosen_test, qt)
        partial_out = phonons.residual_control(chosen, chosen_test, q, qt, pred[:, None], pt[:, None])
        partial_other = partial_by_family(chosen, chosen_test, q, qt)
        parent_partials = []
        for name, parent in PARENTS.items():
            i, j = NAMES.index(name), NAMES.index(parent)
            parent_partials.append(phonons.residual_control(chosen[:, [i]], chosen_test[:, [i]], q[:, [i]],
                                                            qt[:, [i]], q[:, [j]], qt[:, [j]])[0])
        data = dict(nodes=nodes, slope=slopes, offset=offsets, width=h.shape[1],
                    rho=rho, single_r2=phonons.r2(qt, decoded), single=decoded,
                    hidden=chosen_test, ridge=probe["decoded"], rank4=probe["decoded4"],
                    null_r2=phonons.r2(qt, affine_probe(h, ht, q[perm])[3]),
                    partial_output=partial_out, partial_other=partial_other, partial_parent=parent_partials,
                    ablated_raw=ablated, ablated_official=native_ablated, random_nodes=selected[len(nodes):])
        if layer == "late":
            data.update(late_checks(h, ht, q, qt, terms_test - terms.mean(axis=0), pt, model,
                                    fit["seed"], official_ref.dtype, rng))
            if fold == member == 0 and is_kan:
                np.savez_compressed(out / "late-kan-case.npz", train_ids=train_ids, test_ids=test_ids,
                                    train_activation=h, test_activation=ht, mean_final_edges=terms.mean(axis=0),
                                    weights=probe["coefficients"], offsets=probe["intercept"])
        arrays.update({f"{layer}_{key}": value for key, value in data.items()})
    record = dict(member=member, path=str(path.relative_to(ROOT)), sha256=file_hash(path), seed=fit["seed"],
                  fit_sha256=file_hash(path.with_name("result.json")), n_train=len(q), input_features=cols,
                  best_single_inputs=[cols[i] for i in input_nodes], replay_max_mpa=float(np.max(np.abs(pt-raw_ref))),
                  official_replay_max_mpa=float(np.max(np.abs(repaired-official_ref))),
                  final_layer_sum_max_error_mpa=max(err_tr, err_te))
    tf.keras.backend.clear_session()
    return arrays, record


def analyze_fold(fold, frame, qframe, archive, out):
    train_positions, test_positions = (archive[f"{part}_positions_{fold}"] for part in ("train", "test"))
    train_ids = set(frame.index[train_positions])
    test_ids = frame.index[test_positions].tolist()
    assert not train_ids.intersection(test_ids) and len(train_ids) + len(test_ids) == len(frame)
    kdir, mdir = KAN / f"fold-{fold}", MLP / f"fold-{fold}"
    members = read(mdir / "top50/selection.json")["members"]
    assert len(members) == 50 and members == read(mdir / "frozen.json")["members"]["top50"]
    kan_audit = read(kdir / "audit.json")
    mlp_complete = read(mdir / "complete.json")
    assert kan_audit["passed"] and read(mdir / "audit.json")["passed"]
    assert file_hash(kdir / "test.npz") == kan_audit["test_sha256"]
    assert file_hash(mdir / "top50/test.npz") == mlp_complete["results"]["top50"]["sha256"]
    refs = {}
    for arm, directory in (("kan", kdir), ("mlp", mdir / "top50")):
        with np.load(directory / "outer-member-predictions.npz") as z:
            raw, official = z["raw_predictions"], z["predictions"]
            assert raw.shape == official.shape == (len(members), len(test_ids))
            np.testing.assert_array_equal(np.sum(raw != official, axis=1), z["replaced_counts"])
        with np.load(directory / "test.npz") as z:
            np.testing.assert_array_equal(z["ids"], test_ids)
            np.testing.assert_array_equal(z["target"], archive["y"][test_positions])
            np.testing.assert_array_equal(z["prediction"], official.mean(axis=0))
            np.testing.assert_allclose(np.abs(z["prediction"] - z["target"]).mean(),
                                       kan_audit["kan" if arm == "kan" else "modnet"], rtol=0, atol=1e-8)
        refs[arm] = raw, official
    folder = out / f"fold-{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    for arm in ("kan", "mlp"):
        raw, official = refs[arm]
        stacks, records = {}, []
        for i, item in enumerate(members):
            mp = Path(item["directory"]) / "model.pkl"
            kp = kdir / f"member-{i:02d}" / "model.pkl"
            mf, kf = read(mp.with_name("result.json")), read(kp.with_name("result.json"))
            assert all(mf[k] == kf[k] for k in ("train_ids", "validation_ids", "selected_features", "seed", "genes"))
            assert kf["seed"] == item["seed"] and kf["run_fingerprint"] == kan_audit["fp"]
            assert mf["run_fingerprint"] == mlp_complete["fp"] and file_hash(mp) == item["sha256"]
            tr, val = map(set, (kf["train_ids"], kf["validation_ids"]))
            assert not tr.intersection(val) and tr | val == train_ids
            path, fit = (kp, kf) if arm == "kan" else (mp, mf)
            values, record = inspect_member(path, fit, frame, qframe, test_ids, raw[i], official[i], arm, fold, i, folder)
            for key, arr in values.items():
                stacks.setdefault(key, []).append(arr)
            records.append(record)
            if (i + 1) % 10 == 0:
                print(f"Fold {fold}, {arm}: {i + 1}/{len(members)}", flush=True)
        arrays_path = folder / f"{arm}.npz"
        np.savez_compressed(arrays_path, names=NAMES, test_ids=test_ids, q_test=qframe.loc[test_ids].values,
                            target=archive["y"][test_positions], raw=raw, official=official, **stacks)
        write_json(folder / f"{arm}.json", dict(arm=arm, fold=fold, extraction_sha256=file_hash(__file__),
                   arrays_sha256=file_hash(arrays_path),
                   members=records, member_predictions_sha256=file_hash(
                       (kdir if arm == "kan" else mdir / "top50") / "outer-member-predictions.npz")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    args = parser.parse_args()
    manifest = read(INPUT / "manifest.json")
    input_path = INPUT / manifest["archive"]
    assert file_hash(input_path) == manifest["archive_sha256"] == read(KAN / "protocol.json")["input_sha256"]
    assert read(KAN / "comparison.json")["passed"] and read(MLP / "audit.json")["passed"]
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(input_path, allow_pickle=False) as archive:
        frame = pd.DataFrame(archive["X"], index=archive["sample_ids"], columns=archive["columns"])
        assert frame.index.is_unique and frame.shape == (312, 396)
        np.testing.assert_array_equal(np.sort(np.concatenate([archive[f"test_positions_{f}"] for f in range(5)])),
                                      np.arange(len(frame)))
        # The benchmark imputes a few MM columns. Confirm every physical reference used here
        # was observed before imputation, including all atomic fractions for derived quantities.
        src = ROOT / manifest["csv"]
        assert file_hash(src) == manifest["csv_sha256"]
        reference_columns = {spec[2] for spec in CONCEPTS.values() if spec[2] is not None}
        reference_columns.update(c for c in frame if c.startswith("ElementFraction|"))
        cols = sorted(reference_columns)
        observed = pd.read_csv(src, usecols=["material_id", *cols]).set_index("material_id").loc[frame.index, cols]
        assert np.isfinite(observed.values).all()
        np.testing.assert_allclose(observed.values, frame[cols].values, rtol=1e-12, atol=1e-14)
        qframe = concept_frame(frame)
        for fold in args.folds:
            analyze_fold(fold, frame, qframe, archive, OUT)


if __name__ == "__main__":
    main()
