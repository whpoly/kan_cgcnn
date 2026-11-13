"""Compare the chromium-to-node attribution along two routes through existing steels."""
import argparse
import pickle

import audit_steel_observed_pairs as observed
from audit_steel_observed_pairs import CR, OUT, exp, file_hash, np, tf, write_json


def triples(pairs, test_ids, test_q):
    q = np.asarray(test_q)
    result = []
    for j, pair in enumerate(pairs):
        eligible = np.flatnonzero((q > pair["q_low"]) & (q < pair["q_high"]))
        if len(eligible):
            midpoint = (pair["q_low"] + pair["q_high"]) / 2
            k = min(eligible, key=lambda i: (abs(q[i] - midpoint), test_ids[i]))
            result.append(dict(pair=j, group=pair["group"], low=pair["low"], middle=test_ids[k],
                               high=pair["high"], q_middle=q[k]))
    return result


def check(arm, source, frame, test_ids, indices, nodes):
    path, _, digest, train_ids = source
    assert file_hash(path) == digest
    with path.open("rb") as fh:
        model = pickle.load(fh)
    model._restore_model()
    names = list(model.optimal_descriptors[:model.n_feat])
    inputs = exp.p.scaled(model, frame.loc[train_ids + test_ids, names].values).astype(float)
    layer = next(l for l in model.model.layers if not isinstance(l, tf.keras.layers.InputLayer))
    c = names.index(CR)
    n = len(train_ids)
    low, middle, high = (indices + n).T
    start = np.concatenate((low, low, middle))
    end = np.concatenate((high, middle, high))
    if arm == "kan":
        w = layer.coefficients.numpy()[:, nodes].astype(float)
        b = observed.basis(inputs, layer.hidden_coordinates)
        h = np.einsum("pib,ijb->pj", b, w, optimize=True) + layer.bias.numpy()[nodes]
        value = b[:, c] @ w[c].T
        delta = value[end] - value[start]
    else:
        w, bias = layer.get_weights()
        u = inputs @ w[:, nodes].astype(float) + bias[nodes]
        assert layer.activation.__name__ == "elu"
        h = observed.elu(u)
        slope = observed.activation_slopes((u, h, None, "elu"), start, end)
        delta = (inputs[end, c] - inputs[start, c])[:, None] * w[c, nodes] * slope
    sd = h[:n].std(axis=0)
    assert np.min(sd) > 1e-9
    direct, first, second = np.split(delta, 3)
    error = np.abs(direct - (first + second)) / sd
    if arm == "kan":
        assert np.max(error) < 1e-10
    return dict(standardized_error=error, direct=direct, via_middle=first + second)


def run(folds):
    data = exp.load_data("steel", OUT)
    frame = data[0]
    for f in folds:
        folder = OUT / f"fold-{f}"
        design = exp.steel.read(folder / "pairs.json")
        src, _, _, ids = exp.load_fold("steel", f, data)
        assert ids == design["test_ids"]
        cases = triples(design["pairs"], ids, design["test_q"])
        indices = np.array([[ids.index(row[k]) for k in ("low","middle","high")] for row in cases])
        input_audit = [exp.steel.read(exp.steel.OUT / f"fold-{f}/{arm}.json")["members"] for arm in exp.ARMS]
        results, examples = {}, {}
        member_indices = []
        for i in range(50):
            cols = [a[i]["input_features"] for a in input_audit]
            assert cols[0] == cols[1]
            if CR not in cols[0]:
                continue
            member_indices.append(i)
            meta = exp.steel.read(folder / f"member-{i:02d}.json")
            assert file_hash(observed.__file__) == meta["code_sha256"]
            for a, arm in enumerate(exp.ARMS):
                values = check(arm, src[a][i], frame, ids, indices, meta[arm]["physical_nodes"])
                for key, value in values.items():
                    results.setdefault(arm + "_" + key, []).append(value)
                if f == i == 0:
                    j = next(j for j, row in enumerate(cases) if row["group"] == "requested")
                    np.testing.assert_allclose(values["direct"][j, 0], meta[arm]["example"]["input_to_node"], atol=1e-10)
                    examples[arm] = dict(node=meta[arm]["physical_nodes"][0], direct=values["direct"][j, 0],
                                         via_middle=values["via_middle"][j, 0],
                                         error_sd=values["standardized_error"][j, 0])
            tf.keras.backend.clear_session()
        archive = folder / "chromium-edges.npz"
        np.savez_compressed(archive, **results)
        write_json(folder / "chromium-edges.json", dict(fold=f, triples=cases, members=member_indices,
                   example=examples, code_sha256=file_hash(__file__), arrays_sha256=file_hash(archive)))
        print(f"fold {f}: {len(member_indices)} matched members with recorded Cr input", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    run(parser.parse_args().folds)
