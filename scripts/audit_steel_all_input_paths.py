"""Account for strength differences at the original inputs, summing all downstream paths."""
import argparse
import pickle

import audit_steel_molybdenum_case as mo
import audit_steel_observed_pairs as obs
from audit_steel_observed_pairs import exp, file_hash, np, tf, write_json
from audit_steel_physical_strength import match_real


OUT = exp.ROOT / "benchmarks/steel-all-input-paths-20261113"


def pair_set(groups, targets):
    pairs = {}
    for name, entries in groups:
        for entry in entries:
            key = entry["low"], entry["high"]
            if key not in pairs:
                a, b = key
                pairs[key] = dict(low=a,high=b,measured=targets[b]-targets[a],groups=[])
            if name not in pairs[key]["groups"]:
                pairs[key]["groups"].append(name)
    return list(pairs.values())


def contribution(state, x, lo, hi, back, kan):
    if kan:
        return np.einsum("pib,ijb,pj->pi", state[1][hi]-state[1][lo],state[2],back,optimize=True)
    slope = obs.activation_slopes(state,lo,hi)
    return (x[hi]-x[lo])*((slope*back) @ state[2].T)


def member(source, frame, indices, held_pos, raw, arm):
    path, _, digest, fit_ids = source
    assert file_hash(path)==digest
    with path.open("rb") as fh:
        model = pickle.load(fh)
    model._restore_model()
    names = list(model.optimal_descriptors[:model.n_feat])
    x = exp.p.scaled(model,frame[names].values).astype(float)
    layers = [l for l in model.model.layers if not isinstance(l,tf.keras.layers.InputLayer)]
    kan = arm=="kan"
    scale, offset = 1., 0.
    if kan:
        assert model.target_scaling["fit_ids"]==fit_ids
        assert isinstance(layers[-1],tf.keras.layers.Rescaling)
        scale, offset = model.target_scaling["std"], model.target_scaling["mean"]
        layers = layers[:-1]
    states,h,pred = obs.forward(layers,x,kan,scale,offset)
    replay = float(np.max(np.abs(pred[held_pos]-raw)))
    np.testing.assert_allclose(pred[held_pos],raw,atol=.02,rtol=1e-6)
    lo,hi = indices.T
    parts = []
    for start in range(0,len(indices),40):
        a,b = lo[start:start+40],hi[start:start+40]
        _,back = obs.first_contributions(states,h,a,b,kan,scale)
        parts.append(contribution(states[0],x,a,b,back,kan))
    parts = np.vstack(parts)
    delta = pred[hi]-pred[lo]
    error = float(np.max(np.abs(parts.sum(axis=1)-delta)))
    np.testing.assert_allclose(parts.sum(axis=1),delta,rtol=1e-9,atol=1e-5)
    return parts,names,pred,replay,error


def run(folds):
    data = exp.load_data("steel",OUT)
    frame, targets, splits = data
    ids = frame.index.tolist()
    wt,coordinates = mo.chemistry(frame)
    q = exp.steel.concept_frame(frame)
    fold_map = {ids[i]:f for f,(_,te) in enumerate(splits) for i in te}
    natural = mo.natural_pairs(wt,fold_map)
    for f in folds:
        sources,refs,official,held = exp.load_fold("steel",f,data)
        tr,te = splits[f]
        np.testing.assert_array_equal(frame.index[te],held)
        study = exp.steel.read(obs.MATCH_ARCHIVE/f"fold-{f}/design.json")["pairs"]
        test_groups,train_groups = [],[]
        for key in ("molybdenum","ni_equivalent"):
            values = q[key].values
            test = match_real(values,coordinates,tr,te,ids)["pairs"]
            assert [(r["low"],r["high"]) for r in test]==[(r["low"],r["high"]) for r in study[key]["pairs"]]
            test_groups.append((key,test))
            train_groups.append((key,match_real(values,coordinates,tr,tr,ids)["pairs"]))
        test_groups.append(("near_single",[p for p in natural if p["low_fold"]==p["high_fold"]==f]))
        train_groups.append(("near_single",[p for p in natural if f not in (p["low_fold"],p["high_fold"])]))
        train_pairs = pair_set(train_groups,targets)
        test_pairs = pair_set(test_groups,targets)
        pairs = train_pairs+test_pairs
        indices = np.array([[frame.index.get_loc(r[k]) for k in ("low","high")] for r in pairs])
        assert set(indices[:len(train_pairs)].ravel())<=set(tr) and set(indices[len(train_pairs):].ravel())<=set(te)
        folder = OUT/f"fold-{f}"
        folder.mkdir(parents=True,exist_ok=True)
        archives = {}
        validation = {}
        selected_inputs = []
        for a,arm in enumerate(exp.ARMS):
            archive = folder/(arm+".npz")
            contributions = np.zeros((50,len(pairs),len(frame.columns)))
            preds,features,errors = [],[],[]
            for i,s in enumerate(sources[a]):
                parts,names,pred,replay,error = member(s,frame,indices,te,refs[a][i],arm)
                idx = frame.columns.get_indexer(names)
                assert idx.min()>=0
                contributions[i][:,idx] = parts
                preds.append(pred)
                features.append(idx.tolist())
                errors.append((replay,error))
                tf.keras.backend.clear_session()
            np.savez_compressed(archive,terms=contributions,pred=preds,official_test=official[a])
            archives[arm] = file_hash(archive)
            validation[arm] = dict(zip(("replay_max","conservation_max"),np.max(errors,axis=0)))
            selected_inputs.append(features)
        assert selected_inputs[0]==selected_inputs[1]
        write_json(folder/"design.json",dict(fold=f, ids=ids, columns=list(frame.columns),
                   train_indices=tr, test_indices=te, train_pairs=train_pairs,test_pairs=test_pairs,
                   selected_inputs=selected_inputs[0],sha256=archives,validation=validation,
                   code_sha256=file_hash(__file__),upstream_code_sha256=file_hash(obs.__file__),
                   input_sha256=file_hash(exp.steel.INPUT/"inputs.npz")))
        print(f"Verified all inputs in {f}: {len(train_pairs)} training and {len(test_pairs)} test pairs",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds",nargs="+",type=int,default=list(range(5)))
    run(parser.parse_args().folds)
