"""Trace native Mo paths on observed steels in the frozen five-fold comparison."""
import argparse
import pickle

import audit_steel_observed_pairs as obs
from audit_steel_observed_pairs import exp, file_hash, np, tf, write_json
from audit_steel_physical_strength import OUT as MATCH_ARCHIVE, match_real
from pymatgen.core import Element


OUT = exp.ROOT / "benchmarks/steel-observed-molybdenum-20261112"
MO = "ElementFraction|Mo"
CASE = ("mb-steels-073", "mb-steels-042")


def chemistry(frame):
    atomic = frame.filter(regex=r"^ElementFraction\|")
    atomic = atomic.loc[:, atomic.ne(0).any()].copy()
    atomic.columns = [col.split("|")[1] for col in atomic.columns]
    mass = atomic * [float(Element(e).atomic_mass) for e in atomic]
    return 100 * mass.div(mass.sum(axis=1), axis=0), np.sqrt(atomic.values / 2.)


def natural_pairs(wt, split_by_id):
    """Screen all real pairs by composition, independently of yield or predictions."""
    x = wt.drop(columns="Fe")
    i, j = np.triu_indices(len(wt), 1)
    d = x.values[j] - x.values[i]
    col = np.argmax(np.abs(d), axis=1)
    largest = d[np.arange(len(i)), col]
    other = np.abs(d).sum(axis=1) - np.abs(largest)
    eligible = (np.abs(largest) >= .1) & (other <= .2) & (other <= .1 * np.abs(largest))
    pairs = []
    for k in np.flatnonzero(eligible):
        a, b = (i[k], j[k]) if largest[k] > 0 else (j[k], i[k])
        lo, hi = wt.index[[a,b]]
        pairs.append(dict(low=lo, high=hi, element=x.columns[col[k]], gap_wt=abs(largest[k]),
                          other_l1_wt=other[k], low_fold=split_by_id[lo], high_fold=split_by_id[hi]))
    return pairs


def mo_edges(state, x, col, lo, hi, kan):
    if kan:
        return (state[1][hi,col] - state[1][lo,col]) @ state[2][col].T
    delta = x[hi,col] - x[lo,col]
    return delta[:, None] * state[2][col] * obs.activation_slopes(state, lo, hi)


def audit(arm, source, frame, q, test_ids, train_pairs, pairs, raw, official, intermediate):
    path, _, digest, train = source
    assert file_hash(path) == digest
    with path.open("rb") as stream:
        model = pickle.load(stream)
    model._restore_model()
    names = list(model.optimal_descriptors[:model.n_feat])
    all_ids = train + test_ids
    pos = {s:i for i,s in enumerate(all_ids)}
    assert len(pos) == len(all_ids)
    x = exp.p.scaled(model, frame.loc[all_ids,names].values).astype(float)
    layers = [l for l in model.model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    kan = arm == "kan"
    scale, offset = 1., 0.
    if kan:
        assert isinstance(layers[-1], tf.keras.layers.Rescaling)
        assert model.target_scaling["fit_ids"] == train
        scale, offset = model.target_scaling["std"], model.target_scaling["mean"]
        layers = layers[:-1]
    else:
        assert any(l.activation.__name__ != "linear" for l in layers[1:])
    states, h, pred = obs.forward(layers, x, kan, scale, offset)
    n = len(train)
    replay_error = abs(pred[n:] - raw).max()
    np.testing.assert_allclose(pred[n:], raw, rtol=1e-6, atol=.02)
    complete = train_pairs + pairs
    lo = np.array([pos[p["low"]] for p in complete])
    hi = np.array([pos[p["high"]] for p in complete])
    terms, back = obs.first_contributions(states, h, lo, hi, kan, scale)
    total = pred[hi] - pred[lo]
    np.testing.assert_allclose(terms.sum(axis=1),total,atol=1e-5,rtol=1e-10)
    t = len(train_pairs)
    assert t > 0
    k = int(np.ceil(.1*h.shape[1]))
    order = obs.rank_strength(terms[:t], total[:t], max(3,k))
    physical, _, _, decoded = exp.steel.three_node_probe(h[:n], h[n:], q.loc[train].values[:,None], .01)
    physical = physical[0]
    out = dict(total=raw[hi[t:]-n]-raw[lo[t:]-n], official=official[hi[t:]-n]-official[lo[t:]-n],
               raw_endpoint=raw, official_endpoint=official, decoded_mo_at_pct=decoded[:,0],
               physical3=terms[t:][:,physical].sum(axis=1), strength3=terms[t:][:,order[:3]].sum(axis=1),
               strength10pct=terms[t:][:,order[:k]].sum(axis=1))
    record = dict(sha256=digest, width=h.shape[1], physical3_nodes=physical, strength3_nodes=order[:3],
                  has_mo=MO in names, replay_error=replay_error,
                  conservation_error=abs(terms.sum(axis=1)-total).max(), n_train_pairs=t)
    for key in ("direct_all", "direct3", "direct_via_physical3", "all_inputs_via_direct3"):
        out[key] = np.zeros(len(pairs))
    if MO in names:
        col = names.index(MO)
        changes = mo_edges(states[0], x, col, lo, hi, kan)
        mo_mpa = changes * back
        direct = mo_mpa.sum(axis=1)
        nodes = obs.rank_strength(mo_mpa[:t], direct[:t], 3)
        out.update(direct_all=direct[t:], direct3=mo_mpa[t:][:,nodes].sum(axis=1),
                   direct_via_physical3=mo_mpa[t:][:,physical].sum(axis=1),
                   all_inputs_via_direct3=terms[t:][:,nodes].sum(axis=1))
        record["direct3_nodes"] = nodes
        # Check both the edge and its total-node attribution against all input paths.
        _, all_paths = obs.feature_terms(states[0], x, lo[t:], hi[t:], nodes, back[t:], kan)
        np.testing.assert_allclose(all_paths[:,col,:].sum(axis=1),out["direct3"],atol=1e-6,rtol=1e-8)
        np.testing.assert_allclose(all_paths.sum(axis=1),terms[t:][:,nodes],atol=1e-6,rtol=1e-8)
        if intermediate is not None:
            case_index = next(i for i,p in enumerate(pairs) if p["kind"]=="case")
            a, b, c = pos[CASE[0]], pos[intermediate], pos[CASE[1]]
            starts, ends = np.array([a,a,b]), np.array([c,b,c])
            edge_values = mo_edges(states[0], x, col, starts, ends, kan)[:,nodes]
            sd = h[:n, nodes].std(axis=0)
            assert np.min(sd) > 1e-9
            error = (edge_values[1]+edge_values[2]-edge_values[0]) / sd
            if kan:
                assert np.max(np.abs(error)) < 1e-10
            trace = dict(check_ids=[CASE[0], intermediate, CASE[1]], nodes=nodes,
                         direct=edge_values[0], first_leg=edge_values[1], second_leg=edge_values[2],
                         difference_sd=error, direct_mpa=mo_mpa[t+case_index,nodes],
                         node_mpa=terms[t+case_index,nodes])
            if kan:
                trace["curve_values"] = states[0][1][[a,b,c],col] @ states[0][2][col,nodes].T
            record["trace"] = trace
    return record, out, names


def run(folds):
    data = exp.load_data("steel",OUT)
    frame, target, splits = data
    q = 100*frame[MO]
    wt, coord = chemistry(frame)
    ids = frame.index.tolist()
    id_fold = {ids[i]:f for f,(_,te) in enumerate(splits) for i in te}
    natural = natural_pairs(wt,id_fold)
    for f in folds:
        src, raw, official, held = exp.load_fold("steel",f,data)
        tr, te = splits[f]
        matched = match_real(q.values,coord,tr,te,ids)["pairs"]
        saved = exp.steel.read(MATCH_ARCHIVE/f"fold-{f}/design.json")["pairs"]["molybdenum"]["pairs"]
        assert [(r["low"],r["high"]) for r in matched] == [(r["low"],r["high"]) for r in saved]
        pairs = [dict(**r,kind="matched") for r in matched]
        middle = None
        if set(CASE) <= set(held):
            case = next(r for r in natural if (r["low"],r["high"])==CASE)
            pairs.append(dict(**case,kind="case"))
            # Select a third real held-out steel by its Mo coordinate only, for the edge check.
            qs = q.loc[held]
            candidates = qs[(qs>q[CASE[0]]) & (qs<q[CASE[1]])]
            middle = min(candidates.index,key=lambda s:(abs(q[s]-(q[CASE[0]]+q[CASE[1]])/2),s))
        for r in pairs:
            r["measured"] = target[r["high"]] - target[r["low"]]
        folder = OUT/f"fold-{f}"
        folder.mkdir(parents=True,exist_ok=True)
        arrays, records = {}, []
        for i in range(50):
            train = src[0][i][3]
            pos = frame.index.get_indexer(train)
            fit_pairs = match_real(q.values,coord,pos,pos,ids)["pairs"]
            rec, features = {}, []
            for a, arm in enumerate(exp.ARMS):
                rec[arm], result, names = audit(arm,src[a][i],frame,q,held,fit_pairs,pairs,
                                                raw[a][i],official[a][i],middle)
                features.append(names)
                for key,v in result.items():
                    arrays.setdefault(arm+"_"+key,[]).append(v)
            assert features[0] == features[1]
            records.append(rec)
            tf.keras.backend.clear_session()
        archive = folder/"members.npz"
        np.savez_compressed(archive, **arrays)
        write_json(folder/"audit.json", dict(fold=f,test_ids=held,measured=target.loc[held].values,
                   wt=wt.loc[held].to_dict(orient="index"),q=q.loc[held].values,pairs=pairs,middle=middle,
                   natural_same_fold=[r for r in natural if r["low_fold"]==r["high_fold"]==f],
                   natural_mo=[r for r in natural if r["element"]=="Mo" and r["low_fold"]==f],
                   members=records, code_sha256=file_hash(__file__), observed_code_sha256=file_hash(obs.__file__),
                   input_sha256=file_hash(exp.steel.INPUT/"inputs.npz"), arrays_sha256=file_hash(archive)))
        print(f"Verified fold {f}, 50 paired models and {len(matched)} real Mo-matched pairs",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    run(parser.parse_args().folds)
