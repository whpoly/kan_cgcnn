"""Trace named composition inputs through the same three strength-selected first-layer nodes."""
import argparse
import pickle
from pathlib import Path

import audit_steel_equal_nodes as audit
import discover_physical_kan_responses as exp
from run_modnet_official_matched import file_hash, np, tf, write_json


DELTA = "YangSolidSolution|Yang delta"
VALENCE = "ElementProperty|MagpieData avg_dev NValence"


def trace(arm,source,columns,features,ids,nodes):
    path,_,expected,train_ids = source
    assert expected==file_hash(path)
    with path.open("rb") as fp:
        model = pickle.load(fp)
    model._restore_model()
    names = list(model.optimal_descriptors[:model.n_feat])
    assert names == columns
    layers = [l for l in model.model.layers if not isinstance(l,tf.keras.layers.InputLayer)]
    first,tail = layers[0],layers[1:]
    scale = lambda frame: exp.p.scaled(model,frame.loc[:,names].values)
    x0 = scale(features["original"].loc[ids])
    x1 = scale(features["nickel"].loc[ids])
    xt = scale(features["original"].loc[train_ids])
    low = first(x0,training=False).numpy()
    high = first(x1,training=False).numpy()
    y0,y1 = audit.concept.propagate(tail,np.stack([low,high]))
    if arm=="kan":
        weights = first.coefficients.numpy()[:,nodes]
    else:
        assert first.activation.__name__=="elu"
        weights,bias = first.get_weights()
        weights,bias = weights[:,nodes].astype(float),bias[nodes].astype(float)
        train_pre = xt@weights+bias
    results = dict(full=y1-y0)
    namesets = dict(delta=[j for j,name in enumerate(names) if name==DELTA],
                    valence=[j for j,name in enumerate(names) if name==VALENCE],
                    named=[j for j,name in enumerate(names) if not name.startswith("ROOST_")])
    info = dict(groups={g:[names[j] for j in group] for g,group in namesets.items()})
    for key,ix in namesets.items():
        into_low,into_high = x0.copy(),x1.copy()
        into_low[:,ix],into_high[:,ix] = x1[:,ix],x0[:,ix]
        add = first(into_low,training=False).numpy()[:,nodes]-low[:,nodes]
        rem = high[:,nodes]-first(into_high,training=False).numpy()[:,nodes]
        actual = .5*(add+rem)
        if not ix:
            portable = np.zeros_like(actual)
        elif arm=="kan":
            db = exp.basis(x1[:,ix],first.hidden_coordinates)-exp.basis(x0[:,ix],first.hidden_coordinates)
            portable = np.einsum("nib,ijb->nj",db,weights[ix])
            np.testing.assert_allclose(portable,actual,atol=2e-5,rtol=1e-5)
        else:
            background = train_pre-xt[:,ix]@weights[ix]
            s0,s1 = x0[:,ix]@weights[ix],x1[:,ix]@weights[ix]
            # Both architectures receive the same donor descriptor values. The fixed MLP
            # response is its exact joint nonlinear response averaged over training contexts.
            portable = (exp.activation(first,background[:,None,:]+s1)-
                        exp.activation(first,background[:,None,:]+s0)).mean(0)
        outputs = {}
        for label,d in (("contextual",actual),("portable",portable)):
            a,b = low.copy(),high.copy()
            a[:,nodes] += d
            b[:,nodes] -= d
            pa,pb = audit.concept.propagate(tail,np.stack([a,b]))
            outputs[label] = .5*(pa-y0+y1-pb)
        info[key] = dict(node_portable_error_rms=np.mean((portable-actual)**2)**.5,
                         node_contextual_rms=np.mean(actual**2)**.5)
        results.update({key+"_"+label:v for label,v in outputs.items()})
    return results,info


def run(out,folds):
    data = exp.load_data("steel",out)
    features,hashes = audit.load_recipes(out,data[0])
    code = file_hash(__file__)
    for f in folds:
        sources,_,_,ids = exp.load_fold("steel",f,data)
        folder = out/f"fold-{f}"
        parent_paths = [folder/f"physical/member-{m:02d}.json" for m in range(50)]
        parents = [file_hash(p) for p in parent_paths]
        dst = folder/"named-trace.json"
        archive = dst.with_suffix(".npz")
        if dst.exists():
            saved = exp.steel.read(dst)
            if saved["code_sha256"]==code and saved["parents"]==parents and saved["arrays_sha256"]==file_hash(archive):
                continue
        data_by_arm,meta = {},{}
        for arm,src in zip(exp.ARMS,sources):
            items,records = [],[]
            source_audit = exp.steel.read(exp.steel.OUT/f"fold-{f}/{arm}.json")["members"]
            for m,parent in enumerate(parent_paths):
                group = exp.steel.read(parent)[arm]["groups"][1]
                vals,record = trace(arm,src[m],source_audit[m]["input_features"],features,ids,group)
                with np.load(parent.with_suffix(".npz")) as s:
                    np.testing.assert_allclose(vals["full"],s[arm+"_nickel_full"],atol=.015,rtol=1e-6)
                items.append(vals)
                records.append(record)
                tf.keras.backend.clear_session()
            data_by_arm.update({arm+"_"+k:np.stack([v[k] for v in items]) for k in items[0]})
            meta[arm]=records
        d = 100*(features["nickel"].loc[ids,DELTA]-features["original"].loc[ids,DELTA]).values
        v = (features["nickel"].loc[ids,VALENCE]-features["original"].loc[ids,VALENCE]).values
        np.savez_compressed(archive,ids=ids,delta_percentage_points=d,valence_electrons=v,**data_by_arm)
        write_json(dst,dict(fold=f,code_sha256=code,parents=parents,features=hashes,arrays_sha256=file_hash(archive),**meta))
        print("named",f,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds",type=int,nargs="+",choices=range(5),default=list(range(5)))
    parser.add_argument("--output",type=Path,default=audit.OUT)
    args=parser.parse_args()
    run(args.output,args.folds)
