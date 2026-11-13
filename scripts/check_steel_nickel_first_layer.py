"""Check whether upstream nonlinear layers give the MLP material-specific nickel responses."""
import argparse
import pickle
from pathlib import Path

import audit_steel_concept_directions as concept
import audit_steel_physical_strength as study
import discover_physical_kan_responses as exp
from run_modnet_official_matched import file_hash, np, tf, write_json


def replay(arm, source, frame, quantity, ids, raw_reference):
    path, _, source_hash, train_ids = source
    assert file_hash(path) == source_hash
    with path.open("rb") as stream:
        model = pickle.load(stream)
    model._restore_model()
    columns = list(model.optimal_descriptors[:model.n_feat])
    x = exp.p.scaled(model, frame.loc[train_ids+ids,columns].values)
    rep,pred,_,layers,_,_ = exp.p.forward(model,x,arm=="kan")
    h,z = np.split(rep["first"].astype(float),[len(train_ids)])
    base = concept.propagate(layers[1:],z)
    np.testing.assert_allclose(base,pred[len(h):],atol=.01,rtol=1e-6)
    np.testing.assert_allclose(base,raw_reference,atol=.01,rtol=1e-6)
    train_q = quantity.loc[train_ids].values[:,None]
    test_q = quantity.loc[ids].values[:,None]
    probe = exp.steel.fit_readouts(h,z,train_q,test_q)
    w = probe["coefficients"]
    _,_,axes = concept.learn_directions(h,w)
    direction = axes[0][1]["marginal"][0]
    np.testing.assert_allclose(w[:,0]@direction,1,atol=1e-6)
    lower,upper = concept.propagate(layers[1:],np.stack([z-.5*direction,z+.5*direction]))
    return probe["decoded"][:,0], upper-lower


def run(folder, folds):
    data = exp.load_data("steel",folder)
    frame = data[0]
    quantity = exp.steel.concept_frame(frame)["ni_equivalent"]
    for f in folds:
        sources,reference,_,ids = exp.load_fold("steel",f,data)
        path = folder/f"fold-{f}/nickel-first-layer.json"
        archive = path.with_suffix(".npz")
        hashes = {k:file_hash(p) for k,p in dict(code=__file__,latent=concept.__file__).items()}
        model_hashes = [[s[2] for s in src] for src in sources]
        if path.exists():
            saved = exp.steel.read(path)
            if saved["hashes"] == hashes and saved["sources"] == model_hashes and saved["arrays_sha256"] == file_hash(archive):
                continue
        arrays = dict(ids=ids, ni_equivalent=quantity.loc[ids].values)
        for arm,src,ref in zip(exp.ARMS,sources,reference):
            decoded,delta = [],[]
            for m,model_source in enumerate(src):
                score,effect = replay(arm,model_source,frame,quantity,ids,ref[m])
                decoded.append(score)
                delta.append(effect)
                tf.keras.backend.clear_session()
            arrays[arm+"_decoded"] = np.stack(decoded)
            arrays[arm+"_delta"] = np.stack(delta)
        np.savez_compressed(archive,**arrays)
        write_json(path,dict(fold=f,hashes=hashes,sources=model_hashes,arrays_sha256=file_hash(archive)))
        print("first layer",f,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=study.OUT)
    parser.add_argument("--folds",type=int,nargs="+",choices=range(5),default=list(range(5)))
    args = parser.parse_args()
    run(args.output,args.folds)
