"""Trace an exact one-point nickel-equivalent latent intervention to native MPa output terms."""
import argparse
import pickle
from pathlib import Path

import audit_steel_concept_directions as concept
import audit_steel_physical_strength as study
import discover_physical_kan_responses as exp
from scipy.spatial import cKDTree
from run_modnet_official_matched import file_hash, np, tf, write_json


def support(h,z,d):
    """Test both edited endpoints in training PCA space, with a training-only distance cutoff."""
    mean=h.mean(0)
    _,s,v=np.linalg.svd(h-mean,full_matrices=False)
    n=min(12,int(np.searchsorted(np.cumsum(s*s),.95*np.sum(s*s)))+1,int(sum(s>s[0]*1e-6)))
    transform=v[:n].T*(np.sqrt(len(h))/s[:n])
    ht=(h-mean)@transform
    tree=cKDTree(ht)
    distance=tree.query(ht,k=2)[0][:,1]
    cutoff=float(np.quantile(distance,.95))
    def covered(value):
        return tree.query((value-mean)@transform,k=1)[0]<=cutoff+1e-10
    return covered(z)&covered(z-.5*d)&covered(z+.5*d),dict(pcs=n,train_distance_cutoff=cutoff)


def native_terms(arm,model,layers,late):
    if arm=="kan":
        i=sorted(k for k,layer in enumerate(layers) if hasattr(layer,"coefficients"))[-1]
        last=layers[i]
        weights=last.coefficients.numpy()[:,0,:].astype(float)*model.target_scaling["std"]
        def values(h):
            b=exp.basis(h,last.hidden_coordinates)
            return np.einsum("nij,ij->ni",b,weights)
        return values,layers[i:]
    i=max(k for k,layer in enumerate(layers[:-1]) if layer.activation.__name__!="linear")
    tail=layers[i+1:]
    w=np.ones(1)
    for layer in reversed(tail):
        w=layer.get_weights()[0]@w
    assert w.shape==(late.shape[1],)
    return lambda h:h*w,tail


def inspect(arm,record,source,frame,ids,j,stored):
    train_ids=source[3]
    assert file_hash(source[0])==source[2]
    with source[0].open("rb") as stream:
        model=pickle.load(stream)
    model._restore_model()
    cols=list(model.optimal_descriptors[:model.n_feat])
    x=exp.p.scaled(model,frame.loc[train_ids+ids,cols].values)
    rep,pred,_,layers,_,_=exp.p.forward(model,x,arm=="kan")
    hidden=rep["late"].astype(float)
    h,z=np.split(hidden,[len(train_ids)])
    np.testing.assert_allclose(pred[len(h):],stored[arm+"_base"],atol=.01,rtol=1e-6)
    w=np.asarray(record["readout_weights"])[:,j]
    score=z@w+record["intercept"][j]
    # Replaying the float32 network in a fresh process changes the last few bits.
    np.testing.assert_allclose(score,stored[arm+"_decoded"][:,j],atol=2e-5,rtol=1e-6)
    native,tail=native_terms(arm,model,layers,hidden)
    meta,arrays={},{}
    for mode,axis in record["axes"][j]["modes"].items():
        d=np.asarray(axis["direction"])
        np.testing.assert_allclose(w@d,1,atol=1e-6)
        lower,upper=hidden-.5*d,hidden+.5*d
        terms=native(upper)-native(lower)
        upper_pred=concept.propagate(tail,upper)
        lower_pred=concept.propagate(tail,lower)
        full=upper_pred-lower_pred
        np.testing.assert_allclose(terms.sum(1),full,atol=.012,rtol=1e-6)
        tr,te=np.split(terms,[len(h)])
        delta=full[len(h):]
        selected=[]
        residual=full[:len(h)].copy()
        for _ in range(min(3,terms.shape[1])):
            gain=2*(tr.T@residual)-np.sum(tr*tr,axis=0)
            gain[selected]=-np.inf
            k=int(np.argmax(gain))
            selected.append(k)
            residual-=tr[:,k]
        inside,info=support(h,z,d)
        endpoints=np.column_stack([lower_pred[len(h):],upper_pred[len(h):]])
        official=exp.steel.native_postprocess(endpoints,model,source[1],stored[arm+"_official"].dtype)
        arrays.update({mode+"_total":delta,mode+"_official":official[:,1].astype(float)-official[:,0],
                       mode+"_one":te[:,selected[0]],mode+"_three":te[:,selected].sum(1),
                       mode+"_inside":inside,mode+"_individual":te[:,selected]})
        meta[mode]=dict(nodes=selected,latent_support=info,term_replay_max=np.max(abs(terms.sum(1)-full)))
    return meta,arrays


def run(folds,out):
    data=exp.load_data("steel",out)
    j=study.COMPOSITE.index("ni_equivalent")
    code=file_hash(__file__)
    for f in folds:
        sources,_,_,ids=exp.load_fold("steel",f,data)
        folder=out/f"fold-{f}"
        (folder/"nickel").mkdir(exist_ok=True)
        for m in range(50):
            parent=folder/"latent"/f"member-{m:02d}.json"
            doc=exp.steel.read(parent)
            assert doc["code_sha256"]==file_hash(concept.__file__)
            assert doc["arrays_sha256"]==file_hash(parent.with_suffix(".npz"))
            digest=file_hash(parent)
            path=folder/"nickel"/parent.name
            archive=path.with_suffix(".npz")
            if path.exists():
                saved=exp.steel.read(path)
                if saved["code_sha256"]==code and saved["parent_sha256"]==digest and saved["arrays_sha256"]==file_hash(archive):
                    continue
            records,arrays={},dict(ids=ids)
            with np.load(parent.with_suffix(".npz")) as stored:
                np.testing.assert_array_equal(stored["ids"],ids)
                for arm,src in zip(exp.ARMS,sources):
                    records[arm],values=inspect(arm,doc[arm],src[m],data[0],ids,j,stored)
                    arrays.update({arm+"_"+k:v for k,v in values.items()})
                    tf.keras.backend.clear_session()
            np.savez_compressed(archive,**arrays)
            write_json(path,dict(fold=f,member=m,code_sha256=code,parent_sha256=digest,arrays_sha256=file_hash(archive),**records))
            if (m+1)%25==0:
                print("nickel",f,m+1,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--folds",type=int,nargs="+",choices=range(5),default=list(range(5)))
    cli.add_argument("--output",type=Path,default=study.OUT)
    args=cli.parse_args()
    run(args.folds,args.output)
