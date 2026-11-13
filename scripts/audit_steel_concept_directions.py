"""Intervene on one decoded metallurgy coordinate while preserving the other three readouts."""
import argparse
import pickle
from pathlib import Path

import audit_steel_physical_strength as study
import discover_physical_kan_responses as exp
from run_modnet_official_matched import file_hash, np, tf, write_json


NAMES=study.COMPOSITE
MODES=("marginal","conditional")
RANDOM_DRAWS=6


def propagate(layers,hidden):
    arr=np.asarray(hidden,dtype=np.float32)
    shape=arr.shape[:-1]
    flat=arr.reshape(-1,arr.shape[-1])
    responses=[]
    for start in range(0,len(flat),512):
        x=tf.convert_to_tensor(flat[start:start+512])
        for layer in layers:
            x=layer(x,training=False)
        responses.append(np.asarray(x).reshape(-1))
    return np.concatenate(responses).astype(float).reshape(shape)


def learn_directions(h,w):
    """Minimum training-Mahalanobis displacement for a unit readout change."""
    h=h-h.mean(0)
    eigen,vectors=np.linalg.eigh(h.T@h/len(h))
    supported=eigen>max(eigen[-1]*1e-8,1e-15)
    factor=vectors[:,supported]*np.sqrt(eigen[supported])
    axes=factor.T@w
    normalized=axes/np.maximum(np.linalg.norm(axes,axis=0),1e-15)
    all_basis,s,_=np.linalg.svd(normalized,full_matrices=False)
    all_basis=all_basis[:,s>1e-8]
    columns=[]
    for j,a in enumerate(axes.T):
        other=np.delete(normalized,j,axis=1)
        u,s,_=np.linalg.svd(other,full_matrices=False)
        u=u[:,s>1e-8]
        residual=a-u@(u.T@a)
        fraction=np.linalg.norm(residual)/max(np.linalg.norm(a),1e-15)
        candidates={"marginal":a}
        if fraction>=.2:
            candidates["conditional"]=residual
        modes={}
        for key,v in candidates.items():
            denominator=a@v
            if denominator>1e-14:
                white=v/denominator
                d=factor@white
                np.testing.assert_allclose(w[:,j]@d,1,rtol=1e-6,atol=1e-6)
                if key=="conditional":
                    np.testing.assert_allclose(np.delete(w.T@d,j),0,atol=1e-5)
                modes[key]=(d,white)
        columns.append((fraction,modes))
    return factor,all_basis,columns


def audit(arm,source,frame,ids,q,scale,center,raw_reference,official,seed0):
    path,seed,hashed,tr=source
    assert file_hash(path)==hashed
    with path.open("rb") as stream:
        model=pickle.load(stream)
    model._restore_model()
    columns=list(model.optimal_descriptors[:model.n_feat])
    x=exp.p.scaled(model,frame.loc[tr+ids,columns].values)
    rep,pred,_,layers,_,_=exp.p.forward(model,x,arm=="kan")
    idx=(sorted(k for k,l in enumerate(layers) if hasattr(l,"coefficients"))[1] if arm=="kan" else
         max(k for k,l in enumerate(layers[:-1]) if l.activation.__name__!="linear"))
    h,z=np.split(rep["late"].astype(float),[len(tr)])
    tail=layers[idx+1:]
    base=propagate(tail,z)
    np.testing.assert_allclose(base,raw_reference,atol=.01,rtol=1e-6)
    np.testing.assert_allclose(base,pred[len(tr):],atol=.005,rtol=1e-6)
    qtr=q.loc[tr,list(NAMES)].values
    qt=q.loc[ids,list(NAMES)].values
    fit=exp.steel.fit_readouts(h,z,qtr,qt)
    w=fit["coefficients"]
    bias=fit["intercept"]
    decoded=z@w+bias
    dtr=h@w+bias
    np.testing.assert_allclose(decoded,fit["decoded"],rtol=1e-7,atol=1e-7)
    rng=np.random.RandomState(seed0)
    shuffled=exp.steel.fit_readouts(h,z,qtr[rng.permutation(len(qtr))],qt)["decoded"]
    factor,all_basis,dirs=learn_directions(h,w)
    arrays=dict(base=base,official=official,decoded=decoded,null=shuffled)
    records=[]
    for j,name in enumerate(NAMES):
        frac,available=dirs[j]
        step=.1*scale[j]
        shift=np.clip(center[j]-decoded[:,j],-.5*scale[j],.5*scale[j])
        lo,hi=np.quantile(dtr[:,j],[.05,.95])
        arrays[f"{j}_inside"]=(decoded[:,j]-step>=lo)&(decoded[:,j]+step<=hi)
        record=dict(name=name,unique_train_sd_fraction=frac,modes={},train_score_limits=[lo,hi])
        for mode in MODES:
            if mode not in available:
                continue
            direction,white=available[mode]
            modified=z+shift[:,None]*direction
            variants=np.stack([modified,z-step*direction,z+step*direction])
            new,lower,higher=propagate(tail,variants)
            arrays[f"{j}_{mode}_center"]=new
            arrays[f"{j}_{mode}_official"]=exp.steel.native_postprocess(new[:,None],model,seed,official.dtype)[:,0]
            arrays[f"{j}_{mode}_local"]=higher-lower
            np.testing.assert_allclose(modified@w[:,j]+bias[j],decoded[:,j]+shift,rtol=1e-7,atol=1e-6)
            if mode=="conditional":
                np.testing.assert_allclose(np.delete(modified@w+bias, j,axis=1),np.delete(decoded,j,axis=1),
                                           rtol=1e-7,atol=1e-6)
            record["modes"][mode]=dict(direction=direction,whitened_local_length=step*np.linalg.norm(white))
            if mode=="conditional" and factor.shape[1]>all_basis.shape[1]:
                random=rng.normal(size=(factor.shape[1],RANDOM_DRAWS))
                random-=all_basis@(all_basis.T@random)
                random/=np.maximum(np.linalg.norm(random,axis=0),1e-15)
                random=(factor@random*np.linalg.norm(white)).T
                np.testing.assert_allclose(random@w,0,atol=1e-5)
                rand_pred=propagate(tail,z[None,:,:]+shift[None,:,None]*random[:,None,:])
                arrays[f"{j}_matched_random"]=rand_pred-base
        records.append(record)
    meta=dict(nodes=h.shape[1],readout_weights=w,intercept=bias,axes=records,
              source_sha256=hashed,rank=factor.shape[1])
    return meta,arrays


def run(folds,out):
    data=exp.load_data("steel",out)
    frame,target,splits=data
    q,_=study.chemistry(frame)
    code=file_hash(__file__)
    for fold in folds:
        sources,ref,official,ids=exp.load_fold("steel",fold,data)
        q25,center,q75=np.quantile(q.iloc[splits[fold][0]][list(NAMES)].values,[.25,.5,.75],axis=0)
        folder=out/f"fold-{fold}"/"latent"
        folder.mkdir(parents=True,exist_ok=True)
        for m in range(50):
            path=folder/f"member-{m:02d}.json"
            archive=path.with_suffix(".npz")
            hashes=[s[m][2] for s in sources]
            if path.exists():
                saved=exp.steel.read(path)
                if saved["code_sha256"]==code and saved["direct_code_sha256"]==file_hash(study.__file__) and saved["models"]==hashes and saved["arrays_sha256"]==file_hash(archive):
                    continue
            meta={}
            arrays=dict(ids=ids,target=target.loc[ids].values,q=q.loc[ids,list(NAMES)].values)
            for a,src,raw,score in zip(exp.ARMS,sources,ref,official):
                meta[a],values=audit(a,src[m],frame,ids,q,q75-q25,center,raw[m],score[m],81112+1000*fold+m)
                arrays.update({a+"_"+k:v for k,v in values.items()})
                tf.keras.backend.clear_session()
            np.savez_compressed(archive,**arrays)
            write_json(path,dict(fold=fold,member=m,models=hashes,code_sha256=code,direct_code_sha256=file_hash(study.__file__),
                      arrays_sha256=file_hash(archive),center=center,iqr=q75-q25,names=NAMES,**meta))
            if (m+1)%25==0:
                print("latent",fold,m+1,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--folds",nargs="+",type=int,choices=range(5),default=list(range(5)))
    cli.add_argument("--output",type=Path,default=study.OUT)
    args=cli.parse_args()
    run(args.folds,args.output)
