"""Test the strength impact of physically defined steel composition coordinates on paired models."""
import argparse
from pathlib import Path

import audit_kan_turning_points as turns
import discover_physical_kan_responses as exp
from matminer.utils.data import MagpieData
from pymatgen.core import Element
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from run_modnet_official_matched import file_hash, np, tf, write_json


OUT = exp.ROOT / "benchmarks/steel-physical-strength-20261112"
DIRECT = ("size_mismatch", "chromium", "molybdenum", "titanium", "carbon", "nickel")
COMPOSITE = ("size_mismatch", "carbon_nitrogen", "cr_equivalent", "ni_equivalent")
MATCHED = tuple(dict.fromkeys(COMPOSITE + DIRECT))
QUANTILES = (.1, .25, .5, .75, .9)
EDITS = ("all_center", "one_center", "all_chord", "one_chord")
DESIGN = dict(primary="Yang atomic-size mismatch in percent, recomputed from each real formula.",
    direct="Outer-training quantiles supply a common dose grid for every member of both arms. The strongest receiving node per quantity and arm is selected on 64 fixed actual member-training rows by mean absolute original output gradient times the median-reference node change.",
    intervention="Actual outer input with all other coordinates held fixed. Compare native high/low, center dose, and q10-q90 secant on one node or all first-layer routes from the descriptor. Absent direct columns have identity intervention. Outputs are raw MPa, label MAE follows original member repair.",
    matched="Disjoint real pairs per outer fold. Low/high are defined by outer-training terciles and must differ by at least half the outer-training IQR. Match full nominal atomic composition by Hellinger distance. Caliper is training 80th percentile of within-training nearest eligible opposite-group distance. Maximize pair count then minimize distance; no predictions or labels used.")


def chemistry(frame):
    q = exp.steel.concept_frame(frame)
    fractions = frame.filter(regex=r"^ElementFraction\|")
    fractions = fractions.loc[:, fractions.ne(0).any()]
    elements = [Element(n.split("|")[1]) for n in fractions.columns]
    radii = np.array(MagpieData().get_elemental_properties(elements, "MiracleRadius"))
    c = fractions.values
    np.testing.assert_allclose(c.sum(1), 1, atol=1e-12)
    mean_r = c @ radii
    delta = 100*np.sqrt(np.sum(c*(1-radii/mean_r[:,None])**2, axis=1))
    np.testing.assert_allclose(delta, q["size_mismatch"], atol=1e-8, rtol=0)
    return q, np.sqrt(c)/np.sqrt(2)


def match_real(q, coordinates, train, test, ids):
    low_cut,hi_cut=np.quantile(q[train],[1/3,2/3])
    iqr=np.diff(np.quantile(q[train],[.25,.75]))[0]
    def groups(pos):
        lo=pos[q[pos]<=low_cut]
        hi=pos[(q[pos]>=hi_cut)&(q[pos]>low_cut)]
        d=cdist(coordinates[lo],coordinates[hi])
        valid=(q[hi][None,:]-q[lo][:,None]) >= max(.5*iqr,1e-9)
        return lo,hi,d,valid
    _,_,dt,vt=groups(train)
    lo,hi,d,valid=groups(test)
    pairs=[]
    if not len(lo) or not len(hi) or dt.size==0 or not np.any(vt):
        return dict(terciles=[low_cut,hi_cut],iqr=iqr,caliper=None,pairs=pairs)
    ds=np.where(vt,dt,np.inf)
    mins=np.r_[ds.min(0),ds.min(1)]
    caliper=np.quantile(mins[np.isfinite(mins)],.8)
    valid &= d<=caliper
    rows,cols=linear_sum_assignment(np.where(valid,d,1000.))
    for a,b in zip(rows,cols):
        if valid[a,b]:
            pairs.append(dict(low=ids[lo[a]],high=ids[hi[b]],distance=d[a,b],
                              q_low=q[lo[a]],q_high=q[hi[b]]))
    assert len({p[k] for p in pairs for k in ("low","high")})==2*len(pairs)
    return dict(terciles=[low_cut,hi_cut],iqr=iqr,caliper=caliper,pairs=pairs)


def hidden_grid(arm,x,col,scaled):
    changed=np.tile(x,(len(scaled),1,1))
    changed[:,:,col]=np.asarray(scaled)[:,None]
    value=arm.first(changed.reshape(-1,x.shape[1]),training=False).numpy()
    return value.reshape(len(scaled),len(x),-1)


def train_gradient(arm,x):
    h=arm.first(x,training=False)
    with tf.GradientTape() as tape:
        tape.watch(h)
        pred=h
        for layer in arm.tail:
            pred=layer(pred,training=False)
    return h.numpy(),tape.gradient(pred,h).numpy()


def direct_edits(arm,name,levels,qtest,xtrain,htrain,gradient):
    colname=exp.steel.CONCEPTS[name][2]
    if colname not in arm.names:
        original=np.tile(arm.base,(len(levels),1))
        return dict(node=None),dict(all_dose=original,one_dose=original,
                    all_center=arm.base,one_center=arm.base,all_chord=arm.base,one_chord=arm.base)
    col=arm.names.index(colname)
    raw=np.tile(arm.raw[:1],(len(levels),1))
    # Every DIRECT input is an atomic or size fraction expressed in percent here.
    raw[:,col]=levels/100.
    scaled=exp.p.scaled(arm.model,raw)[:,col]
    href=hidden_grid(arm,xtrain,col,scaled[2:3])[0]
    score=np.mean(np.abs((htrain-href)*gradient),axis=0)
    j=int(np.argmax(score))
    hs=hidden_grid(arm,arm.x,col,scaled)
    isolated=np.broadcast_to(arm.h,hs.shape).copy()
    isolated[:,:,j]=hs[:,:,j]
    full,one=np.split(arm.downstream(np.concatenate([hs,isolated])),2)
    left,right=levels[[0,-1]]
    if right>left:
        weight=np.clip((qtest-left)/(right-left),0,1)[:,None]
        inside=(qtest>left)&(qtest<right)
        combined=np.where(inside[:,None],hs[0]+weight*(hs[-1]-hs[0]),arm.h)
    else:
        inside=np.zeros(len(qtest),dtype=bool)
        combined=arm.h.copy()
    component=arm.h.copy()
    component[:,j]=combined[:,j]
    all_chord,one_chord=arm.downstream(np.stack([combined,component]))
    for edited in (all_chord,one_chord):
        np.testing.assert_allclose(edited[~inside],arm.base[~inside],atol=.002,rtol=1e-6)
    return dict(node=j,train_gradient_score=score[j],inside=int(inside.sum())),dict(
        all_dose=full,one_dose=one,all_center=full[2],one_center=one[2],all_chord=all_chord,one_chord=one_chord)


def run(folds,out):
    data=exp.load_data("steel",out)
    frame,target,splits=data
    q,composition=chemistry(frame)
    audit_hash=file_hash(__file__)
    for f in folds:
        sources,ref,official,ids=exp.load_fold("steel",f,data)
        tr,te=splits[f]
        levels={k:turns.nearest_quantiles(q[k].values[tr],QUANTILES) for k in DIRECT}
        pairs={k:match_real(q[k].values,composition,tr,te,frame.index.values) for k in MATCHED}
        folder=out/f"fold-{f}"
        write_json(folder/"design.json",dict(fold=f,audit_sha256=audit_hash,protocol=DESIGN,levels=levels,pairs=pairs))
        for m in range(50):
            dst=folder/f"member-{m:02d}.json"
            arrpath=dst.with_suffix(".npz")
            hashes=[s[m][2] for s in sources]
            if dst.exists():
                meta=exp.steel.read(dst)
                if meta["audit_sha256"]==audit_hash and meta["model_sha256"]==hashes and meta["arrays_sha256"]==file_hash(arrpath):
                    continue
            train_ids=sources[0][m][3]
            rng=np.random.RandomState(121190+1000*f+m)
            fit_ids=[train_ids[i] for i in sorted(rng.choice(len(train_ids),64,replace=False))]
            arrays=dict(ids=ids,target=target.loc[ids].values,q=q.loc[ids,list(MATCHED)].values)
            meta={}
            for idx,(a,src) in enumerate(zip(exp.ARMS,sources)):
                arm=turns.FrozenArm(a,src[m],{"node":0},frame,ids,ref[idx][m])
                xtrain=exp.p.scaled(arm.model,frame.loc[fit_ids,arm.names].values)
                htrain,grad=train_gradient(arm,xtrain)
                dtype=official[idx][m].dtype
                arrays[a+"_base"]=arm.base
                arrays[a+"_official"]=official[idx][m]
                np.testing.assert_allclose(arm.official(arm.base,"steel",dtype),official[idx][m],atol=.01,rtol=1e-6)
                items={}
                for k in DIRECT:
                    items[k],pred=direct_edits(arm,k,levels[k],q.loc[ids,k].values,xtrain,htrain,grad)
                    for key,v in pred.items():
                        arrays[f"{a}_{k}_{key}"]=v
                    for key in EDITS:
                        arrays[f"{a}_{k}_{key}_official"]=arm.official(pred[key],"steel",dtype)
                meta[a]=items
                tf.keras.backend.clear_session()
            np.savez_compressed(arrpath,**arrays)
            write_json(dst,dict(fold=f,member=m,audit_sha256=audit_hash,model_sha256=hashes,
                       arrays_sha256=file_hash(arrpath),**meta))
            if (m+1)%25==0:
                print("direct",f,m+1,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds",nargs="+",type=int,choices=range(5),default=list(range(5)))
    parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args()
    run(args.folds,args.output)
