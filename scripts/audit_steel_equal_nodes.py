"""Test equally many native first-layer nodes on held-out steels using exact forward interventions."""
import argparse
import pickle
from pathlib import Path

import audit_steel_concept_directions as concept
import discover_physical_kan_responses as exp
from pymatgen.core import Element
from scipy.spatial import cKDTree
from run_modnet_official_matched import file_hash, np, pd, tf, write_json


OUT = exp.ROOT/"benchmarks/steel-equal-nodes-20261112"
PREVIOUS = exp.ROOT/"benchmarks/steel-physical-strength-20261112"
GROUPS = ("one","three","five","tenth","physical3","half_a","half_b")
RANDOM = 12


def chemistry_pairs(frame,ids,q):
    raw = frame.loc[ids].filter(regex=r"^ElementFraction\|").copy()
    raw = raw.loc[:,raw.ne(0).any()]
    raw.columns = [c.split("|")[1] for c in raw.columns]
    mass = raw * [float(Element(c).atomic_mass) for c in raw.columns]
    wt = 100*mass.div(mass.sum(axis=1),axis=0)
    ni = wt.Ni.values
    other = wt.drop(columns=["Fe","Ni"]).values
    matches = []
    for l,low in enumerate(q):
        for h in np.flatnonzero((q>=low+.75)&(q<=low+1.25)&(ni>=ni[l]+.75)&(ni<=ni[l]+1.25)):
            distance = np.abs(other[l]-other[h]).sum()
            if distance<=1:
                matches.append((distance,abs(q[h]-low-1),l,h))
    used,selected = set(),[]
    for distance,_,l,h in sorted(matches):
        if l in used or h in used:
            continue
        used.update([l,h])
        selected.append(dict(low=ids[l],high=ids[h],low_index=l,high_index=h,q_gap=q[h]-q[l],
                             ni_gap=ni[h]-ni[l],other_l1_wt_points=distance))
    return selected


def load_recipes(folder,frame):
    variants = {r:[] for r in ("original","nickel","carbon")}
    hashes = {}
    for part in ("mm","roost"):
        path = folder/f"features-{part}.npz"
        meta = exp.steel.read(path.with_suffix(".json"))
        hashes[part] = file_hash(path)
        assert meta["arrays_sha256"]==hashes[part]
        with np.load(path) as d:
            np.testing.assert_array_equal(d["ids"],frame.index)
            for key in variants:
                variants[key].append(pd.DataFrame(d[key],index=d["ids"],columns=d["columns"]))
            np.testing.assert_allclose(d["q"][:,0],exp.steel.concept_frame(frame)["ni_equivalent"].values,atol=1e-8)
            np.testing.assert_allclose(d["q"][:,1:]-d["q"][:,:1],1,atol=1e-8)
    return {r:pd.concat(parts,axis=1) for r,parts in variants.items()},hashes


def training_attributions(tail,lo,hi,full):
    # Integrated gradients only rank candidate nodes on training rows.
    # Every reported subset effect is a finite before/after run of the original network.
    norm = max(np.linalg.norm(full),1e-8)
    for order in (8,16,32):
        points,weights = np.polynomial.legendre.leggauss(order)
        terms = integrate_nodes(tail,lo,hi,(points+1)/2,weights/2)
        error = np.linalg.norm(full-terms.sum(axis=1))/norm
        if error<.005:
            return terms,dict(quadrature=order,relative_sum_error=error)
    # Some large chemistry edits cross thin spline intervals. Refine the affected
    # training rows on a composite grid, keeping the identical accuracy criterion.
    points,weights = np.polynomial.legendre.leggauss(8)
    for cells in (8,16,32,64,128,256,512,1024):
        residual = abs(full-terms.sum(axis=1))
        active = residual > .0025*norm/np.sqrt(len(full))
        alpha = ((np.arange(cells)[:,None]+(points+1)/2)/cells).ravel()
        weights_grid = np.tile(weights/(2*cells),cells)
        terms[active] = integrate_nodes(tail,lo[active],hi[active],alpha,weights_grid)
        error = np.linalg.norm(full-terms.sum(axis=1))/norm
        if error<.005:
            return terms,dict(quadrature=8*cells,relative_sum_error=error)
    raise ArithmeticError(f"Training quadrature relative error {error} exceeded 0.005")


def integrate_nodes(tail,lo,hi,alpha,weights):
    average = np.zeros_like(lo,dtype=float)
    displacement = hi-lo
    width = lo.shape[1]
    step = max(1,384//len(lo))
    for start in range(0,len(alpha),step):
        grid = lo[None,:,:]+alpha[start:start+step,None,None]*displacement[None,:,:]
        x = tf.convert_to_tensor(grid.reshape(-1,width),dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            output = x
            for layer in tail:
                output = layer(output,training=False)
        grad = tape.gradient(output,x).numpy().reshape(grid.shape)
        average += np.einsum("t,tnj->nj",weights[start:start+step],grad)
    return displacement*average


def rank_nodes(terms,full,count):
    residual = full.copy()
    square = np.sum(terms*terms,axis=0)
    nodes = []
    for _ in range(count):
        gain = 2*(terms.T@residual)-square
        gain[nodes] = -np.inf
        best = int(np.argmax(gain))
        nodes.append(best)
        residual -= terms[:,best]
    return nodes


def training_neighborhood(h):
    center = h.mean(0)
    _,s,v = np.linalg.svd(h-center,full_matrices=False)
    n = min(12,np.searchsorted(np.cumsum(s*s),.95*np.sum(s*s))+1,np.sum(s>s[0]*1e-6))
    transform = v[:n].T * (np.sqrt(len(h))/s[:n])
    points = (h-center)@transform
    tree = cKDTree(points)
    cutoff = np.quantile(tree.query(points,k=2)[0][:,1],.95)
    def inside(z):
        return tree.query((z-center)@transform)[0]<=cutoff+1e-10
    return inside


def swap_groups(tail,lo,hi,groups):
    delta = hi-lo
    masks = np.zeros((len(groups),delta.shape[1]))
    for i,nodes in enumerate(groups):
        masks[i,nodes] = 1
    moved = masks[:,None,:]*delta
    add,remove = np.split(concept.propagate(tail,np.concatenate((lo+moved,hi-moved))),2)
    lower,upper = concept.propagate(tail,np.stack([lo,hi]))
    only,necessary = add-lower,upper-remove
    return (only+necessary)/2,only,necessary,upper-lower


def sparse_calibration(h,q,nodes):
    x = h[:,nodes]
    center = x.mean(0)
    c = x-center
    ridge = .01*np.sum((h-h.mean(0))**2)/h.shape[1]
    w = np.linalg.solve(c.T@c+ridge*np.eye(len(nodes)),c.T@(q-q.mean()))
    return w,q.mean()-center@w


def audit(arm,source,reference,official_dtype,frame,q,ids,kind,recipes,seed,real_pairs,previous):
    path,member_seed,digest,train_ids = source
    assert file_hash(path)==digest
    with path.open("rb") as stream:
        model = pickle.load(stream)
    model._restore_model()
    columns = list(model.optimal_descriptors[:model.n_feat])
    all_ids = train_ids+ids
    x = exp.p.scaled(model,frame.loc[all_ids,columns].values)
    rep,pred,_,layers,first,_ = exp.p.forward(model,x,arm=="kan")
    assert first==0
    raw = rep["first"].astype(float)
    n = len(train_ids)
    h,z = raw[:n],raw[n:]
    width = raw.shape[1]
    tail = layers[1:]
    np.testing.assert_allclose(pred[n:],reference,atol=.012,rtol=1e-6)
    qt = q.loc[all_ids].values
    probe = exp.steel.fit_readouts(h,z,qt[:n,None],qt[n:,None])
    w = probe["coefficients"][:,0]
    real = {}
    meta = dict(width=width,nonlinear_tail=sum(hasattr(l,"coefficients") or
                (hasattr(l,"activation") and l.activation.__name__!="linear") for l in tail),model_sha256=digest)
    if kind=="latent":
        _,_,axes = concept.learn_directions(h,w[:,None])
        d = axes[0][1]["marginal"][0]
        np.testing.assert_allclose(w@d,1,atol=1e-7)
        endpoints = {"latent":(raw-.5*d,raw+.5*d)}
    else:
        edited = {}
        for route,features in recipes.items():
            scaled = exp.p.scaled(model,features.loc[all_ids,columns].values)
            edited[route] = layers[0](scaled,training=False).numpy().astype(float)
        base_check = concept.propagate(tail,edited["original"])
        meta["unmodified_feature_replay_max_mpa"] = float(np.max(abs(base_check-pred)))
        meta["unmodified_feature_replay_mean_abs_mpa"] = float(np.mean(abs(base_check-pred)))
        endpoints = {r:(edited["original"],edited[r]) for r in ("nickel","carbon")}
    primary = next(iter(endpoints))
    train_low,train_high = (v[:n] for v in endpoints[primary])
    lower,upper = concept.propagate(tail,np.stack((train_low,train_high)))
    terms,meta["ranking_check"] = training_attributions(tail,train_low,train_high,upper-lower)
    tenth = int(np.ceil(width*.1))
    ranking = rank_nodes(terms,upper-lower,max(5,tenth))
    physical = exp.steel.three_node_probe(h,z,qt[:n,None],.01)[0][0].tolist()
    rng = np.random.RandomState(seed)
    halves = np.array_split(rng.permutation(n),2)
    groups = [ranking[:1],ranking[:3],ranking[:5],ranking[:tenth],physical]
    groups += [rank_nodes(terms[ix],(upper-lower)[ix],3) for ix in halves]
    groups += [rng.choice(width,3,replace=False).tolist() for _ in range(RANDOM)]
    meta["groups"] = groups
    meta["half_jaccard"] = len(set(groups[5])&set(groups[6]))/len(set(groups[5])|set(groups[6]))
    fits = [sparse_calibration(h,qt[:n],g) for g in groups]
    real["estimate_q"] = np.stack([z[:,g]@s+b for g,(s,b) in zip(groups,fits)])
    real["full_estimate_q"] = probe["decoded"][:,0]
    near = training_neighborhood(h)
    for route,(before,after) in endpoints.items():
        lo,hi = before[n:],after[n:]
        estimates,suff,nec,full = swap_groups(tail,lo,hi,groups)
        if route=="latent":
            np.testing.assert_allclose(full,previous,atol=.012,rtol=1e-6)
        base,upper = concept.propagate(tail,np.stack([lo,hi]))
        fixed = exp.steel.native_postprocess(np.column_stack([base,upper]),model,member_seed,official_dtype)
        # Verify both extremes of the swap, including effects of nonlinear interactions.
        np.testing.assert_allclose(concept.propagate(tail,lo+(hi-lo))-base,full,atol=.006,rtol=1e-6)
        change = hi-lo
        g = groups[1]
        add,drop = lo.copy(),hi.copy()
        add[:,g],drop[:,g] = hi[:,g],lo[:,g]
        info = dict(full=full,official=fixed[:,1].astype(float)-fixed[:,0],estimate=estimates,
                    alone=suff,removal=nec,decoded=change@w,
                    subset_decoded=np.stack([change[:,g]@s for g,(s,b) in zip(groups,fits)]),
                    joint_support=near(lo)&near(hi)&near(add)&near(drop))
        real.update({route+"_"+k:v for k,v in info.items()})
    if kind=="physical" and real_pairs:
        lo = z[[p["low_index"] for p in real_pairs]]
        hi = z[[p["high_index"] for p in real_pairs]]
        est,_,_,tot = swap_groups(tail,lo,hi,groups[:5])
        real.update(pair_full=tot,pair_sparse=est)
    return real,meta


def run(folder,folds,kind,members):
    data = exp.load_data("steel",folder)
    frame,y,_ = data
    q = exp.steel.concept_frame(frame)["ni_equivalent"]
    recipes,hashes = load_recipes(folder,frame) if kind=="physical" else (None,{})
    code = file_hash(__file__)
    for f in folds:
        sources,raw,official,ids = exp.load_fold("steel",f,data)
        pairs = chemistry_pairs(frame,ids,q.loc[ids].values)
        base = folder/f"fold-{f}"/kind
        base.mkdir(parents=True,exist_ok=True)
        older = PREVIOUS/f"fold-{f}/nickel-first-layer.npz"
        hashes["previous"] = file_hash(older)
        with np.load(older) as old:
            np.testing.assert_array_equal(old["ids"],ids)
            for m in members:
                path = base/f"member-{m:02d}.json"
                archive = path.with_suffix(".npz")
                models = [src[m][2] for src in sources]
                if path.exists():
                    prior = exp.steel.read(path)
                    if (prior["code_sha256"]==code and prior["inputs"]==hashes and prior["models"]==models
                            and prior["arrays_sha256"]==file_hash(archive)):
                        continue
                arrays = dict(ids=ids,q=q.loc[ids].values,y=y.loc[ids].values)
                metadata = {}
                for arm,src,ref,official_arm in zip(exp.ARMS,sources,raw,official):
                    result,metadata[arm] = audit(arm,src[m],ref[m],official_arm.dtype,frame,q,ids,kind,recipes,
                                       12112026+f*1000+m,pairs,old[arm+"_delta"][m])
                    arrays.update({arm+"_"+k:v for k,v in result.items()})
                    tf.keras.backend.clear_session()
                np.savez_compressed(archive,**arrays)
                write_json(path,dict(code_sha256=code,inputs=hashes,models=models,group_names=GROUPS,
                           fold=f,member=m,pairs=pairs,arrays_sha256=file_hash(archive),**metadata))
                if m==0 or (m+1)%10==0:
                    print(kind,f,m+1,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=("physical","latent"))
    parser.add_argument("--folds",type=int,nargs="+",choices=range(5),default=list(range(5)))
    parser.add_argument("--members",type=int,nargs="+",choices=range(50),default=list(range(50)))
    parser.add_argument("--output",type=Path,default=OUT)
    args = parser.parse_args()
    run(args.output,args.folds,args.kind,args.members)
