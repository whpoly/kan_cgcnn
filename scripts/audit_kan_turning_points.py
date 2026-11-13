"""Find physically labelled extrema in frozen KAN edges and test matched MLP paths."""
import argparse
import pickle
from pathlib import Path

import discover_physical_kan_responses as exp
from scipy.interpolate import BSpline, PPoly
from run_modnet_official_matched import file_hash, np, tf, write_json


OUT = exp.ROOT / "benchmarks/kan-native-turns-20261112"
PROMINENCE = .15
GRID = [.1, .3, .5, .7, .9]
DESIGN = dict(
    population="Every previously train-selected shared physical input (3 per pair), both tasks, all five folds and 50 pairs per fold.",
    discovery="Derivative roots of original cubic spline in the interior of training q10..q90. Adjacent monotone limbs must each have >=max(8,ceil(5% of training rows)) observations and two distinct raw values. >=3 training rows within 10% of bracket width of the root. Use nearest actual training value to root, and actually observed endpoints on both sides. Weaker limb must span >=15% of full central edge range. Also record 10% and 25% cutoffs.",
    holds="Data support independently assessed with actual outer values in the frozen bracket. Need >=3 on both sides, a middle observation within 10% of its width, and reversal >=10% of the training central edge range.",
    neutral="Architecture-independent training quantiles 10/30/50/70/90 (nearest actual values). Compare full models, and whether a reversal of the same full-output direction is visible in the chosen node's isolated path. Require native node reversal >1e-5 and final reversal >1 target unit on each side.",
    edit="Replace one edge response inside the frozen bracket by the chord of the two endpoint responses; both models, MLP endpoints retain the material-specific context. Aggregate one qualifying edge per member (first in frozen selection order), identity for members without turns.",
    coefficient="KAN only. Change a single native coefficient to halve the centered three-point departure from its secant. Choose the largest absolute basis departure. Flank values may move. Verify inside/outside effects with the original Keras layer and restore weights in memory.",
    examples="Show the previously fixed Mo edge. Show a steel turn for the most frequently qualifying elemental fraction, and a phonons turn for minimum CovalentRadius if available (otherwise most frequent). Choose the first qualifying edge in fold/member/input order. Choices depend on training quantities only.")


def nearest_quantiles(x, quantiles):
    values = np.sort(np.asarray(x, dtype=float))
    return values[np.rint(np.asarray(quantiles)*(len(values)-1)).astype(int)]


class PhysicalEdge:
    def __init__(self, recipe, i):
        self.scale = recipe["input_multiplier"][i]
        self.offset = recipe["input_intercept"][i]
        self.bounded = recipe["bounded"]
        self.clip = recipe["clipping"]
        self.knots = np.asarray(recipe["spline_knots"])
        self.weights = np.asarray(recipe["coefficients"][i])
        assert self.scale > 0 and len(self.weights) == 8
        self.spline = BSpline(self.knots, self.weights, 3, extrapolate=False)
        self.bases = BSpline(self.knots, np.eye(8), 3, extrapolate=False)

    def coordinate(self, x):
        z = self.scale*np.asarray(x) + self.offset
        return .5*np.tanh(z) if self.bounded else z

    def basis(self, x):
        return self.bases(np.clip(self.coordinate(x), *self.clip))

    def __call__(self, x):
        return self.spline(np.clip(self.coordinate(x), *self.clip))

    def slope(self, x):
        z = self.coordinate(x)
        chain = .5*self.scale*(1-4*z*z) if self.bounded else self.scale
        interior = (z > self.clip[0]) & (z < self.clip[1])
        return self.spline.derivative()(np.clip(z, *self.clip))*chain*interior

    def turns(self, low, high):
        roots = PPoly.from_spline(self.spline).derivative().roots(extrapolate=False)
        zlow,zhigh = np.clip(self.coordinate([low,high]), *self.clip)
        u = np.unique(roots[np.isfinite(roots) & (roots > zlow) & (roots < zhigh)])
        ends = np.r_[zlow, u, zhigh]
        # Exclude repeated/tangent roots without a direction reversal.
        signs = np.sign(self.spline.derivative()((ends[1:]+ends[:-1])/2))
        u = u[signs[:-1]*signs[1:] < 0]
        return ((np.arctanh(2*u) if self.bounded else u)-self.offset)/self.scale


def weaker_limb(values):
    """Positive for a peak, negative for a valley, zero for a monotone triple."""
    a, b, c = values
    return np.maximum(b-np.maximum(a,c),0) - np.maximum(np.minimum(a,c)-b,0)


def sample_triplet(raw, left, root, right, min_side, train=False):
    window = raw[(raw >= left) & (raw <= right)]
    lo, hi = window[window < root], window[window > root]
    if min(len(lo),len(hi)) < min_side:
        return None
    if train and min(len(np.unique(np.round((v-left)/(right-left),7))) for v in (lo,hi)) < 2:
        return None
    close = window[np.abs(window-root) <= .1*(right-left)]
    if len(close) < (3 if train else 1):
        return None
    middle = close[np.argmin(abs(close-root))]
    if lo.min() >= middle or hi.max() <= middle:
        return None
    return np.array([lo.min(), middle, hi.max()])


def detect(edge, raw):
    low, high = nearest_quantiles(raw, [.1,.9])
    assert high>low
    roots = edge.turns(low,high)
    bounds = np.r_[low, roots, high]
    span = float(np.ptp(edge(bounds)))
    grid = np.linspace(low,high,1001)
    slopes = abs(edge.slope(grid))
    peak = int(np.argmax(slopes))
    supported = slopes >= .8*slopes[peak]
    below, above = np.flatnonzero(~supported[:peak]), np.flatnonzero(~supported[peak+1:])
    start = int(below[-1]+1) if len(below) else 0
    stop = int(peak+above[0]) if len(above) else len(grid)-1
    info = dict(central_range=[low,high], edge_range=span, n_derivative_turns=len(roots),
                sensitive_interval=[grid[start],grid[stop]], max_abs_slope=slopes[peak], max_slope_at=grid[peak])
    best = None
    if span > 1e-7:
        for i, root in enumerate(roots):
            levels = sample_triplet(raw,bounds[i],root,bounds[i+2],max(8,int(np.ceil(.05*len(raw)))),True)
            if levels is None:
                continue
            prominence = float(weaker_limb(edge(levels)))
            if best is None or abs(prominence)>abs(best["prominence"]):
                best = dict(root=root, levels=levels, values=edge(levels), prominence=prominence,
                            relative=abs(prominence)/span, train_counts=[int(np.sum((raw>=levels[0])&(raw<root))),
                                                                        int(np.sum((raw<=levels[2])&(raw>root)))])
    return info,best


class FrozenArm:
    def __init__(self, arm, source, branch, frame, ids, reference):
        path, self.seed, sha, _ = source
        assert file_hash(path) == sha
        with path.open("rb") as stream:
            self.model = pickle.load(stream)
        self.model._restore_model()
        self.arm = arm
        self.record = branch
        self.node = branch["node"]
        self.names = list(self.model.optimal_descriptors[:self.model.n_feat])
        self.raw = frame.loc[ids,self.names].values
        self.x = exp.p.scaled(self.model,self.raw)
        layers = [l for l in self.model.model.layers if not isinstance(l,tf.keras.layers.InputLayer)]
        self.first, self.tail = layers[0], layers[1:]
        self.h = self.first(self.x,training=False).numpy()
        self.base = self.downstream(self.h)
        np.testing.assert_allclose(self.base,reference,atol=.01,rtol=1e-6)
        if arm=="kan":
            self.weights = self.first.coefficients.numpy()
        else:
            assert self.first.activation.__name__=="elu"
            self.weights,bias = self.first.get_weights()
            self.pre = self.x @ self.weights + bias

    def downstream(self,h):
        shape = h.shape[:-1]
        flat = h.reshape(-1,h.shape[-1])
        chunks = []
        for start in range(0,len(flat),512):
            tensor = tf.convert_to_tensor(flat[start:start+512],dtype=tf.float32)
            for layer in self.tail:
                tensor = layer(tensor,training=False)
            chunks.append(tensor.numpy().reshape(-1))
        return np.concatenate(chunks).astype(float).reshape(shape)

    def at_levels(self,name,i,levels):
        """Propagate the full descriptor change and the selected node path for each background."""
        col = self.names.index(name)
        rec = self.record["source_recipe"]
        scaled = (np.asarray(levels)*rec["input_multiplier"][i]+rec["input_intercept"][i]).astype("float32")
        if self.arm=="kan":
            b = self.first.hidden_coordinates
            old = exp.basis(self.x[:,col],b) @ self.weights[col].T
            new = exp.basis(scaled,b) @ self.weights[col].T
            full = self.h[None,:,:] + (new[:,None,:]-old[None,:,:])
        else:
            shift = (scaled[:,None] - self.x[:,col])[:,:,None]*self.weights[col]
            full = exp.activation(self.first,self.pre[None,:,:]+shift)
        direct_rows = np.arange(min(3,len(self.x)))
        checks = np.tile(self.x[direct_rows],(len(levels),1))
        checks[:,col] = np.repeat(scaled,len(direct_rows))
        keras = self.first(checks,training=False).numpy().reshape(len(levels),len(direct_rows),-1)
        np.testing.assert_allclose(full[:,direct_rows],keras,atol=.001,rtol=1e-5)
        path = np.broadcast_to(self.h,full.shape).copy()
        path[:,:,self.node] = full[:,:,self.node]
        prediction = self.downstream(np.concatenate([full,path]))
        yf,yp = np.split(prediction,2)
        return full[:,:,self.node].astype(float),yf,yp

    def edit_node(self,delta):
        edited = self.h.copy()
        edited[:,self.node] += delta
        return self.downstream(edited)

    def official(self,pred,task,dtype):
        if task=="steel":
            return exp.steel.native_postprocess(pred[:,None],self.model,self.seed,dtype)[:,0]
        return pred


def neutral_test(arm,name,i,levels):
    node,full,path = arm.at_levels(name,i,levels)
    count = np.zeros(4,dtype=int)
    for k in range(3):
        a = slice(k,k+3)
        if min(np.diff(levels[a])) <= 0:
            continue
        x,y,z = (weaker_limb(v[a]) for v in (node,full,path))
        nturn = abs(x)>1e-5
        whole = abs(y)>1.
        concordant = (abs(z)>1.) & (np.sign(y)==np.sign(z))
        count += [len(y),sum(nturn),sum(whole),sum(nturn & whole & concordant)]
    if arm.arm=="mlp":
        assert count[1] == 0
    return count


def clip_shape(arm,name,i,levels):
    h,full,path = arm.at_levels(name,i,levels)
    x = arm.raw[:,arm.names.index(name)]
    l,_,r = levels
    t = np.clip((x-l)/(r-l),0,1)
    modified = arm.edit_node(np.where((x>l)&(x<r),(1-t)*h[0]+t*h[2]-arm.h[:,arm.node],0))
    np.testing.assert_allclose(modified[(x<=l)|(x>=r)],arm.base[(x<=l)|(x>=r)],atol=.002,rtol=1e-6)
    node_turn, whole_turn, path_turn = (weaker_limb(v) for v in (h,full,path))
    if arm.arm == "mlp":
        assert np.max(abs(node_turn))<1e-5
    metrics = dict(node_median=float(np.median(node_turn)),whole_turn_rate=np.mean(abs(whole_turn)>1.),
            isolated_turn_rate=np.mean(abs(path_turn)>1.), mean_abs_edit=np.mean(abs(modified-arm.base)),
            mean_abs_edit_inside=np.mean(abs(modified[(x>l)&(x<r)]-arm.base[(x>l)&(x<r)])) if np.any((x>l)&(x<r)) else None)
    return metrics,dict(turn_node=h,turn_whole=full,turn_path=path,chord=modified)


def coefficient_edit(arm, edge, name, levels):
    l,c,r = levels
    t = (c-l)/(r-l)
    contrast = np.array([-(1-t),1.,-t])
    gaps = contrast @ edge.basis(levels)
    k = int(np.argmax(abs(gaps)))
    before = float(contrast @ edge(levels))
    delta = -.5*before/gaps[k]
    col,j = arm.names.index(name),arm.node
    weights = arm.weights.copy()
    np.testing.assert_allclose(weights[col,j],edge.weights,rtol=0,atol=1e-8)
    weights[col,j,k] += delta
    real_delta = float(weights[col,j,k])-float(arm.weights[col,j,k])
    np.testing.assert_allclose(before+real_delta*gaps[k],.5*before,rtol=1e-4,atol=1e-7)
    raw = arm.raw[:,col]
    basis = edge.basis(raw)[:,k]
    pred = arm.edit_node(real_delta*basis)
    outside = basis==0
    np.testing.assert_allclose(pred[outside],arm.base[outside],atol=.002,rtol=1e-6)
    # Edit the actual stored layer in memory and check inputs inside and outside support.
    ix = np.unique([0,np.argmin(basis),np.argmax(basis)])
    try:
        arm.first.coefficients.assign(weights)
        original_graph = arm.model.model(arm.x[ix],training=False).numpy().reshape(-1)
        np.testing.assert_allclose(pred[ix],original_graph,atol=.007,rtol=1e-6)
    finally:
        arm.first.coefficients.assign(arm.weights)
    return dict(index=k,delta=real_delta,secant_before=before,secant_after=before+real_delta*gaps[k],
                outside_count=int(sum(outside)),outside_max=float(np.max(abs(pred[outside]-arm.base[outside]))) if np.any(outside) else None,
                mean_abs_edit=np.mean(abs(pred-arm.base)), graph_check_max=np.max(abs(original_graph-pred[ix]))),pred


def inspect_pair(task,fold,member,data,sources,refs,official,ids,out):
    parent = exp.OUT/task/f"fold-{fold}"/f"member-{member:02d}.json"
    meta = exp.steel.read(parent)
    sha = file_hash(parent)
    assert meta["model_sha256"] == [s[member][2] for s in sources]
    assert meta["analysis_sha256"] == file_hash(exp.__file__)
    folder = out/task/f"fold-{fold}"
    folder.mkdir(parents=True,exist_ok=True)
    dest = folder/f"member-{member:02d}.json"
    archive = dest.with_suffix(".npz")
    code_hash = file_hash(__file__)
    if dest.exists():
        saved = exp.steel.read(dest)
        if saved["audit_sha256"] == code_hash and saved["source_sha256"] == sha and saved["arrays_sha256"] == file_hash(archive):
            return
    frame,target,_ = data
    train_ids = sources[0][member][3]
    arms = {a:FrozenArm(a,s[member],meta[a],frame,ids,ref[member]) for a,s,ref in zip(exp.ARMS,sources,refs)}
    arrays = dict(ids=ids,target=target.loc[ids].values)
    for a,ref in zip(exp.ARMS,official):
        arrays[a+"_base"] = arms[a].base
        arrays[a+"_official_base"] = ref[member]
        np.testing.assert_allclose(arms[a].official(arms[a].base,task,ref[member].dtype),ref[member],atol=.01,rtol=1e-6)
    rows = []
    chosen = None
    for i,name in enumerate(meta["inputs"]):
        raw = frame.loc[train_ids,name].values
        xt = frame.loc[ids,name].values
        edge = PhysicalEdge(meta["kan"]["source_recipe"],i)
        info,turn = detect(edge,raw)
        neutral_levels = nearest_quantiles(raw,GRID)
        neutral = {a:arm_count.tolist() for a,arm_count in
                   ((a,neutral_test(arms[a],name,i,neutral_levels)) for a in exp.ARMS)}
        row = dict(name=name,input_index=i,shape=info,turn=turn,neutral=neutral,neutral_levels=neutral_levels)
        if turn is not None and turn["relative"]>=PROMINENCE:
            levels = turn["levels"]
            test_points = sample_triplet(xt,levels[0],turn["root"],levels[2],3)
            held = float(weaker_limb(edge(test_points))) if test_points is not None else 0.
            row["outer_observed_turn"] = bool(np.sign(held)==np.sign(turn["prominence"]) and abs(held)>=.1*info["edge_range"])
            row["outer_raw_triplet"] = test_points
            row["arms"] = {}
            for a,arm in arms.items():
                row["arms"][a],values = clip_shape(arm,name,i,levels)
                arrays.update({f"e{i}_{a}_{key}":v for key,v in values.items()})
            row["one_coefficient"], arrays[f"e{i}_kan_native"] = coefficient_edit(arms["kan"],edge,name,levels)
            if chosen is None:
                chosen=i
        rows.append(row)
    for a,ref in zip(exp.ARMS,official):
        pred = arrays[a+"_base"] if chosen is None else arrays[f"e{chosen}_{a}_chord"]
        arrays[a+"_selected_chord"] = pred
        arrays[a+"_selected_chord_official"] = arms[a].official(pred,task,ref[member].dtype)
    np.savez_compressed(archive,**arrays)
    write_json(dest,dict(task=task,fold=fold,member=member,source_sha256=sha,model_sha256=meta["model_sha256"],
                  audit_sha256=code_hash,arrays_sha256=file_hash(archive),chosen=chosen,edges=rows))
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=("steel","phonons"),required=True)
    parser.add_argument("--folds",nargs="+",type=int,choices=range(5),default=list(range(5)))
    parser.add_argument("--members",type=int,choices=range(1,51),default=50)
    parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args()
    data=exp.load_data(args.task,args.output/args.task/f"data-check-{args.folds[0]}")
    for fold in args.folds:
        sources,refs,official,ids=exp.load_fold(args.task,fold,data)
        for member in range(args.members):
            inspect_pair(args.task,fold,member,data,sources,refs,official,ids,args.output)
            if (member+1)%10==0 or args.members==1:
                print(args.task,fold,member+1,flush=True)
