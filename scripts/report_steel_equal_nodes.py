"""Summarize matched node interventions, measured chemistry controls and the two fixed steels."""
import json
from pathlib import Path

import numpy as np

from report_physical_kan_responses import ARMS, ROOT, as_json, file_with_arrays, read, sha


OUT = ROOT/"benchmarks/steel-equal-nodes-20261112"
IDS = ("mb-steels-005","mb-steels-020")
ROUTES = {"latent":("latent",),"physical":("nickel","carbon")}
GROUPS = ("one","three","five","tenth","physical3","half_a","half_b")


def relative(actual,estimate):
    norm = np.linalg.norm(actual)
    return np.linalg.norm(estimate-actual)/norm if norm>1e-9 else None


def score(truth,pred):
    return 1-np.mean((pred-truth)**2)/np.var(truth)


def pct(x):
    return np.quantile(x,[.1,.5,.9])


def semantics(truth,estimate):
    return dict(ensemble_r2=score(truth,estimate.mean(0)), ensemble_rmse=np.mean((truth-estimate.mean(0))**2)**.5,
                median_member_r2=np.median([score(truth,row) for row in estimate]))


def describe_group(full,estimate,alone,removal,inside,clean):
    ef,ee = full.mean(0),estimate.mean(0)
    per_member = np.array([relative(f,p) for f,p in zip(full,estimate)])
    significant = abs(ef)>=1
    return dict(ensemble_rel=relative(ef,ee), ensemble_mae=np.mean(abs(ee-ef)),
                members_rel=relative(full,estimate), member_rel_10_50_90=pct(per_member),
                ensemble_sign_match=np.mean(np.sign(ef[significant])==np.sign(ee[significant])) if any(significant) else None,
                alone_rel=relative(full,alone), removal_rel=relative(full,removal),
                mean_interaction_abs_mpa=np.mean(abs(alone-removal)),
                supported_rel=relative(full[inside],estimate[inside]),
                no_repair_rel=relative(full[clean],estimate[clean]))


def load_results(fold,kind,hashes):
    folder = OUT/f"fold-{fold}"/kind
    records,arrays = [],[]
    inputs = {"previous":sha(ROOT/f"benchmarks/steel-physical-strength-20261112/fold-{fold}/nickel-first-layer.npz")}
    if kind=="physical":
        inputs.update({p:sha(OUT/f"features-{p}.npz") for p in ("mm","roost")})
    for m in range(50):
        meta,archive = file_with_arrays(folder/f"member-{m:02d}.json","code_sha256",hashes["audit"])
        assert meta["inputs"]==inputs and meta["group_names"]==list(GROUPS) and (meta["fold"],meta["member"])==(fold,m)
        with np.load(archive) as z:
            arrays.append(dict(z))
        if m:
            for k in ("ids","y","q"):
                np.testing.assert_array_equal(arrays[-1][k],arrays[0][k])
            assert meta["pairs"]==records[0]["pairs"]
        records.append(meta)
    return records,arrays


def inspect(fold,kind,hashes):
    records,data = load_results(fold,kind,hashes)
    def get(key):
        return np.stack([d[key] for d in data])
    q,ids,y = (data[0][k] for k in ("q","ids","y"))
    info = dict(ids=ids,truth=q,y=y,routes={},examples={},readouts={},models=[m["models"] for m in records])
    for arm in ARMS:
        decoded = get(arm+"_estimate_q")
        item = {g:semantics(q,decoded[:,i]) for i,g in enumerate(GROUPS)}
        item["random3_mean_r2"] = np.mean([score(q,z.mean(0)) for z in decoded[:,len(GROUPS):].transpose(1,0,2)])
        item["full"] = semantics(q,get(arm+"_full_estimate_q"))
        item["width"] = [m[arm]["width"] for m in records]
        item["nonlinear_tail"] = [m[arm]["nonlinear_tail"] for m in records]
        item["half_node_jaccard"] = np.mean([m[arm]["half_jaccard"] for m in records])
        item["rank_quadrature_max_rel"] = max(m[arm]["ranking_check"]["relative_sum_error"] for m in records)
        assert item["rank_quadrature_max_rel"] < .005
        if kind=="physical":
            item["natural_replay_max_mpa"] = max(m[arm]["unmodified_feature_replay_max_mpa"] for m in records)
            item["natural_replay_mean_abs_mpa"] = np.mean([m[arm]["unmodified_feature_replay_mean_abs_mpa"] for m in records])
        info["readouts"][arm] = item
    for route in ROUTES[kind]:
        def values(arm,name):
            return get(arm+"_"+route+"_"+name)
        inside = np.logical_and.reduce([values(a,"joint_support") for a in ARMS])
        clean = np.logical_and.reduce([abs(values(a,"full")-values(a,"official"))<.02 for a in ARMS])
        sign_rows = np.logical_and.reduce([abs(values(a,"full").mean(0))>=1 for a in ARMS])
        layer = np.array([all(m[a]["nonlinear_tail"]>0 for a in ARMS) for m in records])
        result = dict(both_nonlinear=int(layer.sum()),near_fraction=inside.mean(),unrepaired_fraction=clean.mean(),
                      both_nontrivial_materials=int(sign_rows.sum()))
        examples = {}
        member_error = []
        for arm in ARMS:
            full,est,alone,rem = (values(arm,k) for k in ("full","estimate","alone","removal"))
            dq,ds = values(arm,"decoded"),values(arm,"subset_decoded")
            item = dict(ensemble_full=full.mean(0), ensemble_official=values(arm,"official").mean(0),
                        ensemble_three=est[:,1].mean(0), fully_decoded_step_mean=dq.mean(),
                        fully_decoded_step_rmse=np.mean((dq.mean(0)-1)**2)**.5)
            for i,name in enumerate(GROUPS):
                v = describe_group(full,est[:,i],alone[:,i],rem[:,i],inside,clean)
                v["decoded_step_mean"] = ds[:,i].mean()
                v["decoded_step_ensemble_rmse"] = np.mean((ds[:,i].mean(0)-1)**2)**.5
                if name=="three":
                    v["both_nonlinear_rel"] = relative(full[layer],est[layer,i])
                    fsign,esign = full.mean(0)[sign_rows],est[:,i].mean(0)[sign_rows]
                    v["common_material_sign_match"] = np.mean(np.sign(fsign)==np.sign(esign)) if len(fsign) else None
                item[name] = v
            random = est[:,len(GROUPS):].transpose(1,0,2)
            item["random3"] = dict(ensemble_rel=np.mean([relative(full.mean(0),draw.mean(0)) for draw in random]),
                                   members_rel=np.mean([relative(full,draw) for draw in random]))
            member_error.append([relative(a,b) for a,b in zip(full,est[:,1])])
            result[arm] = item
            if fold==0:
                for name in IDS:
                    s = ids.tolist().index(name)
                    details = dict(full=full[0,s],estimate=est[0,:len(GROUPS),s],alone=alone[0,1,s],removal=rem[0,1,s],
                                   nodes=records[0][arm]["groups"][1],support_first=bool(inside[0,s]),
                                   ensemble_full=full[:,s].mean(),ensemble_estimate=est[:,:len(GROUPS),s].mean(0),
                                   ensemble_official=values(arm,"official")[:,s].mean(),
                                   ensemble_q=get(arm+"_estimate_q")[:,1,s].mean(),ensemble_dq=ds[:,1,s].mean(),
                                   ensemble_interaction=(alone[:,1,s]-rem[:,1,s]).mean(),
                                   member_full_10_50_90=pct(full[:,s]),member_positive=int((full[:,s]>0).sum()))
                    examples.setdefault(name,dict(q=q[s],yield_strength=y[s]))[arm] = details
        result["kan_wins_member_relative"] = int(np.sum(np.array(member_error[0])<member_error[1]))
        info["routes"][route] = result
        info["examples"][route] = examples
    if kind=="physical":
        pairs = []
        for i,p in enumerate(records[0]["pairs"]):
            l,h = p["low_index"],p["high_index"]
            pairs.append(dict(**p,observed=y[h]-y[l],**{a:dict(full=get(a+"_pair_full")[:,i].mean(),
                         three=get(a+"_pair_sparse")[:,1,i].mean()) for a in ARMS}))
        info["observed_pairs"] = pairs
    return info


def bootstrap(folds,route):
    # Reuse material indices for the two arms within each fold; do not count model seeds as independent data.
    rng = np.random.default_rng(6211126)
    gaps = []
    for f in folds:
        r = f["routes"][route]
        n = len(f["ids"])
        index = rng.integers(0,n,size=(4000,n))
        by_arm = []
        for a in ARMS:
            full,est = r[a]["ensemble_full"],r[a]["ensemble_three"]
            by_arm.append(np.sqrt(((full-est)[index]**2).sum(1)/(full[index]**2).sum(1)))
        gaps.append(by_arm[0]-by_arm[1])
    return np.quantile(np.mean(gaps,axis=0),[.025,.975])


def named_trace(fold,summary,hashes):
    folder = OUT/f"fold-{fold}"
    meta,archive = file_with_arrays(folder/"named-trace.json","code_sha256",hashes["named"])
    assert meta["parents"] == [sha(folder/f"physical/member-{i:02d}.json") for i in range(50)]
    with np.load(archive) as f:
        data = dict(f)
    ids = data["ids"].tolist()
    assert ids == list(summary["ids"])
    result = {}
    for key in ("delta","valence","named"):
        group = {}
        for a in ARMS:
            actual = data[f"{a}_{key}_contextual"]
            portable = data[f"{a}_{key}_portable"]
            assert actual.shape == portable.shape == (50,len(ids))
            present = np.array([bool(m["groups"][key]) for m in meta[a]])
            signal = actual.mean(0)
            error = (portable-actual).mean(0)
            item = dict(present=int(present.sum()), ensemble_mean_abs_signal=np.mean(abs(signal)),
                        members_mean_abs_signal=np.mean(abs(actual[present])) if any(present) else None,
                        ensemble_mean_abs_transfer_error=np.mean(abs(error)), ensemble_rel=relative(signal,portable.mean(0)),
                        max_abs_error=np.max(abs(portable-actual)),
                        member_rel=relative(actual[present],portable[present]))
            if fold==0:
                item["examples"] = {k:dict(m0_true=actual[0,ids.index(k)],m0_portable=portable[0,ids.index(k)],
                                         ensemble_true=signal[ids.index(k)],ensemble_portable=portable[:,ids.index(k)].mean())
                                    for k in IDS}
            group[a] = item
        result[key] = group
    if fold==0:
        result["recipe_descriptors"] = {k:{c:data[c][ids.index(k)] for c in ("delta_percentage_points","valence_electrons")} for k in IDS}
    return result


def plot(folds,path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import font_manager
    for font in ("Microsoft YaHei","Noto Sans CJK SC","SimHei"):
        if font in {f.name for f in font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [font,"DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    colors = ("#126b68","#bc582c")
    fig,axes = plt.subplots(1,3,figsize=(12.6,3.8),layout="constrained",gridspec_kw={"width_ratios":[1.1,1,1]})
    cases = folds[0]["examples"]["nickel"]
    x = np.arange(len(IDS))
    ax = axes[0]
    for j,(arm,color) in enumerate(zip(ARMS,colors)):
        offset = -.13 if j==0 else .13
        marker = "o" if arm=="kan" else "s"
        full = [cases[i][arm]["ensemble_full"] for i in IDS]
        sparse = [cases[i][arm]["ensemble_estimate"][1] for i in IDS]
        ax.vlines(x+offset,full,sparse,color=color,lw=2)
        ax.scatter(x+offset,full,s=66,c=color,marker=marker,label=arm.upper()+" 全部节点",zorder=3)
        ax.scatter(x+offset,sparse,s=63,facecolors="white",edgecolors=color,marker=marker,linewidths=2,label=arm.upper()+" 三节点",zorder=4)
    ax.set(xticks=x,xticklabels=IDS,title="以 Ni 替 Fe，两种钢的预测变化",ylabel="屈服强度差（MPa）",xlim=(-.5,1.5))
    ax.axhline(0,color="#80858b",lw=.85)
    handles,labels = ax.get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,loc="outside upper center",ncols=4,fontsize=9)
    for ax,route,title in zip(axes[1:],("nickel","carbon"),("以 Ni 替 Fe","保留同三节点，改用 C 替 Fe")):
        for offset,arm,color in zip((-.07,.07),ARMS,colors):
            r = [100*f["routes"][route][arm]["three"]["ensemble_rel"] for f in folds]
            ax.plot(np.arange(5)+offset,r,"o" if arm=="kan" else "s",color=color,markerfacecolor="white",
                    markeredgewidth=1.8,ms=7,label=arm.upper())
        ax.set(xticks=range(5),xlabel="留出折",title=title,ylabel="三节点相对还原误差（%，越低越好）",
               xlim=(-.4,4.4),ylim=(0,110))
    for ax in axes:
        ax.grid(axis="y",alpha=.18)
        ax.spines[["right","top"]].set_visible(False)
    fig.savefig(path,dpi=185,bbox_inches="tight")
    plt.close(fig)


def main():
    hashes = {k:sha(ROOT/"scripts"/file) for k,file in dict(audit="audit_steel_equal_nodes.py",
               features="prepare_steel_nickel_paths.py",named="trace_steel_equal_node_inputs.py",report=Path(__file__).name).items()}
    for part in ("mm","roost"):
        feature = read(OUT/f"features-{part}.json")
        assert feature["code_sha256"]==hashes["features"] and feature["arrays_sha256"]==sha(OUT/f"features-{part}.npz")
        assert max(feature["original_max_by_column"].values())<2e-6
    result = dict(code_sha256=hashes,folds={},paired_material_bootstrap={})
    for kind,routes in ROUTES.items():
        folds = [inspect(f,kind,hashes) for f in range(5)]
        assert len({i for f in folds for i in f["ids"]})==sum(len(f["ids"]) for f in folds)==312
        result["folds"][kind] = folds
        result["paired_material_bootstrap"].update({route:bootstrap(folds,route) for route in routes})
    for a,b in zip(*result["folds"].values()):
        assert a["models"]==b["models"]
    for i,fold in enumerate(result["folds"]["physical"]):
        folder = ROOT/f"benchmarks/steel396-physical-concepts-20261112/fold-{i}"
        kan,mlp = [read(folder/(arm+".json"))["members"] for arm in ARMS]
        for m,(k,v) in enumerate(zip(kan,mlp)):
            assert k["input_features"]==v["input_features"]
            assert fold["models"][m]==[k["sha256"],v["sha256"]]
    result["named"] = [named_trace(i,f,hashes) for i,f in enumerate(result["folds"]["physical"])]
    (OUT/"summary.json").write_text(json.dumps(result,default=as_json,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    plot(result["folds"]["physical"],OUT/"equal-nodes.png")


if __name__=="__main__":
    main()
