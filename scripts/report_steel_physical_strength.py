"""Summarize fold-level steel tests with matched material IDs, model pairs and physical units."""
import argparse
import json
from pathlib import Path

import numpy as np

from report_physical_kan_responses import ARMS, ROOT, as_json, file_with_arrays, read, sha


OUT = ROOT / "benchmarks/steel-physical-strength-20261112"
DIRECT = ("size_mismatch", "chromium", "molybdenum", "titanium", "carbon", "nickel")
MODES = ("marginal", "conditional")
NI = "ni_equivalent"
SCRIPTS = {"direct": "audit_steel_physical_strength", "latent": "audit_steel_concept_directions",
           "nickel": "trace_steel_nickel_equivalent", "real": "check_steel_nickel_equivalent",
           "first": "check_steel_nickel_first_layer", "turns": "audit_kan_turning_points"}


def stack(records, key):
    return np.stack([record[key] for record in records]).astype(float)


def describe(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {}
    return dict(mean=values.mean(), median=np.median(values), mean_abs=np.abs(values).mean(),
                positive=int(np.sum(values > 0)), n=int(values.size), p10_p90=np.quantile(values, [.1, .9]))


def r2(truth, pred):
    return 1 - np.mean((truth - pred)**2) / np.var(truth)


def load_fold(folder, hashes):
    design = read(folder/"design.json")
    fold = design["fold"]
    assert design["audit_sha256"] == hashes["direct"]
    metas = {key: [] for key in ("direct", "latent", "nickel")}
    arrays = {key: [] for key in metas}
    for m in range(50):
        filename = f"member-{m:02d}.json"
        for key in metas:
            path = (folder if key == "direct" else folder/key) / filename
            hash_key = "audit_sha256" if key == "direct" else "code_sha256"
            doc, archive = file_with_arrays(path, hash_key, hashes[key])
            assert (doc["fold"], doc["member"]) == (fold, m)
            with np.load(archive) as src:
                values = {name: src[name] for name in src.files}
            if m or key != "direct":
                np.testing.assert_array_equal(values["ids"], arrays["direct"][0]["ids"])
            if key == "latent":
                assert doc["models"] == metas["direct"][m]["model_sha256"]
                assert doc["direct_code_sha256"] == hashes["direct"]
                np.testing.assert_array_equal(values["target"], arrays["direct"][0]["target"])
                for arm in ARMS:
                    np.testing.assert_allclose(values[arm+"_base"], arrays["direct"][m][arm+"_base"],
                                               atol=.02, rtol=1e-6)
                    np.testing.assert_array_equal(values[arm+"_official"], arrays["direct"][m][arm+"_official"])
            if key == "nickel":
                assert doc["parent_sha256"] == sha(folder/"latent"/filename)
            metas[key].append(doc)
            arrays[key].append(values)
    return design, metas, arrays


def real_pairs(pairs, ids, target, preds):
    lookup = {key: i for i, key in enumerate(ids)}
    assert len({p[side] for p in pairs for side in ("low", "high")}) == 2*len(pairs)
    diffs = []
    for pair in pairs:
        l, h = lookup[pair["low"]], lookup[pair["high"]]
        diffs.append(dict(**pair, gap=pair["q_high"]-pair["q_low"], observed=target[h]-target[l],
                          **{arm: preds[arm][h]-preds[arm][l] for arm in ARMS}))
    return dict(n=len(diffs), difference={k: describe([p[k] for p in diffs])
                                          for k in ("gap", "observed", *ARMS)}, pairs=diffs)


def direct_results(records, meta, name, target, levels):
    result = dict(levels=levels)
    for arm in ARMS:
        prefix = f"{arm}_{name}_"
        orig = stack(records, arm+"_base").mean(0)
        official = stack(records, arm+"_official").mean(0)
        dose = stack(records, prefix+"all_dose").mean(0)
        info = dict(present=sum(m[arm][name]["node"] is not None for m in meta),
                    quartile_contrast=describe(dose[3]-dose[1]), mean_dose_predictions=dose.mean(1))
        for k in ("all_center", "one_center", "all_chord", "one_chord"):
            edited = stack(records, prefix+k).mean(0)
            fixed = stack(records, prefix+k+"_official").mean(0)
            info[k] = dict(shift=describe(edited-orig),
                           official_delta_mae=np.mean(abs(fixed-target))-np.mean(abs(official-target)))
        result[arm] = info
    return result


def decoded_results(records, j, target):
    result = dict()
    truth = records[0]["q"][:,j]
    for arm in ARMS:
        pred = stack(records, arm+"_decoded").mean(0)[:,j]
        null = stack(records, arm+"_null").mean(0)[:,j]
        result[arm] = dict(r2=r2(truth, pred), rmse=np.mean((truth-pred)**2)**.5, shuffled_r2=r2(truth, null))
    for mode in MODES:
        suffix = f"_{j}_{mode}_"
        common = [r for r in records if all(arm+suffix+"local" in r for arm in ARMS)]
        stats = dict(common_members=len(common))
        for arm in ARMS if common else ():
            local = stack(common, arm+suffix+"local")
            centered = stack(common, arm+suffix+"center")-stack(common, arm+"_base")
            official = stack(common, arm+"_official").mean(0)
            edited = stack(common, arm+suffix+"official").mean(0)
            info = dict(local_ensemble=describe(local.mean(0)), local_member_means=describe(local.mean(1)),
                        center_ensemble=describe(centered.mean(0)), center_member_mean_abs=np.abs(centered).mean(),
                        official_center_delta_mae=np.mean(abs(edited-target))-np.mean(abs(official-target)))
            if mode == "conditional":
                control_key = f"{arm}_{j}_matched_random"
                control_rows = [r for r in common if control_key in r]
                if control_rows:
                    control = stack(control_rows,control_key)
                    comparison = stack(control_rows,arm+suffix+"center")-stack(control_rows,arm+"_base")
                    info["matched_random_center"] = dict(n=len(control_rows), member_mean_abs=np.abs(control).mean(),
                                                          concept_member_mean_abs=np.abs(comparison).mean())
            stats[arm] = info
        result[mode] = stats
    return result


def one_point(records, meta):
    result = {}
    for mode in MODES:
        index = [i for i,r in enumerate(records) if all(f"{a}_{mode}_total" in r for a in ARMS)]
        common = [records[i] for i in index]
        inside = np.logical_and.reduce([stack(common, f"{a}_{mode}_inside").astype(bool) for a in ARMS])
        stats = dict(n=len(index), members=index, common_supported_fraction=inside.mean())
        for arm in ARMS:
            prefix = f"{arm}_{mode}_"
            full = stack(common, prefix+"total")
            info = dict(ensemble=describe(full.mean(0)), member_means=describe(full.mean(1)),
                        official_ensemble=describe(stack(common,prefix+"official").mean(0)),
                        supported_member_material=describe(full[inside]),
                        marginal_same_members=describe(stack(common, arm+"_marginal_total").mean(0)),
                        within_member_material_sd_median=np.median(full.std(1)),
                        native_sum_max_error=max(meta[i][arm][mode]["term_replay_max"] for i in index))
            for budget in ("one", "three"):
                contribution = stack(common, prefix+budget)
                info[budget] = dict(mean=contribution.mean(),
                                    member_relative_rmse=np.sqrt(np.mean((full-contribution)**2)/np.mean(full**2)),
                                    ensemble_relative_rmse=np.linalg.norm((full-contribution).mean(0)) /
                                                           np.linalg.norm(full.mean(0)))
            stats[arm] = info
        result[mode] = stats
    return result


def native_turns(fold, audit_hash):
    names = {"size_mismatch":"YangSolidSolution|Yang delta", "titanium":"ElementFraction|Ti",
             "molybdenum":"ElementFraction|Mo"}
    rows = {k:[] for k in names}
    folder = ROOT / f"benchmarks/kan-native-turns-20261112/steel/fold-{fold}"
    for m in range(50):
        item = read(folder/f"member-{m:02d}.json")
        assert item["audit_sha256"] == audit_hash and (item["fold"],item["member"]) == (fold,m)
        for name,col in names.items():
            for edge in item["edges"]:
                if edge["name"] == col and "arms" in edge:
                    rows[name].append(dict(member=m, root_percent=100*edge["turn"]["root"],
                                           **{a:edge["arms"][a]["mean_abs_edit"] for a in ARMS}))
    return {k: dict(n=len(v), median={key: np.median([row[key] for row in v])
                                       for key in ("root_percent",*ARMS)} if v else {}) for k,v in rows.items()}


def inspect_fold(folder, real, hashes):
    design, metas, arrays = load_fold(folder, hashes)
    direct, latent, nickel = (arrays[k] for k in ("direct","latent","nickel"))
    ids = direct[0]["ids"].tolist()
    target = direct[0]["target"]
    assert ids == real["ids"]
    np.testing.assert_array_equal(target, real["target"])
    pred = {arm: stack(direct,arm+"_official").mean(0) for arm in ARMS}
    idx = metas["latent"][0]["names"].index(NI)
    np.testing.assert_allclose(latent[0]["q"][:,idx], real["ni_eq"], atol=1e-10)
    observed = {k: real_pairs(v["pairs"],ids,target,pred) for k,v in design["pairs"].items()}
    fixed = [{k:v for k,v in pair.items() if k != "observed"} for pair in real["fixed_ni_pairs"]]
    check = real_pairs(fixed,ids,target,pred)
    metrics = {name:{key:v[key] for key in ("coefficients","mae")} for name,v in real["ols"].items()}
    first_meta = read(folder/"nickel-first-layer.json")
    first_npz = folder/"nickel-first-layer.npz"
    assert first_meta["hashes"] == {"code":hashes["first"], "latent":hashes["latent"]}
    assert first_meta["arrays_sha256"] == sha(first_npz)
    assert first_meta["sources"] == [list(items) for items in zip(*(m["model_sha256"] for m in metas["direct"]))]
    first_results = {}
    with np.load(first_npz) as early:
        np.testing.assert_array_equal(early["ids"], ids)
        np.testing.assert_allclose(early["ni_equivalent"], real["ni_eq"], atol=1e-10)
        for arm in ARMS:
            delta = early[arm+"_delta"]
            first_results[arm] = dict(r2=r2(early["ni_equivalent"],early[arm+"_decoded"].mean(0)),
                                      ensemble=describe(delta.mean(0)), member_means=describe(delta.mean(1)),
                                      within_member_material_sd_median=np.median(delta.std(1)))
    # Predetermined member and first two ordered IDs inside both training regions.
    example = {}
    both = nickel[0]["kan_marginal_inside"] & nickel[0]["mlp_marginal_inside"]
    selected_ids = [i for i in np.argsort(ids) if both[i]][:2]
    for arm in ARMS:
        key = arm+"_marginal_"
        example[arm] = dict(nodes=metas["nickel"][0][arm]["marginal"]["nodes"],
                            sample_range=np.quantile(nickel[0][key+"total"],[.1,.9]),
                            materials=[dict(id=ids[i], all_terms=nickel[0][key+"total"][i],
                                            three_terms=nickel[0][key+"individual"][i],
                                            in_training_region=nickel[0][key+"inside"][i]) for i in selected_ids])
    return dict(fold=design["fold"], ids=ids, counts=len(target),
                official_mae={a:np.mean(abs(p-target)) for a,p in pred.items()},
                matched=observed, fixed_elemental_ni=check, ols=metrics,
                residual_corr_given_ni=real["partial_correlation_given_ni"],
                physical_scale={name:iqr for name,iqr in zip(metas["latent"][0]["names"],metas["latent"][0]["iqr"])},
                direct={k:direct_results(direct,metas["direct"],k,target,design["levels"][k]) for k in DIRECT},
                decoded={k:decoded_results(latent,j,target) for j,k in enumerate(metas["latent"][0]["names"])},
                nickel_one_point=one_point(nickel,metas["nickel"]), first_layer=first_results,
                first_member_example=example,
                same_single_bend=native_turns(design["fold"],hashes["turns"]))


def pair_summary(folds):
    rng = np.random.default_rng(20261112)
    summary = {}
    for name in folds[0]["matched"]:
        groups = [f["matched"][name]["pairs"] for f in folds]
        values = [[p["observed"] for p in g] for g in groups]
        record = dict(pooled_observed=describe([v for group in values for v in group]),
                      pooled_gap=describe([p["gap"] for group in groups for p in group]),
                      pairs_per_fold=[len(g) for g in groups])
        if all(len(g) for g in groups):
            boot = [rng.choice(g,(4000,len(g)),replace=True).mean(1) for g in values]
            record["equal_fold_observed"] = np.mean([np.mean(g) for g in values])
            record["fixed_match_bootstrap_p025_p975"] = np.quantile(np.mean(boot,axis=0),[.025,.975])
        summary[name] = record
    return summary


def extract_curves(folder):
    import pickle
    import discover_physical_kan_responses as exp
    from trace_steel_nickel_equivalent import native_terms

    data = exp.load_data("steel",folder)
    sources,_,_,ids = exp.load_fold("steel",0,data)
    root = folder/"fold-0"
    probe = read(root/"latent/member-00.json")
    trace = read(root/"nickel/member-00.json")
    payload = dict(trace_sha256=sha(root/"nickel/member-00.json"))
    with np.load(root/"nickel/member-00.npz") as verified:
        covered = verified["kan_marginal_inside"] & verified["mlp_marginal_inside"]
        examples = [i for i in np.argsort(ids) if covered[i]][:2]
        for arm,group in zip(ARMS,sources):
            path,_,source_hash,train_ids = group[0]
            assert probe[arm]["source_sha256"] == sha(path) == source_hash
            with path.open("rb") as stream:
                model = pickle.load(stream)
            model._restore_model()
            columns = list(model.optimal_descriptors[:model.n_feat])
            x = exp.p.scaled(model,data[0].loc[train_ids+ids,columns].values)
            rep,_,_,layers,_,_ = exp.p.forward(model,x,arm=="kan")
            h,z = np.split(rep["late"].astype(float),[len(train_ids)])
            node = trace[arm]["marginal"]["nodes"][0]
            axis = probe[arm]["axes"][probe["names"].index(NI)]["modes"]["marginal"]
            step = axis["direction"][node]
            native,_ = native_terms(arm,model,layers,h)
            center = np.median(h,axis=0)
            origin = native(center[None,:])[0,node]
            def ordinate(points):
                batch = np.tile(center,(len(points),1))
                batch[:,node] = points
                return native(batch)[:,node]-origin
            intervals = z[examples,node,None]+np.array([-.5,.5])*step
            lower,upper = np.quantile(h[:,node],[.02,.98])
            grid = np.linspace(min(lower,intervals.min()),max(upper,intervals.max()),241)
            marks = []
            for i,xx in zip(examples,intervals):
                yy = ordinate(xx)
                np.testing.assert_allclose(yy[1]-yy[0],verified[arm+"_marginal_one"][i],atol=.002)
                marks.append(dict(id=ids[i],x=xx,y=yy,delta=yy[1]-yy[0],total=verified[arm+"_marginal_total"][i]))
            payload[arm] = dict(node=node,x=grid,y=ordinate(grid),materials=marks)
    (folder/"nickel-curve-example.json").write_text(json.dumps(payload,default=as_json,indent=2)+"\n",encoding="utf-8")


def draw_curves(case, path):
    from matplotlib import pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(8.7,3.8),layout="constrained")
    colors = ("#097865","#ac4e26")
    for axis,arm in zip(axes,ARMS):
        curve = case[arm]
        axis.plot(curve["x"],curve["y"],color="#343c48",lw=1.7)
        for color,material in zip(colors,curve["materials"]):
            xx,yy = material["x"],material["y"]
            label = f"{material['id']}   {material['delta']:+.2f} MPa"
            axis.plot(xx,yy,color=color,lw=3.2,label=label,zorder=3)
            axis.scatter(xx[:1],yy[:1],s=40,color="white",edgecolor=color,zorder=4)
            axis.scatter(xx[1:],yy[1:],s=40,color=color,zorder=4)
        axis.set(title=f"{arm.upper()} 输出层 · 节点 {curve['node']}",xlabel="该节点的原始激活值",
                 ylabel="该连接的强度贡献（相对训练中位位置，MPa）")
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(axis="y",alpha=.17)
        axis.legend(loc="best",frameon=True,facecolor="white",edgecolor="none",framealpha=.9)
    fig.savefig(path,dpi=190,bbox_inches="tight")
    plt.close(fig)


def draw(folds, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for choice in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        if choice in available:
            plt.rcParams["font.sans-serif"] = [choice, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1,3,figsize=(12.2,4.0),layout="constrained")
    styles = {"observed":("实测","#262b35","D"), "kan":("KAN","#126d6b","o"), "mlp":("MLP","#ba562c","s")}
    x = np.arange(5)
    titles = ("真实钢样的高、低镍当量配对", "模型内镍当量相差 1 点", "同样相差 1 点，固定其余三项")
    for col, (axis, title) in enumerate(zip(axes, titles)):
        if col == 0:
            counts = [f["matched"][NI]["n"] for f in folds]
            data = {key:[f["matched"][NI]["difference"][key]["mean"] for f in folds] for key in styles}
            xlabel = "留出折（钢样对数）"
        else:
            mode = MODES[col-1]
            counts = [f["nickel_one_point"][mode]["n"] for f in folds]
            data = {a:[f["nickel_one_point"][mode][a]["official_ensemble"]["mean"] for f in folds] for a in ARMS}
            xlabel = "留出折（配对模型数）"
        for i,(key,values) in enumerate(data.items()):
            label,color,marker = styles[key]
            axis.plot(x+(i-(len(data)-1)/2)*.14,values,color=color,marker=marker,linestyle="None",
                      markersize=7.5,label=label,markeredgecolor="white",markeredgewidth=.7,zorder=3)
        axis.set(title=title, xlabel=xlabel, xticks=x, xticklabels=[f"{i} ({n})" for i,n in enumerate(counts)])
        axis.set_ylabel("屈服强度差（MPa）" if col == 0 else "预测屈服强度差（MPa）")
        axis.set_xlim(-.5,4.5)
        if col:
            axis.set_ylim(-25,90)
        axis.axhline(0,color="#79828a",linewidth=.9)
        axis.grid(axis="y",alpha=.18)
        axis.spines[["top","right"]].set_visible(False)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,loc="outside upper center",ncols=3)
    fig.savefig(path,dpi=190,bbox_inches="tight")
    plt.close(fig)


def main(folder):
    hashes = {k:sha(ROOT/"scripts"/(v+".py")) for k,v in SCRIPTS.items()}
    chem = read(folder/"nickel-composition-check.json")
    assert chem["code_sha256"] == hashes["real"]
    folds = [inspect_fold(folder/f"fold-{i}",f,hashes) for i,f in enumerate(chem["folds"])]
    all_ids = [i for f in folds for i in f["ids"]]
    assert len(folds) == 5 and len(set(all_ids)) == len(all_ids) == 312
    summary = dict(hashes=dict(**hashes,report=sha(Path(__file__))), folds=folds, matched=pair_summary(folds))
    curve = read(folder/"nickel-curve-example.json")
    assert curve["trace_sha256"] == sha(folder/"fold-0/nickel/member-00.json")
    for arm in ARMS:
        expected = folds[0]["first_member_example"][arm]
        assert curve[arm]["node"] == expected["nodes"][0]
        for m,z in zip(curve[arm]["materials"],expected["materials"]):
            assert m["id"] == z["id"]
            np.testing.assert_allclose(m["delta"],z["three_terms"][0],atol=.002)
    (folder/"summary.json").write_text(json.dumps(summary,default=as_json,ensure_ascii=False,
                                       allow_nan=False,indent=2)+"\n",encoding="utf-8")
    draw(folds,folder/"nickel-equivalent.png")
    draw_curves(curve,folder/"nickel-native-curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--extract-curves",action="store_true",help="Use the trained-model environment for this step.")
    args = parser.parse_args()
    if args.extract_curves:
        extract_curves(args.output)
    else:
        main(args.output)
