"""Summarize the observed Mo comparison and check it against saved predictions."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from report_steel_observed_pairs import ARMS, CONCEPTS, INPUT, ROOT, digest, read, relative_error, r2


OUT = ROOT / "benchmarks/steel-observed-molybdenum-20261112"
CASE = ("mb-steels-073", "mb-steels-042")
KEYS = ("total","physical3","strength3","strength10pct","direct_all","direct3",
        "direct_via_physical3","all_inputs_via_direct3")


def per_pair(pair, endpoints):
    lo, hi = (endpoints[pair[k]] for k in ("low","high"))
    return dict(**pair, measured=hi["measured"]-lo["measured"],
                kan=hi["kan"]-lo["kan"],mlp=hi["mlp"]-lo["mlp"])


def paired_members(arr, j, use, measured):
    stats = {}
    for key in KEYS:
        v = [arr[a+"_"+key][use,j].astype(float) for a in ARMS]
        comparison = dict(kan_closer_to_measured=int((abs(v[0]-measured)<abs(v[1]-measured)).sum()))
        for arm,values in zip(ARMS,v):
            comparison[arm] = dict(mean=float(values.mean()),median=float(np.median(values)),
                q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),
                same_direction=int((values*measured>0).sum()),ensemble_abs_error=abs(values.mean()-measured))
        stats[key] = comparison
    return stats


def pooled(rows):
    y = np.array([r["measured"] for r in rows])
    result = dict(n=len(rows),increases=int((y>0).sum()))
    for arm in ARMS:
        pred = np.array([r[arm]["total"] for r in rows])
        result[arm] = {}
        for key in ("total","physical3","strength3","strength10pct"):
            v = np.array([r[arm][key] for r in rows])
            result[arm][key] = dict(same_measured=int((v*y>0).sum()),
                    mae_measured=float(np.mean(abs(v-y))),relative_to_full=relative_error(v,pred))
    return result


def plot(s):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei","Noto Sans CJK SC","DejaVu Sans"]
    fig,(left,right) = plt.subplots(1,2,figsize=(10.7,4.5),gridspec_kw={"width_ratios":[1,1.12]})
    steel = s["series"]
    x = np.arange(len(steel))
    palette = {"measured":"#252f36","kan":"#167775","mlp":"#ae562c"}
    for key,offset,marker,label in (("measured",-.14,"D","实测"),("kan",0,"o","KAN"),("mlp",.14,"s","MLP")):
        v = [r[key] for r in steel]
        left.scatter(x+offset,v,marker=marker,color=palette[key],label=label,s=45,zorder=3)
    left.set(xticks=x,xticklabels=[f"{r['id'][-3:]}\n{r['wt']['Mo']:.2f} wt.% Mo\n第 {r['fold']} 折" for r in steel],
             xlim=(-.42,2.42),ylim=(1160,1610),ylabel="屈服强度（MPa）",title="三件真实钢样的实测与留出预测")
    left.legend(loc="upper left",ncol=3,frameon=False,fontsize=9,columnspacing=1.2)
    y = np.arange(5)
    for arm,offset,marker in (("kan",-.12,"o"),("mlp",.12,"s")):
        values = [100*f["direct_fidelity"][arm] for f in s["folds"]]
        right.scatter(values,y+offset,marker=marker,color=palette[arm],label=arm.upper(),s=44,zorder=3)
    right.set(yticks=y,yticklabels=[f"第 {f['fold']} 折（{f['matched']['n']} 对钢）" for f in s["folds"]],
              xlim=(40,106),ylim=(4.55,-.6),xlabel="相对均方根误差（%，越低越好）",
              title="用三条路径还原全部钼输入路径的贡献")
    right.legend(frameon=False,loc="lower left",ncol=2,fontsize=9)
    for ax in (left,right):
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",color="#dce3e6",lw=.7)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT/"real-mo-steels.png",dpi=175,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    endpoints, natural, all_mo, matched, folds, examples = {}, [], [], [], [], {}
    code = digest(ROOT/"scripts/audit_steel_molybdenum_case.py")
    origin = digest(INPUT)
    with np.load(INPUT) as z:
        labels = dict(zip(z["sample_ids"],z["y"]))
    replay,conserv = 0.,0.
    for f in range(5):
        folder = OUT/f"fold-{f}"
        d = read(folder/"audit.json")
        path = folder/"members.npz"
        assert d["code_sha256"]==code and d["input_sha256"]==origin and d["arrays_sha256"]==digest(path)
        assert d["observed_code_sha256"]==digest(ROOT/"scripts/audit_steel_observed_pairs.py")
        natural.extend(d["natural_same_fold"])
        all_mo.extend(d["natural_mo"])
        with np.load(path) as z:
            arr = {k:z[k] for k in z.files}
        members, ids = d["members"], d["test_ids"]
        use = np.array([m["kan"]["has_mo"] for m in members])
        assert np.array_equal(use,[m["mlp"]["has_mo"] for m in members])
        assert len(members)==50
        for i,name in enumerate(ids):
            assert name not in endpoints and labels[name]==d["measured"][i]
            endpoints[name] = dict(id=name,wt=d["wt"][name],mo_at_pct=d["q"][i],fold=f,measured=labels[name],
                **{a:float(arr[a+"_official_endpoint"][:,i].astype(float).mean()) for a in ARMS})
        qc, fidelity = {},{}
        mask = np.array([p["kind"]=="matched" for p in d["pairs"]])
        for a in ARMS:
            with np.load(CONCEPTS/f"fold-{f}/{a}.npz") as src:
                np.testing.assert_array_equal(ids,src["test_ids"])
                np.testing.assert_array_equal(arr[a+"_raw_endpoint"],src["raw"])
                np.testing.assert_array_equal(arr[a+"_official_endpoint"],src["official"])
            qc[a] = r2(np.asarray(d["q"]),arr[a+"_decoded_mo_at_pct"].mean(0))
            target = arr[a+"_direct_all"][use].mean(0)[mask]
            three = arr[a+"_direct3"][use].mean(0)[mask]
            fidelity[a] = relative_error(three,target)
            replay = max(replay,max(m[a]["replay_error"] for m in members))
            conserv = max(conserv,max(m[a]["conservation_error"] for m in members))
        local = []
        for j, row in enumerate(d["pairs"]):
            if row["kind"]!="matched":
                continue
            rec = dict(**row, fold=f)
            assert labels[row["high"]]-labels[row["low"]]==row["measured"]
            for a in ARMS:
                rec[a] = {k:float(arr[a+"_"+k][:,j].astype(float).mean()) for k in KEYS}
            local.append(rec)
        matched.extend(local)
        folds.append(dict(fold=f,matched=pooled(local),readout_r2=qc,direct_fidelity=fidelity,n_explicit=int(use.sum())))
        if set(CASE)<=set(ids):
            j = next(j for j,p in enumerate(d["pairs"]) if p["kind"]=="case")
            y = d["pairs"][j]["measured"]
            case = dict(pair=d["pairs"][j],fold=f,explicit_members=int(use.sum()),
                        all=paired_members(arr,j,np.ones(50,dtype=bool),y),explicit=paired_members(arr,j,use,y))
            by_model = {}
            for a in ARMS:
                traces = [m[a]["trace"] for m in members if m[a]["has_mo"]]
                diff = np.abs([r["difference_sd"] for r in traces])
                turns = np.array([r["first_leg"] for r in traces])*np.array([r["second_leg"] for r in traces])<0
                by_model[a] = dict(mean_difference_sd=float(diff.mean()), maximum_difference_sd=float(diff.max()),
                    members_with_a_turn=int(turns.any(axis=1).sum()),observed_turns=int(turns.sum()),first=traces[0])
                lo,hi = [ids.index(s) for s in CASE]
                np.testing.assert_array_equal(arr[a+"_raw_endpoint"][:,[lo,hi]],arr[a+"_official_endpoint"][:,[lo,hi]])
                examples[a] = {s:float(arr[a+"_decoded_mo_at_pct"][:,ids.index(s)].mean()) for s in CASE}
            case["edge_check"] = dict(first_member=int(np.flatnonzero(use)[0]), middle=d["middle"],**by_model)
            param = []
            for a in ARMS:
                audit = read(CONCEPTS/f"fold-{f}/{a}.json")["members"]
                param.append(np.array([read((ROOT/m["path"]).with_name("result.json"))["parameters"] for m in audit]))
            case["parameter_ratio_kan_to_mlp"] = [float(np.min(param[0]/param[1])),float(np.max(param[0]/param[1]))]
    assert len(endpoints)==len(labels)==312
    used = [r[k] for r in matched for k in ("low","high")]
    assert len(set(used))==len(used)==82
    natural = [per_pair(r,endpoints) for r in natural]
    all_mo = [per_pair(r,endpoints) for r in all_mo]
    selected = [r for r in natural if abs(r["measured"])>=50 and abs(r["kan"]-r["measured"])<.5*abs(r["mlp"]-r["measured"])]
    assert [(r["low"],r["high"])for r in selected]==[CASE]
    third, = [r["high"] for r in all_mo if r["low"]==CASE[1]]
    case["decoded_mo_at_pct"]=examples
    middle=case["edge_check"]["middle"]
    result = dict(primary=case,series=[endpoints[s] for s in (*CASE,third)],edge_middle=endpoints[middle],
        within_fold_all_elements=natural,all_folds_natural_mo=all_mo,matched=matched,folds=folds,
        pooled=pooled(matched),other_four_folds=pooled([r for r in matched if r["fold"]!=case["fold"]]),
        mean_fold_direct_fidelity={a:float(np.mean([f["direct_fidelity"][a] for f in folds])) for a in ARMS},
        mean_fold_readout_r2={a:float(np.mean([f["readout_r2"][a] for f in folds])) for a in ARMS},
        equal_fold_three_node_error={a:float(np.mean([f["matched"][a]["physical3"]["relative_to_full"] for f in folds])) for a in ARMS},
        replay_max_mpa=replay,conservation_max_mpa=conserv,input_sha256=origin,analysis_sha256=code)
    (OUT/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    plot(result)


if __name__=="__main__":
    main()
