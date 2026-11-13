"""Summarize train-detected spline extrema, their held-out support and output-level checks."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from report_physical_kan_responses import ROOT, ARMS, as_json, file_with_arrays, read, sha


OUT = ROOT / "benchmarks/kan-native-turns-20261112"
PREVIOUS = ROOT / "benchmarks/physical-edge-discovery-20261112"


def arm_statistics(edges, array_sets, y, arm):
    counts = np.sum([e["neutral"][arm] for e in edges],axis=0)
    eligible = [e for e in edges if "arms" in e]
    original,edit,official,scored = (
        np.mean([arr[arm+"_"+key] for arr in array_sets],axis=0).astype(float)
        for key in ("base","selected_chord","official_base","selected_chord_official"))
    target_mae,edited_mae = abs(official-y).mean(),abs(scored-y).mean()
    return dict(neutral_counts=counts, full_turn_fraction=counts[2]/counts[0],
                full_turn_localized_fraction=counts[3]/counts[2] if counts[2] else None,
                chord_edge_mae_median=np.median([e["arms"][arm]["mean_abs_edit"] for e in eligible]) if eligible else None,
                original_target_mae=target_mae, edited_target_mae=edited_mae, delta_mae=edited_mae-target_mae,
                ensemble_mean_abs_prediction_edit=abs(original-edit).mean())


def recurring(edges):
    by_name = Counter(e["name"] for e in edges)
    result = {}
    for name,total in by_name.items():
        cases = [e for e in edges if e["name"]==name and "arms" in e]
        if not cases:
            continue
        points = [e["turn"]["root"] for e in cases]
        groups = [[e["turn"]["root"] for e in cases if e["fold"]==f] for f in range(5)]
        result[name] = dict(selected=total, turn_count=len(cases), outer_supported=sum(e["outer_observed_turn"] for e in cases),
            root_quartiles=np.quantile(points,[.25,.5,.75]),root_range=[min(points),max(points)],
            fold_counts=[len(g) for g in groups],fold_median=[np.median(g) if g else None for g in groups],
            hidden_peaks=sum(e["turn"]["prominence"]>0 for e in cases),
            majority_full_peak={a:sum(e["full_directions"][a][0]>.5 for e in cases) for a in ARMS},
            majority_full_valley={a:sum(e["full_directions"][a][1]>.5 for e in cases) for a in ARMS})
    return result


def local_polynomial(recipe,i,root,domain,base,display_scale):
    """Translate the actual polynomial span containing a root, without regression."""
    scale=recipe["input_multiplier"][i]/display_scale
    offset=recipe["input_intercept"][i]
    bounded=recipe["bounded"]
    def coordinate(x):
        z=scale*np.asarray(x)+offset
        return .5*np.tanh(z) if bounded else z
    xstar=root*display_scale
    zstar=coordinate(xstar)
    spline=BSpline(recipe["spline_knots"],recipe["coefficients"][i],3)
    knots=np.unique(recipe["spline_knots"])
    k=np.searchsorted(knots,zstar,side="right")-1
    ends=knots[k:k+2]
    if bounded:
        with np.errstate(divide="ignore"):
            ends=np.arctanh(2*ends)
    ends=(ends-offset)/scale
    ends=np.array([max(ends[0],domain[0]*display_scale),min(ends[1],domain[1]*display_scale)])
    multiplier=1 if bounded else scale
    coeffs=np.array([spline(zstar)-base,spline(zstar,nu=1)*multiplier,
                     spline(zstar,nu=2)*multiplier**2/2,spline(zstar,nu=3)*multiplier**3/6])
    x=np.linspace(*ends,251)
    u=coordinate(x)-zstar if bounded else x-xstar
    exact=spline(np.clip(coordinate(x),*recipe["clipping"]))-base
    error=np.max(abs(np.polynomial.polynomial.polyval(u,coeffs)-exact))
    np.testing.assert_allclose(np.polynomial.polynomial.polyval(u,coeffs),exact,atol=2e-8)
    return dict(display_multiplier=display_scale,display_interval=ends,display_root=xstar,
                variable="0.5*tanh(scale*x+offset)-zstar" if bounded else "x-xstar",
                scale=scale,offset=offset,zstar=zstar,coefficients_ascending=coeffs,max_check_error=error)


def example(task,e,folder):
    f,m,i = e["fold"],e["member"],e["input_index"]
    tag = f"{task}/fold-{f}/member-{m:02d}"
    parent = read(PREVIOUS/(tag+".json"))
    recipe = parent["kan"]["source_recipe"]
    result = dict(fold=f,member=m,name=e["name"],index=i,kan_node=parent["kan"]["node"],mlp_node=parent["mlp"]["node"],
                  shape=e["shape"],turn=e["turn"],outer_observed_turn=e.get("outer_observed_turn"))
    if "arms" not in e:
        return result
    levels = np.array(e["turn"]["levels"])
    result.update(coefficient=e["one_coefficient"],arms=e["arms"],outer_triplet=e["outer_raw_triplet"],
                  full_directions=e["full_directions"])
    grid = np.linspace(*e["shape"]["central_range"],401)
    def bases(x):
        values = np.asarray(x)*recipe["input_multiplier"][i]+recipe["input_intercept"][i]
        coord = .5*np.tanh(values) if recipe["bounded"] else values
        return BSpline(recipe["spline_knots"],np.eye(8),3)(np.clip(coord,*recipe["clipping"]))
    coef = np.array(recipe["coefficients"][i])
    original = bases(grid) @ coef
    v = bases(levels) @ coef
    np.testing.assert_allclose(v,e["turn"]["values"],atol=1e-8)
    edited = original + bases(grid)[:,e["one_coefficient"]["index"]]*e["one_coefficient"]["delta"]
    result["grid"] = grid
    result["spline"] = original-v[0]
    result["changed_spline"] = edited-v[0]
    result["spline_at_levels"] = v-v[0]
    result["spline_at_root"] = float(bases(e["turn"]["root"]) @ coef - v[0])
    result["local_polynomial"]=local_polynomial(recipe,i,e["turn"]["root"],e["shape"]["central_range"],v[0],
                                                100 if e["name"].startswith("ElementFraction|") else 1)
    with np.load(folder/(tag+".npz")) as arr:
        result["n_test"] = len(arr["ids"])
        for arm in ARMS:
            for field in ("node","path","whole"):
                triplet = arr[f"e{i}_{arm}_turn_{field}"].astype(float)
                result[arm+"_"+field+"_quantiles"] = np.quantile(triplet-triplet[0], [.1,.5,.9],axis=1)
                if arm=="mlp" and field=="node":
                    result["mlp_node_example_backgrounds"] = (triplet-triplet[0]).T
    return result


def summarize(task, folder, code_hash):
    folds,all_edges,all_ids = [],[],set()
    for f in range(5):
        edges,blocks,group_ids,y = [],[],None,None
        present = 0
        for m in range(50):
            tag=f"{task}/fold-{f}/member-{m:02d}.json"
            record,archive=file_with_arrays(folder/tag,"audit_sha256",code_hash)
            assert record["source_sha256"] == sha(PREVIOUS/tag)
            assert (record["task"],record["fold"],record["member"]) == (task,f,m)
            assert len(record["edges"]) == 3
            qualified = [e["input_index"] for e in record["edges"] if "arms" in e]
            assert record["chosen"] == (qualified[0] if qualified else None)
            present += bool(qualified)
            with np.load(archive) as a:
                ids = a["ids"]
                if group_ids is None:
                    assert all_ids.isdisjoint(map(str,ids))
                    all_ids.update(map(str,ids))
                    group_ids,y = ids,a["target"]
                np.testing.assert_array_equal(group_ids,ids)
                np.testing.assert_array_equal(y,a["target"])
                blocks.append({arm+"_"+key:a[arm+"_"+key] for arm in ARMS
                               for key in ("base","selected_chord","official_base","selected_chord_official")})
                for e in record["edges"]:
                    assert (e["turn"] is not None and e["turn"]["relative"]>=.15) == ("arms" in e)
                    assert e["neutral"]["mlp"][1]==e["neutral"]["mlp"][3]==0
                    if "arms" in e:
                        e["full_directions"]={}
                        assert abs(e["one_coefficient"]["secant_after"]-.5*e["one_coefficient"]["secant_before"])<1e-5
                        for arm in ARMS:
                            arr=a[f"e{e['input_index']}_{arm}_chord"]
                            np.testing.assert_allclose(abs(arr-a[arm+"_base"]).mean(),e["arms"][arm]["mean_abs_edit"])
                            path=a[f"e{e['input_index']}_{arm}_turn_whole"]
                            assert path.shape==(3,len(y)) and np.isfinite(path).all()
                            left,middle,right=path
                            e["full_directions"][arm]=[np.mean(middle-np.maximum(left,right)>1),
                                                       np.mean(np.minimum(left,right)-middle>1)]
                    edges.append(dict(e,fold=f,member=m))
        good = [e for e in edges if "arms" in e]
        fold=dict(fold=f,materials=len(y),selected=len(edges),qualifying=len(good),members_qualifying=present,
            supported=sum(e["outer_observed_turn"] for e in good),
            sensitivity_cutoffs={str(c):sum(e["turn"] is not None and e["turn"]["relative"]>=c for e in edges)
                                 for c in (.1,.15,.25)},
            one_coefficient_median_prediction_edit=np.median([e["one_coefficient"]["mean_abs_edit"] for e in good]) if good else None)
        for arm in ARMS:
            fold[arm]=arm_statistics(edges,blocks,y,arm)
        folds.append(fold)
        all_edges.extend(edges)
    good=[e for e in all_edges if "arms" in e]
    features=recurring(all_edges)
    if task=="steel":
        pool=[n for n in features if n.startswith("ElementFraction|")] or list(features)
    else:
        key="ElementProperty|MagpieData minimum CovalentRadius"
        pool=[key] if key in features else list(features)
    chosen = sorted(pool,key=lambda n:(-features[n]["turn_count"],n))[0] if pool else None
    example_edge=next((e for e in good if e["name"]==chosen),None)
    overall=dict(qualifying=len(good),supported=sum(e["outer_observed_turn"] for e in good),selected=len(all_edges),
                 members_qualifying=sum(f["members_qualifying"] for f in folds),
                 sensitivity_cutoffs={str(c):sum(f["sensitivity_cutoffs"][str(c)] for f in folds) for c in (.1,.15,.25)},
                 one_coefficient_median_prediction_edit=np.mean([f["one_coefficient_median_prediction_edit"] for f in folds
                                                                 if f["one_coefficient_median_prediction_edit"] is not None]) if good else None,
                 graph_edit_max=max((e["one_coefficient"]["graph_check_max"] for e in good),default=0),
                 coefficient_outside_total=sum(e["one_coefficient"]["outside_count"] for e in good),
                 coefficient_outside_max=max((e["one_coefficient"]["outside_max"] for e in good
                                              if e["one_coefficient"]["outside_max"] is not None),default=None))
    for arm in ARMS:
        counts=np.sum([f[arm]["neutral_counts"] for f in folds],axis=0)
        overall[arm]=dict(neutral_counts=counts,full_turn_fraction=counts[2]/counts[0],
            full_turn_localized_fraction=counts[3]/counts[2] if counts[2] else None)
        for key in ("chord_edge_mae_median","original_target_mae","edited_target_mae","delta_mae","ensemble_mean_abs_prediction_edit"):
            values=[f[arm][key] for f in folds if f[arm][key] is not None]
            overall[arm][key]=np.mean(values) if values else None
    result = dict(materials=len(all_ids),folds=folds,overall=overall,features=features,
                  example=example(task,example_edge,folder) if example_edge else None)
    control=read(folder/task/"coefficient-order-control.json")
    assert control["audit_sha256"]==code_hash and control["code_sha256"]==sha(ROOT/"scripts/control_kan_turning_points.py")
    assert control["name"]==chosen
    if chosen:
        info=features[chosen]
        assert control["selected"]==info["selected"] and control["observed"]["count"]==info["turn_count"]
        np.testing.assert_allclose(control["observed"]["median"],info["root_quartiles"][1])
    result["coefficient_order_control"]={k:control[k] for k in ("name","selected","repeats","observed","random_quantiles")}
    result["coefficient_order_control"]["file_sha256"]=sha(folder/task/"coefficient-order-control.json")
    if task=="steel":
        mo = next(e for e in all_edges if e["member"]==e["fold"]==0 and e["name"]=="ElementFraction|Mo")
        result["previous_mo"]=example(task,mo,folder)
    return result


def plot(report, file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","Noto Sans CJK SC","DejaVu Sans"],
        "axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#8b9aa3",
        "axes.labelcolor":"#273b47","text.color":"#192f39","xtick.color":"#445a67","ytick.color":"#445a67",
        "font.size":10,"legend.frameon":False})
    fig,axes=plt.subplots(2,3,figsize=(13.8,7.7))
    fig.subplots_adjust(left=.08,right=.98,bottom=.12,top=.82,wspace=.29,hspace=.54)
    green,orange,purple="#006e61","#b65623","#71509d"
    for row,(task,title,unit) in enumerate((("steel","钢","MPa"),("phonons","声子",r"cm$^{-1}$"))):
        ex=report[task]["example"]
        if ex is None:
            for ax in axes[row]:
                ax.set_visible(False)
            continue
        n=ex["name"]
        pct=n.startswith("ElementFraction|")
        label=n.split("|")[-1]+" 原子含量（at.%）" if pct else (
            "最小共价半径（pm）" if n.endswith("minimum CovalentRadius") else n.split("|")[-1])
        scale=100 if pct else 1
        levels=np.asarray(ex["turn"]["levels"])*scale
        grid=np.asarray(ex["grid"])*scale
        ax,mid,end=axes[row]
        ax.plot(grid,ex["spline"],lw=2.6,c=green)
        ax.plot(grid,ex["changed_spline"],lw=1.9,c=purple,ls="--")
        ax.scatter(levels,ex["spline_at_levels"],s=23,c=green,zorder=4)
        root=ex["turn"]["root"]*scale
        ax.axvline(root,color="#526d7b",alpha=.68,ls=":",lw=1.1)
        ax.scatter([root],[ex["spline_at_root"]],s=49,edgecolors=green,facecolors="white",zorder=5)
        ax.set_ylabel("相对原边左端的激活变化")
        ax.set_title(f"{title} · KAN 原生边    h[{ex['kan_node']}]",loc="left",fontsize=11.5,fontweight="bold")
        backgrounds=np.array(ex["mlp_node_example_backgrounds"])
        for v in backgrounds[np.linspace(0,len(backgrounds)-1,min(24,len(backgrounds)),dtype=int)]:
            mid.plot(levels,v,c=orange,lw=.75,alpha=.20)
        mid.plot(levels,np.array(ex["mlp_node_quantiles"])[1],marker="o",ms=3,c=orange,lw=2.3)
        mid.set_ylabel("相对各自左端的激活变化")
        mid.set_title(f"配对 MLP 首层节点    h[{ex['mlp_node']}]",loc="left",fontsize=11.5,fontweight="bold")
        for arm,col in (("kan",green),("mlp",orange)):
            q=np.array(ex[arm+"_whole_quantiles"])
            end.fill_between(levels,q[0],q[2],color=col,alpha=.13,lw=0)
            end.plot(levels,q[1],marker="o",ms=3,lw=2,c=col)
        end.set_title("完整模型的留出预测",loc="left",fontsize=11.5,fontweight="bold")
        end.set_ylabel(f"相对各自左端的变化（{unit}）")
        for a in axes[row]:
            a.set_xlabel(label)
            a.grid(axis="y",alpha=.27)
            a.set_axisbelow(True)
    fig.text(.08,.95,"一个物理输入的转折能定位到模型的哪一部分？",fontsize=19,fontweight="bold")
    handles=[Line2D([],[],c=green,lw=2.6,label="KAN 原模型"),Line2D([],[],c=purple,ls="--",lw=2,label="只改一个样条系数"),
             Line2D([],[],c=orange,lw=2.2,label="MLP 原模型")]
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.073,.923),ncol=3)
    fig.text(.08,.047,"左列为连续样条，圆圈标出导数换号位置。中列为不同留出背景的三点响应，右列为完整预测的中位数和第 10–90 百分位。",fontsize=9,color="#4c606c")
    fig.text(.08,.022,"后两列的连线仅表示三个输入位置的先后。每个模型保留该材料的其余输入，各自的节点使用各自的激活单位。",fontsize=9,color="#4c606c")
    fig.savefig(file,dpi=155,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,default=OUT)
    out=cli.parse_args().output
    code=sha(ROOT/"scripts/audit_kan_turning_points.py")
    report=dict(audit_sha256=code,report_sha256=sha(Path(__file__)),
                neutral_count_columns=["valid background triples","first node turn","whole output turn >1 target unit",
                                       "whole turn plus native node turn and same-direction isolated output turn >1"],
                statistics="Threshold counts pooled over edges. Intervention magnitudes and MAE use equal-weight fold means; each fold uses its 50 paired members. All saved folds available.")
    for task in ("steel","phonons"):
        report[task]=summarize(task,out,code)
    figure=out/"one-edge-turns.png"
    plot(report,figure)
    report["figure_sha256"]=sha(figure)
    (out/"summary.json").write_text(json.dumps(report,default=as_json,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
