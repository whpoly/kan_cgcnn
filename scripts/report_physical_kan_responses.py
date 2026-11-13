"""Aggregate held-out material responses, verify artifacts and draw fixed-member examples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "benchmarks/physical-edge-discovery-20261112"
ARMS = ("kan", "mlp")
BUDGETS = ("one", "three", "five", "ten", "quarter", "train90", "all")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_with_arrays(path, expected_key, expected_hash):
    row = read(path)
    archive = path.with_suffix(".npz")
    assert row[expected_key] == expected_hash and row["arrays_sha256"] == sha(archive), path
    return row, archive


def r2(y, predictions):
    y = np.asarray(y, dtype=float)
    delta = np.asarray(predictions, dtype=float) - y[:, None]
    return 1 - np.mean(delta**2, axis=0) / max(np.var(y), 1e-20)


def describe_example(names, meta, audit, data, path_data):
    ids = list(map(str, data["ids"]))
    first = ids.index(min(ids))
    ex = dict(id=ids[first], target=data["target"][first], inputs=names, anchor_id=meta["selection"]["anchor_id"])
    for arm in ARMS:
        row = meta[arm]
        ex[arm] = dict(node=row["node"], anchor=row["anchor"], values=data[arm+"_source_values"][first],
                       training_grid=row["grid"], response=row["response"], background_band=row["test_background_p10_p90"],
                       fits=row["curves"], source_recipe=row["source_recipe"], slice_contributions=data[arm+"_slice_components"][first],
                       joint_actual=data[arm+"_component"][first], portable=data[arm+"_portable_component"][first],
                       edge_contrast_levels=audit[arm]["contrast_levels"], edge_contrast_median=audit[arm]["contrast_median"],
                       prediction=data[arm+"_raw_reference"][first], clamped=data[arm+"_remove"][first],
                       output_change=path_data[arm+"_output_change"][first], terminal_nodes=audit[arm]["terminal_nodes"],
                       terminal_changes=path_data[arm+"_terminal_changes"][first], remaining_change=path_data[arm+"_remainder"][first],
                       mean_abs_output_edit=row["metrics"]["mean_abs_output_edit"], whole_node_r2=row["metrics"]["whole_node_r2"])
    return ex


def feature_counts(rows):
    eligible = Counter(n for m, _ in rows for n in m["selection"]["names"])
    selected = Counter(n for m, _ in rows for n in m["inputs"])
    lead = Counter(m["inputs"][0] for m, _ in rows)
    by_name = {}
    for name, total in selected.most_common():
        per_fold = [0]*5
        level_pairs = []
        effects = {arm:[[] for _ in range(5)] for arm in ARMS}
        for m, path in rows:
            if name in m["inputs"]:
                f = m["fold"]
                per_fold[f] += 1
                i = m["inputs"].index(name)
                pairs = np.array(path["kan"]["contrast_levels"])[:, i]
                np.testing.assert_array_equal(pairs, np.array(path["mlp"]["contrast_levels"])[:, i])
                if pairs[1] > pairs[0]:
                    level_pairs.append(pairs)
                    for arm in ARMS:
                        effects[arm][f].append(path[arm]["contrast_median"][i])
        endpoint_stats = {}
        for arm, groups in effects.items():
            values = np.asarray([x for g in groups for x in g])
            endpoint_stats[arm] = dict(fold_median=[np.median(g) if g else None for g in groups],
                positive=int(np.sum(values > .01)), negative=int(np.sum(values < -.01)),
                within_numeric_tolerance=int(np.sum(np.abs(values) <= .01)))
        by_name[name] = dict(selected=total, eligible=eligible[name], lead=lead[name], folds=per_fold,
                             distinct_iqr=len(level_pairs), median_training_endpoints=np.median(level_pairs,axis=0) if level_pairs else None,
                             signed_output_contrast=endpoint_stats)
    groups = Counter(tuple(sorted(m["inputs"])) for m, _ in rows)
    group, freq = groups.most_common(1)[0]
    return dict(features=by_name, distinct_groups=len(groups), most_frequent_group=list(group),
                most_frequent_group_count=freq,
                most_frequent_group_folds=[sum(m["fold"] == f and tuple(sorted(m["inputs"])) == group for m,_ in rows)
                                         for f in range(5)])


def aggregate(task, out, expected):
    result, all_rows, fold_ids = [], [], set()
    example = None
    for f in range(5):
        folder = out/task/f"fold-{f}"
        ref_ids = target = None
        grouped = {arm: [] for arm in ARMS}
        for i in range(50):
            tag = f"member-{i:02d}.json"
            source = folder/tag
            meta, array_file = file_with_arrays(source, "analysis_sha256", expected["discovery"])
            audit, audit_file = file_with_arrays(folder/"paths"/tag, "audit_sha256", expected["audit"])
            assert audit["parent_sha256"] == sha(source)
            assert (meta["task"], meta["fold"], meta["member"]) == (task,f,i)
            assert (audit["task"], audit["fold"], audit["member"]) == (task,f,i)
            assert len(set(meta["inputs"])) == 3
            with np.load(array_file) as data, np.load(audit_file) as paths:
                ids = data["ids"]
                np.testing.assert_array_equal(ids, paths["ids"])
                assert len(paths["donors"]) == 32 and np.all(paths["donors"] != paths["recipients"])
                if i == 0:
                    ref_ids, target = ids, data["target"]
                    assert len(set(map(str, ids))) == len(ids) and fold_ids.isdisjoint(map(str, ids))
                    fold_ids.update(map(str, ids))
                np.testing.assert_array_equal(ids, ref_ids)
                np.testing.assert_array_equal(data["target"], target)
                if i == 0 and f == 0:
                    example = describe_example(meta["inputs"], meta, audit, data, paths)
                for arm in ARMS:
                    original, official = data[arm+"_raw_reference"], data[arm+"_official_reference"]
                    partial, scored = data[arm+"_terminal"], data[arm+"_terminal_official"]
                    np.testing.assert_allclose(partial[:, -1], original, atol=.012, rtol=1e-6)
                    np.testing.assert_allclose(paths[arm+"_output_change"], original-data[arm+"_remove"], atol=.012, rtol=1e-6)
                    np.testing.assert_allclose(np.median(paths[arm+"_edge_iqr_output_change"],axis=0),audit[arm]["contrast_median"])
                    metrics = {k:v for k,v in meta[arm]["metrics"].items() if np.ndim(v) == 0}
                    terminal = meta[arm]["terminal"]
                    metrics.update(audit[arm]["metrics"], width=terminal["width"], k90=terminal["k90"],
                        share90=terminal["k90"]/terminal["width"], first_width=meta[arm]["first_width"])
                    maxima = dict(crossed_exact=audit[arm]["metrics"]["cross_numeric_max"],
                                  output_paths=audit[arm]["metrics"]["output_reconstruction_max"],
                                  natural_first=metrics["first_edit_replay_max"], raw_replay=meta[arm]["raw_replay_max"])
                    grouped[arm].append(dict(metrics=metrics, maxima=maxima, original=original, partial=partial,
                        official=official, scored=scored, individual_r2=r2(original, partial)))
            all_rows.append((meta,audit))
        fold = dict(fold=f, n_test=len(ref_ids))
        for arm, entries in grouped.items():
            averages = {k: np.mean(np.stack([e[k] for e in entries]), axis=0)
                        for k in ("partial", "original", "official", "scored")}
            original = averages["original"].astype(float)
            metrics = {k:np.median([e["metrics"][k] for e in entries]) for k in entries[0]["metrics"]}
            maxima = {k:max(e["maxima"][k] for e in entries) for k in entries[0]["maxima"]}
            fidelity = r2(original, averages["partial"])
            mae = abs(averages["scored"]-target[:, None]).mean(0)
            original_mae = abs(averages["official"]-target).mean()
            np.testing.assert_allclose(mae[-1], original_mae, atol=.01)
            assert np.all(fidelity[-1] > .999999)
            scores = dict(ensemble_raw_r2=fidelity, target_mae=mae,
                          raw_teacher_mae=abs(averages["partial"]-original[:, None]).mean(0),
                          median_member_raw_r2=np.median([e["individual_r2"] for e in entries], axis=0))
            fold[arm] = dict(medians=metrics, maxima=maxima, budgets=scores, original_target_mae=original_mae)
        result.append(fold)
    overall = {}
    for arm in ARMS:
        medians = {k:np.mean([r[arm]["medians"][k] for r in result]) for k in result[0][arm]["medians"]}
        maxima = {k:max(r[arm]["maxima"][k] for r in result) for k in result[0][arm]["maxima"]}
        scores = {k:np.mean([r[arm]["budgets"][k] for r in result], axis=0) for k in result[0][arm]["budgets"]}
        overall[arm] = dict(mean_fold_medians=medians, maxima=maxima, budgets=scores,
                            original_target_mae=np.mean([r[arm]["original_target_mae"] for r in result]))
    return dict(n_materials=len(fold_ids), model_pairs=len(all_rows), n_network_cross_checks_per_arm=32*len(all_rows),
                full_cross_grid_per_arm=sum(50*r["n_test"]**2 for r in result), folds=result, overall=overall,
                equal_terminal_width_pairs=sum(m["kan"]["terminal"]["width"] == m["mlp"]["terminal"]["width"] for m,_ in all_rows),
                discovery=feature_counts(all_rows), fixed_example=example)


def plot_curves(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from scipy.interpolate import BSpline

    plt.rcParams.update({"font.family":"sans-serif",
        "font.sans-serif":["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus":False,
        "font.size":10, "axes.spines.top":False, "axes.spines.right":False, "axes.labelcolor":"#263642",
        "axes.edgecolor":"#a5adb3", "text.color":"#253642", "xtick.color":"#44515e", "ytick.color":"#44515e"})
    fig, axes = plt.subplots(2, 3, figsize=(13.3, 7.2))
    fig.subplots_adjust(left=.083, right=.975, top=.80, bottom=.115, wspace=.29, hspace=.65)
    kan, mlp = "#006e68", "#aa562e"
    features = dict(phonons=["最小共价半径（pm）", "价电子数极差", "p 价电子比例"],
                    steel=["钼含量（at.%）", "价电子数平均绝对偏差", "最近高效堆积团簇的成分距离"])
    for r, (task, title) in enumerate((("phonons","声子"),("steel","钢"))):
        example = report[task]["fixed_example"]
        for i, ax in enumerate(axes[r]):
            g = np.array(example["kan"]["training_grid"])[:, i]
            anchor = example["kan"]["anchor"][i]
            recipe = example["kan"]["source_recipe"]
            spline = BSpline(recipe["spline_knots"], recipe["coefficients"][i], 3)
            def single_edge(raw):
                scaled = raw*recipe["input_multiplier"][i] + recipe["input_intercept"][i]
                coordinate = .5*np.tanh(scaled) if recipe["bounded"] else scaled
                return spline(np.clip(coordinate, *recipe["clipping"]))
            dense = np.linspace(g[0], g[-1], 251)
            y = single_edge(dense) - single_edge(anchor)
            np.testing.assert_allclose(single_edge(g)-single_edge(anchor), np.array(example["kan"]["response"])[:,i], atol=1e-5)
            if task == "steel" and i == 0:
                g, dense, anchor = 100*g, 100*dense, 100*anchor
            avg = np.array(example["mlp"]["response"])[:, i]
            lo, hi = np.array(example["mlp"]["background_band"])[:, :, i]
            ax.axhline(0, color="#8796a1", lw=.8)
            ax.fill_between(g, lo, hi, color=mlp, alpha=.15, linewidth=0)
            ax.plot(g, avg, color=mlp, lw=1.8, linestyle="--")
            ax.plot(dense, y, color=kan, lw=2.3)
            ax.scatter([anchor],[0],s=23,facecolors="white",edgecolors="#384855",zorder=5)
            ax.grid(axis="y",color="#dde2e4",linewidth=.7)
            ax.set_axisbelow(True)
            ax.set_xlabel(features[task][i])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            if i == 0:
                ax.set_ylabel("各自节点的激活变化")
                ax.set_title(f"{title}    KAN h[{example['kan']['node']}]    MLP h[{example['mlp']['node']}]",
                             loc="left",fontsize=11,fontweight="bold",pad=10)
    fig.text(.083,.943,"同一个物理输入换到另一种材料背景，节点响应会怎样变化？",fontsize=19,fontweight="bold")
    handles=[Line2D([],[],color=kan,lw=2.5,label="KAN 原生边"),
             Line2D([],[],color=mlp,lw=1.9,ls="--",label="MLP 训练背景平均"),
             Patch(facecolor=mlp,alpha=.18,label="MLP 留出背景第 10–90 百分位"),
             Line2D([],[],marker="o",markeredgecolor="#384855",markerfacecolor="white",ls="none",label="训练参考值")]
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.077,.917),ncol=4,frameon=False,columnspacing=2)
    fig.text(.083,.033,"固定第 0 折、第 0 个成员。每幅图只改变横轴输入，组内其余两项取参考值。纵轴刻度属于各自网络，隐藏节点的正负方向可不同。",fontsize=9,color="#4e5c66")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def as_json(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    folder = parser.parse_args().output
    hashes = dict(discovery=sha(ROOT/"scripts/discover_physical_kan_responses.py"),
                  audit=sha(ROOT/"scripts/audit_physical_kan_paths.py"), report=sha(Path(__file__)))
    summary = dict(script_sha256=hashes, budgets=BUDGETS,
        independent_unit="Five official outer folds. Member statistics use equal fold means of within-fold medians.",
        portable_error="RMS prediction error divided by RMS actual node change; lower is better. No centering.",
        named_edge_contrast="At the selected node only, set one named input edge to its training 25th then 75th percentile. Report the signed change in the original raw prediction on held-out material backgrounds. Member median then fold median; members with equal quartiles omitted from sign counts; absolute value <=0.01 in target units treated as numerically small.",
        output_fidelity="R2 against each original raw ensemble, training mean as baseline; original native terms with no refit.",
        interpretation="Interventions on descriptor/hidden tensors in a frozen network. No measurement of a material response.")
    for dataset in ("phonons", "steel"):
        summary[dataset] = aggregate(dataset, folder, hashes)
    figpath = folder/"fixed-member-input-responses.png"
    plot_curves(summary, figpath)
    summary["figure_sha256"] = sha(figpath)
    (folder/"summary.json").write_text(json.dumps(summary,default=as_json,indent=2,ensure_ascii=False,allow_nan=False),
                                       encoding="utf-8")
