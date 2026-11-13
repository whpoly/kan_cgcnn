"""Verify and summarize the five-fold steel physical readouts and their phonons reference."""
from pathlib import Path

import analyze_steel396_concepts as steel
from run_modnet_official_matched import file_hash, np, write_json


OUT, ROOT, NAMES = steel.OUT, steel.ROOT, steel.NAMES
PHONONS = ROOT / "benchmarks/formal-phonons-concepts-20261106"
ROBUST = ROOT / "benchmarks/formal-phonons-readout-robustness-20261106"
read = steel.read


def r2(truth, pred):
    residual = np.sum((truth - pred)**2, axis=-2)
    total = np.sum((truth - truth.mean(axis=-2, keepdims=True))**2, axis=-2)
    assert (total > 0).all()
    return 1 - residual / total


def correlation(truth, pred):
    y = truth - truth.mean(axis=-2, keepdims=True)
    p = pred - pred.mean(axis=-2, keepdims=True)
    norm = np.sqrt((y**2).sum(axis=-2) * (p**2).sum(axis=-2))
    return np.divide((y * p).sum(axis=-2), norm, where=norm > 0, out=np.zeros_like(norm))


def inspect_fold(arm, fold):
    folder = OUT / f"fold-{fold}"
    meta = read(folder / f"{arm}.json")
    assert len(meta["members"]) == 50 and meta["extraction_sha256"] == file_hash(steel.__file__)
    assert meta["arrays_sha256"] == file_hash(folder / f"{arm}.npz")
    for member in meta["members"]:
        path = ROOT / member["path"]
        assert member["sha256"] == file_hash(path) and member["fit_sha256"] == file_hash(path.with_name("result.json"))
    with np.load(folder / f"{arm}.npz") as z:
        a = {key: z[key] for key in z.files}
    assert a["names"].tolist() == NAMES and all(np.isfinite(v).all() for v in a.values() if v.dtype.kind == "f")
    raw, official, q, y = a["raw"], a["official"], a["q_test"], a["target"]
    original = (steel.KAN / f"fold-{fold}" if arm == "kan" else steel.MLP / f"fold-{fold}/top50")
    assert meta["member_predictions_sha256"] == file_hash(original / "outer-member-predictions.npz")
    with np.load(original / "test.npz") as z:
        np.testing.assert_array_equal(z["ids"], a["test_ids"])
        np.testing.assert_array_equal(z["target"], y)
        np.testing.assert_array_equal(z["prediction"], official.mean(axis=0))
    official_mae = np.mean(np.abs(official.mean(axis=0) - y))
    raw_mae = np.mean(np.abs(raw.mean(axis=0) - y))
    cr, ni = NAMES.index("chromium"), NAMES.index("nickel")
    shared = a["first_nodes"][:, cr] == a["first_nodes"][:, ni]
    info = dict(fold=fold, test_ids=a["test_ids"].tolist(), official_mae=official_mae, raw_mae=raw_mae,
                replaced=int(np.sum(raw != official)), first_width=np.mean(a["first_width"]),
                late_width=np.mean(a["late_width"]), late_min_width=int(min(a["late_width"])),
                shared_cr_ni_first=int(shared.sum()), shared_cr_ni_opposite_slopes=int(np.sum(
                    shared & (a["first_slope"][:, cr] * a["first_slope"][:, ni] < 0))),
                replay_max=max(r["replay_max_mpa"] for r in meta["members"]),
                official_replay_max=max(r["official_replay_max_mpa"] for r in meta["members"]),
                decomposition_max=max(r["final_layer_sum_max_error_mpa"] for r in meta["members"]))
    rows = {}
    for layer in ("first", "late"):
        def arr(key):
            return a[f"{layer}_{key}"]
        single = arr("single")
        np.testing.assert_allclose(arr("offset")[:, None, :] + arr("slope")[:, None, :] * arr("hidden"), single)
        single_r2 = r2(q, single)
        np.testing.assert_allclose(single_r2, arr("single_r2"), atol=1e-10)
        np.testing.assert_allclose(correlation(q, arr("hidden")), arr("rho"), atol=1e-10)
        strong = (np.abs(arr("rho")) >= .8) & (single_r2 >= .5)
        specific = strong & (np.abs(arr("partial_output")) >= .5) & (np.abs(arr("partial_other")) >= .5)
        ridge = arr("ridge").mean(axis=0)
        metrics = dict(strong=strong.mean(axis=0), specific=specific.mean(axis=0),
                       single_median_r2=np.median(single_r2, axis=0), single_ensemble_r2=r2(q, single.mean(axis=0)),
                       null_median_r2=np.median(arr("null_r2"), axis=0),
                       partial_output_abs_median=np.median(np.abs(arr("partial_output")), axis=0),
                       partial_other_abs_median=np.median(np.abs(arr("partial_other")), axis=0),
                       full_r2=r2(q, ridge), rank4_r2=r2(q, arr("rank4").mean(axis=0)),
                       full_rmse=np.sqrt(np.mean((q - ridge)**2, axis=0)),
                       full_median_r2=np.median(r2(q, arr("ridge")), axis=0),
                       output_r2=r2(q, a["output_only"].mean(axis=0)),
                       direct_input_rate=a["direct"].mean(axis=0), best_input_median_r2=np.median(a["input_r2"], axis=0))
        for group, prefix in ([("single", ""), ("three", "three_")] if layer == "late" else [("single", "")]):
            for kind, base_mae in (("official", official_mae), ("raw", raw_mae)):
                ablated = arr(f"{prefix}ablated_{kind}").mean(axis=0)
                delta = np.mean(np.abs(ablated - y[:, None]), axis=0) - base_mae
                metrics[f"{group}_delta_{kind}"] = delta[:len(NAMES)]
                info[f"{layer}_{group}_random_delta_{kind}"] = delta[len(NAMES):].tolist()
        if layer == "late":
            three = arr("three")
            three_r2 = r2(q, three)
            metrics.update(three_r2=r2(q, three.mean(axis=0)), three_median_r2=np.median(three_r2, axis=0),
                           three_strong=((three_r2 >= .5) & (np.abs(correlation(q, three)) >= .8)).mean(axis=0),
                           unregularized_three_r2=r2(q, arr("ols").mean(axis=0)),
                           unregularized_three_median_r2=np.median(r2(q, arr("ols")), axis=0))
        rows[layer] = {name: {key: values[i] for key, values in metrics.items()} for i, name in enumerate(NAMES)}
        for j, name in enumerate(steel.PARENTS):
            row = rows[layer][name]
            row["partial_parent_abs_median"] = np.median(np.abs(arr("partial_parent")[:, j]))
            if layer == "late":
                truth = q[:, [NAMES.index(name)]]
                parent = arr("parent_only").mean(axis=0)[:, [j]]
                row["parent_only_r2"] = r2(truth, parent)[0]
                row["residual_full_r2"] = r2(truth - parent, arr("parent_residual_decoded").mean(axis=0)[:, [j]])[0]
    return info, rows


def aggregate(arm):
    pairs = [inspect_fold(arm, f) for f in range(5)]
    folds = [info for info, _ in pairs]
    ids = [i for f in folds for i in f["test_ids"]]
    assert len(set(ids)) == len(ids) == 312
    result = dict(folds=folds)
    for layer in ("first", "late"):
        result[layer] = {}
        for name in NAMES:
            rows = [r[layer][name] for _, r in pairs]
            means = {key: np.mean([r[key] for r in rows]) for key in rows[0]}
            result[layer][name] = dict(mean=means, folds=rows, stable=min(r["strong"] for r in rows) >= .7,
                                       specific_stable=min(r["specific"] for r in rows) >= .7)
    return result


def fixed_case():
    folder = OUT / "fold-0"
    with np.load(folder / "kan.npz") as z, np.load(folder / "first-kan-case.npz") as first, np.load(folder / "late-kan-case.npz") as late:
        np.testing.assert_array_equal(z["test_ids"], first["test_ids"])
        np.testing.assert_array_equal(z["test_ids"], late["test_ids"])
        np.testing.assert_allclose(late["test_activation"] @ late["weights"] + late["offsets"], z["late_ridge"][0])
        q = z["q_test"]
        irow = np.argmin(z["test_ids"])
        rows = {}
        for i, name in enumerate(NAMES):
            first_node = z["first_nodes"][0, i]
            sparse = z["late_three_nodes"][0, i]
            weights, offset = z["late_three_slopes"][0, i], z["late_three_offsets"][0, i]
            np.testing.assert_array_equal(first["test_activation"][:, first_node], z["first_hidden"][0, :, i])
            np.testing.assert_allclose(late["test_activation"][:, sparse] @ weights + offset, z["late_three"][0, :, i])
            magnitude = first["mean_abs_centered_edge"][:, first_node]
            cols = first["columns"]
            rows[name] = dict(first_node=int(first_node), slope=z["first_slope"][0, i], offset=z["first_offset"][0, i],
                              first_r2=z["first_single_r2"][0, i], first_r=z["first_rho"][0, i],
                              partial_output=z["first_partial_output"][0, i], partial_other=z["first_partial_other"][0, i],
                              top_first_edges=[dict(name=str(cols[j]), share=float(magnitude[j]/magnitude.sum()))
                                               for j in np.argsort(-magnitude)[:5]],
                              roost_edge_share=magnitude[np.char.startswith(cols, "ROOST_")].sum()/magnitude.sum(),
                              late_three_nodes=sparse, late_three_weights=weights, late_three_offset=offset,
                              late_three_r2=r2(q, z["late_three"][0])[i], late_full_r2=r2(q, z["late_ridge"][0])[i],
                              example_true=q[irow, i], example_three=z["late_three"][0, irow, i])
        return dict(fold=0, member=0, n_test=len(q), example_id=str(z["test_ids"][irow]), quantities=rows,
                    first_weight_replay_error=first["replay_error"].item())


def phonon_reference():
    primary, robust = (read(p / "interpretation.json") for p in (PHONONS, ROBUST))
    for folder in (PHONONS, ROBUST):
        audit = read(folder / "audit.json")
        assert audit["passed"] and audit["models"] == 1000
        assert audit["interpretation_sha256"] == file_hash(folder / "interpretation.json")
    rows = []
    for r in primary["rows"]:
        if r["arm"].startswith("mm_mv") and r["concept"] in ("mean_bond_length", "volume_per_atom", "inverse_cuberoot_volume"):
            rr = next(v for v in robust["rows"] if (v["arm"], v["layer"], v["concept"]) == (r["arm"], r["layer"], r["concept"]))
            rows.append(dict(arm=r["arm"], layer=r["layer"], concept=r["concept"],
                             single_r2=r["mean"]["single_member_test_r2_median"],
                             strong_by_fold=[f["strongly_aligned_fraction"] for f in r["folds"]],
                             delta=r["mean"]["ablation_mae_delta"], full_r2=rr["global_scale_mean_r2"],
                             rank4_r2=rr["rank4_mean_r2"]))
    return rows


def figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    chinese = Path("C:/Windows/Fonts/msyh.ttc")
    if chinese.is_file():
        font_manager.fontManager.addfont(str(chinese))
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(chinese)).get_name(), "DejaVu Sans"]
    plt.rcParams.update({"axes.spines.right": False, "axes.spines.top": False, "axes.unicode_minus": False,
                         "font.size": 9.5, "axes.titlelocation": "left", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), constrained_layout=True)
    blue, rust, green, grey = "#275e9b", "#b3582c", "#14796c", "#64696e"
    def scatter(ax, h, q, slope, offset, color, label):
        ax.scatter(h, q, s=14, alpha=.55, color=color, linewidths=0)
        x = np.array([h.min(), h.max()])
        ax.plot(x, offset + slope*x, color=color, linewidth=1.8, label=label)
    case = PHONONS / "mm_mv_kan/fold-0/member-00"
    row = next(r for r in read(case / "summary.json")["records"] if r["layer"] == "first" and r["concept"] == "mean_bond_length")
    with np.load(case / "arrays.npz") as a:
        idx = steel.phonons.NAMES.index("mean_bond_length")
        scatter(axes[0, 0], a["first_selected_activation_test"][:, idx], a["q_test"][:, idx],
                row["slope"], row["offset"], blue, f'平均键长    R² = {row["test_r2"]:.2f}')
    axes[0, 0].set(title="声子  ·  第一层几何坐标", xlabel=f'节点 h[{row["node"]}]', ylabel="平均键长（Å）")
    axes[0, 0].legend(frameon=False)
    with np.load(OUT / "fold-0/kan.npz") as a:
        for name, label, color in (("chromium", "Cr", rust), ("nickel", "Ni", green)):
            i = NAMES.index(name)
            row = summary["fixed_case"]["quantities"][name]
            assert row["first_node"] == summary["fixed_case"]["quantities"]["chromium"]["first_node"]
            scatter(axes[0, 1], a["first_hidden"][0, :, i], a["q_test"][:, i], row["slope"], row["offset"],
                    color, f'{label}    R² = {row["first_r2"]:.2f}')
        axes[0, 1].set(title="钢  ·  同一节点上的铬与镍", xlabel=f'节点 h[{row["first_node"]}]', ylabel="原子百分比（at.%）")
    axes[0, 1].legend(frameon=False, ncol=2)
    ax = axes[1, 0]
    for name, label, color in (("mean_bond_length", "声子 · 键长", blue), ("volume_per_atom", "声子 · 体积", "#7b6eaf"),
                               ("chromium", "钢 · Cr", rust), ("nickel", "钢 · Ni", green)):
        if name in NAMES:
            values = [100 * r["strong"] for r in summary["steel"]["kan"]["first"][name]["folds"]]
        else:
            row = next(r for r in summary["phonons"] if (r["arm"], r["layer"], r["concept"]) == ("mm_mv_kan", "first", name))
            values = 100 * np.array(row["strong_by_fold"])
        ax.plot(range(5), values, "o-", markersize=4, color=color, label=label)
    ax.axhline(70, linewidth=1, linestyle="--", color=grey)
    ax.set(xticks=range(5), ylim=(0, 105), title="五折中第一层强对应成员的比例", xlabel="外层折", ylabel="每折 50 个成员中的占比（%）")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower left")
    ax = axes[1, 1]
    concepts = ["carbon_nitrogen", "chromium", "nickel", "size_mismatch", "cr_equivalent", "ni_equivalent"]
    ypos = np.arange(len(concepts))[::-1]
    for arm, metric, color, marker, label, dy in [
        ("kan", "output_r2", grey, "x", "仅输出", .13), ("kan", "three_r2", rust, "^", "KAN 三节点", .045),
        ("kan", "full_r2", green, "o", "KAN 整层", -.045), ("mlp", "full_r2", blue, "s", "MLP 整层", -.13)]:
        scores = [summary["steel"][arm]["late"][n]["mean"][metric] for n in concepts]
        ax.scatter(scores, ypos + dy, color=color, marker=marker, s=28, label=label)
    ax.set(yticks=ypos, yticklabels=["C + N", "Cr", "Ni", "尺寸失配 δ", "铬当量", "镍当量"],
           xlim=(-.03, 1.02), title="钢的深层成分读出", xlabel="五折平均留出 R²（成员读出先集成）")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(.24, .49))
    for ax in axes.flat:
        ax.grid(axis="y", color="#d4d8d9", alpha=.5, linewidth=.6)
        ax.set_axisbelow(True)
    fig.savefig(OUT / "phonons-steel-comparison.png", dpi=175)
    plt.close(fig)


def main():
    arms = {arm: aggregate(arm) for arm in ("kan", "mlp")}
    comparison = read(steel.KAN / "comparison.json")
    for arm, key in (("kan", "mean_kan_mpa"), ("mlp", "mean_modnet_mpa")):
        np.testing.assert_allclose(np.mean([f["official_mae"] for f in arms[arm]["folds"]]), comparison[key], atol=1e-8, rtol=0)
    summary = dict(concepts=steel.CONCEPTS, parents=steel.PARENTS, steel=arms, phonons=phonon_reference(), fixed_case=fixed_case(),
                   protocol=dict(single="Select by maximum absolute train Pearson r. Fit affine readout on the same member training IDs.",
                       strong="Outer |r| >= 0.8 and affine R2 >= 0.5; stable if >=70% of 50 members in every one of five folds.",
                       specific="Strong plus |partial r| >= 0.5 separately after train-fitted removal of raw prediction and other-family direct concepts.",
                       ridge="Single layer-wide RMS scale, train means, fixed ridge penalty 0.01. Rank-four uses up to four training PCs.",
                       three="Exploratory follow-up: forward selection of three distinct late native coordinates. Shared layer scale and ridge 0.01 in selection and fit. Unregularized fits retained as sensitivity results.",
                       ablation="First layer: replace selected unit with train mean. Late layer: subtract selected final-edge contributions centered on train. Replay official repair with original seed; retain raw output as control.",
                       random_control="Eight deterministic draws of one unit or three distinct units per member. No alignment with physical labels.",
                       aggregation="Member-median single-unit R2 and fraction strong per fold. Multivariate physical predictions averaged over members before fold R2. Fold means unweighted.",
                       case="Fold 0, member 0. Example row is the first outer ID in lexicographic order."),
                   models={"kan": 250, "mlp": 250}, sources={
                       p.name if p.is_file() and p.suffix == ".py" else str(p.relative_to(ROOT)): file_hash(p)
                       for p in (Path(__file__), Path(steel.__file__), Path(steel.phonons.__file__),
                                 Path(__file__).with_name("audit_formal_phonons_readouts.py"), steel.INPUT / "inputs.npz",
                                 steel.KAN / "protocol.json", steel.KAN / "comparison.json", steel.MLP / "audit.json",
                                 PHONONS / "interpretation.json", ROBUST / "interpretation.json")})
    write_json(OUT / "summary.json", summary)
    figure(summary)
    print("Verified 500 frozen members, five disjoint outer test sets and both original ensemble scores.")


if __name__ == "__main__":
    main()
