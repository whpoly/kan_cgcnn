"""Assemble reproducible descriptive evidence from completed KAN experiments.

This report replays saved physical components, aggregates existing ablations,
and visualizes frozen-model internals. It never trains or selects a new model.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_interpretable_phonons_kan import basis

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/interpretability-review-20261106"
PHYSICAL = ROOT / "benchmarks/interpretable-phonons-kan-20261027"
SOURCES = {}
LABELS = {
    "mean_mass": ("Mean atomic mass", "u"),
    "mean_radius": ("Mean covalent radius", "pm"),
    "radius_deviation": ("Covalent radius average deviation", "pm"),
    "mean_bond_length": ("Mean bond length", r"$\AA$"),
    "bond_length_deviation": ("Bond length standard deviation", r"$\AA$"),
    "electronegativity_deviation": ("Electronegativity average deviation", "Pauling"),
}


def read(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8-sig"))


def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=175, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def check_components():
    members = {}
    max_terms = 0.
    max_prediction = 0.
    for variant in ["physical_additive", "physical_product", "hybrid_additive", "hybrid_product"]:
        group = []
        for path in sorted((PHYSICAL/"fits"/variant).glob("*/edge-functions.json")):
            e = read(path)
            result = read(path.parent/"result.json")
            predictions = path.parent/"predictions.npz"
            assert hashlib.sha256(predictions.read_bytes()).hexdigest() == result["artifacts"]["predictions.npz"]
            SOURCES[str(predictions.relative_to(ROOT))] = result["artifacts"]["predictions.npz"]
            with np.load(predictions) as archive:
                a = {k:archive[k] for k in archive.files}
            for split in ["train", "validation"]:
                b = basis(a[f"x_{split}"]) - np.asarray(e["centers"])
                terms = [np.einsum("nib,ib->ni", b, np.asarray(e["main"]))]
                if e["factors"] is not None:
                    factors = np.asarray(e["factors"])
                    interactions = []
                    for k,(i,j) in enumerate(e["pairs"]):
                        product_center = factors[k,0] @ np.asarray(e["product_centers"][k]) @ factors[k,1]
                        interactions.append((b[:,i] @ factors[k,0])*(b[:,j] @ factors[k,1])-product_center)
                    terms.append(np.column_stack(interactions))
                terms = np.concatenate(terms, axis=1)*e["gate"]*e["target_std"]
                expected = a[f"{split}_terms"]
                max_terms = max(max_terms, float(np.max(abs(terms-expected))))
                np.testing.assert_allclose(terms, expected, atol=1e-7, rtol=0)
                prediction = a["prediction"] if split == "validation" else a["train_prediction"]
                reconstructed = a[f"{split}_backbone"]+float(a["bias"])+terms.sum(axis=1)
                max_prediction = max(max_prediction, float(np.max(abs(prediction-reconstructed))))
                np.testing.assert_allclose(prediction, reconstructed, atol=1e-7, rtol=0)
            group.append((e,a))
        assert len(group) == 10
        members[variant] = group
    return members, dict(verified_models=40, max_term_replay_error=max_terms,
                         max_prediction_replay_error=max_prediction)


def aggregate_components(analysis):
    rows = []
    for variant in ["physical_additive", "physical_product", "hybrid_additive", "hybrid_product"]:
        g = [r for r in analysis["symbolic_and_ablation"] if r["variant"]==variant]
        stability = {r["feature"]:r for r in analysis["main_curve_stability"][variant]}
        for i,name in enumerate(g[0]["component_names"]):
            delta = np.asarray([r["component_removal_validation_mae_delta"][i] for r in g])
            rows.append(dict(variant=variant, component=name,
                mean_abs_training_contribution=float(np.mean([r["train_importance"][i] for r in g])),
                mean_removal_validation_mae_delta=float(delta.mean()),
                removal_positive_runs=int((delta>0).sum()),
                removal_delta_range=[float(delta.min()),float(delta.max())],
                stability=stability.get(name)))
    return rows


def physical_plots(members, rows):
    chosen = ["mean_radius", "radius_deviation", "mean_bond_length", "mean_mass"]
    fig, axes = plt.subplots(2,2,figsize=(11.8,8.1),constrained_layout=True)
    curves = {}
    for ax,name in zip(axes.flat,chosen):
        row = next(r for r in rows if r["variant"]=="physical_product" and r["component"]==name)
        grid = np.linspace(*row["stability"]["range"],160)
        lines = []
        for e,a in members["physical_product"]:
            i = e["short_names"].index(name)
            coordinate = grid*a["scaler_scale"][i]+a["scaler_min"][i]
            response = (basis(coordinate)-np.asarray(e["centers"][i])) @ np.asarray(e["main"][i])
            lines.append(response*e["gate"]*e["target_std"])
        lines = np.asarray(lines)
        for line in lines:
            ax.plot(grid,line,color="#528eab",lw=.65,alpha=.35)
        mean,std = lines.mean(axis=0),lines.std(axis=0)
        ax.fill_between(grid,mean-std,mean+std,color="#247e9b",alpha=.14)
        ax.plot(grid,mean,color="#07566e",lw=2.4)
        ax.axhline(0,color="#9cabb5",lw=.7,ls="--")
        ax.set(xlabel=f"{LABELS[name][0]} ({LABELS[name][1]})",ylabel=r"Exact main component (cm$^{-1}$)")
        ax.set_title(f"Curve correlation {row['stability']['median_pairwise_correlation']:.3f}  |  removal +{row['mean_removal_validation_mae_delta']:.1f} MAE",fontsize=10)
        ax.grid(alpha=.15)
        derivatives = np.gradient(lines,grid,axis=1)
        reference_index = len(grid)//2
        curves[name] = dict(grid=grid.tolist(),responses=lines.tolist(),
            common_support=row["stability"]["range"],
            mean_endpoint_change=float(np.mean(lines[:,-1]-lines[:,0])),
            reference=float(grid[reference_index]),
            mean_local_derivative=float(derivatives[:,reference_index].mean()),
            local_derivative_range=derivatives[:,reference_index].tolist(),
            negative_derivative_fraction=float(np.mean(derivatives<0)),
            derivative_units=f"cm^-1 per {LABELS[name][1]}")
    fig.suptitle("Physical product KAN: directly readable components\n2 seeds x 5 inner splits; bands show run SD, not confidence intervals",fontsize=14)
    savefig(fig,"physical-components")
    return curves


def ale_plot(summary):
    fig,axes=plt.subplots(1,3,figsize=(12.3,3.8),constrained_layout=True)
    for ax,name,unit in zip(axes,["mean_mass","mean_bond_length","mean_covalent_radius"],["u",r"$\AA$","pm"]):
        for arm,color,label in [("mm_modnet","#9b5b32","MLP / MODNet"),("mm_kan","#126682","KAN")]:
            row=next(r for r in summary["rows"] if r["feature"]==name and r["arm"]==arm)
            grid=np.linspace(*row["common_central_domain"],100)
            values=[]
            for fold in range(5):
                curve=read(ROOT/f"benchmarks/native-physical-ale-20261101/{arm}/fold-{fold}.json")["curves"][name]
                values.append(np.interp(grid,curve["x"],curve["ale"]))
            values=np.asarray(values)
            ax.plot(grid,values.mean(axis=0),color=color,lw=2,label=f"{label}, r={row['median_curve_correlation']:.3f}")
            ax.fill_between(grid,values.mean(axis=0)-values.std(axis=0),values.mean(axis=0)+values.std(axis=0),color=color,alpha=.14)
        ax.set(xlabel=f"{name.replace('_',' ')} ({unit})",ylabel=r"ALE response (cm$^{-1}$)")
        ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle("Completed phonons benchmark: stable responses occur in both architectures\nFive outer-fold ensembles; ALE evaluated on outer-training descriptors",fontsize=12)
    savefig(fig,"benchmark-response-comparison")


def native_plot(native):
    r=next(r for r in native["records"] if r["task"]=="phonons" and r["fold"]==0)
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.7),constrained_layout=True)
    for ax,e,label,unit in zip(axes[0], [r["first_layer_examples"][0],r["first_layer_examples"][2]], ["Mean atomic mass","Mean bond length"],["u",r"$\AA$"]):
        ax.plot(e["raw_grid"],e["centered_response"],color="#136b86",lw=2.3)
        ax.axhline(0,color="grey",lw=.6)
        ax.set(xlabel=f"{label} ({unit})",ylabel="First hidden coordinate contribution",
            title=f"Input -> h1[{e['output_node']}]; affine relative error {e['affine_relative_rmse']:.1%}")
        ax.grid(alpha=.15)
    magnitude=np.asarray(r["mean_abs_last_contribution"]);order=np.argsort(-magnitude)
    axes[1,0].bar(range(len(order)),magnitude[order],color="#397f99")
    axes[1,0].set_xticks(range(len(order)),[str(i) for i in order],fontsize=8)
    axes[1,0].set(xlabel="Last latent node index (ranked on training data)",ylabel=r"Mean |centered contribution| (cm$^{-1}$)",title="7 of 16 terms carry 90% of absolute magnitude")
    sample=r["sample"];c=np.asarray(sample["contributions"]);chosen=order[:5]
    values=np.r_[c[chosen],c.sum()-c[chosen].sum()]
    axes[1,1].barh(range(6),values,color=["#b46748" if v<0 else "#327f99" for v in values])
    axes[1,1].set_yticks(range(6),[f"c[{i}]" for i in chosen]+["Other 11"])
    axes[1,1].axvline(0,color="grey",lw=.7)
    axes[1,1].set(xlabel=r"Signed contribution (cm$^{-1}$)",title=f"{sample['id']}: {sample['baseline']:.2f} + ({c.sum():.2f}) = {sample['prediction']:.2f}")
    fig.suptitle("Frozen benchmark KAN: exact, inspectable internal quantities\nPhonons, fold 0, first recorded member; latent nodes have no assigned physical meaning",fontsize=13)
    savefig(fig,"native-intermediates")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
    analysis=read(PHYSICAL/"interpretability-analysis.json")
    members,audit=check_components()
    rows=aggregate_components(analysis)
    curves=physical_plots(members,rows)
    ale=read(ROOT/"benchmarks/native-physical-ale-20261101/summary.json")
    ale_plot(ale)
    native=read(OUT/"native-internals.json")
    native_plot(native)
    benchmarks={name:read(ROOT/path/"analysis.json") for name,path in {
        "phonons":"benchmarks/native-fourway-scaled-kan-20261029",
        "steels":"benchmarks/matbench-expansion-20261101/matbench_steels"}.items()}
    edge=read(ROOT/"benchmarks/native-kan-edge-compression-20261101/interpretation-verdict.json")
    proxy=read(ROOT/"benchmarks/native-physical-proxy-20261101/interpretation-verdict.json")
    example_member=members["physical_product"][0]
    e,a=example_member
    i=int(np.argsort(a["sample_ids"])[0])
    names=list(e["short_names"])+[f"{e['short_names'][j]}*{e['short_names'][k]}" for j,k in e["pairs"]]
    sample=dict(id=str(a["sample_ids"][i]),selection="first validation ID in lexical order; seed 0, inner fold 0",
        baseline=float(a["validation_backbone"][i]+a["bias"]),prediction=float(a["prediction"][i]),
        contributions=dict(zip(names,a["validation_terms"][i].tolist())))
    report=dict(date="2026-11-06",complete=True,training_performed=False,
        completed_benchmarks={name:dict(means=b["means"],comparisons=b["comparisons"]) for name,b in benchmarks.items()},
        physical_component_audit=audit,physical_components=rows,physical_curves=curves,
        physical_sample=sample,physical_model_comparison=analysis["rows"],
        interaction_stability=analysis["product_surface_stability"],
        correlated_descriptor_pairs=analysis["strongly_correlated_physical_features"],
        benchmark_ale=ale["rows"],native_internals_scope=native["scope"],
        native_internals_max_reconstruction_error=max(r["max_validation_reconstruction_error"] for r in native["records"]),
        compression=edge,physical_proxy=proxy,source_sha256=SOURCES,
        limitations=["Post-hoc exploration of already examined experiments; no new confirmatory benchmark.",
            "Physical additive/product experiment has inner validation only; native ensemble scores are outer-fold scores.",
            "Component removal measures dependency of a fixed model, not a physical intervention or retrained feature importance.",
            "Stability is descriptive; overlapping cross-validation splits are not independent replications.",
            "Native hidden-node decomposition is exact but not unique or physically identified.",
            "Stable ALE curves and terminal additive decompositions can also be obtained for MLPs."])
    (OUT/"evidence.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(json.dumps(dict(audit=audit,native_reconstruction_max=report["native_internals_max_reconstruction_error"],sample=sample,curve_parameters={k:{j:v for j,v in d.items() if j not in ["grid","responses","local_derivative_range"]} for k,d in curves.items()}),ensure_ascii=False))


if __name__=="__main__":
    main()
