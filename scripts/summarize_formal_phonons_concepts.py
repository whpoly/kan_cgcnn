"""Summarize all formal phonons members without choosing a new predictor."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"benchmarks/formal-phonons-concepts-20261106"
ARMS=["mm_modnet","mm_kan","mm_mv_modnet","mm_mv_kan"]
LABELS={"mm_modnet":"MM MLP","mm_kan":"MM KAN","mm_mv_modnet":"MM+MV MLP","mm_mv_kan":"MM+MV KAN"}
CONCEPTS=["mean_mass","mean_radius","radius_deviation","mean_bond_length","bond_length_deviation","volume_per_atom",
    "mean_electronegativity","mean_en_difference","coordination","density","inverse_sqrt_mass","inverse_cuberoot_volume"]
SHORT=["Mean mass","Mean radius","Radius spread","Mean bond length","Bond length spread","Volume / atom",
    "Mean EN","EN difference","Coordination","Density","1 / sqrt(mass)","1 / cbrt(volume)"]


def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(fig,name):
    fig.savefig(OUT/f"{name}.png",dpi=170,bbox_inches="tight")
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)


def aggregate(analysis):
    original=read(ROOT/"benchmarks/native-fourway-scaled-kan-20261029/analysis.json")
    protocol=read(OUT/"protocol.json")
    rows=[];models=[];source_hashes={}
    for arm in ARMS:
        folds=[f for f in analysis["folds"] if f["arm"]==arm]
        assert len(folds)==5 and all(f["members"]==50 for f in folds)
        assert abs(np.mean([f["baseline_mae"] for f in folds])-original["means"][arm])<.0001
        prediction_baseline={}
        for fold in range(5):
            baselines=[]
            for member in range(50):
                directory=OUT/arm/f"fold-{fold}"/f"member-{member:02d}"
                r=read(directory/"summary.json")
                assert r["protocol_fingerprint"]==analysis["protocol_fingerprint"]
                assert sha(directory/"arrays.npz")==r["arrays_sha256"]
                source_hashes[str(directory.relative_to(ROOT)/"summary.json")]=sha(directory/"summary.json")
                models.append(r)
                with np.load(directory/"arrays.npz") as arrays:
                    baselines.append(arrays["prediction_only_decoded"])
                    qt=arrays["q_test"]
            decoded=np.mean(baselines,axis=0)
            prediction_baseline[fold]=1-np.sum((qt-decoded)**2,axis=0)/np.sum((qt-qt.mean(axis=0))**2,axis=0)
        for layer in ["first","late"]:
            for concept in CONCEPTS:
                group=[dict(next(x for x in f["rows"] if x["layer"]==layer and x["concept"]==concept)) for f in folds]
                for entry in group:
                    f=entry["fold"]
                    entry["prediction_only_ensemble_r2"]=float(prediction_baseline[f][CONCEPTS.index(concept)])
                    entry["distributed_r2_gain_over_prediction"]=entry["ensemble_multi_decoder_r2"]-entry["prediction_only_ensemble_r2"]
                    with np.load(OUT/arm/f"fold-{f}"/"ensemble-arrays.npz") as arrays:
                        ci=CONCEPTS.index(concept)
                        truth=arrays["q_test"][:,ci]
                        entry["distributed_readout_rmse_physical_units"]=float(np.sqrt(np.mean((arrays[f"{layer}_multi_decoded"][:,ci]-truth)**2)))
                        entry["single_readout_rmse_physical_units"]=float(np.sqrt(np.mean((arrays[f"{layer}_single_decoded"][:,ci]-truth)**2)))
                keys=[k for k,v in group[0].items() if isinstance(v,(int,float)) and k not in ["fold","members"]]
                record=dict(arm=arm,layer=layer,concept=concept,
                    mean={k:float(np.mean([x[k] for x in group])) for k in keys},
                    folds=group,
                    min_strong_fraction=min(x["strongly_aligned_fraction"] for x in group),
                    min_specific_fraction=min(x["concept_specific_fraction"] for x in group),
                    stable_single_node=all(x["strongly_aligned_fraction"]>=.7 for x in group),
                    stable_specific_single_node=all(x["concept_specific_fraction"]>=.7 for x in group),
                    distributed_readout_all_folds_r2_above_half=all(x["ensemble_multi_decoder_r2"]>=.5 for x in group),
                    ablation_positive_folds=sum(x["ablation_mae_delta"]>0 for x in group))
                arm_models=[m for m in models if m["arm"]==arm]
                source_concept={"inverse_sqrt_mass":"mean_mass","inverse_cuberoot_volume":"volume_per_atom"}.get(concept,concept)
                column=protocol["concepts"][source_concept][0]
                record["direct_input_present_fraction"]=float(np.mean([column in m["selected_descriptor_names"] for m in arm_models]))
                record["mean_input_features"]=float(np.mean([m["n_features"] for m in arm_models]))
                record["mean_representation_width"]=float(np.mean([next(x["width"] for x in m["records"] if x["layer"]==layer) for m in arm_models]))
                rows.append(record)
    comparisons=[]
    for bank in ["mm","mm_mv"]:
        for layer in ["first","late"]:
            for concept in CONCEPTS:
                k=next(x for x in rows if x["arm"]==bank+"_kan" and x["layer"]==layer and x["concept"]==concept)
                m=next(x for x in rows if x["arm"]==bank+"_modnet" and x["layer"]==layer and x["concept"]==concept)
                comparisons.append(dict(bank=bank,layer=layer,concept=concept,
                    kan_minus_mlp_mean_single_r2=k["mean"]["single_member_test_r2_median"]-m["mean"]["single_member_test_r2_median"],
                    kan_minus_mlp_mean_distributed_r2=k["mean"]["ensemble_multi_decoder_r2"]-m["mean"]["ensemble_multi_decoder_r2"],
                    kan_single_r2_wins=sum(a["single_member_test_r2_median"]>b["single_member_test_r2_median"] for a,b in zip(k["folds"],m["folds"])),
                    kan_distributed_r2_wins=sum(a["ensemble_multi_decoder_r2"]>b["ensemble_multi_decoder_r2"] for a,b in zip(k["folds"],m["folds"]))))
    return rows,models,comparisons,source_hashes


def heatmaps(rows):
    fig,axes=plt.subplots(2,2,figsize=(17,9.7),constrained_layout=True)
    for li,layer in enumerate(["first","late"]):
        for ki,(key,title) in enumerate([("single_member_test_r2_median","Single selected node: mean of fold medians"),
                                        ("ensemble_multi_decoder_r2","Distributed linear readout: mean fold R2")]):
            matrix=np.array([[next(x for x in rows if x["arm"]==arm and x["layer"]==layer and x["concept"]==c)["mean"][key] for c in CONCEPTS] for arm in ARMS])
            ax=axes[li,ki];im=ax.imshow(matrix,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
            ax.set_yticks(range(4),[LABELS[a] for a in ARMS]);ax.set_xticks(range(12),SHORT,rotation=60,ha="right",fontsize=8)
            ax.set_title(f"{layer.capitalize()} representation | {title}",fontsize=11)
            for i in range(4):
                for j in range(12):
                    value=matrix[i,j];label=f"{value:.0e}" if value < -10 else f"{value:.2f}"
                    ax.text(j,i,label,ha="center",va="center",fontsize=8,color="white" if value>.6 else "black")
            fig.colorbar(im,ax=ax,label="Outer-held-out R2 (negative values share zero color)",shrink=.75)
    fig.suptitle("Original unit-standardized probes: numerical extrapolation failures retained\nDo not infer architectural superiority from failed readouts; see the global-scale robustness audit",fontsize=14)
    save(fig,"physical-information-readout")


def stable_map(rows):
    fig,axes=plt.subplots(1,2,figsize=(15,5.4),constrained_layout=True)
    for ax,layer in zip(axes,["first","late"]):
        matrix=np.array([[next(x for x in rows if x["arm"]==a and x["layer"]==layer and x["concept"]==c)["min_strong_fraction"] for c in CONCEPTS] for a in ARMS])
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
        ax.set_yticks(range(4),[LABELS[a] for a in ARMS]);ax.set_xticks(range(12),SHORT,rotation=65,ha="right",fontsize=8)
        ax.set_title(layer.capitalize()+" representation")
        for i in range(4):
            for j in range(12):
                ax.text(j,i,f"{matrix[i,j]:.0%}"+("*" if matrix[i,j]>=.7 else ""),ha="center",va="center",fontsize=8,color="white" if matrix[i,j]>.6 else "black")
        fig.colorbar(im,ax=ax,label="Minimum strong-member fraction across 5 folds",shrink=.8)
    fig.suptitle("Does the same physical meaning recur in every fold?\nStrong member: held-out |r| >= 0.8 and calibrated R2 >= 0.5; * requires >=70% in EVERY fold",fontsize=13)
    save(fig,"single-node-stability")


def channel_curves():
    fig,axes=plt.subplots(2,2,figsize=(11.5,8),constrained_layout=True)
    rows=[]
    # Fixed illustrative concepts; no search over outer target performance.
    chosen=["inverse_sqrt_mass","mean_radius"]
    for bi,bank in enumerate(["mm","mm_mv"]):
        for ci,concept in enumerate(chosen):
            ax=axes[bi,ci];index=CONCEPTS.index(concept)
            for suffix,color,label in [("modnet","#a1613b","MLP"),("kan","#176b88","KAN")]:
                arm=bank+"_"+suffix;curves=[]
                for fold in range(5):
                    directory=OUT/arm/f"fold-{fold}"
                    with np.load(directory/"member-00/arrays.npz") as a:train=a["q_train"][:,index]
                    with np.load(directory/"ensemble-arrays.npz") as a:
                        q=a["q_test"][:,index];c=a["late_output_shifts"][:,index]
                    edges=np.unique(np.quantile(train,np.linspace(.05,.95,10)))
                    x=[];y=[];counts=[]
                    for left,right in zip(edges[:-1],edges[1:]):
                        mask=(q>=left)&(q<right)
                        if mask.sum()>=4:
                            x.append(float(q[mask].mean()));y.append(float(c[mask].mean()));counts.append(int(mask.sum()))
                    curves.append(dict(x=x,y=y,counts=counts,fold=fold))
                low=max(c["x"][0] for c in curves);high=min(c["x"][-1] for c in curves)
                grid=np.linspace(low,high,100)
                values=np.array([np.interp(grid,c["x"],c["y"]) for c in curves])
                correlations=[float(np.corrcoef(a,b)[0,1]) for a,b in itertools.combinations(values,2) if np.std(a)>1e-8 and np.std(b)>1e-8]
                rho=float(np.median(correlations)) if correlations else None
                ax.plot(grid,values.mean(axis=0),color=color,lw=2,label=f"{label}: curve r={rho:.2f}")
                ax.fill_between(grid,values.mean(axis=0)-values.std(axis=0),values.mean(axis=0)+values.std(axis=0),color=color,alpha=.12)
                rows.append(dict(arm=arm,concept=concept,median_pairwise_curve_correlation=rho,fold_curves=curves))
            ax.axhline(0,lw=.6,color="gray");ax.grid(alpha=.15);ax.legend(fontsize=8)
            ax.set(xlabel="Inverse sqrt mean mass (u^-1/2)" if concept=="inverse_sqrt_mass" else "Mean covalent radius (pm)",
                ylabel="Selected late channel (cm^-1)",title="MM" if bank=="mm" else "MM + MatterVial")
    fig.suptitle("Selected channel response on held-out materials\nConditional bin means; bands are fold SD. Curves do not establish a pure physical contribution.",fontsize=13)
    save(fig,"aligned-channel-responses")
    return rows


def export_readouts(models):
    quantities=[]
    for model in [m for m in models if m["member"]==0]:
        directory=OUT/model["arm"]/f"fold-{model['fold']}"/"member-00"
        with np.load(directory/"arrays.npz") as archive:
            a={k:archive[k] for k in archive.files}
        for layer in ["first","late"]:
            weights=a[f"{layer}_ridge_weights"]
            beta=weights*a["q_train_std"][None,:]/a[f"{layer}_ridge_scale"][:,None]
            intercept=a["q_train_mean"]-a[f"{layer}_ridge_center"]@beta
            for i,concept in enumerate(CONCEPTS):
                r=next(x for x in model["records"] if x["layer"]==layer and x["concept"]==concept)
                quantities.append(dict(arm=model["arm"],fold=model["fold"],member=0,layer=layer,concept=concept,
                    model_path=model["model_path"],model_sha256=model["model_sha256"],
                    single_node=dict(node=r["node"],offset=r["offset"],slope=r["slope"],
                        formula=f"q_hat = {r['offset']:.12g} + ({r['slope']:.12g}) * h[{r['node']}]",
                        outer_r2=r["test_r2"],strongly_aligned=r["strongly_aligned"],concept_specific=r["concept_specific"]),
                    distributed_readout=dict(intercept=float(intercept[i]),coefficients=beta[:,i].tolist(),
                        formula="q_hat = intercept + sum_j coefficients[j] * h[j]",
                        outer_r2=r["ridge_test_r2"],n_terms=beta.shape[0]),
                    meaning="TRAIN-calibrated diagnostic readout in the descriptor's units; does not replace the network or identify a physical mechanism."))
    payload=dict(representative_rule="First saved member of every arm and fold; never selected by outer scores.",
        quantities=quantities,definition="h is the native hidden representation defined in protocol.json; both readouts are post-hoc probes.")
    (OUT/"readable-quantities.json").write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")


def geometry_plot(rows):
    quantities=read(OUT/"readable-quantities.json")["quantities"]
    fig,axes=plt.subplots(2,2,figsize=(11.4,8),constrained_layout=True)
    for j,(name,ci,label) in enumerate(zip(["mean_bond_length","volume_per_atom"],[3,5],
            [r"Mean bond length ($\AA$)",r"Volume per atom ($\AA^3$)"])):
        low=float("inf");high=-float("inf")
        for arm,color,title in [("mm_mv_modnet","#a66a42","MLP"),("mm_mv_kan","#147491","KAN")]:
            with np.load(OUT/arm/"fold-0/member-00/arrays.npz") as z:
                q=z["q_test"][:,ci];pred=z["first_single_decoded"][:,ci]
            c=next(c for c in quantities if c["arm"]==arm and c["fold"]==0 and c["layer"]=="first" and c["concept"]==name)["single_node"]
            axes[0,j].scatter(q,pred,s=17,color=color,alpha=.42,edgecolors="none",label=f"{title} h[{c['node']}], R2={c['outer_r2']:.3f}")
            low=min(low,float(q.min()),float(pred.min()));high=max(high,float(q.max()),float(pred.max()))
        axes[0,j].plot([low,high],[low,high],ls="--",lw=1,color="#697983")
        axes[0,j].set(xlabel="Observed "+label,ylabel="Train-calibrated node readout",title="Fixed example: outer fold 0, member 0")
        axes[0,j].legend(fontsize=8);axes[0,j].grid(alpha=.13)
        for arm,color,offset,title in [("mm_mv_modnet","#a66a42",-.18,"MLP"),("mm_mv_kan","#147491",.18,"KAN")]:
            row=next(r for r in rows if r["arm"]==arm and r["layer"]=="first" and r["concept"]==name)
            vals=[f["strongly_aligned_fraction"] for f in row["folds"]]
            axes[1,j].bar(np.arange(5)+offset,vals,width=.35,color=color,label=title)
            for i,v in enumerate(vals):axes[1,j].text(i+offset,v+.015,f"{v:.0%}",ha="center",fontsize=8)
        axes[1,j].axhline(.7,color="#777",ls="--",lw=.8);axes[1,j].set_xticks(range(5),range(1,6))
        axes[1,j].set(xlabel="Outer fold",ylabel="Strongly aligned members / 50",ylim=(0,1.13),title=label)
        axes[1,j].legend(fontsize=8,loc="lower left")
    fig.suptitle("Geometry-related native channels in the high-accuracy MM + MatterVial models\nAll node choices and calibration use training data; displayed R2 uses outer-held-out materials",fontsize=13)
    save(fig,"geometry-native-evidence")


def main():
    analysis=read(OUT/"analysis.json")
    assert analysis["complete"] and not analysis["smoke"] and analysis["models"]==1000
    rows,models,comparisons,hashes=aggregate(analysis)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
    heatmaps(rows);stable_map(rows);curves=channel_curves()
    export_readouts(models)
    geometry_plot(rows)
    stable=[{k:r[k] for k in ["arm","layer","concept","min_strong_fraction","min_specific_fraction"]} for r in rows if r["stable_single_node"]]
    specific=[{k:r[k] for k in ["arm","layer","concept","min_specific_fraction"]} for r in rows if r["stable_specific_single_node"]]
    result=dict(complete=True,models=1000,rows=rows,comparisons=comparisons,
        stable_single_node_candidates=stable,stable_specific_single_node_candidates=specific,
        aligned_channel_curves=curves,source_summary_sha256=hashes,
        protocol_sha256=sha(OUT/"protocol.json"),analysis_sha256=sha(OUT/"analysis.json"),
        max_prediction_replay_error=analysis["max_prediction_replay_error"],
        max_contribution_reconstruction_error=analysis["max_contribution_reconstruction_error"],
        source_sha256=sha(__file__))
    (OUT/"interpretation.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    (OUT/"audit.json").write_text(json.dumps(dict(passed=True,models=1000,original_fivefold_mae_reproduced=True,
        all_array_hashes_verified=True,protocol_sha256=sha(OUT/"protocol.json"),
        interpretation_sha256=sha(OUT/"interpretation.json")),indent=2),encoding="utf-8")
    print(json.dumps(dict(stable_single_node_candidates=stable,stable_specific_single_node_candidates=specific)))
    for arm in ARMS:
        for layer in ["first","late"]:
            print(arm,layer)
            for r in [r for r in rows if r["arm"]==arm and r["layer"]==layer]:
                m=r["mean"]
                print(r["concept"],"single",round(m["single_member_test_r2_median"],3),"minstrong",r["min_strong_fraction"],
                    "distributed",round(m["ensemble_multi_decoder_r2"],3),"deltaMAE",round(m["ablation_mae_delta"],3))


if __name__=="__main__":main()
