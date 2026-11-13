"""Audit and compare robust physical readouts, retaining failed original probes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summarize_formal_phonons_concepts import ROOT,ARMS,LABELS,CONCEPTS,SHORT

OUT=ROOT/"benchmarks/formal-phonons-readout-robustness-20261106"
PARENT=ROOT/"benchmarks/formal-phonons-concepts-20261106"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    analysis=read(OUT/"analysis.json");assert analysis["complete"] and analysis["models"]==1000
    parent=read(PARENT/"interpretation.json");rows=[];hashes={};max_replay=0.
    for arm in ARMS:
        for fold in range(5):
            for i in range(50):
                folder=OUT/arm/f"fold-{fold}"/f"member-{i:02d}";s=read(folder/"summary.json")
                assert sha(folder/"readouts.npz")==s["arrays_sha256"]
                assert s["protocol_fingerprint"]==analysis["protocol_fingerprint"]
                hashes[str(folder.relative_to(ROOT)/"summary.json")]=sha(folder/"summary.json")
                max_replay=max(max_replay,s["max_prediction_replay_error"])
        for layer in ["first","late"]:
            for concept in CONCEPTS:
                group=[dict(next(r for r in f["rows"] if r["layer"]==layer and r["concept"]==concept),fold=f["fold"])
                    for f in analysis["folds"] if f["arm"]==arm]
                assert len(group)==5
                original=next(r for r in parent["rows"] if r["arm"]==arm and r["layer"]==layer and r["concept"]==concept)
                rows.append(dict(arm=arm,layer=layer,concept=concept,folds=group,
                    global_scale_mean_r2=float(np.mean([r["global_scale_ensemble_r2"] for r in group])),
                    global_scale_min_r2=min(r["global_scale_ensemble_r2"] for r in group),
                    rank4_mean_r2=float(np.mean([r["rank4_ensemble_r2"] for r in group])),
                    rank4_min_r2=min(r["rank4_ensemble_r2"] for r in group),
                    mean_rmse_physical_units=float(np.mean([r["rmse_physical_units"] for r in group])),
                    original_unit_standardized_mean_r2=original["mean"]["ensemble_multi_decoder_r2"],
                    prediction_only_mean_r2=original["mean"]["prediction_only_ensemble_r2"],
                    mean_hidden_width=original["mean_representation_width"],
                    mean_input_features=original["mean_input_features"]))
    comparisons=[]
    for bank in ["mm","mm_mv"]:
        for layer in ["first","late"]:
            for concept in CONCEPTS:
                k=next(r for r in rows if r["arm"]==bank+"_kan" and r["layer"]==layer and r["concept"]==concept)
                m=next(r for r in rows if r["arm"]==bank+"_modnet" and r["layer"]==layer and r["concept"]==concept)
                comparisons.append(dict(bank=bank,layer=layer,concept=concept,
                    global_scale_kan_minus_mlp=k["global_scale_mean_r2"]-m["global_scale_mean_r2"],
                    global_scale_kan_fold_wins=sum(x["global_scale_ensemble_r2"]>y["global_scale_ensemble_r2"] for x,y in zip(k["folds"],m["folds"])),
                    rank4_kan_minus_mlp=k["rank4_mean_r2"]-m["rank4_mean_r2"],
                    rank4_kan_fold_wins=sum(x["rank4_ensemble_r2"]>y["rank4_ensemble_r2"] for x,y in zip(k["folds"],m["folds"]))))
    fig,axes=plt.subplots(2,2,figsize=(17,9.5),constrained_layout=True)
    for li,layer in enumerate(["first","late"]):
        for pi,(key,title) in enumerate([("global_scale_mean_r2","Full representation"),("rank4_mean_r2","Four training principal components")]):
            matrix=np.array([[next(r for r in rows if r["arm"]==arm and r["layer"]==layer and r["concept"]==c)[key] for c in CONCEPTS] for arm in ARMS])
            ax=axes[li,pi];im=ax.imshow(matrix,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
            ax.set_yticks(range(4),[LABELS[a] for a in ARMS]);ax.set_xticks(range(12),SHORT,rotation=60,ha="right",fontsize=8)
            ax.set_title(layer.capitalize()+" | "+title,fontsize=11)
            for i in range(4):
                for j in range(12):ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if matrix[i,j]>.6 else "black")
            fig.colorbar(im,ax=ax,label="Mean outer-fold physical-readout R2",shrink=.75)
    fig.suptitle("Physical information retained by formal phonons models\nOne training RMS scale per layer; fixed ridge penalty; frozen KAN / MLP weights",fontsize=14)
    for ext in ["png","pdf"]:fig.savefig(OUT/f"robust-readout-comparison.{ext}",dpi=170,bbox_inches="tight")
    plt.close(fig)
    payload=dict(complete=True,models=1000,rows=rows,comparisons=comparisons,
        parent_interpretation_sha256=sha(PARENT/"interpretation.json"),source_summary_sha256=hashes,
        max_prediction_replay_error=max_replay,protocol_sha256=sha(OUT/"protocol.json"),source_sha256=sha(__file__))
    (OUT/"interpretation.json").write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    (OUT/"audit.json").write_text(json.dumps(dict(passed=True,models=1000,all_array_hashes_verified=True,
        max_prediction_replay_error=max_replay,interpretation_sha256=sha(OUT/"interpretation.json")),indent=2),encoding="utf-8")
    for r in rows:
        if r["concept"] in ["mean_mass","mean_radius","mean_bond_length","volume_per_atom","inverse_sqrt_mass","inverse_cuberoot_volume"]:
            print(r["arm"],r["layer"],r["concept"],"full",round(r["global_scale_mean_r2"],3),"rank4",round(r["rank4_mean_r2"],3),"min",round(r["global_scale_min_r2"],3),"RMSE",round(r["mean_rmse_physical_units"],3))


if __name__=="__main__":main()
