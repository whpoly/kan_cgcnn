"""Physical-concept audit of every frozen formal phonons KAN and MLP member.

Training descriptors select/calibrate probes; outer-test descriptors evaluate
them. Network weights never change. Outer target labels are only used after
all choices are frozen to report the effect of mean-replacing aligned channels.
"""
from pathlib import Path
import argparse
import gc
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import run_native_fourway as engine
from run_modnet_official_matched import np, tf, file_hash, write_json, digest
from standalone_kan_models import SplineEdgeLayer
from physical_feature_modules import cubic_basis

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "benchmarks/native-fourway-scaled-kan-20261029"
RAW_CONCEPTS = {
    "mean_mass": ("ElementProperty|MagpieData mean AtomicWeight", "mass", "u"),
    "mean_radius": ("ElementProperty|MagpieData mean CovalentRadius", "size", "pm"),
    "radius_deviation": ("ElementProperty|MagpieData avg_dev CovalentRadius", "size", "pm"),
    "mean_bond_length": ("AverageBondLength|mean Average bond length", "geometry", "angstrom"),
    "bond_length_deviation": ("AverageBondLength|std_dev Average bond length", "geometry", "angstrom"),
    "volume_per_atom": ("DensityFeatures|vpa", "volume", "angstrom^3"),
    "mean_electronegativity": ("ElementProperty|MagpieData mean Electronegativity", "electronic", "Pauling"),
    "mean_en_difference": ("ElectronegativityDiff|mean EN difference", "electronic", "Pauling"),
    "coordination": ("CoordinationNumber|mean CN_VoronoiNN", "coordination", "dimensionless"),
    "density": ("DensityFeatures|density", "density", "g/cm^3"),
}
NAMES = list(RAW_CONCEPTS) + ["inverse_sqrt_mass", "inverse_cuberoot_volume"]
FAMILIES = [x[1] for x in RAW_CONCEPTS.values()] + ["mass", "volume"]


def concepts(frame):
    q = frame[[x[0] for x in RAW_CONCEPTS.values()]].values.astype(float)
    assert np.isfinite(q).all()
    assert np.all(q[:,0]>0) and np.all(q[:,5]>0)
    return np.column_stack([q, 1/np.sqrt(q[:,0]), q[:,5]**(-1/3)])


def corr_matrix(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    aa = a-a.mean(axis=0); bb = b-b.mean(axis=0)
    denom = np.sqrt(np.sum(aa*aa,axis=0)[:,None]*np.sum(bb*bb,axis=0)[None,:])
    return np.divide(aa.T@bb,denom,out=np.zeros(denom.shape),where=denom>1e-15)


def paired_corr(a,b):
    return np.diag(corr_matrix(a,b))


def r2(truth, prediction):
    denominator = np.sum((truth-truth.mean(axis=0))**2,axis=0)
    return 1-np.sum((truth-prediction)**2,axis=0)/np.maximum(denominator,1e-12)


def single_probe(h, ht, q, qt):
    correlation = corr_matrix(h,q)
    selected = np.argmax(abs(correlation),axis=0)
    chosen = np.asarray(h[:,selected],dtype=float)
    chosen_test = np.asarray(ht[:,selected],dtype=float)
    dh = chosen-chosen.mean(axis=0)
    slopes = np.sum(dh*(q-q.mean(axis=0)),axis=0)/np.maximum(np.sum(dh*dh,axis=0),1e-15)
    offsets = q.mean(axis=0)-slopes*chosen.mean(axis=0)
    decoded = chosen_test*slopes+offsets
    return dict(selected=selected, train_correlation=correlation[selected,np.arange(len(NAMES))],
        test_correlation=paired_corr(chosen_test,qt), test_r2=r2(qt,decoded),
        slopes=slopes, offsets=offsets, decoded=decoded, chosen_train=chosen,
        chosen_test=chosen_test, correlations=correlation)


def ridge_readout(h, ht, q, qt):
    # Fixed penalty on standardized features/targets, no hyperparameter search.
    center=h.mean(axis=0); scale=np.maximum(h.std(axis=0),1e-8)
    a=(h-center)/scale; at=(ht-center)/scale
    qm=q.mean(axis=0); qs=np.maximum(q.std(axis=0),1e-8)
    weights=np.linalg.solve(a.T@a/len(a)+.01*np.eye(a.shape[1]),a.T@((q-qm)/qs)/len(a))
    decoded=(at@weights)*qs+qm
    return dict(test_r2=r2(qt,decoded),decoded=decoded,weights=weights,
                center=center,scale=scale,target_mean=qm,target_std=qs)


def residual_control(train_h,test_h,q,qt,train_controls,test_controls):
    mean=train_controls.mean(axis=0); sd=np.maximum(train_controls.std(axis=0),1e-8)
    a=np.column_stack([np.ones(len(q)),(train_controls-mean)/sd])
    at=np.column_stack([np.ones(len(qt)),(test_controls-mean)/sd])
    coeff=np.linalg.lstsq(a,np.column_stack([train_h,q]),rcond=None)[0]
    residual=np.column_stack([test_h,qt])-at@coeff
    c=q.shape[1]
    return paired_corr(residual[:,:c],residual[:,c:])


def scaled(model, raw):
    return np.nan_to_num(model._scaler.transform(np.nan_to_num(raw)),nan=-1).astype("float32")


def forward(model, x, kan):
    layers=[l for l in model.model.layers if not isinstance(l,tf.keras.layers.InputLayer)]
    if kan:
        spline_indices=[i for i,l in enumerate(layers) if isinstance(l,SplineEdgeLayer)]
        assert len(spline_indices)==3
        first_index=spline_indices[0]; late_index=spline_indices[1]
    else:
        dense_indices=[i for i,l in enumerate(layers) if isinstance(l,tf.keras.layers.Dense)]
        assert len(dense_indices)==len(layers)
        nonlinear=[i for i in dense_indices[:-1] if layers[i].activation.__name__!="linear"]
        assert nonlinear
        first_index=dense_indices[0]; late_index=nonlinear[-1]
    values=tf.convert_to_tensor(x)
    representations={}
    for i,layer in enumerate(layers):
        values=layer(values,training=False)
        if i==first_index: representations["first"]=values.numpy()
        if i==late_index: representations["late"]=values.numpy()
    prediction=np.asarray(values).reshape(-1).astype(float)
    h=representations["late"]
    if kan:
        layer=layers[spline_indices[-1]]
        coordinate=.5*np.tanh(h) if layer.hidden_coordinates else h
        edges=np.einsum("nik,iok->nio",cubic_basis(tf.convert_to_tensor(coordinate)).numpy(),layer.coefficients.numpy())[:,:,0]
        terms=edges.astype(float)*model.target_scaling["std"]
        baseline=model.target_scaling["mean"]+float(layer.bias.numpy()[0])*model.target_scaling["std"]
    else:
        effective=np.eye(h.shape[1]); bias=np.zeros(h.shape[1])
        for layer in layers[late_index+1:]:
            assert isinstance(layer,tf.keras.layers.Dense) and layer.activation.__name__=="linear"
            weight,offset=layer.get_weights()
            bias=bias@weight+offset; effective=effective@weight
        assert effective.shape[1]==1
        terms=h.astype(float)*effective[:,0]
        baseline=float(bias[0])
    error=float(np.max(abs(prediction-(baseline+terms.sum(axis=1)))))
    np.testing.assert_allclose(prediction,baseline+terms.sum(axis=1),atol=.006,rtol=0)
    return representations,prediction,terms,layers,first_index,error


def first_node_shifts(layers,first_index,h,train_mean,selected,original):
    unique=np.unique(selected)
    perturbed=np.tile(h[None,:,:],(len(unique),1,1))
    for n,node in enumerate(unique):perturbed[n,:,node]=train_mean[node]
    flat=perturbed.reshape(-1,h.shape[1]); outputs=[]
    for start in range(0,len(flat),512):
        values=tf.convert_to_tensor(flat[start:start+512])
        for layer in layers[first_index+1:]:values=layer(values,training=False)
        outputs.append(np.asarray(values).reshape(-1))
    ablated=np.concatenate(outputs).reshape(len(unique),len(h))
    return np.column_stack([original-ablated[np.where(unique==node)[0][0]] for node in selected])


def inspect_member(out,arm,fold,index,job,selection,frame,qframe,split,reference,protocol):
    destination=out/arm/f"fold-{fold}"/f"member-{index:02d}"
    summary_path=destination/"summary.json"
    if summary_path.exists():
        saved=engine.read(summary_path)
        assert saved["protocol_fingerprint"]==digest(protocol)
        assert file_hash(destination/"arrays.npz")==saved["arrays_sha256"]
        return saved
    path=Path(job["directory"])/"model.pkl"
    assert file_hash(path)==selection["model_sha256"][job["directory"]]
    tr_ids=job["train_ids"];te_ids=split["test_ids"]
    assert set(tr_ids)<=set(split["train_ids"]) and not set(tr_ids)&set(te_ids)
    with path.open("rb") as f:model=pickle.load(f)
    model._restore_model()
    cols=list(model.optimal_descriptors[:model.n_feat]);kan=arm.endswith("_kan")
    x=scaled(model,frame.loc[tr_ids,cols].values);xt=scaled(model,frame.loc[te_ids,cols].values)
    q=qframe.loc[tr_ids].values;qt=qframe.loc[te_ids].values
    train,pred,terms,layers,first_index,error_train=forward(model,x,kan)
    test,pred_test,test_terms,_,_,error_test=forward(model,xt,kan)
    replay=float(np.max(abs(pred_test-reference)))
    assert replay<.002,(arm,fold,index,replay)
    centered_terms=test_terms-terms.mean(axis=0)
    # A prediction-only decoder distinguishes a concept-specific representation
    # from mere encoding of the scalar output's correlation with that concept.
    predictor=single_probe(pred[:,None],pred_test[:,None],q,qt)
    arrays={"test_ids":np.asarray(te_ids),"q_test":qt,"prediction":pred_test,
            "prediction_only_decoded":predictor["decoded"]}
    records=[]
    for layer_name in ["first","late"]:
        h=train[layer_name].astype(float);ht=test[layer_name].astype(float)
        single=single_probe(h,ht,q,qt)
        null_q=q[np.random.RandomState(20261106+fold*1000+index).permutation(len(q))]
        null=single_probe(h,ht,null_q,qt)
        multi=ridge_readout(h,ht,q,qt)
        partial_prediction=residual_control(single["chosen_train"],single["chosen_test"],q,qt,pred[:,None],pred_test[:,None])
        partial_other=[]
        for c in range(len(NAMES)):
            others=[j for j in range(len(RAW_CONCEPTS)) if FAMILIES[j]!=FAMILIES[c]]
            partial_other.append(float(residual_control(single["chosen_train"][:,c:c+1],single["chosen_test"][:,c:c+1],
                q[:,c:c+1],qt[:,c:c+1],q[:,others],qt[:,others])[0]))
        if layer_name=="first":
            shifts=first_node_shifts(layers,first_index,test["first"],train["first"].mean(axis=0),single["selected"],pred_test)
        else:
            shifts=centered_terms[:,single["selected"]]
        arrays[f"{layer_name}_single_decoded"]=single["decoded"]
        arrays[f"{layer_name}_multi_decoded"]=multi["decoded"]
        arrays[f"{layer_name}_selected_activation_test"]=single["chosen_test"]
        arrays[f"{layer_name}_selected_activation_train"]=single["chosen_train"]
        arrays[f"{layer_name}_output_shifts"]=shifts
        arrays[f"{layer_name}_ridge_weights"]=multi["weights"]
        arrays[f"{layer_name}_ridge_center"]=multi["center"]
        arrays[f"{layer_name}_ridge_scale"]=multi["scale"]
        arrays[f"{layer_name}_training_correlations"]=single["correlations"]
        for c,name in enumerate(NAMES):
            node=int(single["selected"][c]);rho=float(single["test_correlation"][c]);score=float(single["test_r2"][c])
            aligned=abs(rho)>=.8 and score>=.5
            records.append(dict(layer=layer_name,concept=name,node=node,width=h.shape[1],
                train_correlation=float(single["train_correlation"][c]),test_correlation=rho,
                test_r2=score,slope=float(single["slopes"][c]),offset=float(single["offsets"][c]),
                ridge_test_r2=float(multi["test_r2"][c]),permutation_null_test_r2=float(null["test_r2"][c]),
                prediction_only_test_r2=float(predictor["test_r2"][c]),
                partial_test_correlation_given_prediction=float(partial_prediction[c]),
                partial_test_correlation_given_other_physics=partial_other[c],
                mean_abs_output_shift=float(np.mean(abs(shifts[:,c]))),
                output_shift_over_prediction_sd=float(np.mean(abs(shifts[:,c]))/max(np.std(pred),1e-8)),
                strongly_aligned=bool(aligned),
                concept_specific=bool(aligned and abs(partial_prediction[c])>=.5 and abs(partial_other[c])>=.5)))
    arrays["q_train"]=q
    arrays["q_train_mean"]=q.mean(axis=0);arrays["q_train_std"]=q.std(axis=0)
    destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/"arrays.npz",**arrays)
    result=dict(arm=arm,fold=fold,member=index,model_path=str(path),model_sha256=file_hash(path),
        protocol_fingerprint=digest(protocol),n_train=len(tr_ids),n_test=len(te_ids),n_features=len(cols),
        max_prediction_replay_error=replay,max_contribution_reconstruction_error=max(error_train,error_test),
        selected_descriptor_names=cols,records=records,arrays_sha256=file_hash(destination/"arrays.npz"))
    write_json(summary_path,result)
    del model,train,test,terms,test_terms,layers
    tf.keras.backend.clear_session();gc.collect()
    return result


def summarize_fold(out,arm,fold,records,split,targets,reference):
    directory=out/arm/f"fold-{fold}"
    target=targets.loc[split["test_ids"]].values.reshape(-1)
    original=reference.mean(axis=0).astype(float)
    mae=float(np.mean(abs(original-target)))
    rows=[];saved={"sample_ids":np.asarray(split["test_ids"]),"target":target,"prediction":original}
    loaded=[]
    for i in range(len(records)):
        with np.load(directory/f"member-{i:02d}"/"arrays.npz") as a:
            loaded.append({k:a[k] for k in a.files if "decoded" in k or "shifts" in k or k in ["q_test","q_train_mean","q_train_std"]})
    q=loaded[0]["q_test"]
    saved["q_test"]=q
    for layer in ["first","late"]:
        single=np.mean([a[f"{layer}_single_decoded"] for a in loaded],axis=0)
        multi=np.mean([a[f"{layer}_multi_decoded"] for a in loaded],axis=0)
        shifts=np.mean([a[f"{layer}_output_shifts"] for a in loaded],axis=0)
        saved[f"{layer}_single_decoded"]=single
        saved[f"{layer}_multi_decoded"]=multi
        saved[f"{layer}_output_shifts"]=shifts
        single_r2=r2(q,single);multi_r2=r2(q,multi)
        for c,name in enumerate(NAMES):
            entries=[next(x for x in r["records"] if x["layer"]==layer and x["concept"]==name) for r in records]
            rows.append(dict(arm=arm,fold=fold,layer=layer,concept=name,members=len(entries),
                single_member_test_r2_median=float(np.median([x["test_r2"] for x in entries])),
                single_member_test_abs_correlation_median=float(np.median([abs(x["test_correlation"]) for x in entries])),
                strongly_aligned_fraction=float(np.mean([x["strongly_aligned"] for x in entries])),
                concept_specific_fraction=float(np.mean([x["concept_specific"] for x in entries])),
                ensemble_single_decoder_r2=float(single_r2[c]),ensemble_multi_decoder_r2=float(multi_r2[c]),
                ridge_member_test_r2_median=float(np.median([x["ridge_test_r2"] for x in entries])),
                null_test_r2_median=float(np.median([x["permutation_null_test_r2"] for x in entries])),
                partial_given_prediction_abs_median=float(np.median([abs(x["partial_test_correlation_given_prediction"]) for x in entries])),
                partial_given_other_physics_abs_median=float(np.median([abs(x["partial_test_correlation_given_other_physics"]) for x in entries])),
                mean_member_abs_output_shift=float(np.mean([x["mean_abs_output_shift"] for x in entries])),
                ensemble_abs_output_shift=float(np.mean(abs(shifts[:,c]))),
                baseline_mae=mae,ablated_mae=float(np.mean(abs(original-shifts[:,c]-target))),
                ablation_mae_delta=float(np.mean(abs(original-shifts[:,c]-target)))-mae))
    np.savez_compressed(directory/"ensemble-arrays.npz",**saved)
    result=dict(arm=arm,fold=fold,members=len(records),rows=rows,baseline_mae=mae,
        max_replay_error=max(r["max_prediction_replay_error"] for r in records),
        max_decomposition_error=max(r["max_contribution_reconstruction_error"] for r in records),
        ensemble_arrays_sha256=file_hash(directory/"ensemble-arrays.npz"))
    write_json(directory/"fold-summary.json",result)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=ROOT/"benchmarks/formal-phonons-concepts-20261106")
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--arm",choices=engine.ARMS)
    ap.add_argument("--workers",type=int,default=1)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    p=dict(version=1,dataset="matbench_phonons",run=str(RUN),smoke=args.smoke,
        arms=list(engine.ARMS),folds=[0] if args.smoke else list(range(5)),members_per_fold=2 if args.smoke else 50,
        source_sha256=file_hash(__file__),benchmark_audit_sha256=file_hash(RUN/"artifact-audit.json"),
        concepts=RAW_CONCEPTS,derived_concepts={"inverse_sqrt_mass":"1/sqrt(mean_mass), inertia proxy, not modal effective mass",
            "inverse_cuberoot_volume":"volume_per_atom**(-1/3), inverse geometric length proxy"},
        selection="Per member and layer: maximum absolute TRAIN Pearson correlation, then TRAIN affine calibration to the physical quantity.",
        evaluation="Outer held-out descriptors; all chosen nodes, slopes and offsets frozen before reporting outer statistics.",
        multivariate_readout="Ridge on standardized hidden units and concepts, objective mean_squared_error + 0.01*sum_squared_weights; no tuning.",
        layer_definition="First layer output; late KAN second spline output before final edge; late MLP last nonlinear hidden output, collapsing subsequent linear layers.",
        ablation="First: replace chosen activation by TRAIN mean. Late: remove TRAIN-centered exact output contribution. One node per member per concept, chosen only on TRAIN.",
        stability_rule="Strong member alignment: outer |Pearson r|>=0.8 and calibrated R2>=0.5. Stable candidate: >=70% strong members in EACH of five folds.",
        specificity_rule="Additionally require outer partial |r|>=0.5 after separate TRAIN-fitted removal of model prediction and of other-family raw physical descriptors.",
        permutation_control="One deterministic permutation of TRAIN concept values; no significance claim.",
        limitations=["Post-hoc investigation of previously evaluated benchmarks, not a new confirmatory experiment.",
            "Fifty ensemble members are ten selected candidates times five inner fits, not fifty independent replications.",
            "Correlated descriptors, model width and separately selected inputs limit architectural causal attribution.",
            "A linear readout is a derived diagnostic quantity, not an existing single neuron or a physical law.",
            "Node mean replacement can be off-manifold; its target-error change is model dependence, not physical intervention.",
            "No force constants, mode eigenvectors or dynamical matrices are supplied, so effective-mass or stiffness identification cannot be established."])
    p=json.loads(json.dumps(p))  # Normalize tuples to their on-disk JSON form.
    if (out/"protocol.json").exists():assert engine.read(out/"protocol.json")==p
    else:write_json(out/"protocol.json",p)
    if args.workers>1 and not args.arm:
        started=time.time()
        def launch(arm):
            command=[sys.executable,"-u",str(Path(__file__).resolve()),"--output",str(out),"--arm",arm]
            if args.smoke:command.append("--smoke")
            with (out/f"{arm}.log").open("w",encoding="utf-8") as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            return arm
        finished=[]
        with ThreadPoolExecutor(max_workers=min(args.workers,4)) as pool:
            futures=[pool.submit(launch,arm) for arm in engine.ARMS]
            for future in as_completed(futures):
                finished.append(future.result())
                write_json(out/"status.json",dict(state="analyzing",completed_arms=finished,seconds=time.time()-started,pid=os.getpid()))
                print(json.dumps(dict(completed_arms=finished,seconds=time.time()-started)),flush=True)
        parts=[engine.read(out/f"analysis-{arm}.json") for arm in engine.ARMS]
        final=dict(complete=not args.smoke,smoke=args.smoke,models=sum(x["models"] for x in parts),
            folds=[f for part in parts for f in part["folds"]],protocol_fingerprint=digest(p),
            seconds=time.time()-started,max_prediction_replay_error=max(x["max_prediction_replay_error"] for x in parts),
            max_contribution_reconstruction_error=max(x["max_contribution_reconstruction_error"] for x in parts))
        write_json(out/"analysis.json",final)
        write_json(out/"status.json",dict(state="complete",completed_members=final["models"],seconds=time.time()-started,pid=os.getpid()))
        print(json.dumps({k:v for k,v in final.items() if k!="folds"}),flush=True)
        return
    data_out=out/"data-checks"/(args.arm or "all")
    data_out.mkdir(parents=True,exist_ok=True)
    banks,targets=engine.load_banks(data_out)
    q=concepts(banks["mm"])
    import pandas as pd
    qframe=pd.DataFrame(q,index=banks["mm"].index,columns=NAMES)
    if not (out/"concept-definition.json").exists():
        definition=out/f"concept-definition-{args.arm or 'all'}.json"
        write_json(definition,dict(names=NAMES,families=FAMILIES))
    folds=[];total=0;started=time.time()
    for arm in ([args.arm] if args.arm else engine.ARMS):
        for fold in p["folds"]:
            d=RUN/arm/f"fold-{fold}";selection=engine.read(d/"frozen-selection.json");complete=engine.read(d/"complete.json")
            assert file_hash(d/"frozen-selection.json")==complete["selection_sha256"]
            assert file_hash(d/"outer-predictions.npz")==complete["prediction_sha256"]
            split=engine.read(engine.OLD/"matbench_phonons"/f"fold-{fold}"/"outer-split.json")
            with np.load(d/"outer-predictions.npz") as a:
                assert a["sample_ids"].tolist()==[f"mb-phonons-{int(s[2:])+1:04d}" for s in split["test_ids"]]
                reference=a["members"][:p["members_per_fold"]].copy()
                np.testing.assert_allclose(a["target"],targets.loc[split["test_ids"]].values.reshape(-1),atol=1e-8,rtol=0)
            rows=[]
            for i,job in enumerate(selection["members"][:p["members_per_fold"]]):
                rows.append(inspect_member(out,arm,fold,i,job,selection,banks[arm.rsplit("_",1)[0]],qframe,split,reference[i],p))
                total+=1
                status=dict(state="analyzing",completed_members=total,arm=arm,fold=fold,member=i,seconds=time.time()-started,pid=os.getpid())
                write_json(out/(f"status-{args.arm}.json" if args.arm else "status.json"),status)
                if total%10==0 or args.smoke: print(json.dumps(status),flush=True)
            folds.append(summarize_fold(out,arm,fold,rows,split,targets,reference))
    final=dict(complete=not args.smoke,smoke=args.smoke,models=total,folds=folds,
        protocol_fingerprint=digest(p),seconds=time.time()-started,
        max_prediction_replay_error=max(x["max_replay_error"] for x in folds),
        max_contribution_reconstruction_error=max(x["max_decomposition_error"] for x in folds))
    write_json(out/(f"analysis-{args.arm}.json" if args.arm else "analysis.json"),final)
    write_json(out/(f"status-{args.arm}.json" if args.arm else "status.json"),dict(state="complete",completed_members=total,seconds=time.time()-started,pid=os.getpid()))
    print(json.dumps({k:v for k,v in final.items() if k!="folds"}),flush=True)


if __name__=="__main__":
    main()
