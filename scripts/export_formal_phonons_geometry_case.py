"""Export actual first-layer weights behind fixed formal geometry examples."""
from pathlib import Path
import pickle
import analyze_formal_phonons_concepts as a
from run_modnet_official_matched import np,tf,MODData,write_json,file_hash
from physical_feature_modules import cubic_basis

OUT=a.ROOT/"benchmarks/formal-phonons-concepts-20261106"


def main():
    results=[]
    for arm in ["mm_mv_kan","mm_mv_modnet"]:
        source=a.RUN/arm/"fold-0";selection=a.engine.read(source/"frozen-selection.json");job=selection["members"][0]
        path=Path(job["directory"])/"model.pkl";assert file_hash(path)==selection["model_sha256"][job["directory"]]
        summary=a.engine.read(OUT/arm/"fold-0/member-00/summary.json")
        with np.load(OUT/arm/"fold-0/member-00/arrays.npz") as z:stored={k:z[k] for k in z.files}
        data=MODData.load(job["training_cache"])
        with path.open("rb") as f:model=pickle.load(f)
        model._restore_model();cols=list(model.optimal_descriptors[:model.n_feat])
        raw=data.df_featurized.loc[job["train_ids"],cols].values;x=a.scaled(model,raw)
        layers=[l for l in model.model.layers if not isinstance(l,tf.keras.layers.InputLayer)];first=layers[0]
        actual=first(tf.convert_to_tensor(x),training=False).numpy()
        for concept in ["mean_bond_length","volume_per_atom","inverse_cuberoot_volume"]:
            r=next(r for r in summary["records"] if r["layer"]=="first" and r["concept"]==concept);node=r["node"]
            if arm.endswith("_kan"):
                coords=.5*np.tanh(x) if first.hidden_coordinates else x
                coef=first.coefficients.numpy()[:,node,:]
                contributions=np.einsum("nik,ik->ni",cubic_basis(tf.convert_to_tensor(coords)).numpy(),coef)
                bias=float(first.bias.numpy()[node]);replayed=contributions.sum(axis=1)+bias
                weights=coef.tolist();activation="identity"
            else:
                weights_array,bias_array=first.get_weights();weights=weights_array[:,node].tolist();bias=float(bias_array[node])
                contributions=x*weights_array[:,node]
                replayed=first.activation(tf.convert_to_tensor(contributions.sum(axis=1)+bias)).numpy()
                activation=first.activation.__name__
            error=float(np.max(abs(replayed-actual[:,node])));assert error<.0001
            ci=a.NAMES.index(concept)
            np.testing.assert_allclose(actual[:,node],stored["first_selected_activation_train"][:,ci],atol=.0001,rtol=0)
            center=contributions.mean(axis=0);magnitude=np.mean(abs(contributions-center),axis=0);order=np.argsort(-magnitude)
            groups={"explicit_10_physical_descriptors":0.,"other_matminer_descriptors":0.,"mattervial_embeddings":0.}
            raw_physical={x[0] for x in a.RAW_CONCEPTS.values()}
            for c,value in zip(cols,magnitude):
                group="explicit_10_physical_descriptors" if c in raw_physical else "mattervial_embeddings" if c.startswith(("MEGNet_OFMEncoded_v1_","MVL16_","MVL32_","ROOST_mpgap_","ORB_v3_")) else "other_matminer_descriptors"
                groups[group]+=float(value)
            scale={k:np.asarray(getattr(model._scaler,k)).tolist() for k in ["scale_","min_","mean_","var_"] if hasattr(model._scaler,k)}
            result=dict(arm=arm,fold=0,member=0,concept=concept,node=node,layer_name=first.name,
                model_path=str(path),model_sha256=file_hash(path),features=cols,weights=weights,bias=bias,
                activation=activation,scaler=dict(kind=type(model._scaler).__name__,**scale),
                bounded_coordinates=bool(getattr(first,"hidden_coordinates",False)),
                spline_knots=np.r_[[-.5]*3,np.linspace(-.5,.5,6),[.5]*3].tolist() if arm.endswith("_kan") else None,
                train_mean_preactivation_edge_contributions=center.tolist(),
                input_group_absolute_contribution_fraction={k:v/float(magnitude.sum()) for k,v in groups.items()},
                top_input_edges=[dict(feature=cols[i],mean_abs_centered_contribution=float(magnitude[i])) for i in order[:12]],
                affine_physical_readout=dict(offset=r["offset"],slope=r["slope"],outer_r2=r["test_r2"],
                    partial_r_given_prediction=r["partial_test_correlation_given_prediction"],
                    partial_r_given_other_physics=r["partial_test_correlation_given_other_physics"]),
                weight_replay_max_error=error,
                caveat="Input-group fractions are sums of absolute centered preactivation edge magnitudes, not variance explained or causal shares.")
            results.append(result)
    write_json(OUT/"geometry-native-weights.json",dict(complete=True,selection="fixed fold 0, member 0; same models as protocol",cases=results))
    for r in results:print(r["arm"],r["concept"],r["node"],r["affine_physical_readout"],r["input_group_absolute_contribution_fraction"],flush=True)


if __name__=="__main__":main()
