"""Robustness audit: avoid amplifying nearly constant hidden coordinates.

The initial unit-standardized probe is preserved. This additional fixed probe
uses one RMS scale per layer and a separate four-PC capacity control.
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
from concurrent.futures import ThreadPoolExecutor,as_completed
import analyze_formal_phonons_concepts as a
from run_modnet_official_matched import np,tf,file_hash,write_json,digest

ROOT=a.ROOT
OUT=ROOT/"benchmarks/formal-phonons-readout-robustness-20261106"


def fit_readouts(h,ht,q,qt):
    h=h.astype(float);ht=ht.astype(float)
    mean=h.mean(axis=0);sd=h.std(axis=0)
    scale=max(float(np.sqrt(np.mean(sd**2))),1e-8)
    x=(h-mean)/scale;xt=(ht-mean)/scale
    qm=q.mean(axis=0);qs=np.maximum(q.std(axis=0),1e-8);y=(q-qm)/qs
    covariance=x.T@x/len(x);rhs=x.T@y/len(x)
    w=np.linalg.solve(covariance+.01*np.eye(x.shape[1]),rhs)
    eigenvalues,vectors=np.linalg.eigh(covariance)
    order=np.argsort(-eigenvalues);rank=min(4,x.shape[1]);u=vectors[:,order[:rank]]
    wr=u@np.linalg.solve(u.T@covariance@u+.01*np.eye(rank),u.T@rhs)
    decoded=(xt@w)*qs+qm;decoded4=(xt@wr)*qs+qm
    return dict(decoded=decoded,decoded4=decoded4,r2=a.r2(qt,decoded),r2_rank4=a.r2(qt,decoded4),
        coefficients=w*qs[None,:]/scale,intercept=qm-mean@(w*qs[None,:]/scale),
        coefficients_rank4=wr*qs[None,:]/scale,intercept_rank4=qm-mean@(wr*qs[None,:]/scale),
        rank=rank,layer_rms_scale=scale,unit_sd=sd,
        nearly_constant_fraction=float(np.mean(sd<.001*max(float(np.median(sd)),1e-8))),
        max_test_unit_zscore=float(np.max(abs((ht-mean)/np.maximum(sd,1e-8)))))


def run_arm(out,arm,p):
    dest=out/"data-checks"/arm;dest.mkdir(parents=True,exist_ok=True)
    banks,targets=a.engine.load_banks(dest)
    import pandas as pd
    qframe=pd.DataFrame(a.concepts(banks["mm"]),index=banks["mm"].index,columns=a.NAMES)
    frame=banks[arm.rsplit("_",1)[0]];kan=arm.endswith("_kan");completed=0;started=time.time();folds=[]
    for fold in range(5):
        source=a.RUN/arm/f"fold-{fold}";selection=a.engine.read(source/"frozen-selection.json")
        done=a.engine.read(source/"complete.json")
        assert file_hash(source/"frozen-selection.json")==done["selection_sha256"]
        assert file_hash(source/"outer-predictions.npz")==done["prediction_sha256"]
        split=a.engine.read(a.engine.OLD/"matbench_phonons"/f"fold-{fold}"/"outer-split.json")
        with np.load(source/"outer-predictions.npz") as z:reference=z["members"].copy()
        scores=[];predictions={k:[] for k in ["first","late","first_rank4","late_rank4"]}
        for i,job in enumerate(selection["members"]):
            folder=out/arm/f"fold-{fold}"/f"member-{i:02d}";folder.mkdir(parents=True,exist_ok=True)
            tr_ids=job["train_ids"];te_ids=split["test_ids"]
            assert not set(tr_ids)&set(te_ids)
            q=qframe.loc[tr_ids].values;qt=qframe.loc[te_ids].values
            if (folder/"summary.json").exists():
                summary=a.engine.read(folder/"summary.json")
                assert summary["protocol_fingerprint"]==digest(p)
                assert file_hash(folder/"readouts.npz")==summary["arrays_sha256"]
                with np.load(folder/"readouts.npz") as z:arrays={k:z[k] for k in z.files}
            else:
                path=Path(job["directory"])/"model.pkl"
                assert file_hash(path)==selection["model_sha256"][job["directory"]]
                with path.open("rb") as f:model=pickle.load(f)
                model._restore_model();cols=list(model.optimal_descriptors[:model.n_feat])
                train,pred,_,_,_,_=a.forward(model,a.scaled(model,frame.loc[tr_ids,cols].values),kan)
                test,pred_test,_,_,_,_=a.forward(model,a.scaled(model,frame.loc[te_ids,cols].values),kan)
                error=float(np.max(abs(pred_test-reference[i])));assert error<.002
                arrays={"q_test":qt};records={}
                for layer in ["first","late"]:
                    r=fit_readouts(train[layer],test[layer],q,qt)
                    for key in ["decoded","decoded4","coefficients","intercept","coefficients_rank4","intercept_rank4","unit_sd"]:
                        arrays[f"{layer}_{key}"]=r[key]
                    records[layer]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items() if k not in ["decoded","decoded4","coefficients","intercept","coefficients_rank4","intercept_rank4","unit_sd"]}
                np.savez_compressed(folder/"readouts.npz",**arrays)
                summary=dict(arm=arm,fold=fold,member=i,model_path=str(path),model_sha256=file_hash(path),
                    protocol_fingerprint=digest(p),records=records,max_prediction_replay_error=error,
                    arrays_sha256=file_hash(folder/"readouts.npz"))
                write_json(folder/"summary.json",summary)
                del model,train,test;tf.keras.backend.clear_session();gc.collect()
            scores.append(summary)
            for layer in ["first","late"]:
                predictions[layer].append(arrays[f"{layer}_decoded"])
                predictions[layer+"_rank4"].append(arrays[f"{layer}_decoded4"])
            completed+=1
            write_json(out/f"status-{arm}.json",dict(state="analyzing",members=completed,fold=fold,seconds=time.time()-started,pid=os.getpid()))
            if completed%25==0:print(arm,completed,round(time.time()-started,1),flush=True)
        ensemble={k:np.mean(v,axis=0) for k,v in predictions.items()}
        directory=out/arm/f"fold-{fold}";np.savez_compressed(directory/"ensemble.npz",q_test=qt,**ensemble)
        rows=[]
        for layer in ["first","late"]:
            r=a.r2(qt,ensemble[layer]);r4=a.r2(qt,ensemble[layer+"_rank4"])
            rmse=np.sqrt(np.mean((qt-ensemble[layer])**2,axis=0))
            for c,name in enumerate(a.NAMES):
                rows.append(dict(concept=name,layer=layer,global_scale_ensemble_r2=float(r[c]),
                    rank4_ensemble_r2=float(r4[c]),rmse_physical_units=float(rmse[c]),
                    median_member_r2=float(np.median([s["records"][layer]["r2"][c] for s in scores]))))
        record=dict(arm=arm,fold=fold,rows=rows,max_replay_error=max(s["max_prediction_replay_error"] for s in scores),
            max_test_unit_zscore={l:max(s["records"][l]["max_test_unit_zscore"] for s in scores) for l in ["first","late"]})
        write_json(directory/"summary.json",record);folds.append(record)
    write_json(out/f"analysis-{arm}.json",dict(complete=True,models=completed,folds=folds,protocol_fingerprint=digest(p)))
    write_json(out/f"status-{arm}.json",dict(state="complete",members=completed,seconds=time.time()-started,pid=os.getpid()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=a.engine.ARMS);ap.add_argument("--workers",type=int,default=4);args=ap.parse_args()
    out=OUT;out.mkdir(parents=True,exist_ok=True)
    p=dict(version=1,source_sha256=file_hash(__file__),extraction_source_sha256=file_hash(a.__file__),
        parent_protocol_sha256=file_hash(ROOT/"benchmarks/formal-phonons-concepts-20261106/protocol.json"),
        motivation="Original probes showed extreme held-out extrapolation for some MLP members; do not treat this as architectural superiority without a scale control.",
        scaling="TRAIN center per unit, ONE shared sqrt(mean(unit_variance)) scale per layer; no division by individual tiny unit variances.",
        penalty=.01,rank_control=4,rank_selection="First four TRAIN principal components, or all if layer narrower; no target-driven component selection.",
        inputs="Same 1000 frozen formal phonons models and exactly the same training/outer-test rows as parent.",
        labels="Physical descriptors only; no phonon target labels used to fit or select readouts.",
        limitations=["Exploratory robustness analysis initiated after seeing the original probe instability.",
            "Derived physical readouts do not establish an intrinsic physical neuron or causal mechanism.",
            "Original per-unit standardized results are retained in the parent directory."])
    if (out/"protocol.json").exists():assert a.engine.read(out/"protocol.json")==p
    else:write_json(out/"protocol.json",p)
    if args.arm:run_arm(out,args.arm,p);return
    def launch(arm):
        with (out/f"{arm}.log").open("w",encoding="utf-8") as log:
            subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),"--arm",arm],stdout=log,stderr=subprocess.STDOUT,
                check=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        return arm
    started=time.time();finished=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(launch,arm) for arm in a.engine.ARMS]):
            finished.append(f.result());print(json.dumps(dict(completed_arms=finished,seconds=time.time()-started)),flush=True)
    pieces=[a.engine.read(out/f"analysis-{arm}.json") for arm in a.engine.ARMS]
    write_json(out/"analysis.json",dict(complete=True,models=sum(p["models"] for p in pieces),
        folds=[f for p in pieces for f in p["folds"]],protocol_fingerprint=digest(p),seconds=time.time()-started))
    write_json(out/"status.json",dict(state="complete",models=1000,seconds=time.time()-started))


if __name__=="__main__":main()
