"""Four-way steel comparison using the fixed-50 paper reconstruction protocol."""
import os
import sys
from pathlib import Path
import copy
import json
import pickle
import traceback
import time
import subprocess
import importlib.metadata as md
from datetime import datetime, timezone
import run_modnet_official_matched as nm
import run_mattervial_phonons as mv
import run_native_matbench_expansion as exp
from modnet.hyper_opt.fit_genetic import Individual
from scaled_native_spline import fit_spline
from queue_status_io import write_status

np, tf = nm.np, nm.tf
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'benchmarks/matbench-expansion-20261101/matbench_steels'
REFERENCE = ROOT/'benchmarks/steels-paper-reproduction-20261102/lmp-v3-paper-fixed50'
BASE = ROOT/'benchmarks/steels-paper-fourway-20261106'
ARMS = ('mm_modnet', 'mm_kan', 'mm_mv_kan')
OPTIONS = [50]
from steels_paper_shap_kernel import importance as paper_importance
read = mv.read

def status(out, state, **kw):
    write_status(out/'status.json',dict(state=state,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),**kw))

def evaluate(job):
    job = dict(job)
    arm = Path(job['directory']).relative_to(BASE).parts[1]
    if arm.endswith('_kan') and Path(job['directory']).name != 'pilot':
        job['variant']='bspline'
        nm.fit_kan=lambda i,t,v,variant,fast:fit_spline(i,t,v,variant,fast,seed=job['seed'])
    return nm._evaluate_job(job)

class Pool(nm.SafePool):
    def imap_unordered(self, function, tasks):
        return super().imap_unordered(evaluate,tasks)

class GA(nm.CheckpointGA):
    def _init_run(self,n_jobs=None):
        self.pool=Pool(n_jobs or 1)

def ranking(train,val,folder,arm,seed,fp,smoke):
    folder.mkdir(parents=True,exist_ok=True)
    # No labels, samples, or model selection from validation enter either ranking.
    cols=list(train.df_featurized)
    ids=list(train.df_targets.index)
    assert set(ids).isdisjoint(val.df_targets.index)
    identity=nm.digest([fp,arm,seed,ids,cols])
    path=folder/'ranking.json'
    if path.exists():
        rec=read(path);assert rec['identity']==identity;return rec
    # Paired architectures use the same MODNet-pilot ranking and inner split.
    if arm.endswith('_kan'):
        if arm == 'mm_kan':
            donor = folder.parents[1].parent/'mm_modnet'/folder.parent.name/folder.name/'ranking.json'
        else:
            donor = REFERENCE/folder.relative_to(BASE).parts[0]/'modnet'/folder.parent.name/folder.name/'ranking.json'
        rec = read(donor)
        assert rec['train_ids'] == ids
        assert list(rec['values']) == cols
        assert set(rec['train_ids']).isdisjoint(val.df_targets.index)
        assert nm.file_hash(donor.parent/'pilot/model.pkl') == rec['pilot_sha256']
        rec = dict(rec, identity=identity, ranking_source=str(donor),
                   ranking_source_sha256=nm.file_hash(donor))
        nm.write_json(path, rec)
        return rec
    own=copy.copy(train);own.optimal_features=cols
    nm.set_seed(seed)
    model_args=dict(max_feat=len(cols),num_classes=own.num_classes,multi_label=False,
                    targets=[[list(own.target_names)]],weights={n:1 for n in own.target_names})
    ind=Individual(**model_args)
    ind.genes.update(n_feat=len(cols),act='elu',loss='mae',n_neurons_first_layer=128,
                     fraction1=.5,fraction2=.5,fraction3=1,xscale='minmax',lr=.001,batch_size=32)
    # Fixed pilot. Loss-based stopping; using train also for the diagnostic
    # validation callback avoids feeding any held-out labels to the pilot.
    res=evaluate(dict(individual=ind,train_data=own,val_data=own,directory=str(folder/'pilot'),
                      seed=seed,fast=smoke,variant='modnet',run_fingerprint=fp))
    with (folder/'pilot/model.pkl').open('rb') as stream: m=pickle.load(stream)
    m._restore_model()
    descriptors=list(m.optimal_descriptors[:m.n_feat]);assert descriptors==cols
    x=np.nan_to_num(own.df_featurized[descriptors].values)
    if m._scaler is not None: x=np.nan_to_num(m._scaler.transform(x),nan=-1)
    def pred(z):
        return np.concatenate([np.asarray(m.model(z[i:i+128],training=False),dtype=float).reshape(-1)
                               for i in range(0,len(z),128)])
    vals,ix,shap_settings=paper_importance(x,pred,seed+524,smoke)
    assert vals.sum()>0
    order=np.argsort(-vals,kind='stable'); mass=np.cumsum(vals[order])/vals.sum()
    subsets={str(t):[cols[i] for i in order[:max(2,np.searchsorted(mass,t/100)+1)]] for t in OPTIONS}
    rec=dict(identity=identity,subsets=subsets,train_ids=ids,explained_and_background_ids=[ids[i] for i in ix],
             values=dict(zip(cols,vals)),shap_settings=shap_settings,pilot_sha256=nm.file_hash(folder/'pilot/model.pkl'))
    nm.write_json(path,rec);tf.keras.backend.clear_session()
    return rec

def runfold(out,arm,fold,banks,y,fp,smoke):
    folder=out/arm/f'fold-{fold}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        rec=read(folder/'complete.json');assert rec['fp']==fp;return rec
    split=read(SRC/'inputs/mm'/f'fold-{fold}'/'manifest.json')
    tr,te=split['train_ids'],split['outer_test_ids']
    assert not set(tr)&set(te) and set(tr+te)==set(y.index)
    X=banks['mm_mv']
    pooling=[c for c in X if c.startswith('ROOST_mpgap_LayerMaterialPooling_')]
    assert len(pooling)==64 and len(banks['mm'].columns)==268
    X=X[list(banks['mm'].columns)+pooling] if arm.startswith('mm_mv_') else banks['mm']
    if smoke:X=X.iloc[:,np.linspace(0,len(X.columns)-1,24,dtype=int).tolist()]
    train=mv.make_data(X.loc[tr],y.loc[tr]);train.optimal_features=list(X)
    seed=260902+fold*100000
    nested=2 if smoke else 5
    validations={}
    for option in OPTIONS:
        stage=folder/str(option);stage.mkdir(exist_ok=True)
        sel=stage/'inner-selection.json'
        if sel.exists():
            entry=read(sel);assert entry['fp']==fp;validations[str(option)]=entry;continue
        nm.set_seed(seed)
        ga=GA(train,stage,seed,fp)
        pairs=[];widths=[]
        for k,indices in enumerate(nm.matbench_kfold_splits(ga.train_data,n_splits=nested)):
            it,iv=ga.train_data.split(indices)
            status(out,'inner_ranking',arm=arm,fold=fold,inner=k,option=option)
            rec=ranking(it,iv,folder/f'inner-{k}',arm,seed+k,fp,smoke)
            features=rec['subsets'][str(option)]
            it.optimal_features=features;iv.optimal_features=features
            pairs.append((it,iv));widths.append(len(features))
        ga.train_data.optimal_features=list(X)[:min(widths)]
        ga.inner_data[nested]=pairs
        nm.set_seed(seed+777)
        status(out,'validation_search',arm=arm,fold=fold,option=option,progress=str(stage/'progress.json'))
        try:
            ensemble=ga.run(n_jobs=1,fast=smoke,size_pop=50,num_generations=20,nested=nested,early_stopping=8,refit=False)
        except BaseException:
            if getattr(ga,'pool',None):ga.pool.terminate()
            raise
        jobs=[ga.job_by_path[m.path] for m in ensemble.model]
        members=[dict(directory=j['directory'],seed=j['seed'],inner=int(j['directory'].split('-inner-')[-1]),
                      sha256=nm.file_hash(Path(j['directory'])/'model.pkl')) for j in jobs]
        scores=[]
        for k,(_,iv) in enumerate(pairs):
            subset=[j for j in members if j['inner']==k]
            assert subset and set(iv.df_targets.index).issubset(tr)
            pred,_,_=nm.predict_members(subset,iv,stage,f'validation-{k}')
            scores.extend(abs(pred-iv.df_targets.values.ravel()).tolist())
        assert len(scores)==len(tr)
        entry=dict(fp=fp,validation_mae=float(np.mean(scores)),members=members)
        nm.write_json(sel,entry);validations[str(option)]=entry
    best=50  # User-fixed before this run; never selected from outer performance.
    chosen=validations[str(best)]['members']
    frozen=dict(fp=fp,option=best,threshold_selection='user-fixed-50',validation_maes={k:v['validation_mae'] for k,v in validations.items()},members=chosen)
    nm.write_json(folder/'frozen-selection.json',frozen)
    # First access to outer-test target values in the training/evaluation path.
    test=mv.make_data(X.loc[te],y.loc[te])
    for j in chosen:assert nm.file_hash(Path(j['directory'])/'model.pkl')==j['sha256']
    pred,_,replaced=nm.predict_members(chosen,test,folder,'outer')
    target=test.df_targets.values.ravel()
    np.savez_compressed(folder/'test.npz',ids=np.array(te),prediction=pred,target=target)
    result=dict(fp=fp,arm=arm,fold=fold,threshold=best,mae=float(np.mean(abs(pred-target))),
                n=len(te),native_replacements=replaced,test_sha256=nm.file_hash(folder/'test.npz'))
    nm.write_json(folder/'complete.json',result);return result

def main(smoke):
    out=BASE/('smoke' if smoke else 'formal');out.mkdir(parents=True,exist_ok=True)
    lock=out/'run.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        assert md.version('modnet')=='0.1.13'
        frozen=read(SRC/'protocol.json')
        sources={str(Path(p).resolve()):nm.file_hash(Path(p)) for p in [__file__,nm.__file__,mv.__file__,exp.__file__,
                 *[Path(__file__).with_name(n) for n in ['native_fourway_models.py','relaxed_spline_policy.py',
                 'audit_steels_paper_fourway.py','audit_steels_shap_fold_v3.py','steels_paper_shap_kernel.py','scaled_native_spline.py','run_native_fourway.py','official_kan_models.py','standalone_kan_models.py']]]}
        proto=dict(smoke=smoke,sources=sources,source_protocol=nm.file_hash(SRC/'protocol.json'),input_hashes={
                   p:nm.file_hash(Path(p)) for p in [frozen['csv'],frozen['original']]},options=OPTIONS,feature_bank="MM: 268 MatMiner; MM+MV: same 268 plus 64 MP-gap ROOST pooling features; author exact columns unpublished",
                   selection='user-fixed threshold 50; inner validation selects ensemble only; no threshold sweep or outer-score tuning',
                   design='SHAP 0.44.1 KernelExplainer, up to 300 inner-training rows, nsamples=500, l1_reg=0; fixed full-input pilot; threshold 50 interpreted as cumulative mean absolute attribution, minimum 2 features. Explainer, pilot, threshold semantics and ROOST MP-gap bank remain reconstruction assumptions; not exact author code.',
                   training='MODNet 0.1.13; population 50; up to 20 generations; patience 8; five inner folds; no refit; native outlier-corrected ensemble MAE; identical search settings for scaled B-spline KAN',
                   criterion='Compare paired outer folds and complete fivefold means; historical benchmark and paper are reference results.',
                   paired_selection='Shared MODNet-pilot SHAP ranking within each feature bank, same inner IDs and seed schedule.',
                   baseline_reuse=dict(arm='mm_mv_modnet', directory=str(REFERENCE/('smoke' if smoke else 'formal')),
                                      protocol_sha256=nm.file_hash(REFERENCE/('smoke' if smoke else 'formal')/'protocol.json')),
                   limitations=['Prior outer scores have been observed; protocol change requested by user.',
                                'KAN is an experimental extension of the paper.',
                                'KAN uses train-only target standardization; MODNet uses raw targets.',
                                'Independent GA searches share caps but can stop at different generations.',
                                'No RFE reduction is necessary below the 800-feature cap.'])
        fp=nm.digest(proto)
        if (out/'protocol.json').exists():assert read(out/'protocol.json')==proto
        nm.write_json(out/'protocol.json',proto)
        if not smoke:
            chk=read(BASE/'smoke/audit.json');assert chk['passed'] and chk['sources']==sources
        banks,y=exp.load_banks(frozen,out)
        results={}
        for arm in ARMS:
            if arm == 'mm_mv_kan':
                reference_out = REFERENCE/('smoke' if smoke else 'formal')
                while read(reference_out/'status.json')['state'] != 'complete':
                    if read(reference_out/'status.json')['state'] == 'failed':
                        raise RuntimeError('Reference MODNet failed; inspect before continuing.')
                    status(out, 'waiting_for_reference_modnet', arm=arm)
                    time.sleep(30)
                subprocess.run([sys.executable, str(ROOT/'scripts/audit_steels_shap_fold_v3.py'),
                                'smoke' if smoke else 'formal'], check=True)
            rows=[]
            for fold in ([0] if smoke else range(5)):
                res=runfold(out,arm,fold,banks,y,fp,smoke)
                path=out/arm/f'fold-{fold}'/'test.npz'
                assert res['test_sha256']==nm.file_hash(path)
                with np.load(path) as a:
                    assert abs(res['mae']-np.mean(abs(a['target']-a['prediction'])))<1e-8
                subprocess.run([sys.executable, str(ROOT/'scripts/audit_steels_paper_fourway.py'),
                                'smoke' if smoke else 'formal'], check=True)
                rows.append(res)
            avg=float(np.mean([v['mae'] for v in rows]))
            results[arm]=dict(rows=rows,mean=avg)
            if not smoke and arm=='modnet':
                results[arm].update(delta_vs_paper=avg-85.12,within_one_percent=abs(avg-85.12)<=.8512)
            nm.write_json(out/f'{arm}-verified.json',results[arm])
        reference_out = REFERENCE/('smoke' if smoke else 'formal')
        reference_rows = [read(reference_out/'modnet'/f'fold-{fold}'/'complete.json')
                          for fold in ([0] if smoke else range(5))]
        results['mm_mv_modnet'] = dict(rows=reference_rows,
                                      mean=float(np.mean([row['mae'] for row in reference_rows])),
                                      reused_from=str(reference_out))
        nm.write_json(out/'audit.json',dict(passed=True,smoke=smoke,sources=sources,results=results))
        nm.write_json(out/'fourway-summary.json', dict(complete=True, smoke=smoke, unit='MPa', results=results))
        status(out,'complete',smoke=smoke)
    except BaseException:
        status(out,'failed',traceback=traceback.format_exc());raise
    finally:lock.unlink()

if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('smoke','formal','sequence'):sys.exit('smoke | formal | sequence')
    if sys.argv[1]!='formal':main(True)
    if sys.argv[1]!='smoke':main(False)
