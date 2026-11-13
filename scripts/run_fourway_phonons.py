"""Resumable no-SISSO four-cell benchmark, with a strong-baseline gate.

The published-protocol reproduction is audited separately from this uniform,
strictly nested protocol. No old selected features or weights are reused.
"""
from run_modnet_official_matched import SafePool, digest, file_hash, set_seed, write_json
from fourway_phonons_models import build_model, np, tf
from fourway_phonons_policy import (ARMS, candidate_grid, dense_widths, dense_parameters,
    spline_widths, spline_parameters, choose, baseline_gate, factorial_effects)
from run_mattervial_phonons import CSV, CSV_SHA, ORIGINAL, feature_group, make_data, xgb_rfe
from standalone_kan_fivefold_policy import official_ids, validate_fivefold
from modnet.preprocessing import MODData, get_cross_nmi, nmi_target, get_features_dyn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from pathlib import Path
from datetime import datetime, timezone
import argparse
import contextlib
import gc
import gzip
import importlib.metadata
import inspect
import json
import os
import pickle
import shutil
import time
import traceback
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'benchmarks/official-aligned-20261021'
VERIFIED_SOURCES = set()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def now():
    return datetime.now(timezone.utc).isoformat()


def check_worker_sources(job):
    """A newly spawned worker must execute the frozen source, including on resume."""
    fp=job['run_fingerprint']
    if fp in VERIFIED_SOURCES:return
    p=read(job['protocol_path'])
    if digest(p)!=fp:raise ValueError('Changed parent protocol')
    for path,value in p['source_sha256'].items():
        if file_hash(path)!=value:raise ValueError('Frozen source changed: '+path)
    VERIFIED_SOURCES.add(fp)


def audit_reproduction(out):
    """Re-score both published and locally trained predictions, all 1,265 IDs."""
    with gzip.open(OLD/'reference/results.json.gz', 'rt') as f:
        reference = json.load(f)['tasks']['matbench_phonons']['results']
    rows, seen, artifacts = [], [], {}
    protocol = read(OLD/'matched-protocol.json')
    if protocol['smoke'] or protocol['packages']['modnet'] != '0.1.12':
        raise ValueError('Not the formal MODNet reproduction')
    for fold in range(5):
        directory = OLD/'matbench_phonons'/f'fold-{fold}'
        split = read(directory/'outer-split.json')
        train, test = official_ids(split)
        frame = pd.read_csv(directory/'predictions.csv')
        if frame.sample_id.tolist() != test:
            raise ValueError('Reproduction test IDs changed')
        frozen = read(directory/'frozen-ensemble.json')['members']
        if len(frozen) != 50:
            raise ValueError('Expected fifty official ensemble members')
        for item in frozen:
            p = Path(item['directory'])/'result.json'
            r = read(p)
            if set(r['train_ids']) & set(r['validation_ids']):
                raise ValueError('Reproduction inner overlap')
            if set(r['train_ids']+r['validation_ids']) != set(split['train_ids']):
                raise ValueError('Reproduction outer contamination')
            if any('SISSO' in x.upper() for x in r['selected_features']):
                raise ValueError('Unexpected SISSO in pure baseline')
            if not (p.parent/'model.pkl').is_file():
                raise ValueError('Missing locally trained baseline model')
            artifacts[str(p)] = file_hash(p)
        with np.load(directory/'modnet-member-predictions.npz') as z:
            if z['predictions'].shape != (50, 253):
                raise ValueError('Wrong prediction matrix')
            pred = z['predictions'].mean(axis=0)
        np.testing.assert_allclose(pred, frame.modnet, atol=1e-8, rtol=0)
        mae = float(np.mean(abs(pred-frame.target.values)))
        published = np.array([reference[f'fold_{fold}']['data'][sid] for sid in test])
        pm = float(np.mean(abs(published-frame.target.values)))
        np.testing.assert_allclose(pm, reference[f'fold_{fold}']['scores']['mae'], atol=1e-9, rtol=0)
        np.testing.assert_allclose(mae, read(directory/'baseline.json')['results']['modnet']['mae'], atol=1e-9, rtol=0)
        for name in ('predictions.csv', 'modnet-member-predictions.npz', 'outer-split.json', 'baseline.json'):
            artifacts[str(directory/name)] = file_hash(directory/name)
        rows.append(dict(fold=fold, local_mae=mae, published_mae=pm))
        seen += test
    if len(seen) != len(set(seen)) or len(seen) != 1265:
        raise ValueError('Incomplete reproduction OOF coverage')
    local, published = [float(np.mean([r[key] for r in rows])) for key in ('local_mae', 'published_mae')]
    result = dict(checked_at=now(), complete=True, local_retrained_mae=local, published_mae=published,
        relative_deviation=local/published-1, rows=rows, verified_selected_members=250,
        passed=abs(local/published-1) <= .02, artifacts=artifacts,
        meaning='Audit of previously trained official-protocol baseline; not a new training run under the four-cell protocol')
    write_json(out/'baseline-reproduction-audit.json', result)
    if not result['passed']:
        raise ValueError('Existing formal baseline reproduction did not pass audit')
    return result


def freeze(out, smoke):
    files = [Path(__file__), *[ROOT/'scripts'/name for name in (
        'fourway_phonons_policy.py', 'fourway_phonons_models.py', 'standalone_kan_models.py',
        'physical_feature_modules.py', 'run_modnet_official_matched.py', 'modnet_runtime.py',
        'official_kan_models.py', 'run_mattervial_phonons.py', 'standalone_kan_fivefold_policy.py',
        'audit_fourway_phonons.py')]]
    import modnet.preprocessing, modnet.models.vanilla
    files += [Path(inspect.getfile(m)) for m in (modnet.preprocessing, modnet.models.vanilla)]
    folds = []
    for i in range(5):
        path = OLD/'matbench_phonons'/f'fold-{i}'/'outer-split.json'
        tr, te = official_ids(read(path))
        folds.append(dict(fold=i, train_ids=tr, test_ids=te, source_sha256=file_hash(path)))
    validate_fivefold(folds)
    if smoke:
        folds = [dict(folds[0], train_ids=folds[0]['train_ids'][:80], test_ids=folds[0]['test_ids'][:20])]
    candidates = candidate_grid(smoke)
    architectures = []
    for n, width in sorted({(c['n_feat'], c['width']) for c in candidates}):
        a, b = spline_widths(n, width)
        architectures.append(dict(n_feat=n, width=width, modnet_widths=dense_widths(width), kan_widths=[a,b],
            modnet_parameters=dense_parameters(n,dense_widths(width)), kan_parameters=spline_parameters(n,a,b)))
    p = dict(kind='Four-cell phonons factorial benchmark', smoke=smoke, arms=list(ARMS), folds=folds,
        no_sisso=True, raw_mm=str(ORIGINAL), raw_mm_sha256=file_hash(ORIGINAL), raw_mv=str(CSV), raw_mv_sha256=file_hash(CSV),
        packages={n:importlib.metadata.version(n) for n in ('modnet','tensorflow','numpy','pandas','scikit-learn','xgboost')},
        source_sha256={str(f):file_hash(f) for f in files}, candidates=candidates, architectures=architectures,
        seed=20261029, inner_split_seed=8029, inner_folds=2 if smoke else 5,
        keep_candidates=1 if smoke else 10, max_epochs=2 if smoke else 800,
        rfe_count=32 if smoke else 800,
        budget_per_arm_per_fold=len(candidates)*(2 if smoke else 5),
        total_fits=len(candidates)*(2 if smoke else 5)*len(folds)*4,
        feature_banks='Identical original MatMiner values in both banks; enhanced appends only non-SISSO MEGNet OFM, MVL16/32, ROOST mpgap, ORB v3. No author MatMiner replacement.',
        selection='Within EACH inner train: drop constants, XGB RFE to 800, native MODNet relevance-redundancy ranking with cross NMI fitted there and explicit random_state for both cross/target MI; append any entropy-filtered RFE survivors in sorted-name order. No old rankings. First n_feat selected for each candidate.',
        scaling='Inner TRAIN minmax [-0.5,0.5], target mean/std; same transformations for both networks, no target-informed test transforms.',
        training='Adam, standardized-target MAE, validation MAE early stop patience40/min_delta0.05 cm^-1, restore exact minimum validation-MAE checkpoint; no extra regularization.',
        models='Native MODNet ELU/linear hierarchy; complete 3-layer B-spline predictor with no Dense trunk/residual, near-linear Greville initialization. Shape determined without scores, within5% parameters per corresponding candidate.',
        search='Fixed exhaustive 192-point grid, independently ranked per arm. No adaptive GA stopping, no MODNet-selected configurations reused for KAN.',
        ensemble='Top10 candidates by mean five-inner-fold MAE; retain all five models, equal 50-member average. No test early stopping, no refit, no outlier replacement.',
        baseline_quality_gate='Run all five mm_modnet folds first; require mean MAE <= published34.27510615406341*1.02. Otherwise stop for diagnosis, preserve all results, never retune from outer scores.',
        reporting='All five folds; MAE and sample fold SD; A-B, C-D, A-C, B-D, A-D, (B-D)-(A-C); OOF1265 unique; actual params, fitted epochs, task time and wall time.',
        limitations=['This unified protocol differs from original-paper GA/scaling/early-stopping/feature-selection. Historical reproduction is separate and verified first.',
            'Previously explored dataset and KAN design: not an untouched external confirmation.',
            'No SISSO does not establish absence of external embedding pretraining overlap.',
            'Equal fit/epoch-cap budget, not equal wall time; selected ensemble capacities can differ after independent configuration selection.',
            'The 2% baseline check uses outer results once to assess baseline adequacy, not to choose hyperparameters; conditional evaluation must be disclosed.'])
    if p['raw_mv_sha256'] != CSV_SHA:
        raise ValueError('Unexpected MatterVial source')
    path = out/'protocol.json'
    if path.exists() and digest(read(path)) != digest(p):
        raise ValueError('Frozen protocol changed; refusing to mix runs')
    write_json(path,p)
    for source, sha in p['source_sha256'].items():
        destination=out/'source-snapshot'/(Path(source).parent.name+'--'+Path(source).name)
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists(): shutil.copyfile(source,destination)
        if file_hash(destination)!=sha: raise ValueError('Source snapshot changed')
    return p


def load_banks(out,p):
    """One common MM source; enhanced is a strict additive feature-bank change."""
    original=MODData.load(ORIGINAL)
    expected=[f'id{i}' for i in range(1265)]
    if original.df_targets.index.tolist()!=expected or original.df_featurized.index.tolist()!=expected:
        raise ValueError('Unexpected original sample ordering')
    ids=[f'mb-phonons-{i+1:04d}' for i in range(1265)]
    mm=original.df_featurized.copy();mm.index=ids
    y=pd.Series(original.df_targets.values[:,0],index=ids,name='target')
    header=pd.read_csv(CSV,nrows=0).columns.tolist()
    extras=sorted(c for c in header if 'SISSO' not in c.upper() and feature_group(c) not in (None,'matminer','sisso'))
    extra=pd.read_csv(CSV,usecols=['material_id','target']+extras).set_index('material_id').loc[ids]
    np.testing.assert_allclose(extra.target,y,atol=1e-8,rtol=0)
    extra=extra.drop(columns='target')
    if set(mm.columns)&set(extra.columns) or any('SISSO' in c.upper() for c in list(mm.columns)+extras):
        raise ValueError('Feature-bank contamination')
    mm=mm.replace([np.inf,-np.inf],np.nan).fillna(0)
    extra=extra.replace([np.inf,-np.inf],np.nan).fillna(0)
    enhanced=pd.concat([mm,extra],axis=1)
    pd.testing.assert_frame_equal(enhanced[mm.columns],mm)
    info=dict(mm_features=len(mm.columns),added_features=len(extras),enhanced_features=len(enhanced.columns),
        excluded_sisso_count=sum('SISSO' in c.upper() for c in header),added_feature_names=extras,
        original_mm_identical_in_both_banks=True,targets_match=True,fixed_zero_missing_fill=True)
    write_json(out/'data-audit.json',info)
    return {'mm':mm,'mm_mv':enhanced},y


def prepare_input(job):
    check_worker_sources(job)
    directory=Path(job['directory']);directory.mkdir(parents=True,exist_ok=True)
    signature=digest(job);path=directory/'manifest.json'
    if path.exists():
        r=read(path)
        if r['signature']!=signature: raise ValueError('Changed preprocessing job')
        for name,sha in r['artifacts'].items():
            if file_hash(directory/name)!=sha: raise ValueError('Changed preprocessing artifact')
        return r
    started=time.perf_counter()
    if file_hash(job['cache'])!=job['cache_sha256']: raise ValueError('Changed training-only cache')
    frame=pd.read_pickle(job['cache'])
    tr,va=job['train_ids'],job['validation_ids']
    if set(tr)&set(va) or set(tr+va)!=set(frame.index): raise ValueError('Wrong inner partition')
    if set(frame.index)&set(job['outer_test_ids']): raise ValueError('Outer contamination')
    features=[c for c in frame if c!='target']
    if any('SISSO' in c.upper() for c in features): raise ValueError('SISSO')
    tx=frame.loc[tr,features]
    features=[c for c in features if tx[c].nunique()>1]
    if len(features)<job['count']: raise ValueError('Not enough nonconstant training features')
    set_seed(job['seed'])
    with (directory/'preparation.log').open('w',encoding='utf-8') as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            selected=xgb_rfe(tx[features],frame.loc[tr,'target'],job['count'],directory/'rfe-progress.json',threads=1)
            # MODData.feature_selection does not expose get_cross_nmi's RNG.
            # Compose its native single-target steps with an explicit seed;
            # otherwise get_cross_nmi constructs an unseeded RandomState().
            cross=get_cross_nmi(tx[selected],n_jobs=1,random_state=job['seed'])
            relevance=nmi_target(tx[selected],frame.loc[tr,['target']],random_state=job['seed'])['target']
            ranked=list(get_features_dyn(-1,cross,relevance))
    # Native MI additionally removes features with very low estimated entropy.
    # Retain a fixed candidate dimension without consulting validation data:
    # append those RFE survivors after every ranked feature, in name order.
    entropy_filtered=sorted(set(selected)-set(ranked))
    ranked+=entropy_filtered
    if len(ranked)!=job['count'] or len(set(ranked))!=len(ranked): raise ValueError('Incomplete feature ranking')
    scaler=MinMaxScaler(feature_range=(-.5,.5))
    x=scaler.fit_transform(frame.loc[tr,ranked]).astype('float32')
    xv=scaler.transform(frame.loc[va,ranked]).astype('float32')
    y=frame.loc[tr,'target'].values.astype(float);yv=frame.loc[va,'target'].values.astype(float)
    mean,std=float(y.mean()),float(y.std())
    if not np.isfinite(std) or std<=0: raise ValueError('Invalid training target scale')
    np.savez_compressed(directory/'arrays.npz',x_train=x,x_validation=xv,y_train=y,y_validation=yv)
    with (directory/'scaler.pkl').open('wb') as f: pickle.dump(scaler,f,protocol=4)
    r=dict(signature=signature,run_fingerprint=job['run_fingerprint'],directory=str(directory),
        train_ids=tr,validation_ids=va,outer_test_ids=job['outer_test_ids'],selected_features=ranked,
        feature_selection_fit_ids=tr,cross_nmi_fit_ids=tr,scaling_fit_ids=tr,target_transform_fit_ids=tr,
        target_mean=mean,target_std=std,entropy_filtered_appended=entropy_filtered,seconds=time.perf_counter()-started,
        artifacts={n:file_hash(directory/n) for n in ('arrays.npz','scaler.pkl')})
    write_json(path,r)
    return r


def train_job(job):
    check_worker_sources(job)
    directory=Path(job['directory']);directory.mkdir(parents=True,exist_ok=True)
    signature=digest(job);path=directory/'result.json'
    if path.exists():
        r=read(path)
        if r['signature']!=signature: raise ValueError('Changed fit job')
        for name,sha in r['artifacts'].items():
            if file_hash(directory/name)!=sha: raise ValueError('Changed fit artifact')
        return r
    started=time.perf_counter()
    try:
        inp=read(job['manifest']);source=Path(inp['directory'])
        if digest(inp)!=job['input_digest']: raise ValueError('Changed input manifest')
        if file_hash(source/'arrays.npz')!=inp['artifacts']['arrays.npz']: raise ValueError('Changed arrays')
        c=job['config'];n=c['n_feat'];mean,std=inp['target_mean'],inp['target_std']
        with np.load(source/'arrays.npz') as z:
            x=z['x_train'][:,:n];xv=z['x_validation'][:,:n];y=z['y_train'];yv=z['y_validation']
        np.testing.assert_allclose([mean,std],[y.mean(),y.std()],rtol=0,atol=1e-10)
        model=build_model(job['variant'],c,job['seed'])
        model.compile(optimizer=tf.keras.optimizers.Adam(c['lr']),loss='mae',metrics=['mae'])
        class Best(tf.keras.callbacks.Callback):
            def __init__(self): super().__init__();self.value=float('inf');self.weights=None;self.epoch=0
            def on_epoch_end(self,epoch,logs=None):
                value=float(logs['val_mae'])
                if not np.isfinite(value): raise ValueError('Nonfinite validation score')
                if value<self.value: self.value=value;self.weights=self.model.get_weights();self.epoch=epoch+1
                if (epoch+1)%50==0:
                    write_json(directory/'status.json',dict(state='training',epoch=epoch+1,best_mae=self.value*std,updated_at=now()))
        best=Best();set_seed(job['seed'])
        with (directory/'training.log').open('w',encoding='utf-8') as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                h=model.fit(x,((y-mean)/std).astype('float32'),
                    validation_data=(xv,((yv-mean)/std).astype('float32')),
                    epochs=job['max_epochs'],batch_size=c['batch_size'],verbose=0,
                    callbacks=[best,tf.keras.callbacks.EarlyStopping(monitor='val_mae',patience=40,min_delta=.05/std)])
        if best.weights is None: raise ValueError('No finite checkpoint')
        model.set_weights(best.weights)
        prediction=model.predict(xv,verbose=0).reshape(-1).astype(float)*std+mean
        mae=float(np.mean(abs(prediction-yv)))
        if not np.isfinite(prediction).all() or abs(mae-best.value*std)>.003: raise ValueError('Checkpoint score mismatch')
        np.savez_compressed(directory/'validation.npz',prediction=prediction,target=yv,sample_ids=inp['validation_ids'])
        with (directory/'model.pkl').open('wb') as f:
            pickle.dump(dict(model_json=model.to_json(),weights=model.get_weights()),f,protocol=4)
        r=dict(signature=signature,run_fingerprint=job['run_fingerprint'],directory=str(directory),arm=job['arm'],
            variant=job['variant'],candidate=c['candidate'],config=c,inner=job['inner'],seed=job['seed'],
            manifest=job['manifest'],input_digest=job['input_digest'],train_ids=inp['train_ids'],validation_ids=inp['validation_ids'],
            features=inp['selected_features'][:n],target_mean=mean,target_std=std,parameters=model.count_params(),
            best_epoch=best.epoch,epochs=len(h.history['loss']),validation_mae=mae,history=h.history,
            seconds=time.perf_counter()-started,artifacts={n:file_hash(directory/n) for n in ('model.pkl','validation.npz')})
        write_json(path,r);write_json(directory/'status.json',dict(state='complete',updated_at=now()))
        del model,best,h;tf.keras.backend.clear_session();gc.collect()
        return r
    except Exception:
        write_json(directory/'status.json',dict(state='failed',updated_at=now(),traceback=traceback.format_exc()))
        raise


def evaluate_outer(directory,selection,x,y,test_ids,p):
    vectors,records=[],[]
    for member in selection['members']:
        source=Path(member);r=read(source/'result.json');inp=read(r['manifest'])
        if digest(inp)!=r['input_digest'] or file_hash(source/'model.pkl')!=r['artifacts']['model.pkl']:
            raise ValueError('Changed selected artifact')
        if set(test_ids)&set(r['train_ids']+r['validation_ids']): raise ValueError('Outer leakage')
        with (Path(inp['directory'])/'scaler.pkl').open('rb') as f: scaler=pickle.load(f)
        values=scaler.transform(x.loc[test_ids,inp['selected_features']]).astype('float32')[:,:r['config']['n_feat']]
        with (source/'model.pkl').open('rb') as f: saved=pickle.load(f)
        model=tf.keras.models.model_from_json(saved['model_json']);model.set_weights(saved['weights'])
        prediction=model.predict(values,verbose=0).reshape(-1).astype(float)*r['target_std']+r['target_mean']
        if not np.isfinite(prediction).all(): raise ValueError('Nonfinite test prediction')
        vectors.append(prediction);records.append(r);del model;tf.keras.backend.clear_session()
    matrix=np.array(vectors);target=y.loc[test_ids].values;prediction=matrix.mean(axis=0)
    if matrix.shape!=(p['keep_candidates']*p['inner_folds'],len(test_ids)): raise ValueError('Incorrect ensemble size')
    np.savez_compressed(directory/'outer-predictions.npz',sample_ids=test_ids,target=target,prediction=prediction,members=matrix)
    score=dict(mae=float(np.mean(abs(prediction-target))),ensemble_size=len(records),
        mean_member_mae=float(np.mean(np.mean(abs(matrix-target),axis=1))),
        parameters_mean=float(np.mean([r['parameters'] for r in records])),parameters_sum=sum(r['parameters'] for r in records),
        selected_seconds=sum(r['seconds'] for r in records),prediction_sha256=file_hash(directory/'outer-predictions.npz'))
    return score


def run_arm_fold(out,p,banks,y,arm,fold,jobs,status):
    bank,variant=arm.rsplit('_',1);spec=p['folds'][fold];fp=digest(p)
    directory=out/arm/f"fold-{spec['fold']}";directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():
        r=read(directory/'complete.json')
        if r['run_fingerprint']!=fp or file_hash(directory/'outer-predictions.npz')!=r['prediction_sha256']:
            raise ValueError('Changed completed fold')
        return r
    inputs_dir=out/'inputs'/bank/f"fold-{spec['fold']}";inputs_dir.mkdir(parents=True,exist_ok=True)
    cache=inputs_dir/'outer-training-only.pkl.gz'
    if not cache.exists():
        frame=banks[bank].loc[spec['train_ids']].copy();frame['target']=y.loc[spec['train_ids']]
        frame.to_pickle(cache,compression='gzip',protocol=4)
    cache_sha=file_hash(cache)
    trids=np.array(spec['train_ids'])
    kfold=KFold(n_splits=p['inner_folds'],shuffle=True,random_state=p['inner_split_seed']+spec['fold'])
    prep=[]
    for i,(tr,va) in enumerate(kfold.split(trids)):
        prep.append(dict(directory=str(inputs_dir/f'inner-{i}'),cache=str(cache),cache_sha256=cache_sha,
            train_ids=trids[tr].tolist(),validation_ids=trids[va].tolist(),outer_test_ids=spec['test_ids'],
            count=p['rfe_count'],seed=p['seed']+spec['fold']*100000+i,run_fingerprint=fp,
            protocol_path=str(out/'protocol.json')))
    inputs={}
    status('preparing_features',arm=arm,fold=spec['fold'])
    with SafePool(min(2,jobs)) as pool:
        for inp in pool.imap_unordered(prepare_input,prep):
            i=int(Path(inp['directory']).name.split('-')[-1]);inputs[i]=inp
            status('preparing_features',arm=arm,fold=spec['fold'],prepared=len(inputs),total=p['inner_folds'])
            print('INPUT',arm,spec['fold'],i,flush=True)
    fit_jobs=[]
    for c in p['candidates']:
        for i in range(p['inner_folds']):
            fit_jobs.append(dict(directory=str(directory/'fits'/f"candidate-{c['candidate']:03d}-inner-{i}"),
                arm=arm,variant=variant,config=c,inner=i,seed=p['seed']+spec['fold']*100000+c['candidate']*100+i,
                manifest=str(Path(inputs[i]['directory'])/'manifest.json'),input_digest=digest(inputs[i]),
                max_epochs=p['max_epochs'],run_fingerprint=fp,protocol_path=str(out/'protocol.json')))
    records=[];status('training',arm=arm,fold=spec['fold'],completed=0,total=len(fit_jobs))
    with SafePool(jobs) as pool:
        for r in pool.imap_unordered(train_job,fit_jobs):
            records.append(r)
            status('training',arm=arm,fold=spec['fold'],completed=len(records),total=len(fit_jobs),last_mae=r['validation_mae'])
            print('FIT',arm,spec['fold'],len(records),'/',len(fit_jobs),'MAE',r['validation_mae'],'epoch',r['best_epoch'],flush=True)
    selection=choose(records,p['candidates'],p['inner_folds'],p['keep_candidates'])
    selection.update(run_fingerprint=fp,selection_uses='inner validation only',test_labels_used=False,
        selected_model_sha256={r['directory']:r['artifacts']['model.pkl'] for r in records if r['directory'] in selection['members']})
    write_json(directory/'frozen-selection.json',selection)
    status('outer_evaluation',arm=arm,fold=spec['fold'])
    score=evaluate_outer(directory,selection,banks[bank],y,spec['test_ids'],p)
    score.update(run_fingerprint=fp,arm=arm,fold=spec['fold'],smoke=p['smoke'],search_fits=len(records),
        search_seconds=sum(r['seconds'] for r in records),search_epochs=sum(r['epochs'] for r in records),
        selection_sha256=file_hash(directory/'frozen-selection.json'))
    write_json(directory/'complete.json',score)
    print('OUTER',arm,spec['fold'],score['mae'],flush=True)
    return score


def report(out,p):
    rows=[]
    for arm in ARMS:
        for spec in p['folds']:
            directory=out/arm/f"fold-{spec['fold']}"
            if not (directory/'complete.json').exists():continue
            r=read(directory/'complete.json')
            if r['run_fingerprint']!=digest(p):raise ValueError('Wrong report provenance')
            if file_hash(directory/'outer-predictions.npz')!=r['prediction_sha256']:raise ValueError('Changed predictions')
            with np.load(directory/'outer-predictions.npz') as z:
                np.testing.assert_array_equal(z['sample_ids'],spec['test_ids'])
                np.testing.assert_allclose(z['prediction'],z['members'].mean(axis=0),atol=1e-10,rtol=0)
                np.testing.assert_allclose(np.mean(abs(z['prediction']-z['target'])),r['mae'],atol=1e-9,rtol=0)
            rows.append(r)
    complete=len(rows)==4*len(p['folds'])
    means={arm:float(np.mean([r['mae'] for r in rows if r['arm']==arm])) for arm in ARMS
           if sum(r['arm']==arm for r in rows)==len(p['folds'])}
    effects=factorial_effects(means) if complete else None
    comparisons={}
    statistics={}
    for arm in means:
        rs=[r for r in rows if r['arm']==arm]
        statistics[arm]=dict(mae=means[arm],fold_sample_sd=float(np.std([r['mae'] for r in rs],ddof=1)) if len(rs)>1 else 0,
            search_fits=sum(r['search_fits'] for r in rs),search_seconds=sum(r['search_seconds'] for r in rs),
            search_epochs=sum(r['search_epochs'] for r in rs),mean_member_mae=float(np.mean([r['mean_member_mae'] for r in rs])),
            mean_parameters_per_member=float(np.mean([r['parameters_mean'] for r in rs])),
            mean_parameters_per_ensemble=float(np.mean([r['parameters_sum'] for r in rs])))
    if complete:
        for key,control,alternative in (
            ('mattervial_with_modnet','mm_modnet','mm_mv_modnet'),
            ('mattervial_with_kan','mm_kan','mm_mv_kan'),
            ('kan_with_matminer','mm_modnet','mm_kan'),
            ('kan_with_mattervial','mm_mv_modnet','mm_mv_kan'),
            ('total','mm_modnet','mm_mv_kan')):
            a=[next(r['mae'] for r in rows if r['arm']==control and r['fold']==s['fold']) for s in p['folds']]
            b=[next(r['mae'] for r in rows if r['arm']==alternative and r['fold']==s['fold']) for s in p['folds']]
            comparisons[key]=dict(control=control,alternative=alternative,absolute_improvement=means[control]-means[alternative],
                relative_improvement=1-means[alternative]/means[control],wins=sum(x>y for x,y in zip(a,b)),
                per_fold_absolute_improvement=[x-y for x,y in zip(a,b)])
    if complete and not p['smoke']:
        for arm in ARMS:
            parts=[]
            for spec in p['folds']:
                with np.load(out/arm/f"fold-{spec['fold']}"/'outer-predictions.npz') as z:
                    parts.append({k:z[k].copy() for k in ('sample_ids','target','prediction')})
            ids=np.concatenate([x['sample_ids'] for x in parts])
            if len(ids)!=1265 or len(set(ids))!=1265:raise ValueError('Incomplete OOF coverage')
            np.savez_compressed(out/f'oof-{arm}.npz',**{k:np.concatenate([x[k] for x in parts]) for k in parts[0]})
    analysis=dict(updated_at=now(),complete=complete,smoke=p['smoke'],rows=rows,full_fold_means=means,
        effects=effects,comparisons=comparisons,statistics=statistics,expected_fits=p['total_fits'],
        completed_fits_in_finished_folds=sum(r['search_fits'] for r in rows))
    write_json(out/'analysis.json',analysis)
    lines=['# 无 SISSO 四组统一协议五折比较','',f"完成外折单元：{len(rows)}/{4*len(p['folds'])}；smoke={p['smoke']}。",
        '', '原论文协议的本地复现：34.013377；公开预测复算：34.275106。详见 baseline-reproduction-audit.json。',
        '新统一协议的结果单独统计，不把历史34.013替换进四组表格。','',
        '| 组别 | 外层折 | MAE | 成员数 | 搜索训练次数 |','| --- | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append(f"| {r['arm']} | {r['fold']+1} | {r['mae']:.6f} | {r['ensemble_size']} | {r['search_fits']} |")
    lines+=['','仅完整五折组生成平均值；不把部分折均值冒充正式结果。']
    for arm,mae in means.items():
        sd=float(np.std([r['mae'] for r in rows if r['arm']==arm],ddof=1)) if len(p['folds'])>1 else 0
        lines.append(f'- {arm}: {mae:.6f} ± {sd:.6f}（折间样本标准差）')
    if effects: lines+=['','正值表示降低 MAE（cm⁻¹）：',*[f'- {k}: {v:.6f}' for k,v in effects.items()]]
    if comparisons:
        lines+=['','| 比较 | MAE绝对降低 | 相对降低 | 改善折数 |','| --- | ---: | ---: | ---: |']
        for k,v in comparisons.items():
            lines.append(f"| {k} | {v['absolute_improvement']:.6f} | {v['relative_improvement']:.2%} | {v['wins']}/{len(p['folds'])} |")
    lines+=['','先检查新协议纯MODNet五折是否不劣于34.275106的2%范围；未通过时停止并保留结果，不根据测试分数追调参数。',
        '192配置×5内层折=每组每外折960次拟合；五折四组19,200次。前10配置×5模型固定50成员等权集成。',
        '所有筛选、缩放及目标标准化只拟合内层训练；选模只用内层验证。无SISSO。',
        '当前数据与架构此前被探索过，不是全新独立确认；预训练嵌入的来源限制保留。',
        '完整协议见protocol.json。SMOKE仅证明代码路径，不作为性能结果。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n-jobs',type=int,default=4)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.report_only:report(out,read(out/'protocol.json'));return
    start=time.perf_counter()
    def status(state,**kw):
        write_json(out/'status.json',dict(state=state,pid=os.getpid(),updated_at=now(),session_wall_seconds=time.perf_counter()-start,**kw))
    try:
        status('auditing_historical_reproduction')
        audit=audit_reproduction(out)
        if args.audit_only:status('historical_reproduction_verified',mae=audit['local_retrained_mae']);return
        p=freeze(out,args.smoke);status('loading_audited_feature_banks',expected_total_fits=p['total_fits'])
        banks,y=load_banks(out,p)
        baseline=[]
        for i in range(len(p['folds'])):
            baseline.append(run_arm_fold(out,p,banks,y,'mm_modnet',i,args.n_jobs,status))
            report(out,p)
        gate=baseline_gate([r['mae'] for r in baseline],audit['published_mae'],args.smoke)
        gate['baseline_predictions_sha256']=[r['prediction_sha256'] for r in baseline]
        write_json(out/'baseline-quality-gate.json',gate)
        if not gate['passed']:
            status('baseline_quality_requires_investigation',gate=gate)
            print('BASELINE QUALITY GATE FAILED; no comparator training or test-driven retuning',flush=True)
            return
        for i in range(len(p['folds'])):
            for arm in ARMS[1:]:
                run_arm_fold(out,p,banks,y,arm,i,args.n_jobs,status)
                report(out,p)
        analysis=report(out,p)
        if not analysis['complete']:raise ValueError('Incomplete final results')
        status('final_verification',smoke=args.smoke,completed_fits=p['total_fits'])
        from audit_fourway_phonons import audit as audit_artifacts
        verification=audit_artifacts(out)
        if not verification['passed']:raise ValueError('Final verification failed')
        status('complete',smoke=args.smoke,completed_fits=p['total_fits'],means=analysis['full_fold_means'])
    except Exception:
        status('failed',traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
