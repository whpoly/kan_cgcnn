"""Inner-only paired target-scale diagnostic; never changes the formal run."""
from run_modnet_official_matched import np,pd,tf,MODData,SafePool,set_seed,digest,file_hash,write_json
from native_fourway_models import NativeSplineModel
from standalone_kan_models import SplineEdgeLayer
from official_kan_models import neurons_from_genes
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import argparse
import contextlib
import gc
import json
import pickle
import os
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
CURRENT=ROOT/'benchmarks/native-fourway-20261029'
PREVIOUS=ROOT/'benchmarks/standalone-kan-fivefold-20261028'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def freeze(out):
    path=out/'protocol.json'
    if path.exists():return read(path)
    groups=defaultdict(list)
    for f in (CURRENT/'mm_kan/fold-0/fits').glob('*/result.json'):
        r=read(f);groups[(r['generation'],r['candidate'])].append(r)
    choices=[(np.mean([r['validation_mae'] for r in rows]),key,rows) for key,rows in groups.items() if len(rows)==5]
    _,key,rows=min(choices,key=lambda x:(x[0],x[1]))
    rows=sorted(rows,key=lambda r:r['inner'])
    files=[Path(__file__),ROOT/'scripts/native_fourway_models.py',ROOT/'scripts/standalone_kan_models.py',
           ROOT/'scripts/physical_feature_modules.py',ROOT/'scripts/run_modnet_official_matched.py']
    cache=CURRENT/'inputs/mm/fold-0/training.pkl.gz'
    p=dict(created_at=datetime.now(timezone.utc).isoformat(),kind='Paired inner-only B-spline target-scale diagnostic',
        selected_candidate=list(key),source_members=[dict(path=str(Path(r['directory'])/'result.json'),
        sha256=file_hash(Path(r['directory'])/'result.json')) for r in rows],
        selection='Best candidate with all5 inner scores at diagnosis start. Inner-selected, not independent validation.',
        cache=str(cache),cache_sha256=file_hash(cache),source_sha256={str(f):file_hash(f) for f in files},
        current_protocol_sha256=file_hash(CURRENT/'protocol.json'),formal_run_modified=False,
        fits=6,modes='Raw-scale execution replay on inner0; train-target-standardization on all5 inners.',
        paired='Same raw features/ranking, input scaler, seed, architecture, LR, batch, Adam,800epoch cap and train-loss stopping. min_delta0.001 physical units in both. No best-val checkpoint.',
        scope='Only original first outer-training1012 rows. No outer prediction or target inspection. Frozen formal benchmark continues unchanged.')
    write_json(path,p);return p


def data_for(p,r):
    assert file_hash(p['cache'])==p['cache_sha256']
    data=MODData.load(p['cache']);lookup={sid:i for i,sid in enumerate(data.df_targets.index)}
    assert set(r['train_ids']).isdisjoint(r['validation_ids'])
    assert set(r['train_ids']+r['validation_ids'])==set(lookup)
    return data.split(([lookup[s] for s in r['train_ids']],[lookup[s] for s in r['validation_ids']]))


def prediction(model,data):
    x=data.df_featurized[model.optimal_descriptors[:model.n_feat]].replace([np.inf,-np.inf,np.nan],0).values
    x=np.nan_to_num(model._scaler.transform(np.nan_to_num(x)),nan=-1)
    return model.model.predict(x,verbose=0).reshape(-1).astype(float),x


def inspect_model(model,x,y,mean=0.,std=1.):
    values=np.asarray(x,dtype='float32');layers=[]
    for layer in model.model.layers:
        if not isinstance(layer,SplineEdgeLayer):continue
        coords=.5*np.tanh(values) if layer.hidden_coordinates else values
        layers.append(dict(name=layer.name,input_dim=int(values.shape[1]),
            saturated_fraction=float(np.mean(abs(coords)>.49)),coordinate_std_mean=float(np.std(coords,axis=0).mean()),
            coefficient_abs_max=float(np.max(abs(layer.coefficients.numpy())))))
        values=layer(values).numpy()
    last=[l for l in model.model.layers if isinstance(l,SplineEdgeLayer)][-1]
    coeff=last.coefficients.numpy()[:,0,:];bias=float(last.bias.numpy()[0])
    low=float(coeff.min(axis=1).sum()+bias)*std+mean
    high=float(coeff.max(axis=1).sum()+bias)*std+mean
    yp=values.reshape(-1).astype(float)*std+mean
    return dict(layers=layers,learned_output_lower_bound=low,learned_output_upper_bound=high,
        train_target_above_bound_fraction=float(np.mean(y>high)),train_target_below_bound_fraction=float(np.mean(y<low)),
        train_target_mean=float(y.mean()),train_target_std=float(y.std()),train_target_max=float(y.max()),
        prediction_mean=float(yp.mean()),prediction_std=float(yp.std()),prediction_max=float(yp.max()),
        train_mae=float(np.mean(abs(yp-y))),constant_median_train_mae=float(np.mean(abs(y-np.median(y)))))


def inspect_saved(out,p):
    rows=[]
    for source in p['source_members']:
        assert file_hash(source['path'])==source['sha256'];r=read(source['path'])
        assert file_hash(Path(r['directory'])/'model.pkl')==r['model_sha256']
        tr,va=data_for(p,r)
        with (Path(r['directory'])/'model.pkl').open('rb') as f:model=pickle.load(f)
        model._restore_model();yp,x=prediction(model,tr);pv,_=prediction(model,va)
        mae=float(np.mean(abs(pv-va.df_targets.values[:,0])))
        assert abs(mae-r['validation_mae'])<.001
        stats=inspect_model(model,x,tr.df_targets.values[:,0]);stats.update(inner=r['inner'],seed=r['seed'],
            validation_mae=mae,epochs=r['epochs'],genes=r['genes'],widths=r['spline_widths'],parameters=r['parameters'],
            history_first_train_loss=r['history']['loss'][0],history_final_train_loss=r['history']['loss'][-1],
            history_best_validation_loss=min(r['history']['val_loss']))
        rows.append(stats);del model;tf.keras.backend.clear_session();gc.collect()
    write_json(out/'raw-model-diagnostics.json',dict(rows=rows,mean_validation_mae=float(np.mean([r['validation_mae'] for r in rows]))))
    return rows


def train_job(job):
    p=read(job['protocol']);fp=digest(p)
    for path,sha in p['source_sha256'].items():assert file_hash(path)==sha
    out=Path(job['directory']);out.mkdir(parents=True,exist_ok=True);done=out/'result.json'
    if done.exists():
        r=read(done);assert r['job_signature']==digest(job);return r
    source=p['source_members'][job['inner']];assert file_hash(source['path'])==source['sha256'];r=read(source['path'])
    tr,va=data_for(p,r);y=tr.df_targets.values[:,0].copy();yv=va.df_targets.values[:,0].copy()
    mean,std=(float(y.mean()),float(y.std())) if job['mode']=='standardized' else (0.,1.)
    tr.df_targets=tr.df_targets.copy();va.df_targets=va.df_targets.copy()
    tr.df_targets.iloc[:,0]=(y-mean)/std;va.df_targets.iloc[:,0]=(yv-mean)/std
    g=r['genes'];seed=r['seed'];tf.keras.backend.clear_session();set_seed(seed)
    model=NativeSplineModel(targets=[[list(tr.target_names)]],weights={n:1 for n in tr.target_names},
        num_classes=tr.num_classes,n_feat=g['n_feat'],num_neurons=neurons_from_genes(g),act=g['act'],
        spline_seed=seed,input_scale=g['xscale'])
    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            if (epoch+1)%50==0:write_json(out/'status.json',dict(epoch=epoch+1,mode=job['mode']))
    set_seed(seed);start=time.perf_counter()
    with (out/'training.log').open('w',encoding='utf-8') as f:
        with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            model.fit(tr,val_data=va,loss=g['loss'],lr=g['lr'],epochs=800,batch_size=g['batch_size'],xscale=g['xscale'],
                verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss',patience=30,min_delta=.001/std,
                    restore_best_weights=False),Progress()])
    pv,_=prediction(model,va);pv=pv*std+mean;_,x=prediction(model,tr)
    mae=float(np.mean(abs(pv-yv)));stats=inspect_model(model,x,y,mean,std)
    if job['mode']=='raw_replay':assert abs(mae-r['validation_mae'])<.01
    result=dict(job_signature=digest(job),run_fingerprint=fp,mode=job['mode'],inner=job['inner'],seed=seed,genes=g,
        validation_mae=mae,original_raw_mae=r['validation_mae'],target_mean=mean,target_std=std,
        epochs=len(model.history['loss']),history=model.history,diagnostics=stats,seconds=time.perf_counter()-start,
        train_ids=r['train_ids'],validation_ids=r['validation_ids'],features=r['selected_features'])
    np.savez_compressed(out/'validation.npz',prediction=pv,target=yv,sample_ids=r['validation_ids'])
    model._make_picklable()
    with (out/'model.pkl').open('wb') as f:pickle.dump(model,f,protocol=4)
    result['artifacts']={n:file_hash(out/n) for n in ('model.pkl','validation.npz')}
    write_json(done,result);del model;tf.keras.backend.clear_session();gc.collect();return result


def report(out,p):
    raw=read(out/'raw-model-diagnostics.json')
    normalized=[read(path) for path in sorted((out/'standardized').glob('inner-*/result.json'))]
    complete=len(normalized)==5
    result=dict(complete=complete,normalized_fits=len(normalized),selected_candidate=p['selected_candidate'],
        original_raw_mean_mae=raw['mean_validation_mae'],normalized_mean_mae=float(np.mean([r['validation_mae'] for r in normalized])) if normalized else None,
        rows=[dict(inner=r['inner'],raw_mae=r['original_raw_mae'],normalized_mae=r['validation_mae'],
              normalized_train_mae=r['diagnostics']['train_mae'],epochs=r['epochs']) for r in normalized],
        scope='Single inner-selected configuration, paired5 inners; not an outer accuracy claim; formal run unchanged')
    write_json(out/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    def status(state,**kw):write_json(out/'status.json',dict(state=state,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),**kw))
    try:
        p=freeze(out);status('inspecting');inspect_saved(out,p)
        jobs=[dict(protocol=str(out/'protocol.json'),directory=str(out/'raw_replay/inner-0'),mode='raw_replay',inner=0)]
        jobs += [dict(protocol=str(out/'protocol.json'),directory=str(out/'standardized'/f'inner-{i}'),mode='standardized',inner=i) for i in range(5)]
        status('training',completed=0,total=len(jobs))
        with SafePool(2) as pool:
            for count,r in enumerate(pool.imap_unordered(train_job,jobs),1):
                report(out,p);status('training',completed=count,total=len(jobs))
                print('FIT',count,r['mode'],r['inner'],r['validation_mae'],flush=True)
        result=report(out,p);assert result['complete'];status('complete',completed=6,total=6)
        print(result,flush=True)
    except BaseException:status('failed',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
