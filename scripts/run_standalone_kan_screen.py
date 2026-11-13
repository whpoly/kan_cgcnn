"""Matched standalone KAN replacement screen, entirely within inner splits."""
from run_modnet_official_matched import SafePool,digest,file_hash,set_seed,write_json
from standalone_kan_models import CONFIGS,COUNTS,PHYSICAL_FEATURES,build_model,export_network,np,tf
from run_mattervial_kan_refinement import target_statistics
from pathlib import Path
from datetime import datetime,timezone
import argparse
import gc
import json
import os
import pickle
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'benchmarks/mattervial-kan-refinement-20261026'
SEEDS=(20261027,20271027)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()


def prepare(out):
    old=read(SOURCE/'protocol.json')
    for name,value in old['source_sha256'].items():
        if file_hash(ROOT/'scripts'/name)!=value:raise ValueError('Changed source '+name)
    sources=dict(old['source_sha256'])
    for name in ('standalone_kan_models.py','run_standalone_kan_screen.py'):
        sources[name]=file_hash(ROOT/'scripts'/name)
    for inp in old['source_inputs']:
        info=inp['source']
        if file_hash(inp['source_manifest'])!=inp['source_manifest_sha256']:raise ValueError('Changed manifest')
        if info['train_ids']!=info['feature_selection_fit_ids'] or info['train_ids']!=info['scaling_fit_ids']:
            raise ValueError('Preprocessing scope')
        if set(info['train_ids'])&set(info['validation_ids']):raise ValueError('Overlapping partitions')
        if len(info['selected_features'])!=256 or any('SISSO' in f.upper() for f in info['selected_features']):
            raise ValueError('Invalid features')
        for name,value in info['artifacts'].items():
            if file_hash(Path(info['directory'])/name)!=value:raise ValueError('Changed input')
    protocol=dict(kind='standalone KAN directly replaces the predictor; no residual trunk',
        source_protocol_sha256=file_hash(SOURCE/'protocol.json'),source_sha256=sources,source_inputs=old['source_inputs'],
        configs=CONFIGS,seeds=SEEDS,learning_rate=.001,max_epochs=400,batch_size=32,expected_parameters=COUNTS,
        early_stopping=old['early_stopping'],target_transform='inner TRAIN mean/std; physical-unit prediction',
        optimizer='Adam',loss='normalized-target MAE plus declared sparse/smooth penalties',
        topology=dict(enhanced_kan=[256,17,8,1],physical_kan=[16,17,8,1],
            modnet=[256,96,72,36,36,1],enhanced_mlp=[256,96,72,36,45,1],physical_mlp=[16,75,27,1]),
        layers='pure cubic spline edge functions plus node bias; no Dense/MLP/base-SiLU or residual path in KAN',
        hidden_coordinates='separate 0.5*tanh on each hidden coordinate, no LayerNorm or feature mixing',
        grid='fixed, 8 cubic basis functions, [-0.5,0.5]; out-of-range basis inputs clipped',
        sparse=dict(edge_coefficient_centered_rms_strength=.001,coefficient_second_difference_strength=.001,
            scope='fixed regularization ablation, not a second chance to select the best family result'),
        interpretation=dict(edge_activity='inner TRAIN response mean and standard deviation',
            prune='center each edge using TRAIN mean; remove edges with response std < 5% max within layer; compensate node bias',
            symbolic_library=['constant','linear','quadratic','cubic'],
            symbolic_tolerance='TRAIN response RMSE <= 5% of edge-response SD; otherwise retain spline',
            symbolic_coordinates='per-column scaled descriptors for layer 0; bounded learned coordinates for higher layers',
            audit='evaluate whole network after pruning/symbolic replacement on inner validation; no formula selection on validation'),
        fits=70,outer_test_evaluated=False,source_models_loaded=False,
        limitations=['same two seeds reused for paired comparisons, not new independent confirmation',
            'fixed common optimization budget, not exhaustive tuning of each method',
            'enhanced and physical banks differ in information; compare KAN to MLP within each bank',
            'KAN function graph is interpretable algebra; hidden coordinates and pretrained embeddings are not named physics',
            'iteratively reused inner validation remains exploratory; no outer benchmark claim'])
    path=out/'protocol.json'
    if path.exists() and digest(read(path))!=digest(protocol):raise ValueError('Changed protocol; new directory required')
    write_json(path,protocol);return protocol


def train_job(job):
    out=Path(job['directory']);out.mkdir(parents=True,exist_ok=True);signature=digest(job)
    if (out/'result.json').exists():
        r=read(out/'result.json')
        if r['signature']!=signature:raise ValueError('Changed job')
        for name,value in r['artifacts'].items():
            if file_hash(out/name)!=value:raise ValueError('Changed artifact')
        return r
    started=time.perf_counter()
    try:
        inp=job['input'];info=inp['source'];source=Path(info['directory'])
        if file_hash(inp['source_manifest'])!=inp['source_manifest_sha256'] or file_hash(source/'arrays.npz')!=info['artifacts']['arrays.npz']:
            raise ValueError('Changed inputs')
        features=info['selected_features'] if job['bank']=='enhanced' else PHYSICAL_FEATURES
        indices=[info['selected_features'].index(f) for f in features]
        with np.load(source/'arrays.npz') as a:
            x=a['x_train'][:,indices];xv=a['x_validation'][:,indices];y=a['y_train'];yv=a['y_validation']
        mean,std=target_statistics(y)
        if (mean,std)!=(inp['target_mean'],inp['target_std']):raise ValueError('Target scaling')
        model=build_model(job['bank'],job['variant'],job['seed'])
        model.compile(optimizer=tf.keras.optimizers.Adam(job['lr']),loss='mae',metrics=['mae'])
        class Best(tf.keras.callbacks.Callback):
            def __init__(self):super().__init__();self.value=float('inf');self.weights=None;self.epoch=0
            def on_epoch_end(self,epoch,logs=None):
                value=float(logs['val_mae'])
                if not np.isfinite(value):raise ValueError('Nonfinite validation')
                if value<self.value:self.value=value;self.weights=self.model.get_weights();self.epoch=epoch+1
                if (epoch+1)%50==0:write_json(out/'status.json',dict(state='training',epoch=epoch+1,best_epoch=self.epoch,best_mae=self.value*std,updated_at=now()))
        best=Best();set_seed(job['seed'])
        history=model.fit(x,((y-mean)/std).astype('float32'),validation_data=(xv,((yv-mean)/std).astype('float32')),
            epochs=job['max_epochs'],batch_size=32,verbose=0,
            callbacks=[best,tf.keras.callbacks.EarlyStopping(monitor='val_mae',patience=40,min_delta=.05/std)])
        model.set_weights(best.weights)
        pred=model.predict(xv,verbose=0).reshape(-1).astype(float)*std+mean
        train_pred=model.predict(x,verbose=0).reshape(-1).astype(float)*std+mean
        mae=float(np.mean(abs(pred-yv)))
        if not np.isfinite(pred).all() or abs(mae-best.value*std)>.003:raise ValueError('Checkpoint reproduction')
        np.savez_compressed(out/'predictions.npz',prediction=pred,target=yv,sample_ids=info['validation_ids'],
            train_prediction=train_pred,train_target=y,train_ids=info['train_ids'])
        with (out/'model.pkl').open('wb') as stream:
            pickle.dump(dict(model_json=model.to_json(),weights=model.get_weights(),target_mean=mean,target_std=std),stream,protocol=4)
        artifacts=['predictions.npz','model.pkl']
        if job['variant'] in ('kan','sparse_kan'):
            export=export_network(model,features,x)
            export.update(target_mean=mean,target_std=std)
            write_json(out/'network.json',export);artifacts.append('network.json')
        r=dict(signature=signature,run_fingerprint=job['run_fingerprint'],directory=str(out),bank=job['bank'],
            variant=job['variant'],repeat=job['repeat'],inner=job['inner'],seed=job['seed'],lr=job['lr'],
            input_digest=digest(inp),features=features,train_ids=info['train_ids'],validation_ids=info['validation_ids'],
            target_transform_fit_ids=info['train_ids'],target_mean=mean,target_std=std,parameters=model.count_params(),
            train_mae=float(np.mean(abs(train_pred-y))),validation_mae=mae,best_epoch=best.epoch,epochs=len(history.history['loss']),
            checkpoint_mae=best.value*std,history=history.history,seconds=time.perf_counter()-started,
            artifacts={name:file_hash(out/name) for name in artifacts})
        write_json(out/'result.json',r);write_json(out/'status.json',dict(state='complete',finished_at=now()))
        del model,best,history;tf.keras.backend.clear_session();gc.collect();return r
    except Exception:write_json(out/'status.json',dict(state='failed',updated_at=now(),traceback=traceback.format_exc()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n-jobs',type=int,default=2);args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    def status(state,**kwargs):write_json(out/'status.json',dict(state=state,pid=os.getpid(),updated_at=now(),**kwargs))
    try:
        status('preparing',completed=0,total=70);p=prepare(out);fp=digest(p)
        jobs=[dict(directory=str(out/'fits'/bank/variant/f'seed-{s}-inner-{i}'),bank=bank,variant=variant,repeat=s,inner=i,
            seed=seed+i,lr=p['learning_rate'],max_epochs=p['max_epochs'],expected_parameters=COUNTS[bank][variant],
            input=p['source_inputs'][i],run_fingerprint=fp)
            for s,seed in enumerate(SEEDS) for i in range(5) for bank,variant in CONFIGS]
        records=[]
        with SafePool(args.n_jobs) as pool:
            for r in pool.imap_unordered(train_job,jobs):
                records.append(r)
                rows=[]
                for s in range(2):
                    for bank,v in CONFIGS:
                        group=[r for r in records if r['repeat']==s and r['bank']==bank and r['variant']==v]
                        if group:rows.append(dict(bank=bank,variant=v,repeat=s,completed=len(group),mean_mae=float(np.mean([r['validation_mae'] for r in group]))))
                write_json(out/'progress-summary.json',dict(completed=len(records),rows=rows))
                status('training',completed=len(records),total=70,last_bank=r['bank'],last_variant=r['variant'],last_mae=r['validation_mae'])
                print('FIT',len(records),'/70',r['bank'],r['variant'],'seed',r['repeat'],'inner',r['inner'],'MAE',r['validation_mae'],'epoch',r['best_epoch'],'/',r['epochs'],flush=True)
        status('complete',completed=70,total=70,wall_seconds=time.perf_counter()-start,outer_test_evaluated=False)
        print('COMPLETE',out,flush=True)
    except Exception:status('failed',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
