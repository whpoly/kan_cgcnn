"""80-fit inner-only screen of direct physical KANs and matched MLP controls."""
from run_modnet_official_matched import SafePool, digest, file_hash, set_seed, write_json
from interpretable_phonons_kan import (VARIANTS, PAIRS, PHYSICAL_FEATURES, SHORT_NAMES,
    expected_parameters, build_model, decompose, np, tf)
from run_mattervial_kan_refinement import target_statistics
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import json
import os
import pickle
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'benchmarks/mattervial-kan-refinement-20261026'
SEEDS = (20261027, 20271027)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(output):
    previous = read(SOURCE/'protocol.json')
    for name, value in previous['source_sha256'].items():
        if file_hash(ROOT/'scripts'/name) != value:
            raise ValueError('Changed previous source: '+name)
    sources = dict(previous['source_sha256'])
    for name in ('interpretable_phonons_kan.py', 'run_interpretable_phonons_kan.py'):
        sources[name] = file_hash(ROOT/'scripts'/name)
    for inp in previous['source_inputs']:
        if file_hash(inp['source_manifest']) != inp['source_manifest_sha256']:
            raise ValueError('Input manifest changed')
        info = inp['source']
        if (info['train_ids'] != info['feature_selection_fit_ids'] or
            info['train_ids'] != info['scaling_fit_ids'] or
            set(info['train_ids']) & set(info['validation_ids']) or
            any('SISSO' in f.upper() for f in info['selected_features'])):
            raise ValueError('Invalid input scope')
        for name, value in info['artifacts'].items():
            if file_hash(Path(info['directory'])/name) != value:
                raise ValueError('Changed source artifact')
    references = []
    for repeat in range(2):
        for inner in range(5):
            p = SOURCE/'fits/modnet'/f'lr-0.001-seed-{repeat}-inner-{inner}'/'result.json'
            record = read(p)
            for name, value in record['artifacts'].items():
                if file_hash(p.parent/name) != value:
                    raise ValueError('Baseline artifact changed')
            references.append(dict(path=str(p), sha256=file_hash(p), inner=inner, repeat=repeat,
                                   validation_mae=record['validation_mae'], parameters=record['parameters']))
    protocol = dict(kind='direct physical interpretable KAN inner-only exploration',
        source_protocol_sha256=file_hash(SOURCE/'protocol.json'), source_sha256=sources,
        source_inputs=previous['source_inputs'], baseline_references=references,
        variants=VARIANTS, seeds=SEEDS, learning_rate=.001, max_epochs=400, batch_size=32,
        early_stopping=previous['early_stopping'], parameters={v: expected_parameters(v) for v in VARIANTS},
        physical_features=PHYSICAL_FEATURES, physical_names=SHORT_NAMES, pairs=PAIRS,
        preprocessing='unchanged 256 train-selected no-SISSO features; physical models only see the fixed 16 named columns',
        same_inputs_within_comparison=True, target_scaling='inner TRAIN mean/std only',
        centering='TRAIN marginal spline/linear basis means; TRAIN product means; fixed constants, no target dependence',
        implementation='custom TensorFlow cubic additive and rank-one product KAN; not upstream pykan',
        capacity_pairs={'additive': 'mlp8', 'product': 'mlp12'},
        optimization='Adam fixed .001, normalized MAE, no sparsity/roughness penalty, best inner-val checkpoint',
        interpretation=dict(raw_coordinate_axes=True, exact_components=True, centered_on_train=True,
            ablation='zero each centered term; report train and validation changes, not causal importance',
            masking_threshold=.05, masking_rule='mask terms with TRAIN mean absolute contribution < 5% of max; no retraining',
            symbolic_library=['constant','linear','quadratic','cubic','log(u)','u^-0.5','u^-1'],
            symbolic_rule='fit edge responses on TRAIN coordinates only; simplest within 5% response-SD RMSE, else retain spline',
            cross_fold_curve_grid='intersection of TRAIN 5th-95th percentile ranges, in raw units',
            stability='curve correlations and top-feature overlap across folds/seeds; no significance claim'),
        fits=80, outer_test_evaluated=False,
        limitations=['validation reused for stopping; exploratory, no independent test',
            'physical-only and 256-feature hybrid have different information; capacity claims only within matched pairs',
            'hybrid decomposes only correction; backbone is opaque and decomposition is not unique',
            'products are predeclared hypotheses, not discovered interactions',
            'correlated descriptors and proxy bonding inputs cannot establish causal laws',
            'same two earlier seeds used to pair prior baselines, not fresh seed confirmation'],
        references=['https://arxiv.org/abs/2408.10205','https://github.com/KindXiaoming/pykan',
                    'https://phonopy.github.io/phonopy/formulation.html#dynamical-matrix'])
    path = output/'protocol.json'
    if path.exists() and digest(read(path)) != digest(protocol):
        raise ValueError('Protocol changed; use a new directory')
    write_json(path, protocol)
    return protocol


def train_job(job):
    out = Path(job['directory']); out.mkdir(parents=True, exist_ok=True)
    signature = digest(job)
    if (out/'result.json').exists():
        cached = read(out/'result.json')
        if cached['signature'] != signature:
            raise ValueError('Changed job')
        for name, value in cached['artifacts'].items():
            if file_hash(out/name) != value:
                raise ValueError('Changed output artifact')
        return cached
    started = time.perf_counter()
    try:
        inp = job['input']; src = inp['source']; source = Path(src['directory'])
        if file_hash(inp['source_manifest']) != inp['source_manifest_sha256'] or file_hash(source/'arrays.npz') != src['artifacts']['arrays.npz']:
            raise ValueError('Changed input')
        with np.load(source/'arrays.npz') as a:
            x, xv, y, yv = [a[k] for k in ('x_train','x_validation','y_train','y_validation')]
        mean, scale = target_statistics(y)
        if (mean, scale) != (inp['target_mean'], inp['target_std']):
            raise ValueError('Wrong target scope')
        model = build_model(job['variant'], src['selected_features'], x, job['seed'])
        if model.count_params() != job['expected_parameters']:
            raise ValueError('Wrong model capacity')
        model.compile(optimizer=tf.keras.optimizers.Adam(job['lr']), loss='mae', metrics=['mae'])
        class Best(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__(); self.value=float('inf'); self.weights=None; self.epoch=0
            def on_epoch_end(self, epoch, logs=None):
                value=float(logs['val_mae'])
                if not np.isfinite(value): raise ValueError('Nonfinite validation')
                if value < self.value:
                    self.value=value; self.weights=self.model.get_weights(); self.epoch=epoch+1
                if (epoch+1)%50 == 0:
                    write_json(out/'status.json',dict(state='training',epoch=epoch+1,best_epoch=self.epoch,
                        best_mae=self.value*scale,updated_at=now()))
        best = Best(); set_seed(job['seed'])
        history = model.fit(x, ((y-mean)/scale).astype('float32'),
            validation_data=(xv, ((yv-mean)/scale).astype('float32')),
            epochs=job['max_epochs'], batch_size=32, verbose=0,
            callbacks=[best,tf.keras.callbacks.EarlyStopping(monitor='val_mae',patience=40,min_delta=.05/scale)])
        model.set_weights(best.weights)
        train_prediction=model.predict(x,verbose=0).reshape(-1).astype(float)*scale+mean
        prediction=model.predict(xv,verbose=0).reshape(-1).astype(float)*scale+mean
        mae=float(np.mean(abs(prediction-yv)))
        if not np.isfinite(prediction).all() or abs(mae-best.value*scale) > .003:
            raise ValueError('Checkpoint reproduction mismatch')
        layer=model.get_layer('physical_response')
        exported=layer.export()
        indices=[src['selected_features'].index(f) for f in PHYSICAL_FEATURES]
        with (source/'scaler.pkl').open('rb') as stream: scaler=pickle.load(stream)
        physical_train=(x[:,indices]-scaler.min_[indices])/scaler.scale_[indices]
        physical_validation=(xv[:,indices]-scaler.min_[indices])/scaler.scale_[indices]
        extra=dict(x_train=x[:,indices],x_validation=xv[:,indices],physical_train=physical_train,
            physical_validation=physical_validation,scaler_scale=scaler.scale_[indices],scaler_min=scaler.min_[indices],
            train_ids=src['train_ids'],train_target=y,train_prediction=train_prediction)
        diagnostics={}
        if exported is not None:
            terms,bias=decompose(x[:,indices],exported)
            terms_v,_=decompose(xv[:,indices],exported)
            actual=tf.keras.Model(model.input,layer.output).predict(xv,verbose=0).reshape(-1)
            discrepancy=float(np.max(abs(actual-terms_v.sum(axis=1)-bias))*scale)
            if discrepancy > .003: raise ValueError('Component decomposition mismatch')
            terms*=scale; terms_v*=scale; bias*=scale
            train_backbone=train_prediction-terms.sum(axis=1)-bias
            validation_backbone=prediction-terms_v.sum(axis=1)-bias
            if job['variant'].startswith('physical_'):
                np.testing.assert_allclose(train_backbone,mean,atol=.003,rtol=0)
            else:
                diagnostics['gate']=exported['gate']
            extra.update(train_terms=terms,validation_terms=terms_v,bias=bias,
                         train_backbone=train_backbone,validation_backbone=validation_backbone)
            importance=np.mean(abs(terms),axis=0)
            mask=importance >= .05*importance.max()
            masked=validation_backbone+bias+terms_v[:,mask].sum(axis=1)
            diagnostics.update(decomposition_max_error=discrepancy,train_contribution_abs_mean=importance.tolist(),
                retained_terms=mask.tolist(),masked_validation_mae=float(np.mean(abs(masked-yv))),
                removed_term_mae_delta=(np.mean(abs((prediction[:,None]-terms_v)-yv[:,None]),axis=0)-mae).tolist(),
                backbone_without_branch_mae=float(np.mean(abs(validation_backbone-yv))))
            exported.update(target_mean=mean,target_std=scale)
            write_json(out/'edge-functions.json',exported)
        np.savez_compressed(out/'predictions.npz', sample_ids=src['validation_ids'], target=yv, prediction=prediction, **extra)
        with (out/'model.pkl').open('wb') as stream:
            pickle.dump(dict(model_json=model.to_json(),weights=model.get_weights(),target_mean=mean,target_std=scale),stream,protocol=4)
        artifacts=['predictions.npz','model.pkl']+(['edge-functions.json'] if exported is not None else [])
        record=dict(signature=signature,run_fingerprint=job['run_fingerprint'],directory=str(out),
            variant=job['variant'],inner=job['inner'],repeat=job['repeat'],seed=job['seed'],lr=job['lr'],
            input_digest=digest(inp),parameters=model.count_params(),train_ids=src['train_ids'],validation_ids=src['validation_ids'],
            target_transform_fit_ids=src['train_ids'],target_mean=mean,target_std=scale,
            validation_mae=mae,train_mae=float(np.mean(abs(train_prediction-y))),best_epoch=best.epoch,
            epochs=len(history.history['loss']),history=history.history,checkpoint_mae=best.value*scale,
            diagnostics=diagnostics,seconds=time.perf_counter()-started,artifacts={name:file_hash(out/name) for name in artifacts})
        write_json(out/'result.json',record)
        write_json(out/'status.json',dict(state='complete',finished_at=now()))
        del model,history,best;tf.keras.backend.clear_session();gc.collect()
        return record
    except Exception:
        write_json(out/'status.json',dict(state='failed',updated_at=now(),traceback=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n-jobs',type=int,default=2)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    def status(state,**kwargs):
        write_json(out/'status.json',dict(state=state,pid=os.getpid(),updated_at=now(),**kwargs))
    try:
        status('preparing',completed=0,total=80)
        protocol=prepare(out);fp=digest(protocol)
        jobs=[dict(directory=str(out/'fits'/variant/f'seed-{repeat}-inner-{inner}'),variant=variant,
            inner=inner,repeat=repeat,seed=seed+inner,lr=protocol['learning_rate'],
            max_epochs=protocol['max_epochs'],expected_parameters=protocol['parameters'][variant],
            input=protocol['source_inputs'][inner],run_fingerprint=fp)
            for repeat,seed in enumerate(SEEDS) for inner in range(5) for variant in VARIANTS]
        records=[]
        with SafePool(args.n_jobs) as pool:
            for r in pool.imap_unordered(train_job,jobs):
                records.append(r)
                write_json(out/'progress-summary.json',dict(completed=len(records),
                    rows=[dict(variant=v,repeat=s,completed=sum(r['variant']==v and r['repeat']==s for r in records),
                        mean_mae=float(np.mean([r['validation_mae'] for r in records if r['variant']==v and r['repeat']==s])))
                        for s in range(2) for v in VARIANTS if any(r['variant']==v and r['repeat']==s for r in records)]))
                status('training',completed=len(records),total=80,last_variant=r['variant'],last_mae=r['validation_mae'])
                print('FIT',len(records),'/80',r['variant'],'seed',r['repeat'],'inner',r['inner'],
                      'MAE',r['validation_mae'],'epoch',r['best_epoch'],'/',r['epochs'],flush=True)
        status('complete',completed=80,total=80,wall_seconds=time.perf_counter()-start,outer_test_evaluated=False)
        print('COMPLETE',out,flush=True)
    except Exception:
        status('failed',traceback=traceback.format_exc());raise


if __name__=='__main__': main()
