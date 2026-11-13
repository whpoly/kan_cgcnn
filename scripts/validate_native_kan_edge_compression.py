"""Training-only polynomial compression of frozen KAN edges, not law discovery."""
import argparse
import gc
import os
import pickle
import traceback
from pathlib import Path
import analyze_native_physical_responses as a
from run_modnet_official_matched import np,tf,file_hash,digest,write_json
from physical_feature_modules import cubic_basis
from standalone_kan_models import SplineEdgeLayer
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[1]

def coordinates(values,hidden):
    return np.clip(.5*np.tanh(values) if hidden else values,-.5,.5-1e-6)

def fit_polynomials(observations,degree,indices):
    result=[]
    for layer in observations:
        coords=layer['coordinates'][indices];basis=layer['basis'][indices];original=layer['coefficients']
        coefficients=np.empty((*original.shape[:2],degree+1))
        for j in range(coords.shape[1]):
            design=np.vander(2*coords[:,j],degree+1,increasing=True)
            transform=np.linalg.lstsq(design,basis[:,j,:],rcond=None)[0]
            coefficients[j]=(transform@original[j].T).T
        result.append(dict(coefficients=coefficients,bias=layer['bias'],hidden=layer['hidden']))
    return result

def predict_polynomials(layers,x,scale):
    values=np.asarray(x,dtype=float)
    for layer in layers:
        c=coordinates(values,layer['hidden']);degree=layer['coefficients'].shape[-1]-1
        powers=np.stack([(2*c)**k for k in range(degree+1)],axis=-1)
        values=np.einsum('nik,iok->no',powers,layer['coefficients'])+layer['bias']
    return values.reshape(-1)*scale['std']+scale['mean']

def compress_member(model,train,member_directory,seed,protocol):
    cols=model.optimal_descriptors[:model.n_feat]
    x=np.nan_to_num(model._scaler.transform(np.nan_to_num(train[cols].values)),nan=-1)
    values=x.astype('float32');observations=[]
    for layer in model.model.layers:
        if not isinstance(layer,SplineEdgeLayer):continue
        c=coordinates(values,layer.hidden_coordinates)
        # Use the actual TensorFlow basis and teacher trajectory for fitting.
        basis=cubic_basis(tf.convert_to_tensor(c)).numpy()
        observations.append(dict(coordinates=c,basis=basis,coefficients=layer.coefficients.numpy(),bias=layer.bias.numpy(),hidden=layer.hidden_coordinates))
        values=layer(values).numpy()
    assert len(observations)==3
    teacher=values.reshape(-1)*model.target_scaling['std']+model.target_scaling['mean']
    actual=a.predict(model,train[cols].values)
    np.testing.assert_allclose(teacher,actual,rtol=0,atol=.002)
    splits=list(KFold(n_splits=5,shuffle=True,random_state=seed).split(x))
    scores=[]
    for degree in [1,2,3]:
        losses=[]
        for tr,va in splits:
            fitted=fit_polynomials(observations,degree,tr)
            pred=predict_polynomials(fitted,x[va],model.target_scaling)
            losses.append(float(np.mean(abs(pred-actual[va]))))
        scores.append(dict(degree=degree,fold_teacher_mae=losses,mean_teacher_mae=float(np.mean(losses))))
    tolerance=.01*float(np.std(actual))
    accepted=[s['degree'] for s in scores if s['mean_teacher_mae']<=tolerance]
    degree=min(accepted) if accepted else None
    fitted=fit_polynomials(observations,degree,np.arange(len(x))) if degree is not None else None
    original_count=sum(o['coefficients'].size+o['bias'].size for o in observations)
    count=sum(o['coefficients'].size+o['bias'].size for o in fitted) if fitted else original_count
    recipe=dict(protocol_fingerprint=digest(protocol),degree=degree,kind='polynomial_composition' if fitted else 'original_spline_fallback',
        cv=scores,tolerance=tolerance,tolerance_rule='1 percent of frozen teacher prediction standard deviation on outer train',
        features=list(cols),input_scaler_type=type(model._scaler).__name__,target_scaling=model.target_scaling,
        original_coefficient_entries=original_count,compressed_coefficient_entries=count,
        equation='For each layer: output[o]=bias[o]+sum_i sum_k coefficient[i,o,k]*(2*clip(coordinate[i],-0.5,0.5-1e-6))**k. Hidden coordinates use 0.5*tanh(previous output). Retain original input scaler and inverse target scaling.',
        selection_uses_outer_targets=False)
    if fitted:
        arrays={}
        for i,layer in enumerate(fitted):
            arrays[f'coefficients_{i}']=layer['coefficients'];arrays[f'bias_{i}']=layer['bias'];arrays[f'hidden_{i}']=np.array(layer['hidden'])
        np.savez_compressed(member_directory/'polynomial-coefficients.npz',**arrays)
        recipe['coefficient_sha256']=file_hash(member_directory/'polynomial-coefficients.npz')
    write_json(member_directory/'frozen-recipe.json',recipe)
    return fitted,recipe

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    run=args.run.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__).resolve(),ROOT/'scripts/analyze_native_physical_responses.py',ROOT/'scripts/physical_feature_modules.py',ROOT/'scripts/standalone_kan_models.py']
    p=dict(run=str(run),smoke=args.smoke,degrees=[1,2,3],selection='Lowest degree with mean 5-fold teacher-approximation MAE <= 1% outer-train teacher SD; otherwise retain original model.',seed=20261101,
        source_sha256={str(f):file_hash(f) for f in sources},parent_audit_sha256=file_hash(run/'artifact-audit.json'),
        limitations=['Post-hoc compression of already trained models; not a new equal-budget benchmark.',
            'Teacher CV measures approximation, not independent target prediction. Teachers saw outer-training labels.',
            'Hidden coordinates are learned latent variables, not identified physical quantities.',
            'Input scaling, clipping and tanh remain; a composition of polynomials is not a simple physical law.',
            'No matched MLP compression control in this stage; cannot establish a unique KAN interpretability advantage.',
            'No new choice uses outer target performance; negative sparse-descriptor result is retained separately.'])
    if (out/'protocol.json').exists():assert a.engine.read(out/'protocol.json')==p
    write_json(out/'protocol.json',p);banks,y=a.engine.load_banks(out);rows=[]
    arms=['mm_kan'] if args.smoke else ['mm_kan','mm_mv_kan']
    for arm in arms:
        for fold in (range(1) if args.smoke else range(5)):
            d=out/arm/f'fold-{fold}';d.mkdir(parents=True,exist_ok=True)
            if (d/'complete.json').exists():
                row=a.engine.read(d/'complete.json');assert row['protocol_fingerprint']==digest(p)
                assert file_hash(d/'outer-predictions.npz')==row['prediction_sha256'];rows.append(row);continue
            original=run/arm/f'fold-{fold}';selection=a.engine.read(original/'frozen-selection.json')
            original_complete=a.engine.read(original/'complete.json')
            assert file_hash(original/'frozen-selection.json')==original_complete['selection_sha256']
            assert file_hash(original/'outer-predictions.npz')==original_complete['prediction_sha256']
            split=a.engine.read(a.engine.OLD/'matbench_phonons'/f'fold-{fold}'/'outer-split.json')
            frame=banks[arm.rsplit('_',1)[0]];train=frame.loc[split['train_ids']];test=frame.loc[split['test_ids']]
            jobs=selection['members'][:1] if args.smoke else selection['members']
            with np.load(original/'outer-predictions.npz') as z:reference=z['members'][:len(jobs)].copy()
            predictions=[];recipes=[]
            for i,job in enumerate(jobs):
                md=d/f'member-{i:02d}';md.mkdir(exist_ok=True)
                write_json(out/'status.json',dict(state='compressing',pid=os.getpid(),arm=arm,fold=fold,member=i+1,members=len(jobs)))
                path=Path(job['directory'])/'model.pkl';assert file_hash(path)==selection['model_sha256'][job['directory']]
                with path.open('rb') as f:model=pickle.load(f)
                model._restore_model();fitted,recipe=compress_member(model,train,md,p['seed']+fold*1000+i,p)
                recipe['source_model_sha256']=file_hash(path);recipe['source_model']=str(path);recipe['train_ids']=split['train_ids']
                write_json(md/'frozen-recipe.json',recipe)
                cols=model.optimal_descriptors[:model.n_feat];teacher=a.predict(model,test[cols].values)
                np.testing.assert_allclose(teacher,reference[i],rtol=0,atol=.002)
                if fitted:
                    x=np.nan_to_num(model._scaler.transform(np.nan_to_num(test[cols].values)),nan=-1)
                    pred=predict_polynomials(fitted,x,model.target_scaling)
                    with np.load(md/'polynomial-coefficients.npz') as z:
                        restored=[dict(coefficients=z[f'coefficients_{j}'],bias=z[f'bias_{j}'],hidden=bool(z[f'hidden_{j}'])) for j in range(3)]
                        np.testing.assert_array_equal(pred,predict_polynomials(restored,x,model.target_scaling))
                else:pred=reference[i].copy()
                assert np.isfinite(pred).all();predictions.append(pred);recipes.append(recipe)
                del model,fitted;tf.keras.backend.clear_session();gc.collect()
            matrix=np.asarray(predictions);prediction=matrix.mean(axis=0);teacher=reference.mean(axis=0)
            # No outer targets used until all member recipes have been frozen.
            target=y.loc[split['test_ids']].values[:,0]
            np.savez_compressed(d/'outer-predictions.npz',sample_ids=np.array(split['official_test_ids']),prediction=prediction,teacher=teacher,target=target,members=matrix)
            row=dict(arm=arm,fold=fold,smoke=args.smoke,protocol_fingerprint=digest(p),mae=float(np.mean(abs(prediction-target))),teacher_mae=float(np.mean(abs(teacher-target))),
                teacher_approximation_mae=float(np.mean(abs(prediction-teacher))),degree_counts={str(k):sum(r['degree']==k for r in recipes) for k in [None,1,2,3]},
                original_entries=sum(r['original_coefficient_entries'] for r in recipes),compressed_entries=sum(r['compressed_coefficient_entries'] for r in recipes),
                prediction_sha256=file_hash(d/'outer-predictions.npz'),recipe_sha256={str(md):file_hash(md) for md in sorted(d.glob('member-*/frozen-recipe.json'))})
            write_json(d/'complete.json',row);rows.append(row)
    means={arm:{k:float(np.mean([r[k] for r in rows if r['arm']==arm])) for k in ['mae','teacher_mae','teacher_approximation_mae']} for arm in arms if not args.smoke and sum(r['arm']==arm for r in rows)==5}
    write_json(out/'analysis.json',dict(complete=len(rows)==10,smoke=args.smoke,rows=rows,means=means,limitations=p['limitations']))
    write_json(out/'status.json',dict(state='complete',pid=os.getpid(),smoke=args.smoke,fold_arm_pairs=len(rows)))

if __name__=='__main__':
    try:main()
    except BaseException:
        import sys
        if '--output' in sys.argv:write_json(Path(sys.argv[sys.argv.index('--output')+1])/'status.json',dict(state='failed',pid=os.getpid(),traceback=traceback.format_exc()))
        raise
