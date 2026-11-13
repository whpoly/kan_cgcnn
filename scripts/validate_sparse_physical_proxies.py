"""Post-hoc sparse physical proxies of frozen KAN and MLP teachers.

Select proxy complexity using only outer-training teacher predictions. Outer
labels are used only to score an already frozen formula. This is exploratory
distillation, not a new 5000-fit benchmark or independent physical validation.
"""
import argparse
import gc
import json
import os
import pickle
import traceback
from pathlib import Path
import analyze_native_physical_responses as ale
from run_modnet_official_matched import np, pd, tf, file_hash, digest, write_json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LassoLars
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
ALPHAS=[.0001,.0003,.001,.003,.01,.03,.1,.3,1.]

class PhysicalDictionary(BaseEstimator,TransformerMixin):
    def fit(self,x,y=None):
        x=np.asarray(x,dtype=float);assert np.isfinite(x).all()
        self.terms_=[]
        for j,label in enumerate(ale.FEATURES):
            x0=float(np.median(x[:,j]));x0=x0 if abs(x0)>1e-12 else 1.
            families=['linear','quadratic']
            if np.all(x[:,j]>0):families+=['log','inverse','inverse_sqrt']
            floor=float(x[:,j].min()*1e-6) if np.all(x[:,j]>0) else None
            for family in families:self.terms_.append(dict(feature=label,index=j,family=family,x0=x0,floor=floor))
        return self

    def transform(self,x):
        x=np.asarray(x,dtype=float);columns=[]
        for term in self.terms_:
            raw=x[:,term['index']];family=term['family']
            u=raw/term['x0']
            if family not in ('linear','quadratic'):u=np.maximum(raw,term['floor'])/term['x0']
            columns.append({'linear':lambda:u,'quadratic':lambda:u*u,'log':lambda:np.log(u),
                'inverse':lambda:1/u,'inverse_sqrt':lambda:1/np.sqrt(u)}[family]())
        result=np.column_stack(columns);assert np.isfinite(result).all();return result

def proxy(alpha):
    return TransformedTargetRegressor(regressor=Pipeline([
        ('dictionary',PhysicalDictionary()),('scale',StandardScaler()),
        ('lasso',LassoLars(alpha=alpha,max_iter=5000,normalize=False))]),transformer=StandardScaler())

def export_formula(model):
    pipeline=model.regressor_;scaler=pipeline['scale'];lasso=pipeline['lasso']
    coefficients=lasso.coef_*model.transformer_.scale_[0]/scaler.scale_
    intercept=float(model.transformer_.mean_[0]+model.transformer_.scale_[0]*lasso.intercept_-coefficients@scaler.mean_)
    terms=[dict(t,coefficient=float(c)) for t,c in zip(pipeline['dictionary'].terms_,coefficients) if c!=0]
    return dict(intercept=intercept,terms=terms,n_terms=len(terms),n_features=len({t['feature'] for t in terms}),
        equation='prediction = intercept + sum(coefficient * family(x/x0)); nonlinear inputs clip only at recorded numerical floor',
        response_unit='cm^-1')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    run=args.run.resolve();smoke=args.smoke
    p=dict(run=str(run),smoke=smoke,features=ale.FEATURES,alphas=ALPHAS,
        family_dictionary=['linear','quadratic','log','inverse','inverse_sqrt'],
        selection='5-fold shuffled inner CV of teacher predictions; largest alpha within one standard error of minimum mean MAE.',
        solver='LassoLars on standardized dictionary and target; replaced nonconverging coordinate descent in execution smoke testing.',
        seed=20261101,outer_test_selection=False,
        source_sha256={str(f):file_hash(f) for f in [Path(__file__).resolve(),ROOT/'scripts/analyze_native_physical_responses.py']},
        parent_audit_sha256=file_hash(run/'artifact-audit.json'),
        limitations=['Post-hoc on an already explored dataset; external confirmation still needed.',
          'Teacher CV scores measure approximation only; teacher models may have trained on those inner-validation labels.',
          'Outer-training teacher predictions alone select the proxy; no outer targets used for tuning.',
          'Correlated physical variables admit nonunique formulas. Distillation is not recovery of individual KAN edges.',
          'The new proxy fits are separate exploratory cost, not part of the completed equal-budget benchmark.'])
    if (out/'protocol.json').exists():assert ale.engine.read(out/'protocol.json')==p
    write_json(out/'protocol.json',p)
    banks,y=ale.engine.load_banks(out);results=[]
    arms=ale.engine.ARMS[:1] if smoke else ale.engine.ARMS
    folds=range(1) if smoke else range(5)
    for arm in arms:
        for fold in folds:
            d=out/arm/f'fold-{fold}';d.mkdir(parents=True,exist_ok=True)
            if (d/'complete.json').exists():
                r=ale.engine.read(d/'complete.json');assert r['protocol_fingerprint']==digest(p)
                assert file_hash(d/'frozen-formula.json')==r['formula_sha256'];results.append(r);continue
            split=ale.engine.read(ale.engine.OLD/'matbench_phonons'/f'fold-{fold}'/'outer-split.json')
            frame=banks[arm.rsplit('_',1)[0]];train=frame.loc[split['train_ids']];test=frame.loc[split['test_ids']]
            selection=ale.engine.read(run/arm/f'fold-{fold}'/'frozen-selection.json')
            jobs=selection['members'][:1] if smoke else selection['members']
            with np.load(run/arm/f'fold-{fold}'/'outer-predictions.npz') as z:teacher_test=z['members'][:len(jobs)].mean(axis=0)
            prediction_cache=d/'teacher-training-predictions.npz'
            if prediction_cache.exists():
                with np.load(prediction_cache) as z:
                    assert list(z['sample_ids'])==split['train_ids'];teacher_train=z['prediction'].copy()
            else:
                teacher_train=np.zeros(len(train))
                for i,job in enumerate(jobs):
                    path=Path(job['directory'])/'model.pkl';assert file_hash(path)==selection['model_sha256'][job['directory']]
                    with path.open('rb') as stream:model=pickle.load(stream)
                    model._restore_model();cols=model.optimal_descriptors[:model.n_feat]
                    teacher_train+=ale.predict(model,train[cols].values)/len(jobs)
                    del model;tf.keras.backend.clear_session();gc.collect()
                    write_json(out/'status.json',dict(state='teacher_predictions',pid=os.getpid(),arm=arm,fold=fold,member=i+1,members=len(jobs)))
                np.savez_compressed(prediction_cache,sample_ids=np.array(split['train_ids']),prediction=teacher_train)
            x=train[list(ale.FEATURES.values())].values;xt=test[list(ale.FEATURES.values())].values
            cv=list(KFold(n_splits=5,shuffle=True,random_state=20261101+fold).split(x));scores=[]
            for alpha in ALPHAS:
                losses=[]
                for tr,va in cv:
                    m=proxy(alpha);m.fit(x[tr],teacher_train[tr]);losses.append(float(np.mean(abs(m.predict(x[va])-teacher_train[va]))))
                    assert m.regressor_['lasso'].n_iter_<5000,'Lasso path iteration limit'
                scores.append(dict(alpha=alpha,fold_mae=losses,mean_mae=float(np.mean(losses)),se=float(np.std(losses,ddof=1)/np.sqrt(5))))
            best=min(scores,key=lambda r:r['mean_mae']);threshold=best['mean_mae']+best['se']
            alpha=max(r['alpha'] for r in scores if r['mean_mae']<=threshold)
            model=proxy(alpha);model.fit(x,teacher_train);assert model.regressor_['lasso'].n_iter_<5000
            formula=export_formula(model);formula.update(alpha=alpha,cv=scores,protocol_fingerprint=digest(p),
                train_ids=split['train_ids'],teacher_predictions_sha256=file_hash(prediction_cache),
                selected_before_outer_target_scoring=True)
            write_json(d/'frozen-formula.json',formula)
            prediction=model.predict(xt)
            dictionary=PhysicalDictionary();dictionary.terms_=formula['terms']
            reconstructed=(dictionary.transform(xt)@np.array([t['coefficient'] for t in formula['terms']])+formula['intercept']) if formula['terms'] else np.full(len(xt),formula['intercept'])
            np.testing.assert_allclose(reconstructed,prediction,rtol=1e-10,atol=1e-8)
            # Formula selection has finished; outer targets are first used here.
            target=y.loc[split['test_ids']].values[:,0]
            result=dict(arm=arm,fold=fold,smoke=smoke,protocol_fingerprint=digest(p),
                formula_sha256=file_hash(d/'frozen-formula.json'),alpha=alpha,n_terms=formula['n_terms'],n_features=formula['n_features'],
                outer_target_mae=float(np.mean(abs(prediction-target))),teacher_outer_target_mae=float(np.mean(abs(teacher_test-target))),
                outer_teacher_approximation_mae=float(np.mean(abs(prediction-teacher_test))),
                inner_teacher_approximation_mae=next(r['mean_mae'] for r in scores if r['alpha']==alpha),
                formula_prediction_max_error=float(np.max(abs(reconstructed-prediction))),
                proxy_fits=5*len(ALPHAS)+1,feature_domains={k:[float(x[:,i].min()),float(x[:,i].max())] for i,k in enumerate(ale.FEATURES)})
            np.savez_compressed(d/'outer-predictions.npz',prediction=prediction,teacher=teacher_test,target=target,sample_ids=np.array(split['official_test_ids']))
            result['prediction_sha256']=file_hash(d/'outer-predictions.npz');write_json(d/'complete.json',result);results.append(result)
    means={arm:{key:float(np.mean([r[key] for r in results if r['arm']==arm])) for key in ['outer_target_mae','teacher_outer_target_mae','outer_teacher_approximation_mae','n_terms','n_features']}
           for arm in arms if not smoke and sum(r['arm']==arm for r in results)==5}
    write_json(out/'analysis.json',dict(complete=len(results)==20,smoke=smoke,rows=results,means=means,limitations=p['limitations']))
    write_json(out/'status.json',dict(state='complete',pid=os.getpid(),smoke=smoke,fold_arm_pairs=len(results)))

if __name__=='__main__':
    try:main()
    except BaseException:
        import sys
        if '--output' in sys.argv:write_json(Path(sys.argv[sys.argv.index('--output')+1])/'status.json',dict(state='failed',pid=os.getpid(),traceback=traceback.format_exc()))
        raise
