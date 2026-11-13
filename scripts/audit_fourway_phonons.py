"""Independent artifact checks; never train or select from outer scores."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from fourway_phonons_policy import ARMS, dense_parameters, dense_widths, spline_parameters, spline_widths


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def audit(root):
    root=Path(root);p=read(root/'protocol.json');fingerprint=digest(p)
    checked_inputs={};validation_targets={};checked_fits=0;checked_models=0;max_score_error=0.
    for bank in ('mm','mm_mv'):
        for fold in p['folds']:
            directory=root/'inputs'/bank/f"fold-{fold['fold']}"
            frame=pd.read_pickle(directory/'outer-training-only.pkl.gz')
            assert frame.index.tolist()==fold['train_ids']
            assert not set(frame.index)&set(fold['test_ids'])
            if bank=='mm_mv':
                mm=pd.read_pickle(root/'inputs/mm'/f"fold-{fold['fold']}"/'outer-training-only.pkl.gz')
                pd.testing.assert_frame_equal(frame[mm.columns],mm)
            coverage=[]
            for i in range(p['inner_folds']):
                path=directory/f'inner-{i}'/'manifest.json';info=read(path)
                assert info['run_fingerprint']==fingerprint
                tr,va=info['train_ids'],info['validation_ids'];features=info['selected_features']
                assert not set(tr)&set(va) and set(tr+va)==set(frame.index)
                assert len(features)==len(set(features))==p['rfe_count']
                assert not any('SISSO' in x.upper() for x in features)
                for key in ('feature_selection_fit_ids','cross_nmi_fit_ids','scaling_fit_ids','target_transform_fit_ids'):
                    assert info[key]==tr
                for name,value in info['artifacts'].items():assert sha(path.parent/name)==value
                with (path.parent/'scaler.pkl').open('rb') as f:scaler=pickle.load(f)
                raw=frame.loc[tr,features].values
                np.testing.assert_array_equal(scaler.data_min_,raw.min(axis=0))
                np.testing.assert_array_equal(scaler.data_max_,raw.max(axis=0))
                with np.load(path.parent/'arrays.npz') as z:
                    np.testing.assert_array_equal(z['x_train'],(raw*scaler.scale_+scaler.min_).astype('float32'))
                    np.testing.assert_array_equal(z['x_validation'],(frame.loc[va,features].values*scaler.scale_+scaler.min_).astype('float32'))
                    np.testing.assert_array_equal(z['y_train'],frame.loc[tr,'target'].values)
                    np.testing.assert_array_equal(z['y_validation'],frame.loc[va,'target'].values)
                    np.testing.assert_allclose([info['target_mean'],info['target_std']],[z['y_train'].mean(),z['y_train'].std()],rtol=0,atol=1e-10)
                checked_inputs[str(path.resolve())]=info;coverage+=va
                validation_targets[str(path.resolve())]=frame.loc[va,'target'].values
            assert sorted(coverage)==sorted(frame.index)
    for arm in ARMS:
        bank,variant=arm.rsplit('_',1)
        for fold in p['folds']:
            directory=root/arm/f"fold-{fold['fold']}";selection=read(directory/'frozen-selection.json')
            paths=sorted((directory/'fits').glob('*/result.json'))
            assert len(paths)==p['budget_per_arm_per_fold']
            by_candidate={c['candidate']:[] for c in p['candidates']}
            seen=set()
            for path in paths:
                r=read(path);key=(r['candidate'],r['inner'])
                assert key not in seen;seen.add(key)
                c=p['candidates'][r['candidate']]
                inp=checked_inputs[str(Path(r['manifest']).resolve())]
                assert r['run_fingerprint']==fingerprint and r['arm']==arm and r['variant']==variant
                assert r['config']==c and r['input_digest']==digest(inp)
                assert r['train_ids']==inp['train_ids'] and r['validation_ids']==inp['validation_ids']
                assert r['features']==inp['selected_features'][:c['n_feat']]
                assert r['seed']==p['seed']+fold['fold']*100000+c['candidate']*100+r['inner']
                assert 1<=r['best_epoch']<=r['epochs']<=p['max_epochs']
                expected=dense_parameters(c['n_feat'],dense_widths(c['width'])) if variant=='modnet' else spline_parameters(c['n_feat'],*spline_widths(c['n_feat'],c['width']))
                assert r['parameters']==expected
                with np.load(path.parent/'validation.npz') as z:
                    np.testing.assert_array_equal(z['sample_ids'],r['validation_ids'])
                    np.testing.assert_array_equal(z['target'],validation_targets[str(Path(r['manifest']).resolve())])
                    error=abs(float(np.mean(abs(z['prediction']-z['target'])))-r['validation_mae'])
                    assert error<1e-8;max_score_error=max(max_score_error,error)
                assert sha(path.parent/'validation.npz')==r['artifacts']['validation.npz']
                assert np.isfinite(r['validation_mae'])
                by_candidate[r['candidate']].append(r)
                checked_fits+=1
            ranked=sorted(by_candidate,key=lambda c:(np.mean([r['validation_mae'] for r in by_candidate[c]]),c))
            selected=ranked[:p['keep_candidates']]
            assert selection['selected_candidates']==selected
            members=[r['directory'] for c in selected for r in sorted(by_candidate[c],key=lambda r:r['inner'])]
            assert selection['members']==members
            for member in members:
                r=read(Path(member)/'result.json')
                assert sha(Path(member)/'model.pkl')==r['artifacts']['model.pkl']==selection['selected_model_sha256'][member]
                with (Path(member)/'model.pkl').open('rb') as f:model=pickle.load(f)
                assert sum(w.size for w in model['weights'])==r['parameters']
                layers=json.loads(model['model_json'])['config']['layers']
                if variant=='kan':
                    assert not any(l['class_name']=='Dense' for l in layers)
                    assert sum('SplineEdgeLayer' in l['class_name'] for l in layers)==3
                checked_models+=1
            score=read(directory/'complete.json')
            assert score['search_fits']==len(paths)
            assert score['selection_sha256']==sha(directory/'frozen-selection.json')
            assert score['prediction_sha256']==sha(directory/'outer-predictions.npz')
            with np.load(directory/'outer-predictions.npz') as z:
                assert z['sample_ids'].tolist()==fold['test_ids']
                assert z['members'].shape==(p['inner_folds']*p['keep_candidates'],len(fold['test_ids']))
                np.testing.assert_allclose(z['prediction'],z['members'].mean(axis=0),rtol=0,atol=1e-10)
                np.testing.assert_allclose(np.mean(abs(z['prediction']-z['target'])),score['mae'],rtol=0,atol=1e-9)
    assert checked_fits==p['total_fits']
    gate=read(root/'baseline-quality-gate.json')
    assert gate['passed'] and gate['smoke']==p['smoke']
    result=dict(passed=True,smoke=p['smoke'],verified_inputs=len(checked_inputs),verified_fits=checked_fits,
        verified_selected_models=checked_models,max_recomputed_validation_mae_error=max_score_error,
        scope='Nested fit scope, train-only scaling, targets, common MM values, all candidate budgets, score-based selection, saved validation and outer MAEs, selected model hashes and parameter counts')
    (root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    print(json.dumps(audit(parser.parse_args().output.resolve()),indent=2))
