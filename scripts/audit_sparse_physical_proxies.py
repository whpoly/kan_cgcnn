"""Independent replay audit of the frozen exploratory physical proxies."""
from pathlib import Path
import validate_sparse_physical_proxies as v
from run_modnet_official_matched import np, MODData, file_hash, digest, write_json

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'benchmarks/native-physical-proxy-20261101'
read=v.ale.engine.read
p=read(out/'protocol.json'); analysis=read(out/'analysis.json'); run=Path(p['run'])
assert analysis['complete'] and not p['smoke'] and len(analysis['rows'])==20
for name,sha in p['source_sha256'].items(): assert file_hash(name)==sha
assert file_hash(run/'artifact-audit.json')==p['parent_audit_sha256']
data=MODData.load(v.ale.engine.ORIGINAL)
frame=data.df_featurized.replace([np.inf,-np.inf],np.nan).fillna(0)
rows=[]
for r in analysis['rows']:
    arm,fold=r['arm'],r['fold']; d=out/arm/f'fold-{fold}'
    assert read(d/'complete.json')==r and r['protocol_fingerprint']==digest(p)
    assert file_hash(d/'frozen-formula.json')==r['formula_sha256']
    assert file_hash(d/'outer-predictions.npz')==r['prediction_sha256']
    f=read(d/'frozen-formula.json'); split=read(v.ale.engine.OLD/'matbench_phonons'/f'fold-{fold}'/'outer-split.json')
    assert f['train_ids']==split['train_ids'] and not set(split['train_ids'])&set(split['test_ids'])
    assert f['selected_before_outer_target_scoring'] and f['protocol_fingerprint']==digest(p)
    assert file_hash(d/'teacher-training-predictions.npz')==f['teacher_predictions_sha256']
    scores=f['cv']; best=min(scores,key=lambda x:x['mean_mae'])
    assert f['alpha']==max(x['alpha'] for x in scores if x['mean_mae']<=best['mean_mae']+best['se'])
    with np.load(d/'teacher-training-predictions.npz') as z:
        assert list(z['sample_ids'])==split['train_ids']; train_pred=z['prediction'].copy()
    x=frame.loc[split['train_ids'],list(v.ale.FEATURES.values())].values
    xt=frame.loc[split['test_ids'],list(v.ale.FEATURES.values())].values
    model=v.proxy(f['alpha']); model.fit(x,train_pred)
    # Same fixed alpha; no selection or tuning using held-out observations.
    replay=model.predict(xt)
    with np.load(d/'outer-predictions.npz') as z:
        assert list(z['sample_ids'])==split['official_test_ids']
        np.testing.assert_allclose(z['target'],data.df_targets.loc[split['test_ids']].values[:,0],atol=1e-10)
        np.testing.assert_allclose(z['prediction'],replay,rtol=1e-9,atol=1e-7)
        with np.load(run/arm/f'fold-{fold}'/'outer-predictions.npz') as original:
            np.testing.assert_allclose(z['teacher'],original['members'].mean(axis=0),atol=1e-10)
        for key,a,b in [('outer_target_mae',z['prediction'],z['target']),('teacher_outer_target_mae',z['teacher'],z['target']),('outer_teacher_approximation_mae',z['prediction'],z['teacher'])]:
            assert abs(float(np.mean(abs(a-b)))-r[key])<1e-8
    rows.append(dict(arm=arm,fold=fold,teacher_train_target_mae=float(np.mean(abs(train_pred-data.df_targets.loc[split['train_ids']].values[:,0]))),proxy_train_teacher_mae=float(np.mean(abs(model.predict(x)-train_pred))),n_terms=r['n_terms']))
for arm,means in analysis['means'].items():
    group=[r for r in analysis['rows'] if r['arm']==arm]; assert sorted(r['fold'] for r in group)==list(range(5))
    for key,val in means.items(): assert abs(val-np.mean([r[key] for r in group]))<1e-9
audit=dict(passed=True,complete=True,smoke=False,fold_arm_pairs=20,source_hashes_verified=True,formula_selection_rule_verified=True,fixed_alpha_refits_replay=True,official_targets_and_frozen_teachers_verified=True,analysis_sha256=file_hash(out/'analysis.json'),audit_source_sha256=file_hash(__file__),diagnostics=rows)
write_json(out/'artifact-audit.json',audit)
print({'passed':True,'means':analysis['means'],'term_range':[min(r['n_terms'] for r in rows),max(r['n_terms'] for r in rows)],'train_teacher_target_mae_range':[min(r['teacher_train_target_mae'] for r in rows),max(r['teacher_train_target_mae'] for r in rows)]})
