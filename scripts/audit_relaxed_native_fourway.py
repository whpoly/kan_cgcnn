"""Independent artifact checks for native-protocol four-arm comparisons."""
from native_fourway_features import read, OLD
from run_modnet_official_matched import np, file_hash, digest, write_json
from pathlib import Path
import argparse
from run_modnet_official_matched import MODData
from relaxed_spline_policy import describe
from official_kan_models import neurons_from_genes


def verify_prior(p):
    prior=Path(p['prior_run'])
    assert file_hash(prior/'protocol.json')==p['prior_protocol_sha256']
    for path,sha in p['reused_artifact_sha256'].items():assert file_hash(path)==sha,path
    from audit_native_fourway import audit as original_audit
    result=original_audit(prior)
    baseline=[r for r in result['rows'] if r['arm']=='mm_modnet']
    assert len(baseline)==5 and sum(r['verified_fits'] for r in baseline)==5000
    assert read(prior/'baseline-quality-gate.json')['passed']
    return result


def audit(out, require_complete=False):
    out=Path(out).resolve();p=read(out/'protocol.json');fp=digest(p)
    amendment=read(out/'parameter-matching-amendment.json')
    assert file_hash(out/'protocol.json')==amendment['base_protocol_sha256']
    for path,sha in amendment['source_sha256'].items():assert file_hash(path)==sha,path
    for path,hashes in amendment['preserved_fit_results'].items():
        assert file_hash(path)==hashes['result_sha256']
    for path,sha in p['source_sha256'].items(): assert file_hash(path)==sha,path
    if p['reuse_baseline']: verify_prior(p)
    info=read(out/'data-audit.json')
    assert info['excluded_sisso']==283 and info['added_features']==2396 and info['identical_mm_values']
    rows=[];paired={};initial={};seen={a:[] for a in p['arms']}
    for arm in p['arms']:
        bank=arm.rsplit('_',1)[0]
        for fold in p['folds']:
            reused=p['reuse_baseline'] and arm=='mm_modnet'
            base=Path(p['prior_run']) if reused else out
            directory=base/arm/f'fold-{fold}'
            if not (directory/'complete.json').exists():continue
            r=read(directory/'complete.json')
            assert r['protocol_fingerprint']==(digest(read(base/'protocol.json')) if reused else fp)
            if reused:
                imported=read(out/arm/f'fold-{fold}'/'complete.json')
                expected=dict(r,source_protocol_fingerprint=r['protocol_fingerprint'],
                    protocol_fingerprint=fp,reused_from=str(directory))
                assert imported==expected
                for name in ('outer-predictions.npz','frozen-selection.json','predictions.csv'):
                    assert file_hash(out/arm/f'fold-{fold}'/name)==file_hash(directory/name)
            assert file_hash(directory/'outer-predictions.npz')==r['prediction_sha256']
            assert file_hash(directory/'frozen-selection.json')==r['selection_sha256']
            split=read(OLD/'matbench_phonons'/f'fold-{fold}'/'outer-split.json')
            inp=read(base/'inputs'/bank/f'fold-{fold}'/'manifest.json')
            assert inp['train_ids']==split['train_ids']
            cache=base/'inputs'/bank/f'fold-{fold}'/'training.pkl.gz'
            assert file_hash(cache)==inp['sha256']
            target=MODData.load(cache).df_targets if arm.endswith('kan') else None
            records=[read(path) for path in sorted((directory/'fits').glob('*/result.json'))]
            assert len(records)==r['search_fits']==p['fits_per_arm_fold']
            nested=2 if p['smoke'] else 5;pop=2 if p['smoke'] else 20;generations=2 if p['smoke'] else 10
            expected={(g,c,i) for g in range(generations) for c in range(pop) for i in range(nested)}
            assert {(m['generation'],m['candidate'],m['inner']) for m in records}==expected
            candidates={}
            for m in records:
                result_path=str(Path(m['directory'])/'result.json')
                if not reused and result_path not in amendment['preserved_fit_results']:
                    assert m['amendment_fingerprint']==digest(amendment)
                assert file_hash(Path(m['directory'])/'model.pkl')==m['model_sha256']
                assert set(m['train_ids']).isdisjoint(m['validation_ids'])
                assert set(m['train_ids']+m['validation_ids'])==set(split['train_ids'])
                assert set(m['train_ids']+m['validation_ids']).isdisjoint(split['test_ids'])
                assert m['selected_features']==inp['features'][:m['genes']['n_feat']]
                assert not any('SISSO' in c.upper() for c in m['features'])
                assert m['epochs']<= (1 if p['smoke'] else 800)
                assert np.isfinite(m['validation_mae'])
                key=(fold,m['inner']);partition=(m['train_ids'],m['validation_ids'])
                if key in paired:assert paired[key]==partition
                else:paired[key]=partition
                if m['generation']==0:
                    key=(bank,fold,m['candidate'],m['inner'])
                    if key in initial:assert initial[key]==m['genes']
                    else:initial[key]=m['genes']
                if m['variant']=='bspline':
                    widths=tuple(x[0] for x in neurons_from_genes(m['genes']))
                    mapping=describe(m['genes']['n_feat'],widths)
                    assert m['spline_widths']==mapping['widths']
                    assert m['parameters']==mapping['spline_parameters']
                    assert m['paired_modnet_parameters']==mapping['reference_parameters']
                    if mapping['fallback_used']:assert m['parameter_mapping']==mapping
                    scale=m['target_scaling'];values=target.loc[m['train_ids']].values
                    assert scale['fit_ids']==m['train_ids'] and scale['prediction_units']=='physical'
                    assert scale['mean']==float(values.mean()) and scale['std']==float(values.std())
                    assert scale['early_stopping_delta_normalized']==.001/scale['std']
                key=(m['generation'],m['candidate']);candidates.setdefault(key,[]).append(m)
            groups=[sorted(candidates[k],key=lambda x:x['inner']) for k in sorted(candidates)]
            means=np.array([np.mean([m['validation_mae'] for m in group]) for group in groups])
            expected_members=[m['directory'] for i in means.argsort()[:10] for m in groups[i]]
            selection=read(directory/'frozen-selection.json')
            assert not selection['test_labels_used']
            assert [m['directory'] for m in selection['members']]==expected_members
            for j in selection['members']:
                assert file_hash(Path(j['directory'])/'model.pkl')==selection['model_sha256'][j['directory']]
            with np.load(directory/'outer-predictions.npz') as z:
                assert list(z['sample_ids'])==split['official_test_ids']
                assert z['members'].shape==(len(expected_members),253)
                np.testing.assert_array_equal(z['prediction'],z['members'].mean(axis=0))
                np.testing.assert_allclose(abs(z['prediction']-z['target']).mean(),r['mae'],atol=1e-10)
                seen[arm]+=list(z['sample_ids'])
            rows.append(dict(arm=arm,fold=fold,verified_fits=len(records),mae=r['mae']))
    complete=len(rows)==4*len(p['folds'])
    if require_complete:assert complete
    if complete and not p['smoke']:
        for ids in seen.values():assert len(ids)==len(set(ids))==1265
        assert len(rows)==20 and sum(r['verified_fits'] for r in rows)==20000
    mappings=[read(path) for arm in p['arms'] if arm.endswith('kan') for path in (out/arm).glob('fold-*/fits/*/result.json')]
    result=dict(passed=True,complete=complete,smoke=p['smoke'],rows=rows,
        amendment_fingerprint=digest(amendment),
        relaxed_fit_count=sum(m.get('parameter_mapping',{}).get('fallback_used',False) for m in mappings),
        parameter_difference_range=[float(min(m['parameters']/m['paired_modnet_parameters']-1 for m in mappings)),
            float(max(m['parameters']/m['paired_modnet_parameters']-1 for m in mappings))] if mappings else None,
        checks=['Full GA budget','Native top10 mean-inner-MAE selection','No SISSO or outer contamination',
                'Identical inner partitions across all arms','Paired initial population within bank',
                'Original mapping preserved, otherwise closest feasible spline parameter count',
                'Frozen amendment and inherited fit hashes','Model and input hashes','OOF score reconstruction'])
    write_json(out/'artifact-audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    print(audit(args.output,args.require_complete))
