"""Independent artifact checks for native-protocol four-arm comparisons."""
from native_fourway_features import read, OLD
from run_modnet_official_matched import np, file_hash, digest, write_json
from pathlib import Path
import argparse


def audit(out, require_complete=False):
    out=Path(out).resolve();p=read(out/'protocol.json');fp=digest(p)
    for path,sha in p['source_sha256'].items(): assert file_hash(path)==sha,path
    info=read(out/'data-audit.json')
    assert info['excluded_sisso']==283 and info['added_features']==2396 and info['identical_mm_values']
    rows=[];paired={};initial={};seen={a:[] for a in p['arms']}
    for arm in p['arms']:
        bank=arm.rsplit('_',1)[0]
        for fold in p['folds']:
            directory=out/arm/f'fold-{fold}'
            if not (directory/'complete.json').exists():continue
            r=read(directory/'complete.json');assert r['protocol_fingerprint']==fp
            assert file_hash(directory/'outer-predictions.npz')==r['prediction_sha256']
            assert file_hash(directory/'frozen-selection.json')==r['selection_sha256']
            split=read(OLD/'matbench_phonons'/f'fold-{fold}'/'outer-split.json')
            inp=read(out/'inputs'/bank/f'fold-{fold}'/'manifest.json')
            assert inp['train_ids']==split['train_ids']
            assert file_hash(out/'inputs'/bank/f'fold-{fold}'/'training.pkl.gz')==inp['sha256']
            records=[read(path) for path in sorted((directory/'fits').glob('*/result.json'))]
            assert len(records)==r['search_fits']==p['fits_per_arm_fold']
            nested=2 if p['smoke'] else 5;pop=2 if p['smoke'] else 20;generations=2 if p['smoke'] else 10
            expected={(g,c,i) for g in range(generations) for c in range(pop) for i in range(nested)}
            assert {(m['generation'],m['candidate'],m['inner']) for m in records}==expected
            candidates={}
            for m in records:
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
                    assert abs(m['parameters']/m['paired_modnet_parameters']-1)<=.05
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
    result=dict(passed=True,complete=complete,smoke=p['smoke'],rows=rows,
        checks=['Full GA budget','Native top10 mean-inner-MAE selection','No SISSO or outer contamination',
                'Identical inner partitions across all arms','Paired initial population within bank',
                'Spline parameter matching','Model and input hashes','OOF score reconstruction'])
    write_json(out/'artifact-audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    print(audit(args.output,args.require_complete))
