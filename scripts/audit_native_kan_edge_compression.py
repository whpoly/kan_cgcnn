"""Check recipe policies, saved coefficient shapes, hashes and outer predictions."""
import argparse
import hashlib
import json
from pathlib import Path
from modnet_runtime import configure_official_runtime
configure_official_runtime()
import numpy as np

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve()
    p=read(out/'protocol.json');a=read(out/'analysis.json');run=Path(p['run']);fp=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    assert read(out/'status.json')['state']=='complete'
    assert sha(run/'artifact-audit.json')==p['parent_audit_sha256']
    for f,h in p['source_sha256'].items():assert sha(f)==h
    expected={('mm_kan',0)} if p['smoke'] else {(arm,fold) for arm in ['mm_kan','mm_mv_kan'] for fold in range(5)}
    assert {(r['arm'],r['fold']) for r in a['rows']}==expected
    models=0
    for r in a['rows']:
        d=out/r['arm']/f"fold-{r['fold']}";original=run/r['arm']/f"fold-{r['fold']}"
        assert read(d/'complete.json')==r and r['protocol_fingerprint']==fp and sha(d/'outer-predictions.npz')==r['prediction_sha256']
        original_done=read(original/'complete.json')
        assert sha(original/'outer-predictions.npz')==original_done['prediction_sha256']
        assert sha(original/'frozen-selection.json')==original_done['selection_sha256']
        selection=read(original/'frozen-selection.json');recipes=[]
        paths=sorted(d.glob('member-*/frozen-recipe.json'));assert len(paths)==(1 if p['smoke'] else 50)
        for i,path in enumerate(paths):
            assert sha(path)==r['recipe_sha256'][str(path)]
            recipe=read(path);assert recipe['protocol_fingerprint']==fp and not recipe['selection_uses_outer_targets']
            job=selection['members'][i];source=Path(job['directory'])/'model.pkl'
            assert str(source)==recipe['source_model'] and sha(source)==recipe['source_model_sha256']==selection['model_sha256'][job['directory']]
            assert recipe['original_coefficient_entries']==read(source.parent/'result.json')['parameters']
            candidates=[s['degree'] for s in recipe['cv'] if s['mean_teacher_mae']<=recipe['tolerance']]
            assert recipe['degree']==(min(candidates) if candidates else None)
            for s in recipe['cv']:
                assert len(s['fold_teacher_mae'])==5 and np.isfinite(s['fold_teacher_mae']).all()
                assert abs(np.mean(s['fold_teacher_mae'])-s['mean_teacher_mae'])<1e-10
            if recipe['degree'] is not None:
                coefficients=path.parent/'polynomial-coefficients.npz';assert sha(coefficients)==recipe['coefficient_sha256']
                with np.load(coefficients) as z:
                    count=0
                    for j in range(3):
                        c=z[f'coefficients_{j}'];b=z[f'bias_{j}'];assert c.shape[-1]==recipe['degree']+1 and c.shape[1]==len(b)
                        assert np.isfinite(c).all() and np.isfinite(b).all();count+=c.size+b.size
                    assert count==recipe['compressed_coefficient_entries']<recipe['original_coefficient_entries']
            else:assert recipe['compressed_coefficient_entries']==recipe['original_coefficient_entries']
            recipes.append(recipe);models+=1
        assert sum(x['original_coefficient_entries'] for x in recipes)==r['original_entries']
        assert sum(x['compressed_coefficient_entries'] for x in recipes)==r['compressed_entries']
        with np.load(d/'outer-predictions.npz') as z,np.load(original/'outer-predictions.npz') as orig:
            np.testing.assert_array_equal(z['sample_ids'],orig['sample_ids']);np.testing.assert_array_equal(z['target'],orig['target'])
            np.testing.assert_array_equal(z['teacher'],orig['members'][:len(paths)].mean(axis=0))
            np.testing.assert_array_equal(z['prediction'],z['members'].mean(axis=0))
            assert abs(np.mean(abs(z['prediction']-z['target']))-r['mae'])<1e-10
            assert abs(np.mean(abs(z['teacher']-z['target']))-r['teacher_mae'])<1e-10
    for arm,means in a['means'].items():
        group=[r for r in a['rows'] if r['arm']==arm];assert len(group)==5 and not p['smoke']
        for key,value in means.items():assert abs(value-np.mean([r[key] for r in group]))<1e-10
    result=dict(passed=True,complete=not p['smoke'],smoke=p['smoke'],model_recipes=models,analysis_sha256=sha(out/'analysis.json'),audit_source_sha256=sha(__file__))
    (out/'artifact-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
