"""Replay threshold selection and fold MAE from the locked reconstruction artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from modnet_runtime import configure_official_runtime
configure_official_runtime()
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'benchmarks/steels-paper-reproduction-20261102/lmp-v2'
SRC=ROOT/'benchmarks/matbench-expansion-20261101/matbench_steels'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def audit(mode,arm,fold):
    out=BASE/mode
    d=out/arm/f'fold-{fold}'
    proto=read(out/'protocol.json')
    split=read(SRC/'inputs/mm'/f'fold-{fold}'/'manifest.json')
    train,test=map(set,(split['train_ids'],split['outer_test_ids']))
    assert not train&test and len(train|test)==312
    candidates=proto['options']
    scores={}
    for opt in candidates:
        stage=d/str(opt);rec=read(stage/'inner-selection.json')
        seen=set();errors=[]
        for k in range(2 if mode=='smoke' else 5):
            rank=read(d/f'inner-{k}'/'ranking.json')
            fit_ids=set(rank['train_ids'])
            assert fit_ids < train and set(rank['explained_and_background_ids']) <= fit_ids
            jobs=[j for j in rec['members'] if j['inner']==k]
            assert jobs
            for job in jobs:
                member=Path(job['directory']);fit=read(member/'result.json')
                assert set(fit['train_ids'])==fit_ids and set(fit['validation_ids'])==train-fit_ids
                assert fit['features']==rank['subsets'][str(opt)]
                assert sha(member/'model.pkl')==job['sha256']
            val=fit['validation_ids']
            assert not seen&set(val);seen.update(val)
            with np.load(stage/f'validation-{k}-member-predictions.npz') as z:
                pred=z['predictions'].mean(axis=0)
            errors.extend(abs(pred-np.array([targets[i] for i in val])).tolist())
        assert seen==train
        scores[str(opt)]=float(np.mean(errors))
        assert abs(scores[str(opt)]-rec['validation_mae'])<1e-6
    frozen=read(d/'frozen-selection.json')
    choice=min(candidates,key=lambda o:scores[str(o)])
    assert choice==frozen['option'] and frozen['members']==read(d/str(choice)/'inner-selection.json')['members']
    result=read(d/'complete.json')
    with np.load(d/'test.npz') as z, np.load(d/'outer-member-predictions.npz') as ensemble:
        ids=z['ids'].tolist()
        assert set(ids)==test and len(ids)==len(test)
        assert np.array_equal(z['target'],np.array([targets[i] for i in ids]))
        assert np.array_equal(ensemble['predictions'].mean(axis=0),z['prediction'])
        mae=float(np.mean(abs(z['target']-z['prediction'])))
    assert abs(result['mae']-mae)<1e-8 and sha(d/'test.npz')==result['test_sha256']
    return dict(passed=True,smoke=mode=='smoke',arm=arm,fold=fold,threshold=choice,mae=mae,
                training_count=len(train),external_count=len(test),validation_maes=scores)

if __name__=='__main__':
    import gzip
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','formal']);args=p.parse_args()
    dataset=Path.home()/'.conda/envs/kan-cgcnn-cuda/Lib/site-packages/matminer/datasets/matbench_steels.json.gz'
    with gzip.open(dataset,'rt') as f:data=json.load(f)
    targets={f'id{i}':row[1] for i,row in zip(data['index'],data['data'])}
    paths=list((BASE/args.mode).glob('*/fold-*/complete.json'))
    if not paths:sys.exit('No completed folds yet.')
    for path in paths:
        record=audit(args.mode,path.parents[1].name,int(path.parent.name.split('-')[1]))
        (path.parent/'independent-audit.json').write_text(json.dumps(record,indent=2),encoding='utf8')
        print(record['arm'],record['fold'],'passed')
