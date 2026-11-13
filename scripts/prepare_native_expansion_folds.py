"""Prepare original MODNet features and outer-training rankings for expansion.

No training, no model selection on outer labels. Published predictions are only
used to verify sample membership and separately document reference metrics.
"""
import gzip
import json
import os
import time
import traceback
from pathlib import Path
from run_modnet_official_matched import (np, write_json, file_hash, set_seed,
    matbench_kfold_splits, MODData)
from run_official_modnet_matbench import load_or_featurize, CLASSIFICATION_TASKS
from sklearn.metrics import mean_absolute_error, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/matbench-expansion-20261101/preparation'
REFERENCE=ROOT/'benchmarks/official-aligned-20261021/reference/results.json.gz'
TASKS=['matbench_expt_gap','matbench_expt_is_metal','matbench_glass','matbench_jdft2d','matbench_log_gvrh']

def status(state,**kw):
    write_json(OUT/'status.json',dict(state=state,pid=os.getpid(),updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(tasks=TASKS,seed=20261101,n_jobs=2,
        source_sha256={str(p):file_hash(p) for p in [Path(__file__),ROOT/'scripts/run_official_modnet_matbench.py']},
        reference_sha256=file_hash(REFERENCE),
        feature_selection='Original MODNet 0.1.12 n=-1, precomputed cross-NMI; supervised ranking on outer train only.',
        classification='Two-class MODData; training adapter still requires separate validation.',
        limitation='Outer-training feature ranking is shared with inner validation, matching prior protocol.')
    manifest=OUT/'protocol.json'
    if manifest.exists():assert json.loads(manifest.read_text())==protocol
    write_json(manifest,protocol)
    with gzip.open(REFERENCE,'rt') as f:reference=json.load(f)
    completed=[]
    for index,task in enumerate(TASKS):
        td=OUT/task;td.mkdir(exist_ok=True)
        if (td/'complete.json').exists():
            old=json.loads((td/'complete.json').read_text())
            for f,h in old['files'].items():assert file_hash(f)==h
            completed.append(task);continue
        status('featurizing',task=task,completed_tasks=completed)
        data,metadata=load_or_featurize(task,OUT/'precomputed',2,False)
        cache=OUT/'precomputed'/f'{task}_moddata.pkl.gz'
        files={str(cache):file_hash(cache)};rows=[]
        for fold,(train_pos,test_pos) in enumerate(matbench_kfold_splits(data)):
            status('feature_selection',task=task,fold=fold,completed_tasks=completed)
            prefix='mb-'+task.removeprefix('matbench_').replace('_','-') if hasattr(str,'removeprefix') else 'mb-'+task[len('matbench_'):].replace('_','-')
            official=[f'{prefix}-{int(i)+1:0{len(str(len(data.df_targets)))}d}' for i in test_pos]
            r=reference['tasks'][task]['results'][f'fold_{fold}']
            assert set(official)==set(r['data'])
            y=data.df_targets.iloc[test_pos,0].values
            preds=np.array([r['data'][i] for i in official])
            metric=float(roc_auc_score(y,preds)) if task in CLASSIFICATION_TASKS else float(mean_absolute_error(y,preds))
            if task not in CLASSIFICATION_TASKS:np.testing.assert_allclose(metric,r['scores']['mae'],rtol=0,atol=1e-8)
            # Some historical submissions report a thresholded ROC-AUC; retain both.
            rows.append(dict(fold=fold,recomputed_probability_rocauc=metric if task in CLASSIFICATION_TASKS else None,
                             recomputed_mae=metric if task not in CLASSIFICATION_TASKS else None,published_scores=r['scores']))
            d=td/f'fold-{fold}';d.mkdir(exist_ok=True)
            train,test=data.split((train_pos,test_pos));saved=d/'official-train-moddata.pkl.gz'
            if saved.exists():
                selected=MODData.load(saved)
                assert list(selected.df_targets.index)==list(train.df_targets.index)
                np.testing.assert_array_equal(selected.df_targets.values,train.df_targets.values)
                train=selected
            else:
                set_seed(20261101+index*1000000+fold*100000)
                train.feature_selection(n=-1,use_precomputed_cross_nmi=True,n_jobs=2)
                train.save(saved)
            write_json(d/'outer-split.json',dict(train_ids=list(train.df_targets.index),test_ids=list(test.df_targets.index),
                train_positions=train_pos,test_positions=test_pos,official_test_ids=official,
                selected_features=list(train.get_optimal_descriptors()),training_data_sha256=file_hash(saved),
                feature_selection_scope='outer_train; reused in inner CV, as in MODNet 0.1.12'))
            files[str(saved)]=file_hash(saved);files[str(d/'outer-split.json')]=file_hash(d/'outer-split.json')
        write_json(td/'published-reference-verification.json',dict(task=task,folds=rows,
            meaning='Reference prediction recomputation, not newly trained model performance.'))
        write_json(td/'complete.json',dict(task=task,metadata=metadata,files=files,folds=5))
        completed.append(task)
    status('complete',completed_tasks=completed)

if __name__=='__main__':
    try:main()
    except BaseException:status('failed',traceback=traceback.format_exc());raise
