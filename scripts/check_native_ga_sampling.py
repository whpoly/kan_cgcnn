"""Execution-only boundary tests, including both real worker/model branches."""
from pathlib import Path
import native_ga_sampling as sampling
import run_native_matbench_regression_expansion_v3 as runner
from run_modnet_official_matched import np,pd,MODData,FitGenetic,set_seed,matbench_kfold_splits,write_json,file_hash,digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/native-ga-sampling-check-20261101'

def main():
    OUT.mkdir(parents=True,exist_ok=True);checks=[]
    for n in [40,5000,5001,9000]:
        for classification in [False,True]:
            ids=[f's{i}' for i in range(n)];frame=pd.DataFrame(np.random.RandomState(17).normal(size=(n,8)),index=ids,columns=[f'f{i}' for i in range(8)])
            target=(np.arange(n)%2) if classification else frame.f0.values*3+frame.f1.values
            data=MODData(materials=[None]*n,targets=target[:,None],target_names=['target'],num_classes={'target':2 if classification else 0},structure_ids=ids,df_featurized=frame)
            data.optimal_features=list(frame.columns);data.optimal_features_by_target={'target':list(frame.columns)}
            seed=20261101;set_seed(seed);ga=FitGenetic(data)
            for nested in [2,5]:
                before=np.random.get_state();sampled,folds=sampling.partition_ids(ids,target,seed,nested,classification)
                after=np.random.get_state();assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
                native_ids=np.array(list(ga.train_data.df_targets.index));assert sampled==list(native_ids) and len(sampled)==min(n,5000)
                expected=[(native_ids[tr].tolist(),native_ids[va].tolist()) for tr,va in matbench_kfold_splits(ga.train_data,n_splits=nested,classification=classification)]
                assert folds==expected
                assert set().union(*(set(va) for tr,va in folds))==set(sampled)
                checks.append(dict(rows=n,classification=classification,nested=nested,exact_native_match=True))
            if n==9000 and not classification:worker_data=data
    cache=OUT/'synthetic-training.pkl.gz';worker_data.save(cache)
    files=[Path(__file__).resolve(),Path(runner.__file__).resolve(),Path(sampling.__file__).resolve()]
    p=dict(task='synthetic_sample_cap_probe',seed=19261101,source_sha256={str(f):file_hash(f) for f in files})
    write_json(OUT/'protocol.json',p);fp=digest(p)
    sampled,folds=sampling.partition_ids(worker_data.df_targets.index,worker_data.df_targets.values,p['seed']+1000000,2)
    results=[]
    for arm in ['mm_modnet','mm_kan']:
        tr,va=folds[0];d=OUT/arm
        genes=dict(n_feat=8,n_neurons_first_layer=32,fraction1=.5,fraction2=.5,fraction3=.5,act='relu',lr=.001,batch_size=32,loss='mae',xscale='minmax')
        job=dict(directory=str(d.resolve()),genes=genes,seed=p['seed']+1000000,arm=arm,variant='modnet' if arm.endswith('modnet') else 'bspline',fold=0,generation=0,candidate=0,inner=0,train_ids=tr,validation_ids=va,smoke=True,training_cache=str(cache.resolve()),training_sha256=file_hash(cache),protocol_path=str((OUT/'protocol.json').resolve()),protocol_fingerprint=fp,run_fingerprint=digest(dict(protocol=fp,training_data=file_hash(cache))))
        result=runner.run_fit(job);assert len(set(result['train_ids']+result['validation_ids']))==5000<9000
        assert np.isfinite(result['validation_mae'])
        results.append(dict(arm=arm,sampled_rows=5000,cache_rows=9000,model_sha256=result['model_sha256'],target_scaling_train_only=arm.endswith('kan')))
    write_json(OUT/'artifact-audit.json',dict(passed=True,smoke=True,formal_results=False,checks=checks,worker_probes=results,source_sha256=p['source_sha256']))
    print('Native sampling boundary and both model worker probes passed.')

if __name__=='__main__':main()
