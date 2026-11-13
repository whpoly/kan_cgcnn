"""Execution validation only: classification scaling, probabilities, counts, persistence."""
import copy
import gc
import os
import pickle
import traceback
from pathlib import Path
import native_classification_spline as model_lib
from native_classification_spline import np,tf,NativeClassificationSpline,probabilities
import run_native_matbench_regression_expansion_v2 as regression
from run_modnet_official_matched import MODData,MODNetModel,set_seed,write_json,file_hash,matbench_kfold_splits
from sklearn.metrics import roc_auc_score,log_loss

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/native-classification-adapter-smoke-20261101'
PREP=ROOT/'benchmarks/matbench-expansion-20261101/preparation'
SOURCE=ROOT/'benchmarks/matbench-expansion-20261101/source'
TASKS=['matbench_expt_is_metal','matbench_glass']

def make_data(frame,target,columns):
    data=MODData(materials=[None]*len(frame),targets=target.values,target_names=list(target.columns),
        num_classes={target.columns[0]:2},structure_ids=list(frame.index),df_featurized=frame[columns])
    data.optimal_features=columns;data.optimal_features_by_target={target.columns[0]:columns}
    return data

def main():
    OUT.mkdir(exist_ok=True,parents=True);rows=[]
    for task in TASKS:
        out=OUT/task;out.mkdir(exist_ok=True)
        p=dict(task=task,original=str(PREP/'precomputed'/f'{task}_moddata.pkl.gz'),
            csv=str(next(SOURCE.glob(task+'_*.csv'))))
        banks,y=regression.load_banks(p,out)
        outer=regression.e.read(PREP/task/'fold-0/outer-split.json')
        native=MODData.load(PREP/task/'fold-0/official-train-moddata.pkl.gz')
        assert list(native.num_classes.values())==[2]
        for bank in ['mm','mm_mv']:
            columns=list(native.optimal_features[:24]) if bank=='mm' else list(native.optimal_features[:16])+[c for c in banks[bank] if c not in banks['mm']][:8]
            data=make_data(banks[bank].loc[outer['train_ids']],y.loc[outer['train_ids']],columns)
            tr,va=next(matbench_kfold_splits(data,n_splits=5,classification=True));train,val=data.split((tr,va))
            for family,cls in [('modnet',MODNetModel),('kan',NativeClassificationSpline)]:
                write_json(OUT/'status.json',dict(state='testing',pid=os.getpid(),task=task,bank=bank,family=family))
                tf.keras.backend.clear_session();set_seed(20261101)
                kwargs=dict(targets=[[list(y.columns)]],weights={y.columns[0]:1},num_classes={y.columns[0]:2},
                    n_feat=24,num_neurons=[[32],[16],[8],[4]],multi_label=False)
                if family=='kan':kwargs.update(spline_seed=20261101,input_scale='minmax')
                model=cls(**kwargs);expected=model_lib.dense_count(24,(32,16,8,4)) if family=='modnet' else model_lib.mapping(24,(32,16,8,4))['spline_parameters']
                assert model.model.count_params()==expected
                set_seed(20261101)
                model.fit(train,val_data=val,loss='categorical_crossentropy',lr=.001,epochs=2,batch_size=32,xscale='minmax',verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=.001,patience=30,restore_best_weights=False)])
                pred=probabilities(model,val.df_featurized);target=val.df_targets.values[:,0].astype(int)
                auc=roc_auc_score(target,pred[:,1]);native_auc=-float(model.evaluate(val))
                assert abs(auc-native_auc)<1e-4
                assert model.model.loss=='categorical_crossentropy' and not hasattr(model,'target_scaling')
                model._make_picklable();path=out/(bank+'_'+family+'.pkl')
                with path.open('wb') as f:pickle.dump(model,f,protocol=4)
                with path.open('rb') as f:restored=pickle.load(f)
                restored._restore_model();np.testing.assert_array_equal(probabilities(restored,val.df_featurized),pred)
                rows.append(dict(task=task,bank=bank,family=family,parameters=expected,loss='categorical_crossentropy',
                    validation_probability_auc=float(auc),validation_log_loss=float(log_loss(target,pred)),
                    train_ids=list(train.df_targets.index),validation_ids=list(val.df_targets.index),outer_test_used=False,
                    model_sha256=file_hash(path),serialization_replay_passed=True,target_standardization=False))
                del model,restored;gc.collect()
    write_json(OUT/'artifact-audit.json',dict(passed=True,smoke=True,formal_results=False,rows=rows,
        source_sha256={str(f):file_hash(f) for f in [Path(__file__).resolve(),ROOT/'scripts/native_classification_spline.py']}))
    write_json(OUT/'status.json',dict(state='complete',smoke=True,pid=os.getpid(),fits=len(rows)))

if __name__=='__main__':
    try:main()
    except BaseException:write_json(OUT/'status.json',dict(state='failed',pid=os.getpid(),traceback=traceback.format_exc()));raise
