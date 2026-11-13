"""Training-boundary, pairing, and conditional-budget checks for the quick pilot."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_mattervial_quick_pilot import (np,tf,pd,PHYSICAL_FEATURES,VARIANTS,LRS,SEEDS,
    load_training_rows,select_once,fit_minmax,build_predictor,expansion_gate,
    paired_difference,train_job,summarize,write_json,file_hash,digest)


class QuickPilotChecks(unittest.TestCase):
    def test_loader_never_converts_outer_targets_or_any_sisso(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.csv'
            fields=['material_id','target',PHYSICAL_FEATURES[0],'SISSO_phonons_1','ORB_v3_0','ROOST_oqmd_0']
            with path.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields,['train-a','2','1','DO NOT CONVERT','3','DO NOT CONVERT'])))
                writer.writerow(dict(zip(fields,['outer','INVALID OUTER LABEL','INVALID','INVALID','INVALID','INVALID'])))
            frame,removed=load_training_rows(path,['train-a'])
            self.assertEqual(list(frame.index),['train-a'])
            self.assertEqual(removed,['SISSO_phonons_1'])
            self.assertEqual(list(frame.columns),[PHYSICAL_FEATURES[0],'ORB_v3_0','target'])
            self.assertEqual(float(frame.target.iloc[0]),2.)

    def test_selection_and_scaling_fit_only_training(self):
        rng=np.random.RandomState(19)
        columns=PHYSICAL_FEATURES+[f'ORB_v3_{i}' for i in range(8)]
        train=pd.DataFrame(rng.normal(size=(48,24)),columns=columns)
        target=3*train.iloc[:,19].values+rng.normal(size=48)
        selected,importance=select_once(train,target,20,19)
        self.assertEqual(selected[:16],PHYSICAL_FEATURES)
        self.assertEqual(len(selected),20)
        a,xa,_=fit_minmax(train[selected],np.zeros((6,20)))
        b,xb,_=fit_minmax(train[selected],np.full((6,20),1e8))
        np.testing.assert_array_equal(xa,xb)
        np.testing.assert_array_equal(a.scale_,b.scale_)
        with self.assertRaises(ValueError):
            select_once(train.assign(SISSO_hidden=1),target,20,19)

    def test_paired_capacity_initialization_and_serialization(self):
        features=PHYSICAL_FEATURES+['ORB_v3_0']
        models=[build_predictor(v,features,features,'target',17,100) for v in ('modnet','group_mlp','group_sparse_kan')]
        self.assertEqual(models[1].count_params(),models[2].count_params())
        self.assertEqual(models[1].count_params()-models[0].count_params(),148+4*96)
        weights=[[l.get_weights() for l in m.layers if isinstance(l,tf.keras.layers.Dense)] for m in models]
        for w in weights[1:]:
            np.testing.assert_array_equal(w[0][0][:len(features)],weights[0][0][0])
            for actual,expected in zip(w[1:],weights[0][1:]):
                for a,b in zip(actual,expected):
                    np.testing.assert_array_equal(a,b)
        x=np.random.RandomState(3).normal(size=(8,len(features))).astype('float32')
        restored=tf.keras.models.model_from_json(models[2].to_json())
        restored.set_weights(models[2].get_weights())
        np.testing.assert_allclose(restored(x),models[2](x),rtol=1e-6,atol=1e-6)

    def test_gate_rejects_cherry_picked_fold_and_worse_than_baseline_kan(self):
        self.assertFalse(paired_difference([8,11,11,11,11],[10]*5)['passes'])
        self.assertFalse(paired_difference([9.99]*5,[10]*5)['passes'])
        scores=dict(matminer_modnet=[10]*5,mattervial_modnet=[9.8]*5,
                    mattervial_mlp=[11]*5,mattervial_kan=[10]*5)
        gate=expansion_gate(scores)
        self.assertTrue(gate['mattervial_signal'])
        self.assertFalse(gate['kan_signal'])
        self.assertTrue(gate['expand'])
        with self.assertRaises(ValueError):
            paired_difference([9]*4,[10]*4)

    def test_synthetic_twenty_fit_pipeline_and_resume_integrity(self):
        rng=np.random.RandomState(13)
        features=PHYSICAL_FEATURES+['ORB_v3_0']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            records=[]
            last_job=None
            for inner in range(5):
                folder=root/'inputs'/str(inner)
                folder.mkdir(parents=True)
                x=rng.uniform(-.5,.5,(24,len(features))).astype('float32')
                xv=rng.uniform(-.5,.5,(6,len(features))).astype('float32')
                y=20+5*x[:,0]
                yv=20+5*xv[:,0]
                np.savez_compressed(folder/'arrays.npz',x_train=x,x_validation=xv,y_train=y,y_validation=yv)
                info=dict(directory=str(folder),train_ids=[f't-{inner}-{i}' for i in range(24)],
                    validation_ids=[f'v-{inner}-{i}' for i in range(6)],selected_features=features,
                    target_scale=float(y.std()),artifacts={'arrays.npz':file_hash(folder/'arrays.npz')})
                write_json(folder/'manifest.json',info)
                for variant in VARIANTS:
                    job=dict(directory=str(root/'fits'/variant/str(inner)),variant=variant,inner=inner,
                        repeat=0,seed=SEEDS[0]+inner,lr=LRS[0],max_epochs=1,run_fingerprint='synthetic-test-only',
                        manifest=str(folder/'manifest.json'),manifest_digest=digest(info))
                    records.append(train_job(job))
                    last_job=job
            summary=summarize(root,records,dict(smoke=True),20,0.)
            self.assertEqual(summary['completed_fits'],20)
            self.assertFalse(summary['outer_test_evaluated'])
            self.assertEqual(train_job(last_job),records[-1])
            with (Path(last_job['directory'])/'validation-predictions.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                train_job(last_job)


if __name__=='__main__':
    unittest.main()
