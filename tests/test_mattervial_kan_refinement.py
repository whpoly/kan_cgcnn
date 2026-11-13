"""Numerics, matching, validation checkpoint, and selection-scope checks."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_mattervial_kan_refinement import (np,tf,target_statistics,train_job,choose,delta,LRS,
    write_json,file_hash,digest,VARIANTS,build_model)
from physical_feature_modules import PHYSICAL_FEATURES


class RefinementChecks(unittest.TestCase):
    def setUp(self):
        self.features=PHYSICAL_FEATURES+[f'feature_{i}' for i in range(240)]
        self.x=np.random.RandomState(3).uniform(-.5,.5,(12,256)).astype('float32')

    def test_capacity_pairing_and_initial_prediction_preservation(self):
        models={v:build_model(v,self.features,17) for v in VARIANTS}
        expected=dict(modnet=35653,group_mlp=36185,group_kan=36185,group_sparse_diagnostic=36185,latent_mlp=36391,latent_kan=36391)
        self.assertEqual({v:m.count_params() for v,m in models.items()},expected)
        base=models['modnet'](self.x)
        if isinstance(base,list):base=base[0]
        for v in ('latent_mlp','latent_kan'):
            np.testing.assert_array_equal(models[v](self.x),base)
        np.testing.assert_array_equal(models['group_kan'](self.x),models['group_sparse_diagnostic'](self.x))
        self.assertEqual(float(sum(models['group_kan'].losses)),0.)
        self.assertGreater(float(sum(models['group_sparse_diagnostic'].losses)),0.)
        a=models['latent_mlp'].get_layer('latent_correction').projection.get_weights()
        b=models['latent_kan'].get_layer('latent_correction').projection.get_weights()
        for w1,w2 in zip(a,b):np.testing.assert_array_equal(w1,w2)

    def test_residual_weights_are_active_and_models_roundtrip(self):
        for variant in ('latent_mlp','latent_kan'):
            model=build_model(variant,self.features,21)
            branch=model.get_layer('latent_correction')
            branch.gate.assign(.1)
            with tf.GradientTape() as tape:
                loss=tf.reduce_sum(model(self.x)**2)
            grads=tape.gradient(loss,branch.trainable_weights)
            for grad in grads:
                self.assertIsNotNone(grad)
                self.assertTrue(np.isfinite(grad.numpy()).all())
                self.assertGreater(float(np.abs(grad.numpy()).sum()),0.)
            restored=tf.keras.models.model_from_json(model.to_json())
            restored.set_weights(model.get_weights())
            np.testing.assert_allclose(restored(self.x),model(self.x),rtol=1e-6,atol=1e-6)
            self.assertEqual(branch.correction.count_params(),145)

    def test_target_statistics_are_training_only(self):
        train=np.array([100.,300.,500.])
        mean,std=target_statistics(train)
        self.assertEqual(mean,300.)
        np.testing.assert_allclose((train-mean)/std,[-np.sqrt(1.5),0,np.sqrt(1.5)])
        with self.assertRaises(ValueError):target_statistics([4.,4.])

    def test_complete_grid_selection_excludes_diagnostic_and_lucky_fold(self):
        records=[]
        for v in VARIANTS:
            for lr in LRS:
                for inner in range(5):
                    if v=='group_sparse_diagnostic':score=.1
                    elif v=='modnet':score=11.
                    elif v=='latent_kan':score=(9. if lr==.001 else (1. if inner<4 else 100.))
                    else:score=10.
                    records.append(dict(variant=v,lr=lr,inner=inner,repeat=0,validation_mae=score,directory=f'{v}-{lr}-{inner}'))
        selected=choose(records)
        self.assertEqual(selected['best_kan']['variant'],'latent_kan')
        self.assertEqual(selected['best_kan']['lr'],.001)
        self.assertTrue(selected['confirm'])
        with self.assertRaises(ValueError):choose(records[:-1])
        self.assertFalse(delta([1,11,11,11,11],[10]*5)['passes'])

    def test_real_fit_checkpoint_units_and_resume_hashes(self):
        rng=np.random.RandomState(23)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'inputs';inputs.mkdir()
            x=rng.uniform(-.5,.5,(40,256)).astype('float32');xv=rng.uniform(-.5,.5,(8,256)).astype('float32')
            y=500+100*x[:,0];yv=500+100*xv[:,0]
            np.savez_compressed(inputs/'arrays.npz',x_train=x,x_validation=xv,y_train=y,y_validation=yv)
            info=dict(directory=str(inputs),selected_features=self.features,train_ids=[f't{i}' for i in range(40)],
                validation_ids=[f'v{i}' for i in range(8)],artifacts={'arrays.npz':file_hash(inputs/'arrays.npz')})
            write_json(inputs/'manifest.json',info)
            mean,std=target_statistics(y)
            inp=dict(source=info,source_manifest=str(inputs/'manifest.json'),source_manifest_sha256=file_hash(inputs/'manifest.json'),
                target_mean=mean,target_std=std,target_transform_fit_ids=info['train_ids'])
            for variant in VARIANTS:
                params=35653 if variant=='modnet' else (36391 if variant.startswith('latent_') else 36185)
                job=dict(directory=str(root/variant),variant=variant,inner=0,repeat=0,lr=.001,seed=23,input=inp,
                    max_epochs=2,expected_parameters=params,run_fingerprint='test-only')
                record=train_job(job)
                self.assertLessEqual(abs(record['validation_mae']-record['checkpoint_metric_mae']),.003)
                self.assertEqual(train_job(job),record)
            with (root/variant/'validation-predictions.npz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):train_job(job)


if __name__=='__main__':
    unittest.main()
