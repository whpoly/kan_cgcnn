"""Numerical and experimental-design checks for the physical feature pilot."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from modnet_runtime import configure_official_runtime
configure_official_runtime()
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from scipy.interpolate import BSpline
from physical_feature_modules import PHYSICAL_FEATURES, GroupFeatures, cubic_basis, build_predictor, fit_minmax
from run_phonons_physical_pilot import choose, VARIANTS


class PhysicalPilotTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(51)
        self.x = tf.constant(np.random.default_rng(51).uniform(-.5, .5, (8,16)), dtype=tf.float32)

    def test_basis_matches_scipy_partition_and_boundary_behavior(self):
        x = np.array([-.7,-.5,-.4,-.1,0,.3,.49999,.5,.8],dtype=np.float32)
        knots=np.r_[np.repeat(-.5,3),np.linspace(-.5,.5,6),np.repeat(.5,3)]
        expected=np.stack([BSpline(knots,np.eye(8)[i],3)(np.clip(x,-.5,.5-1e-6)) for i in range(8)],axis=-1)
        actual=cubic_basis(tf.constant(x)).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(actual.sum(axis=-1),1,atol=2e-6)
        self.assertTrue((actual>=-1e-7).all())

    def test_modules_have_exact_equal_active_parameter_counts_and_gradients(self):
        self.assertEqual(len(set(PHYSICAL_FEATURES)),16)
        for kind in ("group_mlp","spline","group_sparse_kan"):
            layer=GroupFeatures(list(range(16)),kind)
            with tf.GradientTape() as tape:
                y=layer(self.x)
                loss=tf.reduce_sum(y**2)+sum(layer.losses)
            self.assertEqual(layer.count_params(),148)
            self.assertEqual(tuple(y.shape),(8,4))
            for gradient in tape.gradient(loss,layer.trainable_weights):
                self.assertIsNotNone(gradient)
                self.assertTrue(np.isfinite(gradient.numpy()).all())
                self.assertGreater(np.abs(gradient.numpy()).sum(),0)

    def test_group_output_locality_and_kan_additivity(self):
        for kind in ("group_mlp","spline","group_sparse_kan"):
            layer=GroupFeatures(list(range(16)),kind)
            first=layer(self.x).numpy()
            changed=self.x.numpy().copy(); changed[:,0]+=.1
            second=layer(tf.constant(changed)).numpy()
            np.testing.assert_array_equal(first[:,1:],second[:,1:])
        layer=GroupFeatures(list(range(16)),"group_sparse_kan")
        a=self.x.numpy().copy(); b=a.copy(); c=a.copy(); d=a.copy()
        b[:,0]+=.03; c[:,1]-=.04; d[:,0]+=.03; d[:,1]-=.04
        np.testing.assert_allclose(layer(d)-layer(b)-layer(c)+layer(a),0,atol=1e-6)

    def test_spline_control_differs_only_by_sparse_penalty(self):
        spline=GroupFeatures(list(range(16)),"spline",target_scale=100)
        kan=GroupFeatures(list(range(16)),"group_sparse_kan",target_scale=100)
        spline(self.x); kan(self.x); kan.set_weights(spline.get_weights())
        np.testing.assert_array_equal(spline(self.x),kan(self.x))
        self.assertGreater(float(sum(kan.losses)),float(sum(spline.losses)))

    def test_scaler_uses_training_only_and_does_not_clip_validation(self):
        train=np.array([[1.,4.],[3.,6.]])
        validation=np.array([[9.,-10.]])
        scaler,x,xv=fit_minmax(train,validation)
        np.testing.assert_array_equal(scaler.data_min_,[1,4])
        np.testing.assert_array_equal(scaler.data_max_,[3,6])
        self.assertGreater(xv[0,0],.5)
        self.assertLess(xv[0,1],-.5)

    def test_models_roundtrip_and_share_the_unreplaced_trunk(self):
        core=PHYSICAL_FEATURES[:12]
        inputs=PHYSICAL_FEATURES
        counts=[]
        baseline=build_predictor("modnet",core,core,"y",12,100)
        trunk=[l.get_weights() for l in baseline.layers if isinstance(l,tf.keras.layers.Dense)]
        for variant in VARIANTS:
            features=core if variant=="modnet" else inputs
            model=build_predictor(variant,core,features,"y",12,100)
            x=self.x[:,:len(features)]
            predicted=model(x)
            clone=tf.keras.models.model_from_json(model.to_json())
            clone.set_weights(model.get_weights())
            np.testing.assert_allclose(clone(x),predicted,atol=1e-6,rtol=1e-6)
            dense=[l for l in model.layers if isinstance(l,tf.keras.layers.Dense)]
            np.testing.assert_array_equal(dense[0].get_weights()[0][:len(core)],trunk[0][0])
            for actual,expected in zip(dense[1:],trunk[1:]):
                for a,b in zip(actual.get_weights(),expected): np.testing.assert_array_equal(a,b)
            if variant in ("group_mlp","spline","group_sparse_kan"): counts.append(model.count_params())
        self.assertEqual(len(set(counts)),1)

    def test_selection_requires_all_folds_and_seeds_not_a_lucky_member(self):
        records=[]
        for variant in VARIANTS:
            for lr in (.0005,.001):
                for inner in range(5):
                    for repeat in range(2):
                        score=(0 if inner<4 else 20) if lr==.0005 else 2
                        records.append(dict(variant=variant,lr=lr,inner=inner,seed_repeat=repeat,
                                            validation_mae=score,directory=f"{variant}-{lr}-{inner}-{repeat}"))
        winners=choose(records,2)
        self.assertTrue(all(w["winner"]["lr"]==.001 for w in winners.values()))
        with self.assertRaises(ValueError): choose(records[:-1],2)


if __name__=="__main__": unittest.main()
