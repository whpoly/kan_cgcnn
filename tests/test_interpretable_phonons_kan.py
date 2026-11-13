from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from interpretable_phonons_kan import *


class PhysicalChecks(unittest.TestCase):
    def setUp(self):
        self.x=np.random.RandomState(8).uniform(-.5,.5,(32,256)).astype('float32')
        self.features=PHYSICAL_FEATURES+[f'other{i}' for i in range(240)]

    def test_basis_and_train_centered_exact_decomposition(self):
        values=np.array([-.7,-.5,-.3,0,.2,.5,.7],dtype='float32')
        np.testing.assert_allclose(basis_numpy(values)[:,:8],cubic_basis(values),atol=2e-6)
        for v in ('physical_additive','physical_product','hybrid_additive','hybrid_product'):
            model=build_model(v,self.features,self.x,17)
            layer=model.get_layer('physical_response')
            if v.startswith('hybrid'): layer.gate.assign(.7)
            terms,bias=decompose(self.x[:,:16],layer.export())
            np.testing.assert_allclose(terms.mean(axis=0),0,atol=1e-7)
            actual=tf.keras.Model(model.input,layer.output)(self.x).numpy().reshape(-1)
            np.testing.assert_allclose(terms.sum(axis=1)+bias,actual,atol=1e-6)

    def test_capacity_roundtrip_and_physical_input_isolation(self):
        for v in VARIANTS:
            model=build_model(v,self.features,self.x,22)
            self.assertEqual(model.count_params(),expected_parameters(v))
            restored=tf.keras.models.model_from_json(model.to_json())
            restored.set_weights(model.get_weights())
            np.testing.assert_allclose(model(self.x),restored(self.x),atol=1e-6)
            if v.startswith('physical'):
                changed=self.x.copy();changed[:,16:]=20
                np.testing.assert_array_equal(model(changed),model(self.x))

    def test_initial_hybrid_predictions_and_active_gradients(self):
        predictions=[]
        for v in VARIANTS:
            model=build_model(v,self.features,self.x,33)
            layer=model.get_layer('physical_response')
            if v.startswith('hybrid'):
                predictions.append(model(self.x).numpy())
                layer.gate.assign(.5)
            with tf.GradientTape() as tape:
                loss=tf.reduce_sum(model(self.x)**2)
            for g in tape.gradient(loss,layer.trainable_weights):
                self.assertIsNotNone(g)
                self.assertTrue(np.isfinite(g.numpy()).all())
                self.assertGreater(float(abs(g.numpy()).sum()),0.)
        for p in predictions[1:]: np.testing.assert_array_equal(p,predictions[0])


if __name__=='__main__': unittest.main()
