from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from standalone_kan_models import *


class StandaloneChecks(unittest.TestCase):
    def test_complete_replacement_capacity_and_serialization(self):
        for bank,variant in CONFIGS:
            model=build_model(bank,variant,77)
            n=256 if bank=='enhanced' else 16
            x=np.random.RandomState(8).uniform(-.6,.6,(15,n)).astype('float32')
            self.assertEqual(model.count_params(),COUNTS[bank][variant])
            if variant in ('kan','sparse_kan'):
                self.assertEqual(sum(isinstance(l,SplineEdgeLayer) for l in model.layers),3)
                self.assertFalse(any(isinstance(l,(tf.keras.layers.Dense,tf.keras.layers.Add,tf.keras.layers.LayerNormalization)) for l in model.layers))
                with tf.GradientTape() as tape:loss=tf.reduce_sum(model(x)**2)+sum(model.losses)
                grads=tape.gradient(loss,model.trainable_weights)
                for g in grads:
                    self.assertIsNotNone(g);self.assertTrue(np.isfinite(g.numpy()).all());self.assertGreater(float(abs(g.numpy()).sum()),0)
            restored=tf.keras.models.model_from_json(model.to_json());restored.set_weights(model.get_weights())
            np.testing.assert_allclose(model(x),restored(x),atol=1e-7)

    def test_regularization_is_only_difference(self):
        for bank in ('enhanced','physical'):
            a=build_model(bank,'kan',91);b=build_model(bank,'sparse_kan',91)
            x=np.random.RandomState(11).uniform(-.5,.5,(12,256 if bank=='enhanced' else 16)).astype('float32')
            np.testing.assert_array_equal(a(x),b(x))
            self.assertEqual(sum(a.losses),0);self.assertGreater(float(sum(b.losses)),0)


if __name__=='__main__':unittest.main()
