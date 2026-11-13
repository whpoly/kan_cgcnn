"""Run with the official environment: python -m unittest discover -s tests -p test_official_matched.py."""
import sys
import os
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from modnet_runtime import configure_official_runtime
configure_official_runtime()
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
             "TF_NUM_INTEROP_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
try:
    import tensorflow as tf
    from modnet.models import MODNetModel
    from modnet.hyper_opt.fit_genetic import Individual
    from official_kan_models import FastKAN, MatchedKANModel, fit_kan
except ImportError:
    tf = None


@unittest.skipIf(tf is None, "Requires the isolated MODNet 0.1.12 environment")
class MatchedModelsTest(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        self.kwargs = dict(targets=[[["y"]]], weights={"y": 1}, n_feat=6,
                           num_neurons=[[8], [6], [4], [3]], act="elu")

    def test_head_preserves_trunk_weights_and_official_linear_fourth_block(self):
        baseline = MODNetModel(**self.kwargs)
        weights = [layer.get_weights() for layer in baseline.model.layers[1:-1]]
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        head = MatchedKANModel(**self.kwargs, variant="kan_head")
        self.assertEqual(sum(isinstance(l, FastKAN) for l in head.model.layers), 1)
        for expected, layer in zip(weights, head.model.layers[1:-1]):
            for left, right in zip(expected, layer.get_weights()):
                np.testing.assert_array_equal(left, right)
        self.assertEqual(head.model.layers[-2].activation.__name__, "linear")
        self.assertIs(MatchedKANModel.fit, MODNetModel.fit)
        self.assertIs(MatchedKANModel.predict, MODNetModel.predict)

    def test_full_replaces_all_dense_layers_and_roundtrips_serialization(self):
        full = MatchedKANModel(**self.kwargs, variant="kan_full")
        self.assertEqual(sum(isinstance(l, FastKAN) for l in full.model.layers), 5)
        x = tf.constant(np.random.default_rng(5).normal(size=(4, 6)), dtype=tf.float32)
        expected = full.model(x).numpy()
        full._make_picklable()
        full._restore_model()
        np.testing.assert_allclose(full.model(x), expected, rtol=1e-6, atol=1e-6)

    def test_rbf_formula_and_input_silu(self):
        layer = FastKAN(2, use_layernorm=False)
        x = tf.constant([[0.2, -0.4], [1.0, 0.5]])
        y = layer(x).numpy()
        bases = np.exp(-((x.numpy()[..., None] - np.array([-2, 0, 2])) / 2) ** 2)
        expected = bases.reshape(2, 6) @ layer.spline.kernel.numpy()
        expected += (x.numpy() / (1 + np.exp(-x.numpy()))) @ layer.base.kernel.numpy()
        expected += layer.base.bias.numpy()
        np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-6)

    def test_training_kwargs_equal_official_individual_evaluate(self):
        individual = Individual(max_feat=10, num_classes={"y": 0}, multi_label=False,
                                targets=[[["y"]]], weights={"y": 1})
        with patch.object(MODNetModel, "fit") as fit, \
             patch.object(MODNetModel, "evaluate", return_value=0.5):
            individual.evaluate("train", "val")
            official = fit.call_args.kwargs.copy()
            fit_kan(individual, "train", "val", "kan_head")
            matched = fit.call_args.kwargs.copy()
        for settings in (official, matched):
            callback = settings.pop("callbacks")[0]
            self.assertEqual(callback.monitor, "loss")
            self.assertEqual(callback.patience, 30)
            self.assertEqual(abs(callback.min_delta), 0.001)
            self.assertFalse(callback.restore_best_weights)
        self.assertEqual(official, matched)

    def test_zero_width_fourth_block_allowed_by_original_ga(self):
        options = dict(self.kwargs, num_neurons=[[32], [8], [2], [0]])
        for variant in ("kan_head", "kan_full"):
            model = MatchedKANModel(**options, variant=variant)
            result = model.model(tf.ones((3, 6))).numpy()
            self.assertEqual(result.shape, (3, 1))
            self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
