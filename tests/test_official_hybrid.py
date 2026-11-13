"""Meaningful architecture, capacity and inner-selection checks for round two."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from modnet_runtime import configure_official_runtime
configure_official_runtime()
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
try:
    import tensorflow as tf
    from modnet.models import MODNetModel
    from official_hybrid_models import (HybridMODNetModel, ResidualReadout, DenseMLP,
                                       fastkan_parameters, mlp_width, fit_hybrid)
    from official_kan_models import FastKAN
    from run_modnet_hybrid_nested import choose_trials, recorded_inner_positions
except ImportError:
    tf = None


@unittest.skipIf(tf is None, "Requires MODNet 0.1.12 and TensorFlow")
class HybridTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        self.options = dict(targets=[[["y"]]], weights={"y": 1}, n_feat=6,
                            num_neurons=[[16], [12], [8], [6]], act="elu")
        self.x = tf.constant(np.random.default_rng(4).normal(size=(5, 6)), dtype=tf.float32)

    def test_both_residual_heads_start_exactly_as_the_paired_modnet(self):
        original = MODNetModel(**self.options)
        expected = original.model(self.x).numpy()
        for variant in ("residual_kan", "residual_mlp"):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(42)
            model = HybridMODNetModel(**self.options, variant=variant, grids=5)
            np.testing.assert_array_equal(model.model(self.x).numpy(), expected)
            self.assertEqual(float(model.model.layers[-1].alpha), 0)
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(model.model(self.x) - 5.0))
            gradient = tape.gradient(loss, model.model.layers[-1].alpha)
            self.assertTrue(np.isfinite(float(gradient)))
            self.assertNotEqual(float(gradient), 0)

    def test_last_hidden_replacement_preserves_other_blocks_and_linear_output(self):
        original = MODNetModel(**self.options)
        weights = [layer.get_weights() for layer in original.model.layers]
        for variant, replacement in (("last_hidden_kan", FastKAN), ("last_hidden_mlp", DenseMLP)):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(42)
            model = HybridMODNetModel(**self.options, variant=variant, grids=8)
            self.assertIsInstance(model.model.layers[-2], replacement)
            self.assertIsInstance(model.model.layers[-1], tf.keras.layers.Dense)
            self.assertEqual(model.model.layers[-1].activation.__name__, "linear")
            for index, layer in enumerate(model.model.layers):
                if index == len(model.model.layers) - 2:
                    continue
                for actual, expected in zip(layer.get_weights(), weights[index]):
                    np.testing.assert_array_equal(actual, expected)

    def test_parameter_pairs_match_actual_keras_counts(self):
        for grids in (3, 5, 8):
            for kan, mlp in (("residual_kan", "residual_mlp"), ("last_hidden_kan", "last_hidden_mlp")):
                left = HybridMODNetModel(**self.options, variant=kan, grids=grids)
                right = HybridMODNetModel(**self.options, variant=mlp, grids=grids)
                self.assertEqual(left.model.count_params(), left.target_parameters)
                self.assertEqual(left.target_parameters, right.target_parameters)
                self.assertLessEqual(abs(right.model.count_params() / left.model.count_params() - 1), .01)

    def test_all_variants_roundtrip_trained_gate_and_predictions(self):
        for variant in ("residual_kan", "last_hidden_kan", "residual_mlp", "last_hidden_mlp"):
            model = HybridMODNetModel(**self.options, variant=variant, grids=5)
            if isinstance(model.model.layers[-1], ResidualReadout):
                model.model.layers[-1].alpha.assign(.3)
            expected = model.model(self.x).numpy()
            model._make_picklable()
            model._restore_model()
            np.testing.assert_allclose(model.model(self.x), expected, rtol=1e-6, atol=1e-6)

    def test_fit_keeps_official_training_settings_except_declared_lr(self):
        genes = dict(n_feat=6, n_neurons_first_layer=16, fraction1=.75,
                     fraction2=.5, fraction3=1., act="elu", loss="mae", lr=.005,
                     batch_size=32, xscale="minmax")
        with patch.object(MODNetModel, "fit") as fit, patch.object(MODNetModel, "evaluate", return_value=.1):
            fit_hybrid(genes, SimpleNamespace(target_names=["y"]), "validation",
                       variant="residual_kan", grids=3, lr_multiplier=.5, seed=42)
        kw = fit.call_args.kwargs
        self.assertEqual(kw["lr"], .0025)
        self.assertEqual(kw["epochs"], 800)
        self.assertEqual(kw["batch_size"], 32)
        self.assertEqual(kw["xscale"], "minmax")
        self.assertEqual(kw["val_data"], "validation")
        stop = kw["callbacks"][0]
        self.assertEqual((stop.monitor, stop.patience, stop.restore_best_weights), ("loss", 30, False))
        self.assertEqual(abs(stop.min_delta), .001)

    def test_selection_uses_mean_of_all_inner_folds_and_stable_tie_break(self):
        records = []
        for trial, losses in enumerate(([0, 0, 0, 0, 100], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2])):
            for inner, loss in enumerate(losses):
                records.append(dict(base=0, trial=trial, inner=inner, validation_mae=loss,
                    grids=3, lr_multiplier=1, directory=f"trial-{trial}-inner-{inner}", seed=inner,
                    parameters=100, target_parameters=100, source_member="source"))
        winners, table = choose_trials(records, 3)
        self.assertEqual(winners[0]["trial"], 1)
        self.assertEqual(winners[0]["mean_validation_mae"], 2)
        with self.assertRaises(ValueError):
            choose_trials(records[:-1], 3)

    def test_saved_inner_order_is_preserved_after_upstream_ga_shuffle(self):
        outer = ["a", "b", "c", "d"]
        source = dict(train_ids=["d", "b", "a"], validation_ids=["c"])
        train, val = recorded_inner_positions(outer, source)
        self.assertEqual(train.tolist(), [3, 1, 0])
        self.assertEqual(val.tolist(), [2])
        with self.assertRaises(ValueError):
            recorded_inner_positions(outer, dict(train_ids=["a", "b"], validation_ids=["b", "c", "d"]))


if __name__ == "__main__":
    unittest.main()
