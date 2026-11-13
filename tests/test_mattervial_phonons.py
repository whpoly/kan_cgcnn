"""Scientific-integrity checks for the new MatterVial pipeline."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_mattervial_phonons import (np, pd, tf, make_data, prepare_inner, read,
    feature_group, build_control, choose_controls, VARIANTS, PHYSICAL_FEATURES)


class MatterVialChecks(unittest.TestCase):
    def test_feature_selection_is_invariant_to_validation_labels_and_values(self):
        rng = np.random.RandomState(17)
        x = pd.DataFrame(rng.normal(size=(40, 6)), index=[f"train-{i}" for i in range(40)],
                         columns=[f"feature-{j}" for j in range(6)])
        y = 3 * x.iloc[:, 2].values + rng.normal(size=40)
        xv = pd.DataFrame(rng.normal(size=(10, 6)), index=[f"val-{i}" for i in range(10)], columns=x.columns)
        with tempfile.TemporaryDirectory() as tmp:
            features, matrices = [], []
            for index, multiplier in enumerate((1, 10000)):
                train = make_data(x.copy(), y)
                val = make_data(xv * multiplier, np.arange(10) * multiplier)
                directory = Path(tmp) / str(index)
                prepared, _ = prepare_inner(train, val, directory, "test", 17, 4, 1)
                info = read(directory / "manifest.json")
                self.assertEqual(info["feature_selection_fit_ids"], list(x.index))
                self.assertEqual(info["cross_nmi_fit_ids"], list(x.index))
                features.append(info["selected_features"])
                matrices.append(prepared.cross_nmi)
            self.assertEqual(features[0], features[1])
            pd.testing.assert_frame_equal(matrices[0], matrices[1])
            # Altering cached data must not silently resume.
            p = Path(tmp) / "0/train.pkl.gz"
            with p.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                prepare_inner(make_data(x, y), make_data(xv, np.arange(10)), Path(tmp) / "0", "test", 17, 4, 1)

    def test_feature_groups_exclude_unrequested_embeddings_and_metadata(self):
        for name in ("target", "structure", "material_id", "ROOST_oqmd_x", "MEGNet_MatMinerEncoded_v1_1"):
            self.assertIsNone(feature_group(name))
        self.assertEqual(feature_group("MEGNet_OFMEncoded_v1_0"), "megnet_ofm")
        self.assertEqual(feature_group("SISSO_matbench_phonons_1"), "sisso")
        self.assertEqual(feature_group(PHYSICAL_FEATURES[0]), "matminer")

    def test_equal_capacity_shared_trunk_and_reload(self):
        genes = dict(n_neurons_first_layer=32, fraction1=.75, fraction2=.5, fraction3=.5, act="elu")
        features = ["extra"] + PHYSICAL_FEATURES
        x = np.random.RandomState(21).uniform(-.5, .5, (9, len(features))).astype("float32")
        models = [build_control(v, features, genes, 21, 50) for v in VARIANTS]
        self.assertEqual(models[1].count_params(), models[2].count_params())
        self.assertEqual(models[1].count_params() - models[0].count_params(), 148 + 4 * 32)
        baseline = [l.get_weights() for l in models[0].layers if isinstance(l, tf.keras.layers.Dense)]
        for model in models:
            dense = [l.get_weights() for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
            np.testing.assert_array_equal(dense[0][0][:len(features)], baseline[0][0])
            for actual, expected in zip(dense[1:], baseline[1:]):
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a, b)
            clone = tf.keras.models.model_from_json(model.to_json())
            clone.set_weights(model.get_weights())
            np.testing.assert_allclose(clone(x), model(x), rtol=1e-6, atol=1e-6)

    def test_candidate_selection_requires_complete_inner_folds(self):
        rows = []
        for variant in VARIANTS:
            for factor in (.5, 1., 2.):
                for fold in range(5):
                    mae = ((0 if fold < 4 else 100) if factor == .5 else 10 * factor)
                    rows.append(dict(variant=variant, template=0, lr_factor=factor, inner=fold,
                        validation_mae=mae, directory=f"{variant}/{factor}/{fold}"))
        choices = choose_controls(rows, 1, 5)
        for choice in choices.values():
            self.assertEqual(choice["candidates"][0]["selected"]["lr_factor"], 1.)
            self.assertEqual(len(choice["members"]), 5)
        with self.assertRaises(ValueError):
            choose_controls(rows[:-1], 1, 5)


if __name__ == "__main__":
    unittest.main()
