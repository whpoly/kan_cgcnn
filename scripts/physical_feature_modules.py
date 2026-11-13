"""Four small physical feature modules; no sample-wise LayerNorm or test data."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from modnet.models import MODNetModel


PHYSICAL_GROUPS = {
    "mass": ["ElementProperty|MagpieData mean AtomicWeight", "ElementProperty|MagpieData range AtomicWeight",
             "ElementProperty|MagpieData avg_dev AtomicWeight", "ElementProperty|MagpieData maximum AtomicWeight"],
    "size_density": ["ElementProperty|MagpieData mean CovalentRadius", "ElementProperty|MagpieData avg_dev CovalentRadius",
                     "DensityFeatures|density", "DensityFeatures|vpa"],
    "local_bonding": ["AverageBondLength|mean Average bond length", "AverageBondLength|std_dev Average bond length",
                      "CoordinationNumber|mean CN_VoronoiNN", "StructuralHeterogeneity|avg_dev neighbor distance variation"],
    "electronic": ["ElementProperty|MagpieData mean Electronegativity", "ElementProperty|MagpieData avg_dev Electronegativity",
                   "ElementProperty|MagpieData mean NValence", "ElementProperty|MagpieData avg_dev NValence"],
}
PHYSICAL_FEATURES = [name for names in PHYSICAL_GROUPS.values() for name in names]
VARIANTS = ("modnet", "physical_input", "group_mlp", "spline", "group_sparse_kan")
WIDTHS = [[96], [72], [36], [36]]


def cubic_basis(x):
    """Eight clamped cubic B-splines on five uniform intervals, [-.5,.5]."""
    breaks = np.linspace(-.5, .5, 6)
    knots = tf.cast(np.r_[np.repeat(-.5, 3), breaks, np.repeat(.5, 3)], x.dtype)
    clipped = tf.clip_by_value(x, -.5, .5 - 1e-6)[..., None]
    basis = tf.cast((clipped >= knots[:-1]) & (clipped < knots[1:]), x.dtype)
    for degree in range(1, 4):
        count = 11 - degree
        left = tf.math.divide_no_nan(clipped - knots[:count], knots[degree:degree + count] - knots[:count])
        right = tf.math.divide_no_nan(knots[degree + 1:degree + 1 + count] - clipped,
                                     knots[degree + 1:degree + 1 + count] - knots[1:1 + count])
        basis = left * basis[..., :count] + right * basis[..., 1:count + 1]
    return basis


@tf.keras.utils.register_keras_serializable(package="PhysicalPilot")
class GroupFeatures(tf.keras.layers.Layer):
    """Four groups of four inputs, each producing one feature; 148 parameters.

    Spline and KAN share the additive function class. KAN adds a soft edge/group
    sparsity penalty; both use the same coefficient second-difference penalty.
    These four features are NOT claimed to be explicit cross-group interactions.
    """
    def __init__(self, indices, kind, target_scale=1., sparse_strength=.01,
                 smooth_strength=.001, **kwargs):
        super().__init__(**kwargs)
        if len(indices) != 16 or kind not in ("group_mlp", "spline", "group_sparse_kan"):
            raise ValueError((indices, kind))
        self.indices, self.kind = list(indices), kind
        self.target_scale = float(target_scale)
        self.sparse_strength, self.smooth_strength = float(sparse_strength), float(smooth_strength)

    def build(self, shape):
        if self.kind == "group_mlp":
            # Per group: (4*6+6) + (6*1+1) = 37, total 148 active parameters.
            self.first = self.add_weight("first", (4, 4, 6), initializer="glorot_uniform")
            self.hidden_bias = self.add_weight("hidden_bias", (4, 6), initializer="zeros")
            self.second = self.add_weight("second", (4, 6), initializer="glorot_uniform")
        else:
            self.coefficients = self.add_weight("coefficients", (4, 4, 8), initializer=tf.keras.initializers.RandomNormal(stddev=.05))
            self.base = self.add_weight("base", (4, 4), initializer=tf.keras.initializers.RandomNormal(stddev=.05))
        self.bias = self.add_weight("bias", (4,), initializer="zeros")
        super().build(shape)

    def call(self, inputs):
        x = tf.reshape(tf.gather(inputs, self.indices, axis=-1), (-1, 4, 4))
        if self.kind == "group_mlp":
            hidden = tf.nn.elu(tf.einsum("ngi,gih->ngh", x, self.first) + self.hidden_bias)
            return tf.einsum("ngh,gh->ng", hidden, self.second) + self.bias
        response = tf.einsum("ngib,gib->ng", cubic_basis(x), self.coefficients)
        response += tf.einsum("ngi,gi->ng", x, self.base) + self.bias
        differences = self.coefficients[..., 2:] - 2 * self.coefficients[..., 1:-1] + self.coefficients[..., :-2]
        self.add_loss(self.target_scale * self.smooth_strength * tf.reduce_mean(tf.square(differences)))
        if self.kind == "group_sparse_kan":
            edge_energy = tf.reduce_mean(tf.square(self.coefficients), axis=-1) + tf.square(self.base)
            edge_norm = tf.reduce_mean(tf.sqrt(edge_energy + 1e-8))
            group_norm = tf.reduce_mean(tf.sqrt(tf.reduce_mean(edge_energy, axis=-1) + 1e-8))
            self.add_loss(self.target_scale * self.sparse_strength * (edge_norm + .25 * group_norm))
        return response

    def get_config(self):
        return dict(super().get_config(), indices=self.indices, kind=self.kind, target_scale=self.target_scale,
                    sparse_strength=self.sparse_strength, smooth_strength=self.smooth_strength)


def build_predictor(variant, base_features, input_features, target_name, seed, target_scale):
    """Identical official Dense trunk and shared initial weights for all variants."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    original = MODNetModel(targets=[[[target_name]]], weights={target_name: 1},
                          n_feat=len(base_features), num_neurons=WIDTHS, act="elu")
    if variant == "modnet":
        return original.model
    if variant not in VARIANTS or input_features[:len(base_features)] != base_features:
        raise ValueError("Invalid variant or feature order.")
    inputs = tf.keras.Input(shape=(len(input_features),), name="physical_inputs")
    x = inputs
    if variant != "physical_input":
        positions = [input_features.index(name) for name in PHYSICAL_FEATURES]
        extra = GroupFeatures(positions, variant, target_scale=target_scale, name="physical_module")(inputs)
        x = tf.keras.layers.Concatenate(name="augmented_inputs")([inputs, extra])
    originals = [layer for layer in original.model.layers if isinstance(layer, tf.keras.layers.Dense)]
    for index, old in enumerate(originals):
        new = tf.keras.layers.Dense.from_config(old.get_config())
        x = new(x)
        if index == 0:
            kernel, bias = old.get_weights()
            extended = tf.keras.initializers.GlorotUniform(seed=seed + 109)(new.kernel.shape).numpy()
            extended[:len(base_features)] = kernel
            new.set_weights([extended, bias])
        else:
            new.set_weights(old.get_weights())
    return tf.keras.Model(inputs, x)


def fit_minmax(train, validation):
    from sklearn.preprocessing import MinMaxScaler
    # All 16 physical features and core features use per-column training-only scaling.
    scaler = MinMaxScaler(feature_range=(-.5, .5))
    return scaler, scaler.fit_transform(train).astype("float32"), scaler.transform(validation).astype("float32")
