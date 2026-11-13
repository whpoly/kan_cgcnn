"""TensorFlow KAN replacements that inherit MODNet 0.1.12's fit/predict code.

No alternate scaler, target transform, optimizer, stopping rule or prediction
postprocessing is introduced here. Single-target regression only.
"""
from __future__ import annotations

import tensorflow as tf
from modnet.models import MODNetModel


@tf.keras.utils.register_keras_serializable(package="MatchedMODNet")
class FastKAN(tf.keras.layers.Layer):
    """Three Gaussian bases per input plus a SiLU/Dense residual branch."""

    def __init__(self, units, num_grids=3, grid_min=-2.0, grid_max=2.0,
                 use_layernorm=True, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.num_grids = int(num_grids)
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.use_layernorm = bool(use_layernorm)
        self.norm = None
        # Activation belongs on the input, not after the output projection.
        self.base = tf.keras.layers.Dense(self.units)
        self.spline = tf.keras.layers.Dense(
            self.units, use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))

    def build(self, input_shape):
        self.input_width = int(input_shape[-1])
        if self.use_layernorm and self.input_width > 1:
            self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        super().build(input_shape)

    def call(self, inputs):
        z = self.norm(inputs) if self.norm is not None else inputs
        grid = tf.linspace(tf.cast(self.grid_min, inputs.dtype),
                           tf.cast(self.grid_max, inputs.dtype), self.num_grids)
        width = (self.grid_max - self.grid_min) / (self.num_grids - 1)
        bases = tf.exp(-tf.square((z[..., None] - grid) / width))
        bases = tf.reshape(bases, [tf.shape(inputs)[0], self.input_width * self.num_grids])
        return self.spline(bases) + self.base(tf.nn.silu(inputs))

    def get_config(self):
        return dict(super().get_config(), units=self.units, num_grids=self.num_grids,
                    grid_min=self.grid_min, grid_max=self.grid_max,
                    use_layernorm=self.use_layernorm)


class MatchedKANModel(MODNetModel):
    """Replace only the final Dense, or every Dense, preserving official fit."""

    def __init__(self, *args, variant="kan_head", **kwargs):
        if variant not in ("kan_head", "kan_full"):
            raise ValueError(variant)
        self.variant = variant
        super().__init__(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        original = super().build_model(*args, **kwargs)
        if self._multi_target or any(self.num_classes.values()):
            raise ValueError("This matched benchmark supports single-target regression.")
        output_names = set(original.output_names)

        def clone(layer):
            if isinstance(layer, tf.keras.layers.Dense) and (
                self.variant == "kan_full" or layer.name in output_names
            ):
                return FastKAN(layer.units, name=layer.name)
            return layer.__class__.from_config(layer.get_config())

        replacement = tf.keras.models.clone_model(original, clone_function=clone)
        # Head-only pairs start with identical trunk weights for the same seed.
        for old in original.layers:
            new = replacement.get_layer(old.name)
            if not isinstance(new, FastKAN):
                new.set_weights(old.get_weights())
        return replacement


def neurons_from_genes(genes):
    first = genes["n_neurons_first_layer"]
    return [[int(first)], [int(first * genes["fraction1"])],
            [int(first * genes["fraction1"] * genes["fraction2"])],
            [int(first * genes["fraction1"] * genes["fraction2"] * genes["fraction3"])]]


def fit_kan(individual, training_data, validation_data, variant, fast=False):
    """Mirror Individual.evaluate, changing only the model's Dense layers."""
    genes = individual.genes
    model = MatchedKANModel(
        targets=individual.targets, weights=individual.weights,
        n_feat=genes["n_feat"], num_neurons=neurons_from_genes(genes),
        act=genes["act"], num_classes=individual.num_classes,
        multi_label=individual.multi_label, variant=variant)
    model.fit(
        training_data, val_data=validation_data, loss=genes["loss"],
        lr=genes["lr"], epochs=1 if fast else 800,
        batch_size=genes["batch_size"], xscale=genes["xscale"],
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="loss", min_delta=0.001, patience=30,
            restore_best_weights=False)], verbose=0)
    individual.val_loss = model.evaluate(validation_data)
    individual.model = model
    return individual
