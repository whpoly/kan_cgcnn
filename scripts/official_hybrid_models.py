"""Round-two KAN ablations and active, parameter-matched MLP controls."""
from __future__ import annotations

import math
import tensorflow as tf
from modnet.models import MODNetModel
from official_kan_models import FastKAN, neurons_from_genes

VARIANTS = ("residual_kan", "last_hidden_kan", "residual_mlp", "last_hidden_mlp")
PAIRS = {"residual_kan": "residual_mlp", "last_hidden_kan": "last_hidden_mlp"}


def fastkan_parameters(inputs, outputs, grids):
    return (grids + 1) * inputs * outputs + outputs + (2 * inputs if inputs > 1 else 0)


def mlp_width(inputs, outputs, target):
    """Nearest active two-Dense parameter count; no unused padding weights."""
    ideal = (target - outputs) / (inputs + outputs + 1)
    options = {max(1, math.floor(ideal)), max(1, math.ceil(ideal))}
    return min(options, key=lambda n: (abs(n * (inputs + outputs + 1) + outputs - target), n))


@tf.keras.utils.register_keras_serializable(package="MatchedHybrid")
class DenseMLP(tf.keras.layers.Layer):
    def __init__(self, units, width, **kwargs):
        super().__init__(**kwargs)
        self.units, self.width = int(units), int(width)
        self.hidden = tf.keras.layers.Dense(self.width, activation="elu")
        self.output_layer = tf.keras.layers.Dense(self.units)

    def call(self, inputs):
        return self.output_layer(self.hidden(inputs))

    def get_config(self):
        return dict(super().get_config(), units=self.units, width=self.width)


@tf.keras.utils.register_keras_serializable(package="MatchedHybrid")
class ResidualReadout(tf.keras.layers.Layer):
    """Original linear readout + zero-gated KAN or matched MLP correction."""
    def __init__(self, units, grids=3, correction="kan", width=None, **kwargs):
        super().__init__(**kwargs)
        self.units, self.grids = int(units), int(grids)
        self.correction_kind, self.width = correction, width
        self.linear = tf.keras.layers.Dense(self.units)
        if correction == "kan":
            self.correction = FastKAN(self.units, num_grids=self.grids)
        elif correction == "mlp":
            self.correction = DenseMLP(self.units, int(width))
        else:
            raise ValueError(correction)
        self.alpha = self.add_weight(name="alpha", shape=(), initializer="zeros", trainable=True)

    def call(self, inputs):
        return self.linear(inputs) + self.alpha * self.correction(inputs)

    def get_config(self):
        return dict(super().get_config(), units=self.units, grids=self.grids,
                    correction=self.correction_kind, width=self.width)


class HybridMODNetModel(MODNetModel):
    def __init__(self, *args, variant, grids=3, parameter_tolerance=0.01, **kwargs):
        if variant not in VARIANTS or grids < 2:
            raise ValueError((variant, grids))
        self.variant, self.grids = variant, int(grids)
        self.parameter_tolerance = parameter_tolerance
        super().__init__(*args, **kwargs)

    def build_model(self, *args, **kwargs):
        original = super().build_model(*args, **kwargs)
        if self._multi_target or any(self.num_classes.values()):
            raise ValueError("Single-target regression only.")
        dense = [layer for layer in original.layers if isinstance(layer, tf.keras.layers.Dense)]
        if len(dense) != 5:
            raise ValueError("Expected the official four-hidden-block single-target architecture.")
        last_hidden, output = dense[-2:]
        target = output if self.variant.startswith("residual") else last_hidden
        inputs, units = int(target.input_shape[-1]), target.units
        if inputs < 1 or units < 1:
            raise ValueError("Round-two selected configurations must have positive layer widths.")
        branch_params = fastkan_parameters(inputs, units, self.grids)
        width = mlp_width(inputs, units, branch_params)
        base_parameters = original.count_params()
        self.target_parameters = (base_parameters + branch_params + 1
                                  if self.variant.startswith("residual")
                                  else base_parameters - target.count_params() + branch_params)

        def clone(layer):
            if layer.name != target.name:
                return layer.__class__.from_config(layer.get_config())
            if self.variant.startswith("residual"):
                return ResidualReadout(units, self.grids,
                    correction="kan" if self.variant == "residual_kan" else "mlp",
                    width=width if self.variant == "residual_mlp" else None, name=layer.name)
            if self.variant == "last_hidden_kan":
                return FastKAN(units, num_grids=self.grids, name=layer.name)
            return DenseMLP(units, width, name=layer.name)

        replacement = tf.keras.models.clone_model(original, clone_function=clone)
        for old in original.layers:
            new = replacement.get_layer(old.name)
            if old.name != target.name:
                new.set_weights(old.get_weights())
            elif isinstance(new, ResidualReadout):
                new.linear.set_weights(old.get_weights())
        self.parameter_error = (replacement.count_params() / self.target_parameters) - 1
        if abs(self.parameter_error) > self.parameter_tolerance:
            raise ValueError(f"MLP parameter mismatch exceeds tolerance: {self.parameter_error}")
        self.mlp_hidden_width = width if self.variant.endswith("mlp") else None
        return replacement


def fit_hybrid(genes, train, validation, *, variant, grids, lr_multiplier,
               seed, smoke=False, callbacks=()):
    targets = [[[name] for name in train.target_names]]
    if len(train.target_names) != 1:
        raise ValueError("Only one regression target is supported.")
    model = HybridMODNetModel(targets=targets, weights={train.target_names[0]: 1},
                             n_feat=genes["n_feat"], num_neurons=neurons_from_genes(genes),
                             act=genes["act"], variant=variant, grids=grids)
    # Preserve the paired member's minibatch RNG after architecture initialization.
    tf.keras.utils.set_random_seed(seed)
    model.fit(train, val_data=validation, loss=genes["loss"],
              lr=genes["lr"] * lr_multiplier, epochs=1 if smoke else 800,
              batch_size=genes["batch_size"], xscale=genes["xscale"],
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.001,
                         patience=30, restore_best_weights=False), *callbacks], verbose=0)
    return model, float(model.evaluate(validation))
