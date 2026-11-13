"""Train-only target scaling for the native B-spline adapter.

Stored models predict physical units, including after native pickle restore.
No trainable output layer is introduced and no validation statistics are used.
"""
import copy
from run_native_fourway import NativeSplineModel, neurons_from_genes, tf, np


def fit_spline(individual, train, val, variant, fast=False, seed=0):
    assert list(train.target_names) == list(val.target_names)
    assert train.df_targets.shape[1] == 1
    y = train.df_targets.values.copy()
    mean, std = float(y.mean()), float(y.std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError('Invalid training target standard deviation')
    normalized = []
    for data in (train, val):
        clone = copy.copy(data)
        clone.df_targets = (data.df_targets.copy() - mean) / std
        normalized.append(clone)
    g = individual.genes
    model = NativeSplineModel(targets=individual.targets, weights=individual.weights,
        num_classes=individual.num_classes, multi_label=False, n_feat=g['n_feat'],
        num_neurons=neurons_from_genes(g), act=g['act'], spline_seed=seed, input_scale=g['xscale'])
    model.fit(normalized[0], val_data=normalized[1], loss=g['loss'], lr=g['lr'],
        epochs=1 if fast else 800, batch_size=g['batch_size'], xscale=g['xscale'], verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30,
            min_delta=.001/std, restore_best_weights=False)])
    # The inverse transform is part of the serialized inference graph. This is
    # required because the native ensemble calls model.model.predict directly.
    parameters = model.model.count_params()
    output = tf.keras.layers.Rescaling(std, mean, dtype='float64',
        name='physical_target_units')(model.model.output)
    model.model = tf.keras.Model(model.model.input, output)
    assert model.model.count_params() == parameters
    model.min_y, model.max_y = y.min(axis=0), y.max(axis=0)
    model.target_scaling = dict(mean=mean, std=std, ddof=0,
        fit_ids=list(train.df_targets.index), prediction_units='physical',
        early_stopping_delta_physical=.001, early_stopping_delta_normalized=.001/std)
    model.normalized_history = copy.deepcopy(model.history)
    for key in ('loss', 'val_loss', 'mean_absolute_error', 'val_mean_absolute_error'):
        if key in model.history:
            model.history[key] = [float(value)*std for value in model.history[key]]
    individual.val_loss = model.evaluate(val)
    individual.model = model
