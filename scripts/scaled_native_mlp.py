"""Native MODNet with the same train-only target transform as the spline arm."""
import copy
from run_modnet_official_matched import MODNetModel, np, tf
from official_kan_models import neurons_from_genes


def evaluate_standardized(individual, train_data, val_data, fast=False):
    assert list(train_data.target_names) == list(val_data.target_names)
    assert train_data.df_targets.shape[1] == 1
    y = train_data.df_targets.values.copy()
    mean, std = float(y.mean()), float(y.std())
    assert np.isfinite(std) and std > 0
    normalized = []
    for data in (train_data, val_data):
        clone = copy.copy(data)
        clone.df_targets = (data.df_targets.copy() - mean) / std
        normalized.append(clone)
    genes = individual.genes
    model = MODNetModel(targets=individual.targets, weights=individual.weights,
        num_classes=individual.num_classes, multi_label=individual.multi_label,
        n_feat=genes['n_feat'], num_neurons=neurons_from_genes(genes), act=genes['act'])
    model.fit(normalized[0], val_data=normalized[1], loss=genes['loss'], lr=genes['lr'],
        epochs=1 if fast else 800, batch_size=genes['batch_size'], xscale=genes['xscale'], verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30,
            min_delta=.001/std, restore_best_weights=False)])
    parameters = model.model.count_params()
    output = tf.keras.layers.Rescaling(std, mean, dtype='float64',
        name='physical_target_units')(model.model.output)
    model.model = tf.keras.Model(model.model.input, output)
    assert model.model.count_params() == parameters
    model.min_y, model.max_y = y.min(axis=0), y.max(axis=0)
    model.target_scaling = dict(mean=mean, std=std, ddof=0,
        fit_ids=list(train_data.df_targets.index), prediction_units='physical',
        early_stopping_delta_physical=.001, early_stopping_delta_normalized=.001/std)
    model.normalized_history = copy.deepcopy(model.history)
    # Match the existing spline adapter's recorded physical loss curves.
    for key in ('loss', 'val_loss', 'mean_absolute_error', 'val_mean_absolute_error'):
        if key in model.history:
            model.history[key] = [float(value)*std for value in model.history[key]]
    individual.val_loss = model.evaluate(val_data)
    individual.model = model
