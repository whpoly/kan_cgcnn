"""Read-only diagnostics of frozen phonons models; performs no model fitting."""
import json
import pickle
from pathlib import Path

from run_modnet_official_matched import np, tf, MODData, file_hash, write_json
from native_fourway_features import ROOT, OLD, read
from standalone_kan_models import SplineEdgeLayer


def main():
    run = ROOT / 'benchmarks/phonons-rawtarget-modnet012-20261115'
    out = ROOT / 'benchmarks/phonons-rawtarget-diagnosis-20261115'
    out.mkdir(parents=True, exist_ok=True)
    training = MODData.load(OLD / 'matbench_phonons/fold-0/official-train-moddata.pkl.gz')
    q95 = float(np.quantile(training.df_targets.values, .95))
    result = dict(scope='Completed outer fold 0; diagnostics only, no training or model selection',
                  training_target_q95=q95, models={}, saved_model_diagnostics=[])
    predictions = {}
    for arm in ('mm_modnet', 'mm_kan'):
        directory = run / arm / 'fold-0'
        complete = read(directory / 'complete.json')
        assert file_hash(directory / 'outer-predictions.npz') == complete['prediction_sha256']
        selected = read(directory / 'frozen-selection.json')['members']
        records = [read(Path(j['directory']) / 'result.json') for j in selected]
        with np.load(directory / 'outer-predictions.npz') as z:
            y, pred = z['target'].copy(), z['prediction'].copy()
            predictions[arm] = (z['sample_ids'].copy(), y, pred)
        errors = np.abs(pred-y)
        stats = dict(test_mae=float(errors.mean()), prediction_max=float(pred.max()),
                     target_max=float(y.max()), n_test=len(y), epochs800=sum(r['epochs'] == 800 for r in records),
                     mean_epochs=float(np.mean([r['epochs'] for r in records])),
                     mean_final_epoch_training_mae=float(np.mean([r['history']['loss'][-1] for r in records])),
                     mean_member_validation_mae=float(np.mean([r['validation_mae'] for r in records])),
                     parameters_mean=complete['parameters_mean'], high_target_tail_count=int(np.sum(y > q95)),
                     high_target_tail_mae=float(errors[y > q95].mean()),
                     remainder_mae=float(errors[y <= q95].mean()),
                     high_target_tail_fraction_of_absolute_error=float(errors[y > q95].sum()/errors.sum()))
        result['models'][arm] = stats
        if arm != 'mm_kan':
            continue
        for index, (job, rec) in enumerate(zip(selected, records)):
            path = Path(job['directory']) / 'model.pkl'
            assert file_hash(path) == rec['model_sha256']
            with path.open('rb') as stream:
                model = pickle.load(stream)
            assert getattr(model, 'target_scaling', None) is None
            model._restore_model()
            layers = [layer for layer in model.model.layers if isinstance(layer, SplineEdgeLayer)]
            assert len(layers) == 3
            coeff = layers[-1].coefficients.numpy()[:, 0, :].astype(float)
            bias = float(layers[-1].bias.numpy()[0])
            item = dict(member=index, source=str(path), lower_bound=float(coeff.min(axis=1).sum()+bias),
                        upper_bound=float(coeff.max(axis=1).sum()+bias), epochs=rec['epochs'])
            if index < 5:
                frame = training.df_featurized.loc[rec['train_ids'], rec['selected_features']]
                values = np.nan_to_num(model._scaler.transform(frame.values), nan=-1).astype('float32')
                layer_stats = []
                for layer in layers:
                    coords = .5*np.tanh(values) if layer.hidden_coordinates else values
                    layer_stats.append(dict(name=layer.name, hidden_coordinate_transform=layer.hidden_coordinates,
                        near_boundary_fraction=float(np.mean(abs(coords) > .49)),
                        coefficient_abs_max=float(np.max(abs(layer.coefficients.numpy())))))
                    values = layer(values).numpy()
                item['training_layers'] = layer_stats
            result['saved_model_diagnostics'].append(item)
            del model
            tf.keras.backend.clear_session()
        upper = float(np.mean([r['upper_bound'] for r in result['saved_model_diagnostics']]))
        lower = float(np.mean([r['lower_bound'] for r in result['saved_model_diagnostics']]))
        assert pred.max() <= upper + .01 and pred.min() >= lower - .01
        result['ensemble_output_envelope'] = dict(lower=lower, upper=upper,
            test_targets_above_upper=int(np.sum(y > upper)),
            unavoidable_mae_for_these_frozen_weights=float(np.mean(np.maximum(y-upper, 0)+np.maximum(lower-y, 0))),
            interpretation='Bound for saved coefficients, not a limit on the architecture after different training')
    np.testing.assert_array_equal(predictions['mm_modnet'][0], predictions['mm_kan'][0])
    np.testing.assert_array_equal(predictions['mm_modnet'][1], predictions['mm_kan'][1])
    previous = ROOT / 'benchmarks/native-spline-scale-diagnosis-20261029'
    assert read(previous / 'verification.json')['passed']
    result['previous_paired_inner_diagnostic'] = read(previous / 'summary.json')
    write_json(out / 'diagnosis.json', result)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharex=True, sharey=True)
    names = {'mm_modnet': 'Original MM-MLP', 'mm_kan': 'MM-KAN: raw targets'}
    for axis, arm, color in zip(axes, ('mm_modnet', 'mm_kan'), ('#2563a6', '#d76a27')):
        _, y, pred = predictions[arm]
        axis.scatter(y, pred, s=19, alpha=.65, c=color, edgecolors='none')
        axis.plot([0, 3800], [0, 3800], '--', color='#777777', lw=1, label='Perfect prediction')
        axis.set(xlim=(0, 3800), ylim=(0, 3800), xlabel=r'True frequency (cm$^{-1}$)')
        axis.set_title(names[arm] + '\nMAE = {:.2f} cm$^{{-1}}$'.format(result['models'][arm]['test_mae']))
        axis.grid(alpha=.18)
        if arm == 'mm_kan':
            upper = result['ensemble_output_envelope']['upper']
            axis.axhline(upper, color='#a33a1c', ls=':', lw=1.5,
                         label='Upper bound of saved ensemble')
        axis.legend(loc='upper left', frameon=False, fontsize=8)
    axes[0].set_ylabel(r'Predicted frequency (cm$^{-1}$)')
    fig.suptitle('Phonons: same 253 test materials, first outer fold, 50 models per ensemble', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'first-fold-predictions.png', dpi=170)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k != 'saved_model_diagnostics'}, indent=2))
    print('TOP_CANDIDATE_TRAINING_LAYERS', json.dumps(result['saved_model_diagnostics'][:5], indent=2))


if __name__ == '__main__':
    main()
