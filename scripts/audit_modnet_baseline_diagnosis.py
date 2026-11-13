"""Verify paired ablation inputs, target scaling and saved numerical outputs."""
from run_modnet_baseline_diagnosis import ROOT, OLD, MODES, read, file_hash, np, MODData, write_json
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = read(out / 'protocol.json')
    data = MODData.load(OLD / 'matbench_phonons/fold-0/official-train-moddata.pkl.gz')
    outer_ids = set(data.df_targets.index)
    records = {}
    for path in (out / 'diagnostics').glob('fold-0/*/*/diagnostic.json'):
        r = read(path)
        records[r['mode'], r['member_index']] = r
        assert set(r['train_ids']).isdisjoint(r['validation_ids'])
        assert set(r['train_ids'] + r['validation_ids']) == outer_ids
        assert r['target_fit_ids'] == r['train_ids']
        assert len(r['features']) == r['genes']['n_feat']
        assert not any('SISSO' in f.upper() for f in r['features'])
        for name, sha in r['artifacts'].items():
            assert file_hash(path.parent / name) == sha
        with np.load(path.parent / 'validation.npz') as z:
            assert list(z['sample_ids']) == r['validation_ids']
            np.testing.assert_array_equal(z['target'], data.df_targets.loc[r['validation_ids']].values[:, 0])
            np.testing.assert_allclose(abs(z['prediction'] - z['target']).mean(), r['validation_mae'], atol=1e-10)
        y = data.df_targets.loc[r['train_ids']].values[:, 0]
        normalized = r['mode'] in ('target_only', 'target_checkpoint', 'new_features_training')
        np.testing.assert_allclose(r['target_mean'], y.mean() if normalized else 0, atol=1e-10)
        np.testing.assert_allclose(r['target_std'], y.std() if normalized else 1, atol=1e-10)
        if r['mode'] == 'native' and not protocol['smoke']:
            assert abs(r['validation_mae'] - r['original_validation_mae']) <= .01
    checks = 0
    for i in range(1 if protocol['smoke'] else 5):
        for modes in (MODES[:4], MODES[4:]):
            selected = [records[m, i] for m in modes if (m, i) in records]
            for a, b in zip(selected, selected[1:]):
                for key in ('train_ids', 'validation_ids', 'seed', 'genes'):
                    assert a[key] == b[key], (key, a['mode'], b['mode'])
                if a['mode'] != 'new_split_legacy':
                    assert a['features'] == b['features']
                checks += 1
        if protocol['smoke']:
            # With one epoch, changing checkpoint selection must not change
            # initialization, minibatch order, or the resulting predictions.
            for a, b in (('native', 'checkpoint_only'), ('target_only', 'target_checkpoint')):
                with np.load(Path(records[a, i]['directory']) / 'validation.npz') as za:
                    with np.load(Path(records[b, i]['directory']) / 'validation.npz') as zb:
                        np.testing.assert_array_equal(za['prediction'], zb['prediction'])
    expected = protocol['diagnostics_fits']
    replay_checks = []
    all_test_ids = []
    for fold in range(5):
        directory = out / 'replay' / f'fold-{fold}'
        if not (directory / 'complete.json').exists():
            continue
        score = read(directory / 'complete.json')
        historical = OLD / 'matbench_phonons' / f'fold-{fold}'
        split = read(historical / 'outer-split.json')
        test_ids = split['official_test_ids']
        all_test_ids.extend(test_ids)
        assert set(split['train_ids']).isdisjoint(split['test_ids'])
        members = sorted(directory.glob('replay/member-*/diagnostic.json'))
        assert len(members) == 50
        for path in members:
            r = read(path)
            assert set(r['train_ids'] + r['validation_ids']) == set(split['train_ids'])
            assert abs(r['validation_mae'] - r['original_validation_mae']) <= .01
            for name, sha in r['artifacts'].items():
                assert file_hash(path.parent / name) == sha
        with np.load(directory / 'modnet-member-predictions.npz') as fresh:
            with np.load(historical / 'modnet-member-predictions.npz') as old:
                assert fresh['predictions'].shape == (50, len(test_ids))
                np.testing.assert_array_equal(fresh['predictions'], old['predictions'])
                np.testing.assert_array_equal(fresh['raw_predictions'], fresh['predictions'])
                assert np.sum(fresh['replaced_counts']) == 0
        assert score['mae'] == read(historical / 'baseline.json')['results']['modnet']['mae']
        replay_checks.append(dict(fold=fold,verified_members=len(members),mae=score['mae'],
                                  member_predictions_identical=True))
    if len(replay_checks) == 5:
        assert len(all_test_ids) == len(set(all_test_ids)) == 1265
        summary = read(out / 'replay-summary.json')
        assert summary['complete'] and summary['verified']
        np.testing.assert_allclose(summary['mae'], np.mean([r['mae'] for r in replay_checks]), atol=1e-10)
    result = dict(passed=True, completed=len(records), expected=expected,
                  complete=len(records) == expected, smoke=protocol['smoke'], paired_comparisons=checks,
                  replay_complete=len(replay_checks) == 5, replay_checks=replay_checks,
                  checks=['Outer test rows excluded', 'Same genes/seed/rows within paired comparisons',
                          'Target statistics fit only on training rows', 'Saved predictions reproduce MAE',
                          'No SISSO; selected features have specified count', 'Model and prediction hashes intact',
                          'Native refits match historical validation scores within0.01cm^-1'])
    write_json(out / 'diagnostic-audit.json', result)
    print(result)


if __name__ == '__main__':
    main()
