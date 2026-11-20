"""Descriptive outer-test report for finished, predeclared input budgets.

Read the live run without changing its protocol, fits, selections or completion
markers. The training runner never reads this report; it selects using inner CV.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import run_phonons_standardized_input_budget as scan

native, np = scan.native, scan.np


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    assert run != out and run not in out.parents
    out.mkdir(parents=True, exist_ok=True)
    p = scan.read(run / 'protocol.json')
    fingerprint = native.digest(p)
    scan.load_protocol(str(run / 'protocol.json'), fingerprint)
    assert not p['smoke'] and len(p['folds']) == 5
    budgets = sorted(set(args.budgets))
    assert budgets and all(k in p['coarse_budgets'] for k in budgets)
    references = {}
    for filename, sha in p['reference_sha256'].items():
        assert native.file_hash(filename) == sha, filename
    for fold in p['folds']:
        for budget in budgets:
            path = scan.dimension_folder(run, fold['fold'], budget) / 'validation-summary.json'
            record = scan.read(path)
            assert record['protocol_fingerprint'] == fingerprint and record['ensemble_size'] == 50
            references[str(path)] = native.file_hash(path)
    protocol = dict(run=str(run), protocol_fingerprint=fingerprint, budgets=budgets,
        validation_summary_sha256=references, reporter_sha256=native.file_hash(__file__),
        source_sha256=p['source_sha256'],
        purpose='User-requested interim comparison of all currently finished, predeclared input caps.',
        outer_results_observed_before_final_dimension_selection=True,
        selection_policy_unchanged=True,
        limitation='The smallest qualifying completed cap is descriptive, not an unbiased final selected optimum. '
                   'The live runner continues its frozen inner-validation-only selection algorithm. Its later '
                   'selection-before-scoring audit refers to its own final scoring stage, not global test blinding.')
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        assert native.digest(scan.read(protocol_path)) == native.digest(protocol)
    else:
        native.write_json(protocol_path, protocol)
    assert native.file_hash(scan.ORIGINAL) == p['original_sha256']
    original = native.MODData.load(scan.ORIGINAL)
    rows, baseline_rows, ids_all = [], [], []
    completed_members = 0
    for fold in p['folds']:
        number = fold['fold']
        frame = original.df_featurized.loc[fold['test_ids']]
        target = original.df_targets.loc[fold['test_ids']].values[:, 0]
        ids_all.extend(fold['official_test_ids'])
        for arm in scan.ARMS:
            base = fold['arms'][arm]
            archive = Path(p['reference']) / arm / f'fold-{number}' / 'outer-predictions.npz'
            with np.load(archive) as z:
                np.testing.assert_array_equal(z['target'], target)
                assert z['sample_ids'].tolist() == fold['official_test_ids']
                np.testing.assert_array_equal(z['prediction'], z['members'].mean(axis=0))
                np.testing.assert_allclose(np.abs(z['prediction']-target).mean(), base['complete']['mae'], atol=1e-10, rtol=0)
            baseline_rows.append(dict(arm=arm, fold=number, mae=base['complete']['mae'],
                raw_mae=base['complete']['raw_mae'], parameters_mean=base['complete']['parameters_mean'],
                parameters_sum=base['complete']['parameters_sum'], input_union=base['input_union']))
        for budget in budgets:
            source = scan.dimension_folder(run, number, budget)
            destination = scan.dimension_folder(out, number, budget)
            destination.mkdir(parents=True, exist_ok=True)
            validation = scan.read(source / 'validation-summary.json')
            done = destination / 'score.json'
            if done.exists():
                result = scan.read(done)
                assert result['protocol_fingerprint'] == fingerprint
                assert native.file_hash(destination / 'outer-predictions.npz') == result['prediction_sha256']
                rows.append(result)
                completed_members += 50
                continue
            matrix, raw_matrix, repairs, parameters, hashes = [], [], [], [], []
            for index in range(50):
                rec = scan.read(source / f'member-{index:02d}' / 'evaluation.json')
                assert rec['protocol_fingerprint'] == fingerprint
                assert rec['fold'] == number and rec['budget'] == budget and rec['index'] == index
                original_member = fold['arms'][scan.ARMS[1]]['members'][index]
                assert rec['seed'] == original_member['seed']
                assert rec['hidden_widths'] == original_member['hidden_widths']
                assert rec['input_features'] == min(budget, original_member['genes']['n_feat'])
                assert rec['target_scaling'] == original_member['target_scaling']
                expected = scan.spline.spline_count(rec['input_features'], *rec['hidden_widths'])
                assert rec['parameters'] == expected
                model = scan.load_network(rec['model_directory'], rec['model_sha256'],
                    fold['ranking'][:rec['input_features']], rec['target_scaling'], expected)
                raw = scan.raw_prediction(model, frame)
                prediction, repaired = scan.native_prediction(raw, model, rec['seed'])
                matrix.append(prediction); raw_matrix.append(raw); repairs.append(repaired)
                parameters.append(expected); hashes.append(rec['model_sha256'])
                scan.release(model)
                completed_members += 1
                if (index+1) % 10 == 0:
                    native.write_json(out / 'status.json', dict(state='scoring', pid=os.getpid(), updated_at=now(),
                        completed_members=completed_members, total_members=250*len(budgets), fold=number, budget=budget))
            matrix, raw_matrix = np.asarray(matrix), np.asarray(raw_matrix)
            assert matrix.shape == raw_matrix.shape == (50, 253)
            prediction, raw_prediction = matrix.mean(axis=0), raw_matrix.mean(axis=0)
            assert sum(parameters) == validation['parameters_sum']
            np.testing.assert_allclose(np.mean(parameters), validation['parameters_mean'], atol=1e-9, rtol=0)
            np.savez_compressed(destination / 'outer-predictions.npz', sample_ids=fold['official_test_ids'],
                target=target, prediction=prediction, raw_prediction=raw_prediction, members=matrix, raw_members=raw_matrix)
            result = dict(protocol_fingerprint=fingerprint, fold=number, budget=budget, input_union=validation['input_union'],
                mae=float(np.abs(prediction-target).mean()), raw_mae=float(np.abs(raw_prediction-target).mean()),
                parameters_mean=float(np.mean(parameters)), parameters_sum=sum(parameters), ensemble_size=50,
                repaired_member_predictions=sum(repairs), model_sha256=hashes,
                prediction_sha256=native.file_hash(destination / 'outer-predictions.npz'))
            native.write_json(done, result); rows.append(result)
            print(json.dumps({k:result[k] for k in ['fold','budget','mae','parameters_mean','repaired_member_predictions']}), flush=True)
    assert len(ids_all) == len(set(ids_all)) == 1265
    baselines = {arm: {key: float(np.mean([r[key] for r in baseline_rows if r['arm'] == arm]))
                      for key in ('mae','raw_mae','parameters_mean','parameters_sum','input_union')}
                 for arm in scan.ARMS}
    baseline = baselines[scan.ARMS[0]]
    curve = []
    for budget in budgets:
        group = [r for r in rows if r['budget'] == budget]
        assert sorted(r['fold'] for r in group) == list(range(5))
        item = dict(budget=budget, folds=5, ensemble_size=50,
            **{key: float(np.mean([r[key] for r in group])) for key in
               ('mae','raw_mae','parameters_mean','parameters_sum','input_union')})
        item.update(relative_mae_change=item['mae']/baseline['mae']-1,
            parameter_reduction_vs_mlp=1-item['parameters_sum']/baseline['parameters_sum'],
            parameter_reduction_vs_full_kan=1-item['parameters_sum']/baselines[scan.ARMS[1]]['parameters_sum'],
            input_reduction_vs_mlp=1-item['input_union']/baseline['input_union'],
            repaired_member_predictions=sum(r['repaired_member_predictions'] for r in group),
            fold_mae=[r['mae'] for r in sorted(group, key=lambda r:r['fold'])])
        curve.append(item)
    summary = dict(complete=True, interim=True, generated_at=now(), live_run_still_in_progress=True,
        baselines=baselines, curve=curve, rows=rows,
        minimum_completed_cap_no_higher_mae=min([r['budget'] for r in curve if r['mae']<=baseline['mae']] or [None]),
        minimum_completed_cap_within_5_percent=min([r['budget'] for r in curve if r['mae']<=1.05*baseline['mae']] or [None]),
        limitation=protocol['limitation'])
    native.write_json(out / 'summary.json', summary)
    # Independent reconstruction from the saved 1,265 paired test predictions.
    for budget in budgets:
        errors = []
        for fold in p['folds']:
            directory = scan.dimension_folder(out, fold['fold'], budget)
            record = scan.read(directory / 'score.json')
            assert native.file_hash(directory / 'outer-predictions.npz') == record['prediction_sha256']
            with np.load(directory / 'outer-predictions.npz') as z:
                np.testing.assert_array_equal(z['prediction'], z['members'].mean(axis=0))
                errors.extend(np.abs(z['prediction']-z['target']).tolist())
        assert len(errors)==1265
        expected = next(r['mae'] for r in curve if r['budget']==budget)
        np.testing.assert_allclose(np.mean(errors), expected, rtol=0, atol=1e-10)
    native.write_json(out / 'artifact-audit.json', dict(passed=True, complete=True, interim=True,
        source_protocol_fingerprint=fingerprint, budgets=budgets, test_samples_per_budget=1265,
        models_per_fold=50, actual_model_parameters_checked=True, test_predictions_recomputed=True,
        fixed_hidden_widths_checked=True, source_run_modified=False))
    native.write_json(out / 'status.json', dict(state='complete', complete=True, pid=os.getpid(), updated_at=now(),
        completed_members=250*len(budgets), total_members=250*len(budgets)))
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}), flush=True)


if __name__ == '__main__':
    main()
