"""Frozen 20-fit, training-only MatterVial screen; extend to 80 only on signal.

This deliberately uses a lighter selection/training protocol than the paper.
Only split IDs are reused. No previous model scores, selected features, model
weights, SISSO values, or outer-test targets enter this experiment.
"""
from __future__ import annotations

from run_modnet_official_matched import SafePool, digest, file_hash, set_seed, write_json
from run_mattervial_phonons import CSV, CSV_SHA, feature_group
from physical_feature_modules import PHYSICAL_FEATURES, PHYSICAL_GROUPS, WIDTHS, build_predictor, fit_minmax
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'benchmarks/mattervial-phonons-20261023/run/paper_features_sisso_unverified/fold-0/inputs'
VARIANTS = ('matminer_modnet', 'mattervial_modnet', 'mattervial_mlp', 'mattervial_kan')
LABELS = dict(zip(VARIANTS, ('A MatMiner MODNet', 'B MatterVial MODNet', 'C MatterVial grouped MLP', 'D MatterVial grouped KAN')))
SEEDS = (20261026, 20271026)
LRS = (.001, .0005)
FEATURE_COUNT = 256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def now():
    return datetime.now(timezone.utc).isoformat()


def read_splits(source):
    """Read only ID lists from existing manifests, never their feature lists/scores."""
    splits = []
    for inner in range(5):
        path = source / f'inner-{inner}' / 'manifest.json'
        saved = read(path)
        splits.append(dict(inner=inner, train_ids=saved['train_ids'], validation_ids=saved['validation_ids'],
                           source=str(path), source_sha256=file_hash(path)))
    universe = set(splits[0]['train_ids'] + splits[0]['validation_ids'])
    covered = []
    for item in splits:
        tr, va = item['train_ids'], item['validation_ids']
        if len(tr) != len(set(tr)) or len(va) != len(set(va)) or set(tr) & set(va):
            raise ValueError('Overlapping/duplicate inner IDs.')
        if set(tr + va) != universe:
            raise ValueError('Inner folds do not share the same training universe.')
        covered += va
    if len(universe) != 1012 or len(covered) != len(set(covered)) or set(covered) != universe:
        raise ValueError('Expected an exact five-fold partition of 1012 outer-training IDs.')
    return splits, sorted(universe)


def load_training_rows(path, allowed_ids):
    """Discard non-training rows before numeric conversion; never select SISSO.

    Reading the source CSV necessarily scans its lines. Non-training values and
    labels are not converted, stored, scored, or used for any fit/decision.
    """
    allowed = set(allowed_ids)
    rows, found = [], []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        excluded_sisso = [c for c in names if 'SISSO' in c.upper()]
        columns = [c for c in names if 'SISSO' not in c.upper() and feature_group(c) not in (None, 'sisso')]
        for row in reader:
            key = row['material_id']
            if key not in allowed:
                continue
            found.append(key)
            rows.append([float(row[c]) if row[c] else np.nan for c in columns] + [float(row['target'])])
    if set(found) != allowed or len(found) != len(allowed):
        raise ValueError('Training CSV IDs missing or duplicated.')
    frame = pd.DataFrame(rows, index=found, columns=columns + ['target']).loc[allowed_ids]
    if not np.isfinite(frame['target']).all():
        raise ValueError('Nonfinite training targets.')
    frame.loc[:, columns] = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    return frame, excluded_sisso


def select_once(train_x, train_y, count, seed):
    if len(set(PHYSICAL_FEATURES) & set(train_x.columns)) != 16 or count < 16:
        raise ValueError('All fixed physical features are required.')
    if any('SISSO' in c.upper() for c in train_x.columns):
        raise ValueError('SISSO must never enter feature selection.')
    selector = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=.1,
        tree_method='hist', max_bin=256, importance_type='gain', objective='reg:squarederror',
        subsample=1., colsample_bytree=1., reg_lambda=1., n_jobs=1, random_state=seed)
    selector.fit(train_x.values.astype('float32'), np.asarray(train_y))
    importance = dict(zip(train_x.columns, map(float, selector.feature_importances_)))
    ranked = sorted((c for c in train_x.columns if c not in PHYSICAL_FEATURES),
                    key=lambda c: (-importance[c], c))
    selected = list(PHYSICAL_FEATURES) + ranked[:count - 16]
    if len(selected) != count or len(set(selected)) != count:
        raise ValueError('Insufficient selected inputs.')
    return selected, importance


def prepare_input(job):
    directory = Path(job['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    signature = digest(job)
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        info = read(manifest_path)
        if info['signature'] != signature:
            raise ValueError('Changed feature preparation.')
        for name in ('arrays.npz', 'scaler.pkl'):
            if file_hash(directory / name) != info['artifacts'][name]:
                raise ValueError('Changed prepared inputs.')
        return info
    started = time.perf_counter()
    if file_hash(job['training_cache']) != job['training_cache_sha256']:
        raise ValueError('Training-only cache changed.')
    frame = pd.read_pickle(job['training_cache'])
    split = job['split']
    cols = [c for c in frame if c != 'target' and
            (job['bank'] == 'mattervial' or feature_group(c) == 'matminer')]
    tr = frame.loc[split['train_ids'], cols]
    va = frame.loc[split['validation_ids'], cols]
    y = frame.loc[split['train_ids'], 'target'].values
    yv = frame.loc[split['validation_ids'], 'target'].values
    selected, importance = select_once(tr, y, job['feature_count'], SEEDS[0] + split['inner'])
    scaler, x, xv = fit_minmax(tr[selected].values, va[selected].values)
    np.savez_compressed(directory / 'arrays.npz', x_train=x, x_validation=xv, y_train=y, y_validation=yv)
    with (directory / 'scaler.pkl').open('wb') as stream:
        pickle.dump(scaler, stream, protocol=4)
    info = dict(signature=signature, run_fingerprint=job['run_fingerprint'], inner=split['inner'], bank=job['bank'],
        directory=str(directory), selected_features=selected, candidate_feature_count=len(cols),
        train_ids=split['train_ids'], validation_ids=split['validation_ids'],
        feature_selection_fit_ids=split['train_ids'], scaling_fit_ids=split['train_ids'],
        target_scale=float(np.std(y)), importance=importance,
        selected_group_counts={g: sum(feature_group(c) == g for c in selected) for g in sorted(set(map(feature_group, cols)))},
        seconds=time.perf_counter() - started, artifacts={name: file_hash(directory / name) for name in ('arrays.npz', 'scaler.pkl')},
        validation_out_of_scaling_range_fraction=float(np.mean((xv < -.5) | (xv > .5))))
    write_json(manifest_path, info)
    return info


def make_protocol(splits, smoke=False):
    dependencies = ('run_mattervial_quick_pilot.py', 'run_mattervial_phonons.py', 'run_modnet_official_matched.py',
                    'official_kan_models.py', 'physical_feature_modules.py', 'modnet_runtime.py')
    return dict(kind='training_only_mattervial_20_then_conditional_80', smoke=smoke, outer_fold=0,
        outer_test_evaluated=False, data_source=str(CSV), csv_sha256=CSV_SHA,
        splits=splits, variants=VARIANTS, trunk_widths=WIDTHS, features=FEATURE_COUNT,
        physical_groups=PHYSICAL_GROUPS, physical_outputs=4, module_parameters=148,
        source_sha256={name: file_hash(ROOT / 'scripts' / name) for name in dependencies},
        packages={name: importlib.metadata.version(name) for name in ('modnet','tensorflow','xgboost','numpy','pandas','scikit-learn')},
        feature_selection='one XGB fit per bank/inner TRAIN only, gain ranking, fixed 16 + top 240, deterministic name tie-break',
        feature_selection_fits=10, sisso='ALL SISSO columns excluded before numeric conversion and selection',
        preprocessing='fixed zero imputation; inner-training-only per-column minmax [-0.5,0.5]',
        missing_pretrained_features='All frozen embeddings inherited from published data; provenance limitations remain.',
        max_epochs=1 if smoke else 300, batch_size=32, learning_rates=LRS, seeds=SEEDS,
        initial_lr=LRS[0], initial_seed=SEEDS[0], optimizer='Adam', loss='raw target MAE plus fixed module penalties',
        early_stopping=dict(monitor='mae', patience=30, min_delta=.001, restore_best_weights=False),
        penalties=dict(smooth=.001, sparse=.01, scaling='inner-training target standard deviation'),
        budget='same fit count, epoch cap, batch size, optimizer, training-MAE stopping rule; actual epochs/time may differ',
        stage1_fits=20, expanded_total_fits=80, expanded_additional_fits=60,
        expansion_gate=dict(min_improved_folds=4, min_relative_mean_improvement=.01,
            mattervial='B versus A', kan='D versus C AND positive mean improvement versus B',
            trigger='either MatterVial or KAN passes; expand ALL four groups together',
            interpretation='exploratory compute allocation heuristic, not a significance test'),
        selection='stage1 no tuning; stage2 LR chosen using mean over all five inner folds and both seeds',
        limitations=['Not exact paper replication; no comparison to published outer-test MAE.',
            'Five validation folds share training data; one seed cannot establish seed stability.',
            '300-epoch cap can underfit; an absent signal is conditional on this small budget.',
            'These training samples/inner splits have been used in previous exploratory experiments.'])


def train_job(job):
    directory = Path(job['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    signature = digest(job)
    path = directory / 'result.json'
    if path.exists():
        saved = read(path)
        if saved['signature'] != signature:
            raise ValueError('Changed training job.')
        for name, sha in saved['artifacts'].items():
            if file_hash(directory / name) != sha:
                raise ValueError('Changed saved model/predictions.')
        return saved
    try:
        info = read(job['manifest'])
        if digest(info) != job['manifest_digest']:
            raise ValueError('Input manifest changed.')
        array_path = Path(info['directory']) / 'arrays.npz'
        if file_hash(array_path) != info['artifacts']['arrays.npz']:
            raise ValueError('Arrays changed.')
        with np.load(array_path) as arrays:
            x, xv, y, yv = [arrays[k] for k in ('x_train','x_validation','y_train','y_validation')]
        if len(x) != len(info['train_ids']) or len(xv) != len(info['validation_ids']):
            raise ValueError('Arrays and IDs do not agree.')
        features = info['selected_features']
        kind = {'mattervial_mlp':'group_mlp','mattervial_kan':'group_sparse_kan'}.get(job['variant'], 'modnet')
        started = time.perf_counter()
        model = build_predictor(kind, features, features, 'target', job['seed'], info['target_scale'])
        module = next((l for l in model.layers if l.name == 'physical_module'), None)
        module_parameters = 0 if module is None else module.count_params()
        if module_parameters != (0 if kind == 'modnet' else 148):
            raise ValueError('Wrong module capacity.')
        set_seed(job['seed'])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=job['lr']), loss='mae', metrics=['mae'])
        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 25 == 0:
                    write_json(directory / 'status.json', dict(state='training', epoch=epoch+1,
                        seconds=time.perf_counter()-started, updated_at=now(), metrics=logs))
        with (directory / 'training.log').open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                history = model.fit(x, y, validation_data=(xv, yv), epochs=job['max_epochs'], batch_size=32, verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='mae', patience=30, min_delta=.001,
                        restore_best_weights=False), Progress()])
        pred = model.predict(xv, verbose=0).reshape(-1).astype('float64')
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite predictions.')
        np.savez_compressed(directory / 'validation-predictions.npz', sample_ids=info['validation_ids'], target=yv, prediction=pred)
        with (directory / 'model.pkl').open('wb') as stream:
            pickle.dump(dict(model_json=model.to_json(), weights=model.get_weights(), input_manifest=job['manifest']), stream, protocol=4)
        record = dict(signature=signature, run_fingerprint=job['run_fingerprint'], variant=job['variant'],
            inner=job['inner'], repeat=job['repeat'], seed=job['seed'], lr=job['lr'], directory=str(directory),
            train_ids=info['train_ids'], validation_ids=info['validation_ids'], selected_features=features,
            input_manifest_digest=job['manifest_digest'], parameters=model.count_params(), module_parameters=module_parameters,
            validation_mae=float(np.mean(abs(pred-yv))), epochs=len(history.history['loss']),
            hit_epoch_cap=len(history.history['loss']) == job['max_epochs'], seconds=time.perf_counter()-started,
            history=history.history, artifacts={name:file_hash(directory / name) for name in ('model.pkl','validation-predictions.npz')})
        write_json(path, record)
        write_json(directory / 'status.json', dict(state='complete', finished_at=now(), seconds=record['seconds']))
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        return record
    except Exception:
        write_json(directory / 'status.json', dict(state='failed', traceback=traceback.format_exc(), updated_at=now()))
        raise


def paired_difference(candidate, reference):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != (5,) or reference.shape != (5,) or not np.isfinite(candidate + reference).all():
        raise ValueError('All five finite paired folds are required.')
    delta = candidate - reference
    relative = float((reference.mean() - candidate.mean()) / reference.mean())
    return dict(delta_mae_by_fold=delta.tolist(), mean_delta_mae=float(delta.mean()),
        relative_mean_improvement=relative, improved_folds=int((delta < 0).sum()),
        passes=int((delta < 0).sum()) >= 4 and relative >= .01)


def expansion_gate(scores):
    mv = paired_difference(scores['mattervial_modnet'], scores['matminer_modnet'])
    kan = paired_difference(scores['mattervial_kan'], scores['mattervial_mlp'])
    kan_base = paired_difference(scores['mattervial_kan'], scores['mattervial_modnet'])
    kan_signal = kan['passes'] and kan_base['mean_delta_mae'] < 0
    return dict(mattervial=mv, kan_vs_mlp=kan, kan_vs_modnet=kan_base,
        mattervial_signal=mv['passes'], kan_signal=kan_signal, expand=mv['passes'] or kan_signal,
        interpretation='Screening heuristic only; not statistical significance or an outer-test result.')


def summarize(output, records, protocol, stage, wall_seconds):
    expected = 20 if stage == 20 else 80
    if len(records) != expected:
        raise ValueError('Summary requires complete paired training grid.')
    combos = [(LRS[0], 0)] if stage == 20 else [(lr,r) for lr in LRS for r in range(2)]
    keys = {(r['variant'],r['inner'],r['lr'],r['repeat']) for r in records}
    if keys != {(v,i,lr,r) for v in VARIANTS for i in range(5) for lr,r in combos}:
        raise ValueError('Missing/duplicate fit in summary.')
    by_key = {(r['variant'],r['inner'],r['lr'],r['repeat']):r for r in records}
    for inner in range(5):
        for lr, repeat in combos:
            rows = [by_key[(v,inner,lr,repeat)] for v in VARIANTS]
            for key in ('train_ids','validation_ids','seed'):
                if len({digest(r[key]) for r in rows}) != 1:
                    raise ValueError('Unpaired splits/seeds.')
            if len({r['input_manifest_digest'] for r in rows[1:]}) != 1:
                raise ValueError('B/C/D must use identical prepared inputs.')
            if rows[0]['parameters'] != rows[1]['parameters'] or rows[2]['parameters'] != rows[3]['parameters']:
                raise ValueError('A/B or C/D capacity mismatch.')
    for r in records:
        with np.load(Path(r['directory']) / 'validation-predictions.npz') as p:
            if p['sample_ids'].tolist() != r['validation_ids']:
                raise ValueError('Prediction IDs mismatch.')
            np.testing.assert_allclose(float(np.mean(abs(p['prediction']-p['target']))), r['validation_mae'], atol=1e-12, rtol=0)
    rows, chosen_scores = [], {}
    for variant in VARIANTS:
        members = [r for r in records if r['variant'] == variant]
        candidates = []
        for lr in sorted(set(r['lr'] for r in members)):
            selected = [r for r in members if r['lr'] == lr]
            fold_scores = [float(np.mean([r['validation_mae'] for r in selected if r['inner']==i])) for i in range(5)]
            candidates.append(dict(lr=lr, fold_mae=fold_scores, mean_mae=float(np.mean(fold_scores)),
                seed_means={str(s):float(np.mean([r['validation_mae'] for r in selected if r['repeat']==s])) for s in sorted(set(r['repeat'] for r in selected))}))
        choice = min(candidates, key=lambda c:(c['mean_mae'],c['lr']))
        chosen_scores[variant] = choice['fold_mae']
        rows.append(dict(variant=variant, selected_lr=choice['lr'], mean_mae=choice['mean_mae'], fold_mae=choice['fold_mae'],
            candidates=candidates, parameters=members[0]['parameters'], module_parameters=members[0]['module_parameters'],
            fit_seconds=sum(r['seconds'] for r in members), epochs=[r['epochs'] for r in members],
            cap_hits=sum(r['hit_epoch_cap'] for r in members)))
    gate = expansion_gate(chosen_scores) if stage == 20 else None
    summary = dict(stage=stage, smoke=protocol['smoke'], completed_fits=len(records), inner_only=True,
        outer_test_evaluated=False, no_sisso=True, paired_splits_inputs_seeds_and_capacity_verified=True,
        predictions_recomputed=True, results=rows, expansion_gate=gate, wall_seconds=wall_seconds,
        selection_bias='stage 80 selects LR on these same inner validation scores; exploratory only')
    write_json(output / f'stage-{stage}-summary.json', summary)
    report = [f'# MatterVial phonons quick pilot: {stage} fits', '',
        'Only the 1012 training samples of outer fold 0; all five recorded inner splits. No outer-test evaluation.',
        'All SISSO removed. 256 inputs per group; B/C/D share exactly the same selected/scaled inputs.',
        'One XGBoost selection per bank/fold (10 selector fits in addition to neural fits); all fitted inside inner training.',
        'Fixed 96/72/36/36 MODNet trunk. Group MLP/KAN: 16 physical inputs -> 4 features, 148 module parameters.',
        'Same maximum 300 epochs, batch 32, Adam, train-MAE early stopping. Raw validation predictions, no outlier repair.',
        'MAE unit: cm^-1. Inner validation only; not a paper reproduction or an unbiased final benchmark.', '',
        '| Group | Mean MAE | Inner 1 | Inner 2 | Inner 3 | Inner 4 | Inner 5 | Parameters | Fit minutes | Epoch cap hits |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        report.append(f"| {LABELS[row['variant']]} | {row['mean_mae']:.6f} | " +
            ' | '.join(f'{x:.6f}' for x in row['fold_mae']) +
            f" | {row['parameters']} | {row['fit_seconds']/60:.2f} | {row['cap_hits']} |")
    report += ['', f'Observed run wall time: {wall_seconds/60:.2f} minutes. Fit minutes are summed worker times.',
        'A signal requires >=4/5 improved folds and >=1% mean MAE reduction. KAN must also beat B on mean MAE.',
        'This is a prespecified compute-allocation rule, not statistical significance. One seed does not establish seed stability.',
        'Absence of a signal only applies to the fixed feature/training budget; frequent epoch-cap hits indicate possible underfitting.']
    if gate is not None:
        report += ['', f"MatterVial signal: {gate['mattervial_signal']}; KAN signal: {gate['kan_signal']}; expand all groups: {gate['expand']}."]
    else:
        report += ['', 'Stage 80 reports mean member MAE over two seeds at the LR selected using all inner folds; these are not ensemble scores.']
    (output / f'stage-{stage}-report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-jobs', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    def status(state, **kwargs):
        write_json(output / 'status.json', dict(state=state, pid=os.getpid(), updated_at=now(), **kwargs))
    try:
        status('preparing', phase='freeze_protocol', completed=0, total=20)
        splits, allowed_ids = read_splits(SOURCE)
        protocol = make_protocol(splits, args.smoke)
        fingerprint = digest(protocol)
        protocol_path = output / 'protocol.json'
        if protocol_path.exists() and digest(read(protocol_path)) != fingerprint:
            raise ValueError('Protocol/source changed. Use a new output directory.')
        write_json(protocol_path, protocol)
        if file_hash(CSV) != CSV_SHA:
            raise ValueError('Published source CSV changed.')
        cache = output / 'training-only-no-sisso.pkl.gz'
        audit_path = output / 'input-audit.json'
        if audit_path.exists():
            audit = read(audit_path)
            if audit['run_fingerprint'] != fingerprint or file_hash(cache) != audit['training_cache_sha256']:
                raise ValueError('Training-only cache changed.')
        else:
            frame, removed = load_training_rows(CSV, allowed_ids)
            if len(removed) != 283 or len(frame.columns)-1 != 4455:
                raise ValueError('Unexpected feature bank counts.')
            frame.to_pickle(cache)
            audit = dict(run_fingerprint=fingerprint, training_ids=allowed_ids, training_samples=len(frame),
                removed_sisso_columns=removed, removed_sisso_count=len(removed), retained_features=len(frame.columns)-1,
                matminer_features=sum(feature_group(c)=='matminer' for c in frame),
                training_cache_sha256=file_hash(cache), outer_targets_converted_or_stored=False,
                outer_test_evaluated=False, only_split_ids_reused=True,
                frozen_feature_pretraining_provenance='Inherited; removing SISSO alone does not audit external pretraining labels.')
            write_json(audit_path, audit)
            del frame
        jobs = [dict(directory=str(output / 'inputs' / f"inner-{s['inner']}" / bank), bank=bank,
                split=s, training_cache=str(cache), training_cache_sha256=audit['training_cache_sha256'],
                run_fingerprint=fingerprint, feature_count=FEATURE_COUNT) for s in splits for bank in ('matminer','mattervial')]
        inputs = {}
        with SafePool(args.n_jobs) as pool:
            for done, info in enumerate(pool.imap_unordered(prepare_input, jobs), 1):
                inputs[(info['inner'],info['bank'])] = info
                status('preparing', phase='train_only_feature_selection', completed_selectors=done, total_selectors=10, completed=0, total=20)
                print('PREPARED', done, '/10', info['inner'], info['bank'], info['seconds'], flush=True)
        def make_jobs(combinations):
            result=[]
            for lr, repeat in combinations:
                for inner in range(5):
                    for variant in VARIANTS:
                        bank = 'matminer' if variant=='matminer_modnet' else 'mattervial'
                        info=inputs[(inner,bank)]
                        result.append(dict(directory=str(output/'fits'/variant/f'lr-{lr}-seed-{repeat}-inner-{inner}'),
                            variant=variant, inner=inner, lr=lr, repeat=repeat, seed=SEEDS[repeat]+inner,
                            max_epochs=protocol['max_epochs'], run_fingerprint=fingerprint,
                            manifest=str(Path(info['directory'])/'manifest.json'), manifest_digest=digest(info)))
            return result
        records=[]
        def execute(jobs, total):
            status('training', completed=len(records), total=total)
            with SafePool(args.n_jobs) as pool:
                for record in pool.imap_unordered(train_job, jobs):
                    records.append(record)
                    status('training', completed=len(records), total=total, last_variant=record['variant'],
                        last_inner=record['inner'], last_validation_mae=record['validation_mae'])
                    print('FIT',len(records),'/',total,record['variant'],record['inner'],record['validation_mae'],record['epochs'],record['seconds'],flush=True)
        execute(make_jobs([(LRS[0],0)]), 20)
        summary=summarize(output,records,protocol,20,time.perf_counter()-start)
        expand=summary['expansion_gate']['expand'] and not args.smoke
        write_json(output/'expansion-decision.json',dict(run_fingerprint=fingerprint,decided_at=now(),
            stage20_sha256=file_hash(output/'stage-20-summary.json'),expand=expand,gate=summary['expansion_gate']))
        if expand:
            execute(make_jobs([(LRS[0],1),(LRS[1],0),(LRS[1],1)]),80)
            summarize(output,records,protocol,80,time.perf_counter()-start)
        status('complete', completed=len(records), total=len(records), expanded=expand, smoke=args.smoke,
               outer_test_evaluated=False, wall_seconds=time.perf_counter()-start)
        print('COMPLETE',output,flush=True)
    except Exception:
        status('failed',traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
