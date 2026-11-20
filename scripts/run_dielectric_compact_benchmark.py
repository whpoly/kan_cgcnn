"""Deferred dielectric comparison: retained standardized MLP, new compact KAN GA.

No old KAN worker or fixed-width input scan is supported by this runner.
The existing steel/phonons training and GA implementations are imported intact.
"""
import argparse
import copy
import gc
import importlib.metadata
import os
import shutil
import time
import traceback
from pathlib import Path

import run_standardized_matbench_pair as pair
import run_kan_compact_expansion_ga as kan
from official_kan_models import neurons_from_genes as mlp_neurons

ROOT, base, engine = kan.ROOT, kan.base, kan.engine
np, MODData = kan.np, kan.MODData
read, write, sha, digest = kan.read, kan.write_json, kan.file_hash, kan.digest
TASK = 'matbench_dielectric'
OLD_REFERENCE = ROOT/'benchmarks/dielectric-standardized-modnet012-20261116'
TEMPLATE = ROOT/'benchmarks/steels-kan-compact-expansion-ga-20261117/protocol.json'
BUDGETS = [16,48,32,64,80,96,112,128,160,192]
MLP_ARM = 'mm_scaled_modnet'


def neurons(genes):
    if 'kan_a' in genes:
        return [[genes['kan_a']], [genes['kan_b']], [1]]
    return mlp_neurons(genes)


def run_fit(job):
    if job['arm'] == MLP_ARM:
        assert job['variant'] == 'modnet' and 'kan_a' not in job['genes']
        return pair.run_fit(job)
    assert job['arm'].startswith('kan_compact_expansion_k') and job['variant'] == 'bspline', 'Old KAN training is disabled'
    return kan.run_fit(job)


# These process-local adapters also execute when a spawned worker imports us.
kan.native.neurons_from_genes = neurons
engine.run_fit = run_fit


def sources():
    result = dict(read(OLD_REFERENCE/'protocol.json')['source_sha256'])
    result.update(read(TEMPLATE)['source_sha256'])
    result[str(Path(__file__).resolve())] = sha(__file__)
    for path, expected in result.items():
        assert sha(path) == expected, path
    return result


def save_protocol(out, protocol):
    path = out/'protocol.json'
    if path.exists():
        assert digest(read(path)) == digest(protocol), 'Frozen protocol changed'
    else:
        write(path, protocol)
    for path, expected in protocol['source_sha256'].items():
        src = Path(path)
        dest = out/'source-snapshot'/(src.parent.name+'--'+src.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
        assert sha(dest) == expected
    return protocol


def audit_fold(folder, split, fingerprint, members):
    rec = read(folder/'complete.json')
    assert rec['protocol_fingerprint'] == fingerprint and rec['ensemble_size'] == members
    for name, key in [('outer-predictions.npz','prediction_sha256'),('frozen-selection.json','selection_sha256')]:
        assert sha(folder/name) == rec[key]
    selected = read(folder/'frozen-selection.json')
    assert not selected['test_labels_used'] and len(selected['members']) == members
    partitions, parameter_counts = {}, []
    for job in selected['members']:
        r = read(Path(job['directory'])/'result.json')
        assert sha(Path(job['directory'])/'model.pkl') == r['model_sha256']
        assert set(job['train_ids']).isdisjoint(job['validation_ids'])
        assert set(job['train_ids']+job['validation_ids']) == set(split['train_ids'])
        assert r['target_scaling']['fit_ids'] == job['train_ids']
        assert r['target_scaling']['ddof'] == 0
        partition = dict(train_ids=job['train_ids'], validation_ids=job['validation_ids'])
        key = str(job['inner'])
        if key in partitions:
            assert partitions[key] == partition
        partitions[key] = partition
        parameter_counts.append(r['parameters'])
    assert sum(parameter_counts) == rec['parameters_sum']
    np.testing.assert_allclose(np.mean(parameter_counts), rec['parameters_mean'], rtol=0, atol=1e-10)
    with np.load(folder/'outer-predictions.npz') as z:
        assert z['sample_ids'].tolist() == split['official_test_ids']
        assert z['members'].shape[0] == members
        for prefix in ('', 'raw_'):
            np.testing.assert_array_equal(z[prefix+'prediction'], z[prefix+'members'].mean(axis=0))
            np.testing.assert_allclose(abs(z[prefix+'prediction']-z['target']).mean(),rec[prefix+'mae'],rtol=0,atol=1e-10)
    return rec, partitions


def existing_mlp():
    old = read(OLD_REFERENCE/'protocol.json')
    result = {}
    for fold in range(5):
        folder = OLD_REFERENCE/MLP_ARM/f'fold-{fold}'
        if not (folder/'complete.json').exists():
            continue
        split = read(pair.OLD/TASK/f'fold-{fold}'/'outer-split.json')
        rec, _ = audit_fold(folder, split, digest(old), 50)
        assert rec['arm'] == MLP_ARM
        result[str(fold)] = dict(directory=str(folder), protocol_fingerprint=digest(old),
            sha256={str(folder/name):sha(folder/name) for name in ('complete.json','frozen-selection.json','outer-predictions.npz')})
    return result


def freeze_mlp(out, smoke, input_audit):
    old = read(OLD_REFERENCE/'protocol.json')
    p = copy.deepcopy(old)
    reuse = {} if smoke else existing_mlp()
    p.update(kind='Standardized MM-MLP reference for the new dielectric compact-KAN benchmark',
        benchmark_family=MLP_ARM, protocol_path=str(out/'protocol.json'), output=str(out),
        smoke=smoke, arms=[MLP_ARM], folds=[0] if smoke else list(range(5)),
        source_sha256=sources(), input_sha256=input_audit['input_sha256'],
        fits_per_arm_fold=8 if smoke else 1000, reused_mlp_folds=reuse,
        reference_sha256={str(OLD_REFERENCE/'protocol.json'):sha(OLD_REFERENCE/'protocol.json')},
        model_reuse='Only already completed, verified MLP folds; never old KAN models',
        task_order=[TASK])
    for rec in reuse.values():
        p['reference_sha256'].update(rec['sha256'])
    assert {name:importlib.metadata.version(name) for name in p['packages']} == p['packages']
    return save_protocol(out, p)


def mlp_directory(out, p, fold):
    reused = p['reused_mlp_folds'].get(str(fold))
    return Path(reused['directory']) if reused else out/MLP_ARM/f'fold-{fold}'


def report_mlp(out, p):
    rows = []
    for fold in p['folds']:
        folder = mlp_directory(out,p,fold)
        if (folder/'complete.json').exists():
            rows.append(dict(read(folder/'complete.json'), reference_directory=str(folder),
                reused_completed_mlp=str(fold) in p['reused_mlp_folds']))
    complete = len(rows) == len(p['folds'])
    summary = dict(task=TASK, benchmark_family=MLP_ARM, smoke=p['smoke'],complete=complete,rows=rows,
        protocol_fingerprint=digest(p),mean={key:float(np.mean([r[key] for r in rows])) for key in
            ('mae','raw_mae','parameters_mean','parameters_sum')} if rows else {})
    if complete:
        ids = []
        for row in rows:
            fold = row['fold']; folder = Path(row['reference_directory'])
            split = read(pair.OLD/TASK/f'fold-{fold}'/'outer-split.json')
            expected = p['reused_mlp_folds'].get(str(fold),{}).get('protocol_fingerprint',digest(p))
            audit_fold(folder,split,expected,8 if p['smoke'] else 50)
            ids.extend(split['official_test_ids'])
        assert len(ids) == len(set(ids))
        if not p['smoke']:
            assert len(ids) == 4764
        write(out/'artifact-audit.json',dict(passed=True,complete=True,task=TASK,smoke=p['smoke'],
            benchmark_family=MLP_ARM,outer_samples=len(ids),source_folds=len(rows),
            reused_completed_mlp_folds=[r['fold'] for r in rows if r['reused_completed_mlp']],
            old_kan_training_disabled=True,predictions_parameters_and_model_hashes_verified=True))
    write(out/'summary.json', summary)
    return summary


def freeze_kan(out, smoke, reference):
    summary, audit = read(reference/'summary.json'), read(reference/'artifact-audit.json')
    assert summary['complete'] and audit['passed'] and not audit['smoke']
    assert audit['benchmark_family'] == MLP_ARM and audit['outer_samples'] == 4764
    old = read(OLD_REFERENCE/'protocol.json')
    p = copy.deepcopy(read(TEMPLATE))
    budgets = [min(BUDGETS),max(BUDGETS)] if smoke else BUDGETS
    references = {str(path):sha(path) for path in [TEMPLATE,OLD_REFERENCE/'protocol.json',
        reference/'protocol.json',reference/'summary.json',reference/'artifact-audit.json']}
    folds = []
    for row in summary['rows']:
        fold = row['fold']
        if smoke and fold != 0:
            continue
        folder = Path(row['reference_directory'])
        native_folder = pair.OLD/TASK/f'fold-{fold}'
        cache = native_folder/'official-train-moddata.pkl.gz'
        split = read(native_folder/'outer-split.json')
        rec, partitions = audit_fold(folder,split,row['protocol_fingerprint'],50)
        assert set(partitions) == set(map(str,range(5)))
        for name in ('complete.json','frozen-selection.json','outer-predictions.npz'):
            references[str(folder/name)] = sha(folder/name)
        folds.append(dict(fold=fold,cache=str(cache),cache_sha256=sha(cache),ranking=split['selected_features'],
            train_ids=split['train_ids'],test_ids=split['test_ids'],official_test_ids=split['official_test_ids'],
            expected_inner_splits=partitions,mlp_reference=rec,
            fixed_width_references={str(n):None for n in budgets}))
    assert len(folds) == (1 if smoke else 5) and min(len(f['ranking']) for f in folds) >= max(budgets)
    p.update(task=TASK,units='unitless',samples=4764,benchmark_family='kan_compact_expansion',
        protocol_path=str(out/'protocol.json'),output=str(out),reference=str(reference),baseline=str(reference),
        original=old['original'],original_sha256=sha(old['original']),source_sha256=sources(),
        reference_sha256=references,input_sha256=old['input_sha256'],smoke=smoke,folds=folds,budgets=budgets,
        seed=old['seed'],fits_per_arm_fold=8 if smoke else 1000,
        fixed_width_reference_available=False,
        restart='Only identically signed new KAN fits are resumable. No old KAN model or old fixed-width input scan is reused.',
        interpretation='New dielectric task extension of the frozen steel/phonons 1%-MAE compact-GA method. '
            'Previously seen dielectric baseline folds make this exploratory; original outer-training feature ranking is retained.')
    p['search_space']['input_dependent_maxima'] = {
        str(n):dict(zip(('a_max','b_max'),kan.width_policy.limits(n,kan.WIDTH_PROFILE))) for n in budgets}
    assert p['training'] == old['training'] and p['packages'] == old['packages']
    return save_protocol(out,p)


def report_kan(out,p):
    rows = [read(out/f'k-{n:03d}'/f"fold-{f['fold']}"/'complete.json') for n in p['budgets'] for f in p['folds']
        if (out/f'k-{n:03d}'/f"fold-{f['fold']}"/'complete.json').exists()]
    curve = []
    for n in p['budgets']:
        group = [r for r in rows if r['budget'] == n]
        if len(group) != len(p['folds']):
            continue
        item = dict(budget=n,**{k:float(np.mean([r[k] for r in group])) for k in
            ('mae','raw_mae','parameters_mean','parameters_sum')})
        reference = {k:float(np.mean([r['mlp_reference'][k] for r in group])) for k in
            ('mae','parameters_mean','parameters_sum')}
        item.update(mlp=reference,relative_mae_change_vs_mlp=item['mae']/reference['mae']-1,
            parameter_reduction_vs_mlp=1-item['parameters_sum']/reference['parameters_sum'])
        curve.append(item)
    complete = len(rows) == len(p['budgets'])*len(p['folds'])
    summary = dict(task=TASK,units=p['units'],benchmark_family='kan_compact_expansion',smoke=p['smoke'],
        complete=complete,protocol_fingerprint=digest(p),rows=rows,curve=curve,budgets=p['budgets'],
        total_search_fits=sum(r['search_fits'] for r in rows),
        minimum_measured_input_strict=min([r['budget'] for r in curve if r['mae']<=r['mlp']['mae']] or [None]),
        limitation=p['interpretation'])
    if complete:
        for n in p['budgets']:
            ids=[]
            for f in p['folds']:
                folder=out/f'k-{n:03d}'/f"fold-{f['fold']}"
                audit_fold(folder,f,digest(p),8 if p['smoke'] else 50)
                assert read(folder/'ga-stopping-audit.json')['passed']
                ids.extend(f['official_test_ids'])
            assert len(ids)==len(set(ids))
            if not p['smoke']:assert len(ids)==4764
        write(out/'artifact-audit.json',dict(passed=True,complete=True,task=TASK,smoke=p['smoke'],
            benchmark_family='kan_compact_expansion',outer_samples=len(ids),budgets=p['budgets'],
            accuracy_size_selection_and_stopping_verified=True,input_dependent_width_bounds_verified=True,
            width_policy=kan.width_policy.describe(kan.WIDTH_PROFILE),relative_mae_tolerance=.01,
            old_kan_models_reused=False,source_mlp_reference=p['baseline']))
    write(out/'summary.json',summary)
    return summary


def require_smoke(path,p):
    assert path is not None
    a,sp=read(path/'artifact-audit.json'),read(path/'protocol.json')
    assert a['passed'] and a['complete'] and a['smoke'] and a['task']==TASK
    assert sp['benchmark_family']==p['benchmark_family'] and sp['training']==p['training']
    assert sp['source_sha256']==p['source_sha256']
    for name,expected in sp['source_sha256'].items():assert sha(name)==expected,name
    if p['benchmark_family']=='kan_compact_expansion':
        assert sp['preference']==p['preference']
        assert sp['search_space']['width_policy']==p['search_space']['width_policy']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['preflight','mlp_smoke','mlp','kan_smoke','kan'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--mlp-reference',type=Path)
    parser.add_argument('--smoke-audit',type=Path)
    args=parser.parse_args();assert args.workers>0
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=out/'run.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    started=time.perf_counter()
    def progress(state,**values):
        write(out/'status.json',dict(state=state,pid=os.getpid(),task=TASK,stage=args.stage,
            workers=args.workers,updated_at=engine.now(),elapsed_seconds=time.perf_counter()-started,**values))
    try:
        progress('preflight')
        sources()
        if args.stage=='preflight':
            original,audit=pair.audit_inputs(TASK,out)
            reused=existing_mlp()
            assert min(f['ranked_features'] for f in audit['folds'])>=max(BUDGETS)
            write(out/'readiness-audit.json',dict(passed=True,task=TASK,training_started=False,
                samples=4764,budgets=BUDGETS,reuse_mlp_folds=list(map(int,reused)),
                mlp_folds_remaining=[f for f in range(5) if str(f) not in reused],
                new_kan_outer_ga_runs=len(BUDGETS)*5,width_policy=kan.width_policy.describe(kan.WIDTH_PROFILE),
                relative_mae_tolerance=.01,source_sha256=sources(),input_sha256=audit['input_sha256']))
            progress('preflight_passed',training_started=False);return
        smoke=args.stage.endswith('_smoke')
        if args.stage.startswith('mlp'):
            original,audit=pair.audit_inputs(TASK,out)
            p=freeze_mlp(out,smoke,audit)
            if not smoke:require_smoke(args.smoke_audit,p)
            for fold in p['folds']:
                if str(fold) not in p['reused_mlp_folds']:
                    progress('starting_fold',fold=fold,arm=MLP_ARM)
                    base.run_fold(out,p,original,MLP_ARM,fold,args.workers,progress)
                report_mlp(out,p)
            summary=report_mlp(out,p)
        else:
            assert args.mlp_reference is not None
            p=freeze_kan(out,smoke,args.mlp_reference.resolve())
            if not smoke:require_smoke(args.smoke_audit,p)
            write(out/'operator-selfcheck.json',kan.self_check())
            write(out/'preference-selfcheck.json',kan.preference.self_check())
            for n in p['budgets']:
                for fold in p['folds']:
                    progress('starting_fold',budget=n,fold=fold['fold'])
                    kan.run_fold(out,p,fold,n,args.workers,progress)
                    report_kan(out,p)
            summary=report_kan(out,p)
        assert summary['complete']
        progress('complete',complete=True,smoke=smoke)
    except BaseException:
        progress('failed',complete=False,traceback=traceback.format_exc());raise
    finally:
        lock.unlink()


if __name__=='__main__':main()
