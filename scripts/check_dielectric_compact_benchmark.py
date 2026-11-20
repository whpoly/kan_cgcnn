"""Read-only data, routing and dependency checks; no model is fitted."""
import copy
import gc
import json
from pathlib import Path
from unittest.mock import patch

import run_dielectric_compact_benchmark as runner
import run_dielectric_compact_sequence as queue


def main():
    out=runner.ROOT/'benchmarks/dielectric-compact-benchmark-review-20261118'
    out.mkdir(parents=True,exist_ok=True)
    plan=queue.make_plan(2)
    queue.check_sources(plan)
    assert [s['phase'] for s in plan['stages']]==['mlp_smoke','mlp','kan_smoke','kan']
    assert all('run_dielectric_compact_benchmark.py' in s['command'][2] for s in plan['stages'])
    assert plan['widths']['profile']=='moderate_expansion' and plan['relative_mae_tolerance']==.01
    assert runner.engine.run_fit is runner.run_fit
    assert runner.kan.native.neurons_from_genes is runner.neurons
    selection=runner.read(runner.OLD_REFERENCE/runner.MLP_ARM/'fold-0/frozen-selection.json')
    genes=selection['members'][0]['genes']
    assert runner.neurons(genes)==runner.mlp_neurons(genes)
    assert runner.neurons({'kan_a':32,'kan_b':16})==[[32],[16],[1]]
    with patch.object(runner.pair,'run_fit',return_value='mlp') as mlp, patch.object(runner.kan,'run_fit',return_value='kan') as kan:
        assert runner.run_fit(dict(arm=runner.MLP_ARM,variant='modnet',genes=genes))=='mlp'
        assert runner.run_fit(dict(arm='kan_compact_expansion_k016',variant='bspline',genes={}))=='kan'
        try:
            runner.run_fit(dict(arm='mm_scaled_kan',variant='bspline',genes={}))
        except AssertionError:
            pass
        else:
            raise AssertionError('Old KAN route was accepted')
        assert mlp.call_count==kan.call_count==1

    for state in ['running','failed','paused','stopped_by_user_for_new_benchmark']:
        with patch.object(queue,'read',return_value={'state':state}), patch.object(queue.upstream,'complete') as verify:
            assert not queue.dependency_ready(plan)[0]
            verify.assert_not_called()
    def dependency_read(path):
        if Path(path).name=='status.json':return dict(state='complete',complete=True)
        return dict(task_order=['matbench_steels','matbench_phonons'],stages=[1,2,3,4])
    with patch.object(queue,'read',side_effect=dependency_read),patch.object(queue.upstream,'complete',return_value=True) as verify:
        assert queue.dependency_ready(plan)[0] and verify.call_count==4
    with patch.object(queue,'read',side_effect=dependency_read),patch.object(queue.upstream,'complete',return_value=False):
        try:
            queue.dependency_ready(plan)
        except AssertionError:
            pass
        else:
            raise AssertionError('Unaudited dependency was accepted')

    audit=runner.read(queue.PREFLIGHT/'input-audit.json')
    partial=out/'partial-mlp-reference'
    p=runner.freeze_mlp(partial,False,audit)
    assert p['arms']==[runner.MLP_ARM] and sorted(p['reused_mlp_folds'])==['0','1']
    report=runner.report_mlp(partial,p)
    assert not report['complete'] and len(report['rows'])==2
    assert not (partial/'artifact-audit.json').exists()

    # Full-input MLP and each fixed-input KAN must use the same inner samples.
    checked=0
    for fold in range(5):
        cache=runner.pair.OLD/runner.TASK/f'fold-{fold}'/'official-train-moddata.pkl.gz'
        data=runner.MODData.load(cache)
        seed=p['seed']+1000000+fold*100000
        runner.kan.set_seed(seed)
        full=runner.engine.EqualBudgetGA(data,out/'unused',cache,p,runner.MLP_ARM,fold,seed,lambda *a,**k:None,1)
        ids=runner.np.asarray(runner.kan.sample_ids(full.train_data))
        expected={str(i):dict(train_ids=ids[tr].tolist(),validation_ids=ids[va].tolist())
            for i,(tr,va) in enumerate(runner.kan.matbench_kfold_splits(full.train_data,n_splits=5))}
        if fold<2:
            saved=runner.read(runner.OLD_REFERENCE/runner.MLP_ARM/f'fold-{fold}'/'frozen-selection.json')
            assert all(expected[str(j['inner'])]==dict(train_ids=j['train_ids'],validation_ids=j['validation_ids']) for j in saved['members'])
        for n in [16,192]:
            fixed=copy.copy(data);fixed.optimal_features=data.get_optimal_descriptors()[:n]
            runner.kan.set_seed(seed)
            ga=runner.kan.DirectGA(fixed,out/'unused',cache,p,f'kan_compact_expansion_k{n:03d}',fold,seed,lambda *a,**k:None,1)
            ids=runner.np.asarray(runner.kan.sample_ids(ga.train_data))
            for i,(tr,va) in enumerate(runner.kan.matbench_kfold_splits(ga.train_data,n_splits=5)):
                assert expected[str(i)]==dict(train_ids=ids[tr].tolist(),validation_ids=ids[va].tolist())
                checked+=1
            del ga,fixed
        del data,full
        gc.collect()
    assert checked==50
    report=dict(passed=True,training_started=False,old_kan_route_rejected=True,
        completed_mlp_folds_reused=[0,1],missing_mlp_folds=[2,3,4],partial_mlp_not_marked_complete=True,
        original_inner_partitions_verified=checked,mlp_and_new_kan_worker_routes_verified=True,
        upstream_completion_and_audit_gate_verified=True,stages=[s['phase'] for s in plan['stages']],
        source_sha256={**plan['source_sha256'],str(Path(__file__).resolve()):runner.sha(__file__)})
    runner.write(out/'adapter-queue-audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
