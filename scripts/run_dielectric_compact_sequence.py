"""Queue the new dielectric benchmark after the existing steel/phonons queue."""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from run_standardized_matbench_sequence import read,write,sha,now
import run_kan_compact_expansion_sequence as upstream
import kan_input_width_policy as widths

ROOT=Path(__file__).resolve().parents[1]
AFTER=ROOT/'benchmarks/kan-compact-expansion-ga-sequence-20261117'
PREFLIGHT=ROOT/'benchmarks/dielectric-compact-benchmark-preflight-20261118'
BUDGETS=[16,48,32,64,80,96,112,128,160,192]


def make_plan(workers):
    readiness=read(PREFLIGHT/'readiness-audit.json')
    assert readiness['passed'] and not readiness['training_started']
    assert readiness['budgets']==BUDGETS and readiness['relative_mae_tolerance']==.01
    runner=ROOT/'scripts/run_dielectric_compact_benchmark.py'
    mlp_smoke=ROOT/'benchmarks/dielectric-standardized-mlp-smoke-20261118'
    mlp=ROOT/'benchmarks/dielectric-standardized-mlp-reference-20261118'
    kan_smoke=ROOT/'benchmarks/dielectric-compact-expansion-ga-smoke-20261118'
    kan=ROOT/'benchmarks/dielectric-compact-expansion-ga-20261118'
    common=[sys.executable,'-u',str(runner),'--workers',str(workers)]
    stages=[]
    for phase,out,extra in [('mlp_smoke',mlp_smoke,[]),
        ('mlp',mlp,['--smoke-audit',str(mlp_smoke)]),
        ('kan_smoke',kan_smoke,['--mlp-reference',str(mlp)]),
        ('kan',kan,['--mlp-reference',str(mlp),'--smoke-audit',str(kan_smoke)])]:
        stages.append(dict(phase=phase,output=str(out),smoke=phase.endswith('_smoke'),
            benchmark_family='mm_scaled_modnet' if phase.startswith('mlp') else 'kan_compact_expansion',
            command=common+['--stage',phase,'--output',str(out)]+extra))
    sources=dict(readiness['source_sha256'])
    for path in [Path(__file__).resolve(),Path(upstream.__file__).resolve(),ROOT/'scripts/run_standardized_matbench_sequence.py']:
        sources[str(path)]=sha(path)
    return dict(task='matbench_dielectric',workers=workers,dependency=str(AFTER),
        dependency_plan_sha256=sha(AFTER/'plan.json'),readiness_audit=str(PREFLIGHT/'readiness-audit.json'),
        readiness_audit_sha256=sha(PREFLIGHT/'readiness-audit.json'),source_sha256=sources,stages=stages,
        task_order=['matbench_steels','matbench_phonons','matbench_dielectric'],
        authorization='User stopped old dielectric and requested the new benchmark later.',
        budgets=BUDGETS,widths=widths.describe('moderate_expansion'),relative_mae_tolerance=.01,
        selection='Same frozen compact GA as steel/phonons: 1% validation-MAE bands, then fewer actual parameters.',
        ensemble='10 configurations x 5 inner members, selected before outer test scoring.',
        target_scaling='Inner-training z-score only, ddof=0; MAE reported in original target units.',
        mlp='Retain verified completed standardized MLP folds 0,1; train missing folds 2,3,4 using the identical MLP method.',
        kan='Fresh GA at every fixed input count and outer fold; never resume old dielectric KAN or fixed-width scan.',
        additional_formal_ga_runs=53,maximum_additional_formal_fits=53000,
        deferred_training='No dielectric smoke or formal fitting before the upstream queue is complete and audited.')


def check_sources(plan):
    assert sha(AFTER/'plan.json')==plan['dependency_plan_sha256']
    assert sha(PREFLIGHT/'readiness-audit.json')==plan['readiness_audit_sha256']
    for path,expected in plan['source_sha256'].items():assert sha(path)==expected,path


def dependency_ready(plan):
    state=read(Path(plan['dependency'])/'status.json')
    if state.get('state')!='complete':
        return False,state
    assert state.get('complete') is True
    p=read(Path(plan['dependency'])/'plan.json')
    assert p['task_order']==['matbench_steels','matbench_phonons']
    assert all(upstream.complete(stage) for stage in p['stages'])
    return True,state


def stage_complete(stage):
    out=Path(stage['output'])
    if not (out/'status.json').exists() or read(out/'status.json').get('state')!='complete':return False
    a,p,s=[read(out/name) for name in ('artifact-audit.json','protocol.json','summary.json')]
    assert a['passed'] and a['complete'] and s['complete']
    assert a['task']==p['task']=='matbench_dielectric'
    assert a['smoke']==p['smoke']==stage['smoke']
    assert a['benchmark_family']==p['benchmark_family']==stage['benchmark_family']
    if not stage['smoke']:assert a['outer_samples']==4764
    if stage['benchmark_family']=='kan_compact_expansion':
        assert p['budgets']==([16,192] if stage['smoke'] else BUDGETS)
        assert p['preference']['relative_mae_tolerance']==.01
        assert p['search_space']['width_policy']==widths.describe('moderate_expansion')
        assert a['accuracy_size_selection_and_stopping_verified'] and not a['old_kan_models_reused']
    for key in ('source_sha256','reference_sha256','input_sha256'):
        for path,expected in p.get(key,{}).items():assert sha(path)==expected,path
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();assert args.workers>0
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    plan=make_plan(args.workers)
    check_sources(plan)
    if (out/'plan.json').exists():assert read(out/'plan.json')==plan,'Frozen plan changed'
    else:write(out/'plan.json',plan)
    if args.prepare_only:
        print(str(out/'plan.json'));return
    lock=out/'run.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
    state=dict(pid=os.getpid(),started_at=now(),workers=args.workers,completed_stages=[],training_started=False)
    child=None
    try:
        while True:
            check_sources(plan)
            ready,dependency=dependency_ready(plan)
            if ready:break
            state.update(state='waiting_for_steel_and_phonons',updated_at=now(),
                dependency=str(AFTER),dependency_state={k:dependency.get(k) for k in ('state','task','phase','updated_at')})
            write(out/'status.json',state)
            time.sleep(30)
        for stage in plan['stages']:
            check_sources(plan)
            if not stage_complete(stage):
                stdout,stderr=out/(stage['phase']+'.stdout.log'),out/(stage['phase']+'.stderr.log')
                with stdout.open('a',encoding='utf-8') as log,stderr.open('a',encoding='utf-8') as err:
                    child=subprocess.Popen(stage['command'],cwd=ROOT,stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                    state.update(state='running',training_started=True,phase=stage['phase'],child_pid=child.pid,
                        stage_output=stage['output'],command=stage['command'],updated_at=now())
                    write(out/'status.json',state)
                    while True:
                        try:
                            code=child.wait(timeout=15);break
                        except subprocess.TimeoutExpired:
                            state['updated_at']=now();write(out/'status.json',state)
                    if code:raise RuntimeError(stage['phase']+' failed: '+str(code))
                assert stage_complete(stage),'Child exited without a passed audit'
            state['completed_stages'].append(dict(phase=stage['phase'],output=stage['output'],verified_at=now()))
            state.update(state='stage_complete',updated_at=now());write(out/'status.json',state)
        state.update(state='complete',complete=True,updated_at=now());write(out/'status.json',state)
    except BaseException:
        if child is not None and child.poll() is None:child.terminate()
        state.update(state='failed',complete=False,traceback=traceback.format_exc(),updated_at=now())
        write(out/'status.json',state);raise
    finally:lock.unlink()


if __name__=='__main__':main()
