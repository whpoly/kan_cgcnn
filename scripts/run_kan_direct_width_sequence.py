"""Audited, resumable steels then phonons fixed-input/direct-width GA queue."""
import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

from run_standardized_matbench_sequence import read, write, sha, now

ROOT=Path(__file__).resolve().parents[1]
TASKS=(('matbench_steels',312,[16,48,32,64,80,96,112]),
       ('matbench_phonons',1265,[16,48,32,64,80,96,112,128,160,192]))


def plan_for(workers):
    runner=ROOT/'scripts/run_kan_direct_width_ga.py'
    stages=[]
    for task,samples,budgets in TASKS:
        short=task[len('matbench_'):]
        smoke=ROOT/'benchmarks'/f'{short}-kan-direct-width-ga-smoke-20261117'
        formal=ROOT/'benchmarks'/f'{short}-kan-direct-width-ga-20261117'
        common=[sys.executable,'-u',str(runner),'--task',task,'--workers',str(workers)]
        stages.append(dict(task=task,samples=samples,phase='smoke',smoke=True,output=str(smoke),
            budgets=[min(budgets),max(budgets)],command=common+['--output',str(smoke),'--smoke']))
        stages.append(dict(task=task,samples=samples,phase='formal',smoke=False,output=str(formal),budgets=budgets,
            command=common+['--output',str(formal),'--smoke-audit',str(smoke)]))
    sources=[runner,Path(__file__).resolve(),ROOT/'scripts/run_standardized_matbench_sequence.py']
    return dict(task_order=[x[0] for x in TASKS],workers=workers,stages=stages,
        source_sha256={str(path):sha(path) for path in sources},
        authorized_scope='Redo KAN GA for all previous main input dimensions in steels and phonons; directly optimize a,b.',
        supersedes='The proposed fixed half-width 16/48 ablation was not launched and is replaced by this GA search.',
        total_ga_runs=85,maximum_formal_member_fits=85000,
        budget='20 candidates x at most 10 generations x 5 inner splits; early stop after 4 unchanged-best generations.',
        ensemble='Native top 10 configurations x 5 inner members; no refit; independent GA per input count and outer fold.',
        baseline='Retain existing standardized MM-MLP and original fixed-width input scans as references.',
        inputs='Exact fixed input count, using the original outer-training MM ranking; no outer-test model selection.',
        widths='Direct independent integer genes a in [1,256], b in [1,128]; two hidden layers, no fixed ratio or virtual-MLP mapping.',
        scoring='Inner-validation MAE drives GA; record actual parameters without an unrequested hard parameter ceiling.',
        running_dielectric='Independent existing sequence continues; no frozen source edits or process interruption.')


def complete(stage):
    out=Path(stage['output'])
    if not (out/'status.json').exists() or read(out/'status.json').get('state')!='complete':return False
    audit,p,s=[read(out/name) for name in ('artifact-audit.json','protocol.json','summary.json')]
    assert audit['passed'] and audit['complete'] and s['complete']
    assert audit['smoke']==stage['smoke'] and p['smoke']==stage['smoke']
    assert p['task']==stage['task'] and p['budgets']==stage['budgets']
    for key in ('source_sha256','reference_sha256'):
        for path,expected in p[key].items():assert sha(path)==expected,path
    if not stage['smoke']:assert audit['outer_samples']==stage['samples']
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();assert args.workers>0
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    plan=plan_for(args.workers)
    if (out/'plan.json').exists():assert read(out/'plan.json')==plan,'Frozen queue plan changed'
    else:write(out/'plan.json',plan)
    if args.prepare_only:
        print(str(out/'plan.json'));return
    lock=out/'run.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    state=dict(pid=os.getpid(),started_at=now(),completed_stages=[],workers=args.workers)
    child=None
    try:
        for stage in plan['stages']:
            for path,expected in plan['source_sha256'].items():assert sha(path)==expected,path
            if not complete(stage):
                name=stage['task']+'--'+stage['phase']
                stdout,stderr=out/(name+'.stdout.log'),out/(name+'.stderr.log')
                with stdout.open('a',encoding='utf-8') as log,stderr.open('a',encoding='utf-8') as err:
                    child=subprocess.Popen(stage['command'],cwd=ROOT,stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                    state.update(state='running',task=stage['task'],phase=stage['phase'],stage_output=stage['output'],
                        child_pid=child.pid,command=stage['command'],stdout=str(stdout),stderr=str(stderr),updated_at=now())
                    write(out/'status.json',state)
                    while True:
                        try:
                            code=child.wait(timeout=15);break
                        except subprocess.TimeoutExpired:
                            state['updated_at']=now();write(out/'status.json',state)
                    if code:raise RuntimeError(name+' failed: '+str(code))
                assert complete(stage),'Child exited without a complete, passed audit'
            state['completed_stages'].append(dict(task=stage['task'],phase=stage['phase'],verified_at=now(),output=stage['output']))
            state.update(state='stage_complete',updated_at=now());write(out/'status.json',state)
        state.update(state='complete',complete=True,updated_at=now());write(out/'status.json',state)
    except BaseException:
        if child is not None and child.poll() is None:child.terminate()
        state.update(state='failed',complete=False,traceback=traceback.format_exc(),updated_at=now())
        write(out/'status.json',state);raise
    finally:lock.unlink()


if __name__=='__main__':main()
