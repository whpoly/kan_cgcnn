"""Version 2 classification queue: accept independently audited preliminary smoke recovery."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/matbench-expansion-20261101'
SMOKE=ROOT/'benchmarks/classification-fourway-smoke-v2-20261101'
TASKS=['matbench_expt_is_metal','matbench_glass']
RUNNER=ROOT/'scripts/run_native_matbench_classification_v2.py'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(**kw):
    path=OUT/'queue-classification-status.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(pid=os.getpid(),updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw),indent=2));temp.replace(path)

def run(task,smoke):
    out=(SMOKE if smoke else OUT)/task;out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists():
        state=read(out/'status.json')
        if state['state']=='complete':
            audit=read(out/'artifact-audit.json');assert audit['passed'] and audit['smoke']==smoke
            p=read(out/'protocol.json')
            for f,h in {**p['source_sha256'],**p['input_sha256']}.items():assert sha(f)==h
            return
        raise RuntimeError('Existing incomplete run requires PID review before queue resumes: '+str(out))
    prepared=read(OUT/'preparation'/task/'complete.json')
    for f,h in prepared['files'].items():assert sha(f)==h
    csvs=list((OUT/'source').glob(task+'_*.csv'));assert len(csvs)==1
    cmd=[sys.executable,'-u',str(RUNNER),'--task',task,'--csv',str(csvs[0]),'--output',str(out),'--n-jobs','2' if smoke else '3']
    if smoke:cmd.append('--smoke')
    with (out/'run.stdout.log').open('a') as stdout,(out/'run.stderr.log').open('a') as stderr:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
        write(state='smoke_testing' if smoke else 'running',task=task,child_pid=child.pid,command=cmd)
        code=child.wait()
    state=read(out/'status.json') if (out/'status.json').exists() else {}
    if code or state.get('state')!='complete':raise RuntimeError(f'{task}: exit={code}, status={state}')
    audit=read(out/'artifact-audit.json');assert audit['passed'] and audit['complete'] and audit['smoke']==smoke

def main():
    lock=OUT/'queue-classification.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        preliminary=ROOT/'benchmarks/classification-fourway-smoke-20261101/matbench_expt_is_metal'
        while True:
            state=read(preliminary/'status.json')
            if state['state']=='complete':
                audit=read(preliminary/'artifact-audit.json');assert audit['passed'] and audit['smoke']
                break
            if state['state']=='failed':
                recovery=read(preliminary/'recovery-audit.json')
                assert recovery['passed'] and recovery['kind']=='audit_only_correction_no_retraining'
                assert sha(preliminary/'artifact-audit.json')==recovery['artifact_audit_sha256']
                assert read(preliminary/'artifact-audit.json')['passed']
                break
            write(state='waiting_for_preliminary_smoke');time.sleep(15)
        for task in TASKS:run(task,True)
        while True:
            state=read(OUT/'queue-additional-status.json')
            if state['state']=='complete':break
            if state['state']=='failed':raise RuntimeError('Additional regression queue failed; investigate before classification')
            write(state='waiting_for_all_first_wave_regressions',smoke_tasks_passed=TASKS);time.sleep(30)
        for task in TASKS:run(task,False)
        write(state='complete',tasks=TASKS)
    except BaseException:write(state='failed',traceback=traceback.format_exc());raise
    finally:lock.unlink()

if __name__=='__main__':main()
