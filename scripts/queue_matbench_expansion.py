"""Sequential, resumable launch of the two regression tasks with audited caches."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/matbench-expansion-20261101'
JOBS=[('matbench_steels','matbench_steels_featurizedMM2020Comp_mattervial.csv'),
      ('matbench_dielectric','matbench_dielectric_featurizedMM2020Struct_mattervial.csv')]

def write(**kw):
    p=OUT/'queue-status.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(pid=os.getpid(),updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw),indent=2),encoding='utf-8');t.replace(p)

def main():
    lock=OUT/'queue.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Queue lock exists; verify its process before resuming.')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        for task,csv in JOBS:
            out=OUT/task;out.mkdir(exist_ok=True)
            status=out/'status.json'
            if status.exists() and json.loads(status.read_text())['state']=='complete':continue
            cmd=[sys.executable,'-u',str(ROOT/'scripts/run_native_matbench_expansion.py'),
                 '--task',task,'--csv',str(OUT/'source'/csv),'--output',str(out),'--n-jobs','3']
            with (out/'run.stdout.log').open('a') as stdout,(out/'run.stderr.log').open('a') as stderr:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
                write(state='running',task=task,child_pid=child.pid,command=cmd)
                code=child.wait()
            result=json.loads(status.read_text()) if status.exists() else {}
            if code or result.get('state')!='complete':
                write(state='needs_investigation',task=task,exit_code=code,child_status=result);return
        write(state='complete',tasks=[x[0] for x in JOBS])
    except BaseException as exc:
        write(state='failed',error=repr(exc));raise
    finally:lock.unlink()

if __name__=='__main__':main()
