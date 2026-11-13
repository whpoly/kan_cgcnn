"""Wait for existing steel/dielectric queue, then run prepared additional regressions."""
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
JOBS=[('matbench_expt_gap','matbench_expt_gap_featurizedMM2020_mattervial.csv'),
      ('matbench_jdft2d','matbench_jdft2d_featurizedMM2020_mattervial.csv'),
      ('matbench_log_gvrh','matbench_log_gvrh_featurizedMM2020Struct_mattervial.csv')]

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(**kw):
    p=OUT/'queue-additional-status.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(pid=os.getpid(),updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw),indent=2),encoding='utf-8');t.replace(p)

def main():
    lock=OUT/'queue-additional.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        smoke=read(ROOT/'benchmarks/matbench-expansion-smoke-v2-20261101/matbench_expt_gap/artifact-audit.json')
        assert smoke['passed'] and smoke['complete'] and smoke['smoke']
        while True:
            parent=read(OUT/'queue-status.json')
            if parent['state']=='complete':break
            if parent['state']!='running':raise RuntimeError('Prior regression queue needs investigation: '+str(parent))
            write(state='waiting_for_steels_and_dielectric');time.sleep(30)
        for task,csv in JOBS:
            prepared=OUT/'preparation'/task/'complete.json'
            while not prepared.exists():
                state=read(OUT/'preparation/status.json')
                if state['state']=='failed':raise RuntimeError('Native feature preparation failed')
                write(state='waiting_for_preparation',task=task);time.sleep(30)
            for path,h in read(prepared)['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
            out=OUT/task;out.mkdir(exist_ok=True)
            status=out/'status.json'
            if status.exists() and read(status)['state']=='complete':continue
            cmd=[sys.executable,'-u',str(ROOT/'scripts/run_native_matbench_regression_expansion_v2.py'),
                 '--task',task,'--csv',str(OUT/'source'/csv),'--output',str(out),'--n-jobs','3']
            with (out/'run.stdout.log').open('a') as stdout,(out/'run.stderr.log').open('a') as stderr:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
                write(state='running',task=task,child_pid=child.pid,command=cmd);code=child.wait()
            state=read(status) if status.exists() else {}
            if code or state.get('state')!='complete':raise RuntimeError(f'{task}: exit={code}, status={state}')
        write(state='complete',tasks=[j[0] for j in JOBS])
    except BaseException:write(state='failed',traceback=traceback.format_exc());raise
    finally:lock.unlink()

if __name__=='__main__':main()
