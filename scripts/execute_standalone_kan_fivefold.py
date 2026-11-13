"""Durable one-shot pipeline: remaining folds, then independent five-fold report."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--n-jobs',type=int,default=4)
    parser.add_argument('--analysis-python',type=Path,required=True);a=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    def status(state,**kwargs):
        path=out/'pipeline-status.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),**kwargs),indent=2),encoding='utf-8');tmp.replace(path)
    try:
        command=[sys.executable,'-u',str(root/'scripts/run_standalone_kan_fivefold.py'),'--output',str(out),'--n-jobs',str(a.n_jobs)]
        p=subprocess.Popen(command,cwd=str(root),stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        status('training',training_pid=p.pid);code=p.wait()
        if code:status('failed',stage='training',returncode=code);return
        status('auditing')
        with (out/'final-analysis.log').open('w',encoding='utf-8') as log:
            p=subprocess.run([str(a.analysis_python),str(root/'scripts/analyze_standalone_kan_fivefold.py')],cwd=str(root),
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONIOENCODING='utf-8'),
                creationflags=subprocess.CREATE_NO_WINDOW,timeout=1800)
        if p.returncode:status('failed',stage='audit',returncode=p.returncode);return
        result=json.loads((out/'analysis.json').read_text(encoding='utf-8'))
        status('complete',verified_core_fits=result['verified_core_fits'],completed_folds=result['completed_folds'],report=str(out/'report.md'))
    except Exception as error:status('failed',reason=repr(error));raise


if __name__=='__main__':main()
