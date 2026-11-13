"""Hidden detached launcher for the complete five-fold training/report pipeline."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--n-jobs',type=int,default=4);parser.add_argument('--analysis-python',type=Path,required=True);a=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();record=a.record.resolve();out.mkdir(parents=True,exist_ok=True)
    if record.parent!=out or record.exists():raise ValueError('New launch record required')
    command=[sys.executable,'-u',str(root/'scripts/execute_standalone_kan_fivefold.py'),'--output',str(out),
             '--n-jobs',str(a.n_jobs),'--analysis-python',str(a.analysis_python)]
    stdout=out/(record.stem+'.stdout.log');stderr=out/(record.stem+'.stderr.log')
    with stdout.open('xb',buffering=0) as so,stderr.open('xb',buffering=0) as se:
        p=subprocess.Popen(command,cwd=str(root),env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdin=subprocess.DEVNULL,
            stdout=so,stderr=se,close_fds=True,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    record.write_text(json.dumps(dict(pid=p.pid,started_at=datetime.now(timezone.utc).isoformat(),arguments=command,
        stdout=str(stdout),stderr=str(stderr),jobs=a.n_jobs),indent=2),encoding='utf-8')

if __name__=='__main__':main()
