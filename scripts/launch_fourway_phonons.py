"""Hidden detached launcher; logs and lifecycle stay local to this benchmark."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n-jobs',type=int,default=4)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    status=out/'status.json'
    if status.exists():
        prior=json.loads(status.read_text(encoding='utf-8'))
        if prior['state'] not in ('complete','failed','baseline_quality_requires_investigation','historical_reproduction_verified'):
            # Never launch a second writer into a potentially active directory.
            raise ValueError('Run may already be active; inspect status before explicitly resuming')
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    record=out/f'launch-{stamp}.json'
    command=[sys.executable,'-u',str(root/'scripts/run_fourway_phonons.py'),'--output',str(out),'--n-jobs',str(args.n_jobs)]
    so,se=out/(record.stem+'.stdout.log'),out/(record.stem+'.stderr.log')
    with so.open('xb',buffering=0) as stdout,se.open('xb',buffering=0) as stderr:
        process=subprocess.Popen(command,cwd=str(root),env=dict(os.environ,PYTHONIOENCODING='utf-8'),
            stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    result=dict(pid=process.pid,started_at=datetime.now(timezone.utc).isoformat(),arguments=command,
                stdout=str(so),stderr=str(se),jobs=args.n_jobs)
    record.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
