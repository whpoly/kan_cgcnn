"""Detach a resumable quick pilot with hidden Windows processes and direct logs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--n-jobs',type=int,default=2)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output,record=args.output.resolve(),args.record.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if record.parent!=output or record.exists():
        raise ValueError('Use a fresh launch record inside the output directory.')
    out=output/(record.stem+'.stdout.log')
    err=output/(record.stem+'.stderr.log')
    argv=[sys.executable,'-u',str(root/'scripts/run_mattervial_quick_pilot.py'),'--output',str(output),'--n-jobs',str(args.n_jobs)]
    with out.open('xb',buffering=0) as stdout,err.open('xb',buffering=0) as stderr:
        process=subprocess.Popen(argv,cwd=str(root),env=dict(os.environ,PYTHONIOENCODING='utf-8'),
            stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    record.write_text(json.dumps(dict(pid=process.pid,started_at=datetime.now(timezone.utc).isoformat(),
        arguments=argv,stdout=str(out),stderr=str(err),jobs=args.n_jobs),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
