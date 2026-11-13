"""One-shot final analysis step for the already running durable training job."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import subprocess
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--analysis-python',type=Path,required=True);a=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve()
    expected=root/'benchmarks/standalone-kan-confirmation-20261028'
    if out!=expected.resolve():raise ValueError('Analyzer is scoped to this frozen study')
    status_path=out/'report-status.json'
    def record(state,**kwargs):
        data=dict(state=state,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),**kwargs)
        tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(status_path)
    record('waiting_for_training')
    deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline:
        status=json.loads((out/'status.json').read_text(encoding='utf-8-sig'))
        if status['state']=='failed':
            record('failed',reason='Training reported failure; inspect training status.json');return
        if status['state']=='complete':
            record('auditing')
            with (out/'final-analysis.log').open('w',encoding='utf-8') as log:
                process=subprocess.run([str(a.analysis_python),str(root/'scripts/analyze_standalone_kan_confirmation.py')],
                    cwd=str(root),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    env=dict(os.environ,PYTHONIOENCODING='utf-8'),timeout=900,creationflags=subprocess.CREATE_NO_WINDOW)
            if process.returncode:
                record('failed',reason='Independent audit failed; inspect final-analysis.log',returncode=process.returncode)
            else:
                result=json.loads((out/'analysis.json').read_text(encoding='utf-8'))
                record('complete',verified_fits=result['verified_fits'],gate_passed=result['gate']['passed'],
                       outer_test_evaluated=result['outer_test_evaluated'],report=str(out/'report.md'))
            return
        time.sleep(15)
    record('failed',reason='Six-hour training wait expired; training was not terminated')


if __name__=='__main__':main()
