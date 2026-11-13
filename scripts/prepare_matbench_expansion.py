"""Extract requested, already-downloaded MatterVial CSVs into a new workspace."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import traceback

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
    source=Path(__file__).resolve().parents[1]/'benchmarks/mattervial-phonons-20261023/source'
    tasks=['steels','jdft2d','expt_gap','dielectric','expt_is_metal','glass','log_gvrh']
    names={f'matbench_{x}' for x in tasks};records=[]
    with tarfile.open(source/'featurized_datasets.tar.gz','r|gz') as archive:
        for member in archive:
            base=Path(member.name).name
            task=next((x for x in names if base.startswith(x+'_featurized')),None)
            if task is None or not member.isfile():continue
            write(out/'status.json',dict(state='extracting',pid=os.getpid(),task=task,completed=records))
            target=out/base
            if not target.exists():
                temp=target.with_suffix('.partial')
                with archive.extractfile(member) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
                assert temp.stat().st_size==member.size
                temp.replace(target)
            assert target.stat().st_size==member.size
            h=hashlib.sha256()
            with target.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            records.append(dict(task=task,path=str(target),bytes=member.size,sha256=h.hexdigest()))
            write(out/'inventory.json',records)
            if len(records)==len(tasks):break
    assert {r['task'] for r in records}==names
    write(out/'status.json',dict(state='complete',pid=os.getpid(),tasks=tasks))

if __name__=='__main__':main()
