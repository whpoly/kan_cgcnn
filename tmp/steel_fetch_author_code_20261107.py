import sys, json, urllib.request, ssl, certifi, concurrent.futures as cf, hashlib
from pathlib import Path
OUT=Path('benchmarks/steels-paper-reproduction-20261102/source-search-20261107')
def run(item):
    repo,sha,path=item
    url='https://raw.githubusercontent.com/rogeriog/'+repo+'/'+sha+'/'+path
    dest=OUT/'code'/repo/path
    try:
        with urllib.request.urlopen(url,timeout=35,context=ssl.create_default_context(cafile=certifi.where())) as r: b=r.read()
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
        if path.endswith('.ipynb'):
            obj=json.loads(b); txt='\n'.join(''.join(c['source']) for c in obj['cells'] if c['cell_type']=='code')
            dest=dest.with_suffix('.code.py');dest.write_text(txt,encoding='utf-8')
        return dict(repo=repo,path=path,url=url,sha256=hashlib.sha256(b).hexdigest())
    except Exception as e:return dict(url=url,error=str(e))
jobs=[]
for repo in ['MatterVial_SupportData','MatterVial','MatterVial_public','modnet_misctools','modnet_gnn_enhanced_v0.2.1']:
    t=json.loads((OUT/(repo+'-tree.json')).read_text());seen=set()
    for x in t['tree']:
        p=x['path']
        wanted=(repo=='MatterVial_SupportData' and p.endswith(('.py','.md','.json')) and 'metrics' not in p and ('modnet_mattervial' in p or p.endswith('.py'))) or p.endswith('.ipynb') or (repo in ['MatterVial','MatterVial_public'] and p.endswith('.py') and ('interpreter/' in p or 'roost_models' in p)) or repo=='modnet_misctools' and p.endswith('.py')
        if wanted and x['sha'] not in seen and x.get('size',0)<2000000:
            seen.add(x['sha']);jobs.append((repo,t['sha'],p))
results=list(cf.ThreadPoolExecutor(max_workers=6).map(run,jobs))
(OUT/'downloaded-code-provenance.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps({'requested':len(jobs),'downloaded':sum('error' not in r for r in results),'errors':[r for r in results if 'error' in r]}))
