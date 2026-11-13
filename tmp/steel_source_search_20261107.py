import concurrent.futures as cf
import json, urllib.request, hashlib, ssl, certifi
from pathlib import Path

OUT=Path('benchmarks/steels-paper-reproduction-20261102/source-search-20261107')
OUT.mkdir(parents=True,exist_ok=True)
def fetch(item):
    key,url=item
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'steel-method-reproduction-audit'})
        with urllib.request.urlopen(req,timeout=40,context=ssl.create_default_context(cafile=certifi.where())) as r: raw=r.read()
        data=json.loads(raw)
        (OUT/(key+'.json')).write_bytes(raw)
        return key,data
    except Exception as e: return key,{'error':str(e),'url':url}

if __name__=='__main__':
    repos={'MatterVial_SupportData':'main','MatterVial':'master','MatterVial_public':'master','modnet':'master','modnet_gnn_enhanced_v0.2.1':'master','modnet_misctools':'master','modnet-pt':'main'}
    jobs=[]
    for repo,branch in repos.items():
        base='https://api.github.com/repos/rogeriog/'+repo
        for kind,path in [('tree','git/trees/'+branch+'?recursive=1'),('branches','branches?per_page=100'),('tags','tags?per_page=100')]:
            jobs.append((repo+'-'+kind,base+'/'+path))
    for repo in ['MatterVial_SupportData','MatterVial']:
        base='https://api.github.com/repos/rogeriog/'+repo
        for kind,path in [('commits','commits?per_page=100'),('releases','releases?per_page=100'),('issues','issues?state=all&per_page=100'),('forks','forks?per_page=100')]:
            jobs.append((repo+'-'+kind,base+'/'+path))
    jobs += [('figshare','https://api.figshare.com/v2/articles/27132093')]
    results=dict(cf.ThreadPoolExecutor(max_workers=5).map(fetch,jobs))
    summary={}
    for key,d in results.items():
        if isinstance(d,dict) and 'error' in d: summary[key]=d
        elif key.endswith('-tree'):
            paths=[x['path'] for x in d['tree'] if any(t in x['path'].lower() for t in ['rerank','steel','shap']) and not any(t in x['path'] for t in ['shap_values/','shap_plots/','shapmeta','shapvalues/'])]
            summary[key]={'sha':d['sha'],'truncated':d.get('truncated'),'paths':paths[:70]}
        elif key.endswith(('-branches','-tags')): summary[key]=[x['name'] for x in d]
        elif key.endswith('-commits'): summary[key]=[{'sha':x['sha'],'message':x['commit']['message'][:150]} for x in d]
        elif key.endswith('-issues'): summary[key]=[{'number':x['number'],'title':x['title'],'body':x.get('body'),'url':x['html_url']} for x in d]
        elif key.endswith('-forks'): summary[key]=[x['full_name'] for x in d]
        elif key.endswith('-releases'): summary[key]=[{'tag':x['tag_name'],'assets':[a['name'] for a in x['assets']]} for x in d]
        else: summary[key]={'url':d.get('url'),'files':[{'name':f['name'],'size':f['size'],'download_url':f['download_url']} for f in d.get('files',[])]}
    (OUT/'discovery-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
