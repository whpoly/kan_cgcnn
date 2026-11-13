"""Compress descriptive ALE curves into predeclared dimensionless shape families.

This is not target prediction or physical-law discovery. It uses no target
labels. Bin cross-validation measures curve compression only.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import traceback
from modnet_runtime import configure_official_runtime
configure_official_runtime()
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

FAMILIES=['constant','linear','log','inverse','inverse_sqrt','quadratic']
LIMITS=['ALE is a conditional model response, not a causal intervention or physical law.',
        'Bin CV measures compression of a model-response curve, not predictive accuracy on materials.',
        'Overlapping training folds prevent interpreting curve agreement as independent replication.',
        'Correlated features and pretrained embeddings prevent identifying unique mechanisms.',
        'x/x0 is dimensionless; response coefficients carry cm^-1. No extrapolation beyond observed domain.']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)

def design(x,x0,family):
    u=np.asarray(x)/x0
    if family=='constant':return np.ones((len(u),1))
    v={'linear':lambda:u,'log':lambda:np.log(u),'inverse':lambda:1/u,
       'inverse_sqrt':lambda:1/np.sqrt(u),'quadratic':lambda:u}[family]()
    return np.column_stack([np.ones(len(u)),v,u*u]) if family=='quadratic' else np.column_stack([np.ones(len(u)),v])

def fit_curve(c):
    x=np.array(c['x']);y=np.array(c['ale']);w=np.array(c['counts'],dtype=float)
    keep=w>0;x=x[keep];y=y[keep];w=w[keep]
    x0=float(np.median(x));x0=x0 if abs(x0)>1e-12 else 1.
    candidates=[]
    for family in FAMILIES:
        if family in ('log','inverse','inverse_sqrt') and (np.any(x/x0<=0)):continue
        a=design(x,x0,family)
        if len(x)<=a.shape[1]+1:continue
        errors=[]
        for k in range(len(x)):
            mask=np.arange(len(x))!=k
            b=np.linalg.lstsq(a[mask]*np.sqrt(w[mask,None]),y[mask]*np.sqrt(w[mask]),rcond=None)[0]
            errors.append(float(a[k]@b-y[k]))
        coef=np.linalg.lstsq(a*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
        residual=a@coef-y;variance=np.average((y-np.average(y,weights=w))**2,weights=w)
        candidates.append(dict(family=family,coefficients=coef.tolist(),x0=x0,
            bin_cv_rmse=float(np.sqrt(np.average(np.square(errors),weights=w))),
            curve_rmse=float(np.sqrt(np.average(residual**2,weights=w))),
            curve_r2=float(1-np.average(residual**2,weights=w)/variance) if variance>1e-12 else None,
            n_parameters=len(coef)))
    candidates.sort(key=lambda d:(round(d['bin_cv_rmse'],10),d['n_parameters']))
    best=candidates[0]
    expressions=dict(constant='a',linear='a + b*(x/x0)',log='a + b*log(x/x0)',inverse='a + b/(x/x0)',
                     inverse_sqrt='a + b/sqrt(x/x0)',quadratic='a + b*(x/x0) + c*(x/x0)^2')
    return dict(best=best,expression=expressions[best['family']],candidates=candidates,
                domain=[float(x.min()),float(x.max())],n_bins=len(x),
                end_to_end_change=float(y[-1]-y[0]),mean_abs_ale=c['mean_abs_ale'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ale',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    out=args.output.resolve();src=args.ale.resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=dict(source_sha256=sha(__file__),ale_protocol_sha256=sha(src/'protocol.json'),families=FAMILIES,
                  selection='Weighted leave-one-bin-out curve RMSE; ties prefer fewer coefficients.',limitations=LIMITS)
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    write(path,protocol)
    while True:
        state=json.loads((src/'status.json').read_text())
        if state['state']=='failed':raise RuntimeError('ALE prerequisite failed: '+str(state))
        if state['state']=='complete':break
        if not args.wait:raise RuntimeError('ALE prerequisite still running')
        write(out/'status.json',dict(state='waiting_for_ale',pid=os.getpid()))
        time.sleep(30)
    summary=json.loads((src/'summary.json').read_text());assert summary['complete'] and not summary['smoke']
    rows=[];sources={}
    for arm in ['mm_modnet','mm_kan','mm_mv_modnet','mm_mv_kan']:
        for fold in range(5):
            p=src/arm/f'fold-{fold}.json';sources[str(p)]=sha(p);d=json.loads(p.read_text())
            assert d['members']==50
            for feature,curve in d['curves'].items():rows.append(dict(arm=arm,fold=fold,feature=feature,**fit_curve(curve)))
    comparisons=[]
    for label,mlp,kan in [('MM','mm_modnet','mm_kan'),('MM+MatterVial','mm_mv_modnet','mm_mv_kan')]:
        for feature in sorted({r['feature'] for r in rows}):
            rec=dict(bank=label,feature=feature)
            for name,arm in [('mlp',mlp),('kan',kan)]:
                group=[r for r in rows if r['arm']==arm and r['feature']==feature]
                response=next(r for r in summary['rows'] if r['arm']==arm and r['feature']==feature)
                rec[name]=dict(mean_abs_ale=response['mean_abs_ale'],
                    median_curve_correlation=response['median_curve_correlation'],
                    preferred_families=[r['best']['family'] for r in group],
                    end_to_end_changes=[r['end_to_end_change'] for r in group],
                    bin_cv_rmse=[r['best']['bin_cv_rmse'] for r in group],
                    curve_r2=[r['best']['curve_r2'] for r in group])
            comparisons.append(rec)
    write(out/'formulas.json',dict(complete=True,rows=rows,comparisons=comparisons,source_files_sha256=sources,
        target_prediction_mae=None,physical_law_discovered=False,limitations=LIMITS))
    write(out/'status.json',dict(state='complete',pid=os.getpid(),curves=len(rows)))

if __name__=='__main__':
    try:main()
    except BaseException:
        import sys
        if '--output' in sys.argv:write(Path(sys.argv[sys.argv.index('--output')+1])/'status.json',dict(state='failed',pid=os.getpid(),traceback=traceback.format_exc()))
        raise
