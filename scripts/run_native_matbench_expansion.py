"""Versioned regression expansion of the audited phonons protocol.

Initial supported tasks have already verified native feature/split caches.
No edits to frozen phonons sources. Classification is deliberately rejected.
"""
import argparse
import copy
import gc
import os
import pickle
import shutil
import time
import traceback
from pathlib import Path
import run_native_fourway as e
import native_fourway_models as models
from scaled_native_spline import fit_spline
import relaxed_spline_policy as policy
from run_modnet_official_matched import np, pd, tf, MODData, write_json, file_hash, digest, set_seed

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'benchmarks/official-aligned-20261021'
SUPPORTED={'matbench_steels':0,'matbench_dielectric':2}
models.matched_widths=policy.matched_widths
e.matched_widths=policy.matched_widths
DATA={}


def run_fit(job):
    e.check_sources(job['protocol_path'],job['protocol_fingerprint'])
    p=e.read(job['protocol_path']);directory=Path(job['directory']);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'result.json';signature=digest(job)
    if path.exists():
        r=e.read(path);assert r['job_signature']==signature
        assert file_hash(directory/'model.pkl')==r['model_sha256'];return r
    key=(job['training_cache'],job['training_sha256'])
    if key not in DATA:
        assert file_hash(key[0])==key[1];DATA[key]=MODData.load(key[0])
    data=DATA[key];lookup={s:i for i,s in enumerate(data.df_targets.index)}
    assert set(job['train_ids']).isdisjoint(job['validation_ids'])
    assert set(job['train_ids']+job['validation_ids'])==set(lookup)
    train,val=data.split(([lookup[x] for x in job['train_ids']],[lookup[x] for x in job['validation_ids']]))
    # Reuse only semantically identical old native baseline fits, with provenance.
    prior=OLD/p['task']/f"fold-{job['fold']}"/'modnet'/directory.name
    r=None
    if job['arm']=='mm_modnet' and not job['smoke'] and (prior/'result.json').exists():
        legacy=e.read(prior/'result.json')
        match=all(legacy[k]==job[k] for k in ['genes','seed','train_ids','validation_ids'])
        match=match and legacy['features']==list(train.get_optimal_descriptors()) and not legacy['fast']
        if match:
            shutil.copyfile(prior/'model.pkl',directory/'model.pkl')
            r=dict(legacy,directory=str(directory),reused_from=str(prior),
                   reused_result_sha256=file_hash(prior/'result.json'),
                   reused_model_sha256=file_hash(prior/'model.pkl'))
    if r is None:
        individual=e.IntegerBoundIndividual(max_feat=len(data.optimal_features),num_classes=data.num_classes,
            multi_label=False,targets=[[list(data.target_names)]],weights={n:1 for n in data.target_names})
        individual.genes=dict(job['genes'])
        fitjob=dict(individual=individual,train_data=train,val_data=val,directory=str(directory),
            seed=job['seed'],fast=job['smoke'],variant=job['variant'],run_fingerprint=job['run_fingerprint'])
        original=e.native.fit_kan
        try:
            e.native.fit_kan=lambda i,t,v,variant,fast=False:fit_spline(i,t,v,variant,fast,seed=job['seed'])
            r=e.native.evaluate_job(fitjob)
        finally:e.native.fit_kan=original
    r.update(job_signature=signature,model_sha256=file_hash(directory/'model.pkl'),
             arm=job['arm'],task=p['task'],fold=job['fold'],generation=job['generation'],candidate=job['candidate'],inner=job['inner'])
    if job['variant']=='bspline':
        widths=tuple(x[0] for x in e.neurons_from_genes(job['genes']))
        mapping=policy.describe(job['genes']['n_feat'],widths)
        with (directory/'model.pkl').open('rb') as f:model=pickle.load(f)
        r.update(parameter_mapping=mapping,paired_modnet_parameters=mapping['reference_parameters'],
                 spline_widths=mapping['widths'],target_scaling=model.target_scaling)
        assert r['parameters']==mapping['spline_parameters']
    write_json(path,r);return r


e.run_fit=run_fit  # also installed on Windows spawn import


def load_banks(p,out):
    task=p['task'];data=MODData.load(p['original'])
    mm=data.df_featurized.replace([np.inf,-np.inf],np.nan).fillna(0)
    frame=pd.read_csv(p['csv']).set_index('material_id')
    short=task.replace('matbench_','').replace('_','-')
    ids=[f'mb-{short}-{i+1:0{len(str(len(mm)))}d}' for i in range(len(mm))]
    assert frame.index.is_unique and set(ids)==set(frame.index)
    frame=frame.loc[ids]
    np.testing.assert_allclose(frame['target'],data.df_targets.iloc[:,0],rtol=0,atol=1e-8)
    if 'composition' in frame:
        from pymatgen.core import Composition
        for a,b in zip(data.df_structure.iloc[:,0],frame.composition):
            assert a.composition.fractional_composition.almost_equals(Composition(b).fractional_composition)
    else:
        import json
        from pymatgen.core import Structure
        for a,b in zip(data.df_structure.iloc[:,0],frame.structure):
            s=Structure.from_dict(json.loads(b));assert list(a.species)==list(s.species)
            np.testing.assert_allclose(a.lattice.matrix,s.lattice.matrix,atol=1e-8)
            np.testing.assert_allclose(a.frac_coords,s.frac_coords,atol=1e-8)
    prefixes=('ROOST_mpgap_',) if task=='matbench_steels' else ('MEGNet_OFMEncoded_v1_','MVL16_','MVL32_','ROOST_mpgap_','ORB_v3_')
    extras=sorted(c for c in frame if c.startswith(prefixes) and 'SISSO' not in c.upper())
    assert extras
    mv=frame[extras].copy();mv.index=mm.index;mv=mv.replace([np.inf,-np.inf],np.nan).fillna(0)
    enhanced=pd.concat([mm,mv],axis=1);assert enhanced.columns.is_unique
    assert not any('SISSO' in c.upper() for c in enhanced)
    write_json(out/'data-audit.json',dict(task=task,samples=len(mm),identities_verified=True,
        mm_features=len(mm.columns),added_features=len(extras),excluded_sisso=sum('SISSO' in c.upper() for c in frame),
        extras=extras,ids=ids,original_mm_preserved=True))
    return dict(mm=mm,mm_mv=enhanced),data.df_targets


def prepare(out,p,banks,y,bank,fold,n_jobs,progress):
    source=OLD/p['task']/f'fold-{fold}';split=e.read(source/'outer-split.json')
    cache=out/'inputs'/bank/f'fold-{fold}'/'training.pkl.gz';manifest=cache.parent/'manifest.json'
    cache.parent.mkdir(exist_ok=True,parents=True)
    if manifest.exists():
        m=e.read(manifest);assert m['protocol_fingerprint']==digest(p) and file_hash(cache)==m['sha256'];return cache,split
    data=MODData.load(source/'official-train-moddata.pkl.gz')
    pd.testing.assert_frame_equal(data.df_featurized,banks['mm'].loc[split['train_ids']],check_names=False)
    np.testing.assert_array_equal(data.df_targets.values,y.loc[split['train_ids']].values)
    if bank=='mm_mv':
        enhanced=e.make_data(banks[bank].loc[split['train_ids']],y.loc[split['train_ids']])
        if p['smoke']:
            cols=list(data.optimal_features[:16])+[c for c in banks[bank] if c not in banks['mm']][:8]
            enhanced.df_featurized=enhanced.df_featurized[cols]
        data=e.extend_ranking(enhanced,data,cache.parent/'feature-selection',p['seed']+1000000+fold*100000,
            n_jobs,lambda state,**kw:progress(state,bank=bank,fold=fold,**kw))
    if p['smoke']:data.optimal_features=data.optimal_features[:24]
    data.save(cache)
    write_json(manifest,dict(protocol_fingerprint=digest(p),sha256=file_hash(cache),
        train_ids=list(data.df_targets.index),features=data.optimal_features,bank=bank,fold=fold,outer_test_ids=split['test_ids']))
    return cache,split


e.prepare=prepare


def freeze(out,task,csv,smoke):
    base=e.read(ROOT/'benchmarks/native-fourway-scaled-kan-20261029/protocol.json')
    sources=dict(base['source_sha256'])
    for name in ['run_native_matbench_expansion.py','relaxed_spline_policy.py','scaled_native_spline.py']:
        f=ROOT/'scripts'/name;sources[str(f)]=file_hash(f)
    original=OLD/'precomputed'/f'{task}_moddata.pkl.gz'
    p=dict(task=task,protocol_path=str(out/'protocol.json'),source_sha256=sources,
        original=str(original),csv=str(csv),input_sha256={str(original):file_hash(original),str(csv):file_hash(csv)},
        seed=20261021+SUPPORTED[task]*1000000-1000000,
        arms=e.ARMS,folds=[0] if smoke else list(range(5)),smoke=smoke,ga=base['ga'],
        fits_per_arm_fold=8 if smoke else 1000,training=base['training'],
        published_mae=e.read(OLD/task/'published-reference-verified.json')['mean_mae'],
        kan='Inherited target-standardized near-linear 3-layer B-spline with closest integer parameter mapping.',
        predictions='Raw ensemble mean primary; native outlier-repaired MAE only baseline alignment diagnostic.',
        baseline_gate='Before KAN, five-fold native repaired mean <= published MODNet mean*1.02; stop for investigation if failed.',
        limitations=[base['limitations'][0],base['limitations'][2],base['limitations'][3],base['limitations'][4],
          'KAN design and target standardization are inherited from phonons; no tuning on this task outer-test scores.',
          'KAN uses train-only target standardization while native MODNet uses raw targets; architecture and scaling effects are not fully separated.',
          'Fixed 5000 fits per arm extends early-stopped native GA. Identical old baseline fits may be reused with source hashes.'])
    for fold in p['folds']:
        for name in ['outer-split.json','official-train-moddata.pkl.gz']:
            f=OLD/task/f'fold-{fold}'/name;p['input_sha256'][str(f)]=file_hash(f)
    if (out/'protocol.json').exists():assert e.read(out/'protocol.json')==__import__('json').loads(__import__('json').dumps(p,default=e.native.jsonable))
    write_json(out/'protocol.json',p)
    for f,h in sources.items():
        dest=out/'source-snapshot'/(Path(f).parent.name+'--'+Path(f).name);dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copyfile(f,dest)
        assert file_hash(dest)==h
    return p


def audit(out,p):
    rows=[];partitions={};initial={}
    for f,h in {**p['source_sha256'],**p['input_sha256']}.items():assert file_hash(f)==h
    for arm in e.ARMS:
        for fold in p['folds']:
            d=out/arm/f'fold-{fold}';r=e.read(d/'complete.json');split=e.read(OLD/p['task']/f'fold-{fold}'/'outer-split.json')
            assert r['protocol_fingerprint']==digest(p)
            assert file_hash(d/'outer-predictions.npz')==r['prediction_sha256']
            assert file_hash(d/'frozen-selection.json')==r['selection_sha256']
            records=[e.read(f) for f in (d/'fits').glob('*/result.json')]
            assert len(records)==p['fits_per_arm_fold']==r['search_fits']
            pop,gens,nested=(2,2,2) if p['smoke'] else (20,10,5)
            assert {(x['generation'],x['candidate'],x['inner']) for x in records}=={(g,c,i) for g in range(gens) for c in range(pop) for i in range(nested)}
            manifest=e.read(out/'inputs'/arm.rsplit('_',1)[0]/f'fold-{fold}'/'manifest.json')
            cache=out/'inputs'/arm.rsplit('_',1)[0]/f'fold-{fold}'/'training.pkl.gz'
            assert file_hash(cache)==manifest['sha256']
            targets=MODData.load(cache).df_targets if arm.endswith('kan') else None
            groups={}
            for m in records:
                assert file_hash(Path(m['directory'])/'model.pkl')==m['model_sha256']
                assert set(m['train_ids']).isdisjoint(m['validation_ids'])
                assert set(m['train_ids']+m['validation_ids'])==set(split['train_ids'])
                assert set(split['train_ids']).isdisjoint(split['test_ids'])
                assert m['selected_features']==manifest['features'][:m['genes']['n_feat']]
                assert not any('SISSO' in c.upper() for c in m['features'])
                key=(fold,m['inner']);partition=(m['train_ids'],m['validation_ids'])
                if key in partitions:assert partitions[key]==partition
                else:partitions[key]=partition
                if m['generation']==0:
                    key=(arm.rsplit('_',1)[0],fold,m['candidate'],m['inner'])
                    if key in initial:assert initial[key]==m['genes']
                    else:initial[key]=m['genes']
                if 'reused_from' in m:
                    assert file_hash(Path(m['reused_from'])/'result.json')==m['reused_result_sha256']
                    assert m['model_sha256']==m['reused_model_sha256']
                if arm.endswith('kan'):
                    scale=m['target_scaling'];values=targets.loc[m['train_ids']].values
                    assert scale['fit_ids']==m['train_ids'] and scale['mean']==float(values.mean()) and scale['std']==float(values.std())
                    mapping=policy.describe(m['genes']['n_feat'],tuple(x[0] for x in e.neurons_from_genes(m['genes'])))
                    assert m['parameter_mapping']==mapping and m['parameters']==mapping['spline_parameters']
                groups.setdefault((m['generation'],m['candidate']),[]).append(m)
            groups=[sorted(groups[k],key=lambda x:x['inner']) for k in sorted(groups)]
            order=np.argsort([np.mean([x['validation_mae'] for x in group]) for group in groups])[:10]
            expected=[x['directory'] for i in order for x in groups[i]]
            assert expected==[x['directory'] for x in e.read(d/'frozen-selection.json')['members']]
            with np.load(d/'outer-predictions.npz') as z:
                assert list(z['sample_ids'])==split['official_test_ids']
                np.testing.assert_array_equal(z['prediction'],z['members'].mean(axis=0))
                assert abs(float(np.mean(abs(z['prediction']-z['target'])))-r['mae'])<1e-10
            rows.append(dict(arm=arm,fold=fold,mae=r['mae'],fits=len(records),reused_fits=sum('reused_from' in m for m in records)))
    write_json(out/'artifact-audit.json',dict(passed=True,complete=True,smoke=p['smoke'],rows=rows))


def report(out,p):
    summary=e.report(out,p)
    summary['task']=p['task'];summary['metric']='mae';write_json(out/'analysis.json',summary)
    path=out/'report.md'
    path.write_text(path.read_text(encoding='utf-8').replace('four-way phonons comparison',f"four-way {p['task']} comparison"),encoding='utf-8')
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=SUPPORTED,required=True)
    ap.add_argument('--csv',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--n-jobs',type=int,default=3);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    def progress(state,**kw):write_json(out/'status.json',dict(state=state,pid=os.getpid(),task=args.task,updated_at=e.now(),wall_seconds=time.monotonic()-start,**kw))
    try:
        progress('freezing');p=freeze(out,args.task,args.csv.resolve(),args.smoke)
        progress('loading_data');banks,y=load_banks(p,out)
        for arm in e.ARMS:
            for fold in p['folds']:
                e.run_fold(out,p,banks,y,arm,fold,args.n_jobs,progress);summary=report(out,p)
            if arm=='mm_modnet' and not args.smoke:
                repaired=float(np.mean([x['native_repaired_mae'] for x in summary['rows'] if x['arm']==arm]))
                passed=repaired<=p['published_mae']*1.02
                write_json(out/'baseline-quality-gate.json',dict(passed=passed,raw_mae=summary['means'][arm],native_repaired_mae=repaired,published_mae=p['published_mae']))
                if not passed:progress('baseline_quality_requires_investigation');return
        progress('auditing');audit(out,p);progress('complete',complete=True,smoke=args.smoke)
    except BaseException:progress('failed',traceback=traceback.format_exc());raise

if __name__=='__main__':main()
