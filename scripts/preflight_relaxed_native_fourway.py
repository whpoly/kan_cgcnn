"""Check legacy mapping compatibility and execute the formerly failing candidate."""
import run_native_fourway as engine
import native_fourway_models as models
legacy_mapping = models.matched_widths
import resume_relaxed_native_fourway as resumed
from relaxed_spline_policy import describe, mapping, dense_count, spline_count
from run_native_fourway import ROOT, Path, read, file_hash, digest, write_json, np, pickle, MODData
from diagnose_native_spline_scale import prediction
from itertools import product


def main():
    root=ROOT/'benchmarks/native-fourway-scaled-kan-20261029'
    out=ROOT/'benchmarks/relaxed-native-preflight-20261030';out.mkdir(parents=True,exist_ok=True)
    p=resumed.scaled.freeze(out,False);resumed.freeze_amendment(out)
    # Network geometry is checked independently of any score or target value.
    configurations={(540,(32,32,16,4))}
    for f in (root/'mm_kan').glob('fold-*/fits/*/result.json'):
        r=read(f);w=tuple(x[0] for x in engine.neurons_from_genes(r['genes']))
        configurations.add((r['genes']['n_feat'],w))
    for n,first,f1,f2,f3 in product((1,10,100,540,963,2000,4455),(32,320,896),
                                  (1,.75,.5,.25),(1,.75,.5,.25),(1,.75,.5,.25)):
        configurations.add((n,(first,int(first*f1),int(first*f1*f2),int(first*f1*f2*f3))))
    unchanged=0;fallback=[]
    for n,w in sorted(configurations):
        d=describe(n,w)
        try:old=legacy_mapping(n,w)
        except ValueError:old=None
        if old:
            assert tuple(d['widths'][:2])==old and not d['fallback_used'];unchanged+=1
        else:
            assert d['fallback_used'];fallback.append(d['signed_relative_difference'])
            # Independently check nearest geometry over the legacy candidate set.
            target=dense_count(n,w)
            import math
            errors=[abs(spline_count(n,a,b)/target-1)
                for a in range(1,max(513,int(math.sqrt(target))+2))
                for b in range(max(1,round(a*8/17)-8),round(a*8/17)+9)]
            assert abs(d['signed_relative_difference'])==min(errors)
    write_json(out/'geometry-check.json',dict(passed=True,checked=len(configurations),
        unchanged_legacy_mappings=unchanged,fallbacks=len(fallback),
        fallback_signed_range=[min(fallback),max(fallback)]))
    print('GEOMETRY',len(configurations),unchanged,len(fallback),flush=True)
    # Derive the failed job's identities from its frozen sibling worker status.
    src=root/'mm_kan/fold-3/fits'
    records=[read(f) for f in src.glob('*/result.json')]
    template=next(r for r in records if r['inner']==0)
    cache=root/'inputs/mm/fold-3/training.pkl.gz';sha=file_hash(cache);fp=digest(p)
    genes=dict(template['genes']);genes.update(n_feat=540,n_neurons_first_layer=32,fraction1=1,
        fraction2=.5,fraction3=.25)
    job=dict(directory=str(out/'fit'),genes=genes,seed=p['seed']+1000000+300000+500,
        arm='mm_kan',variant='bspline',fold=3,generation=0,candidate=5,inner=0,
        train_ids=template['train_ids'],validation_ids=template['validation_ids'],smoke=True,
        training_cache=str(cache),training_sha256=sha,protocol_path=p['protocol_path'],
        protocol_fingerprint=fp,run_fingerprint=digest(dict(protocol=fp,training_data=sha)))
    result=resumed.run_fit(job)
    assert result['parameter_mapping']['fallback_used']
    assert result['parameters']==17695 and result['paired_modnet_parameters']==18969
    with (out/'fit/model.pkl').open('rb') as f:model=pickle.load(f)
    model._restore_model();data=MODData.load(cache);lookup={x:i for i,x in enumerate(data.df_targets.index)}
    tr,va=data.split(([lookup[x] for x in job['train_ids']],[lookup[x] for x in job['validation_ids']]))
    pred,_=prediction(model,va)
    assert np.isfinite(pred).all()
    assert abs(float(np.mean(abs(pred-va.df_targets.values[:,0])))-result['validation_mae'])<.001
    before=file_hash(out/'fit/model.pkl');again=resumed.run_fit(job)
    assert again==result and file_hash(out/'fit/model.pkl')==before
    write_json(out/'preflight.json',dict(passed=True,parameter_mapping=result['parameter_mapping'],
        checks=['Legacy geometry unchanged','Nearest fallback verified','Failing geometry trains',
                'Physical-unit serialization','Cached worker is reused without retraining'],
        accuracy_claim=False,epochs=1))
    print('PREFLIGHT PASSED',flush=True)


if __name__=='__main__':main()
