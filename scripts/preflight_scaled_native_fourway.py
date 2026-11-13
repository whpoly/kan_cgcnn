"""Replay a diagnosed inner fit through the corrected formal worker and pickle."""
import run_scaled_native_fourway as corrected
from run_native_fourway import ROOT, read, file_hash, digest, write_json, MODData, np, pickle
from diagnose_native_spline_scale import prediction


def main():
    out=ROOT/'benchmarks/scaled-native-preflight-20261029';out.mkdir(parents=True,exist_ok=True)
    p=corrected.freeze(out,False)
    diagnosis=ROOT/'benchmarks/native-spline-scale-diagnosis-20261029'
    dp=read(diagnosis/'protocol.json')
    source=read(dp['source_members'][0]['path'])
    reference=read(diagnosis/'standardized/inner-0/result.json')
    cache=dp['cache'];sha=file_hash(cache);fp=digest(p)
    job=dict(directory=str(out/'fit'),genes=source['genes'],seed=source['seed'],
        arm='mm_kan',variant='bspline',fold=0,generation=source['generation'],candidate=source['candidate'],inner=0,
        train_ids=source['train_ids'],validation_ids=source['validation_ids'],smoke=False,
        training_cache=cache,training_sha256=sha,protocol_path=p['protocol_path'],
        protocol_fingerprint=fp,run_fingerprint=digest(dict(protocol=fp,training_data=sha)))
    r=corrected.run_fit(job)
    assert abs(r['validation_mae']-reference['validation_mae'])<.001
    assert r['epochs']==reference['epochs']
    data=MODData.load(cache);lookup={sid:i for i,sid in enumerate(data.df_targets.index)}
    train,val=data.split(([lookup[s] for s in job['train_ids']],[lookup[s] for s in job['validation_ids']]))
    with (out/'fit/model.pkl').open('rb') as handle:model=pickle.load(handle)
    model._restore_model()
    pred,_=prediction(model,val)
    with np.load(diagnosis/'standardized/inner-0/validation.npz') as z:
        np.testing.assert_allclose(pred,z['prediction'],atol=.001,rtol=0)
    np.testing.assert_array_equal(model.min_y,train.df_targets.values.min(axis=0))
    np.testing.assert_array_equal(model.max_y,train.df_targets.values.max(axis=0))
    assert model.target_scaling['fit_ids']==job['train_ids']
    result=dict(passed=True,raw_mae=source['validation_mae'],diagnostic_scaled_mae=reference['validation_mae'],
        corrected_formal_mae=r['validation_mae'],epochs=r['epochs'],
        checks=['Inner-only target statistics','Physical early-stopping threshold',
                'Identical seeded training replay','Pickle restores physical predictions','Raw physical bounds preserved'])
    write_json(out/'preflight.json',result);print(result)


if __name__=='__main__':main()
