"""One original first-generation fit through the new adapter, no new tuning."""
from run_native_fourway import ROOT, OLD, read, freeze, file_hash, digest, run_fit, write_json


def main():
    out=ROOT/'benchmarks/native-fourway-preflight-20261029';out.mkdir(parents=True,exist_ok=True)
    p=freeze(out,False)
    name='generation-00-candidate-00-inner-0'
    source=read(OLD/'matbench_phonons/fold-0/modnet'/name/'result.json')
    cache=OLD/'matbench_phonons/fold-0/official-train-moddata.pkl.gz';sha=file_hash(cache);fp=digest(p)
    job=dict(directory=str(out/'mm_modnet/fold-0/fits'/name),genes=source['genes'],seed=source['seed'],
        arm='mm_modnet',variant='modnet',fold=0,generation=0,candidate=0,inner=0,
        train_ids=source['train_ids'],validation_ids=source['validation_ids'],smoke=False,
        training_cache=str(cache),training_sha256=sha,protocol_path=p['protocol_path'],
        protocol_fingerprint=fp,run_fingerprint=digest(dict(protocol=fp,training_data=sha)))
    result=run_fit(job)
    assert result['historical_prefix_check']['passed']
    summary=dict(passed=True,old_mae=source['validation_mae'],new_mae=result['validation_mae'],
        difference=result['validation_mae']-source['validation_mae'],old_epochs=source['epochs'],new_epochs=result['epochs'],
        purpose='Execution equivalence of original first GA candidate; not a hyperparameter search or four-arm result')
    write_json(out/'preflight.json',summary);print(summary)


if __name__=='__main__':main()
