"""Trace complete trained KAN/MLP differences on pairs of archived phonon crystals."""
import argparse
import os

for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[variable]='1'

from audit_steel_all_input_paths import member, obs
from audit_steel_observed_pairs import exp, file_hash, np, tf, write_json
from prepare_phonons_physical_pairs import OUT, read


def run(folds):
    catalogue=read(OUT/'catalogue.json')
    data=exp.load_data('phonons',OUT/f'data-check-{folds[0]}')
    frame,_,_=data
    for f in folds:
        sources,refs,official,held=exp.load_fold('phonons',f,data)
        pairs=[r for r in catalogue['pairs'] if r['fold']==f]
        test=frame.loc[held]
        indices=np.array([[test.index.get_loc(p['lo']),test.index.get_loc(p['hi'])] for p in pairs])
        folder=OUT/f'fold-{f}'; folder.mkdir(parents=True,exist_ok=True)
        info={}
        for arm,models,raw,ensemble in zip(exp.ARMS,sources,refs,official):
            # Retain all signed descriptor contributions, even when selected input sets differ.
            all_terms=np.zeros((len(models),len(pairs),len(frame.columns)),dtype='float64')
            predictions=[]; errors=[]; selected=[]
            for i,source in enumerate(models):
                terms,names,pred,replay,closure=member(source,test,indices,np.arange(len(test)),raw[i],arm)
                cols=frame.columns.get_indexer(names)
                assert min(cols)>=0 and len(set(cols))==len(cols)
                all_terms[i][:,cols]=terms
                predictions.append(pred); errors.append((replay,closure)); selected.append(cols.tolist())
                tf.keras.backend.clear_session()
            true_delta=ensemble[:,indices[:,1]].astype(float)-ensemble[:,indices[:,0]].astype(float)
            archive=folder/(arm+'.npz')
            ensemble_closure=float(np.max(abs(all_terms.mean(0).sum(axis=1)-true_delta.mean(0))))
            np.testing.assert_allclose(all_terms.mean(0).sum(axis=1),true_delta.mean(0),atol=.012,rtol=1e-6)
            np.savez_compressed(archive,terms=all_terms,member_predictions=predictions,archived_members=ensemble)
            info[arm]=dict(sha256=file_hash(archive),selected=selected,models=[s[2] for s in models],
                replay_max=float(np.max(errors,axis=0)[0]),member_closure_max=float(np.max(errors,axis=0)[1]),
                archive_ensemble_delta_max_error=ensemble_closure)
            print(f'fold {f} {arm}: traced {len(pairs)} differences, {len(models)} frozen models',flush=True)
        write_json(folder/'trace.json',dict(fold=f,pairs=pairs,held=held,columns=list(frame.columns),arms=info,
               pairs_sha256=file_hash(OUT/'catalogue.json'),code_sha256=file_hash(__file__),
               method_sha256=file_hash(obs.__file__),member_function_sha256=file_hash(obs.exp.ROOT/'scripts/audit_steel_all_input_paths.py')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds',nargs='+',type=int,default=list(range(5)))
    run(parser.parse_args().folds)
