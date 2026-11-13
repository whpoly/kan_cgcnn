"""Decode the global physical readout from equal numbers of real hidden activations."""
import argparse
import os
import pickle

for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[name]='1'
import numpy as np


def formula_terms(q,formula):
    parts=[]
    for x,var in zip(q.T,formula['variables']):
        v=np.clip(x,var['lo'],var['hi'])
        y=var['coef'][0]*v
        for b,t in zip(var['coef'][1:],var['knots']):
            y+=b*np.maximum(v-t,0)
        parts.append(y)
    return np.column_stack(parts)


def decode_channels(hidden,q,n):
    """Greedy forward linear regression. Both selection and calibration use the member's fit IDs."""
    x=hidden.astype(float)-hidden[:n].mean(axis=0,dtype=float)
    norm=np.sqrt((x[:n]**2).sum(0)); x/=np.where(norm>1e-10,norm,1.)
    mean=q[:n].mean(0); centered=q[:n]-mean
    decoded=np.empty((2,len(q)-n,q.shape[1])); ranks=[]
    for j in range(q.shape[1]):
        candidates=x[:n]; denom=(candidates*candidates).sum(0)
        orth=[]; chosen=[]; residual=centered[:,j].copy()
        for k in range(4):
            cross=candidates.T@residual
            score=np.divide(cross**2,denom,out=np.full(len(denom),-np.inf),where=denom>1e-8)
            if chosen: score[chosen]=-np.inf
            i=int(np.argmax(score)); assert np.isfinite(score[i]); chosen.append(i)
            v=candidates[:,i].copy()
            for u in orth: v-=u*(u@v)
            v/=np.linalg.norm(v); orth.append(v)
            denom-=np.square(v@candidates); residual-=v*(v@residual)
            if k in (0,3):
                coeff=np.linalg.lstsq(candidates[:,chosen],centered[:,j],rcond=None)[0]
                decoded[int(k==3),:,j]=mean[j]+x[n:,chosen]@coeff
        ranks.append(chosen)
    return decoded,ranks


def run(folds):
    import audit_steel_observed_pairs as obs
    from prepare_phonons_physical_pairs import EXTENDED, OUT, read
    from run_modnet_official_matched import file_hash, tf, write_json
    exp=obs.exp
    design=read(OUT/'readout.json')
    with np.load(OUT/'readout.npz') as z:
        ids=z['ids'].tolist(); cols=z['quantities'].tolist()
        q=z['q'][:,[cols.index(name) for name in EXTENDED]]
        from_quantities=z['readouts'][1,1].copy()
    data=exp.load_data('phonons',OUT/f'data-check-{folds[0]}'); frame=data[0]
    assert ids==frame.index.tolist()
    for f in folds:
        sources,refs,_,held=exp.load_fold('phonons',f,data)
        ho=frame.index.get_indexer(held); folder=OUT/f'fold-{f}'
        choice=design['folds'][f]['designs']['extended']; assert choice['chosen'][:2]==EXTENDED
        records={}
        for a,(arm,members,archived,formula) in enumerate(zip(exp.ARMS,sources,refs,choice['two_formulas'])):
            physical=formula['intercept']+formula_terms(q[ho],formula).sum(1)
            np.testing.assert_allclose(physical,from_quantities[ho,a],atol=1e-7)
            predicted=[]; physical_outputs=[]; controls=[]; selection=[]; errors=[]
            kan=arm=='kan'
            for i,(path,_,sha,train_ids) in enumerate(members):
                assert file_hash(path)==sha
                with path.open('rb') as fh: model=pickle.load(fh)
                model._restore_model(); n=len(train_ids)
                pos=frame.index.get_indexer(train_ids+held)
                names=model.optimal_descriptors[:model.n_feat]
                x=exp.p.scaled(model,frame.loc[train_ids+held,names].values).astype(float)
                layers=[l for l in model.model.layers if not isinstance(l,tf.keras.layers.InputLayer)]
                if kan:
                    assert isinstance(layers[-1],tf.keras.layers.Rescaling) and model.target_scaling['fit_ids']==train_ids
                    layers=layers[:-1]; scale=model.target_scaling['std']; offset=model.target_scaling['mean']
                else: scale,offset=1.,0.
                states,_,y=obs.forward(layers,x,kan,scale,offset)
                np.testing.assert_allclose(y[n:],archived[i],atol=.02,rtol=1e-6)
                errors.append(float(abs(y[n:]-archived[i]).max()))
                units=([s[0] for s in states[1:]] if kan else [s[1] for s in states[:-1]])
                widths=[h.shape[1] for h in units]; hidden=np.column_stack(units)
                del states,units,x
                decoded,chosen=decode_channels(hidden,q[pos],n); predicted.append(decoded)
                physical_outputs.append([formula['intercept']+formula_terms(v,formula).sum(1) for v in decoded])
                yy=y[:n]-y[:n].mean(); qq=q[pos][:n]; slopes=(yy@(qq-qq.mean(0)))/(yy@yy)
                controls.append(qq.mean(0)+(y[n:]-y[:n].mean())[:,None]*slopes)
                selection.append(dict(widths=widths,nodes=chosen))
                tf.keras.backend.clear_session()
            archive=folder/f'{arm}-channels.npz'
            np.savez_compressed(archive,decoded=predicted,physical_outputs=physical_outputs,output_only_decoded=controls)
            records[arm]=dict(sha256=file_hash(archive),replay_max=max(errors),selection=selection)
            print(f'fold {f} {arm} decoded both physical quantities from hidden activations',flush=True)
        write_json(folder/'channels.json',dict(fold=f,held=held,arms=records,quantities=EXTENDED,
                   code_sha256=file_hash(__file__),readout_sha256=file_hash(OUT/'readout.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds',nargs='+',type=int,default=list(range(5)))
    run(parser.parse_args().folds)
