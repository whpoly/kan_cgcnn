"""Test fixed explanation budgets on original crystal pairs and audit matched training configurations."""
import argparse
import os
import pickle
from pathlib import Path

for var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[var]='1'

import audit_steel_observed_pairs as obs
from audit_steel_all_input_paths import contribution
from native_fourway_features import load_banks
from prepare_phonons_physical_pairs import OUT as PRIOR, matched_pairs, read
from run_modnet_official_matched import file_hash, np, tf, write_json
from scipy.stats import rankdata


P=obs.exp.p
OUT=obs.exp.ROOT/'benchmarks/phonons-equal-budget-20261113'
ARMS=('kan','modnet')
BUDGETS=(1,3,5,10,'quarter')


def source(bank,arm,fold):
    folder=P.RUN/f'{bank}_{arm}'/f'fold-{fold}'
    complete=read(folder/'complete.json')
    for path,key in [('frozen-selection.json','selection_sha256'),('outer-predictions.npz','prediction_sha256')]:
        assert file_hash(folder/path)==complete[key]
    return folder,read(folder/'frozen-selection.json')


def forward(path,fit,frame,arm):
    assert file_hash(path)==fit['model_sha256']
    with path.open('rb') as stream: model=pickle.load(stream)
    model._restore_model(); names=list(model.optimal_descriptors[:model.n_feat]); assert names==fit['selected_features']
    x=P.scaled(model,frame[names].values).astype(float)
    layers=[l for l in model.model.layers if not isinstance(l,tf.keras.layers.InputLayer)]
    scale,offset=1.,0.; kan=arm=='kan'
    if kan:
        assert model.target_scaling['fit_ids']==fit['train_ids'] and isinstance(layers[-1],tf.keras.layers.Rescaling)
        layers=layers[:-1];scale,offset=model.target_scaling['std'],model.target_scaling['mean']
    states,_,pred=obs.forward(layers,x,kan,scale,offset)
    return names,states,pred,scale,x


def native_terms(states,x,indices,kan,scale):
    blocks=[[] for _ in states]
    for k in range(0,len(indices),36):
        lo,hi=indices[k:k+36].T; back=np.full((len(lo),1),scale)
        for layer in range(len(states)-1,0,-1):
            state=states[layer]
            if kan:
                back=np.einsum('pib,ijb,pj->pi',obs.spline_slopes(state,lo,hi),state[2],back,optimize=True)
                h=state[0]
            else:
                back=(back*obs.activation_slopes(state,lo,hi))@state[2].T
                h=states[layer-1][1]
            blocks[layer].append((h[hi]-h[lo])*back)
        blocks[0].append(contribution(states[0],x,lo,hi,back,kan))
    return [np.vstack(v) for v in blocks]


def greedy(terms,total,limit):
    residual=total.copy(); order=[]; norm=(terms**2).sum(0)
    for _ in range(limit):
        score=2*(residual@terms)-norm
        score[order]=-np.inf
        j=int(np.argmax(score)); order.append(j); residual-=terms[:,j]
    return order


def choose_layer(blocks,delta,fit_mask,nt):
    best={}
    for layer,v in enumerate(blocks):
        width=v.shape[1]; quarter=max(1,int(np.ceil(width/4)))
        order=greedy(v[fit_mask],delta[fit_mask],min(width,max(10,quarter)))
        path=np.cumsum(v[:,order],axis=1)
        for cap in BUDGETS:
            k=quarter if cap=='quarter' else cap
            if k>width:continue
            train_mse=float(np.mean((path[fit_mask,k-1]-delta[fit_mask])**2))
            if cap not in best or train_mse<best[cap][0]:
                best[cap]=(train_mse,path[nt:,k-1],dict(layer=layer,nodes=order[:k],width=width))
    assert len(best)==len(BUDGETS)
    return np.stack([best[k][1] for k in BUDGETS]),[best[k][2] for k in BUDGETS]


def design(records,f,split):
    train=[dict(r,fold=0) for r in records['rows'] if r['fold']!=f and r['prototype']]
    tr=[p for p in matched_pairs(train) if p['kind']=='substitution']
    for p in tr:p['fold']=f
    te=[p for p in records['pairs'] if p['fold']==f and p['kind']=='substitution']
    tr_ids=sorted({p[k] for p in tr for k in ('lo','hi')}); ids=tr_ids+split['test_ids']
    assert set(tr_ids)<=set(split['train_ids']) and not set(tr_ids).intersection(split['test_ids'])
    return tr,te,ids,np.array([[ids.index(p[k]) for k in ('lo','hi')] for p in tr+te])


def explain(bank,fold,frame,target,records):
    split=read(P.engine.OLD/'matbench_phonons'/f'fold-{fold}/outer-split.json')
    tr,te,ids,idx=design(records,fold,split); nt=len(tr); study=frame.loc[ids]; nheld=len(split['test_ids'])
    save=OUT/bank/f'fold-{fold}'; save.mkdir(parents=True,exist_ok=True)
    models=[]; provenance={}; cols=frame.columns
    for arm in ARMS:
        folder,selection=source(bank,arm,fold); all_terms=np.zeros((len(idx),len(cols))); native=[]; choices=[]; training=[]; validation=[]; deltas=[]; replays=[]; closure=[]
        with np.load(folder/'outer-predictions.npz') as z:
            refs=z['members'].astype(float); official=z['prediction'].astype(float)
            np.testing.assert_array_equal(z['sample_ids'],split['official_test_ids']); np.testing.assert_allclose(target.loc[split['test_ids']],z['target'],atol=1e-8,rtol=0)
            np.testing.assert_allclose(refs.mean(0),official,atol=.002)
        for i,member in enumerate(selection['members']):
            path=Path(member['directory']);fit=read(path/'result.json'); assert fit['model_sha256']==selection['model_sha256'][str(path)]
            names,states,pred,scale,x=forward(path/'model.pkl',fit,study,arm)
            np.testing.assert_allclose(pred[-nheld:],refs[i],atol=.02,rtol=1e-6);replays.append(max(abs(pred[-nheld:]-refs[i])))
            delta=pred[idx[:,1]]-pred[idx[:,0]]; blocks=native_terms(states,x,idx,arm=='kan',scale)
            errs=[max(abs(block.sum(1)-delta)) for block in blocks]; assert max(errs)<1e-5;closure.append(max(errs))
            ci=cols.get_indexer(names); assert min(ci)>=0
            all_terms[:,ci]+=blocks[0]/len(refs)
            fit_ids=set(fit['train_ids']); mask=np.array([j<nt and p['lo'] in fit_ids and p['hi'] in fit_ids for j,p in enumerate(tr+te)])
            assert mask.sum()>20; chosen,meta=choose_layer(blocks[1:],delta,mask,nt)
            native.append(chosen); choices.append(meta); training.append(fit['train_ids']); deltas.append(delta); validation.append(dict(model=fit['model_sha256'],fit_ids=len(fit_ids),training_pairs=int(mask.sum()),parameters=fit['parameters']))
            tf.keras.backend.clear_session()
        models.append(training); np.testing.assert_allclose(all_terms.sum(1),np.mean(deltas,axis=0),atol=1e-6)
        out=save/f'{arm}.npz'; np.savez_compressed(out,input_terms=all_terms,native=native,deltas=deltas,official=official,member_outer=refs)
        provenance[arm]=dict(choices=choices,validation=validation,replay_max=max(replays),closure_max=max(closure),sha256=file_hash(out),official_archive_sha256=file_hash(folder/'outer-predictions.npz'))
        print(bank,fold,arm,'all layers and inputs traced',flush=True)
    assert models[0]==models[1]
    write_json(save/'design.json',dict(bank=bank,fold=fold,held=split['test_ids'],columns=list(cols),train_pairs=tr,test_pairs=te,arms=provenance,
               budgets=BUDGETS,source_sha256=file_hash(__file__),catalogue_sha256=file_hash(PRIOR/'catalogue.json')))


def matched_config(bank,fold,frame,target):
    split=read(P.engine.OLD/'matbench_phonons'/f'fold-{fold}/outer-split.json'); held=split['test_ids']; test=frame.loc[held]
    roots=[]; genome=[]; scores=[]; published={}
    for arm in ARMS:
        origin,sel=source(bank,arm,fold); roots.append(Path(sel['members'][0]['directory']).parent.parent)
        gen=read(roots[-1]/'generation-00.json'); genome.append(gen['genes']); scores.append(gen['mean_validation_mae'])
        with np.load(origin/'outer-predictions.npz') as z:
            np.testing.assert_array_equal(z['sample_ids'],split['official_test_ids']);np.testing.assert_allclose(z['target'],target.loc[held],rtol=0,atol=1e-8)
            published[arm]={m['directory']:row for m,row in zip(sel['members'],z['members'])}
    assert genome[0]==genome[1] and len(genome[0])==20
    joint=np.lexsort((np.arange(20),rankdata(scores[0])+rankdata(scores[1])))
    pred=np.empty((2,20,5,len(held)));params=np.empty((2,20,5),int);direct_errors=[];manifest=[]
    for c,genes in enumerate(genome[0]):
        for m in range(5):
            both=[]; scalers=[]
            for a,(arm,root) in enumerate(zip(ARMS,roots)):
                path=root/'fits'/f'generation-00-candidate-{c:02d}-inner-{m}';fit=read(path/'result.json');both.append(fit);assert fit['genes']==genes
                assert file_hash(path/'model.pkl')==fit['model_sha256']
                with (path/'model.pkl').open('rb') as stream:model=pickle.load(stream)
                model._restore_model(); names=list(model.optimal_descriptors[:model.n_feat]); assert names==fit['selected_features']
                x=P.scaled(model,test[names].values); scalers.append(x)
                if arm=='kan':assert model.target_scaling['fit_ids']==fit['train_ids']
                y=np.asarray(model.model(x,training=False)).reshape(-1);assert np.isfinite(y).all();pred[a,c,m]=y
                if str(path) in published[arm]:
                    diff=np.abs(y-published[arm][str(path)]);np.testing.assert_allclose(diff,0,atol=.012);direct_errors.append(max(diff))
                params[a,c,m]=fit['parameters'];tf.keras.backend.clear_session()
            for key in ('train_ids','validation_ids','selected_features','seed'): assert both[0][key]==both[1][key]
            np.testing.assert_array_equal(*scalers); assert not set(held)&(set(both[0]['train_ids'])|set(both[0]['validation_ids']))
            manifest.append([fit['model_sha256'] for fit in both])
    dir_=OUT/bank/f'fold-{fold}';dir_.mkdir(parents=True,exist_ok=True);path=dir_/'common-genomes.npz'
    np.savez_compressed(path,pred=pred,parameters=params,joint_order=joint,scores=scores,target=target.loc[held].values,official_ids=split['official_test_ids'])
    write_json(dir_/'common-genomes.json',dict(sha256=file_hash(path),model_hashes_by_candidate_member=manifest,code_sha256=file_hash(__file__),
          matched_pairs=len(manifest),published_replays=len(direct_errors),published_max_error=max(direct_errors,default=0),genes=genome[0]))
    print(bank,fold,'all 100 identical-configuration pairs checked',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['native','matched']);parser.add_argument('--folds',nargs='+',type=int,required=True)
    args=parser.parse_args();records=read(PRIOR/'catalogue.json');banks,yt=load_banks(OUT/f'data-check-{args.mode}-{args.folds[0]}');target=yt.iloc[:,0]
    for f in args.folds:
        for b in ('mm','mm_mv'):
            if args.mode=='native':explain(b,f,banks[b],target,records)
            else:matched_config(b,f,banks[b],target)
