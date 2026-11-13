"""Represent the five fixed native terms by functions of physical coordinates on observed pairs."""
import argparse
from pathlib import Path

from audit_phonons_explanation_budget import ARMS, OUT, P, PRIOR, design, forward, load_banks, np, obs, read, source, tf, write_json, file_hash


def selected_terms(states,indices,choice,kan,scale):
    target_layer=choice['layer']+1
    rows=[]
    for start in range(0,len(indices),36):
        lo,hi=indices[start:start+36].T;back=np.full((len(lo),1),scale)
        for l in range(len(states)-1,target_layer-1,-1):
            layer=states[l]
            if kan:
                back=np.einsum('pib,ijb,pj->pi',obs.spline_slopes(layer,lo,hi),layer[2],back,optimize=True)
                h=layer[0]
            else:
                back=(back*obs.activation_slopes(layer,lo,hi))@layer[2].T; h=states[l-1][1]
        channels=choice['nodes']; assert len(channels)==5
        rows.append((h[hi][:,channels]-h[lo][:,channels])*back[:,channels])
    return np.vstack(rows)


def coordinate_blocks(q,indices,train):
    lo,hi=indices.T; fit=np.unique(indices[train]); blocks=[]; specs=[]
    for values in q.T:
        small,large=np.min(values[fit]),np.max(values[fit]); vals=np.clip(values,small,large)
        knots=np.unique(np.quantile(values[fit],[.25,.5,.75],method='nearest'));knots=knots[(knots>small)&(knots<large)]
        value=np.column_stack([vals,*[np.maximum(vals-t,0) for t in knots]])
        diff=value[hi]-value[lo]; rms=np.sqrt(np.mean(diff[train]**2,axis=0)); keep=rms>1e-8*max(1,abs(large),abs(small))
        blocks.append(diff[:,keep]/rms[keep]);specs.append(dict(lo=small,hi=large,knots=knots,keep=np.flatnonzero(keep),rms=rms[keep]))
    return blocks,specs


def ridge(x,terms,train):
    a=x[train];weights=np.linalg.solve(a.T@a+.005*len(a)*np.eye(a.shape[1]),a.T@terms[train])
    return x@weights,weights


def fit_curves(q,idx,train,target,names,nt):
    blocks,specs=coordinate_blocks(q,idx,train); measured=[names.index(n) for n in ('shortest_distance','inverse_sqrt_lightest_mass')]
    predictions={}; equations={}; selected={}; all_est=[]; weights=[]
    for b in blocks:
        if b.shape[1]:
            estimate,w=ridge(b,target,train)
        else:estimate=np.zeros_like(target);w=np.empty((0,target.shape[1]))
        all_est.append(estimate);weights.append(w)
    err=np.mean((np.array(all_est)[:,train]-target[train])**2,axis=1)
    err[[i for i,b in enumerate(blocks) if not b.shape[1]]]=np.inf
    def equation(i,w):
        s=specs[i]
        return dict(name=names[i],lo=s['lo'],hi=s['hi'],knots=s['knots'],keep=s['keep'],coef=w/s['rms'])
    for title,candidates in [('one_of_two',measured),('one_of_fourteen',list(range(len(names))))]:
        order=np.asarray(candidates)[np.argmin(err[candidates,:5],axis=0)]
        predictions[title]=np.column_stack([all_est[i][nt:,j] for j,i in enumerate(order)])
        equations[title]=[[equation(i,weights[i][:,j])] for j,i in enumerate(order)];selected[title]=[names[i] for i in order]
    x=np.column_stack([blocks[i] for i in measured]); est,w=ridge(x,target,train); predictions['both']=est[nt:,:5]; predictions['direct']=est[nt:,5:]
    par=[];pos=0
    for i in measured:
        end=pos+blocks[i].shape[1];par.append([equation(i,w[pos:end,j]) for j in range(target.shape[1])]);pos=end
    equations['both']=[[t[j] for t in par] for j in range(5)];equations['direct']=[[t[5] for t in par]]
    for mode,eq in equations.items():
        restored=eval_equations(eq,q,names,idx[nt:]);np.testing.assert_allclose(restored,predictions[mode],atol=1e-7,rtol=1e-9)
    return predictions,equations,selected


def eval_equations(eq,q,names,idx):
    vals=np.zeros((len(q),len(eq)));lo,hi=idx.T
    for j,curves in enumerate(eq):
        for spec in curves:
            v=np.clip(q[:,names.index(spec['name'])],spec['lo'],spec['hi'])
            basis=np.column_stack([v,*[np.maximum(v-t,0) for t in spec['knots']]])
            vals[:,j]+=basis[:,spec['keep']]@spec['coef']
    return vals[hi]-vals[lo]


def distill(bank,f,frame,data,q_full,names):
    folder=OUT/bank/f'fold-{f}';design_path=folder/'design.json';meta=read(design_path)
    split=read(P.engine.OLD/'matbench_phonons'/f'fold-{f}/outer-split.json');tr,te,ids,idx=design(data,f,split);nt=len(tr)
    assert meta['train_pairs']==tr and meta['test_pairs']==te
    q=q_full[frame.index.get_indexer(ids)];real=frame.loc[ids];k=meta['budgets'].index(5)
    for arm in ARMS:
        _,sel=source(bank,arm,f); models=[];outer={i:[] for i in ['one_of_two','one_of_fourteen','both','direct','native']};provenance=meta['arms'][arm]
        original=folder/(arm+'.npz'); assert file_hash(original)==provenance['sha256']
        with np.load(original) as z:
            native=z['native']; diffs=z['deltas']
        for i,m in enumerate(sel['members']):
            path=Path(m['directory']);fit=read(path/'result.json');_,states,pred,scale,_=forward(path/'model.pkl',fit,real,arm)
            assert fit['model_sha256']==provenance['validation'][i]['model']
            delta=pred[idx[:,1]]-pred[idx[:,0]];np.testing.assert_allclose(diffs[i],delta,atol=1e-7)
            five=selected_terms(states,idx,provenance['choices'][i][k],arm=='kan',scale)
            np.testing.assert_allclose(five[nt:].sum(1),native[i,k],atol=1e-7);outer['native'].append(five[nt:])
            fit_ids=set(fit['train_ids']);train=np.array([j<nt and p['lo'] in fit_ids and p['hi'] in fit_ids for j,p in enumerate(tr+te)])
            results,equations,chosen=fit_curves(q,idx,train,np.column_stack([five,delta]),names,nt)
            for mode,v in results.items():outer[mode].append(v)
            models.append(dict(member=i,model=fit['model_sha256'],selected=chosen,equations=equations));tf.keras.backend.clear_session()
        output=folder/(arm+'-physical.npz');np.savez_compressed(output,**outer)
        write_json(folder/(arm+'-physical.json'),dict(members=models,code_sha256=file_hash(__file__),native_design_sha256=file_hash(design_path),out_sha256=file_hash(output)))
        print(bank,f,arm,'exported five-node physical formulas',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folds',type=int,nargs='+',required=True);args=parser.parse_args()
    data=read(PRIOR/'catalogue.json');banks,_=load_banks(OUT/f'data-check-physical-{args.folds[0]}')
    with np.load(PRIOR/'readout.npz') as z:
        ids=z['ids'].tolist();q_full=z['q'];names=z['quantities'].tolist()
    for b in banks: assert banks[b].index.tolist()==ids
    for f in args.folds:
        for bank in ('mm','mm_mv'):distill(bank,f,banks[bank],data,q_full,names)
