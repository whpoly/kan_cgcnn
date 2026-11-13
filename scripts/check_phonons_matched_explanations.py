"""Repeat the five-node physical explanation on shared, validation-ranked training configurations."""
import argparse
from pathlib import Path

from audit_phonons_explanation_budget import ARMS, OUT, P, PRIOR, design, file_hash, forward, greedy, load_banks, native_terms, np, read, source, tf, write_json
from distill_phonon_nodes_to_physics import coordinate_blocks, eval_equations, ridge


def five_or_fewer(blocks,delta,mask,cap):
    choices=[]
    for l,b in enumerate(blocks):
        if b.shape[1]<cap:continue
        order=greedy(b[mask],delta[mask],cap); terms=b[:,order]
        choices.append((np.mean((terms[mask].sum(1)-delta[mask])**2),terms,dict(layer=l,nodes=order,width=b.shape[1])))
    _,terms,description=min(choices,key=lambda c:c[0]);return terms,description


def two_variables(q,idx,mask,terms,names,nt):
    used=['shortest_distance','inverse_sqrt_lightest_mass']; coords=q[:,[names.index(n) for n in used]]
    blocks,specs=coordinate_blocks(coords,idx,mask);est,w=ridge(np.column_stack(blocks),terms,mask)
    formulas=[[] for _ in range(terms.shape[1])];pos=0
    for name,block,s in zip(used,blocks,specs):
        end=pos+block.shape[1]
        for j,curves in enumerate(formulas):curves.append(dict(name=name,lo=s['lo'],hi=s['hi'],knots=s['knots'],keep=s['keep'],coef=w[pos:end,j]/s['rms']))
        pos=end
    np.testing.assert_allclose(eval_equations(formulas,q,names,idx[nt:]),est[nt:],atol=1e-7,rtol=1e-8)
    return est[nt:].sum(1),formulas


def run(bank,f,frame,qfull,names,records):
    folder=OUT/bank/f'fold-{f}'; archive=folder/'common-genomes.npz'; metadata=read(folder/'common-genomes.json')
    assert file_hash(archive)==metadata['sha256']
    split=read(P.engine.OLD/'matbench_phonons'/f'fold-{f}/outer-split.json');train,test,ids,idx=design(records,f,split)
    with np.load(archive) as z:
        order=z['joint_order'][:10]; original=z['pred'];params=z['parameters']
    nt=len(train); nh=len(split['test_ids']); sub=frame.loc[ids];q=qfull[frame.index.get_indexer(ids)]
    delta_all=[];native_all=[];physical_all=[];models=[];caps={}
    for a,arm in enumerate(ARMS):
        _,sel=source(bank,arm,f);root=Path(sel['members'][0]['directory']).parent
        deltas=[];five_terms=[];formulas=[];descriptions=[]
        for cand in order:
            for m in range(5):
                path=root/f'generation-00-candidate-{cand:02d}-inner-{m}';fit=read(path/'result.json')
                assert fit['model_sha256']==metadata['model_hashes_by_candidate_member'][cand*5+m][a]
                _,states,pred,scale,x=forward(path/'model.pkl',fit,sub,arm)
                np.testing.assert_allclose(pred[-nh:],original[a,cand,m],atol=.015,rtol=1e-6)
                delta=pred[idx[:,1]]-pred[idx[:,0]]; blocks=native_terms(states,x,idx,arm=='kan',scale)
                for b in blocks:np.testing.assert_allclose(b.sum(1),delta,atol=1e-5,rtol=1e-9)
                fitids=set(fit['train_ids']);mask=np.array([j<nt and p['lo'] in fitids and p['hi'] in fitids for j,p in enumerate(train+test)])
                if a==0:caps[cand,m]=min(5,max(b.shape[1] for b in blocks[1:]))
                cap=caps[cand,m];assert max(b.shape[1] for b in blocks[1:])>=cap
                terms,c=five_or_fewer(blocks[1:],delta,mask,cap)
                pred_phys,eq=two_variables(q,idx,mask,terms,names,nt)
                deltas.append(delta[nt:]);five_terms.append(terms[nt:].sum(1));formulas.append(pred_phys);descriptions.append(dict(candidate=int(cand),inner=m,sha=fit['model_sha256'],nodes=c,equations=eq,parameters=int(params[a,cand,m])))
                tf.keras.backend.clear_session()
        delta_all.append(deltas);native_all.append(five_terms);physical_all.append(formulas);models.append(descriptions)
        print(bank,f,arm,'traced the shared top ten configurations',flush=True)
    path=folder/'common-physical.npz';np.savez_compressed(path,delta=delta_all,native=native_all,physical=physical_all,order=order)
    write_json(folder/'common-physical.json',dict(sha256=file_hash(path),common_sha256=file_hash(archive),code_sha256=file_hash(__file__),models=models,test_pairs=test))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folds',nargs='+',type=int,required=True);args=parser.parse_args()
    records=read(PRIOR/'catalogue.json');banks,_=load_banks(OUT/f'data-check-matched-explanation-{args.folds[0]}')
    with np.load(PRIOR/'readout.npz') as z:
        ids=z['ids'].tolist();q=z['q'];names=z['quantities'].tolist()
    for b in banks:assert list(banks[b].index)==ids
    for f in args.folds:
        for bank in ('mm','mm_mv'):run(bank,f,banks[bank],q,names,records)
