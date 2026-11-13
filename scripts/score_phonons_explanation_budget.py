"""Score train-selected native explanations and matched predictions on the official phonon folds."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/phonons-equal-budget-20261113'
PRIOR=ROOT/'benchmarks/phonons-all-input-paths-20261113'
ARMS=('kan','modnet')
CAPS=(1,3,5,10)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y,p):
    error=np.asarray(p)-y
    return dict(mae=np.mean(abs(error)),rmse=np.sqrt(np.mean(error**2)),r2=1-(error**2).sum()/((y-y.mean())**2).sum(),
                relative_to_zero=1-(error**2).sum()/(y**2).sum(),correct_direction=int(np.sum(y*p>0)),within_half=int(np.sum(abs(error)<=.5*abs(y))))


def paired_scores(y,p):
    errs=np.abs(p-y[:,None]); scores={a:metrics(y,p[:,i]) for i,a in enumerate(ARMS)}
    scores['kan_lower_absolute_error']=int(sum(errs[:,0]<errs[:,1]));scores['n']=len(y)
    scores['kan_mae_reduction_percent']=100*(1-errs[:,0].mean()/errs[:,1].mean());return scores


def named_families(cols):
    # Split element properties by the underlying named attribute, never count an anonymous vector as a physical term.
    names=[]; indices=[]
    for i,name in enumerate(cols):
        family,sep,label=name.partition('|')
        if not sep:continue
        group='ElementProperty/'+label.split()[-1] if family=='ElementProperty' else family
        names.append(group); indices.append(i)
    return names,np.array(indices)


def select(v,y,ntrain,names,shared):
    train=v[:,:ntrain]; targets=y[:,:ntrain]; valid=np.linalg.norm(train,axis=1)>1e-12
    units=1 if shared else len(v); scores=[]; selected=[]
    for i in range(units):
        arms=list(range(len(v))) if shared else [i]
        mats=train[arms]; dy=targets[arms]; scale=np.sqrt(np.mean(dy**2,axis=1)); mats=mats/scale[:,None,None]; r=dy/scale[:,None]
        norm=np.sum(mats**2,axis=(0,1)); eligible=np.all(valid[arms],axis=0); assert sum(eligible)>=max(CAPS)
        order=[]
        for _ in range(max(CAPS)):
            gain=2*np.einsum('ap,apj->j',r,mats)-norm;gain[~eligible]=-np.inf;gain[order]=-np.inf
            j=int(np.argmax(gain));order.append(j);r-=mats[:,:,j]
        selected.append([names[j] for j in order])
        scores.append(np.cumsum(v[arms,ntrain:][:,:,order],axis=2)[:,:,np.array(CAPS)-1])
    return np.concatenate(scores,axis=0),selected


def physical_formulas(folder,arm,meta,qty,qlo,qhi):
    curves=read(folder/(arm+'-physical.json'));file=folder/(arm+'-physical.npz')
    assert curves['out_sha256']==digest(file) and curves['native_design_sha256']==digest(folder/'design.json')
    modes=['one_of_two','one_of_fourteen','both','direct']; h=len(qlo)
    explained={mode:np.zeros((h,len(qty))) for mode in modes};counts={mode:Counter() for mode in modes[:2]}
    combined=defaultdict(lambda:defaultdict(float)); nmodels=len(curves['members'])
    with np.load(file) as z:
        raw={k:z[k] for k in modes}; native=z['native']
    for i,m in enumerate(curves['members']):
        assert m['model']==meta['validation'][i]['model']
        for mode in modes:
            table=np.zeros((h,len(qty))); eq=m['equations'][mode];check=np.zeros((h,len(eq)))
            for j,node in enumerate(eq):
                for curve in node:
                    k=qty.index(curve['name']);a=np.clip(qlo[:,k],curve['lo'],curve['hi']);b=np.clip(qhi[:,k],curve['lo'],curve['hi'])
                    steps=np.column_stack([b-a,*[np.maximum(b-t,0)-np.maximum(a-t,0) for t in curve['knots']]])
                    v=steps[:,curve['keep']]@curve['coef']; table[:,k]+=v;check[:,j]+=v
                    if mode=='both':
                        for basis,coef in zip(curve['keep'],curve['coef']):
                            threshold=curve['lo'] if basis==0 else curve['knots'][basis-1]
                            combined[curve['name']][threshold]+=coef/nmodels;combined[curve['name']][curve['hi']]-=coef/nmodels
            np.testing.assert_allclose(check,raw[mode][i],atol=1e-7,rtol=1e-9);explained[mode]+=table/len(curves['members'])
            if mode in counts:counts[mode].update(m['selected'][mode])
    detail={mode:dict(sse=float(np.square(raw[mode]-native).sum()),signal=float(np.square(native).sum())) for mode in modes[:3]}
    compact={}
    for name,c in combined.items():
        k=qty.index(name);items=sorted((t,a) for t,a in c.items() if a!=0.);knots,coef=map(np.array,zip(*items))
        step=np.maximum(qhi[:,k,None]-knots,0)-np.maximum(qlo[:,k,None]-knots,0)
        np.testing.assert_allclose(step@coef,explained['both'][:,k],atol=1e-7,rtol=1e-9)
        compact[name]=dict(knots=knots,coefficients=coef)
    return explained,dict(selected={k:dict(v) for k,v in counts.items()},term_errors=detail,two_curves=compact)


def read_native(bank,f,rows):
    folder=OUT/bank/f'fold-{f}'; d=read(folder/'design.json'); nt=len(d['train_pairs']);pairs=d['test_pairs']; cols=d['columns']; held=d['held']
    assert d['catalogue_sha256']==digest(PRIOR/'catalogue.json')
    arrays=[]; files=[];node=[];delta=[]; select_layers={}; all_pred=[]
    for a in ARMS:
        meta=d['arms'][a]; file=folder/(a+'.npz'); assert digest(file)==meta['sha256']; files.append(meta['sha256'])
        with np.load(file) as z:
            v=z['input_terms'];y=z['deltas'].mean(0);np.testing.assert_allclose(v.sum(1),y,atol=1e-5)
            arrays.append(v);delta.append(y);node.append(z['native'].mean(0).T);pred=z['official'];all_pred.append(pred)
            ii=np.array([[held.index(p['lo']),held.index(p['hi'])] for p in pairs]);np.testing.assert_allclose(pred[ii[:,1]]-pred[ii[:,0]],y[nt:],atol=.002)
        select_layers[a]=[dict(layers=dict(Counter(c[i]['layer'] for c in meta['choices'])),median_chosen=np.median([len(c[i]['nodes']) for c in meta['choices']]),
                              median_width=np.median([c[i]['width'] for c in meta['choices']])) for i in range(len(d['budgets']))]
    y=np.array(delta); v=np.array(arrays); actual=np.array([rows[p['hi']]['target']-rows[p['lo']]['target'] for p in pairs]); names,ix=named_families(cols)
    family_names=sorted(set(names)); groups=np.array(names); group_inputs={g:np.flatnonzero(groups==g) for g in family_names}; physical=v[:,:,ix]
    families=np.stack([physical[:,:,group_inputs[g]].sum(2) for g in family_names],axis=2);native=np.array(node)
    results={'full':y[:,nt:].T}; orders={};
    for k,cap in enumerate(d['budgets']):results['nodes_'+str(cap)]=native[:,:,k].T
    for kind,values,labels in [('families',families,family_names),('scalars',physical,[cols[k] for k in ix])]:
        for common in (True,False):
            key=kind+('_shared' if common else '_own'); pred,orders[key]=select(values,y,nt,labels,common)
            for k,cap in enumerate(CAPS):results[key+'_'+str(cap)]=pred[:,:,k].T
    family_counts={g:len(z) for g,z in group_inputs.items()}; inventory={g:int(sum(np.any(abs(v[:, :nt,ix[z]])>1e-12,axis=(0,1)))) for g,z in group_inputs.items()}
    material=np.array([rows[h]['target'] for h in held]);single=paired_scores(material,np.stack(all_pred,axis=1))
    with np.load(PRIOR/'readout.npz') as z:
        ids=z['ids'].tolist(); qty=z['quantities'].tolist(); q=z['q'];qlo=q[[ids.index(p['lo']) for p in pairs]]; qhi=q[[ids.index(p['hi']) for p in pairs]]
    by_qty=[];calibration={}
    for i,a in enumerate(ARMS):
        values,calibration[a]=physical_formulas(folder,a,d['arms'][a],qty,qlo,qhi);by_qty.append(values)
    for k in by_qty[0]:results['formula_'+k]=np.stack([p[k].sum(1) for p in by_qty],axis=1)
    ledger=[]
    for i,p in enumerate(pairs):
        rec=dict(lo=rows[p['lo']]['official'],hi=rows[p['hi']]['official'],formula=[rows[p['lo']]['formula'],rows[p['hi']]['formula']],fold=f,
                 prototype=p['prototype'],dfpt=actual[i],predictions={k:v[i].tolist() for k,v in results.items()})
        rec['five_shared_family_terms']={a:{g:families[j,nt+i,family_names.index(g)] for g in orders['families_shared'][0][:5]} for j,a in enumerate(ARMS)}
        rec['physical_equation_terms']={mode:{a:{name:by_qty[j][mode][i,k] for k,name in enumerate(qty) if np.any(by_qty[j][mode][:,k]!=0)} for j,a in enumerate(ARMS)} for mode in by_qty[0]}
        ledger.append(rec)
    return dict(fold=f,n_train=nt,n_test=len(pairs),selected=orders,layers_by_budget=select_layers,
          family_column_counts=family_counts,family_varying_columns=inventory,matrices=files,physical_calibration=calibration,material_mae=single),ledger,material,np.stack(all_pred,axis=1)


def matched(bank,f):
    folder=OUT/bank/f'fold-{f}'; d=read(folder/'common-genomes.json');path=folder/'common-genomes.npz';assert d['sha256']==digest(path)
    with np.load(path) as z:
        out=z['pred'].mean(2);n=z['parameters'];y=z['target']; order=z['joint_order'];scores=z['scores']
    assert out.shape==(2,20,len(y)) and n.shape==(2,20,5)
    predictions={k:out[:,order[:lim]].mean(1).T for k,lim in [('joint_best',1),('joint_top10',10),('all20',20)]}
    predictions['arm_own_top10']=np.stack([out[i,np.argsort(scores[i],kind='stable')[:10]].mean(0) for i in range(2)],axis=1)
    errors=np.abs(out-y).mean(2); ratio=n[0]/n[1]-1
    meta=dict(pairs=d['matched_pairs'],all_candidates_kan_lower=int(sum(errors[0]<errors[1])),
              max_relative_parameter_difference=np.max(abs(ratio)),published_overlap=d['published_replays'],joint_order=order,
              paired_candidate_maes=errors,parameter_ratio_range=[np.min(ratio)+1,np.max(ratio)+1],
              forecasts={k:paired_scores(y,v) for k,v in predictions.items()})
    return meta,predictions


def matched_explanations(bank,pairs):
    comparison={}
    for label,n in [('joint_best',5),('joint_top10',50)]:
        material=[]
        for f in range(5):
            folder=OUT/bank/f'fold-{f}';meta=read(folder/'common-physical.json');archive=folder/'common-physical.npz'
            assert meta['sha256']==digest(archive) and meta['common_sha256']==digest(folder/'common-genomes.npz')
            held=[p for p in pairs if p['fold']==f];assert len(held)==len(meta['test_pairs'])
            tr=read(folder/'design.json');assert tr['test_pairs']==meta['test_pairs']
            with np.load(archive) as z:
                values={k:z[key][:,:n].mean(1).T for k,key in [('full','delta'),('nodes_5','native'),('formula_both','physical')]}
            for i,row in enumerate(held):material.append(dict(dfpt=row['dfpt'],fold=f,predictions={k:v[i] for k,v in values.items()}))
        comparison[label]=compare_pairs(material)
    return comparison


def compare_pairs(rows):
    y=np.array([r['dfpt'] for r in rows]); keys=rows[0]['predictions'];teacher=np.array([r['predictions']['full'] for r in rows]);result={}
    for k in keys:
        pred=np.array([r['predictions'][k] for r in rows]); dfpt=paired_scores(y,pred)
        fidelity={a:metrics(teacher[:,i],pred[:,i]) for i,a in enumerate(ARMS)}
        per=[]
        for f in sorted({r['fold'] for r in rows}):
            sel=np.array([r['fold']==f for r in rows]);per.append(dict(fold=f,n=int(sum(sel)),dfpt=paired_scores(y[sel],pred[sel]),
                       fidelity={a:metrics(teacher[sel,i],pred[sel,i]) for i,a in enumerate(ARMS)}))
        result[k]=dict(dfpt=dfpt,fidelity=fidelity,by_fold=per,folds_kan_lower_dfpt_mae=sum(p['dfpt']['kan']['mae']<p['dfpt']['modnet']['mae'] for p in per),
               folds_kan_lower_fidelity_mae=sum(p['fidelity']['kan']['mae']<p['fidelity']['modnet']['mae'] for p in per))
    return result


def analyse():
    catalogue=read(PRIOR/'catalogue.json');rows={r['id']:r for r in catalogue['rows']}; report={}
    for bank in ('mm','mm_mv'):
        folds=[];material=[];official=[];matches=[];ledger=[];forecasts=[]
        for f in range(5):
            meta,records,y,p=read_native(bank,f,rows);folds.append(meta);ledger+=records; material.append(y);official.append(p)
            m,v=matched(bank,f);matches.append(m);forecasts.append(v)
        target=np.concatenate(material); assert len(target)==1265 and len(ledger)==58; used=set(); disjoint=[]
        for record in ledger:
            pair={record['lo'],record['hi']}
            if not pair.intersection(used):used.update(pair);disjoint.append(record)
        assert len(disjoint)==26
        pooled=paired_scores(target,np.concatenate(official)); pooled['folds_kan_lower_mae']=sum(v['material_mae']['kan']['mae']<v['material_mae']['modnet']['mae'] for v in folds)
        equality={k:paired_scores(target,np.concatenate([v[k] for v in forecasts])) for k in forecasts[0]}
        for k,v in equality.items():v['folds_kan_lower_mae']=sum(m['forecasts'][k]['kan']['mae']<m['forecasts'][k]['modnet']['mae'] for m in matches)
        report[bank]=dict(official_all_crystals=pooled,folds=folds,substitutions=compare_pairs(ledger),disjoint_substitutions=compare_pairs(disjoint),
                  pairs=ledger,identical_configuration_controls=dict(pooled=equality,folds=matches,five_node_explanations=matched_explanations(bank,ledger)))
    return dict(banks=report,arms=ARMS,physical_groups='Existing Matminer family; ElementProperty split by the individual Magpie property. All learned embedding columns excluded.',
                input_source=digest(PRIOR/'catalogue.json'),archive_code=digest(ROOT/'scripts/audit_phonons_explanation_budget.py'),report_code=digest(Path(__file__)))


def plot(report):
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#aab7bc','text.color':'#223744','axes.labelcolor':'#223744'})
    fig,axes=plt.subplots(1,2,figsize=(10.9,4.4));fig.subplots_adjust(left=.09,right=.97,bottom=.18,top=.78,wspace=.34);colors=['#17696b','#ac5334']
    for k,bank in enumerate(('mm','mm_mv')):
        s=report['banks'][bank]['substitutions']; ax=axes[k]
        for i,a in enumerate(ARMS):ax.plot(CAPS,[s['nodes_'+str(cap)]['fidelity'][a]['mae'] for cap in CAPS],'-o',lw=2,color=colors[i],label=['KAN','MLP'][i])
        ax.set(title=['Matminer 描述符','Matminer + MatterVial'][k],xticks=CAPS,xlabel='每个成员保留的原生节点数',ylabel='还原整网峰差的平均绝对误差（cm⁻¹）');ax.grid(axis='y',alpha=.16)
        ax.legend(frameon=False)
    fig.suptitle('58 对真实晶体替换，所有隐藏层均可参与选择',fontsize=15,fontweight='bold',y=.98)
    fig.savefig(OUT/'native-terms.png',dpi=170,facecolor='white');plt.close(fig)

    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.5,4.65));fig.subplots_adjust(left=.13,right=.98,bottom=.2,top=.76,wspace=.38)
    labels=['Matminer','Matminer\n+ MatterVial'];y=np.array([1.,0.]);banks=('mm','mm_mv')
    for a,arm in enumerate(ARMS):
        values=[100*report['banks'][bank]['substitutions']['formula_both']['fidelity'][arm]['r2'] for bank in banks];height=y+(.14 if a==0 else -.14)
        ax.barh(height,values,height=.22,color=colors[a],label=['KAN','MLP'][a]);
        for h,v in zip(height,values):ax.text(v-1.5,h,f'{v:.1f}%',ha='right',va='center',color='white')
    ax.set(yticks=y,yticklabels=labels,xlim=(0,100),xlabel='真实晶体对上的峰差 R²（%）');ax.set_title('用距离和质量复算五个节点',loc='left',fontweight='bold',pad=29)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.0),ncol=2,frameon=False,borderaxespad=0)
    case=max(report['banks']['mm_mv']['pairs'],key=lambda row:abs(row['dfpt'])); g=case['physical_equation_terms']['both']; yy=np.array([1.,0.]);start=np.zeros(2)
    for key,name,c in [('shortest_distance','距离项','#376b86'),('inverse_sqrt_lightest_mass','质量项','#738834')]:
        values=np.array([-g[a][key] for a in ARMS]);bx.barh(yy,values,left=start,height=.35,color=c,label=name)
        for h,left,w in zip(yy,start,values):bx.text(left+w/2,h,f'{w:.0f}',ha='center',va='center',color='white')
        start+=values
    for h,x in zip(yy,start):bx.text(x+11,h,f'{x:.0f}',va='center',fontsize=9)
    ref=abs(case['dfpt']);bx.axvline(ref,color='#313b44',ls='--',lw=1);bx.text(ref-8,.52,f'DFPT  {ref:.0f}',rotation=90,ha='right',va='center',fontsize=9)
    bx.set(yticks=yy,yticklabels=['KAN','MLP'],xlim=(0,830),xlabel='峰位下降量（cm⁻¹）');bx.set_title('LiH → LiI，两项可直接相加',loc='left',fontweight='bold',pad=29)
    bx.legend(loc='lower left',bbox_to_anchor=(0,1.0),ncol=2,frameon=False,borderaxespad=0)
    for a in (ax,bx):a.grid(axis='x',alpha=.12);a.set_axisbelow(True)
    fig.savefig(OUT/'physical-effectiveness.png',dpi=175,facecolor='white');plt.close(fig)


if __name__=='__main__':
    report=analyse()
    def encode(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v))
    (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2,default=encode),encoding='utf8');plot(report)
    curves={bank:[{a:f['physical_calibration'][a]['two_curves'] for a in ARMS} for f in v['folds']] for bank,v in report['banks'].items()}
    (OUT/'two-physical-curves.json').write_text(json.dumps(dict(function='F(x) = sum_i coefficients[i] * max(x - knots[i], 0). Output is F(B)-F(A), summed over the two coordinates, in cm^-1.',
            units={'shortest_distance':'angstrom','inverse_sqrt_lightest_mass':'u^(-1/2)'},folds_by_input_bank=curves),indent=2,default=encode,allow_nan=False),encoding='utf8')
