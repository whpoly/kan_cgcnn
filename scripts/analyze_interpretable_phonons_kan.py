"""Audit complete runs and test train-fitted symbolic approximations on inner validation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import itertools
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'benchmarks/interpretable-phonons-kan-20261027'
LABELS={'physical_additive':'Physical additive KAN','physical_product':'Physical product KAN',
        'physical_mlp8':'Physical MLP (8)','physical_mlp12':'Physical MLP (12)',
        'hybrid_additive':'MODNet + additive KAN','hybrid_product':'MODNet + product KAN',
        'hybrid_mlp8':'MODNet + MLP (8)','hybrid_mlp12':'MODNet + MLP (12)'}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def check(c,msg):
    if not c: raise ValueError(msg)


def basis(x):
    x=np.asarray(x,dtype=float)
    knots=np.r_[[-.5]*3,np.linspace(-.5,.5,6),[.5]*3]
    z=np.clip(x,-.5,.5-1e-6)[...,None]
    b=((z>=knots[:-1])&(z<knots[1:])).astype(float)
    for degree in range(1,4):
        n=11-degree
        d1=knots[degree:degree+n]-knots[:n]
        d2=knots[degree+1:degree+1+n]-knots[1:1+n]
        a=np.divide(z-knots[:n],d1,out=np.zeros(z.shape[:-1]+(n,)),where=d1!=0)
        c=np.divide(knots[degree+1:degree+1+n]-z,d2,out=np.zeros(z.shape[:-1]+(n,)),where=d2!=0)
        b=a*b[...,:n]+c*b[...,1:n+1]
    return np.concatenate([b,x[...,None]],axis=-1)


def design(raw,kind,mean,std,reference):
    z=(np.asarray(raw)-mean)/std
    if kind in ('constant','linear','quadratic','cubic'):
        degree={'constant':0,'linear':1,'quadratic':2,'cubic':3}[kind]
        return np.column_stack([z**i for i in range(degree+1)])
    u=np.asarray(raw)/reference
    # This helper is called only on the supported positive domain.
    value=np.log(u) if kind=='log' else (u**-.5 if kind=='inverse_sqrt' else u**-1)
    return np.column_stack([np.ones(len(u)),value])


def symbolic_edge(train_raw,train_response,val_raw,val_response):
    """Select solely on training edge responses, never validation labels/errors."""
    mean=float(np.mean(train_raw));std=max(float(np.std(train_raw)),1e-12)
    ref=float(np.median(train_raw))
    kinds=['constant','linear','quadratic','cubic']
    if np.all(train_raw>0): kinds+=['log','inverse_sqrt','inverse']
    candidates=[]
    response_std=float(np.std(train_response))
    for kind in kinds:
        a=design(train_raw,kind,mean,std,ref)
        coef=np.linalg.lstsq(a,train_response,rcond=None)[0]
        fit=a@coef
        rmse=float(np.sqrt(np.mean((fit-train_response)**2)))
        candidates.append(dict(kind=kind,coefficients=coef.tolist(),parameters=len(coef),train_rmse=rmse,
            train_relative_rmse=rmse/max(response_std,1e-12),coordinate_mean=mean,coordinate_std=std,positive_reference=ref))
    eligible=[c for c in candidates if c['train_rmse']<=max(.05*response_std,1e-10)]
    if not eligible:
        return train_response.copy(),val_response.copy(),dict(kind='retained_spline',candidates=candidates,domain_fallbacks=0)
    chosen=dict(min(eligible,key=lambda c:(c['parameters'],c['train_rmse'],c['kind'])))
    train=design(train_raw,chosen['kind'],mean,std,ref)@chosen['coefficients']
    valid=np.ones(len(val_raw),dtype=bool)
    if chosen['kind'] in ('log','inverse_sqrt','inverse'): valid=val_raw>0
    val=val_response.copy()
    val[valid]=design(val_raw[valid],chosen['kind'],mean,std,ref)@chosen['coefficients']
    # Canonical train centering; the original edge response has zero train mean.
    offset=float(train.mean());train-=offset;val-=offset
    chosen.update(centering_offset=offset,domain_fallbacks=int((~valid).sum()),candidates=candidates)
    return train,val,chosen


def explain(record,arrays,edge):
    bt=basis(arrays['x_train'])-np.asarray(edge['centers'])
    bv=basis(arrays['x_validation'])-np.asarray(edge['centers'])
    main=np.asarray(edge['main']);scale=edge['target_std']*edge['gate']
    train_main=np.einsum('nib,ib->ni',bt,main)
    val_main=np.einsum('nib,ib->ni',bv,main)
    full_t=[train_main];full_v=[val_main];simple_t=[];simple_v=[];formulas=[];simple_product_centers=[]
    for i in range(16):
        st,sv,formula=symbolic_edge(arrays['physical_train'][:,i],train_main[:,i],
                                  arrays['physical_validation'][:,i],val_main[:,i])
        simple_t.append(st);simple_v.append(sv)
        formulas.append(dict(component=edge['short_names'][i],**formula))
    if edge['factors'] is not None:
        factors=np.asarray(edge['factors']);products_t=[];products_v=[]
        for p,(i,j) in enumerate(edge['pairs']):
            left=bt[:,i]@factors[p,0];right=bt[:,j]@factors[p,1]
            left_v=bv[:,i]@factors[p,0];right_v=bv[:,j]@factors[p,1]
            center=factors[p,0]@np.asarray(edge['product_centers'][p])@factors[p,1]
            products_t.append(left*right-center);products_v.append(left_v*right_v-center)
            lt,lv,lf=symbolic_edge(arrays['physical_train'][:,i],left,arrays['physical_validation'][:,i],left_v)
            rt,rv,rf=symbolic_edge(arrays['physical_train'][:,j],right,arrays['physical_validation'][:,j],right_v)
            new_center=float(np.mean(lt*rt))
            simple_product_centers.append(new_center)
            simple_t.append(lt*rt-new_center);simple_v.append(lv*rv-new_center)
            formulas.extend([dict(component=f'product_{p}_left_{edge["short_names"][i]}',**lf),
                             dict(component=f'product_{p}_right_{edge["short_names"][j]}',**rf)])
        full_t.append(np.column_stack(products_t));full_v.append(np.column_stack(products_v))
    full_t=np.concatenate(full_t,axis=1)*scale;full_v=np.concatenate(full_v,axis=1)*scale
    np.testing.assert_allclose(full_t,arrays['train_terms'],atol=1e-7,rtol=0)
    np.testing.assert_allclose(full_v,arrays['validation_terms'],atol=1e-7,rtol=0)
    np.testing.assert_allclose(full_t.mean(axis=0),0,atol=1e-5,rtol=0)
    reconstructed=arrays['validation_backbone']+arrays['bias']+full_v.sum(axis=1)
    np.testing.assert_allclose(reconstructed,arrays['prediction'],atol=1e-7,rtol=0)
    simplified=arrays['validation_backbone']+arrays['bias']+np.column_stack(simple_v).sum(axis=1)*scale
    importance=np.mean(abs(full_t),axis=0);mask=importance>=.05*importance.max()
    masked=arrays['validation_backbone']+arrays['bias']+full_v[:,mask].sum(axis=1)
    names=list(edge['short_names'])+([f'{edge["short_names"][i]}*{edge["short_names"][j]}' for i,j in edge['pairs']]
                                    if edge['factors'] is not None else [])
    symbolic_mae=float(np.mean(abs(simplified-arrays['target'])))
    masked_mae=float(np.mean(abs(masked-arrays['target'])))
    np.testing.assert_allclose(masked_mae,record['diagnostics']['masked_validation_mae'],atol=1e-9)
    return dict(variant=record['variant'],repeat=record['repeat'],inner=record['inner'],
        validation_mae=record['validation_mae'],symbolic_validation_mae=symbolic_mae,
        symbolic_mae_delta=symbolic_mae-record['validation_mae'],
        symbolic_prediction_shift_mae=float(np.mean(abs(simplified-arrays['prediction']))),
        simplified_edges=sum(f['kind']!='retained_spline' for f in formulas),total_edges=len(formulas),
        symbolic_domain_fallbacks=sum(f['domain_fallbacks'] for f in formulas),
        masked_validation_mae=masked_mae,masked_mae_delta=masked_mae-record['validation_mae'],
        retained_terms=int(mask.sum()),total_terms=len(mask),
        component_names=names,train_importance=importance.tolist(),
        train_correction_abs_mean=float(np.mean(abs(full_t.sum(axis=1)+float(arrays['bias'])))),
        correction_std_over_prediction_std=float(np.std(full_t.sum(axis=1))/max(np.std(arrays['train_prediction']),1e-12)),
        # A magnitude ratio is not explained variance or percent of physical truth.
        component_removal_validation_mae_delta=record['diagnostics']['removed_term_mae_delta'],formulas=formulas,
        symbolic_product_centers=simple_product_centers,physical_response_multiplier=scale,
        physical_response_bias=float(arrays['bias']),target_mean=edge['target_mean'])


def write_formulas(directory,detail):
    """Human-readable candidates, including cases that remain numerical splines."""
    lines=['# 训练集拟合的边函数简化候选','',
        '这些是已训练模型的近似表达，不是从材料中确认的物理定律。公式只拟合训练边响应；选择没有使用验证标签。', '',
        f"结构：{detail['variant']}，种子编号 {detail['repeat']+1}，内层折 {detail['inner']+1}。",'',
        f"原验证 MAE {detail['validation_mae']:.6f}；替换后 {detail['symbolic_validation_mae']:.6f}；预测平均偏移 {detail['symbolic_prediction_shift_mae']:.6f} cm⁻¹。",'',
        f"将下表 16 个主效应相加，并加乘积项，最后乘 {detail['physical_response_multiplier']:.10g}，再加分支常数 {detail['physical_response_bias']:.10g} cm⁻¹。",'',
        ('独立模型还需加目标训练均值 '+str(detail['target_mean'])+' cm⁻¹。'
         if detail['variant'].startswith('physical') else '混合模型还需加深层主干预测（原单位）；下表不能解释深层主干。'),'',
        '| 分量 | 近似表达 | 坐标定义 | 训练相对 RMSE |',
        '| --- | --- | --- | ---: |']
    for f in detail['formulas']:
        if f['kind']=='retained_spline':
            expr='保留数值样条，见 edge-functions.json';coord='原始物理量逐列缩放';error='—'
        else:
            coefficients=f['coefficients'];kind=f['kind']
            if kind in ('constant','linear','quadratic','cubic'):
                expr=' + '.join(f'({c:.9g})'+('' if k==0 else f' z^{k}') for k,c in enumerate(coefficients))
                coord=f"z=(x−{f['coordinate_mean']:.9g})/{f['coordinate_std']:.9g}"
            else:
                term={'log':'log(u)','inverse_sqrt':'u^(−1/2)','inverse':'u^(−1)'}[kind]
                expr=f"({coefficients[0]:.9g}) + ({coefficients[1]:.9g}) {term}"
                coord=f"u=x/{f['positive_reference']:.9g}, x>0"
            expr+=f" − ({f['centering_offset']:.9g})"
            error=f"{f['train_relative_rmse']:.5f}"
        lines.append(f"| {f['component']} | {expr} | {coord} | {error} |")
    if detail['symbolic_product_centers']:
        lines+=['','每个乘积项等于对应 left 函数 × right 函数，再减去以下训练均值：','']
        for p,c in enumerate(detail['symbolic_product_centers']):lines.append(f'- 乘积 {p}: {c:.12g}')
    lines+=['',f"正值域之外使用原样条回退的边×验证样本数：{detail['symbolic_domain_fallbacks']}。任何回退或保留样条都表示尚未获得全域解析表达。",'',
        '公式中的质量、半径等沿用数据集原始单位，z 或 u 为无量纲坐标；输出乘数将模型响应转回 cm⁻¹。不要把拟合系数解释成独立的力常数。']
    (directory/'symbolic-candidates.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def curve_stability(variant,members,output):
    fig,axes=plt.subplots(4,4,figsize=(15,12))
    rows=[]
    for i,ax in enumerate(axes.flat):
        lo=max(np.quantile(a['physical_train'][:,i],.05) for _,a,_ in members)
        hi=min(np.quantile(a['physical_train'][:,i],.95) for _,a,_ in members)
        name=members[0][2]['short_names'][i]
        if hi<=lo:
            ax.set_title(name+' (no common range)');rows.append(dict(feature=name,overlap=False));continue
        grid=np.linspace(lo,hi,100);curves=[]
        for _,a,e in members:
            scaled=grid*a['scaler_scale'][i]+a['scaler_min'][i]
            c=(basis(scaled)-np.asarray(e['centers'][i]))@np.asarray(e['main'][i])
            curves.append(c*e['gate']*e['target_std'])
        curves=np.array(curves)
        correlations=[]
        for a,b in itertools.combinations(curves,2):
            if a.std()>1e-8 and b.std()>1e-8: correlations.append(float(np.corrcoef(a,b)[0,1]))
        for c in curves: ax.plot(grid,c,lw=.6,alpha=.25,color='#257d9b')
        ax.plot(grid,curves.mean(axis=0),color='#073b4c',lw=2)
        ax.fill_between(grid,curves.mean(axis=0)-curves.std(axis=0),curves.mean(axis=0)+curves.std(axis=0),alpha=.15,color='#257d9b')
        ax.axhline(0,color='gray',lw=.5);ax.set_title(name,fontsize=9)
        ax.set_xlabel('Original descriptor units',fontsize=8);ax.set_ylabel('Contribution (cm$^{-1}$)',fontsize=8)
        rows.append(dict(feature=name,overlap=True,range=[float(lo),float(hi)],
            median_pairwise_correlation=float(np.median(correlations)) if correlations else None,
            positive_endpoint_trends=int(np.sum(curves[:,-1]>curves[:,0])),curves=len(curves),
            mean_curve_span=float(np.ptp(curves.mean(axis=0)))))
    fig.suptitle(LABELS[variant]+' | 5 folds x 2 seeds\nMain-effect functions; common training 5-95% range; bands = across-fit SD, not confidence intervals',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(output/f'curves-{variant}.png',dpi=160)
    fig.savefig(output/f'curves-{variant}.pdf');plt.close(fig)
    return rows


def product_stability(variant,members,output):
    """Compare complete products; factor sign/scale alone is not identifiable."""
    fig,axes=plt.subplots(2,2,figsize=(12,9));rows=[]
    pairs=members[0][2]['pairs'];names=members[0][2]['short_names']
    for p,((i,j),ax) in enumerate(zip(pairs,axes.flat)):
        lows=[max(np.quantile(a['physical_train'][:,k],.05) for _,a,_ in members) for k in (i,j)]
        highs=[min(np.quantile(a['physical_train'][:,k],.95) for _,a,_ in members) for k in (i,j)]
        if any(hi<=lo for lo,hi in zip(lows,highs)):
            ax.set_title('No common supported range');rows.append(dict(pair=[names[i],names[j]],supported_cells=0));continue
        ex=np.linspace(lows[0],highs[0],21);ey=np.linspace(lows[1],highs[1],21)
        gx=(ex[:-1]+ex[1:])/2;gy=(ey[:-1]+ey[1:])/2
        surfaces=[];supported=np.ones((20,20),dtype=bool)
        for _,a,e in members:
            f=np.asarray(e['factors'][p]);center=f[0]@np.asarray(e['product_centers'][p])@f[1]
            bx=basis(gx*a['scaler_scale'][i]+a['scaler_min'][i])-np.asarray(e['centers'][i])
            by=basis(gy*a['scaler_scale'][j]+a['scaler_min'][j])-np.asarray(e['centers'][j])
            surface=(np.outer(bx@f[0],by@f[1])-center)*e['gate']*e['target_std']
            surfaces.append(surface)
            hist,_,_=np.histogram2d(a['physical_train'][:,i],a['physical_train'][:,j],bins=(ex,ey))
            supported &= hist>=1
        surfaces=np.asarray(surfaces);mean=surfaces.mean(axis=0)
        image=np.where(supported,mean,np.nan)
        size=float(np.nanmax(abs(image))) if supported.any() else 1.
        mesh=ax.pcolormesh(ex,ey,image.T,cmap='RdBu_r',vmin=-max(size,1e-5),vmax=max(size,1e-5),shading='flat')
        fig.colorbar(mesh,ax=ax,label='Contribution (cm$^{-1}$)')
        ax.set_xlabel(names[i]+' (raw units)');ax.set_ylabel(names[j]+' (raw units)')
        ax.set_title(f'{names[i]} x {names[j]}\n{supported.sum()}/400 occupied cells shared by all fits',fontsize=10)
        values=surfaces[:,supported];correlations=[]
        if supported.sum()>=5:
            for a,b in itertools.combinations(values,2):
                if a.std()>1e-8 and b.std()>1e-8:correlations.append(float(np.corrcoef(a,b)[0,1]))
        rows.append(dict(pair=[names[i],names[j]],supported_cells=int(supported.sum()),
            median_surface_correlation=float(np.median(correlations)) if correlations else None,
            mean_supported_abs_contribution=float(np.mean(abs(values))) if supported.any() else None))
    fig.suptitle(LABELS[variant]+' | complete pair contributions\nBlank cells lack joint training support in at least one fold; associations, not causal effects',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(output/f'interactions-{variant}.png',dpi=170)
    fig.savefig(output/f'interactions-{variant}.pdf');plt.close(fig)
    return rows


def main():
    p=read(RUN/'protocol.json');status=read(RUN/'status.json')
    check(status['state']=='complete' and status['completed']==80,'Run not complete')
    fp=digest(p)
    for name,value in p['source_sha256'].items(): check(sha(ROOT/'scripts'/name)==value,'Changed source '+name)
    records=[];details=[];members={};prediction_delta=0
    for path in sorted((RUN/'fits').glob('*/*/result.json')):
        r=read(path);inp=p['source_inputs'][r['inner']];info=inp['source']
        job=dict(directory=r['directory'],variant=r['variant'],inner=r['inner'],repeat=r['repeat'],seed=r['seed'],
            lr=p['learning_rate'],max_epochs=p['max_epochs'],expected_parameters=p['parameters'][r['variant']],input=inp,run_fingerprint=fp)
        check(r['signature']==digest(job) and r['run_fingerprint']==fp,'Job signature')
        check(r['seed']==p['seeds'][r['repeat']]+r['inner'],'Seed mismatch')
        check(r['input_digest']==digest(inp),'Input digest')
        check(r['train_ids']==info['train_ids']==r['target_transform_fit_ids'],'Train scope')
        check(r['validation_ids']==info['validation_ids'],'Validation IDs')
        check(sha(inp['source_manifest'])==inp['source_manifest_sha256'],'Manifest changed')
        for name,value in info['artifacts'].items():check(sha(Path(info['directory'])/name)==value,'Input changed')
        for name,value in r['artifacts'].items():check(sha(path.parent/name)==value,'Fit artifact changed')
        with np.load(path.parent/'predictions.npz') as a: arrays={k:a[k] for k in a.files}
        with np.load(Path(info['directory'])/'arrays.npz') as a:
            np.testing.assert_array_equal(arrays['target'],a['y_validation'])
            np.testing.assert_allclose([r['target_mean'],r['target_std']],
                [a['y_train'].astype(float).mean(),a['y_train'].astype(float).std()],rtol=0,atol=1e-10)
        check(arrays['sample_ids'].tolist()==info['validation_ids'],'Prediction IDs')
        mae=float(np.mean(abs(arrays['prediction']-arrays['target'])))
        prediction_delta=max(prediction_delta,abs(mae-r['validation_mae']))
        check(abs(mae-r['validation_mae'])<1e-10,'MAE mismatch')
        check(r['best_epoch']==int(np.argmin(r['history']['val_mae']))+1,'Checkpoint epoch')
        check(abs(mae-r['checkpoint_mae'])<.003 and r['epochs']<=400,'Checkpoint score/budget')
        with (path.parent/'model.pkl').open('rb') as stream: model=pickle.load(stream)
        check(sum(w.size for w in model['weights'])==r['parameters']==p['parameters'][r['variant']],'Capacity')
        records.append(r)
        if 'edge-functions.json' in r['artifacts']:
            edge=read(path.parent/'edge-functions.json')
            detail=explain(r,arrays,edge);details.append(detail)
            write_formulas(path.parent,detail)
            members.setdefault(r['variant'],[]).append((r,arrays,edge))
    keys={(r['variant'],r['repeat'],r['inner']) for r in records}
    expected={(v,s,i) for v in p['variants'] for s in range(2) for i in range(5)}
    check(keys==expected and len(records)==80,'Incomplete grid')
    baseline={}
    for b in p['baseline_references']:
        check(sha(b['path'])==b['sha256'],'Changed baseline')
        r=read(b['path']);info=p['source_inputs'][b['inner']]['source']
        check(r['train_ids']==info['train_ids'] and r['validation_ids']==info['validation_ids'],'Baseline split')
        check(r['selected_features']==info['selected_features'] and r['seed']==p['seeds'][b['repeat']]+b['inner'],'Baseline features/seed')
        baseline[(b['repeat'],b['inner'])]=b['validation_mae']
    rows=[]
    for v in p['variants']:
        fits=[r for r in records if r['variant']==v]
        scores=[[r['validation_mae'] for r in sorted(fits,key=lambda r:r['inner']) if r['repeat']==s] for s in range(2)]
        item=dict(variant=v,parameters=fits[0]['parameters'],seed_mae=[float(np.mean(a)) for a in scores],
            fold_mae=scores,mean_mae=float(np.mean(scores)),train_mae=float(np.mean([r['train_mae'] for r in fits])),
            mean_seconds=float(np.mean([r['seconds'] for r in fits])),cap_hits=sum(r['epochs']==400 for r in fits))
        if not v.split('_')[1].startswith('mlp'):
            d=[a for a in details if a['variant']==v]
            compare=v.split('_')[0]+'_'+('mlp8' if v.endswith('additive') else 'mlp12')
            other={(r['repeat'],r['inner']):r['validation_mae'] for r in records if r['variant']==compare}
            item.update(matched_mlp=compare,mlp_wins=sum(r['validation_mae']<other[(r['repeat'],r['inner'])] for r in fits),
                modnet_wins=sum(r['validation_mae']<baseline[(r['repeat'],r['inner'])] for r in fits),
                mean_retained_terms=float(np.mean([a['retained_terms'] for a in d])),
                symbolic_mae=float(np.mean([a['symbolic_validation_mae'] for a in d])),
                masked_mae=float(np.mean([a['masked_validation_mae'] for a in d])),
                symbolic_prediction_shift=float(np.mean([a['symbolic_prediction_shift_mae'] for a in d])),
                simplified_edge_fraction=float(sum(a['simplified_edges'] for a in d)/sum(a['total_edges'] for a in d)),
                correction_abs_mean=float(np.mean([a['train_correction_abs_mean'] for a in d])),
                correction_std_ratio=float(np.mean([a['correction_std_over_prediction_std'] for a in d])))
            sets=[set(np.argsort(a['train_importance'])[-5:]) for a in d]
            item['top5_jaccard']=float(np.mean([len(a&b)/len(a|b) for a,b in itertools.combinations(sets,2)]))
        rows.append(item)
    figures=RUN/'figures';figures.mkdir(exist_ok=True)
    stability={v:curve_stability(v,items,figures) for v,items in members.items()}
    products={v:product_stability(v,items,figures) for v,items in members.items() if v.endswith('product')}
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for scope,ax in zip(('physical','hybrid'),axes):
        items=[r for r in rows if r['variant'].startswith(scope+'_')]
        x=np.arange(4)
        for seed in range(2):ax.bar(x+(seed-.5)*.34,[r['seed_mae'][seed] for r in items],.34,label=f'Seed {seed+1}')
        if scope=='hybrid':ax.axhline(np.mean(list(baseline.values())),ls='--',color='gray',label='MODNet two-seed mean')
        ax.set_xticks(x,['Additive KAN','Product KAN','MLP (8)','MLP (12)'],rotation=15)
        ax.set_ylabel('Inner-validation MAE (cm$^{-1}$)');ax.set_title('16 physical inputs' if scope=='physical' else '256 inputs + physical correction')
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figures/'validation-comparison.png',dpi=180);fig.savefig(figures/'validation-comparison.pdf');plt.close(fig)
    training_correlations=[]
    for _,arrays,edge in members['physical_additive']:
        if _['repeat']==0:
            training_correlations.append(np.corrcoef(arrays['physical_train'],rowvar=False))
    correlation_pairs=[]
    for i,j in itertools.combinations(range(16),2):
        values=[float(c[i,j]) for c in training_correlations]
        if min(abs(v) for v in values)>=.8:
            correlation_pairs.append(dict(features=[p['physical_names'][i],p['physical_names'][j]],
                                          fold_correlations=values))
    result=dict(analyzed_at=datetime.now(timezone.utc).isoformat(),analyzer_sha256=sha(__file__),run_fingerprint=fp,
        audited_fits=80,mae_max_difference=prediction_delta,outer_test_evaluated=False,rows=rows,
        baseline_seed_mae=[float(np.mean([baseline[(s,i)] for i in range(5)])) for s in range(2)],
        symbolic_and_ablation=details,main_curve_stability=stability,product_surface_stability=products,
        strongly_correlated_physical_features=correlation_pairs,
        caveats=p['limitations'],wall_minutes=status['wall_seconds']/60)
    (RUN/'interpretability-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 物理坐标 KAN：验证与解释性结果','',
        f"完成并核验 80 次训练，墙钟 {result['wall_minutes']:.2f} 分钟。MAE 单位 cm⁻¹，越低越好。全部来自相同五个内层划分，未评估外层测试。",'',
        '两个种子均不选择新学习率；学习率固定 0.001。物理独立模型仅用 16 个描述符，混合模型保留 256 个增强特征。不同信息量的模型不能只按参数量直接归因；KAN/MLP 配对使用相同输入和预算。','',
        '| 模型 | 种子 1 | 种子 2 | 参数量 | 每次秒数 | 到达 400 轮上限 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append(f"| {LABELS[r['variant']]} | {r['seed_mae'][0]:.3f} | {r['seed_mae'][1]:.3f} | {r['parameters']:,} | {r['mean_seconds']:.1f} | {r['cap_hits']}/10 |")
    lines+=['',f"同协议增强 MODNet 参考：{result['baseline_seed_mae'][0]:.3f} / {result['baseline_seed_mae'][1]:.3f}；它已有对应结果，本轮未重复训练。",'',
        '精确分量遮蔽与简单函数替换（两个种子的五折均值）：','',
        '| KAN | 原始 MAE | 遮蔽小项后 | 简式替换后 | 平均保留项数 | 简式边比例 | 胜对应 MLP | 胜 MODNet |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        if 'matched_mlp' in r:lines.append(f"| {LABELS[r['variant']]} | {r['mean_mae']:.3f} | {r['masked_mae']:.3f} | {r['symbolic_mae']:.3f} | {r['mean_retained_terms']:.1f} | {r['simplified_edge_fraction']:.1%} | {r['mlp_wins']}/10 | {r['modnet_wins']}/10 |")
    lines+=['','遮蔽仅按训练贡献阈值决定；没有重训或实际压缩参数。简单函数选择仅拟合训练样本上的边函数输出，验证标签没有参与公式选择；未达到精度门槛的边仍保留样条，因此简式边比例不足 100% 时不是完整解析公式。','',
        '训练贡献排序及单变量曲线稳定性：','',
        '| KAN | 前五项集合平均 Jaccard | 中位曲线相关性 ≥ 0.7 的变量 | 修正量平均绝对值 |',
        '| --- | ---: | ---: | ---: |']
    for r in rows:
        if 'matched_mlp' in r:
            stable=sum(a.get('median_pairwise_correlation') is not None and a['median_pairwise_correlation']>=.7 for a in stability[r['variant']])
            lines.append(f"| {LABELS[r['variant']]} | {r['top5_jaccard']:.3f} | {stable}/16 | {r['correction_abs_mean']:.3f} |")
    lines+=['','修正量大小不能解释为“解释了多少百分比的物理”；独立模型该列表示去掉目标均值后的模型输出。相关性阈值是描述性统计，既不证明因果，也不是统计显著性。乘积模型此处只统计主效应曲线，乘积因子存在符号/尺度不唯一性。','',
        '![配对验证成绩](figures/validation-comparison.png)','',
        '## 模型解释的范围','',
        '独立 KAN：常数 + 16 条物理变量函数，乘积版另外包含四个预设两变量项。所有分量可精确加回预测。混合 KAN：同样分解只覆盖物理残差分支，深层 MODNet 主干仍不透明，分支之间的相关信息使物理分配并不唯一。','',
        '本轮不强制平均质量的单调性或平方根规律。声子动力学涉及质量加权力常数矩阵；PhDOS 峰值不等于一个有效弹簧的频率。键长、电负性和配位数只是成键代理，不能把学出的乘积项直接命名为力常数。','',
        '所有 16 变量的跨种子、跨折曲线位于 `figures/curves-*.png` 和 PDF，横轴使用原始描述符单位。只画各训练分布 5%—95% 区间的交集；阴影是十次训练曲线的标准差，不是置信区间。完整两变量贡献图位于 `figures/interactions-*.png`，只保留每个训练划分都至少有一个样本的二维格子，空白区不支持解释。完整曲面相关性见 JSON，避免将因子任意的符号和尺度当作不稳定物理机制。','',
        '每个模型的具体函数系数、无量纲化参考、保留的样条、域外回退次数、逐项遮蔽误差和五折成绩见 `interpretability-analysis.json`；每次训练的精确边函数见 `fits/*/*/edge-functions.json`，可读近似公式见对应 `symbolic-candidates.md`。','',
        f"五个训练划分中均满足绝对 Pearson 相关系数至少 0.8 的描述符对有 {len(correlation_pairs)} 对，具体清单见 JSON。强相关输入之间的归因可能互相替代，不能把单条曲线直接解释为控制其他材料属性后的真实实验效应。",'',
        '本轮是定预算初筛，达到 400 轮上限的模型可能尚未充分优化，不能把结果推广为整个方法类别的能力上限。所有结构均用同样优化协议，但同样协议不保证对不同结构同样合适。','',
        '公式选择、早停和既往探索都使用内层数据，因此只能提出可检验的经验关系，不能宣称发现材料定律或超过论文正式外层 benchmark。', '',
        '参考：[KAN 2.0](https://arxiv.org/abs/2408.10205)、[Phonopy 动力学矩阵](https://phonopy.github.io/phonopy/formulation.html#dynamical-matrix)。']
    (RUN/'interpretability-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('audited_fits','mae_max_difference','rows','baseline_seed_mae','wall_minutes')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
