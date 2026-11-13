"""Independent network reconstruction, train-only pruning and symbolic audit."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import itertools
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'benchmarks/standalone-kan-screen-20261027'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def check(condition,msg):
    if not condition:raise ValueError(msg)


class ScalerState:
    """Only deserialize training scaler attributes; never invoke another sklearn version."""
    pass


class ScalerStateReader(pickle.Unpickler):
    def find_class(self,module,name):
        if name=='MinMaxScaler' and module.startswith('sklearn.preprocessing'):
            return ScalerState
        return super().find_class(module,name)


def basis(x):
    x=np.asarray(x,dtype=float)
    knots=np.r_[[-.5]*3,np.linspace(-.5,.5,6),[.5]*3]
    z=np.clip(x,-.5,.5-1e-6)[...,None]
    b=((z>=knots[:-1])&(z<knots[1:])).astype(float)
    for d in range(1,4):
        n=11-d;dl=knots[d:d+n]-knots[:n];dr=knots[d+1:d+1+n]-knots[1:1+n]
        l=np.divide(z-knots[:n],dl,out=np.zeros(z.shape[:-1]+(n,)),where=dl!=0)
        r=np.divide(knots[d+1:d+1+n]-z,dr,out=np.zeros(z.shape[:-1]+(n,)),where=dr!=0)
        b=l*b[...,:n]+r*b[...,1:n+1]
    return b


def forward(network,x,collect=False):
    values=np.asarray(x,dtype=float);states=[]
    for layer in network['layers']:
        z=.5*np.tanh(values) if layer['hidden_coordinates'] else values
        b=basis(z);coeff=np.asarray(layer['coefficients'])
        if collect:states.append((z,b))
        if 'degrees' not in layer:
            values=np.einsum('nik,iok->no',b,coeff,optimize=True)+layer['bias']
        else:
            degrees=np.asarray(layer['degrees']);keep=degrees<0
            values=np.einsum('nik,iok->no',b,coeff*keep[...,None],optimize=True)+layer['bias']
            q=(z-np.asarray(layer['coordinate_mean']))/np.asarray(layer['coordinate_std'])
            pc=np.asarray(layer['polynomial_coefficients'])
            for degree in range(4):values+=np.einsum('ni,io->no',q**degree,pc[:,:,degree],optimize=True)
    prediction=values.reshape(-1)*network['target_std']+network['target_mean']
    return (prediction,states) if collect else prediction


def transformations(network,x):
    """No validation data enter edge statistics, masks, or polynomial selection."""
    _,states=forward(network,x,collect=True)
    pruned=json.loads(json.dumps(network));symbolic=json.loads(json.dumps(network));summaries=[]
    for index,(layer,(z,b)) in enumerate(zip(network['layers'],states)):
        c=np.asarray(layer['coefficients']);mean=b.mean(axis=0)
        cov=np.einsum('nik,nil->ikl',b,b,optimize=True)/len(b)-np.einsum('ik,il->ikl',mean,mean)
        response_mean=np.einsum('ik,iok->io',mean,c)
        response_std=np.sqrt(np.maximum(np.einsum('iok,ikl,iol->io',c,cov,c,optimize=True),0))
        keep=response_std>=.05*response_std.max()
        pruned['layers'][index]['coefficients']=(c*keep[...,None]).tolist()
        pruned['layers'][index]['bias']=(np.asarray(layer['bias'])+(response_mean*(~keep)).sum(axis=0)).tolist()
        degrees=np.full(c.shape[:2],-1,dtype=int);poly=np.zeros(c.shape[:2]+(4,))
        means=z.mean(axis=0);stds=np.maximum(z.std(axis=0),1e-12)
        for i in range(c.shape[0]):
            q=(z[:,i]-means[i])/stds[i]
            for degree in range(4):
                design=np.column_stack([q**k for k in range(degree+1)])
                transfer=np.linalg.lstsq(design,b[:,i,:],rcond=None)[0]
                residual=b[:,i,:]-design@transfer
                error_cov=residual.T@residual/len(b)
                rmse=np.sqrt(np.maximum(np.einsum('ok,kl,ol->o',c[i],error_cov,c[i],optimize=True),0))
                accept=(degrees[i]<0)&(rmse<=np.maximum(.05*response_std[i],1e-10))
                degrees[i,accept]=degree
                poly[i,accept,:degree+1]=(transfer@c[i,accept].T).T
        symbolic['layers'][index].update(degrees=degrees.tolist(),polynomial_coefficients=poly.tolist(),
            coordinate_mean=means.tolist(),coordinate_std=stds.tolist())
        summaries.append(dict(layer=layer['name'],total_edges=int(keep.size),kept_edges=int(keep.sum()),
            active_input_coordinates=int(np.any(keep,axis=1).sum()),symbolic_edges=int((degrees>=0).sum()),
            degree_counts={str(d):int((degrees==d).sum()) for d in range(-1,4)},
            training_response_std=response_std.tolist(),training_response_mean=response_mean.tolist()))
    return pruned,symbolic,summaries


def physical_ale(network,x,raw_scale,raw_min):
    """First-order ALE of the FINAL network output, on inner training inputs."""
    curves=[]
    for i in range(x.shape[1]):
        edges=np.unique(np.quantile(x[:,i],np.linspace(0,1,11)))
        if len(edges)<2:
            curves.append(dict(feature=network['features'][i],constant=True));continue
        bins=np.clip(np.searchsorted(edges,x[:,i],side='right')-1,0,len(edges)-2)
        low=x.copy();high=x.copy();low[:,i]=edges[bins];high[:,i]=edges[bins+1]
        changes=forward(network,high)-forward(network,low)
        count=np.bincount(bins,minlength=len(edges)-1)
        mean=np.divide(np.bincount(bins,weights=changes,minlength=len(count)),count,
                       out=np.zeros(len(count)),where=count>0)
        accumulated=np.r_[0,np.cumsum(mean)]
        accumulated-=np.average((accumulated[:-1]+accumulated[1:])/2,weights=count)
        curves.append(dict(feature=network['features'][i],constant=False,
            raw_edges=((edges-raw_min[i])/raw_scale[i]).tolist(),effects=accumulated.tolist(),bin_counts=count.tolist()))
    return curves


def audit_network(path,r,inp,plot_curves=True):
    identity=digest(dict(analyzer=sha(__file__),network=sha(path/'network.json'),
        input=inp['source']['artifacts']['arrays.npz'],scaler=inp['source']['artifacts']['scaler.pkl'],
        numpy=np.__version__,curves=plot_curves))
    cached_path=path/'interpretation.json'
    if cached_path.exists():
        cached=read(cached_path)
        if cached.get('analysis_fingerprint')==identity and all(
            sha(path/name)==value for name,value in cached.get('derived_artifacts',{}).items()):
            return cached
    network=read(path/'network.json');info=inp['source']
    indices=[info['selected_features'].index(f) for f in r['features']]
    with np.load(Path(info['directory'])/'arrays.npz') as a:
        x=a['x_train'][:,indices].astype(float);xv=a['x_validation'][:,indices].astype(float);yv=a['y_validation']
    original=forward(network,xv)
    with np.load(path/'predictions.npz') as a:saved=a['prediction']
    maximum=float(np.max(abs(original-saved)))
    check(maximum<.01,'Independent network reconstruction error: '+str(maximum))
    pruned,symbolic,layers=transformations(network,x)
    p=forward(pruned,xv);s=forward(symbolic,xv)
    check(np.isfinite(p).all() and np.isfinite(s).all(),'Nonfinite transformed network')
    detail=dict(bank=r['bank'],variant=r['variant'],repeat=r['repeat'],inner=r['inner'],
        original_mae=r['validation_mae'],numpy_max_prediction_difference=maximum,
        pruned_mae=float(np.mean(abs(p-yv))),pruned_prediction_shift=float(np.mean(abs(p-original))),
        symbolic_mae=float(np.mean(abs(s-yv))),symbolic_prediction_shift=float(np.mean(abs(s-original))),
        edges=sum(l['total_edges'] for l in layers),kept_edges=sum(l['kept_edges'] for l in layers),
        symbolic_edges=sum(l['symbolic_edges'] for l in layers),layers=layers,
        diagnostics=network['diagnostics'])
    if r['bank']=='physical' and plot_curves:
        # Training-derived scaler is a trusted artifact generated by the preceding run.
        with (Path(info['directory'])/'scaler.pkl').open('rb') as stream:scaler=ScalerStateReader(stream).load()
        detail['ale']=physical_ale(network,x,scaler.scale_[indices],scaler.min_[indices])
    for name,data in [('pruned-network.json',pruned),('symbolic-network.json',symbolic)]:
        (path/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    detail['analysis_fingerprint']=identity
    detail['derived_artifacts']={name:sha(path/name) for name in ('pruned-network.json','symbolic-network.json')}
    (path/'interpretation.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return detail


def plot_ale(details,output):
    results={}
    for variant in ('kan','sparse_kan'):
        group=[d for d in details if d['bank']=='physical' and d['variant']==variant]
        if len(group)!=10:continue
        fig,axes=plt.subplots(4,4,figsize=(15,12));rows=[]
        names=['Mean mass','Mass range','Mass deviation','Maximum mass','Mean radius','Radius deviation',
               'Density','Volume per atom','Mean bond length','Bond length deviation','Coordination',
               'Neighbor variation','Mean electronegativity','EN deviation','Mean valence','Valence deviation']
        for i,ax in enumerate(axes.flat):
            curves=[d['ale'][i] for d in group]
            if any(c['constant'] for c in curves):ax.set_title(names[i]+' (constant)');continue
            # Restrict to the common interior quantile bins; no boundary extrapolation.
            lo=max(c['raw_edges'][1] for c in curves);hi=min(c['raw_edges'][-2] for c in curves)
            if hi<=lo:ax.set_title(names[i]+' (no overlap)');continue
            grid=np.linspace(lo,hi,100)
            values=np.array([np.interp(grid,c['raw_edges'],c['effects']) for c in curves])
            correlations=[float(np.corrcoef(a,b)[0,1]) for a,b in itertools.combinations(values,2) if a.std()>1e-8 and b.std()>1e-8]
            for v in values:ax.plot(grid,v,color='#00838f',alpha=.22,lw=.7)
            ax.plot(grid,values.mean(axis=0),color='#00363a',lw=2)
            ax.fill_between(grid,values.mean(axis=0)-values.std(axis=0),values.mean(axis=0)+values.std(axis=0),color='#00838f',alpha=.15)
            ax.axhline(0,color='gray',lw=.5);ax.set_title(names[i],fontsize=10)
            ax.set_xlabel('Original descriptor units',fontsize=8);ax.set_ylabel('Final-output ALE (cm$^{-1}$)',fontsize=8)
            rows.append(dict(feature=curves[0]['feature'],median_curve_correlation=float(np.median(correlations)) if correlations else None,
                             positive_endpoint_trends=int((values[:,-1]>values[:,0]).sum())))
        fig.suptitle(f'Standalone physical {variant} | 5 folds x 2 seeds\nALE of complete output; training data only; bands are across-fit SD, not confidence intervals',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output/f'physical-ale-{variant}.png',dpi=160)
        fig.savefig(output/f'physical-ale-{variant}.pdf');plt.close(fig);results[variant]=rows
    return results


def main():
    p=read(RUN/'protocol.json');status=read(RUN/'status.json')
    check(status['state']=='complete' and status['completed']==70,'Training is incomplete')
    fp=digest(p)
    for name,value in p['source_sha256'].items():check(sha(ROOT/'scripts'/name)==value,'Changed training source')
    records=[];details=[];mae_difference=0
    for path in sorted((RUN/'fits').glob('*/*/*/result.json')):
        r=read(path);inp=p['source_inputs'][r['inner']];info=inp['source']
        job=dict(directory=r['directory'],bank=r['bank'],variant=r['variant'],repeat=r['repeat'],inner=r['inner'],
            seed=r['seed'],lr=p['learning_rate'],max_epochs=p['max_epochs'],expected_parameters=p['expected_parameters'][r['bank']][r['variant']],
            input=inp,run_fingerprint=fp)
        check(r['signature']==digest(job) and r['run_fingerprint']==fp,'Job fingerprint')
        check(r['input_digest']==digest(inp),'Input digest')
        check(r['seed']==p['seeds'][r['repeat']]+r['inner'],'Unpaired seed')
        check(r['train_ids']==info['train_ids']==r['target_transform_fit_ids'],'Wrong training scope')
        check(r['validation_ids']==info['validation_ids'],'Wrong validation IDs')
        check(sha(inp['source_manifest'])==inp['source_manifest_sha256'],'Input manifest changed')
        for name,value in info['artifacts'].items():check(sha(Path(info['directory'])/name)==value,'Changed input artifact')
        for name,value in r['artifacts'].items():check(sha(path.parent/name)==value,'Changed fit artifact')
        check(not any('SISSO' in f.upper() for f in r['features']),'SISSO found')
        if r['bank']=='enhanced':check(r['features']==info['selected_features'],'Enhanced feature mismatch')
        else:check(r['features']==info['selected_features'][:16],'Physical feature mismatch')
        with np.load(path.parent/'predictions.npz') as a:
            check(a['sample_ids'].tolist()==info['validation_ids'],'Wrong prediction IDs')
            mae=float(np.mean(abs(a['prediction']-a['target'])))
            with np.load(Path(info['directory'])/'arrays.npz') as source:
                np.testing.assert_array_equal(a['target'],source['y_validation'])
                np.testing.assert_allclose([r['target_mean'],r['target_std']],
                    [source['y_train'].astype(float).mean(),source['y_train'].astype(float).std()],atol=1e-10,rtol=0)
        mae_difference=max(mae_difference,abs(mae-r['validation_mae']))
        check(abs(mae-r['validation_mae'])<1e-10,'MAE mismatch')
        check(1<=r['best_epoch']<=r['epochs']<=400 and r['best_epoch']==int(np.argmin(r['history']['val_mae']))+1,'Best checkpoint')
        check(abs(mae-r['checkpoint_mae'])<.003,'Checkpoint units')
        with (path.parent/'model.pkl').open('rb') as stream:model=pickle.load(stream)
        check(sum(w.size for w in model['weights'])==r['parameters']==p['expected_parameters'][r['bank']][r['variant']],'Capacity mismatch')
        records.append(r)
        if r['variant'] in ('kan','sparse_kan'):
            detail=audit_network(path.parent,r,inp);details.append(detail)
            print('AUDIT',r['bank'],r['variant'],r['repeat'],r['inner'],flush=True)
    expected={(b,v,s,i) for b,v in p['configs'] for s in range(2) for i in range(5)}
    keys={(r['bank'],r['variant'],r['repeat'],r['inner']) for r in records}
    check(len(records)==70 and keys==expected,'Incomplete grid')
    rows=[]
    for bank,v in p['configs']:
        fits=[r for r in records if r['bank']==bank and r['variant']==v]
        values=[[r['validation_mae'] for r in sorted(fits,key=lambda r:r['inner']) if r['repeat']==s] for s in range(2)]
        item=dict(bank=bank,variant=v,seed_mae=[float(np.mean(x)) for x in values],fold_mae=values,
            mean_mae=float(np.mean(values)),train_mae=float(np.mean([r['train_mae'] for r in fits])),parameters=fits[0]['parameters'],
            mean_seconds=float(np.mean([r['seconds'] for r in fits])),epoch_cap_hits=sum(r['epochs']==400 for r in fits))
        if v in ('kan','sparse_kan'):
            d=[a for a in details if a['bank']==bank and a['variant']==v]
            references={(r['repeat'],r['inner']):r['validation_mae'] for r in records if r['bank']==bank and r['variant']=='matched_mlp'}
            item.update(mlp_wins=sum(r['validation_mae']<references[(r['repeat'],r['inner'])] for r in fits),
                pruned_mae=float(np.mean([x['pruned_mae'] for x in d])),symbolic_mae=float(np.mean([x['symbolic_mae'] for x in d])),
                mean_kept_edges=float(np.mean([x['kept_edges'] for x in d])),total_edges=d[0]['edges'],
                symbolic_edge_fraction=sum(x['symbolic_edges'] for x in d)/sum(x['edges'] for x in d),
                symbolic_prediction_shift=float(np.mean([x['symbolic_prediction_shift'] for x in d])))
            if bank=='enhanced':
                base={(r['repeat'],r['inner']):r['validation_mae'] for r in records if r['variant']=='modnet'}
                item['modnet_wins']=sum(r['validation_mae']<base[(r['repeat'],r['inner'])] for r in fits)
        rows.append(item)
    figures=RUN/'figures';figures.mkdir(exist_ok=True)
    ale=plot_ale(details,figures)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for bank,ax in zip(('enhanced','physical'),axes):
        group=[r for r in rows if r['bank']==bank];x=np.arange(len(group))
        for s in range(2):ax.bar(x+(s-.5)*.34,[r['seed_mae'][s] for r in group],.34,label=f'Seed {s+1}')
        ax.set_xticks(x,[r['variant'] for r in group],rotation=15);ax.set_ylabel('Inner-validation MAE (cm$^{-1}$)')
        ax.set_title('256 enhanced features' if bank=='enhanced' else '16 physical descriptors');ax.legend()
    fig.tight_layout();fig.savefig(figures/'comparison.png',dpi=180);fig.savefig(figures/'comparison.pdf');plt.close(fig)
    result=dict(analyzed_at=datetime.now(timezone.utc).isoformat(),analyzer_sha256=sha(__file__),run_fingerprint=fp,
        audited_fits=70,max_saved_mae_difference=mae_difference,rows=rows,network_audits=details,physical_ale_stability=ale,
        outer_test_evaluated=False,wall_minutes=status['wall_seconds']/60,limitations=p['limitations'])
    (RUN/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 完整 KAN 直接替代 MODNet：结果','',
        f"完成并核验 70 次训练，墙钟 {result['wall_minutes']:.2f} 分钟。KAN 无 MODNet 主干、Dense 层和残差支路。全部是内层验证，MAE 单位 cm⁻¹。",'',
        '| 输入组 | 模型 | 种子 1 | 种子 2 | 参数量 | 平均每次秒数 | 达 400 轮上限 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append(f"| {r['bank']} | {r['variant']} | {r['seed_mae'][0]:.3f} | {r['seed_mae'][1]:.3f} | {r['parameters']:,} | {r['mean_seconds']:.1f} | {r['epoch_cap_hits']}/10 |")
    lines+=['','同一输入组内 KAN/MLP 仅差 1 个参数。两种输入组的信息不同，不能将组间误差变化都解释为结构效果。正则版是固定消融，不能以两个 KAN 中的最好结果获得额外未披露的搜索优势。','',
        '| 输入组 | KAN | 原始 MAE（两种子均值） | 剪枝后 | 简式替换后 | 保留边数 | 简式边比例 | 胜配对 MLP |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        if 'pruned_mae' in r:lines.append(f"| {r['bank']} | {r['variant']} | {r['mean_mae']:.3f} | {r['pruned_mae']:.3f} | {r['symbolic_mae']:.3f} | {r['mean_kept_edges']:.1f}/{r['total_edges']} | {r['symbolic_edge_fraction']:.1%} | {r['mlp_wins']}/10 |")
    lines+=['','剪枝只按训练边响应标准差作决定，并把删除边的训练均值补入节点偏置；没有重新训练，也没有实际缩减权重张量。简式只按训练边拟合误差选择，随后从原输入运行整个替换网络。未通过门槛的边保留样条，不能将部分成功称为完整解析公式。','',
        '![结果对比](figures/comparison.png)','',
        '物理输入组的 `physical-ale-*.png` 展示原始物理量对完整模型输出的一阶累积局部效应（ALE），而非某个隐藏节点的任意坐标曲线。曲线使用训练数据条件分箱，不构成因果效应，也不是精确的加性预测分解；相关特征和未展示的高阶交互仍然影响解释。','',
        '每次 KAN 的 `network.json` 保存完整函数网络；`pruned-network.json` 保存偏置补偿后的掩码模型；`symbolic-network.json` 保存各边多项式次数、标准化坐标和系数；`interpretation.json` 保存整个网络的替换误差。隐藏坐标经过逐维 tanh 后进入下一层边函数，不能命名为力常数或有效质量。','',
        '增强输入含有预训练嵌入。可以审计完整 KAN 的函数图，但不能仅凭它由 KAN 构成就声称它是一条可读物理公式。当前固定网格、固定学习率和 400 轮预算的负结果也不能证明 KAN 类别没有可用配置。','',
        '未评估外层测试；所有结论是已反复使用内层验证的探索。训练时间为并行 worker 内每次累计耗时，不是隔离的速度基准。']
    (RUN/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(audited_fits=70,wall_minutes=result['wall_minutes'],rows=rows),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
