"""Validate and summarize the physical readout, native paths and fair hidden-node probes."""
from collections import Counter
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from probe_phonons_two_quantities import formula_terms


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/phonons-all-input-paths-20261113'
ARMS=('kan','mlp')
COMPOSITION={'AtomicOrbitals','AtomicPackingEfficiency','BandCenter','ElementFraction','ElementProperty',
             'IonProperty','Miedema','Stoichiometry','TMetalFraction','ValenceOrbital','YangSolidSolution',
             'ElectronegativityDiff','OxidationStates'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    return dict(r2=1-((y-p)**2).sum()/((y-y.mean())**2).sum(),mae=np.mean(abs(y-p)))


def batch_r2(y,p):
    return 1-((y-p)**2).sum(axis=-2)/((y-y.mean(axis=-2,keepdims=True))**2).sum(axis=-2)


def family(name):
    return name.split('|')[0] if '|' in name else name.split('_')[0]


def group(n):
    fam=family(n)
    if (fam in {'AverageBondLength','RadialDistributionFunction','GeneralizedRDF','GaussianSymmFunc'}
            or n=='DensityFeatures|vpa' or (fam=='AGNIFingerPrint' and 'AGNI eta=' in n)
            or (fam=='VoronoiFingerprint' and any(k in n for k in ('Voro_vol','Voro_area','Voro_dist')))):
        return 'distance_geometry'
    if fam=='ElementProperty' and n.endswith(' AtomicWeight'):
        return 'elemental_masses'
    if fam in COMPOSITION: return 'other_composition'
    if fam in {'ORB','ROOST','MEGNet','MVL16','MVL32'}: return 'learned_embeddings'
    return 'other_descriptors'


def selection_stats(pairs):
    result={}
    for a in ARMS:
        term=[p['models'][a] for p in pairs]
        y=np.array([t['delta'] for t in term]); pred=np.array([t['formula_delta'] for t in term]); ref=np.array([p['dfpt_delta'] for p in pairs]);
        result[a]=dict(pairs=len(y),two_quantity_vs_teacher=stats(y,pred),two_quantity_vs_dfpt=stats(ref,pred),
            two_quantity_half_error_count=int(np.sum(abs(y-pred)<=.5*abs(y))),
            geometry_half_error_count=int(np.sum(abs(y-[t['groups']['distance_geometry'] for t in term])<=.5*abs(y))),
            net_over_absolute_median=np.median([t['net_over_absolute'] for t in term]),
            absolute_over_net_median=np.median([1/t['net_over_absolute'] for t in term]),
            net_over_family_absolute_median=np.median([t['net_over_family_absolute'] for t in term]),
            model_vs_dfpt=stats(ref,y))
    for k,op in [('net_over_absolute',np.greater),('net_over_family_absolute',np.greater),('gross',np.less)]:
        result['kan_'+k+'_wins']=sum(bool(op(p['models']['kan'][k],p['models']['mlp'][k])) for p in pairs)
    return result


def analyse():
    data=read(OUT/'catalogue.json'); ro=read(OUT/'readout.json')
    assert digest(OUT/'readout.npz')==ro['arrays_sha256']
    rows={v['id']:v for v in data['rows']}
    with np.load(OUT/'readout.npz') as z:
        ids=z['ids'].tolist(); names=z['quantities'].tolist(); q=z['q'][:,-2:]
        assert names[-2:]==['shortest_distance','inverse_sqrt_lightest_mass']
        target=z['target']; direct=z['readouts'][1,1]; teacher=z['teacher']; all_stages=z['readouts']
    pos={v:i for i,v in enumerate(ids)}; assert len(pos)==len(rows)==1265
    estimated=np.full((2,2,len(q),2),np.nan); represented=np.full((2,2,len(q)),np.nan); ctrl=np.full((2,len(q),2),np.nan)
    fold_stats=[]; output=[]; matched=[]; seen=set()
    for f in range(5):
        d=OUT/f'fold-{f}'; t=read(d/'trace.json'); ch=read(d/'channels.json')
        assert t['pairs_sha256']==digest(OUT/'catalogue.json') and ch['readout_sha256']==digest(OUT/'readout.json')
        assert t['held']==ch['held'] and not seen.intersection(t['held']); seen.update(t['held'])
        test=np.array([pos[i] for i in t['held']]); pp=t['pairs']; cols=t['columns']; assert pp==[p for p in data['pairs'] if p['fold']==f]
        left=np.array([pos[p['lo']] for p in pp]); right=np.array([pos[p['hi']] for p in pp])
        families=np.array([family(c) for c in cols]); label=np.array([group(c) for c in cols]); fam_mat=np.equal(families[:,None],np.unique(families)[None,:]).astype(float)
        select=[t['arms'][a]['selected'] for a in ARMS]; exact=np.array([k for k,(u,v) in enumerate(zip(*select)) if u==v],dtype=int)
        pairs=[dict(**p,low=rows[p['lo']]['official'],high=rows[p['hi']]['official'],
                    dfpt_delta=target[b]-target[a],distance_delta=q[b,0]-q[a,0],inverse_mass_delta=q[b,1]-q[a,1],
                    models={}) for p,a,b in zip(pp,left,right)]
        fs=dict(fold=f,arms={},identical_input_pairs=len(exact)); subseq=[j for j,p in enumerate(pp) if p['kind']=='substitution']
        compare=[]
        for ai,a in enumerate(ARMS):
            ar=t['arms'][a]; assert ar['sha256']==digest(d/(a+'.npz')) and ch['arms'][a]['sha256']==digest(d/f'{a}-channels.npz')
            with np.load(d/(a+'.npz')) as z:
                terms=z['terms']; stored=z['archived_members'].astype(float)
            assert terms.shape==(50,len(pp),len(cols)); np.testing.assert_allclose(stored.mean(0),teacher[test,ai],atol=1e-9)
            v=terms.mean(0); delta=v.sum(1); gross=abs(v).sum(1); fam_term=v@fam_mat; fg=abs(fam_term).sum(1)
            np.testing.assert_allclose(fam_term.sum(1),delta,atol=1e-8)
            np.testing.assert_allclose(delta,teacher[right,ai]-teacher[left,ai],atol=.012,rtol=1e-6)
            formula=ro['folds'][f]['designs']['extended']['two_formulas'][ai]
            physical=formula_terms(q[right],formula)-formula_terms(q[left],formula)
            np.testing.assert_allclose(physical.sum(1),direct[right,ai]-direct[left,ai],atol=1e-7)
            if len(exact):
                members=terms[exact][:,subseq]; compare.append(abs(members.sum(2))/abs(members).sum(2))
            for j,p in enumerate(pairs):
                groups={g:v[j,label==g].sum() for g in np.unique(label)}
                np.testing.assert_allclose(sum(groups.values()),delta[j],atol=1e-8)
                p['models'][a]=dict(delta=delta[j],formula_delta=physical[j].sum(),formula_terms=physical[j],groups=groups,
                            positive=v[j,v[j]>0].sum(),negative=v[j,v[j]<0].sum(),gross=gross[j],net_over_absolute=abs(delta[j])/gross[j],
                            net_over_family_absolute=abs(delta[j])/fg[j],geometry_absolute_input_share=abs(v[j,label=='distance_geometry']).sum()/gross[j])
            with np.load(d/f'{a}-channels.npz') as z:
                decode=z['decoded']; pred=z['physical_outputs']; base=z['output_only_decoded']
            estimated[ai][:,test]=decode.mean(0); represented[ai][:,test]=pred.mean(0); ctrl[ai,test]=base.mean(0)
            fs['arms'][a]=dict(decoding_median_member_r2=np.median(batch_r2(q[test],decode),axis=0),
                 two_quantities_vs_teacher=stats(teacher[test,ai],direct[test,ai]),
                 hidden_vs_teacher=[stats(teacher[test,ai],p) for p in pred.mean(0)],
                 selected_width_range=[min(map(len,select[ai])),max(map(len,select[ai]))],
                 common_input_fraction=np.mean([len(set(u).intersection(v))/len(set(u).union(v)) for u,v in zip(*select)]),
                 replay_max=ar['replay_max'],closure_max=ar['member_closure_max'],official_delta_max_error=ar['archive_ensemble_delta_max_error'])
        if len(exact): matched.append((f,compare))
        fold_stats.append(fs); output.extend(pairs)
    assert len(seen)==len(q) and np.isfinite(estimated).all() and np.isfinite(represented).all()
    subs=[p for p in output if p['kind']=='substitution']; same=[p for p in output if p['kind']=='same_composition']
    invariant=[p for p in subs if p['same_group'] and abs(p['inverse_mass_delta'])<1e-12]
    case=max(invariant,key=lambda p:abs(p['dfpt_delta']))
    disjoint=[]; used=set()
    for p in subs:
        if {p['lo'],p['hi']}.isdisjoint(used):
            disjoint.append(p); used.update((p['lo'],p['hi']))
    cohorts=dict(substitution=subs,same_composition=same,same_group=[p for p in subs if p['same_group']],same_lightest_isovalent=invariant,
             disjoint_substitution=disjoint,large_agreement=[p for p in subs if abs(p['dfpt_delta'])>60 and all(abs(v['delta']-p['dfpt_delta'])<=.5*abs(p['dfpt_delta']) for v in p['models'].values())])
    cohort={k:dict(unique_crystals=len({p[i] for p in pp for i in ('lo','hi')}),stats=selection_stats(pp),
            per_fold=[len([p for p in pp if p['fold']==f]) for f in range(5)]) for k,pp in cohorts.items()}
    for f in range(5): fold_stats[f]['substitutions']=selection_stats([p for p in subs if p['fold']==f])
    light_signs=dict(dfpt=sum(p['dfpt_delta']*p['distance_delta']<0 for p in invariant),
                     **{a:sum(p['models'][a]['delta']*p['distance_delta']<0 for p in invariant) for a in ARMS})
    global_scores={}; h=np.array(['H' in rows[i]['elements'] for i in ids]); keep={'all':np.ones(len(q),bool),'without_H':~h,'with_H':h,'dfpt_at_most_1000':target<=1000}
    for k,mask in keep.items():
        global_scores[k]=dict(n=int(mask.sum()),arms={a:dict(physical_vs_teacher=stats(teacher[mask,i],direct[mask,i]),
            physical_vs_dfpt=stats(target[mask],direct[mask,i]),hidden_vs_teacher=[stats(teacher[mask,i],p[mask]) for p in represented[i]],
            decoded_quantity_r2=batch_r2(q[mask],estimated[i][:,mask]),output_only_decoded_r2=batch_r2(q[mask],ctrl[i,mask])) for i,a in enumerate(ARMS)})
    common=np.concatenate([np.stack(x,axis=0).reshape(2,-1) for _,x in matched],axis=1)
    summary=dict(readout_metrics=global_scores,all_physical_candidates=names,folds=fold_stats,cohorts=cohort,
       same_lightest_isovalent_direction=light_signs,case=dict(pair=case,endpoints=[rows[case['lo']],rows[case['hi']]],
             endpoint_teacher=teacher[[pos[case['lo']],pos[case['hi']]]],endpoint_readout=direct[[pos[case['lo']],pos[case['hi']]]]),
       matched_native_control=dict(matched_members=sum(f['identical_input_pairs'] for f in fold_stats),folds=[f for f,_ in matched],
               member_by_material_pairs=common.shape[1],kan_wins=int(np.sum(common[0]>common[1])),median_net_over_absolute=np.median(common,axis=1),
               fold_median_net_over_absolute=[np.median(x,axis=(1,2)) for _,x in matched]),
       native_replay_max=max(f['arms'][a]['replay_max'] for f in fold_stats for a in ARMS),
       native_closure_max=max(f['arms'][a]['closure_max'] for f in fold_stats for a in ARMS),
       native_official_delta_max_error=max(f['arms'][a]['official_delta_max_error'] for f in fold_stats for a in ARMS),
       readout_sha256=digest(OUT/'readout.json'),catalogue_sha256=digest(OUT/'catalogue.json'),code_sha256=digest(Path(__file__)),pairs=output)
    summary['structure_crosscheck']=check_structures(rows,case)
    return summary


def check_structures(rows,case):
    import pandas as pd
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    archive=ROOT/'benchmarks/mattervial-phonons-20261023/source/extracted/featurized_datasets/matbench_phonons_featurizedMM2020Struct_mattervial.csv'
    source=pd.read_csv(archive,usecols=['material_id','structure','target']).set_index('material_id')
    assert len(source)==len(rows) and source.index.is_unique
    distance_error=0.; primitive=Counter(); sizes={}; checked={}
    for key,r in rows.items():
        v=source.loc[r['official']]; st=Structure.from_dict(json.loads(v['structure'])); comp=st.composition
        assert set(comp.as_dict())==set(r['elements']) and abs(v.target-r['target'])<1e-8
        for el,fraction in r['chemistry']:
            assert abs(comp.get_atomic_fraction(el)-fraction)<1e-8
        np.testing.assert_allclose(st.volume/len(st),r['volume_per_atom'],rtol=1e-8)
        d=r['shortest_distance']; centers,near,_,dist=st.get_neighbor_list(d*1.002)
        assert len(dist) and abs(min(dist)-d)<1e-6
        distance_error=max(distance_error,abs(min(dist)-d))
        if r['prototype']:
            sg=SpacegroupAnalyzer(st,symprec=.01); assert sg.get_space_group_number()==r['sg']
            primitive_size=len(sg.find_primitive()); primitive[primitive_size]+=1
            assert primitive_size==2
            sizes.setdefault(r['prototype'],set()).add(len(st))
        if key in (case['lo'],case['hi']):
            shell=dist<=d*1.001
            checked[key]=dict(sites=len(st),space_group=r['sg'],nearest_coordination=np.bincount(centers[shell]).tolist(),
              nearest_species=sorted({'–'.join(sorted((st[i].specie.symbol,st[j].specie.symbol))) for i,j in zip(centers[shell],near[shell])}),
              nearest_distance=min(dist))
    return dict(second_archive=str(archive.relative_to(ROOT)),sha256=digest(archive),rows=len(rows),max_distance_error=distance_error,
        prototype_primitive_site_counts=dict(primitive),cell_sizes_by_prototype={k:sorted(v) for k,v in sizes.items()},case_endpoints=checked)


def plot(s):
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10.5,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#bbc3cb','text.color':'#253947',
                         'axes.labelcolor':'#253947','xtick.color':'#405462','ytick.color':'#405462'})
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7)); colors=['#126a72','#b25a38']
    fig.subplots_adjust(left=.13,right=.98,bottom=.18,top=.77,wspace=.49)
    ax=axes[0]; ax.set_title('两个物理量对整网预测的解释度',loc='left',fontweight='bold',pad=33)
    y=np.arange(3)[::-1]; labels=['直接用晶体物理量','每个量读 1 个节点','每个量读 4 个节点']; g=s['readout_metrics']['all']['arms']
    for i,a in enumerate(ARMS):
        vals=100*np.array([g[a]['physical_vs_teacher']['r2'],*[v['r2'] for v in g[a]['hidden_vs_teacher']]])
        loc=y+(.17 if i==0 else -.17); ax.barh(loc,vals,height=.26,color=colors[i],label=a.upper())
        for yy,v in zip(loc,vals): ax.text(v-1,yy,f'{v:.1f}%',ha='right',va='center',color='white',fontsize=9)
    ax.set(yticks=y,yticklabels=labels,xlim=(0,100),xlabel='五折合并的留出 R²（%）');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.0),ncol=2,frameon=False,fontsize=9,borderaxespad=0)
    ax=axes[1]; ax.set_title('同原型替换中，输入贡献的抵消程度',loc='left',fontweight='bold',pad=33)
    subs=[p for p in s['pairs'] if p['kind']=='substitution']; case=s['case']['pair']
    ax.plot([0,1],[0,1],color='#98a3a9',linestyle='--',lw=1,zorder=0)
    for f in range(5):
        xx=[p['models']['mlp']['net_over_absolute'] for p in subs if p['fold']==f]; yy=[p['models']['kan']['net_over_absolute'] for p in subs if p['fold']==f]
        ax.scatter(xx,yy,s=32,color=colors[0],alpha=.62,edgecolors='white',linewidths=.45)
    xx=case['models']['mlp']['net_over_absolute']; yy=case['models']['kan']['net_over_absolute'];ax.scatter(xx,yy,s=60,color=colors[1],zorder=3)
    ax.annotate('BN → BAs',(xx,yy),xytext=(xx+.10,yy+.14),va='center',arrowprops={'arrowstyle':'-','color':colors[1]},color=colors[1],fontsize=10)
    ax.set(xlabel='MLP  |净峰差| / 各输入贡献绝对值之和',ylabel='KAN  |净峰差| / 各输入贡献绝对值之和',xlim=(0,1),ylim=(0,1),aspect='equal');ax.grid(alpha=.12)
    fig.savefig(OUT/'phonons-physical-readout.png',dpi=180,facecolor='white'); plt.close(fig)

    from matplotlib.colors import LogNorm
    with np.load(OUT/'readout.npz') as z:
        distance,inv_mass=z['q'][:,-2:].T; y=z['target']; ids=z['ids'].tolist()
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.8,4.7)); fig.subplots_adjust(left=.08,right=.98,bottom=.18,top=.86,wspace=.40)
    ax.set_title('晶体中的最短原子间距与参考峰位',loc='left',fontweight='bold',pad=13)
    pts=ax.scatter(distance,y,c=inv_mass**-2,norm=LogNorm(vmin=1,vmax=145),cmap='viridis_r',s=12,alpha=.65,linewidths=0)
    cb=fig.colorbar(pts,ax=ax,fraction=.045,pad=.02); cb.set_ticks([1,10,100],labels=['1','10','100']);cb.set_label('最轻元素的质量（u）')
    ax.set(xlabel='最短原子间距（Å）',ylabel='DFPT 参考峰位（cm⁻¹）',ylim=(0,3850)); ax.grid(alpha=.12)
    for r,shift in zip(s['case']['endpoints'],[(-.30,380),(.14,330)]):
        j=ids.index(r['id']); x,y0=distance[j],y[j];ax.scatter(x,y0,s=65,facecolors='none',edgecolors='#ab4626',linewidths=1.4)
        ax.annotate(r['formula'],(x,y0),xytext=(x+shift[0],y0+shift[1]),color='#873317',arrowprops={'arrowstyle':'-','color':'#873317'})
    bx.set_title('最轻元素相同的 13 对同族替换',loc='left',fontweight='bold',pad=13)
    pairs=[p for p in s['pairs'] if p.get('same_group') and abs(p['inverse_mass_delta'])<1e-12]
    for p in pairs:
        xx=p['distance_delta']; yy=[p['dfpt_delta'],*[p['models'][a]['delta'] for a in ARMS]]
        bx.plot([xx,xx],[min(yy),max(yy)],color='#b0bbc1',lw=1.2)
    for label,ys,col,mark in [('DFPT',[p['dfpt_delta'] for p in pairs],'#253947','o'),
            ('KAN',[p['models']['kan']['delta'] for p in pairs],colors[0],'^'),('MLP',[p['models']['mlp']['delta'] for p in pairs],colors[1],'s')]:
        bx.scatter([p['distance_delta'] for p in pairs],ys,s=28,c=col,marker=mark,label=label,zorder=3)
    c=s['case']['pair'];bx.annotate('BN → BAs',(c['distance_delta'],c['dfpt_delta']),xytext=(.59,-571),color='#253947',arrowprops={'arrowstyle':'-','color':'#52636b'})
    bx.axhline(0,color='#85939b',lw=.8);bx.grid(alpha=.12);bx.legend(loc='upper right',ncol=3,fontsize=9,frameon=False,handlelength=1)
    bx.set(xlabel='最短原子间距的增量（Å）',ylabel='峰位变化（cm⁻¹）',ylim=(-615,135));fig.savefig(OUT/'phonons-observed-distances.png',dpi=180,facecolor='white');plt.close(fig)


if __name__=='__main__':
    summary=analyse()
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x))
    (OUT/'summary.json').write_text(json.dumps(summary,default=convert,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf8')
    plot(summary)
