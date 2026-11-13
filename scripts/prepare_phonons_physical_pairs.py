"""Build a real-crystal pair catalogue and fit small readouts to frozen phonon ensembles."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import json

import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import analyze_formal_phonons_concepts as audit
from native_fourway_features import ORIGINAL, MODData
from run_modnet_official_matched import file_hash, write_json
from steel_three_element_readout import feature_blocks, fit_three


OUT = audit.ROOT / 'benchmarks/phonons-all-input-paths-20261113'
PROXY = audit.ROOT / 'benchmarks/native-physical-proxy-20261101'
ARMS = ['kan', 'mlp']
EXTENDED = ['shortest_distance', 'inverse_sqrt_lightest_mass']
PROTOTYPES = {225: ('rocksalt', 6, 1.), 216: ('zincblende', 4, np.sqrt(3)/2),
              221: ('cesium_chloride', 8, np.sqrt(3)/2 * 2**(1/3))}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def record(structure, raw, fold, official, label):
    c = structure.composition.element_composition
    elems = sorted(el.symbol for el in c)
    vpa = structure.volume / c.num_atoms
    assert structure.is_ordered
    # This includes periodic images of the same site, including in one-site cells.
    radius = 3.
    while True:
        a, b, _, distances = structure.get_neighbor_list(radius)
        if len(distances):
            break
        radius *= 2
    short = float(min(distances))
    light = min(float(el.atomic_mass) for el in c)
    mass = float(c.weight) / c.num_atoms
    formula_key = tuple((el, round(c.get_atomic_fraction(el), 9)) for el in elems)
    sg = None
    proto = None
    if len(elems) == 2 and abs(c.get_atomic_fraction(elems[0])-.5) < 1e-9:
        sg = SpacegroupAnalyzer(structure, symprec=.01).get_space_group_number()
        if sg in PROTOTYPES:
            name, count, ratio = PROTOTYPES[sg]
            if radius < short*1.001:
                a, b, _, distances = structure.get_neighbor_list(short*1.002)
            shell = distances <= short*1.001
            if (np.all(np.bincount(a[shell], minlength=len(structure)) == count)
                    and all(structure[i].specie.symbol != structure[j].specie.symbol for i,j in zip(a[shell],b[shell]))
                    and abs(short/vpa**(1/3)/ratio-1) < .001):
                proto = name
    return dict(id=raw, official=official, fold=fold, formula=c.reduced_formula, chemistry=formula_key,
                prototype=proto, sg=sg, elements=elems, mass=mass, lightest_mass=light,
                density=float(structure.density), volume_per_atom=vpa, shortest_distance=short,
                groups={el.symbol:int(el.group) for el in c}, weights={el.symbol:float(el.atomic_mass) for el in c},
                target=float(label))


def matched_pairs(records):
    pairs = []
    for f in range(5):
        for a,b in combinations([r for r in records if r['fold']==f],2):
            if a['chemistry']==b['chemistry']:
                a,b = sorted([a,b],key=lambda r:(r['shortest_distance'],r['official']))
                pairs.append(dict(fold=f,kind='same_composition',lo=a['id'],hi=b['id']))
            elif a['prototype'] and a['prototype']==b['prototype']:
                compa,compb = set(a['elements']),set(b['elements'])
                if len(compa & compb)!=1:
                    continue
                suba,subb = next(iter(compa-compb)),next(iter(compb-compa))
                if a['weights'][suba]>b['weights'][subb]:
                    a,b,suba,subb = b,a,subb,suba
                pairs.append(dict(fold=f,kind='substitution',prototype=a['prototype'],lo=a['id'],hi=b['id'],
                    common=next(iter(compa & compb)),sub_a=suba,sub_b=subb,same_group=a['groups'][suba]==b['groups'][subb]))
    return pairs


def scores(truth, pred):
    return dict(r2=float(1-np.sum((truth-pred)**2)/np.sum((truth-np.mean(truth))**2)),
                mae=float(np.mean(abs(truth-pred))))


def two_formulas(q,names,source,tr,reference):
    blocks,specs=feature_blocks(q,tr)
    x=np.column_stack(blocks); mu=source[tr].mean(axis=0)
    beta=np.linalg.solve(x[tr].T@x[tr]+.005*len(tr)*np.eye(x.shape[1]), x[tr].T@(source[tr]-mu))
    np.testing.assert_allclose(x@beta+mu,reference,atol=1e-7,rtol=1e-10)
    formulas=[]
    for a in range(source.shape[1]):
        bias=mu[a]; pos=0; variables=[]
        for name,spec in zip(names,specs):
            end=pos+len(spec['mean']); coef=beta[pos:end,a]/spec['sd']; pos=end
            bias-=coef@spec['mean']; variables.append(dict(name=name,lo=spec['lo'],hi=spec['hi'],knots=spec['knots'],coef=coef))
        formulas.append(dict(intercept=bias,variables=variables))
    return formulas


def main():
    mod = MODData.load(ORIGINAL)
    frame = mod.df_featurized
    ids = list(frame.index)
    assert ids==list(mod.df_structure.index)==list(mod.df_targets.index)
    splits = [read(audit.engine.OLD/'matbench_phonons'/f'fold-{f}/outer-split.json') for f in range(5)]
    mapping = {i:(f,j) for f,s in enumerate(splits) for i,j in zip(s['test_ids'],s['official_test_ids'])}
    assert len(mapping)==len(set(j for f,j in mapping.values()))==len(ids)==1265
    target = mod.df_targets.iloc[:,0].values.astype(float)
    rows = [record(st,i,*mapping[i],y) for i,st,y in zip(ids,mod.df_structure.iloc[:,0],target)]
    q = audit.concepts(frame)
    qnames = audit.NAMES + EXTENDED
    q = np.column_stack([q,[r['shortest_distance'] for r in rows], [r['lightest_mass']**-.5 for r in rows]])
    assert np.isfinite(q).all() and np.all(q[:,3]>0)
    deviations = {}
    for name,field in [('mean_mass','mass'),('volume_per_atom','volume_per_atom'),('density','density')]:
        vals = np.array([r[field] for r in rows]); stored=q[:,qnames.index(name)]
        deviations[name] = float(max(abs(vals-stored)/vals))
        np.testing.assert_allclose(stored,vals,rtol=.0015)
    pairs = matched_pairs(rows)
    write_json(OUT/'catalogue.json',dict(rows=rows,pairs=pairs,quantities=qnames,structure_source=str(ORIGINAL.relative_to(audit.ROOT)),
               structure_source_sha256=file_hash(ORIGINAL),max_relative_difference_vs_archived=deviations))
    print('catalogue', Counter(r['prototype'] for r in rows), Counter((p['fold'],p['kind']) for p in pairs), flush=True)
    folds = []
    full = np.full((len(ids),2),np.nan); predictions = np.full((2,3,len(ids),2),np.nan)
    for f,s in enumerate(splits):
        tr = frame.index.get_indexer(s['train_ids']); te = frame.index.get_indexer(s['test_ids'])
        source = np.full((len(ids),2),np.nan)
        for a,suffix in enumerate(['kan','modnet']):
            prox = PROXY/f'mm_mv_{suffix}'/f'fold-{f}'
            orig = audit.RUN/f'mm_mv_{suffix}'/f'fold-{f}'
            assert file_hash(orig/'outer-predictions.npz')==read(orig/'complete.json')['prediction_sha256']
            assert file_hash(prox/'teacher-training-predictions.npz')==read(prox/'frozen-formula.json')['teacher_predictions_sha256']
            with np.load(prox/'teacher-training-predictions.npz') as z:
                assert set(z['sample_ids'])==set(s['train_ids'])
                source[frame.index.get_indexer(z['sample_ids']),a]=z['prediction']
            with np.load(orig/'outer-predictions.npz') as z:
                np.testing.assert_array_equal(z['sample_ids'],s['official_test_ids'])
                np.testing.assert_allclose(z['target'],target[te],rtol=0,atol=1e-8)
                source[te,a]=z['members'].astype(float).mean(0)
        assert np.isfinite(source).all(); full[te]=source[te]
        designs = {}
        for k,(design,names) in enumerate([('original',audit.NAMES),('extended',qnames)]):
            sub = q[:,:len(names)]
            chosen,formulas,stages=fit_three(sub,names,source,tr)
            predictions[k,:,te,:]=np.stack([v[te] for v in stages],axis=1)
            own=[fit_three(sub,names,source[:,a:a+1],tr)[0] for a in range(2)]
            designs[design]=dict(chosen=chosen,independently_chosen=own,formulas=formulas,
                         two_formulas=two_formulas(sub[:,[names.index(name) for name in chosen[:2]]],chosen[:2],source,tr,stages[1]),
                         fold_scores=[[scores(source[te,a],v[te,a]) for a in range(2)] for v in stages])
        folds.append(dict(fold=f,designs=designs))
    metrics = {}
    for k,name in enumerate(['original','extended']):
        metrics[name]=[dict(teacher=[scores(full[:,a],p[:,a]) for a in range(2)],
                            dfpt=[scores(target,p[:,a]) for a in range(2)]) for p in predictions[k]]
    assert np.isfinite(predictions).all() and np.isfinite(full).all()
    np.savez_compressed(OUT/'readout.npz',ids=ids,quantities=qnames,q=q,target=target,teacher=full,readouts=predictions)
    write_json(OUT/'readout.json',dict(folds=folds,pooled=metrics,original_models=[scores(target,x) for x in full.T],
        arms=ARMS,arrays_sha256=file_hash(OUT/'readout.npz'),code_sha256=file_hash(__file__)))
    print('readout', [(v['designs']['extended']['chosen']) for v in folds],metrics['extended'],flush=True)


if __name__=='__main__':
    main()
