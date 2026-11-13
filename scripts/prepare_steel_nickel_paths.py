"""Recompute the frozen models' inputs for the same lawful Ni-equivalent changes in both arms."""
import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"benchmarks/steel-equal-nodes-20261112"
ORIGINAL = ROOT/"benchmarks/steels-paper-reproduction-20261102/mm-roost396"
AUDIT = ROOT/"benchmarks/steel396-physical-concepts-20261112"
GITDIR = ORIGINAL.parent/"source-search-20261107/mattervial-history.git"
MATTERVIAL = "f58652097cebe96f383f17aa6798f4364e99564e"
ROUTES = ("original","nickel","carbon")


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_grid(recipes):
    columns = {key:[] for key in ROUTES}
    actual = []
    for text in recipes:
        comp = Composition(text)
        columns["original"].append(comp)
        original = comp.get_el_amt_dict()
        # Keep the source formula's amount scale; swap exactly one gram in 100 g for the Ni path.
        for name,el,step in (("nickel","Ni",1.),("carbon","C",1/30)):
            amounts = original.copy()
            mass = float(comp.weight)*step/100
            amounts["Fe"] -= mass/float(Element("Fe").atomic_mass)
            amounts[el] = amounts.get(el,0)+mass/float(Element(el).atomic_mass)
            assert amounts["Fe"]>0
            edited = Composition(amounts)
            np.testing.assert_allclose(float(comp.weight),float(edited.weight),rtol=1e-12)
            columns[name].append(edited)
        score = [100*(c.get_wt_fraction("Ni")+30*c.get_wt_fraction("C")+.5*c.get_wt_fraction("Mn"))
                 for c in [columns[k][-1] for k in ROUTES]]
        np.testing.assert_allclose(np.subtract(score[1:],score[0]),1,atol=1e-10,rtol=0)
        actual.append(score)
    return columns,np.array(actual)


def packing_rows(featurizer,recipes):
    """Same multiset of efficient clusters as matminer, queried in the populated element dimensions."""
    from scipy.spatial import cKDTree
    by_elements = {}
    all_comp = [comp for route in ROUTES for comp in recipes[route]]
    result = [[] for _ in all_comp]
    for i,comp in enumerate(all_comp):
        ranked = sorted(comp.element_composition.items(),key=lambda e:-e[1])
        elements = tuple(sorted(e for e,_ in ranked[:featurizer.max_types]))
        by_elements.setdefault(elements,[]).append(i)
    count_cache = {}
    for elements,indices in by_elements.items():
        radii = np.array(featurizer._data_source.get_elemental_properties(elements,"MiracleRadius"))
        small = featurizer.find_ideal_cluster_size(radii.min()/radii.max())[0]
        large = featurizer.find_ideal_cluster_size(radii.max()/radii.min())[0]
        k = len(elements)
        clusters = []
        for size in range(small,large+1):
            key = (size,k)
            if key not in count_cache:
                bars = np.array(list(itertools.combinations(range(1,size+k),k-1)),dtype=int)
                boundaries = np.column_stack((np.zeros(len(bars),dtype=int),bars,np.full(len(bars),size+k)))
                count_cache[key] = np.diff(boundaries,axis=1)-1
            counts = count_cache[key]
            mean_r = np.einsum("ij,j->i",counts,radii)/size
            ideal = featurizer.get_ideal_radius_ratio(size)
            for j,radius in enumerate(radii):
                fit = counts[np.abs(1-ideal*mean_r/radius)<featurizer.threshold].copy()
                if len(fit):
                    fit[:,j] += 1
                    clusters.append(fit/(size+1))
        tree = cKDTree(np.concatenate(clusters)) if clusters else None
        largest_k = max(featurizer.n_nearest)
        for i in indices:
            comp = all_comp[i]
            fraction = comp.fractional_composition
            kept = np.array([fraction[e] for e in elements])
            tail = sum(value**2 for e,value in fraction.items() if e not in elements)
            distances = np.ones(largest_k)
            if tree is not None:
                n = min(tree.n,largest_k)
                dist = np.atleast_1d(tree.query(kept,k=n)[0])
                distances[:n] = np.sqrt(dist**2+tail)
            result[i] = list(featurizer.compute_simultaneous_packing_efficiency(comp)) + [
                         distances[:n].mean() for n in featurizer.n_nearest]
    # Check the reference implementation on the exact edited examples in the user request.
    for start in range(0,len(all_comp),len(recipes["original"])):
        for i in (start+4,start+19):
            np.testing.assert_allclose(result[i],featurizer.featurize(all_comp[i]),atol=3e-8,rtol=1e-6)
    return dict(zip(ROUTES,np.split(np.array(result),len(ROUTES))))


def matminer_grid(columns, recipes):
    from modnet.featurizers.presets import CompositionOnlyFeaturizer
    available = {type(f).__name__:f for f in CompositionOnlyFeaturizer().composition_featurizers}
    output = {}
    metadata = {}
    for group in dict.fromkeys(c.split("|")[0] for c in columns):
        featurizer = available[group]
        chosen = [c for c in columns if c.split("|")[0]==group]
        indices = [featurizer.feature_labels().index(c.split("|",1)[1]) for c in chosen]
        if group=="AtomicPackingEfficiency":
            packing = packing_rows(featurizer,recipes)
            for route,values in packing.items():
                for c,series in zip(chosen,values[:,indices].T):
                    output[route,c] = series
            print("matminer",group,flush=True)
            continue
        for route in ROUTES:
            rows = []
            for comp in recipes[route]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    values = featurizer.featurize(comp)
                record = []
                for i,name in zip(indices,chosen):
                    v = values[i]
                    if group=="AtomicOrbitals" and name.endswith("_character"):
                        v = {"s":1,"p":2,"d":3,"f":4}.get(v,0)
                    elif group=="AtomicOrbitals" and name.endswith("_element"):
                        v = Element(v).Z if isinstance(v,str) else -1
                    record.append(float(v) if np.isfinite(v) else 0.)
                rows.append(record)
            for c,series in zip(chosen,np.array(rows).T):
                output[route,c] = series
        print("matminer",group,flush=True)
    return {r:np.column_stack([output[r,c] for c in columns]) for r in ROUTES},metadata


def roost_grid(columns, recipes):
    import torch
    torch.set_num_threads(2)
    # Load only the upstream inference modules and the two pinned official model files.
    subfiles = ["featurizers/roost_models.py","packages/roost/core.py","packages/roost/segments.py",
                "packages/roost/roost/data.py","packages/roost/roost/model.py",
                "featurizers/custom_models/matscholar-embedding.json"]
    subfiles += [f"featurizers/custom_models/{k}_checkpoint-r0.pth.tar" for k in ("mpgap","oqmd_eform")]
    with tempfile.TemporaryDirectory() as temp:
        for sub in subfiles:
            relative = "mattervial/"+sub
            path = Path(temp)/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("wb") as out:
                subprocess.run(["git","--git-dir="+str(GITDIR),"show",MATTERVIAL+":"+relative],stdout=out,check=True)
        sys.path.insert(0,temp)
        from mattervial.featurizers.roost_models import get_RoostFeatures
        comps = pd.Series([c for r in ROUTES for c in recipes[r]],index=[f"steel-{i}" for i in range(3*len(recipes["original"]))])
        numeric = [(np.core.multiarray.scalar,"numpy.core.multiarray.scalar"),np.dtype,
                   type(np.dtype("float64")),type(np.dtype("float32")),type(np.dtype("int64"))]
        with torch.serialization.safe_globals(numeric), warnings.catch_warnings():
            warnings.simplefilter("ignore",FutureWarning)
            frames = [get_RoostFeatures(comps,model_type=k,batch_size=32) for k in ("mpgap","oqmd_eform")]
        values = pd.concat(frames,axis=1).loc[comps.index,columns].values
    return dict(zip(ROUTES,np.split(values,3))),dict(commit=MATTERVIAL)


def run(part,out):
    manifest = json.loads((ORIGINAL/"manifest.json").read_text())
    source = ROOT/manifest["csv"]
    assert hash_file(source)==manifest["csv_sha256"]
    selection = set()
    for f in range(5):
        for arm in ("kan","mlp"):
            meta = json.loads((AUDIT/f"fold-{f}/{arm}.json").read_text())
            selection.update(c for m in meta["members"] for c in m["input_features"])
    with np.load(ORIGINAL/"inputs.npz") as data:
        ids = data["sample_ids"].astype(str)
        all_cols = list(data["columns"])
        columns = [c for c in all_cols if c in selection and c.startswith("ROOST_")==(part=="roost")]
        expected = data["X"][:,[all_cols.index(c) for c in columns]]
    original = pd.read_csv(source,usecols=["material_id","composition"]).set_index("material_id").loc[ids,"composition"]
    recipes,points = physical_grid(original)
    grid,meta = (matminer_grid if part=="mm" else roost_grid)(columns,recipes)
    delta = np.abs(grid["original"]-expected)
    maxima = delta.max(0)
    assert np.isfinite(delta).all()
    # Never silently substitute frozen learned descriptors for unverified recomputations.
    out.mkdir(parents=True,exist_ok=True)
    path = out/f"features-{part}.npz"
    np.savez_compressed(path,ids=ids,columns=columns,q=points,**grid)
    record = dict(code_sha256=hash_file(__file__),arrays_sha256=hash_file(path),csv_sha256=hash_file(source),
                  original_max_by_column=dict(zip(columns,maxima.tolist())),**meta)
    path.with_suffix(".json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print("largest discrepancies",[(columns[i],float(maxima[i])) for i in np.argsort(maxima)[-10:]],flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part",choices=("mm","roost"))
    parser.add_argument("--output",type=Path,default=OUT)
    args = parser.parse_args()
    run(args.part,args.output)
