"""Coefficient-order control for the named physical inputs in the turning-point audit."""
import argparse
from collections import Counter
from pathlib import Path

import audit_kan_turning_points as audit
import discover_physical_kan_responses as exp
from run_modnet_official_matched import file_hash, np, write_json


REPEATS=200


def named_input(task,folder):
    counts=Counter()
    for f in range(5):
        for m in range(50):
            for e in exp.steel.read(folder/task/f"fold-{f}/member-{m:02d}.json")["edges"]:
                if e["turn"] is not None and e["turn"]["relative"]>=audit.PROMINENCE:
                    counts[e["name"]]+=1
    radius="ElementProperty|MagpieData minimum CovalentRadius"
    if task=="phonons" and radius in counts:
        return radius
    candidates=[n for n in counts if n.startswith("ElementFraction|")] if task=="steel" else list(counts)
    return sorted(candidates or counts,key=lambda n:(-counts[n],n))[0] if counts else None


def result_stats(values):
    if not values:
        return dict(count=0,median=None,iqr_width=None)
    low,med,high=np.quantile(values,[.25,.5,.75])
    return dict(count=len(values),median=med,iqr_width=high-low)


def run(task,folder):
    name=named_input(task,folder)
    data=exp.load_data(task,folder/task/"data-check-0")
    observed,randomized,parents=[],[[] for _ in range(REPEATS)],[]
    for f in range(5):
        sources,_,_,_=exp.load_fold(task,f,data)
        for m in range(50):
            tag=f"{task}/fold-{f}/member-{m:02d}.json"
            meta=exp.steel.read(exp.OUT/tag)
            if name not in meta["inputs"]:
                continue
            record=exp.steel.read(folder/tag)
            assert record["source_sha256"]==file_hash(exp.OUT/tag)
            assert record["audit_sha256"]==file_hash(audit.__file__)
            parents.append(file_hash(folder/tag))
            i=meta["inputs"].index(name)
            raw=data[0].loc[sources[0][m][3],name].values
            recipe=meta["kan"]["source_recipe"]
            edge=audit.PhysicalEdge(recipe,i)
            _,actual=audit.detect(edge,raw)
            saved=record["edges"][i]["turn"]
            assert (saved is None)==(actual is None)
            if actual is not None:
                np.testing.assert_allclose(actual["root"],saved["root"],atol=1e-9)
                if actual["relative"]>=audit.PROMINENCE:
                    observed.append(actual["root"])
            coefficients=np.array(recipe["coefficients"])
            altered=coefficients.copy()
            rng=np.random.RandomState(151126+1000*f+m)
            for rep in range(REPEATS):
                altered[i]=rng.permutation(coefficients[i])
                shuffled=audit.PhysicalEdge(dict(recipe,coefficients=altered),i)
                _,turn=audit.detect(shuffled,raw)
                if turn is not None and turn["relative"]>=audit.PROMINENCE:
                    randomized[rep].append(turn["root"])
    sims=[result_stats(v) for v in randomized]
    quantiles={}
    for key in ("count","median","iqr_width"):
        values=[s[key] for s in sims if s[key] is not None]
        quantiles[key]=np.quantile(values,[.05,.5,.95]) if values else None
    write_json(folder/task/"coefficient-order-control.json",dict(name=name,selected=len(parents),repeats=REPEATS,
        description="Independently permute the eight trained coefficients within each selected edge. Keep input, training support, scaler, knots, node and coefficient multiset fixed. Reapply identical detection. q05/q50/q95 across replications are descriptive; the permutation also changes smoothness.",
        observed=result_stats(observed),random_quantiles=quantiles,replicates=sims,parents_sha256=parents,
        code_sha256=file_hash(__file__),audit_sha256=file_hash(audit.__file__)))
    print(task,name,len(parents),result_stats(observed),quantiles,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=("steel","phonons"),required=True)
    parser.add_argument("--output",type=Path,default=audit.OUT)
    args=parser.parse_args()
    run(args.task,args.output)
