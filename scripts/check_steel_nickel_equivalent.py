"""Check the nickel-equivalent interpretation against elemental Ni in actual held-out steels."""
import argparse
from pathlib import Path

import audit_steel_physical_strength as study
import discover_physical_kan_responses as exp
from pymatgen.core import Element
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from run_modnet_official_matched import file_hash, np, write_json


def predict_ols(columns,y,tr,te):
    x=np.column_stack([np.ones(len(y)),*columns])
    w,_,rank,_=np.linalg.lstsq(x[tr],y[tr],rcond=None)
    assert rank==x.shape[1]
    return dict(coefficients=w,prediction=x[te]@w,mae=np.mean(abs(x[te]@w-y[te])))


def run(out):
    frame,y,folds=exp.load_data("steel",out)
    q,composition=study.chemistry(frame)
    raw=frame.filter(regex=r"^ElementFraction\|")
    raw=raw.loc[:,raw.ne(0).any()].copy()
    raw.columns=[e.split("|")[1] for e in raw.columns]
    kg=raw*[float(Element(e).atomic_mass) for e in raw.columns]
    wt=100*kg.div(kg.sum(axis=1),axis=0)
    ni=wt.Ni.values
    add=(30*wt.C+.5*wt.Mn).values
    ni_eq=q["ni_equivalent"].values
    cr=q["cr_equivalent"].values
    np.testing.assert_allclose(ni+add,ni_eq,atol=1e-11)
    data=[]
    for f,(tr,te) in enumerate(folds):
        baseline=exp.steel.read(out/f"fold-{f}/design.json")["pairs"]["ni_equivalent"]
        low_cut,high_cut=baseline["terciles"]
        lo=te[ni_eq[te]<=low_cut]
        hi=te[(ni_eq[te]>=high_cut)&(ni_eq[te]>low_cut)]
        distance=cdist(composition[lo],composition[hi])
        valid=((ni_eq[hi][None,:]-ni_eq[lo][:,None]>=.5*baseline["iqr"]) &
               (abs(ni[hi][None,:]-ni[lo][:,None])<=1.) & (distance<=baseline["caliper"]))
        il,ir=linear_sum_assignment(np.where(valid,distance,1000.))
        pairs=[]
        for a,b in zip(il,ir):
            if valid[a,b]:
                l,h=lo[a],hi[b]
                pairs.append(dict(low=frame.index[l],high=frame.index[h],ni_low=ni[l],ni_high=ni[h],
                                  q_low=ni_eq[l],q_high=ni_eq[h],distance=distance[a,b],observed=y.iloc[h]-y.iloc[l]))
        xsets={"ni_only":[ni],"ni_eq_only":[ni_eq],"ni_plus_carbon_manganese":[ni,add],
               "ni_plus_cr_eq":[ni,cr],"ni_plus_cr_eq_plus_carbon_manganese":[ni,cr,add]}
        ols={name:predict_ols(cols,y.values,tr,te) for name,cols in xsets.items()}
        # Partial out actual Ni with a training calibration; test residual strength correlation.
        x=np.column_stack([np.ones(len(y)),ni])
        joined=np.column_stack([y,add])
        residual=joined[te]-x[te]@np.linalg.lstsq(x[tr],joined[tr],rcond=None)[0]
        data.append(dict(fold=f,ids=frame.index[te].tolist(),target=y.iloc[te].values,
                    weights=wt.iloc[te][["Ni","C","Mn"]].values,ni_eq=ni_eq[te],fixed_ni_pairs=pairs,
                    partial_correlation_given_ni=np.corrcoef(residual.T)[0,1],ols=ols))
    write_json(out/"nickel-composition-check.json",dict(code_sha256=file_hash(__file__),
               design="Exploratory follow-up to four specified composition coordinates. Repeat the same disjoint real-pair match with absolute Ni mass percent difference <=1 point. All OLS coefficients trained on full outer-training materials only. Real Ni, carbon, manganese and Cr equivalent as indicated.",
               folds=data))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,default=study.OUT)
    run(cli.parse_args().output)
