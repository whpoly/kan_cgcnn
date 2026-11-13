"""Fit a small, explicit additive readout to paired frozen predictions on real training steels."""
import numpy as np


def feature_blocks(q, train):
    specs, blocks = [], []
    for x in q.T:
        lo,hi = x[train].min(),x[train].max()
        clipped = np.clip(x,lo,hi)
        knots = np.unique(np.quantile(x[train],[.25,.5,.75],method="nearest"))
        knots = knots[(knots>lo) & (knots<hi)]
        terms = np.column_stack([clipped, *(np.maximum(clipped-t,0) for t in knots)])
        mean,sd = terms[train].mean(axis=0),terms[train].std(axis=0)
        assert np.all(sd>1e-8)
        specs.append(dict(lo=float(lo),hi=float(hi),knots=knots.tolist(),mean=mean,sd=sd))
        blocks.append((terms-mean)/sd)
    return blocks,specs


def fit_three(q,names,y,tr):
    """Joint variable selection uses training rows only; each teacher gets its own MPa coefficients."""
    blocks,specs = feature_blocks(q,tr)
    mu,scale = y[tr].mean(axis=0),y[tr].std(axis=0)
    target = (y[tr]-mu)/scale
    chosen = []
    stages = []
    for _ in range(3):
        candidates = []
        for i in range(len(blocks)):
            if i in chosen:
                continue
            x = np.column_stack([blocks[k] for k in chosen+[i]])
            xt = x[tr]
            gram = xt.T@xt + .005*len(tr)*np.eye(xt.shape[1])
            coefficients = np.linalg.solve(gram,xt.T@target)
            loss = np.mean((xt@coefficients-target)**2)
            candidates.append((loss,i,coefficients,x@coefficients*scale+mu))
        _,i,beta,pred = min(candidates,key=lambda c:c[0])
        chosen.append(i)
        stages.append(pred)
    formulas = []
    for a in range(y.shape[1]):
        pos = 0
        bias = mu[a]
        spec = []
        for i in chosen:
            col = specs[i]
            n = len(col["mean"])
            coeff = beta[pos:pos+n,a]*scale[a]/col["sd"]
            bias -= coeff@col["mean"]
            spec.append(dict(name=names[i],lo=col["lo"],hi=col["hi"],knots=col["knots"],coef=coeff.tolist()))
            pos += n
        formula = dict(intercept=float(bias),variables=spec)
        replay = np.full(len(y),formula["intercept"])
        for i,col in zip(chosen,spec):
            x = np.clip(q[:,i],col["lo"],col["hi"])
            replay += col["coef"][0]*x
            for b,t in zip(col["coef"][1:],col["knots"]):
                replay += b*np.maximum(x-t,0)
        np.testing.assert_allclose(replay,stages[-1][:,a],rtol=1e-10,atol=1e-7)
        formulas.append(formula)
    return [names[i] for i in chosen],formulas,stages
