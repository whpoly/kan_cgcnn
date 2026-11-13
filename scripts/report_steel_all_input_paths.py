"""Verify the native composition ledger and compare it with a global three-variable readout."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_steel_observed_pairs import ARMS, CONCEPTS, INPUT, ROOT, digest, read, r2
from steel_three_element_readout import fit_three
from trace_steel_composition_effects import CompositionPaths


OUT = ROOT/"benchmarks/steel-all-input-paths-20261113"
CASE = ("mb-steels-073","mb-steels-042")


def family(name):
    if name.startswith("ElementFraction|"):
        return "element_fraction"
    if name.startswith("ValenceOrbital|") or name.split()[-1].endswith(("Valence","Unfilled")):
        return "valence"
    if name.endswith(" MeltingT"):
        return "element_melting"
    if name.endswith((" CovalentRadius"," GSvolume_pa","Yang delta")):
        return "size"
    if name.endswith((" Number"," AtomicWeight"," Column"," Row"," MendeleevNumber")):
        return "periodic_and_mass"
    if name.endswith(" GSmagmom"):
        return "element_magnetic_moment"
    if name.endswith(" Electronegativity"):
        return "electronegativity"
    if name.startswith("Stoichiometry|"):
        return "stoichiometry"
    return "other"


def core_values(c):
    with np.load(CONCEPTS/"fold-0/kan.npz") as z:
        names=z["names"].tolist()
    q=np.full((len(c.ids),len(names)),np.nan)
    for f in range(5):
        with np.load(CONCEPTS/f"fold-{f}/kan.npz") as z:
            assert z["names"].tolist()==names
            q[[c.ids.index(s) for s in z["test_ids"]]]=z["q_test"]
    assert np.isfinite(q).all()
    for e in ("Al","Si","Nb","V","W","Mn","N"):
        names.append(e)
        q=np.column_stack((q,100*c.c[:,c.e.index(e)]))
    return q,names


def pair_ledger(v,shares,valid,learned,driver):
    ledger=v[:,None]*shares
    elemental=ledger.sum(axis=0)
    rem_named=v[~valid & ~learned].sum()
    embedding=v[learned].sum()
    gross=np.abs(ledger).sum()+np.abs(v[~valid]).sum()
    total=float(v.sum())
    np.testing.assert_allclose(elemental.sum()+rem_named+embedding,total,atol=1e-7)
    return dict(total=total, assigned=float(elemental[driver]),
                other_elements=float(elemental.sum()-elemental[driver]),undistributed_named=float(rem_named),
                embedding=float(embedding),signed_share=float(elemental[driver]/total) if abs(total)>1e-6 else None,
                abs_share=float(abs(ledger[:,driver]).sum()/gross) if gross>1e-6 else None)


def case_detail(c,i,j,shares,valid,terms,learned,labels,driver):
    detail={}
    versions={"primary":(shares,valid)}
    for coordinate,side in (("weight","low"),("weight","high"),("atomic","symmetric")):
        versions[coordinate+"_"+side]=c.shares(i,j,coordinate,side)
    reverse,_=c.shares(j,i)
    np.testing.assert_allclose(shares,reverse,rtol=1e-7,atol=1e-6)
    conditions=np.abs(shares).sum(axis=1)
    for arm,members in terms.items():
        avg=members.mean(axis=0)
        part=avg*shares[:,driver]
        full=members.sum(axis=1)
        native=members@shares[:,driver]
        gross=np.abs(members)@np.abs(shares)
        denominator=gross.sum(axis=1)+np.abs(members[:,~valid]).sum(axis=1)
        fractions=native/full
        info=dict(versions={name:pair_ledger(avg,w,v,learned,driver) for name,(w,v) in versions.items()},
            elements=dict(zip(c.labels,(avg@shares).tolist())),mo_member_iqr=np.quantile(native,[.25,.5,.75]).tolist(),
            signed_ratio_iqr=np.quantile(fractions,[.25,.5,.75]).tolist(),
            members_within_half=int((abs(full-native)<.5*abs(full)).sum()),
            members_above_half_signed=int((fractions>.5).sum()),members_over_150_pct=int((fractions>1.5).sum()),
            pooled_member_abs_share=float(gross[:,driver].sum()/denominator.sum()),
            exclude_condition_above_five=float(part[conditions<=5].sum()),
            groups={},descriptors=[])
        for label in sorted(set(labels)):
            mask=labels==label
            info["groups"][label]=dict(mo_path=float(part[mask].sum()),
                nonzero=int((abs(part[mask])>1e-6).sum()))
        for k in np.argsort(-abs(part)):
            if abs(part[k])<1e-7:
                break
            info["descriptors"].append(dict(name=c.names[k],original=[c.x[i,k],c.x[j,k]],
                signed_descriptor_delta_from_mo=(c.x[j,k]-c.x[i,k])*shares[k,driver],
                descriptor_total_mpa=avg[k],mo_path_mpa=part[k],cancellation_factor=conditions[k]))
        detail[arm]=info
    return detail


def stats(rows,namespace):
    selected=[r for r in rows if namespace in r["groups"]]
    out=dict(n=len(selected),folds=[])
    for arm in ARMS:
        full=np.array([r[namespace][arm]["total"] for r in selected])
        attr=np.array([r[namespace][arm]["assigned"] for r in selected])
        absp=np.array([r[namespace][arm]["abs_share"] for r in selected])
        out[arm]=dict(within_half=int((abs(full-attr)<.5*abs(full)).sum()),abs_majority=int((absp>.5).sum()),
                      rms_relative=float(np.linalg.norm(full-attr)/np.linalg.norm(full)))
    for f in range(5):
        cases=[r for r in selected if r["fold"]==f]
        out["folds"].append(dict(fold=f,n=len(cases),**{a:sum(abs(r[namespace][a]["total"]-r[namespace][a]["assigned"])
            <.5*abs(r[namespace][a]["total"]) for r in cases) for a in ARMS}))
    return out


def figure(result):
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","Noto Sans CJK SC","DejaVu Sans"]
    fig,(left,right)=plt.subplots(1,2,figsize=(11.3,4.6),gridspec_kw={"width_ratios":[1.08,1]})
    case=result["case"]
    yy=np.arange(5)
    for a,off,mark,color in (("kan",-.12,"o","#17756c"),("mlp",.12,"s","#ae592e")):
        v=case[a]["versions"]["primary"]
        direct=next(row["mo_path_mpa"] for row in case[a]["descriptors"] if row["name"]=="ElementFraction|Mo")
        melting=case[a]["groups"]["element_melting"]["mo_path"]
        valence=case[a]["groups"]["valence"]["mo_path"]
        other=v["assigned"]-direct-melting-valence
        levels=[direct,melting,valence,other,v["assigned"],v["total"]]
        left.scatter(levels,np.arange(len(levels))+off,marker=mark,color=color,s=47,label=a.upper(),zorder=3)
        right.scatter([z["metrics"][a]["official_r2"] for z in result["readout"]["folds"]],
                      yy+off,marker=mark,color=color,s=47,label=a.upper(),zorder=3)
    left.axvline(0,color="#929e9e",lw=1)
    left.axvline(case["measured"],color="#666876",lw=1.2,ls="--")
    left.text(case["measured"]-5,1.45,"实测差\n+268.7",ha="right",color="#595a67",fontsize=9)
    left.set(yticks=np.arange(6),yticklabels=["钼原子分数","元素熔点统计","价电子统计","其他钼路径",
                 "钼路径合计","完整预测差"],ylim=(5.6,-.6),xlim=(-10,290),xlabel="073 → 042，贡献（MPa）",
                 title="钼成分差如何传到强度输出")
    right.axvline(.5,color="#899496",lw=.9,ls="--")
    right.set(yticks=yy,yticklabels=[f"第 {f} 折" for f in range(5)],ylim=(4.6,-.6),xlim=(.35,.8),
              xlabel="留出钢样上对基准正式预测的 R²",title="Ti、C、Co 三变量公式")
    for ax in (left,right):
        ax.grid(axis="y",color="#e1e6e6",lw=.7)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    right.legend(frameon=False,loc="upper right",fontsize=9)
    fig.tight_layout(w_pad=2.6)
    fig.savefig(OUT/"physical-explanation.png",dpi=175,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    with np.load(INPUT) as z:
        c=CompositionPaths(z)
        y=z["y"].copy()
    q,qnames=core_values(c)
    learned=np.array([n.startswith("ROOST_") for n in c.names])
    labels=np.array([family(n) for n in c.names])
    local,folds=[],[]
    input_sha=digest(INPUT)
    script_sha=digest(ROOT/"scripts/audit_steel_all_input_paths.py")
    max_replay=max_conservation=0.
    for f in range(5):
        p=OUT/f"fold-{f}"
        d=read(p/"design.json")
        assert d["ids"]==c.ids and d["columns"]==c.names and d["input_sha256"]==input_sha and d["code_sha256"]==script_sha
        assert d["upstream_code_sha256"]==digest(ROOT/"scripts/audit_steel_observed_pairs.py")
        tr,te=np.array(d["train_indices"]),np.array(d["test_indices"])
        assert not set(tr)&set(te)
        split=len(d["train_pairs"])
        native,official,raw,terms={},{},{},{}
        for a in ARMS:
            arc=p/(a+".npz")
            assert digest(arc)==d["sha256"][a]
            with np.load(arc) as z, np.load(CONCEPTS/f"fold-{f}/{a}.npz") as orig:
                native[a]=z["pred"].mean(axis=0)
                official[a]=z["official_test"].astype(float)
                raw[a]=orig["raw"].astype(float)
                np.testing.assert_allclose(native[a][te],raw[a].mean(axis=0),atol=.002,rtol=1e-6)
                np.testing.assert_array_equal(official[a],orig["official"])
                terms[a]=z["terms"][:,split:]
            max_replay=max(max_replay,d["validation"][a]["replay_max"])
            max_conservation=max(max_conservation,d["validation"][a]["conservation_max"])
        pred=np.column_stack([native[a] for a in ARMS])
        chosen,formula,stages=fit_three(q,qnames,pred,tr)
        fixed=np.logical_and.reduce([np.all(abs(official[a]-raw[a])<.01,axis=0) for a in ARMS])
        fm={}
        for ia,a in enumerate(ARMS):
            test=stages[-1][te,ia]
            separate,_,separate_pred=fit_three(q,qnames,pred[:,ia:ia+1],tr)
            fm[a]=dict(independently_selected=separate,independent_raw_r2=r2(raw[a].mean(0),separate_pred[-1][te,0]),
                raw_r2=r2(raw[a].mean(0),test),official_r2=r2(official[a].mean(0),test),
                unaffected_r2=r2(raw[a].mean(0)[fixed],test[fixed]),measured_r2=r2(y[te],test),
                mae_model=float(np.mean(abs(test-raw[a].mean(0)))),mae_measured=float(np.mean(abs(test-y[te]))),
                full_measured_r2=r2(y[te],official[a].mean(0)),
                budget_raw_r2=[r2(raw[a].mean(0),v[te,ia]) for v in stages])
        folds.append(dict(fold=f,variables=chosen,formulas=dict(zip(ARMS,formula)),metrics=fm,
                          n=len(te),unaffected=int(fixed.sum()),test_ids=[c.ids[i] for i in te],
                          y=y[te].tolist(),readout=stages[-1][te].tolist()))
        for k,row in enumerate(d["test_pairs"]):
            i,j=[c.ids.index(row[key]) for key in ("low","high")]
            assert i in te and j in te
            sh,ok=c.shares(i,j)
            t={a:terms[a][:,k] for a in ARMS}
            for a in ARMS:
                np.testing.assert_allclose(t[a].sum(axis=1),raw[a][:,list(te).index(j)]-raw[a][:,list(te).index(i)],
                                           atol=.003,rtol=1e-6)
            assert abs(y[j]-y[i]-row["measured"])<1e-7
            rec=dict(**row,fold=f)
            for group in row["groups"]:
                if group=="ni_equivalent":
                    continue
                ix=c.labels.index("Mo") if group=="molybdenum" else int(np.argmax(abs(c.w[j,c.driver]-c.w[i,c.driver])))
                rec[group]=dict(driver=c.labels[ix],**{a:pair_ledger(t[a].mean(0),sh,ok,learned,ix) for a in ARMS})
            if (c.ids[i],c.ids[j])==CASE:
                ix=c.labels.index("Mo")
                case=dict(low=c.ids[i],high=c.ids[j],measured=row["measured"],fold=f,wt_low=dict(zip(c.e,100*c.w[i])),
                    wt_high=dict(zip(c.e,100*c.w[j])),mo_at_pct=[100*c.c[h,c.e.index("Mo")] for h in (i,j)],
                    surrogate_delta=dict(zip(ARMS,stages[-1][j]-stages[-1][i])),
                    **case_detail(c,i,j,sh,ok,t,learned,labels,ix))
            local.append(rec)
    assert len({sid for fold in folds for sid in fold["test_ids"]})==len(y)==312
    readout=dict(folds=folds,mean_fold={a:{m:float(np.mean([f["metrics"][a][m] for f in folds]))
        for m in ("raw_r2","official_r2","unaffected_r2","measured_r2","mae_model","mae_measured","full_measured_r2")}
        for a in ARMS})
    readout["observed_quartile_effects"]={}
    for name in folds[0]["variables"]:
        if not all(name in f["variables"] for f in folds):
            continue
        values=q[:,qnames.index(name)]
        levels=np.quantile(values,[.25,.75],method="nearest")
        effect=dict(at_pct=levels.tolist(),observed_examples=[c.ids[np.flatnonzero(values==x)[0]] for x in levels])
        for a in ARMS:
            effect[a]=[]
            for f in folds:
                var=next(v for v in f["formulas"][a]["variables"] if v["name"]==name)
                x=np.clip(levels,var["lo"],var["hi"])
                z=var["coef"][0]*x
                for b,knot in zip(var["coef"][1:],var["knots"]):
                    z+=b*np.maximum(x-knot,0)
                effect[a].append(float(z[1]-z[0]))
        readout["observed_quartile_effects"][name]=effect
    result=dict(case=case,readout=readout,native_generalization={g:stats(local,g) for g in ("near_single","molybdenum")},
                pairs=local,verified_formulas=len(c.specs),formula_rebuild_max=c.rebuild_error,
                max_replay_mpa=max_replay,max_native_conservation_mpa=max_conservation,input_sha256=input_sha)
    (OUT/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    figure(result)


if __name__=="__main__":
    main()
