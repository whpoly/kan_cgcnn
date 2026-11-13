"""Report the matched-pair, node-budget, and chromium-edge checks on original steels."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Element


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/steel-observed-pairs-20261112"
CONCEPTS = ROOT / "benchmarks/steel396-physical-concepts-20261112"
INPUT = ROOT / "benchmarks/steels-paper-reproduction-20261102/mm-roost396/inputs.npz"
ARMS = ("kan", "mlp")
GROUPS = ("physical3", "strength1", "strength3", "strength5", "strength10pct")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r2(truth, estimate):
    return float(1 - np.sum((truth - estimate) ** 2) / np.sum((truth - truth.mean()) ** 2))


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def metrics(rows):
    y = np.array([r["measured"] for r in rows])
    out = dict(count=len(rows), positive=int((y > 0).sum()), measured=float(y.mean()),
               measured_median=float(np.median(y)), q_gap=float(np.mean([r["q_gap"] for r in rows])))
    for arm in ARMS:
        raw = np.array([r[arm]["raw"] for r in rows])
        values = {}
        for key in ("raw", "official") + GROUPS:
            d = np.array([r[arm][key] for r in rows])
            values[key] = dict(mean=float(d.mean()), mae_measured=float(np.mean(np.abs(d-y))),
                same_measured=int((np.sign(d) == np.sign(y)).sum()),
                correct_measured_decreases=int(((y < 0) & (d < 0)).sum()),
                correct_measured_increases=int(((y > 0) & (d > 0)).sum()),
                same_model=int((np.sign(d) == np.sign(raw)).sum()),
                mae_model=float(np.mean(np.abs(d-raw))), relative_model=relative_error(d, raw))
        out[arm] = values
    return out


def plot(report):
    matched = [r for r in report["pairs"] if r["group"] == "matched"]
    folds = report["folds"]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), gridspec_kw={"width_ratios":[1.08, 1]})
    left, right = axes
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dce3e6", linewidth=.7)
        ax.set_axisbelow(True)
    left.axhline(0, color="#89969b", lw=.9)
    left.scatter([r["q_gap"] for r in matched], [r["measured"] for r in matched], s=32,
                 color="#587487", alpha=.84, linewidths=0, label="40 对按成分匹配的钢样")
    for group, lab, marker, col, offset in (("requested","020 → 005","*","#ab512d",(-12,-17)),
                                            ("near_unit","038 → 016","D","#286d65",(7,-2))):
        r = next(r for r in report["pairs"] if r["group"] == group)
        left.scatter(r["q_gap"], r["measured"], marker=marker, s=95 if marker=="*" else 38, color=col, zorder=3)
        left.annotate(lab, (r["q_gap"], r["measured"]), xytext=offset, textcoords="offset points",
                      ha="right" if offset[0]<0 else "left", color=col, fontsize=9)
    for row in (r for r in report["pairs"] if r["group"] == "requested_near_unit"):
        left.scatter(row["q_gap"], row["measured"], marker="^", s=46, color="#722d62", zorder=3)
        left.annotate(f"{row['low'][-3:]} → {row['high'][-3:]}", (row["q_gap"], row["measured"]),
                      xytext=(8, 1), textcoords="offset points", fontsize=9, color="#722d62")
    left.set(title="真实钢样之间的实测屈服强度差", xlabel="镍当量差（质量百分比当量点）",
             ylabel="实测屈服强度差（MPa）")
    left.legend(loc="upper left", frameon=False, fontsize=9)
    y = np.arange(5)
    for arm, off, color, marker in (("kan",-.13,"#216f71","o"),("mlp",.13,"#ad522c","s")):
        val = [100*f["matched"][arm]["strength3"]["relative_model"] for f in folds]
        right.scatter(val, y+off, color=color, marker=marker, s=44, label=arm.upper(), zorder=3)
    right.set(title="用三个按强度选定的第一层节点还原预测差",
              xlabel="相对均方根误差（%，越低越好）",
              yticks=y, yticklabels=[f"第 {f['fold']} 折（{f['matched']['count']} 对）" for f in folds],
              xlim=(50,101), ylim=(4.65,-.65))
    right.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "observed-steels.png", dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    code_hash = digest(ROOT / "scripts/audit_steel_observed_pairs.py")
    edge_hash = digest(ROOT / "scripts/check_steel_observed_edge.py")
    pairs, folds, examples, records = [], [], {}, {arm: [] for arm in ARMS}
    all_ids = []
    input_data = np.load(INPUT)
    y_by_id = dict(zip(input_data["sample_ids"], input_data["y"]))
    for f in range(5):
        folder = OUT / f"fold-{f}"
        design = read(folder / "pairs.json")
        ids = design["test_ids"]
        all_ids += ids
        q = np.asarray(design["test_q"])
        p = design["pairs"]
        lo = [ids.index(row["low"]) for row in p]
        hi = [ids.index(row["high"]) for row in p]
        for row in p:
            assert row["y_low"] == y_by_id[row["low"]] and row["y_high"] == y_by_id[row["high"]]
        values = {}
        for i in range(50):
            meta_path = folder / f"member-{i:02d}.json"
            m = read(meta_path)
            path = meta_path.with_suffix(".npz")
            assert m["code_sha256"] == code_hash and m["arrays_sha256"] == digest(path)
            with np.load(path) as z:
                for k in z.files:
                    values.setdefault(k, []).append(z[k])
            for arm in ARMS:
                records[arm].append(m[arm])
                if f == i == 0:
                    examples[arm] = m[arm]["example"]
        values = {k: np.array(v) for k,v in values.items()}
        q_scores = {}
        for arm in ARMS:
            with np.load(CONCEPTS / f"fold-{f}/{arm}.npz") as original:
                np.testing.assert_array_equal(ids, original["test_ids"])
                col = original["names"].tolist().index("ni_equivalent")
                np.testing.assert_allclose(q, original["q_test"][:,col], atol=1e-10)
                for key in ("raw", "official"):
                    delta = original[key][:,hi] - original[key][:,lo]
                    np.testing.assert_allclose(values[arm+"_"+key], delta, rtol=1e-6, atol=.001)
            q_scores[arm] = {key:r2(q, values[arm+"_q_"+key].mean(axis=0))
                             for key in ("physical3", "strength3")}
            for j,row in enumerate(p):
                row[arm] = {k:float(values[arm+"_"+k][:,j].astype(float).mean())
                            for k in ("raw","official") + GROUPS}
                row[arm]["corrected_members"] = int(values[arm+"_corrected"][:,j].sum())
                row["fold"] = f
        pairs.extend(p)
        matched = [r for r in p if r["group"] == "matched"]
        clean = [r for r in matched if not any(r[a]["corrected_members"] for a in ARMS)]
        edge = read(folder / "chromium-edges.json")
        path = folder / "chromium-edges.npz"
        assert edge["code_sha256"] == edge_hash and edge["arrays_sha256"] == digest(path)
        mask = [j for j,t in enumerate(edge["triples"]) if t["group"] == "matched"]
        cr = dict(members=len(edge["members"]), triples=len(mask))
        with np.load(path) as z:
            for arm in ARMS:
                error = z[arm+"_standardized_error"][:,mask]
                cr[arm] = dict(mean_error_sd=float(error.mean()), max_error_sd=float(error.max()),
                               above_one_percent_sd=float((error > .01).mean()))
        if f == 0:
            examples["chromium_triple"] = next(r for r in edge["triples"] if r["group"]=="requested")
            examples["chromium_paths"] = edge["example"]
            examples["near_unit_candidates"] = design["requested_higher_near_unit_candidates"]
            examples["all_rows_higher_near_unit_candidates"] = design["all_rows_higher_near_unit_candidates"]
        folds.append(dict(fold=f, readout_r2=q_scores, matched=metrics(matched), clean=metrics(clean), chromium=cr))
    assert len(set(all_ids)) == len(all_ids) == len(y_by_id) == 312
    cols = input_data["columns"].tolist()
    frac_idx = [i for i,name in enumerate(cols) if name.startswith("ElementFraction|")]
    elements = [cols[i].split("|")[1] for i in frac_idx]
    fractions = input_data["X"][:,frac_idx]
    mass = fractions * np.array([float(Element(el).atomic_mass) for el in elements])
    wt = 100 * mass / mass.sum(axis=1, keepdims=True)
    ids = list(input_data["sample_ids"])
    steel_ids = sorted({p[k] for p in pairs if p["group"]!="matched" for k in ("low","high")} |
                       {examples["chromium_triple"]["middle"]})
    examples["chemistry_wt_pct"] = {s:{el:float(wt[ids.index(s), elements.index(el)])
                                           for el in ("Ni","C","Mn","Cr","Si","Fe")} for s in steel_ids}
    for p in pairs:
        for k in ("low","high"):
            row = wt[ids.index(p[k])]
            eq = sum(c*row[elements.index(el)] for el,c in (("Ni",1),("C",30),("Mn",.5)))
            np.testing.assert_allclose(eq, p["q_"+k], atol=1e-8, rtol=0)
    matched = [r for r in pairs if r["group"] == "matched"]
    endpoints = [r[k] for r in matched for k in ("low","high")]
    assert len(set(endpoints)) == len(endpoints) == 80
    clean = [r for r in matched if not any(r[a]["corrected_members"] for a in ARMS)]
    equal_fold = {}
    checks = {}
    for arm in ARMS:
        equal_fold[arm] = {}
        for group in ("matched","clean"):
            equal_fold[arm][group] = {key:float(np.mean([f[group][arm][key]["relative_model"] for f in folds]))
                                       for key in GROUPS}
        equal_fold[arm]["q_r2"] = {key:np.mean([f["readout_r2"][arm][key] for f in folds])
                                    for key in ("physical3","strength3")}
        equal_fold[arm]["chromium_error_sd"] = np.mean([f["chromium"][arm]["mean_error_sd"] for f in folds])
        rec = records[arm]
        checks[arm] = {k:max(m[k] for m in rec) for k in
                       ("replay_max_mpa","conservation_max_mpa","descriptor_sum_max_mpa")}
        for key in ("n_train_pairs","first_width"):
            checks[arm][key] = [min(m[key] for m in rec), max(m[key] for m in rec)]
    result = dict(input_archive=str(INPUT.relative_to(ROOT)), input_sha256=digest(INPUT),
                  target="measured yield strength in MPa", code_sha256=code_hash, edge_code_sha256=edge_hash,
                  folds=folds, pooled=metrics(matched), pooled_clean=metrics(clean), equal_fold=equal_fold,
                  validation=checks, examples=examples, pairs=pairs)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    plot(result)


if __name__ == "__main__":
    main()
