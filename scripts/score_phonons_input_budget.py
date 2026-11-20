"""Score fixed-input phonons KANs and export the paired outer predictions."""

import argparse
import pickle

from benchmark_phonons_input_budget import ORIGINAL, OUTPUT, REFERENCE, member_dir, native, read


np = native.np


def restore(directory, fit, names, frame):
    with (directory / "model.pkl").open("rb") as stream:
        model = pickle.load(stream)
    model._restore_model()
    assert list(model.optimal_descriptors[:model.n_feat]) == names == fit["selected_features"]
    assert model.model.input_shape[-1] == model.n_feat == len(names)
    if fit["variant"] == "bspline":
        assert model.target_scaling["fit_ids"] == fit["train_ids"]
    x = np.nan_to_num(model._scaler.transform(frame[names].to_numpy()), nan=-1).astype("float32")
    prediction = np.asarray(model.model(x, training=False)).ravel()
    assert prediction.shape == (len(frame),) and np.isfinite(prediction).all()
    native.tf.keras.backend.clear_session()
    return prediction


def formal(arm, fold, truth, ids):
    folder = REFERENCE / arm / f"fold-{fold['fold']}"
    key = "modnet" if arm == "mm_modnet" else "kan"
    archive = folder / "outer-predictions.npz"
    assert native.file_hash(archive) == fold[key]["prediction_sha256"]
    with np.load(archive) as z:
        np.testing.assert_array_equal(z["sample_ids"], ids)
        np.testing.assert_array_equal(z["target"], truth)
        matrix, prediction = z["members"], z["prediction"]
    assert matrix.shape == (50, len(truth)) and np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix.mean(axis=0), prediction, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.mean(abs(prediction - truth)), read(folder / "complete.json")["mae"],
                               rtol=0, atol=1e-9)
    return prediction, matrix


def modnet_inner_reference(fold, frame, target):
    """For each validation crystal, average the ten MODNet members that held it out."""
    folder = REFERENCE / "mm_modnet" / f"fold-{fold['fold']}"
    source = read(folder / "frozen-selection.json")
    matrices = [[] for _ in range(5)]
    for member in source["members"]:
        inner = member["inner"]
        expected = fold["members"][inner]
        path = native.Path(member["directory"])
        fit = read(path / "result.json")
        assert fit["variant"] == "modnet" and fit["train_ids"] == expected["train_ids"]
        assert fit["validation_ids"] == expected["val_ids"]
        assert native.file_hash(path / "model.pkl") == source["model_sha256"][str(path)]
        val_ids = expected["val_ids"]
        names = fold["ranking"][:member["genes"]["n_feat"]]
        prediction = restore(path, fit, names, frame.loc[val_ids])
        np.testing.assert_allclose(np.mean(abs(prediction - target.loc[val_ids].to_numpy())),
                                   fit["validation_mae"], rtol=1e-5, atol=.003)
        matrices[inner].append(prediction)
    assert all(len(matrix) == 10 for matrix in matrices)
    return [np.mean(abs(np.mean(matrix, axis=0) - target.loc[expected["val_ids"]].to_numpy()))
            for expected, matrix in zip(fold["members"], matrices)]


def score_fold(fold, budgets, frame, target, fingerprint):
    number = fold["fold"]
    test = frame.loc[fold["test_ids"]]
    ids = fold["official_test_ids"]
    truth = target.loc[test.index].to_numpy()
    assert ids == [f"mb-phonons-{int(sid[2:]) + 1:04d}" for sid in test.index]
    inner_modnet = modnet_inner_reference(fold, frame, target)
    modnet, _ = formal("mm_modnet", fold, truth, ids)
    full, matrix = formal("mm_kan", fold, truth, ids)
    original_five, controls, validation = [], [], []
    src = REFERENCE / "mm_kan" / f"fold-{number}"
    frozen = read(src / "frozen-selection.json")
    for expected in fold["members"]:
        index = expected["index_in_reference"]
        source = frozen["members"][index]
        path = native.Path(source["directory"])
        fit = read(path / "result.json")
        sha = native.file_hash(path / "model.pkl")
        assert fit["model_sha256"] == frozen["model_sha256"][str(path)] == sha
        names = fold["ranking"][:expected["genes"]["n_feat"]]
        prediction = restore(path, fit, names, test)
        np.testing.assert_allclose(prediction, matrix[index], rtol=1e-5, atol=.01)
        original_five.append(matrix[index])
        controls.append(dict(sha256=sha, max_replay_error=float(np.max(abs(prediction - matrix[index])))))
    rows = native.pd.DataFrame(dict(fold=number, sample_id=ids, target=truth, mm_modnet=modnet,
                                    mm_kan=full, kan_original_five=np.mean(original_five, axis=0)))
    for budget in budgets:
        names = fold["ranking"][:budget]
        predictions, hashes, losses = [], [], []
        for expected in fold["members"]:
            path = member_dir(number, budget, expected["inner"])
            fit = read(path / "result.json")
            assert fit["run_fingerprint"] == fingerprint and not fit["fast"]
            assert fit["genes"] == dict(expected["genes"], n_feat=budget)
            assert fit["seed"] == expected["seed"] and fit["train_ids"] == expected["train_ids"]
            assert fit["validation_ids"] == expected["val_ids"] and fit["features"] == names
            ids_for_fit = fit["train_ids"] + fit["validation_ids"]
            assert set(ids_for_fit).isdisjoint(test.index)
            assert sorted(ids_for_fit) == sorted(fold["train_ids"])
            both = native.pd.concat((frame.loc[fit["validation_ids"]], test))
            prediction = restore(path, fit, names, both)
            nval = len(fit["validation_ids"])
            val_loss = np.mean(abs(target.loc[fit["validation_ids"]].to_numpy() - prediction[:nval]))
            np.testing.assert_allclose(val_loss, fit["validation_mae"], atol=.003, rtol=1e-5)
            predictions.append(prediction[nval:])
            hashes.append(native.file_hash(path / "model.pkl"))
            losses.append(val_loss)
        rows[f"kan_{budget}"] = np.mean(predictions, axis=0)
        validation.append(dict(budget=budget, inner_mae=losses, models_sha256=hashes))
    return rows, dict(fold=number, fits=validation, inner_modnet_mae=inner_modnet, controls=controls)


def apply_validation_rule(rows, audits, margin):
    """Choose a budget using only that outer fold's inner errors."""
    choices = []
    for audit in audits:
        limit = np.mean(audit["inner_modnet_mae"]) * (1 + margin)
        eligible = [r["budget"] for r in audit["fits"] if np.mean(r["inner_mae"]) <= limit]
        # If the grid misses the reference, use its best inner-MAE setting.
        best = min(audit["fits"], key=lambda r: (np.mean(r["inner_mae"]), r["budget"]))
        k = min(eligible) if eligible else best["budget"]
        mask = rows.fold == audit["fold"]
        rows.loc[mask, "kan_inner_selected"] = rows.loc[mask, f"kan_{k}"]
        choices.append(dict(fold=audit["fold"], budget=k, threshold=limit, inner_reference=np.mean(audit["inner_modnet_mae"]),
                            within_inner_margin=bool(eligible)))
    return choices


def aggregate(rows, protocol, inner_choices):
    folds = protocol["folds"]
    y = rows.target.to_numpy()
    base_error = abs(rows.mm_modnet.to_numpy() - y)
    groups = [np.flatnonzero(rows.fold.to_numpy() == f) for f in range(5)]
    assert len(rows) == rows.sample_id.nunique() == 1265 and all(len(i) == 253 for i in groups)
    base = np.mean([base_error[i].mean() for i in groups])
    reference_union = np.array([f["modnet"]["union"] for f in folds])
    budgets = protocol["budgets"]
    choices = [("mm_modnet", reference_union),
               ("mm_kan", [f["kan"]["union"] for f in folds]),
               ("kan_original_five", [f["members"][0]["genes"]["n_feat"] for f in folds])]
    choices += [(f"kan_{k}", [k] * 5) for k in budgets]
    choices += [("kan_inner_selected", [item["budget"] for item in inner_choices])]
    result = []
    for key, counts in choices:
        err = abs(rows[key].to_numpy() - y)
        mae = np.array([err[i].mean() for i in groups])
        deltas = np.array([(err[i] - base_error[i]).mean() for i in groups])
        ratio = mae.mean() / base - 1
        np.testing.assert_allclose(mae.mean(), err.mean(), atol=1e-12, rtol=0)
        result.append(dict(model=key, inputs_by_fold=counts, mean_inputs=np.mean(counts),
                           reduction=1 - np.mean(counts) / reference_union.mean(),
                           fold_mae=mae, mean_mae=mae.mean(), delta_mae=deltas.mean(),
                           relative_change=ratio, fold_deltas=deltas,
                           folds_at_or_below_modnet=int(np.sum(deltas <= 0)),
                           folds_within_margin=int(sum(err[i].mean() <= base_error[i].mean() *
                                                       (1 + protocol["relative_mae_margin"]) for i in groups)),
                           passes_mean_margin=bool(ratio <= protocol["relative_mae_margin"]),
                           passes_no_increase=bool(ratio <= 0)))
    points = result[3:-1]
    return dict(samples=len(rows), unit="cm^-1", relative_margin=protocol["relative_mae_margin"],
                margin_mae=base * (1 + protocol["relative_mae_margin"]), results=result,
                inner_selected=inner_choices,
                smallest_tested_passing={name:next((int(r["mean_inputs"]) for r in points if r[flag]), None)
                                         for name, flag in (("within_margin", "passes_mean_margin"),
                                                            ("no_increase", "passes_no_increase"))},
                smallest_passing_all_five=next((int(r["mean_inputs"]) for r in points
                                                if r["folds_within_margin"] == 5), None))


def draw(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    baseline = summary["results"][0]
    points = summary["results"][3:-1]
    xs = np.array([r["mean_inputs"] for r in points])
    ys = np.array([r["mean_mae"] for r in points])
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    ax.plot(xs, ys, "-o", label="KAN, five-member mean", color="#17665e", lw=2, markersize=5)
    ax.axhline(baseline["mean_mae"], color="#334155", lw=1.5,
               label=f"Full MODNet ({baseline['mean_inputs']:.1f} inputs on average)")
    ax.axhline(summary["margin_mae"], color="#a65d18", ls="--", lw=1.3, label="MODNet MAE +5%")
    ax.set(xlabel="Matminer descriptors per crystal", ylabel="Five-fold MAE (cm⁻¹)",
           title="Matbench Phonons · 1,265 held-out crystal predictions")
    ax.xaxis.set_major_locator(MultipleLocator(32))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / "feature-count-vs-mae.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5),
                        help="Score one completed fold and save its predictions and audit.")
    args = parser.parse_args()
    protocol = read(OUTPUT / "protocol.json")
    assert native.file_hash(ORIGINAL) == protocol["raw_data_sha256"]
    fingerprint = native.digest(protocol)
    data = native.MODData.load(ORIGINAL)
    frame = data.df_featurized.replace([np.inf, -np.inf], np.nan).fillna(0)
    target = data.df_targets.iloc[:, 0]
    assert frame.index.is_unique and frame.index.equals(target.index)
    scores, audits = [], []
    folds = protocol["folds"] if args.fold is None else [protocol["folds"][args.fold]]
    for fold in folds:
        rows, audit = score_fold(fold, protocol["budgets"], frame, target, fingerprint)
        scores.append(rows)
        audits.append(audit)
        print(f"Verified outer fold {fold['fold']}, {len(rows)} crystals", flush=True)
    if args.fold is not None:
        directory = OUTPUT / f"fold-{args.fold}"
        scores[0].to_csv(directory / "budget-predictions.csv", index=False, float_format="%.17g")
        native.write_json(directory / "budget-audit.json", audits[0])
        return
    rows = native.pd.concat(scores, ignore_index=True)
    inner_choices = apply_validation_rule(rows, audits, protocol["relative_mae_margin"])
    result = aggregate(rows, protocol, inner_choices)
    predictions = OUTPUT / "oof-predictions.csv"
    rows.to_csv(predictions, index=False, float_format="%.17g")
    result.update(protocol_sha256=native.file_hash(OUTPUT / "protocol.json"),
                  predictions_sha256=native.file_hash(predictions), scoring_sha256=native.file_hash(__file__))
    native.write_json(OUTPUT / "model-audit.json", audits)
    native.write_json(OUTPUT / "summary.json", result)
    draw(result)
    for row in result["results"]:
        print(f"{row['model']:>18} {row['mean_inputs']:6.1f} inputs   {row['mean_mae']:.4f} cm^-1")


if __name__ == "__main__":
    main()
