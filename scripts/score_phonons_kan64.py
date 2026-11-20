"""Freeze KAN choices using inner validation, then score the five outer folds."""

import gc
import pickle
from pathlib import Path

from benchmark_phonons_input_budget import ORIGINAL
from train_phonons_kan64 import DENSE, N_FEATURES, OUTPUT, PRIOR, native, read, verify_fit


np, pd = native.np, native.pd


def select_configurations(plan, records, fingerprint):
    """Reads result metadata only; no outer predictions or targets are loaded."""
    choices = []
    for fold in plan["folds"]:
        inner_mae = {}
        for config in fold["configurations"]:
            losses = []
            for inner in range(5):
                record = records[fold["fold"], config["name"], inner]
                path = Path(record["directory"])
                assert record["reused"] == ("reuse" in config)
                assert native.file_hash(path / "model.pkl") == record["model_sha256"]
                assert native.file_hash(path / "result.json") == record["result_sha256"]
                result = read(path / "result.json")
                verify_fit(result, fold, config, inner,
                           plan["prior_fingerprint"] if record["reused"] else fingerprint)
                assert result["validation_mae"] == record["validation_mae"]
                losses.append(result["validation_mae"])
            inner_mae[config["name"]] = dict(member_mae=losses, mean_mae=float(np.mean(losses)))
        assert len(inner_mae) == 10
        chosen = min(inner_mae, key=lambda name: (inner_mae[name]["mean_mae"], name))
        choices.append(dict(fold=fold["fold"], refitted_choice=chosen, validation=inner_mae))
    selection = dict(protocol_fingerprint=fingerprint, fits_sha256=native.file_hash(OUTPUT / "fits.json"),
                     test_labels_used=False, rule=plan["selection_rule"], choices=choices)
    path = OUTPUT / "selection.json"
    if path.exists():
        assert read(path) == selection, "Previously frozen selection changed"
    else:
        native.write_json(path, selection)
    return choices


def restore_and_audit(record, config, fit, names, frame, training_targets):
    path = Path(record["directory"])
    with (path / "model.pkl").open("rb") as stream:
        model = pickle.load(stream)
    model._restore_model()
    assert list(model.optimal_descriptors[:model.n_feat]) == names == fit["selected_features"]
    assert model.model.input_shape[-1] == model.n_feat == N_FEATURES
    assert list(model.spline_widths) == config["hidden_widths"] + [1]
    scaling = model.target_scaling
    assert scaling["fit_ids"] == fit["train_ids"] and scaling["prediction_units"] == "physical"
    np.testing.assert_allclose([scaling["mean"], scaling["std"]],
                               [training_targets.mean(), training_targets.std(ddof=0)], rtol=0, atol=1e-10)
    parameters = sum(int(np.prod(w.shape)) for w in model.model.trainable_weights)
    assert parameters == model.model.count_params() == fit["parameters"] == record["parameters"]
    layers = [layer for layer in model.model.layers if hasattr(layer, "coefficients")]
    widths = [N_FEATURES] + config["hidden_widths"] + [1]
    assert len(layers) == 3
    for layer, nin, nout in zip(layers, widths, widths[1:]):
        assert tuple(layer.coefficients.shape) == (nin, nout, 8)
        assert tuple(layer.bias.shape) == (nout,)
        assert np.isfinite(layer.coefficients.numpy()).all() and np.isfinite(layer.bias.numpy()).all()
    x = np.nan_to_num(model._scaler.transform(frame[names].to_numpy()), nan=-1).astype("float32")
    prediction = np.asarray(model.model(x, training=False)).ravel()
    assert prediction.shape == (len(frame),) and np.isfinite(prediction).all()
    del model, layers
    native.tf.keras.backend.clear_session()
    gc.collect()
    return prediction


def mlp_parameters(fold, config_name, dense_plan, dense_summary):
    config = next(c for c in dense_plan["folds"][fold]["configurations"] if c["name"] == config_name)
    counts = []
    for inner in range(5):
        result = read(DENSE / f"fold-{fold}" / config_name / f"inner-{inner}" / "result.json")
        assert result["genes"] == dict(config["original_genes"], n_feat=N_FEATURES)
        widths = [N_FEATURES] + [w[0] for w in result["num_neurons"]] + [1]
        count = sum((a + 1) * b for a, b in zip(widths, widths[1:]))
        assert count == result["parameters"] == dense_summary["folds"][fold]["parameters_by_config"][config_name]
        counts.append(count)
    return dict(per_member=float(np.mean(counts)), ensemble_total=sum(counts))


def score_fold(fold, choice, records, frame, target, paired, dense_plan, dense_summary):
    number = fold["fold"]
    names, held, ids = fold["features"], fold["test_ids"], fold["official_test_ids"]
    reference = paired.loc[ids]
    truth = target.loc[held].to_numpy()
    assert (reference["fold"] == number).all()
    assert ids == [f"mb-phonons-{int(sid[2:]) + 1:04d}" for sid in held]
    np.testing.assert_allclose(reference["target"].to_numpy(), truth, rtol=0, atol=1e-10)
    outer, candidate_rows, validation_archive, outer_archive = {}, [], {}, {}
    for config in fold["configurations"]:
        name = config["name"]
        evaluate_outer = name in ("kan-00", choice["refitted_choice"])
        predictions, params, replay_errors, epochs = [], [], [], []
        for inner in range(5):
            record = records[number, name, inner]
            fit = read(Path(record["directory"]) / "result.json")
            val = fit["validation_ids"]
            indices = val + held if evaluate_outer else val
            pred = restore_and_audit(record, config, fit, names, frame.loc[indices, names],
                                     target.loc[fit["train_ids"]].to_numpy())
            loss = float(np.mean(abs(pred[:len(val)] - target.loc[val].to_numpy())))
            np.testing.assert_allclose(loss, fit["validation_mae"], rtol=1e-5, atol=.003)
            validation_archive[f"{name}_inner_{inner}_ids"] = np.asarray(val)
            validation_archive[f"{name}_inner_{inner}_prediction"] = pred[:len(val)]
            if evaluate_outer:
                predictions.append(pred[len(val):])
            params.append(fit["parameters"])
            epochs.append(fit["epochs"])
            replay_errors.append(abs(loss - fit["validation_mae"]))
        assert len(set(params)) == 1
        if evaluate_outer:
            matrix = np.asarray(predictions)
            assert matrix.shape == (5, 253)
            outer[name] = matrix.mean(axis=0)
            outer_archive[name] = matrix
        candidate_rows.append(dict(fold=number, configuration=name, selected=name == choice["refitted_choice"],
                      original_generation=config["generation"], original_candidate=config["candidate"],
                      original_features=config["original_genes"]["n_feat"], features=N_FEATURES,
                      hidden_1=config["hidden_widths"][0], hidden_2=config["hidden_widths"][1],
                      lr=config["original_genes"]["lr"], batch_size=config["original_genes"]["batch_size"],
                      input_scaling=config["original_genes"]["xscale"], parameters_per_member=params[0],
                      parameters_per_ensemble=sum(params), mean_inner_mae=choice["validation"][name]["mean_mae"],
                      mean_epochs=float(np.mean(epochs)), members_at_epoch_limit=sum(e == 800 for e in epochs),
                      max_validation_replay_error=max(replay_errors), reused=name == "kan-00"))
        print(f"Verified fold={number} {name}; outer evaluated={evaluate_outer}", flush=True)
    np.testing.assert_allclose(outer["kan-00"], reference["kan_64"].to_numpy(), rtol=1e-5, atol=.003)
    output = OUTPUT / f"fold-{number}"
    np.savez_compressed(output / "validation-members.npz", **validation_archive)
    np.savez_compressed(output / "outer-members.npz", sample_ids=np.asarray(ids), **outer_archive)
    selected_prediction = (reference["kan_64"].to_numpy() if choice["refitted_choice"] == "kan-00"
                           else outer[choice["refitted_choice"]])
    rows = pd.DataFrame(dict(fold=number, sample_id=ids, target=truth,
                        kan_64_frozen=reference["kan_64"].to_numpy(),
                        kan_64_reselected=selected_prediction,
                        mlp_64_frozen=reference["dense_64_five"].to_numpy(),
                        mlp_64_reselected=reference["dense_64_inner_selected"].to_numpy()))
    dense_choice = dense_summary["selected_configurations"][number]["refitted_choice"]
    lookup = {r["configuration"]: r for r in candidate_rows}
    parameters = {}
    for label, name in (("kan_64_frozen", "kan-00"), ("kan_64_reselected", choice["refitted_choice"])):
        parameters[label] = dict(per_member=lookup[name]["parameters_per_member"],
                                 ensemble_total=lookup[name]["parameters_per_ensemble"])
    parameters["mlp_64_frozen"] = mlp_parameters(number, "dense-00", dense_plan, dense_summary)
    parameters["mlp_64_reselected"] = mlp_parameters(number, dense_choice, dense_plan, dense_summary)
    return rows, candidate_rows, dict(fold=number, kan_choice=choice["refitted_choice"], mlp_choice=dense_choice,
                        parameters=parameters, outer_predictions_sha256=native.file_hash(output / "outer-members.npz"),
                        validation_predictions_sha256=native.file_hash(output / "validation-members.npz"))


def main():
    plan = read(OUTPUT / "protocol.json")
    trained = read(OUTPUT / "fits.json")
    fingerprint = native.digest(plan)
    assert trained["protocol_fingerprint"] == fingerprint
    records = {(r["fold"], r["configuration"], r["inner"]): r for r in trained["records"]}
    assert len(records) == len(trained["records"]) == 250
    choices = select_configurations(plan, records, fingerprint)
    selection_sha = native.file_hash(OUTPUT / "selection.json")
    print("Frozen all five choices using inner validation only. Starting outer evaluation.", flush=True)
    # All choices have been written before opening any outer target or prediction file.
    assert native.file_hash(ORIGINAL) == plan["raw_data_sha256"]
    dense_plan, dense_summary = read(DENSE / "protocol.json"), read(DENSE / "summary.json")
    assert native.file_hash(DENSE / "protocol.json") == plan["dense_protocol_sha256"] == dense_summary["protocol_sha256"]
    assert native.file_hash(DENSE / "oof-predictions.csv") == dense_summary["predictions_sha256"]
    paired = pd.read_csv(DENSE / "oof-predictions.csv", float_precision="round_trip").set_index("sample_id")
    original = native.MODData.load(ORIGINAL)
    frame = original.df_featurized.replace([np.inf, -np.inf], np.nan).fillna(0)
    target = original.df_targets.iloc[:, 0]
    rows, candidates, audits = [], [], []
    for fold, choice in zip(plan["folds"], choices):
        assert fold["fold"] == choice["fold"]
        result, checks, audit = score_fold(fold, choice, records, frame, target, paired, dense_plan, dense_summary)
        rows.append(result)
        candidates.extend(checks)
        audits.append(audit)
    full = pd.concat(rows, ignore_index=True)
    assert len(full) == full.sample_id.nunique() == 1265
    assert set(full.sample_id) == set(paired.index) and full.groupby("fold").size().tolist() == [253] * 5
    assert np.isfinite(full.drop(columns="sample_id").to_numpy()).all()
    estimates = {}
    for name in full.columns.drop(["fold", "sample_id", "target"]):
        errors = (full[name] - full.target).abs()
        fold_mae = errors.groupby(full["fold"]).mean().to_numpy()
        np.testing.assert_allclose(errors.mean(), fold_mae.mean(), rtol=0, atol=1e-10)
        counts = [a["parameters"][name] for a in audits]
        estimates[name] = dict(mean_mae=float(fold_mae.mean()), fold_mae=fold_mae,
            features=64, members_per_fold=5, mean_parameters_per_member=float(np.mean([x["per_member"] for x in counts])),
            mean_parameters_per_ensemble=float(np.mean([x["ensemble_total"] for x in counts])))
    expected = {"kan_64_frozen": "kan_64", "mlp_64_frozen": "dense_64_five", "mlp_64_reselected": "dense_64_inner_selected"}
    for name, old in expected.items():
        np.testing.assert_allclose(estimates[name]["fold_mae"], dense_summary["results"][old]["fold_mae"], rtol=0, atol=1e-10)
    comparisons = {}
    selected = estimates["kan_64_reselected"]
    for baseline in ("kan_64_frozen", "mlp_64_frozen", "mlp_64_reselected"):
        ref = estimates[baseline]
        difference = ref["fold_mae"] - selected["fold_mae"]
        comparisons[baseline] = dict(baseline_minus_reselected_kan_by_fold=difference,
                    mean_mae_reduction=float(difference.mean()),
                    relative_mae_reduction=float(difference.mean() / ref["mean_mae"]),
                    folds_reselected_kan_lower=int((difference > 0).sum()),
                    relative_parameter_reduction=1-selected["mean_parameters_per_member"]/ref["mean_parameters_per_member"])
    csv = OUTPUT / "oof-predictions.csv"
    full.to_csv(csv, index=False, float_format="%.17g")
    pd.DataFrame(candidates).to_csv(OUTPUT / "candidate-comparison.csv", index=False, float_format="%.17g")
    assert native.file_hash(OUTPUT / "selection.json") == selection_sha
    native.write_json(OUTPUT / "summary.json", dict(samples=1265, unit="cm^-1", results=estimates,
                comparisons=comparisons, folds=audits, selected_configurations=choices,
                limitation=plan["limitation"], protocol_sha256=native.file_hash(OUTPUT / "protocol.json"),
                selection_sha256=selection_sha, fits_sha256=native.file_hash(OUTPUT / "fits.json"),
                predictions_sha256=native.file_hash(csv), scorer_sha256=native.file_hash(__file__),
                dense_summary_sha256=native.file_hash(DENSE / "summary.json"),
                all_model_weights_and_target_scalers_verified=True,
                max_validation_replay_error=max(r["max_validation_replay_error"] for r in candidates),
                members_at_epoch_limit=sum(r["members_at_epoch_limit"] for r in candidates)))
    for name, result in estimates.items():
        print(f"{name:25} MAE={result['mean_mae']:.6f} parameters/member={result['mean_parameters_per_member']:.2f}", flush=True)


if __name__ == "__main__":
    main()
