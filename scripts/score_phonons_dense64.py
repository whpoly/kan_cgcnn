"""Evaluate the frozen 64-descriptor Dense and KAN ensembles on identical phonons crystals."""

from benchmark_phonons_input_budget import ORIGINAL
from score_phonons_input_budget import restore
from train_phonons_dense64 import N_FEATURES, OUTPUT, PRIOR, REFERENCE, model_dir, native, read


np, pd = native.np, native.pd


def selections(plan, prior, fingerprint, records):
    """Choose each five-member configuration before opening outer targets or predictions."""
    choices = []
    for fold, previous in zip(plan["folds"], prior["folds"]):
        number = fold["fold"]
        names = fold["features"]
        assert number == previous["fold"] and names == previous["ranking"][:N_FEATURES]
        assert fold["official_test_ids"] == previous["official_test_ids"]
        validation_scores = {}
        for config in fold["configurations"]:
            name = config["name"]
            errors = []
            for inner, split in enumerate(fold["splits"]):
                path = model_dir(number, name, inner)
                record = records[number, name, inner]
                assert native.file_hash(path / "result.json") == record["result_sha256"]
                assert native.file_hash(path / "model.pkl") == record["model_sha256"]
                fit = read(path / "result.json")
                assert fit["run_fingerprint"] == fingerprint and fit["variant"] == "modnet" and not fit["fast"]
                assert fit["genes"] == dict(config["original_genes"], n_feat=N_FEATURES)
                assert fit["features"] == fit["selected_features"] == names and fit["seed"] == config["seeds"][inner]
                assert fit["train_ids"] == split["train_ids"] and fit["validation_ids"] == split["val_ids"]
                assert set(fit["train_ids"] + fit["validation_ids"]).isdisjoint(previous["test_ids"])
                errors.append(fit["validation_mae"])
            validation_scores[name] = float(np.mean(errors))
        dense_names = [c["name"] for c in fold["configurations"] if c["name"].startswith("dense-")]
        assert len(dense_names) == 10 and len(validation_scores) == 11
        chosen = min(dense_names, key=lambda name: (validation_scores[name], name))
        choices.append(dict(fold=number, mean_inner_mae=validation_scores, refitted_choice=chosen))
    return choices


def score_fold(fold, previous, choice, data, target, paired, kan_audit):
    number = fold["fold"]
    names, held, ids = fold["features"], previous["test_ids"], fold["official_test_ids"]
    selected = paired.loc[ids]
    truth = target.loc[held].to_numpy()
    assert (selected["fold"] == number).all()
    assert ids == [f"mb-phonons-{int(sid[2:]) + 1:04d}" for sid in held]
    np.testing.assert_allclose(selected["target"], truth, atol=1e-10, rtol=0)

    old = REFERENCE / "mm_modnet" / f"fold-{number}"
    archive = old / "outer-predictions.npz"
    assert native.file_hash(archive) == previous["modnet"]["prediction_sha256"]
    with np.load(archive) as z:
        np.testing.assert_array_equal(z["sample_ids"], ids)
        np.testing.assert_array_equal(z["target"], truth)
        np.testing.assert_allclose(z["prediction"], selected["mm_modnet"], rtol=0, atol=1e-10)
        frozen = read(old / "frozen-selection.json")["members"]
        best = fold["configurations"][0]
        kept = [i for i, m in enumerate(frozen) if (m["generation"], m["candidate"]) ==
                (best["generation"], best["candidate"])]
        assert len(kept) == 5
        dense_full_five = z["members"][kept].mean(axis=0)

    frame = data.loc[:, names]
    original_kan = next(row for row in kan_audit["fits"] if row["budget"] == N_FEATURES)
    kan_predictions = []
    for inner, sha in enumerate(original_kan["models_sha256"]):
        path = PRIOR / f"fold-{number}" / "k-064" / f"inner-{inner}"
        fit = read(path / "result.json")
        assert native.file_hash(path / "model.pkl") == sha and fit["variant"] == "bspline"
        assert fit["train_ids"] == fold["splits"][inner]["train_ids"]
        assert fit["validation_ids"] == fold["splits"][inner]["val_ids"]
        kan_predictions.append(restore(path, fit, names, frame.loc[held]))
    np.testing.assert_allclose(np.mean(kan_predictions, axis=0), selected["kan_64"], rtol=1e-5, atol=.003)

    config_predictions, parameters = {}, {}
    for config in fold["configurations"]:
        name = config["name"]
        members, params = [], []
        for inner in range(5):
            path = model_dir(number, name, inner)
            fit = read(path / "result.json")
            val = fit["validation_ids"]
            pred = restore(path, fit, names, frame.loc[val + held])
            np.testing.assert_allclose(np.mean(abs(pred[:len(val)] - target.loc[val].to_numpy())),
                                       fit["validation_mae"], rtol=1e-5, atol=.003)
            members.append(pred[len(val):])
            params.append(fit["parameters"])
        assert len(set(params)) == 1 and len(members) == 5
        assert np.asarray(members).shape == (5, len(ids))
        config_predictions[name] = np.asarray(members, dtype=float)
        parameters[name] = params[0]
    kept_names = [name for name in config_predictions if name.startswith("dense-")]
    dense_members = np.stack([config_predictions[name] for name in kept_names])
    scores = pd.DataFrame(dict(fold=number, sample_id=ids, target=truth, kan_64=selected["kan_64"].to_numpy(),
                               dense_64_five=config_predictions["dense-00"].mean(axis=0),
                               dense_64_inner_selected=config_predictions[choice["refitted_choice"]].mean(axis=0),
                               dense_64_fifty=dense_members.mean(axis=(0, 1)),
                               dense_64_kan_reference=config_predictions["kan-reference"].mean(axis=0),
                               dense_full_five=dense_full_five, dense_full_fifty=selected["mm_modnet"].to_numpy()))
    member_archive = OUTPUT / f"fold-{number}" / "outer-members.npz"
    np.savez_compressed(member_archive, sample_ids=ids, dense_configurations=kept_names, dense=dense_members,
                        kan_reference_dense=config_predictions["kan-reference"])
    return scores, dict(fold=number, parameters_by_config=parameters, predictions_sha256=native.file_hash(member_archive))


def main():
    plan = read(OUTPUT / "protocol.json")
    prior = read(PRIOR / "protocol.json")
    assert native.file_hash(PRIOR / "protocol.json") == plan["prior_protocol_sha256"]
    fingerprint = native.digest(plan)
    trained = read(OUTPUT / "fits.json")
    assert fingerprint == trained["protocol_fingerprint"]
    records = {(r["fold"], r["configuration"], r["inner"]): r for r in trained["records"]}
    assert len(records) == len(trained["records"]) == 275
    choices = selections(plan, prior, fingerprint, records)
    native.write_json(OUTPUT / "selection.json", dict(protocol_fingerprint=fingerprint, choices=choices))
    assert native.file_hash(ORIGINAL) == prior["raw_data_sha256"]
    assert native.file_hash(PRIOR / "oof-predictions.csv") == plan["paired_predictions_sha256"]
    paired = pd.read_csv(PRIOR / "oof-predictions.csv", float_precision="round_trip").set_index("sample_id")
    assert len(paired) == 1265 and paired.index.is_unique
    original = native.MODData.load(ORIGINAL)
    data = original.df_featurized.replace([np.inf, -np.inf], np.nan).fillna(0)
    target = original.df_targets.iloc[:, 0]
    kan_audit = read(PRIOR / "model-audit.json")
    rows, audits = [], []
    for f, prev, choice, kan in zip(plan["folds"], prior["folds"], choices, kan_audit):
        assert f["fold"] == choice["fold"] == kan["fold"]
        row, audit = score_fold(f, prev, choice, data, target, paired, kan)
        rows.append(row)
        audits.append(audit)
        print(f"Verified {len(row)} held-out crystals in fold {f['fold']}", flush=True)
    full = pd.concat(rows, ignore_index=True)
    assert full["sample_id"].is_unique and set(full["sample_id"]) == set(paired.index)
    assert full.groupby("fold").size().tolist() == [253] * 5
    estimates = {}
    for name in full.columns.drop(["fold", "sample_id", "target"]):
        errors = (full[name] - full.target).abs()
        fold_mae = errors.groupby(full["fold"]).mean().to_numpy()
        np.testing.assert_allclose(errors.mean(), fold_mae.mean(), rtol=0, atol=1e-10)
        estimates[name] = dict(mean_mae=float(fold_mae.mean()), fold_mae=fold_mae)
    assert np.isfinite(full.drop(columns="sample_id").to_numpy()).all()
    folds_kan = estimates["kan_64"]["fold_mae"]
    comparisons = {}
    for name in ("dense_64_five", "dense_64_inner_selected", "dense_64_fifty", "dense_64_kan_reference"):
        diff = estimates[name]["fold_mae"] - folds_kan
        comparisons[name] = dict(dense_minus_kan=diff, mean_dense_minus_kan=float(diff.mean()),
                                 kan_reduction_relative_to_dense=float(diff.mean() / estimates[name]["mean_mae"]),
                                 folds_kan_lower=int((diff > 0).sum()))
    csv = OUTPUT / "oof-predictions.csv"
    full.to_csv(csv, index=False, float_format="%.17g")
    native.write_json(OUTPUT / "summary.json", dict(samples=len(full), unit="cm^-1", results=estimates,
                      comparisons=comparisons, folds=audits, selected_configurations=choices,
                      protocol_sha256=native.file_hash(OUTPUT / "protocol.json"),
                      predictions_sha256=native.file_hash(csv), scorer_sha256=native.file_hash(__file__)))
    for name, result in estimates.items():
        print(f"{name:25} {result['mean_mae']:.6f}", flush=True)


if __name__ == "__main__":
    main()
