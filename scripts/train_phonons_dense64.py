"""Refit the selected Phonons MM Dense MODNet configurations at exactly 64 inputs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain
from pathlib import Path

from benchmark_phonons_input_budget import OUTPUT as PRIOR, REFERENCE, native, read, training_data
from modnet.hyper_opt.fit_genetic import Individual
from native_fourway_models import dense_count


OUTPUT = PRIOR.parent / "phonons-mm-dense64-20261114"
N_FEATURES = 64


def model_dir(fold, configuration, inner):
    return OUTPUT / f"fold-{fold}" / configuration / f"inner-{inner}"


def fold_plan(fold):
    number = fold["fold"]
    folder = REFERENCE / "mm_modnet" / f"fold-{number}"
    source = folder / "frozen-selection.json"
    assert native.file_hash(source) == fold["modnet"]["selection_sha256"]
    saved = read(source)
    assert saved["test_labels_used"] is False
    frozen = saved["members"]
    generations = Path(frozen[0]["directory"]).parent.parent
    ranking = sorted((score, gen, i) for gen in range(10) for i, score in
                     enumerate(read(generations / f"generation-{gen:02d}.json")["mean_validation_mae"]))
    assert len(frozen) == 50 and len(ranking) == 200
    configurations = []
    for rank, (_, gen, candidate) in enumerate(ranking[:10]):
        group = sorted((m for m in frozen if (m["generation"], m["candidate"]) == (gen, candidate)),
                       key=lambda m: m["inner"])
        assert [m["inner"] for m in group] == list(range(5))
        genes = group[0]["genes"]
        for member, split in zip(group, fold["members"]):
            assert member["genes"] == genes and not member["smoke"]
            assert member["train_ids"] == split["train_ids"] and member["validation_ids"] == split["val_ids"]
        configurations.append(dict(name=f"dense-{rank:02d}", generation=gen, candidate=candidate,
                                   original_genes=genes, seeds=[m["seed"] for m in group]))
    kan = fold["members"]
    assert all(m["genes"] == kan[0]["genes"] for m in kan)
    configurations.append(dict(name="kan-reference", generation=fold["best_generation"],
                               candidate=fold["best_candidate"], original_genes=kan[0]["genes"],
                               seeds=[m["seed"] for m in kan]))
    return dict(fold=number, features=fold["ranking"][:N_FEATURES], cache=fold["cache"],
                cache_sha256=fold["cache_sha256"], original_dense_selection_sha256=native.file_hash(source),
                official_test_ids=fold["official_test_ids"],
                splits=[dict(train_ids=m["train_ids"], val_ids=m["val_ids"]) for m in kan],
                configurations=configurations)


def freeze():
    prior = read(PRIOR / "protocol.json")
    summary = read(PRIOR / "summary.json")
    assert native.file_hash(PRIOR / "protocol.json") == summary["protocol_sha256"]
    assert native.file_hash(PRIOR / "oof-predictions.csv") == summary["predictions_sha256"]
    plan = dict(features=N_FEATURES, task="matbench_phonons", bank="mm", trainer_sha256=native.file_hash(__file__),
                prior_protocol_sha256=summary["protocol_sha256"], paired_predictions_sha256=summary["predictions_sha256"],
                training_rule="Native Dense MODNet, raw target, original per-configuration optimizer, scaling, seeds "
                              "and train-loss stopping. Only n_feat is changed to 64. No outer data in fitting.",
                comparisons=dict(frozen_five="Primary. Five inner members of the original validation-winning Dense configuration.",
                                 inner_selected_five="Five inner members of the original shortlist configuration with lowest "
                                                     "new mean inner validation MAE at 64; ties use lower original rank.",
                                 original_fifty="All ten original Dense configurations, five inner members each, at 64.",
                                 kan_reference_five="Dense widths for the original KAN parameter match, with its genes "
                                                    "and seeds. At 64 the trainable parameter counts can differ."),
                folds=[fold_plan(f) for f in prior["folds"]])
    path = OUTPUT / "protocol.json"
    if path.exists() and read(path) != plan:
        raise ValueError("The Dense training protocol differs from the saved run")
    native.write_json(path, plan)
    return plan


def fit(job):
    fold, config, inner, fingerprint = job
    data, positions = training_data(fold["cache"], fold["cache_sha256"])
    names = fold["features"]
    assert len(names) == len(set(names)) == N_FEATURES and names == list(data.get_optimal_descriptors())[:N_FEATURES]
    split = fold["splits"][inner]
    assert set(split["train_ids"]).isdisjoint(split["val_ids"])
    assert set(split["train_ids"] + split["val_ids"]) == set(positions)
    tr, va = data.split(tuple([positions[sid] for sid in split[key]] for key in ("train_ids", "val_ids")))
    tr.optimal_features, va.optimal_features = names, names
    target = list(tr.target_names)
    individual = Individual(max_feat=N_FEATURES, targets=[[target]], weights={t: 1 for t in target},
                            num_classes=tr.num_classes, multi_label=False)
    individual.genes = dict(config["original_genes"], n_feat=N_FEATURES)
    directory = model_dir(fold["fold"], config["name"], inner)
    result = native.evaluate_job(dict(individual=individual, train_data=tr, val_data=va, directory=str(directory),
                                      seed=config["seeds"][inner], fast=False, variant="modnet",
                                      run_fingerprint=fingerprint))
    assert result["variant"] == "modnet" and result["genes"] == individual.genes
    assert result["train_ids"] == split["train_ids"] and result["validation_ids"] == split["val_ids"]
    assert result["features"] == result["selected_features"] == names
    assert result["parameters"] == dense_count(N_FEATURES, [v[0] for v in result["num_neurons"]])
    return dict(fold=fold["fold"], configuration=config["name"], inner=inner, epochs=result["epochs"],
                model_sha256=native.file_hash(directory / "model.pkl"), result_sha256=native.file_hash(directory / "result.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    plan = freeze()
    fingerprint = native.digest(plan)
    # Finish the paired five-member comparisons first, then the rest of the original ensemble.
    config_order = chain([0, 10], range(1, 10))
    jobs = [(f, f["configurations"][i], inner, fingerprint) for i in config_order
            for f in plan["folds"] for inner in range(5)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fit, job) for job in jobs]
        for done in as_completed(futures):
            row = done.result()
            records.append(row)
            print(f"{len(records)}/{len(jobs)} fold={row['fold']} {row['configuration']} "
                  f"inner={row['inner']} epochs={row['epochs']}", flush=True)
    assert len(records) == 275
    records.sort(key=lambda r: (r["fold"], r["configuration"], r["inner"]))
    native.write_json(OUTPUT / "fits.json", dict(protocol_fingerprint=fingerprint, records=records))


if __name__ == "__main__":
    main()
