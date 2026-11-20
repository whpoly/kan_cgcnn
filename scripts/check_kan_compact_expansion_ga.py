"""Exercise accuracy/size ranking, actual GA parent ordering and early stopping."""
import ast
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import kan_accuracy_size_policy as policy
import run_kan_compact_expansion_ga as runner


def main():
    out = runner.ROOT/'benchmarks/kan-compact-expansion-ga-review-20261117'
    out.mkdir(parents=True, exist_ok=True)
    report = policy.self_check()
    report['width_operator_selfcheck'] = runner.self_check()
    wp = runner.width_policy
    assert runner.WIDTH_PROFILE == 'moderate_expansion'
    assert wp.limits(16,'funnel') == (16,8)
    assert wp.limits(48,'funnel') == (48,24)
    assert wp.limits(112,'funnel') == (112,56)
    assert wp.limits(192,'funnel') == (192,96)
    assert wp.limits(16,'moderate_expansion') == (32,16)
    assert wp.limits(48,'moderate_expansion') == (96,48)
    assert wp.limits(112,'moderate_expansion') == (224,112)
    assert wp.limits(192,'moderate_expansion') == (384,192)
    assert wp.valid(16,32,16,'moderate_expansion')
    assert wp.valid(192,384,192,'moderate_expansion')
    assert not wp.valid(16,33,16,'moderate_expansion')
    assert not wp.valid(16,32,17,'moderate_expansion')
    assert wp.project(48,1000,1000,'moderate_expansion') == (96,48)
    assert wp.project(48,4,48,'moderate_expansion') == (4,4)
    assert wp.initial_values(96) == [1,2,4,8,16,32,64,96]
    assert wp.project(16,1000,1000,'funnel') == (16,8)
    assert wp.project(16,2,8,'funnel') == (2,2)
    assert wp.project(16,0,-1,'funnel') == (1,1)
    assert wp.initial_values(48) == [1,2,4,8,16,32,48]
    assert not wp.valid(16,17,8,'funnel') and not wp.valid(16,16,9,'funnel')
    assert not wp.valid(16,2,3,'funnel')
    rng = random.Random(20261117)
    for _ in range(500):
        losses = [rng.uniform(10, 11) for _ in range(40)]
        parameters = [rng.randint(1, 100000) for _ in losses]
        order, bands = policy.accuracy_size_order(losses, parameters)
        assert sorted(order) == list(range(len(losses)))
        assert losses[order[0]] <= min(losses)*1.01 + 1e-12
        best_band = [i for i, loss in enumerate(losses) if loss <= min(losses)*1.01]
        assert parameters[order[0]] == min(parameters[i] for i in best_band)
        mae_only = sorted(range(len(losses)), key=lambda i: losses[i])[:10]
        assert sum(parameters[i] for i in order[:10]) <= sum(parameters[i] for i in mae_only)
        assert sum(losses[i] for i in order[:10]) <= 1.01*sum(losses[i] for i in mae_only)+1e-10
        rank = {index: pos for pos, index in enumerate(order)}
        for a in range(len(losses)):
            for b in range(len(losses)):
                if losses[a] < losses[b] and parameters[a] <= parameters[b]:
                    assert rank[a] < rank[b]
    report['random_tradeoff_cases'] = 500
    report['selected_mean_candidate_mae_within_1pct_of_mae_only_top10'] = True
    report['selected_parameters_never_exceed_mae_only_top10_same_archive'] = True

    old = ast.parse((runner.ROOT/'scripts/run_kan_direct_width_ga.py').read_text())
    new = ast.parse(Path(runner.__file__).read_text())
    for name in ('direct_fit', 'run_fit', 'prepare'):
        nodes = [next(n for n in tree.body if getattr(n, 'name', None) == name) for tree in (old, new)]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
    report['training_and_splits_unchanged'] = True
    prior = ast.parse((runner.ROOT/'scripts/run_kan_compact_dim_ga.py').read_text())
    for name in ('DirectIndividual','DirectGA','verify_genes','stopping_audit','self_check'):
        nodes = [next(n for n in tree.body if getattr(n,'name',None)==name) for tree in (prior,new)]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
    report['only_width_profile_changed_in_selection_and_operators'] = True

    class Probe(runner.DirectGA):
        def __init__(self):
            self.directory = out/'synthetic-integration'
            self.directory.mkdir(parents=True, exist_ok=True)
            self.generation = 0

        def _init_run(self, **kwargs):
            self.pool = SimpleNamespace(close=lambda: None, join=lambda: None, terminate=lambda: None)

        def initialization_population(self, size_pop, multi_label):
            self.pop = [runner.DirectIndividual(16, {'y':0}, targets=[[['y']]], weights={'y':1}) for _ in range(size_pop)]
            for i, ind in enumerate(self.pop):
                a, b = [(16,8),(4,2),(1,1)][min(i,2)]
                ind.genes.update(kan_a=a, kan_b=b)
                runner.verify_genes(ind.genes,16)

        def function_fitness(self, population, nested, **kwargs):
            rows, models, individuals = [], [], []
            for candidate, source in enumerate(population):
                # No new candidate improves either champion after generation 0.
                loss = ([100.,100.5,101.1] + [200.]*17)[candidate] if self.generation == 0 else 300.+candidate
                refs = []
                for inner in range(nested):
                    folder = self.directory/'fits'/f'generation-{self.generation:02d}-candidate-{candidate:02d}-inner-{inner}'
                    folder.mkdir(parents=True, exist_ok=True)
                    g = source.genes
                    runner.write_json(folder/'result.json', dict(genes=g, validation_mae=loss,
                        parameters=runner.spline.spline_count(g['n_feat'],g['kan_a'],g['kan_b'])))
                    refs.append(runner.engine.ModelReference(str(folder/'model.pkl'), source.targets,source.weights,source.num_classes))
                ind = copy.copy(source)
                ind.model, ind.val_loss = refs[-1], loss
                rows.append([loss]*nested)
                individuals.append(ind)
                models.append(runner.engine.EnsembleMODNetModel(modnet_models=refs))
            values = runner.np.asarray(rows)
            runner.write_json(self.directory/f'generation-{self.generation:02d}.json', dict(
                genes=[i.genes for i in population], validation_mae=values, mean_validation_mae=values.mean(axis=1)))
            self.generation += 1
            return values.mean(axis=1), runner.np.asarray(models), runner.np.asarray(individuals)

    original_choices = runner.random.choices
    parent_calls = []
    def checked_choices(population, weights, k):
        assert population[0].genes['kan_a'] == 4 and population[0].genes['kan_b'] == 2
        assert weights[0] == 1 and weights[1] == .5
        parent_calls.append(len(population))
        return original_choices(population, weights=weights, k=k)

    runner.set_seed(20261117)
    probe = Probe()
    with patch.object(runner.random, 'choices', checked_choices):
        ensemble = probe.run(n_jobs=1, **runner.GA)
    selected = [dict(directory=str(Path(ref.path).parent)) for ref in ensemble.model]
    audit = runner.stopping_audit(probe.directory, selected, {'smoke':False})
    assert audit['passed'] and audit['generations'] == 4 and len(selected) == 50
    assert parent_calls == [20,20,40,40,60,60]
    assert Path(selected[0]['directory']).name == 'generation-00-candidate-01-inner-0'
    report.update(actual_ga_parent_weighting_and_final_selection_verified=True,
        actual_ga_early_stop_generations=4, ensemble_members=50,
        source_sha256={str(path):runner.file_hash(path) for path in
            (Path(runner.__file__),Path(policy.__file__),Path(wp.__file__),Path(__file__))})
    runner.write_json(out/'policy-integration-audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
