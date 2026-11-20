"""Deterministic MAE-band ranking with an explicit parameter preference."""
import math


TOLERANCE = 0.01


def accuracy_size_order(losses, parameters, tolerance=TOLERANCE):
    """Rank the best remaining 1%-MAE band by size, then open the next band.

    Anchoring each band to its minimum avoids non-transitive pairwise ties.
    The first-ranked candidate is always within tolerance of the global best.
    Remaining bands allow a fixed-size ensemble without duplicating models.
    """
    assert len(losses) == len(parameters) > 0
    assert 0 <= tolerance < 1
    assert all(math.isfinite(float(x)) and x >= 0 for x in losses)
    assert all(int(x) == x and x > 0 for x in parameters)
    remaining = set(range(len(losses)))
    order, bands = [], []
    while remaining:
        anchor = min(float(losses[i]) for i in remaining)
        limit = anchor * (1 + tolerance)
        eligible = [i for i in remaining if losses[i] <= limit + 1e-12]
        eligible.sort(key=lambda i: (parameters[i], losses[i], i))
        bands.append(dict(anchor_mae=anchor, maximum_mae=limit, candidate_indices=eligible))
        order.extend(eligible)
        remaining.difference_update(eligible)
    return order, bands


def parent_weights(size):
    """Positive rank weights: better accuracy/size ranks are more likely."""
    assert size > 0
    return [1.0 / (rank + 1) for rank in range(size)]


def champion_state(losses, order):
    """Track both the compact preferred candidate and best-accuracy candidate."""
    return (order[0], min(range(len(losses)), key=lambda i: (losses[i], i)))


def should_stop(history, patience=4):
    return len(history) >= patience and len({tuple(x) for x in history[-patience:]}) == 1


def pareto_indices(losses, parameters):
    """Indices not dominated on both MAE and size, retaining exact ties."""
    return [i for i in range(len(losses)) if not any(
        losses[j] <= losses[i] and parameters[j] <= parameters[i]
        and (losses[j] < losses[i] or parameters[j] < parameters[i])
        for j in range(len(losses)))]


def self_check():
    # A 0.5% MAE increase buys a tenfold reduction; a 1.1% increase does not.
    losses, sizes = [100., 100.5, 101.1], [100000, 10000, 1000]
    order, bands = accuracy_size_order(losses, sizes)
    assert order == [1, 0, 2]
    assert bands[0]['candidate_indices'] == [1, 0]
    assert accuracy_size_order([100., 101.], [100000, 1000])[0] == [1, 0]
    assert accuracy_size_order([100., 100.5], [1000, 1000])[0] == [0, 1]
    assert accuracy_size_order([0., 0., .001], [1000, 100, 1])[0] == [1, 0, 2]
    assert pareto_indices(losses, sizes) == [0, 1, 2]
    assert pareto_indices([100., 101., 99.], [1000, 2000, 3000]) == [0, 2]
    assert not should_stop([(0, 0)] * 3)
    assert should_stop([(0, 0)] * 4)
    assert not should_stop([(0, 0), (0, 0), (0, 0), (1, 0)])
    assert not should_stop([(0, 0), (0, 0), (0, 0), (0, 1)])
    assert not should_stop([(0, 0), (1, 0), (0, 0), (0, 0)])
    weights = parent_weights(20)
    assert all(a > b > 0 for a, b in zip(weights, weights[1:]))
    return dict(passed=True, relative_mae_tolerance=TOLERANCE,
        similar_mae_prefers_smaller=True, outside_band_accuracy_takes_priority=True,
        exact_boundary_and_ties_verified=True, no_pairwise_tie_cycles=True,
        smaller_champion_or_accuracy_improvement_resets_early_stop=True,
        example=dict(mae=losses, parameters=sizes, ranking=order))
