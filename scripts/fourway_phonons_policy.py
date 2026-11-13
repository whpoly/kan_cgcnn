"""Frozen, score-independent factorial design and complete-budget selection."""
import itertools
import math

ARMS = ('mm_modnet', 'mm_mv_modnet', 'mm_kan', 'mm_mv_kan')


def candidate_grid(smoke=False):
    if smoke:
        return [dict(candidate=i, n_feat=n, width=w, lr=.001, batch_size=32)
                for i, (n, w) in enumerate(((16, 8), (32, 16)))]
    return [dict(candidate=i, n_feat=n, width=w, lr=lr, batch_size=b)
            for i, (n, w, lr, b) in enumerate(itertools.product(
                (128, 256, 512, 800), (64, 96, 192, 320),
                (.0003, .001, .003, .01), (32, 64, 128)))]


def dense_widths(width):
    return [width, max(1, int(width * .75)), max(1, int(width * .375)),
            max(1, int(width * .375))]


def dense_parameters(n, widths):
    sizes = [n] + list(widths) + [1]
    return sum((a + 1) * b for a, b in zip(sizes, sizes[1:]))


def spline_parameters(n, a, b):
    return 8 * (n * a + a * b + b) + a + b + 1


def spline_widths(n, width):
    """Closest 17:8 shape within 5% of the paired native MODNet capacity.

    Uses parameter counts only, never any fit or validation score. At n=256,
    width=96 this preserves the previously frozen KAN 17:8 topology.
    """
    target = dense_parameters(n, dense_widths(width))
    choices = []
    for a in range(1, 513):
        for b in range(max(1, round(a * 8 / 17) - 8), round(a * 8 / 17) + 9):
            error = abs(spline_parameters(n, a, b) / target - 1)
            choices.append((error, abs(math.log(b / a / (8 / 17))), a, b))
    acceptable = [x for x in choices if x[0] <= .05]
    if not acceptable:
        raise ValueError('No capacity-matched KAN shape')
    _, _, a, b = min(acceptable, key=lambda x: (x[1], x[0], x[2], x[3]))
    return [a, b]


def choose(records, candidates, inner_count, keep):
    expected = {(c['candidate'], i) for c in candidates for i in range(inner_count)}
    keys = [(r['candidate'], r['inner']) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Selection requires the entire unique candidate x fold budget')
    if any(not math.isfinite(r['validation_mae']) for r in records):
        raise ValueError('Nonfinite fit cannot enter selection')
    ranked = []
    for c in candidates:
        rows = sorted((r for r in records if r['candidate'] == c['candidate']), key=lambda r: r['inner'])
        ranked.append(dict(candidate=c['candidate'], config=c,
            mean_validation_mae=sum(r['validation_mae'] for r in rows) / inner_count,
            members=[r['directory'] for r in rows]))
    ranked.sort(key=lambda r: (r['mean_validation_mae'], r['candidate']))
    selected = ranked[:keep]
    return dict(ranking=ranked, selected_candidates=[r['candidate'] for r in selected],
                members=[m for r in selected for m in r['members']])


def baseline_gate(maes, reference, smoke=False):
    if smoke:
        return dict(passed=True, smoke=True, establishes_reproduction=False)
    if len(maes) != 5 or any(not math.isfinite(x) for x in maes):
        raise ValueError('Five finite baseline folds required')
    mean = sum(maes) / 5
    return dict(passed=mean <= reference * 1.02, mean_mae=mean,
                published_mae=reference, maximum_mae=reference * 1.02,
                relative_deviation=mean / reference - 1, smoke=False,
                meaning='One predeclared baseline quality check; no test-score-driven retuning; not statistical equivalence')


def factorial_effects(scores):
    if set(scores) != set(ARMS):
        raise ValueError('All four cells required')
    a, b, c, d = [scores[x] for x in ARMS]
    return dict(mattervial_with_modnet=a-b, mattervial_with_kan=c-d,
                kan_with_matminer=a-c, kan_with_mattervial=b-d,
                total=a-d, interaction=(b-d)-(a-c))
