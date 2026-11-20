"""KAN width limits derived from each fixed input count, never a global cap."""


PROFILES = ('funnel', 'moderate_expansion')
DEFAULT_PROFILE = 'funnel'


def limits(n, profile='funnel'):
    assert isinstance(n, int) and n > 0 and profile in PROFILES
    if profile == 'funnel':
        return n, max(1, n // 2)
    return 2 * n, n


def initial_values(maximum):
    assert maximum >= 1
    values, width = [], 1
    while width <= maximum:
        values.append(width)
        width *= 2
    return sorted(set(values + [maximum]))


def project(n, a, b, profile='funnel'):
    a_max, b_max = limits(n, profile)
    a = int(max(1, min(a_max, a)))
    b = int(max(1, min(b_max, a, b)))
    return a, b


def valid(n, a, b, profile='funnel'):
    a_max, b_max = limits(n, profile)
    return type(a) is int and type(b) is int and 1 <= a <= a_max and 1 <= b <= min(a, b_max)


def describe(profile='funnel'):
    assert profile in PROFILES
    return dict(profile=profile,
        a='1 <= a <= n' if profile == 'funnel' else '1 <= a <= 2*n',
        b='1 <= b <= min(a,max(1,floor(n/2)))' if profile == 'funnel' else '1 <= b <= min(a,n)',
        initialization='Powers of two within each conditional bound, plus the exact bound.',
        crossover='Uniform gene crossover followed by projection of b to min(a,input-dependent b maximum).',
        mutation='Quarter-width steps, then project both widths back into input-dependent bounds.',
        fixed_width_ratio=False)
