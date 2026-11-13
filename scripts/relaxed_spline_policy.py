"""Preserve every feasible legacy mapping; otherwise minimize parameter error.

This is an architecture-only decision, independent of training/validation scores.
The fallback accepts either more or fewer parameters and has no fatal tolerance.
"""
import math
from functools import lru_cache


def dense_count(n, widths):
    sizes = [n] + list(widths) + [1]
    return sum((a + 1) * b for a, b in zip(sizes, sizes[1:]))


def spline_count(n, a, b):
    return 8 * (n * a + a * b + b) + a + b + 1


@lru_cache(None)
def mapping(n, widths):
    target = dense_count(n, widths)
    if n < 1 or target <= 0:
        raise ValueError('Invalid feature count or reference parameter count')
    feasible, closest = [], None
    for a in range(1, max(513, int(math.sqrt(target)) + 2)):
        for b in range(max(1, round(a * 8 / 17) - 8), round(a * 8 / 17) + 9):
            count = spline_count(n, a, b)
            error = abs(count / target - 1)
            ratio_error = abs(math.log(b / a / (8 / 17)))
            if error <= .05:
                feasible.append((ratio_error, error, a, b))
            choice = (error, ratio_error, a, b)
            if closest is None or choice < closest:
                closest = choice
    if feasible:
        _, _, a, b = min(feasible)
        return a, b, False
    _, _, a, b = closest
    return a, b, True


def matched_widths(n, widths):
    return mapping(int(n), tuple(widths))[:2]


def describe(n, widths):
    a, b, relaxed = mapping(int(n), tuple(widths))
    reference = dense_count(n, widths)
    count = spline_count(n, a, b)
    return dict(widths=[a,b,1], fallback_used=relaxed, reference_parameters=reference,
        spline_parameters=count, signed_relative_difference=count/reference-1,
        rule='legacy_5pct' if not relaxed else 'closest_integer_network_either_direction')
