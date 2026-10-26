"""Uncertainty intervals for M4 (eval/methodology.md "Uncertainty"; playbook, milestone 03).

- Wilson 95% interval for proportions.
- Bootstrap 95% interval (resample rows with replacement, 2.5th / 97.5th percentiles) for
  latency, cost and ratio statistics such as question-clustered claim support.
- Paired bootstrap for differences between two systems that answered the same questions: each
  resample draws ONE set of question indices and recomputes both systems on it.
Pure Python, seeded, so every interval is reproducible."""

import math
import random
import statistics
from collections.abc import Callable, Sequence

Z95 = 1.96
N_RESAMPLES = 2000
SEED = 20261024


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for successes / n; None when n is 0."""
    if n < 0 or not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n")
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    """2.5th and 97.5th percentiles (linear interpolation between order statistics)."""
    if len(values) < 2:
        raise ValueError("need at least 2 bootstrap values")
    cuts = statistics.quantiles(values, n=40, method="inclusive")
    return cuts[0], cuts[-1]


def _interval(values: list[float]) -> list[float] | None:
    """None when fewer than 2 resamples had a defined statistic (e.g. a system with no items)."""
    return list(percentile_interval(values)) if len(values) >= 2 else None


def resample_indices(n: int, n_resamples: int = N_RESAMPLES, seed: int = SEED):
    """The index lists every bootstrap here draws from: same seed, same resamples."""
    if n < 1:
        raise ValueError("nothing to resample")
    rng = random.Random(seed)
    for _ in range(n_resamples):
        yield rng.choices(range(n), k=n)


def bootstrap_ci(
    rows: Sequence, stat: Callable[[list], float | None], n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> dict:  # fmt: skip
    """Point estimate of `stat(rows)` and its bootstrap 95% interval. Resamples where `stat`
    returns None (e.g. nothing answered) are skipped and counted."""
    values = [v for idx in resample_indices(len(rows), n_resamples, seed)
              if (v := stat([rows[i] for i in idx])) is not None]  # fmt: skip
    return {"estimate": stat(list(rows)), "ci95": _interval(values), "resamples_used": len(values),
            "resamples": n_resamples, "seed": seed}  # fmt: skip


def paired_bootstrap_ci(
    rows_a: Sequence, rows_b: Sequence, stat: Callable[[list], float | None],
    n_resamples: int = N_RESAMPLES, seed: int = SEED,
) -> dict:  # fmt: skip
    """Difference stat(B) - stat(A) for two systems on the same questions (row i of A and row i
    of B are the same question) and its paired bootstrap 95% interval."""
    if len(rows_a) != len(rows_b):
        raise ValueError("paired rows must cover the same questions")
    diffs = []
    for idx in resample_indices(len(rows_a), n_resamples, seed):
        a, b = stat([rows_a[i] for i in idx]), stat([rows_b[i] for i in idx])
        if a is not None and b is not None:
            diffs.append(b - a)
    ci = _interval(diffs)
    a, b = stat(list(rows_a)), stat(list(rows_b))
    return {"estimate": None if a is None or b is None else b - a, "ci95": ci,
            "includes_zero": None if ci is None else ci[0] <= 0 <= ci[1],
            "resamples_used": len(diffs),
            "resamples": n_resamples, "seed": seed}  # fmt: skip
