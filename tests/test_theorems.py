"""Monte Carlo verification of Theorems 1 and 2.

Both theorems state: for concave, monotone increasing gratification functions
and i.i.d. thresholds Lambda_1, Lambda_2, the expected double-jerk stimulation
rate

    E[R_S] = E[ min( S(f1)/Lambda_1, S(1 - f1)/Lambda_2 ) ]

is maximized at f1 = 1/2, achieving the bound S(1/2) * E[min(1/Lambda_1,
1/Lambda_2)]. Theorem 1 is the shaft-to-shaft statement (spatial fractions,
girth matching); Theorem 2 is the identical statement for tip-to-tip
(temporal fractions, shaft-length matching).
"""

import numpy as np
import pytest

from tiptotip.gratification import S_sqrt, sample_thresholds


def expected_rate(f1, lam1, lam2, S=S_sqrt):
    return np.minimum(S(f1) / lam1, S(1.0 - f1) / lam2).mean()


@pytest.fixture(scope="module")
def thresholds():
    rng = np.random.default_rng(42)
    n = 400_000
    return sample_thresholds(rng, n), sample_thresholds(rng, n)


@pytest.mark.parametrize("S", [S_sqrt, lambda f: np.clip(f, 0, 1) ** 0.25,
                               lambda f: 1 - (1 - np.clip(f, 0, 1)) ** 2])
def test_rate_is_maximized_at_equal_split(thresholds, S):
    """Sweep f1 over [0, 1]: the empirical maximum sits at f1 = 1/2 for every
    concave monotone S tried (the paper's sqrt and two others)."""
    lam1, lam2 = thresholds
    grid = np.linspace(0.0, 1.0, 101)
    rates = np.array([expected_rate(f, lam1, lam2, S) for f in grid])
    assert grid[np.argmax(rates)] == pytest.approx(0.5, abs=0.011)
    # and the profile is symmetric: E[R_S](f) == E[R_S](1-f) in distribution
    assert rates[20] == pytest.approx(rates[80], rel=0.02)


def test_maximum_achieves_the_theorem_bound(thresholds):
    """At f1 = 1/2 the rate equals S(1/2) * E[min(1/Lambda_1, 1/Lambda_2)],
    the bound of Theorems 1 and 2, and no other split exceeds it."""
    lam1, lam2 = thresholds
    bound = S_sqrt(0.5) * np.minimum(1.0 / lam1, 1.0 / lam2).mean()
    assert expected_rate(0.5, lam1, lam2) == pytest.approx(bound, rel=1e-12)
    for f in np.linspace(0.0, 1.0, 41):
        assert expected_rate(f, lam1, lam2) <= bound * (1 + 1e-9)


def test_bound_requires_identical_distributions():
    """The proof of Theorem 1 leans on Lambda_1, Lambda_2 being identically
    distributed (the symmetrization step). With mismatched distributions the
    optimum genuinely moves away from 1/2 — confirming the assumption is
    load-bearing, not decorative."""
    rng = np.random.default_rng(7)
    n = 400_000
    lam1 = rng.gamma(2.0, 0.5, n)   # mean 1.0
    lam2 = rng.gamma(2.0, 2.0, n)   # mean 4.0 — much needier
    grid = np.linspace(0.01, 0.99, 99)
    rates = np.array([expected_rate(f, lam1, lam2) for f in grid])
    assert grid[np.argmax(rates)] < 0.45  # more hand for the needier member
