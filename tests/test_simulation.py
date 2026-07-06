"""Verify the ordinal claims of Sec. 4 / Fig. 8."""

import numpy as np
import pytest

from tiptotip.simulate import sample_audience, run_presenter, variance_sweep


def test_per_hand_speedup_is_capped_at_sqrt_two():
    """With an identical audience, tip-to-tip double jerking gives each member
    T(1/2) = sqrt(1/2) gratification per jerk, so a pair finishes in
    Lambda/sqrt(.5) jerks versus 2*Lambda singly: per-hand speedup is exactly
    sqrt(2), and threshold heterogeneity only lowers it (the pair is held
    hostage by its needier member). This is why the paper's Fig. 8 performance
    of ~2 must be a two-hands-vs-one comparison: per hand, 2*T(1/2) = sqrt(2)
    is a hard ceiling."""
    rng = np.random.default_rng(0)
    aud = sample_audience(rng, 1000, variance_scale=0.0)
    aud["lam"][:] = 1.0  # remove threshold noise to isolate the geometry
    baseline = aud["lam"].sum()
    perf = baseline / run_presenter(aud, "A")
    assert perf == pytest.approx(np.sqrt(2.0), rel=1e-9)

    # heterogeneous thresholds can only hurt
    aud["lam"] = rng.gamma(2.0, 0.5, 1000)
    perf_het = aud["lam"].sum() / run_presenter(aud, "A")
    assert perf_het < np.sqrt(2.0)


def test_sorting_dominates_under_variance():
    """The paper's headline claim: at high variance, presenter A (leg+shaft
    sorted) beats B (leg sorted) beats C (unsorted)."""
    res = variance_sweep([1.0], n=1500, trials=10, seed=99)
    a, b, c = res["A"][0], res["B"][0], res["C"][0]
    assert a > b > c


def test_sorted_presenter_stays_stiff_as_variance_grows():
    """'Presenter A remains strong even in the presence of member variation,
    [while] B and C demonstrate increasingly flaccid performance.'"""
    variances = [0.0, 0.5, 1.0]
    res = variance_sweep(variances, n=1500, trials=10, seed=123)
    drop = lambda p: res[p][0] - res[p][-1]
    assert drop("A") < 0.2                 # A: nearly flat
    assert drop("C") > 2 * drop("A")       # C: increasingly flaccid
    assert np.all(np.diff(res["C"]) < 0)   # C decays monotonically


def test_presenters_identical_when_audience_is_uniform():
    """With zero variance there is nothing to sort: all presenters tie."""
    rng = np.random.default_rng(5)
    aud = sample_audience(rng, 1000, variance_scale=0.0)
    totals = {p: run_presenter(aud, p, rng=np.random.default_rng(1)) for p in "ABC"}
    assert totals["A"] == pytest.approx(totals["B"], rel=1e-6)
    # C pairs in random order, but with identical geometry only the Lambda
    # pairing differs — allow the small stochastic wiggle
    assert totals["C"] == pytest.approx(totals["A"], rel=0.05)
