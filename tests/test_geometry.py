"""Verify the geometric claims of Sec. 1.2."""

import numpy as np
import pytest

from tiptotip.geometry import (
    contact_angle_formula,
    contact_angle_numeric,
    coverage_fractions,
    small_shaft_arc_numeric,
    tip_to_tip_penalty,
    shaft_to_shaft_penalty,
)

RADII = [(1.0, 1.0), (0.5, 1.0), (0.9, 1.1), (0.25, 2.0), (1e-3, 1.0), (0.999, 1.0)]


@pytest.mark.parametrize("r,R", RADII)
def test_contact_angle_formula_matches_tangent_construction(r, R):
    """theta = pi + 2*arcsin((R-r)/(R+r)) agrees with an independent numeric
    construction of the taut band's external tangent lines."""
    assert contact_angle_formula(r, R) == pytest.approx(
        contact_angle_numeric(r, R), abs=1e-10
    )


def test_equal_girth_gives_half_coverage_each():
    f, F = coverage_fractions(1.0, 1.0)
    assert f == pytest.approx(0.5)
    assert F == pytest.approx(0.5)


@pytest.mark.parametrize("r,R", RADII)
def test_coverage_fractions_sum_to_one(r, R):
    """Paper Eq. (2): the hand always covers exactly one full shaft's worth
    of circumference in total. Checked against the numeric construction."""
    f, F = coverage_fractions(r, R)
    assert f + F == pytest.approx(1.0)
    theta_small = small_shaft_arc_numeric(r, R)
    assert theta_small / (2 * np.pi) == pytest.approx(f, abs=1e-10)


def test_disparate_girths_starve_the_smaller_shaft():
    """As R/r grows, the larger shaft hogs the hand; coverage of the smaller
    shaft falls monotonically toward zero."""
    ratios = np.linspace(1.0, 200.0, 500)
    f = np.array([coverage_fractions(1.0, R)[0] for R in ratios])
    assert np.all(np.diff(f) < 0)
    assert f[-1] < 0.06


def test_concavity_makes_double_jerking_beneficial():
    """Sec. 1.2.1's key observation: for concave S with S(0)=0, jerking two
    shafts at half contact beats one shaft at full contact,
    2*S(1/2) >= S(1), strictly for strictly concave S."""
    for a in (0.25, 0.5, 0.75, 0.99):
        S = lambda x: x**a
        assert 2 * S(0.5) >= S(1.0)
    assert 2 * np.sqrt(0.5) > 1.0  # the paper's sqrt choice: 2*S(1/2) = sqrt(2)


def test_shaft_to_shaft_pays_greater_mismatch_penalty():
    """Sec. 1.2: 'a greater penalty for mismatch is paid in the shaft-to-shaft
    scenario' — for every vertical displacement delta."""
    ell, L = 4.5, 6.5
    delta = np.linspace(0.0, ell + L, 200)
    t2t = tip_to_tip_penalty(ell, L, delta)
    s2s = shaft_to_shaft_penalty(ell, L, delta)
    assert np.all(s2s <= t2t + 1e-12)


def test_no_jerking_when_shafts_cannot_bridge_the_gap():
    ell, L = 4.5, 6.5
    assert tip_to_tip_penalty(ell, L, ell + L + 0.1) == 0.0
    assert shaft_to_shaft_penalty(ell, L, L + 0.1) == 0.0
    # tip-to-tip remains feasible in the region where shaft-to-shaft fails
    assert tip_to_tip_penalty(ell, L, L + 0.1) > 0.0


def test_zero_displacement_is_penalty_free():
    assert tip_to_tip_penalty(5.0, 5.0, 0.0) == pytest.approx(1.0)
    assert shaft_to_shaft_penalty(5.0, 5.0, 0.0) == pytest.approx(1.0)
