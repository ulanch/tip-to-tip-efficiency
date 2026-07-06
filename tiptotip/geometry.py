"""Geometric claims from Sec. 1.2 of the paper.

Two results are checked here:

1. The taut-hand contact angle around two circular cross-sections of radii
   r <= R (paper Fig. 5):

       theta = pi + 2*arcsin((R - r) / (R + r))

   giving fractional coverages F = theta/(2*pi) on the larger shaft and
   f = 1 - F on the smaller. We verify this with an independent numerical
   construction of the common external tangent lines.

2. The leg-length mismatch penalties (paper Sec. 1.2, Figs. 3-4):

       tip-to-tip:      sqrt((l + L)^2 - delta^2) / (l + L)
       shaft-to-shaft:  sqrt(max(l, L)^2 - delta^2) / max(l, L)

   The shaft-to-shaft configuration always pays at least as large a penalty,
   and becomes infeasible (penalty 0) at a smaller vertical displacement.
"""

import numpy as np
from scipy.optimize import brentq


def contact_angle_formula(r, R):
    """Paper's closed form for the angle subtended by the hand on the larger
    shaft: theta = pi + 2*arcsin((R - r)/(R + r))."""
    r, R = np.asarray(r, dtype=float), np.asarray(R, dtype=float)
    return np.pi + 2.0 * np.arcsin((R - r) / (R + r))


def contact_angle_numeric(r, R):
    """Independent verification of the contact angle.

    Model the cross-section exactly as the paper does: two externally tangent
    circles (large one of radius R centered at the origin, small one of radius
    r centered at (R + r, 0)) with the hand as a taut band around both. The
    band consists of the two common external tangent segments plus the two
    outer arcs. We locate the upper tangent line by root-finding, with no use
    of the paper's closed form.

    A candidate tangent point on the large circle at polar angle phi has
    tangent line with unit normal (cos phi, sin phi) at distance R from the
    origin. The band is tangent to the small circle when the distance from the
    small circle's center to that line also equals r.

    Returns the angle of the large circle's covered arc.
    """
    d = R + r  # center-to-center distance for externally tangent circles

    def signed_gap(phi):
        # distance from small-circle center to the candidate tangent line,
        # minus the small radius; zero iff the line is tangent to both circles
        return (d * np.cos(phi) - R) + r

    # the upper tangent point on the large circle lies in (0, pi/2] when
    # R > r and at exactly pi/2 when R == r
    phi = brentq(signed_gap, 1e-12, np.pi / 2 + 1e-12, xtol=1e-14)

    # sanity: the tangent line must not cut either circle
    p = np.array([R * np.cos(phi), R * np.sin(phi)])
    n = p / np.linalg.norm(p)
    assert abs(np.dot(p, n) - R) < 1e-9
    assert abs(abs(np.dot(np.array([d, 0.0]), n) - R) - r) < 1e-9

    # by symmetry the lower tangent point is at -phi; the hand covers the
    # large circle's arc on the far side, from phi through pi to 2*pi - phi
    return 2.0 * np.pi - 2.0 * phi


def coverage_fractions(r, R):
    """Fractional hand coverage (f, F) of the smaller and larger shafts.

    Paper Eqs. (1)-(2): F = theta/(2*pi), f = 1 - F.
    """
    theta = contact_angle_formula(r, R)
    F = theta / (2.0 * np.pi)
    return 1.0 - F, F


def small_shaft_arc_numeric(r, R):
    """Covered arc on the *smaller* shaft, from the same tangent construction.

    Used to verify f + F = 1 independently: the band touches the small circle
    at the same normal direction as the tangent line, so the covered arc on
    the small circle (the near side) spans 2*phi.
    """
    theta_large = contact_angle_numeric(r, R)
    phi = (2.0 * np.pi - theta_large) / 2.0
    return 2.0 * phi


def tip_to_tip_penalty(ell, L, delta):
    """Gratification penalty for vertically displaced shafts, tip-to-tip.

    sqrt((l + L)^2 - delta^2) / (l + L); zero when the combined shafts cannot
    bridge the height difference.
    """
    ell, L, delta = np.broadcast_arrays(
        np.asarray(ell, float), np.asarray(L, float), np.asarray(delta, float)
    )
    span = ell + L
    return np.sqrt(np.clip(span**2 - delta**2, 0.0, None)) / span


def shaft_to_shaft_penalty(ell, L, delta):
    """Gratification penalty for vertically displaced shafts, shaft-to-shaft.

    sqrt(max(l, L)^2 - delta^2) / max(l, L); zero when even the longer shaft
    cannot bridge the height difference.
    """
    ell, L, delta = np.broadcast_arrays(
        np.asarray(ell, float), np.asarray(L, float), np.asarray(delta, float)
    )
    m = np.maximum(ell, L)
    return np.sqrt(np.clip(m**2 - delta**2, 0.0, None)) / m
