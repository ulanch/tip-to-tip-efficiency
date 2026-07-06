"""The gratification model of Secs. 1.1 and 2.

Per-jerk gratification is S(f_s) * T(f_t) in [0, 1], where S and T are
monotone increasing, concave, and zero at zero (axioms A1-A3). The paper uses
sqrt(f) for both in its simulations.

The gratification threshold (paper Eq. 3):

    Lambda = Z + g(age) * (1/sqrt(T0) + C)

where Z ~ N(0, sigma^2), g is the age-dependency function of Fig. 6, and T0
is the time in hours since the member's last gratification (Fig. 7). If
Lambda <= 0 the member is instantly gratified.
"""

import numpy as np

DEFAULT_C = 0.3


def S_sqrt(f):
    """The paper's choice of gratification function, S(f) = T(f) = sqrt(f)."""
    return np.sqrt(np.clip(f, 0.0, 1.0))


def age_modifier(age):
    """Age-dependency function g, qualitatively reproducing paper Fig. 6.

    The paper describes g as having a local minimum at age 45 and increasing
    monotonically afterwards, plotted over ages 20-60 with values roughly in
    [0, 2]. We model the adult domain (18+) with a smooth curve that has a
    local maximum near 28 and a local minimum at 45, then rescale to span
    Fig. 6's plotted range. The paper gives no closed form, so this is a
    qualitative stand-in with the stated critical points.
    """
    age = np.asarray(age, dtype=float)
    hump = 1.6 * np.exp(-(((age - 28.0) / 8.0) ** 2))  # local max near 28
    senescence = 1.7 * np.clip((age - 45.0) / 15.0, 0.0, None) ** 2
    return 0.3 + hump + senescence


def time_modifier(t0_hours, C=DEFAULT_C):
    """Dependence on time since last gratification (paper Fig. 7):
    proportional to 1/sqrt(T0) plus a constant floor C."""
    return 1.0 / np.sqrt(np.asarray(t0_hours, dtype=float)) + C


def sample_thresholds(rng, n, sigma=0.15, C=DEFAULT_C):
    """Draw i.i.d. gratification thresholds Lambda per paper Eq. 3.

    Ages are drawn uniformly over the adult range 18-70 and T0 uniformly over
    0.5-24 hours (the paper leaves both distributions unspecified). Members
    with Lambda <= 0 are 'instantly gratified'; their stimulation rate 1/Lambda
    is unbounded, which makes E[min(1/Lambda_1, 1/Lambda_2)] diverge, so we
    floor Lambda at 0.05 to keep expectations finite. The theorems hold for
    any i.i.d. positive thresholds, so the floor does not affect what is
    being verified.
    """
    age = rng.uniform(18.0, 70.0, n)
    t0 = rng.uniform(0.5, 24.0, n)
    z = rng.normal(0.0, sigma, n)
    lam = z + age_modifier(age) * time_modifier(t0, C)
    return np.maximum(lam, 0.05)
