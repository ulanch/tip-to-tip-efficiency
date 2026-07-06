"""Reproduction of the paper's Sec. 4 simulation (Fig. 8).

Three presenters perform tip-to-tip double jerking on the same audience:

    A: sorts by leg-length, then by shaft-length within leg-length bins
    B: sorts by leg-length only
    C: does not sort

Girth, shaft length, and leg length are truncated normals centered on the
paper's stated means (2.0 in diameter, 5.5 in shaft, 31 in leg). Performance
is the audience-wide stimulation rate (members gratified per jerk) normalized
by the single-jerk baseline on the same audience.

The paper does not specify its variance units, threshold parameters, or how
infeasible pairings are handled, so absolute performance values differ from
Fig. 8; the claims we verify are ordinal — A >= B >= C, with A robust to
variance and B and C increasingly flaccid.
"""

import numpy as np

from tiptotip.geometry import tip_to_tip_penalty
from tiptotip.gratification import S_sqrt, sample_thresholds

# per-attribute standard deviations (inches) at variance scale 1.0:
# girth diameter, shaft length, leg length
BASE_SIGMA = np.array([0.6, 2.0, 6.0])
MEANS = np.array([2.0, 5.5, 31.0])


def sample_audience(rng, n, variance_scale, sigma_lambda=0.15):
    """Sample an audience of n members.

    Attributes are truncated normals (resampled below a small positive floor)
    with standard deviations variance_scale * BASE_SIGMA. Returns a dict of
    arrays: girth, shaft, leg, lam.
    """
    sig = variance_scale * BASE_SIGMA
    out = {}
    for key, mean, s in zip(("girth", "shaft", "leg"), MEANS, sig):
        x = rng.normal(mean, s, n) if s > 0 else np.full(n, mean)
        floor = 0.2 * mean
        bad = x < floor
        while bad.any():  # truncate by resampling
            x[bad] = rng.normal(mean, s, bad.sum())
            bad = x < floor
        out[key] = x
    out["lam"] = sample_thresholds(rng, n, sigma=sigma_lambda)
    return out


def _pair_jerks_tip_to_tip(shaft1, shaft2, leg1, leg2, lam1, lam2):
    """Jerks required to gratify both members of a tip-to-tip pair.

    Spatial contact is full for both members (f_s = 1, Sec. 1.2.1). Jerk time
    splits across the shafts in proportion to their lengths (Sec. 1.2.2), and
    both members pay the leg-length mismatch penalty of Fig. 3. When the
    combined shafts cannot bridge the height difference, no double jerk is
    possible and the presenter falls back to single-jerking each member.
    """
    delta = abs(leg1 - leg2)
    pen = tip_to_tip_penalty(shaft1, shaft2, delta)
    if pen <= 0.0:
        return lam1 + lam2  # single-jerk fallback: S(1)T(1) = 1 per jerk
    ft1 = shaft1 / (shaft1 + shaft2)
    g1 = S_sqrt(1.0) * S_sqrt(ft1) * pen
    g2 = S_sqrt(1.0) * S_sqrt(1.0 - ft1) * pen
    return max(lam1 / g1, lam2 / g2)


def run_presenter(audience, strategy, rng=None):
    """Total jerks for one presenter to gratify the whole audience.

    strategy: 'A' (sort leg then shaft), 'B' (sort leg), 'C' (no sorting).
    Members are paired adjacently in the resulting order, as in Sec. 3
    (leg-length bins at one-centimeter resolution).
    """
    n = len(audience["lam"])
    leg, shaft = audience["leg"], audience["shaft"]
    if strategy == "A":
        # bin leg length at 1 cm resolution, sort by shaft length within bins
        leg_cm_bins = np.round(leg * 2.54)
        order = np.lexsort((shaft, leg_cm_bins))
    elif strategy == "B":
        order = np.argsort(leg, kind="stable")
    elif strategy == "C":
        order = rng.permutation(n) if rng is not None else np.arange(n)
    else:
        raise ValueError(f"unknown presenter {strategy!r}")

    total = 0.0
    for i in range(0, n - 1, 2):
        a, b = order[i], order[i + 1]
        total += _pair_jerks_tip_to_tip(
            shaft[a], shaft[b], leg[a], leg[b],
            audience["lam"][a], audience["lam"][b],
        )
    if n % 2:  # odd member out gets a single jerk
        total += audience["lam"][order[-1]]
    return total


def variance_sweep(variances, n=2000, trials=20, seed=1337):
    """Performance of presenters A, B, C across audience variance.

    Returns dict mapping 'A'/'B'/'C' to arrays of mean normalized performance,
    one per variance value.

    Normalization: double-jerking presenters work a pair with each hand while
    the baseline single-jerks one member with one hand, so the double-jerk
    rate is doubled before dividing by the baseline. This is the only reading
    consistent with the paper's Fig. 8, which reaches performance ~2: with
    T(f) = sqrt(f), a per-hand comparison is hard-capped at
    2*T(1/2) = sqrt(2) ~ 1.41 (see tests/test_simulation.py).
    """
    rng = np.random.default_rng(seed)
    results = {p: np.zeros(len(variances)) for p in "ABC"}
    for i, v in enumerate(variances):
        perf = {p: [] for p in "ABC"}
        for _ in range(trials):
            aud = sample_audience(rng, n, v)
            baseline = aud["lam"].sum()  # single jerks: 1 gratification/jerk
            for p in "ABC":
                perf[p].append(2.0 * baseline / run_presenter(aud, p, rng=rng))
        for p in "ABC":
            results[p][i] = np.mean(perf[p])
    return results
