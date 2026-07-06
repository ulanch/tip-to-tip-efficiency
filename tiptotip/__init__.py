"""Numerical verification of "Optimal Tip-to-Tip Efficiency" (Chugtai & Gilfoyle, 2014).

This package independently verifies the paper's geometric identities and its
two optimality theorems, and reproduces its figures.
"""

from tiptotip.geometry import (
    contact_angle_formula,
    contact_angle_numeric,
    coverage_fractions,
    tip_to_tip_penalty,
    shaft_to_shaft_penalty,
)
from tiptotip.gratification import (
    S_sqrt,
    age_modifier,
    time_modifier,
    sample_thresholds,
)
from tiptotip.simulate import sample_audience, run_presenter, variance_sweep

__all__ = [
    "contact_angle_formula",
    "contact_angle_numeric",
    "coverage_fractions",
    "tip_to_tip_penalty",
    "shaft_to_shaft_penalty",
    "S_sqrt",
    "age_modifier",
    "time_modifier",
    "sample_thresholds",
    "sample_audience",
    "run_presenter",
    "variance_sweep",
]
