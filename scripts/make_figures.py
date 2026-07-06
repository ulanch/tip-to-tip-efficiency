"""Regenerate every figure in figures/. Run: python scripts/make_figures.py"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tiptotip.geometry import (
    contact_angle_formula,
    contact_angle_numeric,
    coverage_fractions,
    tip_to_tip_penalty,
    shaft_to_shaft_penalty,
)
from tiptotip.gratification import S_sqrt, age_modifier, time_modifier, sample_thresholds
from tiptotip.simulate import variance_sweep

FIGDIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3})


def fig_contact_geometry():
    """Paper Fig. 5 + Eqs. (1)-(2): the taut-hand cross-section and the
    resulting coverage fractions, formula vs independent numeric check."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    r, R = 0.55, 1.0
    d = R + r
    theta = contact_angle_formula(r, R)
    phi = (2 * np.pi - theta) / 2

    t = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(R * np.cos(t), R * np.sin(t), color="0.6", lw=1)
    ax1.plot(d + r * np.cos(t), r * np.sin(t), color="0.6", lw=1)

    # covered arcs (the hand)
    t_big = np.linspace(phi, 2 * np.pi - phi, 300)
    ax1.plot(R * np.cos(t_big), R * np.sin(t_big), color="crimson", lw=3,
             label=r"hand on larger shaft ($\theta$)")
    t_small = np.linspace(-phi, phi, 200)
    ax1.plot(d + r * np.cos(t_small), r * np.sin(t_small), color="royalblue",
             lw=3, label=r"hand on smaller shaft ($2\pi-\theta$)")
    # tangent segments closing the band
    for s in (1, -1):
        p_big = np.array([R * np.cos(phi), s * R * np.sin(phi)])
        p_small = np.array([d + r * np.cos(phi), s * r * np.sin(phi)])
        ax1.plot(*zip(p_big, p_small), color="0.2", lw=2)
    ax1.annotate(r"$\theta=\pi+2\arcsin\frac{R-r}{R+r}$", xy=(-1.45, -1.55),
                 fontsize=11)
    ax1.set_aspect("equal")
    ax1.set_title("Fig. 5 geometry: taut hand around two shafts")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlim(-1.7, 2.4)
    ax1.set_ylim(-1.7, 1.7)

    ratios = np.linspace(1, 12, 200)
    fF = np.array([coverage_fractions(1.0, R_) for R_ in ratios])
    ax2.plot(ratios, fF[:, 1], color="crimson", label="F (larger shaft), Eq. (1)")
    ax2.plot(ratios, fF[:, 0], color="royalblue", label="f (smaller shaft), Eq. (2)")
    pick = np.linspace(1, 12, 12)
    num = np.array([contact_angle_numeric(1.0, R_) / (2 * np.pi) for R_ in pick])
    ax2.plot(pick, num, "k.", ms=7, label="independent tangent construction")
    ax2.plot(pick, 1 - num, "k.", ms=7)
    ax2.axhline(0.5, color="0.7", ls="--", lw=1)
    ax2.set_xlabel("girth ratio  R / r")
    ax2.set_ylabel("fractional hand coverage")
    ax2.set_title("Coverage fractions: closed form vs numeric  (f + F = 1)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGDIR / "contact_geometry.png")
    plt.close(fig)


def fig_mismatch_penalty():
    """Sec. 1.2 leg-length mismatch penalties, tip-to-tip vs shaft-to-shaft."""
    ell, L = 4.5, 6.5
    delta = np.linspace(0, ell + L, 400)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(delta, tip_to_tip_penalty(ell, L, delta), color="crimson",
            label=r"tip-to-tip:  $\sqrt{(\ell+L)^2-\Delta^2}\,/\,(\ell+L)$")
    ax.plot(delta, shaft_to_shaft_penalty(ell, L, delta), color="royalblue",
            label=r"shaft-to-shaft:  $\sqrt{\max(\ell,L)^2-\Delta^2}\,/\max(\ell,L)$")
    ax.axvline(L, color="royalblue", ls=":", lw=1)
    ax.axvline(ell + L, color="crimson", ls=":", lw=1)
    ax.annotate("shaft-to-shaft\ninfeasible", xy=(L + 0.15, 0.55), fontsize=8,
                color="royalblue")
    ax.annotate("all jerking\nimpossible", xy=(ell + L - 1.6, 0.75), fontsize=8,
                color="crimson")
    ax.set_xlabel(r"vertical shaft displacement $\Delta$ (in), $\ell=4.5$, $L=6.5$")
    ax.set_ylabel("gratification penalty factor")
    ax.set_title("Shaft-to-shaft always pays the greater mismatch penalty")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "mismatch_penalty.png")
    plt.close(fig)


def fig_theorem():
    """Theorems 1 & 2: E[R_S] over the contact split, against the bound."""
    rng = np.random.default_rng(42)
    n = 400_000
    lam1, lam2 = sample_thresholds(rng, n), sample_thresholds(rng, n)
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for S, name, color in ((S_sqrt, r"$S(f)=\sqrt{f}$ (paper)", "crimson"),
                           (lambda f: np.clip(f, 0, 1) ** 0.25,
                            r"$S(f)=f^{1/4}$", "royalblue")):
        rates = [np.minimum(S(f) / lam1, S(1 - f) / lam2).mean() for f in grid]
        bound = S(0.5) * np.minimum(1 / lam1, 1 / lam2).mean()
        ax.plot(grid, rates, color=color, label=name)
        ax.axhline(bound, color=color, ls="--", lw=1,
                   label=rf"bound $S(1/2)\,E[\min(1/\Lambda_1,1/\Lambda_2)]$")
    ax.axvline(0.5, color="0.6", ls=":", lw=1)
    ax.set_xlabel(r"contact fraction $f_1$ given to member 1  ($f_2 = 1-f_1$)")
    ax.set_ylabel(r"expected stimulation rate  $E[R_S]$")
    ax.set_title("Theorems 1 & 2: equal splits are optimal and meet the bound\n"
                 r"(Monte Carlo, $4\times10^5$ threshold pairs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "theorem_bound.png")
    plt.close(fig)


def fig_figure8():
    """Reproduction of paper Fig. 8: presenters A/B/C across variance."""
    variances = np.linspace(0, 1, 9)
    res = variance_sweep(variances, n=2000, trials=20)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    styles = {"A": ("crimson", "A: sorts by leg-length and shaft-length"),
              "B": ("seagreen", "B: sorts by leg-length only"),
              "C": ("royalblue", "C: does not sort")}
    for p, (color, label) in styles.items():
        ax.plot(variances, res[p], color=color, marker="o", ms=4, label=label)
    ax.set_xlabel("variance scale of girth, shaft length, and leg length")
    ax.set_ylabel("performance (two-handed, vs one-handed single-jerk rate)")
    ax.set_title("Fig. 8 reproduced: sorted presenters stay firm,\n"
                 "unsorted performance is increasingly flaccid")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figure8_reproduction.png")
    plt.close(fig)


def fig_threshold_model():
    """Paper Figs. 6 & 7: the gratification threshold's dependency functions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    age = np.linspace(18, 70, 300)
    ax1.plot(age, age_modifier(age), color="crimson")
    ax1.axvline(45, color="0.6", ls=":", lw=1)
    ax1.annotate("local min at 45", xy=(45.5, 0.35), fontsize=8)
    ax1.set_xlabel("age (years)")
    ax1.set_ylabel("modifier  g(age)")
    ax1.set_title("Fig. 6: age dependency (qualitative)")
    t = np.linspace(0.2, 25, 300)
    ax2.plot(t, time_modifier(t), color="royalblue")
    ax2.axhline(time_modifier(np.inf), color="0.6", ls="--", lw=1)
    ax2.annotate("floor C", xy=(21, 0.38), fontsize=8)
    ax2.set_xlabel("hours since last gratification")
    ax2.set_ylabel(r"modifier  $1/\sqrt{T_0}+C$")
    ax2.set_title("Fig. 7: time-since-gratification dependency")
    fig.tight_layout()
    fig.savefig(FIGDIR / "threshold_model.png")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_contact_geometry, fig_mismatch_penalty, fig_theorem,
               fig_figure8, fig_threshold_model):
        fn()
        print(f"wrote {fn.__name__}")
    print(f"figures in {FIGDIR}")
