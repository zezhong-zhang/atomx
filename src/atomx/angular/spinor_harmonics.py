"""Real spherical and spinor spherical harmonics for 3-D orbital viz.

These helpers are independent of the physical calculation and are only
used by :mod:`atomx.viz.plotly3d` to paint the angular factor of an
orbital onto a Cartesian sampling grid.

Definitions
-----------
Non-relativistic (Schrödinger) orbitals:

    ψ(r, θ, φ) = R_{n l}(r) · Y_{l m}(θ, φ)
               = (P(r) / r) · Y_{l m}(θ, φ)

We plot the **signed real** spherical harmonic
:func:`real_spherical_harmonic` so positive / negative lobes are
distinguishable.

Dirac (relativistic) orbitals have a 2-component spinor angular factor,
the *spinor spherical harmonic* Ω^κ_μ defined via Clebsch–Gordan
coefficients:

    Ω^κ_μ(θ, φ) = Σ_{m_s} C(l, μ−m_s; 1/2, m_s; j, μ) · Y_{l, μ−m_s}
                        · χ_{m_s}

with
    l = kappa        if kappa > 0
    l = −kappa − 1   if kappa < 0,
    j = |kappa| − 1/2,
    μ ∈ {−j, …, +j}.

:func:`spinor_density` returns the non-negative angular density
|Ω^κ_μ(θ, φ)|² summed over the two spin projections.  The
large-component density picks up the (l, j) channel, and the
small-component density picks up the (l', j) channel with
l' = |kappa| if kappa > 0 else kappa (one unit of angular momentum
away), which is exactly what folding P²·|Ω^κ_μ|² + Q²·|Ω^{−κ}_μ|²
does automatically.
"""

from __future__ import annotations

import numpy as np
from scipy import special as _sp

from atomx.angular.clebsch import clebsch_gordan


# scipy ≥ 1.15 deprecates ``sph_harm`` in favour of ``sph_harm_y`` with
# a different argument order.  Centralise the compatibility shim once so
# the rest of the module can call a single helper.
if hasattr(_sp, "sph_harm_y"):
    def _Y(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return _sp.sph_harm_y(l, m, theta, phi)
else:  # pragma: no cover — legacy scipy fallback
    def _Y(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return _sp.sph_harm(m, l, phi, theta)


# ---------------------------------------------------------------------------
# Real spherical harmonics (Schrödinger)
# ---------------------------------------------------------------------------


def real_spherical_harmonic(
    l: int, m: int, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """Real-valued spherical harmonic Y_{l,m}(θ, φ).

    Uses the Condon–Shortley convention and the standard real
    combination:

        Y_{l,m}^real =  √2 · (−1)^m · Re[Y_l^|m|]    if m > 0
        Y_{l,m}^real =  Y_l^0                        if m = 0
        Y_{l,m}^real =  √2 · (−1)^m · Im[Y_l^|m|]    if m < 0

    Parameters
    ----------
    l, m
        Orbital angular-momentum quantum numbers with ``|m| ≤ l``.
    theta, phi
        Polar and azimuthal angles (radians), broadcastable arrays.
    """
    if abs(m) > l:
        raise ValueError(f"Require |m| <= l; got l={l}, m={m}.")

    Y = _Y(l, abs(m), theta, phi)
    if m == 0:
        return Y.real
    if m > 0:
        return np.sqrt(2.0) * ((-1) ** m) * Y.real
    return np.sqrt(2.0) * ((-1) ** m) * Y.imag


# ---------------------------------------------------------------------------
# Dirac spinor spherical harmonics
# ---------------------------------------------------------------------------


def _l_j_from_kappa(kappa: int) -> tuple[int, float]:
    """Return (l, j) for the given Dirac quantum number κ."""
    if kappa == 0:
        raise ValueError("kappa must be non-zero for a Dirac orbital.")
    l = kappa if kappa > 0 else -kappa - 1
    j = abs(kappa) - 0.5
    return l, j


def _valid_mu_values(j: float) -> np.ndarray:
    """Return the allowed magnetic projections  μ ∈ {−j, −j+1, …, +j}."""
    n = int(round(2.0 * j)) + 1
    return np.linspace(-j, +j, n)


def spinor_density(
    kappa: int, mu: float, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """Angular density ``|Ω^κ_μ(θ, φ)|²`` for the Dirac spinor.

    The density is the probability-weighted sum over the two spin
    projections ``m_s = ±1/2`` of the squared magnitude of the
    CG-combined Y_{l, μ−m_s}.  It is strictly non-negative and
    integrates to unity on the sphere.

    Parameters
    ----------
    kappa
        Dirac quantum number (nonzero integer).
    mu
        Magnetic projection; must satisfy ``|μ| ≤ j = |κ| − 1/2`` and
        ``(j − μ)`` must be an integer (i.e. half-integer μ for
        half-integer j).
    theta, phi
        Polar and azimuthal angles, broadcastable arrays.
    """
    l, j = _l_j_from_kappa(kappa)
    if abs(mu) > j + 1e-9:
        raise ValueError(
            f"|mu| must be <= j = {j}; got mu={mu}."
        )
    # CG uses twice-the-angular-momentum integers (see clebsch.py).
    tj, tmu = int(round(2 * j)), int(round(2 * mu))
    # one-half spin
    ts = 1
    # possible spin projections ±1/2 → tm_s ∈ {−1, +1}
    density = np.zeros_like(np.asarray(theta, dtype=float) * np.asarray(phi, dtype=float))
    for tms in (-1, +1):
        tml = tmu - tms
        ml = tml // 2  # CG is zero unless this is an integer, check below
        if tml % 2 != 0 or abs(ml) > l:
            continue
        cg = clebsch_gordan(2 * l, tml, ts, tms, tj, tmu)
        if cg == 0.0:
            continue
        Y = _Y(l, ml, theta, phi)
        density = density + (cg ** 2) * np.abs(Y) ** 2
    return density


__all__ = [
    "real_spherical_harmonic",
    "spinor_density",
    "_l_j_from_kappa",
    "_valid_mu_values",
]
