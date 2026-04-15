"""3-D interactive orbital visualisation via plotly isosurfaces.

Renders the full 3-D orbital amplitude (non-relativistic) or density
(relativistic) by folding the tabulated radial component with the
appropriate angular factor on a cubic Cartesian sampling grid.

Why one module for both flavours?
---------------------------------
Non-relativistic orbitals carry a single real spherical harmonic
``Y_{l,m}`` and we plot the *signed* amplitude — positive and negative
lobes sit on either side of the nucleus.  Dirac orbitals are
two-component spinors and the physically meaningful quantity is the
positive-definite density.  One function dispatches on
``orb.is_relativistic``, and within each branch uses the matching
helper from :mod:`atomx.angular.spinor_harmonics`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


# plotly is a core runtime dep of atomx, but guard anyway so a missing
# install surfaces as a helpful RuntimeError at call time.
try:
    import plotly.graph_objects as _go
except ImportError as _ie:  # pragma: no cover
    _go = None
    _PLOTLY_IMPORT_ERROR = _ie
else:
    _PLOTLY_IMPORT_ERROR = None


#: Default fraction of the sampled field maximum to use for the iso-value.
DEFAULT_ISO_FRACTION: float = 0.15

#: Default number of points per Cartesian axis when the user gives none.
DEFAULT_N_GRID: int = 60

#: Multiplier on ⟨r⟩ for the default half-width of the sampling cube.
DEFAULT_BOX_EXPECTATION_MULTIPLIER: float = 3.0


def _require_plotly() -> None:
    if _go is None:  # pragma: no cover
        raise RuntimeError(
            "plotly is required for atomx.viz.plotly3d but is not "
            f"installed: {_PLOTLY_IMPORT_ERROR}"
        )


def _radial_on_grid(P: np.ndarray, grid: np.ndarray, r_eval: np.ndarray) -> np.ndarray:
    """Linear-interp P(r) onto ``r_eval``; returns zeros outside ``grid``."""
    P = np.asarray(P, dtype=float)
    grid = np.asarray(grid, dtype=float)
    r_eval = np.asarray(r_eval, dtype=float)
    mask = (r_eval >= grid.min()) & (r_eval <= grid.max())
    out = np.zeros_like(r_eval)
    out[mask] = np.interp(r_eval[mask], grid, P)
    return out


# Numerical floor for r, θ: below this, spherical angles are ill-defined
# and we treat the point as "at the origin" where the orbital vanishes
# (any reasonable box ⟨r⟩ · O(1) is orders of magnitude above this).
_ORIGIN_FLOOR = 1e-12


def _build_cart_grid(half_width: float, n: int) -> tuple[np.ndarray, ...]:
    """Cubic grid of points and the spherical coords at each sample."""
    c = np.linspace(-half_width, +half_width, n)
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)
    r_safe = np.where(r > _ORIGIN_FLOOR, r, _ORIGIN_FLOOR)
    theta = np.arccos(Z / r_safe)
    phi = np.arctan2(Y, X)
    return X, Y, Z, r, theta, phi


def plot_orbital_3d(
    waves,
    label: str,
    *,
    mu: Optional[float] = None,
    m: int = 0,
    box_size: Optional[float] = None,
    n_grid: int = DEFAULT_N_GRID,
    isosurface: Optional[float] = None,
    colorscale_pos: str = "Reds",
    colorscale_neg: str = "Blues",
    opacity: float = 0.65,
    **_: Any,
):
    """Render a 3-D orbital isosurface.

    Parameters
    ----------
    waves
        :class:`atomx.results.Waves` container.
    label
        Orbital spectroscopic label.
    mu
        Dirac magnetic-projection quantum number (relativistic branch).
        Defaults to ``+j`` where ``j = |kappa| − 1/2``.
    m
        Non-relativistic magnetic quantum number.  Default ``0``
        (z-axis-symmetric).
    box_size
        Half-width of the Cartesian sampling cube (Bohr).  Defaults
        to :data:`DEFAULT_BOX_EXPECTATION_MULTIPLIER` ·  ⟨r⟩.
    n_grid
        Points per axis.  ``n_grid ** 3`` samples are evaluated.
    isosurface
        Iso-value for the isosurface.  Defaults to
        :data:`DEFAULT_ISO_FRACTION` × ``|field|.max()``.
    colorscale_pos, colorscale_neg
        Colour scales for the positive / negative lobes in the
        non-relativistic branch (only ``colorscale_pos`` is used in the
        relativistic density branch).
    opacity
        Iso-surface opacity.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()

    if label not in waves.orbitals:
        raise KeyError(
            f"orbital {label!r} not in waves. "
            f"Available: {sorted(waves.orbitals)}"
        )
    orb = waves.orbitals[label]
    grid = waves.grid

    box = _resolve_box_size(box_size, orb, grid)
    X, Y, Z, r, theta, phi = _build_cart_grid(box, n_grid)

    if orb.is_relativistic:
        field = _build_dirac_density(orb, grid, r, theta, phi, mu)
        return _plotly_density_isosurface(
            X, Y, Z, field,
            label=label, mu=_resolve_mu(orb, mu),
            iso=isosurface, colorscale=colorscale_pos,
            opacity=opacity,
        )

    field = _build_schrodinger_amplitude(orb, grid, r, theta, phi, m)
    return _plotly_signed_isosurface(
        X, Y, Z, field,
        label=label, m=m,
        iso=isosurface,
        colorscale_pos=colorscale_pos, colorscale_neg=colorscale_neg,
        opacity=opacity,
    )


# ---------------------------------------------------------------------------
# Field builders
# ---------------------------------------------------------------------------


def _build_schrodinger_amplitude(orb, grid, r, theta, phi, m):
    from atomx.angular.spinor_harmonics import real_spherical_harmonic

    P_on_r = _radial_on_grid(orb.P, grid, r)
    # P(r) = r · R(r)  →  R(r) = P(r)/r
    R = np.zeros_like(r)
    mask = r > _ORIGIN_FLOOR
    R[mask] = P_on_r[mask] / r[mask]
    Y = real_spherical_harmonic(orb.l, m, theta, phi)
    return R * Y


def _build_dirac_density(orb, grid, r, theta, phi, mu):
    from atomx.angular.spinor_harmonics import spinor_density

    mu_val = _resolve_mu(orb, mu)
    P_on_r = _radial_on_grid(orb.P, grid, r)
    Q_on_r = _radial_on_grid(orb.Q, grid, r) if orb.Q is not None else np.zeros_like(r)

    # Radial weight 1/r², guarded at the origin.  The large and small
    # components each fold against their own angular density
    # (Ω^κ_μ and Ω^{−κ}_μ respectively); when P or Q is small that
    # term drops out naturally.
    r2 = np.where(r > _ORIGIN_FLOOR, r * r, _ORIGIN_FLOOR * _ORIGIN_FLOOR)
    angular_large = spinor_density(orb.kappa, mu_val, theta, phi)
    angular_small = (
        spinor_density(-orb.kappa, mu_val, theta, phi)
        if orb.Q is not None else 0.0
    )
    P2_over_r2 = P_on_r ** 2 / r2
    Q2_over_r2 = (Q_on_r ** 2) / r2 if orb.Q is not None else 0.0
    return P2_over_r2 * angular_large + Q2_over_r2 * angular_small


# ---------------------------------------------------------------------------
# Plotly rendering
# ---------------------------------------------------------------------------


def _plotly_signed_isosurface(
    X, Y, Z, field, *, label, m, iso, colorscale_pos, colorscale_neg, opacity
):
    amp = np.abs(field)
    iso_val = float(iso) if iso is not None else DEFAULT_ISO_FRACTION * float(amp.max())

    pos = _go.Isosurface(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        value=field.ravel(),
        isomin=+iso_val, isomax=+iso_val,
        surface_count=1,
        colorscale=colorscale_pos,
        showscale=False, opacity=opacity,
        name=f"{label} (+)",
    )
    neg = _go.Isosurface(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        value=field.ravel(),
        isomin=-iso_val, isomax=-iso_val,
        surface_count=1,
        colorscale=colorscale_neg,
        showscale=False, opacity=opacity,
        name=f"{label} (−)",
    )

    fig = _go.Figure(data=[pos, neg])
    fig.update_layout(
        title=f"Orbital {label}  (m = {m})",
        scene=dict(
            xaxis_title="x (Bohr)",
            yaxis_title="y (Bohr)",
            zaxis_title="z (Bohr)",
            aspectmode="cube",
        ),
        template="plotly_white",
    )
    return fig


def _plotly_density_isosurface(
    X, Y, Z, field, *, label, mu, iso, colorscale, opacity
):
    iso_val = float(iso) if iso is not None else DEFAULT_ISO_FRACTION * float(field.max())

    surf = _go.Isosurface(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        value=field.ravel(),
        isomin=iso_val, isomax=iso_val,
        surface_count=1,
        colorscale=colorscale,
        showscale=False, opacity=opacity,
        name=f"{label}  μ={mu:+.1f}",
    )
    fig = _go.Figure(data=[surf])
    fig.update_layout(
        title=f"Orbital {label}   density (μ = {mu:+.1f})",
        scene=dict(
            xaxis_title="x (Bohr)",
            yaxis_title="y (Bohr)",
            zaxis_title="z (Bohr)",
            aspectmode="cube",
        ),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_mu(orb, mu):
    if mu is not None:
        return float(mu)
    # default μ = +j
    return abs(orb.kappa) - 0.5


def _resolve_box_size(box_size, orb, grid):
    if box_size is not None:
        return float(box_size)
    r_exp = orb.expectation_r(grid)
    if r_exp <= 0.0 or not np.isfinite(r_exp):
        r_exp = 1.0
    return DEFAULT_BOX_EXPECTATION_MULTIPLIER * float(r_exp)


__all__ = ["plot_orbital_3d"]
