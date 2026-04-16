"""3-D interactive orbital visualisation via plotly isosurfaces.

Renders the full 3-D orbital amplitude or density by folding the
tabulated radial component with the appropriate angular factor on a
cubic Cartesian sampling grid.

The default ``mode="all"`` builds a **dropdown selector** that lets
the user switch between:

1. **Total (m-averaged)** — spherically-symmetric radial shell.
2. **Non-relativistic Y_{l,m}** entries for each ``m ∈ {−l, …, +l}``
   (signed amplitude, textbook-style lobes).
3. **Relativistic Ω^κ_μ** entries for each ``μ ∈ {−j, …, +j}``
   (spinor density, physical for Dirac orbitals).

Group 3 appears only when the orbital carries a small component
(``kappa ≠ None``).  For Schrödinger orbitals only groups 1 + 2 are
shown.  Pinning ``m=`` or ``mu=`` directly produces a single surface
with no dropdown (legacy / quick-look mode).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


try:
    import plotly.graph_objects as _go
except ImportError as _ie:  # pragma: no cover
    _go = None
    _PLOTLY_IMPORT_ERROR = _ie
else:
    _PLOTLY_IMPORT_ERROR = None


DEFAULT_ISO_FRACTION: float = 0.15
DEFAULT_N_GRID: int = 60
DEFAULT_BOX_EXPECTATION_MULTIPLIER: float = 3.0
_ORIGIN_FLOOR: float = 1e-12


def _require_plotly() -> None:
    if _go is None:  # pragma: no cover
        raise RuntimeError(
            "plotly is required for atomx.viz.plotly3d but is not "
            f"installed: {_PLOTLY_IMPORT_ERROR}"
        )


# ---------------------------------------------------------------------------
# Cartesian grid + radial interpolation
# ---------------------------------------------------------------------------

def _radial_on_grid(P, grid, r_eval):
    P = np.asarray(P, dtype=float)
    grid = np.asarray(grid, dtype=float)
    r_eval = np.asarray(r_eval, dtype=float)
    mask = (r_eval >= grid.min()) & (r_eval <= grid.max())
    out = np.zeros_like(r_eval)
    out[mask] = np.interp(r_eval[mask], grid, P)
    return out


def _build_cart_grid(half_width: float, n: int):
    c = np.linspace(-half_width, +half_width, n)
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)
    r_safe = np.where(r > _ORIGIN_FLOOR, r, _ORIGIN_FLOOR)
    theta = np.arccos(Z / r_safe)
    phi = np.arctan2(Y, X)
    return X, Y, Z, r, theta, phi


# ---------------------------------------------------------------------------
# Field builders
# ---------------------------------------------------------------------------

def _R_over_r(orb, grid, r):
    """P(r)/r on the Cartesian sampling grid (the radial factor)."""
    P_on_r = _radial_on_grid(orb.P, grid, r)
    R = np.zeros_like(r)
    mask = r > _ORIGIN_FLOOR
    R[mask] = P_on_r[mask] / r[mask]
    return R


def _build_nr_amplitude(orb, grid, r, theta, phi, m: int):
    """Signed amplitude ψ = P(r)/r · Y_{l,m}(θ,φ)."""
    from atomx.angular.spinor_harmonics import real_spherical_harmonic
    return _R_over_r(orb, grid, r) * real_spherical_harmonic(orb.l, m, theta, phi)


def _build_rel_density(orb, grid, r, theta, phi, mu: float):
    """Spinor density (P²+Q²)/r² · |Ω^κ_μ|² + Q² term."""
    from atomx.angular.spinor_harmonics import spinor_density
    r2 = np.where(r > _ORIGIN_FLOOR, r * r, _ORIGIN_FLOOR ** 2)
    P_on_r = _radial_on_grid(orb.P, grid, r)
    Q_on_r = _radial_on_grid(orb.Q, grid, r) if orb.Q is not None else np.zeros_like(r)
    ang_large = spinor_density(orb.kappa, mu, theta, phi)
    ang_small = spinor_density(-orb.kappa, mu, theta, phi) if orb.Q is not None else 0.0
    return P_on_r ** 2 / r2 * ang_large + Q_on_r ** 2 / r2 * ang_small


def _build_total_density(orb, grid, r):
    """m-averaged (spherically symmetric) density  (P²+Q²)/r² · (2j+1)/(4π)
    for the relativistic case, or (2l+1)/(4π) for NR."""
    r2 = np.where(r > _ORIGIN_FLOOR, r * r, _ORIGIN_FLOOR ** 2)
    P_on_r = _radial_on_grid(orb.P, grid, r)
    rho = P_on_r ** 2 / r2
    if orb.Q is not None:
        Q_on_r = _radial_on_grid(orb.Q, grid, r)
        rho = rho + Q_on_r ** 2 / r2
    # The angular average of Σ |Y_lm|² = (2l+1)/(4π), ditto for Ω.
    # We normalise so the shell value matches the per-m scale.
    if orb.kappa is not None:
        j = abs(orb.kappa) - 0.5
        degeneracy = 2 * j + 1
    else:
        degeneracy = 2 * orb.l + 1
    return rho * degeneracy / (4.0 * np.pi)


# ---------------------------------------------------------------------------
# Trace builders (return list of go.Isosurface objects)
# ---------------------------------------------------------------------------

def _signed_traces(X, Y, Z, field, *, name, iso, cs_pos, cs_neg, opacity):
    """Two traces (positive + negative lobes) for a signed amplitude."""
    amp = np.abs(field)
    iso_val = float(iso) if iso is not None else DEFAULT_ISO_FRACTION * float(amp.max())
    if iso_val <= 0:
        iso_val = 1e-6
    flat = dict(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=field.ravel())
    pos = _go.Isosurface(**flat, isomin=+iso_val, isomax=+iso_val,
                          surface_count=1, colorscale=cs_pos,
                          showscale=False, opacity=opacity, name=f"{name} (+)")
    neg = _go.Isosurface(**flat, isomin=-iso_val, isomax=-iso_val,
                          surface_count=1, colorscale=cs_neg,
                          showscale=False, opacity=opacity, name=f"{name} (−)")
    return [pos, neg]


def _density_traces(X, Y, Z, field, *, name, iso, colorscale, opacity):
    """Single trace for a non-negative density isosurface."""
    fmax = float(field.max())
    iso_val = float(iso) if iso is not None else DEFAULT_ISO_FRACTION * fmax
    if iso_val <= 0:
        iso_val = 1e-6
    flat = dict(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=field.ravel())
    return [_go.Isosurface(**flat, isomin=iso_val, isomax=iso_val,
                            surface_count=1, colorscale=colorscale,
                            showscale=False, opacity=opacity, name=name)]


# ---------------------------------------------------------------------------
# Dropdown assembly
# ---------------------------------------------------------------------------

# Each "view" is (label_text, list_of_traces).
_View = Tuple[str, list]


def _nr_views(orb, grid, X, Y, Z, r, theta, phi, *, iso, cs_pos, cs_neg, opacity) -> List[_View]:
    """Build one view per m ∈ {−l, …, +l} under Y_{l,m}."""
    views: List[_View] = []
    for m in range(-orb.l, orb.l + 1):
        field = _build_nr_amplitude(orb, grid, r, theta, phi, m)
        name = f"Y({orb.l},{m:+d})"
        traces = _signed_traces(X, Y, Z, field, name=name, iso=iso,
                                cs_pos=cs_pos, cs_neg=cs_neg, opacity=opacity)
        views.append((name, traces))
    return views


def _rel_views(orb, grid, X, Y, Z, r, theta, phi, *, iso, colorscale, opacity) -> List[_View]:
    """Build one view per μ ∈ {−j, …, +j} under Ω^κ_μ."""
    from atomx.angular.spinor_harmonics import _valid_mu_values
    j = abs(orb.kappa) - 0.5
    views: List[_View] = []
    for mu in _valid_mu_values(j):
        field = _build_rel_density(orb, grid, r, theta, phi, mu)
        name = f"Ω(κ={orb.kappa:+d}, μ={mu:+.1f})"
        traces = _density_traces(X, Y, Z, field, name=name, iso=iso,
                                  colorscale=colorscale, opacity=opacity)
        views.append((name, traces))
    return views


def _total_view(orb, grid, X, Y, Z, r, *, iso, colorscale, opacity) -> _View:
    """Total (m-averaged) radial-shell density."""
    field = _build_total_density(orb, grid, r)
    name = "Total (m-averaged)"
    traces = _density_traces(X, Y, Z, field, name=name, iso=iso,
                              colorscale=colorscale, opacity=opacity)
    return (name, traces)


def _assemble_dropdown(views: List[_View], *, label: str, default_title: str) -> "_go.Figure":
    """Combine pre-rendered views into a single Figure with a dropdown."""
    fig = _go.Figure()
    buttons = []
    n_per = [len(v[1]) for v in views]
    total_traces = sum(n_per)

    # Add all traces (hidden by default).
    for _, traces in views:
        for tr in traces:
            tr.visible = False
            fig.add_trace(tr)

    # Build visibility toggles.
    offset = 0
    for idx, (view_name, _traces) in enumerate(views):
        vis = [False] * total_traces
        for k in range(n_per[idx]):
            vis[offset + k] = True
            if idx == 0:
                fig.data[offset + k].visible = True
        buttons.append(dict(
            label=view_name, method="update",
            args=[{"visible": vis}, {"title": f"{label}  —  {view_name}"}],
        ))
        offset += n_per[idx]

    fig.update_layout(
        title=f"{label}  —  {views[0][0]}" if views else label,
        updatemenus=[dict(
            type="dropdown", direction="down",
            x=0.02, xanchor="left", y=1.12, yanchor="top",
            buttons=buttons,
        )],
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
# Public entry point
# ---------------------------------------------------------------------------

def plot_orbital_3d(
    waves,
    label: str,
    *,
    mode: str = "all",
    m: Optional[int] = None,
    mu: Optional[float] = None,
    box_size: Optional[float] = None,
    n_grid: int = DEFAULT_N_GRID,
    isosurface: Optional[float] = None,
    colorscale_pos: str = "Reds",
    colorscale_neg: str = "Blues",
    colorscale_density: str = "Reds",
    opacity: float = 0.65,
    **_: Any,
):
    """Render a 3-D orbital isosurface with an interactive dropdown.

    Parameters
    ----------
    waves
        :class:`atomx.results.Waves` container.
    label
        Orbital spectroscopic label (already resolved by caller).
    mode
        Which angular decompositions to include in the dropdown:

        ``"all"`` (default)
            Total + non-relativistic Y_{l,m} + relativistic Ω^κ_μ
            (the latter only for Dirac orbitals).
        ``"nr"``
            Total + non-relativistic Y_{l,m} only.
        ``"rel"``
            Total + relativistic Ω^κ_μ only (Dirac orbitals only).
        ``"auto"``
            Legacy single-surface mode: renders one m=0 (NR) or μ=+j
            (Dirac) surface with no dropdown.
    m
        If set, renders a single Y_{l,m} surface — no dropdown.
    mu
        If set, renders a single Ω^κ_μ surface — no dropdown.
    box_size
        Half-width of the Cartesian sampling cube (Bohr).
    n_grid
        Points per axis.
    isosurface
        Iso-value.  Default = 15 % of field maximum.
    colorscale_pos, colorscale_neg
        Signed-amplitude lobe colours (NR branch).
    colorscale_density
        Density iso-surface colour (relativistic branch + Total).
    opacity
        Surface opacity.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _require_plotly()

    if label not in waves.orbitals:
        raise KeyError(
            f"orbital {label!r} not found. Available: {sorted(waves.orbitals)}")
    orb = waves.orbitals[label]
    grid = waves.grid
    box = _resolve_box_size(box_size, orb, grid)
    X, Y, Z, r, theta, phi = _build_cart_grid(box, n_grid)

    iso = isosurface
    kw_nr = dict(iso=iso, cs_pos=colorscale_pos, cs_neg=colorscale_neg, opacity=opacity)
    kw_rl = dict(iso=iso, colorscale=colorscale_density, opacity=opacity)

    # ── Pinned single-surface modes ──────────────────────────────────
    if m is not None:
        field = _build_nr_amplitude(orb, grid, r, theta, phi, m)
        fig = _go.Figure(data=_signed_traces(X, Y, Z, field,
                         name=f"Y({orb.l},{m:+d})", **kw_nr))
        fig.update_layout(title=f"{label}  Y({orb.l},{m:+d})",
                          scene=dict(aspectmode="cube"), template="plotly_white")
        return fig

    if mu is not None:
        field = _build_rel_density(orb, grid, r, theta, phi, float(mu))
        fig = _go.Figure(data=_density_traces(X, Y, Z, field,
                         name=f"Ω(μ={mu:+.1f})", **kw_rl))
        fig.update_layout(title=f"{label}  Ω(μ={mu:+.1f})",
                          scene=dict(aspectmode="cube"), template="plotly_white")
        return fig

    if mode == "auto":
        # Legacy behavior: one surface, no dropdown.
        if orb.is_relativistic:
            mu_val = abs(orb.kappa) - 0.5
            field = _build_rel_density(orb, grid, r, theta, phi, mu_val)
            fig = _go.Figure(data=_density_traces(X, Y, Z, field,
                             name=f"Ω(μ={mu_val:+.1f})", **kw_rl))
        else:
            field = _build_nr_amplitude(orb, grid, r, theta, phi, 0)
            fig = _go.Figure(data=_signed_traces(X, Y, Z, field,
                             name="Y(l,0)", **kw_nr))
        fig.update_layout(title=f"{label}", scene=dict(aspectmode="cube"),
                          template="plotly_white")
        return fig

    # ── Dropdown modes ───────────────────────────────────────────────
    views: List[_View] = []

    # 1. Total (m-averaged)
    views.append(_total_view(orb, grid, X, Y, Z, r, **kw_rl))

    # 2. NR Y_{l,m} entries
    if mode in ("all", "nr"):
        views.extend(_nr_views(orb, grid, X, Y, Z, r, theta, phi, **kw_nr))

    # 3. Relativistic Ω^κ_μ entries (only for Dirac orbitals)
    if mode in ("all", "rel") and orb.is_relativistic:
        views.extend(_rel_views(orb, grid, X, Y, Z, r, theta, phi, **kw_rl))

    return _assemble_dropdown(views, label=label, default_title=views[0][0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_box_size(box_size, orb, grid):
    if box_size is not None:
        return float(box_size)
    r_exp = orb.expectation_r(grid)
    if r_exp <= 0.0 or not np.isfinite(r_exp):
        r_exp = 1.0
    return DEFAULT_BOX_EXPECTATION_MULTIPLIER * float(r_exp)


__all__ = ["plot_orbital_3d"]
