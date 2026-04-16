"""3-D interactive orbital visualisation via plotly isosurfaces.

Uses **two permanent traces** (positive lobe + negative/density lobe)
and a ``restyle`` dropdown to swap the field data.  This renders
distinct red/blue lobes for signed Y_{l,m} amplitudes and a single
density surface for Ω^κ_μ, and works reliably in both HTML export
and Jupyter notebook renderers.

Dropdown layout::

    Total (m-averaged)
    ── Non-relativistic (Y_lm) ──
    Y(l, -l)  …  Y(l, +l)
    ── Relativistic (Ω^κ_μ) ──      (Dirac orbitals only)
    Ω(κ, μ=-j)  …  Ω(κ, μ=+j)
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


def _require_plotly():
    if _go is None:  # pragma: no cover
        raise RuntimeError(
            "plotly is required for atomx.viz.plotly3d but is not "
            f"installed: {_PLOTLY_IMPORT_ERROR}")


# ── grid + radial helpers ────────────────────────────────────────────

def _radial_on_grid(P, grid, r_eval):
    P, grid, r_eval = (np.asarray(a, dtype=float) for a in (P, grid, r_eval))
    mask = (r_eval >= grid.min()) & (r_eval <= grid.max())
    out = np.zeros_like(r_eval)
    out[mask] = np.interp(r_eval[mask], grid, P)
    return out


def _build_cart_grid(half_width, n):
    c = np.linspace(-half_width, +half_width, n)
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)
    r_safe = np.where(r > _ORIGIN_FLOOR, r, _ORIGIN_FLOOR)
    theta = np.arccos(Z / r_safe)
    phi = np.arctan2(Y, X)
    return X, Y, Z, r, theta, phi


def _R_over_r(orb, grid, r):
    P_on_r = _radial_on_grid(orb.P, grid, r)
    R = np.zeros_like(r)
    mask = r > _ORIGIN_FLOOR
    R[mask] = P_on_r[mask] / r[mask]
    return R


# ── field builders ───────────────────────────────────────────────────

def _build_nr_amplitude(orb, grid, r, theta, phi, m):
    from atomx.angular.spinor_harmonics import real_spherical_harmonic
    return _R_over_r(orb, grid, r) * real_spherical_harmonic(orb.l, m, theta, phi)


def _build_rel_density(orb, grid, r, theta, phi, mu):
    from atomx.angular.spinor_harmonics import spinor_density
    r2 = np.where(r > _ORIGIN_FLOOR, r * r, _ORIGIN_FLOOR ** 2)
    P_on_r = _radial_on_grid(orb.P, grid, r)
    Q_on_r = _radial_on_grid(orb.Q, grid, r) if orb.Q is not None else np.zeros_like(r)
    ang_L = spinor_density(orb.kappa, mu, theta, phi)
    ang_S = spinor_density(-orb.kappa, mu, theta, phi) if orb.Q is not None else 0.0
    return P_on_r ** 2 / r2 * ang_L + Q_on_r ** 2 / r2 * ang_S


def _build_total_density(orb, grid, r):
    r2 = np.where(r > _ORIGIN_FLOOR, r * r, _ORIGIN_FLOOR ** 2)
    P_on_r = _radial_on_grid(orb.P, grid, r)
    rho = P_on_r ** 2 / r2
    if orb.Q is not None:
        rho = rho + _radial_on_grid(orb.Q, grid, r) ** 2 / r2
    deg = (2 * (abs(orb.kappa) - 0.5) + 1) if orb.kappa is not None else (2 * orb.l + 1)
    return rho * deg / (4.0 * np.pi)


# ── view catalogue ───────────────────────────────────────────────────
# (label, field_3d, is_signed)
_View = Tuple[str, np.ndarray, bool]

# Section separator "button" label prefix.
_SEP_NR = "── Y_lm (non-rel) ──"
_SEP_REL = "── Ω^κ_μ (relativistic) ──"


def _collect_views(orb, grid, r, theta, phi, mode) -> List[_View]:
    views: List[_View] = []

    # Total
    views.append(("Total (m-averaged)", _build_total_density(orb, grid, r), False))

    # NR section
    if mode in ("all", "nr"):
        for m in range(-orb.l, orb.l + 1):
            f = _build_nr_amplitude(orb, grid, r, theta, phi, m)
            views.append((f"Y({orb.l},{m:+d})", f, True))

    # Relativistic section
    if mode in ("all", "rel") and orb.is_relativistic:
        from atomx.angular.spinor_harmonics import _valid_mu_values
        j = abs(orb.kappa) - 0.5
        for mu in _valid_mu_values(j):
            f = _build_rel_density(orb, grid, r, theta, phi, mu)
            views.append((f"Ω(κ={orb.kappa:+d}, μ={mu:+.1f})", f, False))

    return views


# ── iso value helper ─────────────────────────────────────────────────

def _iso_for_field(field, user_iso):
    if user_iso is not None:
        return float(user_iso)
    peak = float(np.abs(field).max())
    return max(DEFAULT_ISO_FRACTION * peak, 1e-10)


# ── dropdown figure (2-trace restyle) ────────────────────────────────

def _assemble_dropdown_figure(
    X, Y, Z, views, *, label, user_iso,
    cs_pos, cs_neg, cs_density, opacity,
):
    """Two permanent traces: ``trace[0]`` = positive/density lobe,
    ``trace[1]`` = negative lobe (or transparent when density-mode).
    The dropdown restyles both traces simultaneously."""
    x_flat = X.ravel().tolist()
    y_flat = Y.ravel().tolist()
    z_flat = Z.ravel().tolist()
    n_pts = len(x_flat)

    first_name, first_field, first_signed = views[0]
    iso0 = _iso_for_field(first_field, user_iso)

    # For signed: trace0 = +iso (red), trace1 = -iso (blue).
    # For density: trace0 = +iso (red), trace1 = transparent zero-field.
    zero_field = [0.0] * n_pts

    if first_signed:
        v0, v1 = first_field.ravel().tolist(), first_field.ravel().tolist()
        imin0, imax0 = +iso0, +iso0
        imin1, imax1 = -iso0, -iso0
    else:
        v0, v1 = first_field.ravel().tolist(), zero_field
        imin0, imax0 = iso0, iso0
        imin1, imax1 = 0.0, 0.0

    common = dict(showscale=False, opacity=opacity,
                  caps=dict(x_show=False, y_show=False, z_show=False))

    trace_pos = _go.Isosurface(
        x=x_flat, y=y_flat, z=z_flat, value=v0,
        isomin=imin0, isomax=imax0, surface_count=1,
        colorscale=cs_pos, name="(+)", **common,
    )
    trace_neg = _go.Isosurface(
        x=x_flat, y=y_flat, z=z_flat, value=v1,
        isomin=imin1, isomax=imax1, surface_count=1,
        colorscale=cs_neg, name="(−)", **common,
    )
    fig = _go.Figure(data=[trace_pos, trace_neg])

    # Build dropdown buttons.
    buttons = []
    nr_started = False
    rel_started = False

    for vname, vfield, vsigned in views:
        iso_v = _iso_for_field(vfield, user_iso)
        v_list = vfield.ravel().tolist()

        # Insert section-separator labels.
        if vname.startswith("Y(") and not nr_started:
            nr_started = True
            buttons.append(dict(
                label=_SEP_NR, method="restyle",
                args=[{}],  # no-op: separator label only
            ))
        if vname.startswith("Ω(") and not rel_started:
            rel_started = True
            buttons.append(dict(
                label=_SEP_REL, method="restyle",
                args=[{}],
            ))

        if vsigned:
            buttons.append(dict(
                label=vname, method="restyle",
                args=[{
                    "value": [v_list, v_list],
                    "isomin": [+iso_v, -iso_v],
                    "isomax": [+iso_v, -iso_v],
                    "colorscale": [cs_pos, cs_neg],
                    "opacity": [opacity, opacity],
                }],
            ))
        else:
            buttons.append(dict(
                label=vname, method="restyle",
                args=[{
                    "value": [v_list, zero_field],
                    "isomin": [iso_v, 0.0],
                    "isomax": [iso_v, 0.0],
                    "colorscale": [cs_density, cs_density],
                    "opacity": [opacity, 0.0],
                }],
            ))

    fig.update_layout(
        title=dict(text=f"{label}  —  {first_name}", y=0.95, x=0.5, xanchor="center"),
        margin=dict(t=100, l=10, r=10, b=10),
        updatemenus=[dict(
            type="dropdown", direction="down",
            x=0.0, xanchor="left", y=1.02, yanchor="bottom",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#999", font=dict(size=12),
            buttons=buttons,
        )],
        scene=dict(
            xaxis_title="x (Bohr)", yaxis_title="y (Bohr)", zaxis_title="z (Bohr)",
            aspectmode="cube",
        ),
        template="plotly_white",
    )
    return fig


# ── public entry ─────────────────────────────────────────────────────

def plot_orbital_3d(
    waves, label: str, *,
    mode: str = "all",
    m: Optional[int] = None,
    mu: Optional[float] = None,
    box_size: Optional[float] = None,
    n_grid: int = DEFAULT_N_GRID,
    isosurface: Optional[float] = None,
    colorscale_pos: str = "Reds",
    colorscale_neg: str = "Blues",
    colorscale_density: str = "Reds",
    opacity: float = 0.55,
    **_: Any,
):
    """Render 3-D orbital isosurface(s) with interactive dropdown.

    Parameters
    ----------
    mode : {"all", "nr", "rel", "auto"}
    m : int, optional — pin to one Y_{l,m} (no dropdown).
    mu : float, optional — pin to one Ω^κ_μ (no dropdown).
    """
    _require_plotly()
    if label not in waves.orbitals:
        raise KeyError(f"orbital {label!r} not found. Available: {sorted(waves.orbitals)}")
    orb = waves.orbitals[label]
    grid = waves.grid
    box = _resolve_box_size(box_size, orb, grid)
    X, Y, Z, r, theta, phi = _build_cart_grid(box, n_grid)
    x_flat, y_flat, z_flat = X.ravel().tolist(), Y.ravel().tolist(), Z.ravel().tolist()
    common = dict(showscale=False, opacity=opacity,
                  caps=dict(x_show=False, y_show=False, z_show=False))

    # ── pinned single surface ────────────────────────────────────
    if m is not None:
        field = _build_nr_amplitude(orb, grid, r, theta, phi, m)
        iso = _iso_for_field(field, isosurface)
        fig = _go.Figure(data=[
            _go.Isosurface(x=x_flat, y=y_flat, z=z_flat, value=field.ravel().tolist(),
                           isomin=+iso, isomax=+iso, surface_count=1,
                           colorscale=colorscale_pos, **common),
            _go.Isosurface(x=x_flat, y=y_flat, z=z_flat, value=field.ravel().tolist(),
                           isomin=-iso, isomax=-iso, surface_count=1,
                           colorscale=colorscale_neg, **common),
        ])
        fig.update_layout(title=f"{label}  Y({orb.l},{m:+d})",
                          scene=dict(aspectmode="cube"), template="plotly_white")
        return fig

    if mu is not None:
        field = _build_rel_density(orb, grid, r, theta, phi, float(mu))
        iso = _iso_for_field(field, isosurface)
        fig = _go.Figure(data=[
            _go.Isosurface(x=x_flat, y=y_flat, z=z_flat, value=field.ravel().tolist(),
                           isomin=iso, isomax=iso, surface_count=1,
                           colorscale=colorscale_density, **common),
        ])
        fig.update_layout(title=f"{label}  Ω(μ={mu:+.1f})",
                          scene=dict(aspectmode="cube"), template="plotly_white")
        return fig

    if mode == "auto":
        if orb.is_relativistic:
            return plot_orbital_3d(waves, label, mu=abs(orb.kappa) - 0.5,
                                   n_grid=n_grid, box_size=box_size,
                                   isosurface=isosurface, opacity=opacity,
                                   colorscale_density=colorscale_density)
        return plot_orbital_3d(waves, label, m=0, n_grid=n_grid, box_size=box_size,
                               isosurface=isosurface, opacity=opacity,
                               colorscale_pos=colorscale_pos,
                               colorscale_neg=colorscale_neg)

    # ── full dropdown ────────────────────────────────────────────
    views = _collect_views(orb, grid, r, theta, phi, mode)
    return _assemble_dropdown_figure(
        X, Y, Z, views, label=label, user_iso=isosurface,
        cs_pos=colorscale_pos, cs_neg=colorscale_neg,
        cs_density=colorscale_density, opacity=opacity,
    )


def _resolve_box_size(box_size, orb, grid):
    if box_size is not None:
        return float(box_size)
    r_exp = orb.expectation_r(grid)
    if r_exp <= 0.0 or not np.isfinite(r_exp):
        r_exp = 1.0
    return DEFAULT_BOX_EXPECTATION_MULTIPLIER * float(r_exp)


__all__ = ["plot_orbital_3d"]
