"""Plotly (interactive HTML) backend for atomx results.

Mirrors the function names of :mod:`atomx.viz.mpl` so users can swap
backends with a one-line change.  All functions return a
``plotly.graph_objects.Figure`` which can be displayed via
``fig.show()`` (notebook) or saved to HTML via ``fig.write_html(...)``.

Import is guarded so that ``import atomx.viz`` succeeds even when
``plotly`` is not installed; calling any function without plotly
raises a helpful error.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

try:
    import plotly.graph_objects as _go
except ImportError as _ie:  # pragma: no cover
    _go = None
    _PLOTLY_IMPORT_ERROR = _ie
else:
    _PLOTLY_IMPORT_ERROR = None


def _require_plotly():
    if _go is None:  # pragma: no cover
        raise RuntimeError(
            "plotly is required for atomx.viz.plotly but is not installed: "
            f"{_PLOTLY_IMPORT_ERROR}\n"
            "Install with:  pip install plotly"
        )


# ---------------------------------------------------------------------------
# Radial orbital
# ---------------------------------------------------------------------------

def plot_radial_orbital(
    wavefunctions,
    label: str,
    *,
    log_x: bool = False,
    show_small_component: bool = True,
    title: Optional[str] = None,
):
    """Interactive plot of P(r), Q(r) for one orbital."""
    _require_plotly()
    if label not in wavefunctions.orbitals:
        available = ", ".join(sorted(wavefunctions.orbitals.keys()))
        raise KeyError(
            f"orbital {label!r} not in wavefunctions. Available: {available}")
    orb = wavefunctions.orbitals[label]
    r = wavefunctions.grid

    fig = _go.Figure()
    fig.add_trace(_go.Scatter(
        x=r, y=orb.P, name=f"P(r) [{label}]",
        mode="lines", line=dict(width=2),
    ))
    if show_small_component and orb.Q is not None:
        fig.add_trace(_go.Scatter(
            x=r, y=orb.Q, name=f"Q(r) [{label}]",
            mode="lines", line=dict(width=1.5, dash="dash"),
        ))
    fig.update_layout(
        title=title or f"Orbital {label}  (ε = {orb.energy:.6f} Ha)",
        xaxis=dict(title="r (Bohr)",
                   type="log" if log_x else "linear"),
        yaxis=dict(title="radial wavefunction", zeroline=True),
        template="plotly_white",
    )
    return fig


def plot_orbital_density(
    wavefunctions,
    labels: Optional[Sequence[str]] = None,
    *,
    log_x: bool = False,
    include_r2: bool = True,
    title: Optional[str] = None,
):
    """Interactive radial density for one or more orbitals."""
    _require_plotly()
    r = wavefunctions.grid
    if labels is None:
        labels = list(wavefunctions.orbitals.keys())

    fig = _go.Figure()
    for lbl in labels:
        if lbl not in wavefunctions.orbitals:
            continue
        orb = wavefunctions.orbitals[lbl]
        rho = orb.P ** 2
        if orb.Q is not None:
            rho = rho + orb.Q ** 2
        y = 4.0 * np.pi * rho if include_r2 else rho
        fig.add_trace(_go.Scatter(x=r, y=y, name=lbl, mode="lines"))

    fig.update_layout(
        title=title or "Radial electron density",
        xaxis=dict(title="r (Bohr)",
                   type="log" if log_x else "linear"),
        yaxis=dict(
            title=("4π (P² + Q²)" if include_r2 else "P² + Q²"),
        ),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

def plot_spectrum(
    spec,
    *,
    show_centroid: bool = False,
    title: Optional[str] = None,
):
    """Interactive spectrum plot (``Spectrum.wavelength_nm`` vs values)."""
    _require_plotly()
    wl = spec.wavelength_nm
    y = spec.values

    fig = _go.Figure()
    fig.add_trace(_go.Scatter(x=wl, y=y, mode="lines", name="spectrum"))
    if show_centroid and y.sum() > 0:
        centroid = float(np.trapezoid(y * wl, wl) / np.trapezoid(y, wl))
        fig.add_vline(x=centroid, line_dash="dot", line_color="black",
                      annotation_text=f"⟨λ⟩ = {centroid:.2f} nm",
                      annotation_position="top right")
    fig.update_layout(
        title=title or "Spectrum",
        xaxis_title="wavelength (nm)",
        yaxis_title=spec.unit,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# QED breakdown
# ---------------------------------------------------------------------------

def plot_qed_breakdown(
    components: dict,
    *,
    reference_key: Optional[str] = None,
    title: Optional[str] = None,
):
    """Bar chart of energy components as an interactive plot."""
    _require_plotly()
    items = [(k, v) for k, v in components.items()
             if isinstance(v, (int, float))]
    if not items:
        raise ValueError("No numeric components to plot.")
    names = [k for k, _ in items]
    values = np.array([float(v) for _, v in items])
    if reference_key is not None:
        if reference_key not in components:
            raise KeyError(
                f"reference_key {reference_key!r} not in components")
        values = values - float(components[reference_key])

    colors = ["steelblue" if v >= 0 else "crimson" for v in values]
    fig = _go.Figure(data=[
        _go.Bar(x=names, y=values, marker_color=colors,
                text=[f"{v:+.3e}" for v in values],
                textposition="outside"),
    ])
    fig.update_layout(
        title=title or "QED / correction breakdown",
        yaxis_title="energy (Hartree)",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# SCF convergence
# ---------------------------------------------------------------------------

def plot_scf_convergence(
    residuals: Sequence[float],
    energies: Optional[Sequence[float]] = None,
    *,
    title: Optional[str] = None,
):
    """Interactive SCF convergence plot with log-y residual."""
    _require_plotly()
    residuals = np.asarray(residuals, dtype=float)
    iters = np.arange(1, len(residuals) + 1)

    fig = _go.Figure()
    fig.add_trace(_go.Scatter(
        x=iters, y=np.maximum(residuals, 1e-16),
        mode="lines+markers", name="residual", yaxis="y",
    ))
    if energies is not None:
        energies = np.asarray(energies, dtype=float)
        fig.add_trace(_go.Scatter(
            x=iters, y=energies,
            mode="lines+markers", name="total energy",
            yaxis="y2", line=dict(color="crimson", dash="dash"),
        ))
        fig.update_layout(yaxis2=dict(
            title="total energy (Hartree)",
            overlaying="y", side="right",
        ))
    fig.update_layout(
        title=title or "SCF convergence",
        xaxis_title="SCF iteration",
        yaxis=dict(title="residual", type="log"),
        template="plotly_white",
    )
    return fig
