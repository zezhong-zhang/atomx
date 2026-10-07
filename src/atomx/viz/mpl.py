"""matplotlib rendering of atomx results.

All functions accept one or more atomx result objects, return a
``matplotlib.figure.Figure``, and support a ``ax=`` keyword so they
can be embedded in existing grids.  No function calls ``plt.show()``
— display is the caller's responsibility.

Available renderers
-------------------
:func:`plot_radial_orbital`
    P(r), Q(r) for one Dirac orbital on log or linear axes.
:func:`plot_orbital_density`
    4π r² · [P² + Q²]   radial electron density for selected orbitals.
:func:`plot_grotrian`
    Energy-level diagram with optional transition arrows.
:func:`plot_spectrum`
    A :class:`~atomx.spectrum.Spectrum` as a line plot, with optional
    resonance / centroid overlay.
:func:`plot_qed_breakdown`
    Bar chart of the QED energy ladder (one bar per correction).
:func:`plot_scf_convergence`
    Residual / energy curves from an SCF run.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from atomx.constants import HC_EV_NM

# matplotlib is listed as a core-viz dep in pyproject.  Import guarded
# so `import atomx.viz.mpl` works even in headless no-matplotlib envs —
# functions raise a helpful error on call.
try:
    import matplotlib.pyplot as _plt
except ImportError as _ie:  # pragma: no cover
    _plt = None
    _MPL_IMPORT_ERROR = _ie
else:
    _MPL_IMPORT_ERROR = None


def _require_mpl():
    if _plt is None:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for atomx.viz.mpl but is not installed: "
            f"{_MPL_IMPORT_ERROR}"
        )


# ---------------------------------------------------------------------------
# Radial orbital
# ---------------------------------------------------------------------------

def plot_radial_orbital(
    waves,
    label: str,
    *,
    ax=None,
    log_x: bool = True,
    show_small_component: bool = True,
    title: Optional[str] = None,
):
    """Plot P(r) (and Q(r)) for one orbital.

    Parameters
    ----------
    waves : :class:`atomx.results.Waves`
    label : str
        Orbital key, e.g. ``"2p3/2"``.
    ax : :class:`matplotlib.axes.Axes`, optional
        Target axes.  A new figure is created if omitted.
    log_x : bool
        Use log-x axis (useful for short-range structure in heavy atoms).
    show_small_component : bool
        Overlay Q(r) as a dashed line when the orbital is relativistic.
    title : str, optional
        Axes title.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
    """
    _require_mpl()
    if label not in waves.orbitals:
        available = ", ".join(sorted(waves.orbitals.keys()))
        raise KeyError(f"orbital {label!r} not in waves. "
                       f"Available: {available}")
    orb = waves.orbitals[label]
    r = waves.grid

    if ax is None:
        fig, ax = _plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    ax.plot(r, orb.P, label=f"P(r)  [{label}]", linewidth=1.6)
    if show_small_component and orb.Q is not None:
        ax.plot(r, orb.Q, label=f"Q(r)  [{label}]",
                linewidth=1.3, linestyle="--")

    if log_x:
        ax.set_xscale("log")
    ax.axhline(0, color="0.6", linewidth=0.6)
    ax.set_xlabel(r"$r$  (Bohr)")
    ax.set_ylabel("radial wavefunction")
    if title is None:
        eps = orb.energy
        title = (f"Orbital {label}  "
                 r"($\varepsilon$ = " f"{eps:.6f} Ha)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig


def plot_orbital_density(
    waves,
    labels: Optional[Sequence[str]] = None,
    *,
    ax=None,
    log_x: bool = True,
    include_r2: bool = True,
    title: Optional[str] = None,
):
    """Radial electron density ``r² · (P² + Q²)`` for one or more orbitals.

    Parameters
    ----------
    labels : list of str, optional
        Orbitals to plot; default = all in ``waves.orbitals``.
    include_r2 : bool
        If ``True`` (default) plot the radial probability
        ``4π r² (P² + Q²)`` (peak-shaped).  If ``False`` plot the bare
        density ``P² + Q²``.
    """
    _require_mpl()
    r = waves.grid
    if labels is None:
        labels = list(waves.orbitals.keys())

    if ax is None:
        fig, ax = _plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for lbl in labels:
        if lbl not in waves.orbitals:
            continue
        orb = waves.orbitals[lbl]
        rho = orb.P ** 2
        if orb.Q is not None:
            rho = rho + orb.Q ** 2
        if include_r2:
            # In atomx P(r) = r R(r), so P² already carries one factor of r²;
            # additional r² appears only in the full 3-D volume measure.
            weight = 4.0 * np.pi
            ax.plot(r, weight * rho, label=lbl, linewidth=1.6)
        else:
            ax.plot(r, rho, label=lbl, linewidth=1.6)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(r"$r$  (Bohr)")
    ax.set_ylabel(
        r"$4\pi\,(P^2 + Q^2)$" if include_r2 else r"$P^2 + Q^2$"
    )
    ax.set_title(title or "Radial electron density")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Grotrian diagram
# ---------------------------------------------------------------------------

def plot_grotrian(
    levels: Sequence,
    transitions: Optional[Sequence] = None,
    *,
    ax=None,
    energy_unit: str = "eV",
    title: Optional[str] = None,
    arrow_label: str = "wavelength",
):
    """Grotrian (energy-level) diagram with optional transition arrows.

    Parameters
    ----------
    levels : sequence
        Each entry is an object or dict with ``energy`` (in ``energy_unit``)
        and an optional ``label`` / ``J`` / ``L`` / ``S``.  If LJ/LS are
        provided, levels are horizontally offset by L-grouping (Grotrian
        convention).
    transitions : sequence, optional
        Each entry is an object or dict with ``upper``, ``lower`` (indices
        into ``levels``) and an optional ``wavelength_nm`` / ``A_einstein`` /
        ``multipole``.  Drawn as arrows with optional λ label.
    energy_unit : {"eV", "cm^-1", "Hartree"}
        Axis unit for the vertical energy axis.
    arrow_label : {"wavelength", "A_einstein", "multipole", "none"}
        What to annotate each arrow with.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
    """
    _require_mpl()
    if ax is None:
        fig, ax = _plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    # Extract (energy, L-group, label) from each level
    def _E(lv):
        return float(_get(lv, "energy"))

    def _get(lv, key, default=None):
        if hasattr(lv, key):
            return getattr(lv, key)
        if isinstance(lv, dict):
            return lv.get(key, default)
        return default

    # Group by L if available, else by index group
    L_values = []
    for lv in levels:
        L = _get(lv, "L")
        if L is None:
            L = 0
        L_values.append(int(L))
    unique_L = sorted(set(L_values))
    L_to_x = {L: i for i, L in enumerate(unique_L)}

    energies = [_E(lv) for lv in levels]
    xs = [L_to_x[L] for L in L_values]

    # Draw each level as a short horizontal bar
    for i, (E, x, lv) in enumerate(zip(energies, xs, levels, strict=False)):
        ax.plot([x - 0.3, x + 0.3], [E, E], color="navy", linewidth=2.0)
        lbl = _get(lv, "label", f"lvl{i}")
        if lbl:
            ax.text(x + 0.35, E, f" {lbl}", va="center", fontsize=8)

    # Draw transitions
    if transitions:
        for t in transitions:
            u = int(_get(t, "upper"))
            lo = int(_get(t, "lower"))
            if not (0 <= u < len(levels)) or not (0 <= lo < len(levels)):
                continue
            x_u, x_l = xs[u], xs[lo]
            E_u, E_l = energies[u], energies[lo]
            ax.annotate(
                "", xy=(x_l, E_l), xytext=(x_u, E_u),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0),
            )
            if arrow_label != "none":
                val = _get(t, arrow_label, None)
                if val is not None:
                    midx = 0.5 * (x_u + x_l)
                    midy = 0.5 * (E_u + E_l)
                    if arrow_label == "wavelength":
                        txt = f"{float(val):.1f} nm"
                    elif arrow_label == "A_einstein":
                        txt = f"{float(val):.2e} s⁻¹"
                    else:
                        txt = str(val)
                    ax.text(midx, midy, txt, fontsize=7,
                            color="crimson", rotation=0)

    # Axes
    if unique_L:
        l_labels = "SPDFGHIKLMNO"
        ax.set_xticks(list(L_to_x.values()))
        ax.set_xticklabels([l_labels[L] if len(l_labels) > L else f"L={L}"
                            for L in unique_L])
        ax.set_xlim(-0.5, len(unique_L) - 0.5)
    else:
        ax.set_xticks([])

    ax.set_ylabel(f"Energy ({energy_unit})")
    ax.set_title(title or "Grotrian diagram")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

def plot_spectrum(
    spec,
    *,
    ax=None,
    show_centroid: bool = False,
    resonances: Optional[Sequence] = None,
    title: Optional[str] = None,
):
    """Render a :class:`~atomx.spectrum.Spectrum` as a line plot.

    Parameters
    ----------
    show_centroid : bool
        If ``True``, draw a vertical line at the Spectrum's
        area-weighted centroid.
    resonances : sequence, optional
        Iterable of objects/dicts with ``E_res_eV`` and ``gamma_eV`` —
        renders each as a vertical line.
    """
    _require_mpl()
    if ax is None:
        fig, ax = _plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    wl = spec.wavelength_nm
    y = spec.values
    ax.plot(wl, y, color="tab:blue", linewidth=1.2)
    ax.set_xlabel(r"wavelength (nm)")
    ax.set_ylabel(spec.unit)
    ax.set_title(title or "Spectrum")

    if show_centroid and y.sum() > 0:
        centroid = float(np.trapezoid(y * wl, wl) / np.trapezoid(y, wl))
        ax.axvline(centroid, color="k", linewidth=0.6, linestyle=":")
        ax.text(centroid, y.max() * 0.95, f" ⟨λ⟩ = {centroid:.2f} nm",
                fontsize=8)

    if resonances:
        for r in resonances:
            if hasattr(r, "E_res_eV"):
                # Convert eV → nm if any ;-)
                E_eV = float(r.E_res_eV)
                lam_nm = HC_EV_NM / E_eV
                ax.axvline(lam_nm, color="crimson", alpha=0.3, linewidth=0.8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# QED breakdown
# ---------------------------------------------------------------------------

def plot_qed_breakdown(
    components: dict,
    *,
    reference_key: Optional[str] = None,
    ax=None,
    title: Optional[str] = None,
):
    """Bar-chart of QED/correction energy components.

    Parameters
    ----------
    components : dict
        Mapping ``{name: E_Hartree}``, e.g. the ``energy_components``
        attribute of an :class:`SCFResult`.
    reference_key : str, optional
        If given, all bars are plotted as differences relative to this
        component (e.g. ``reference_key="E_nonrel"`` gives corrections
        above a non-relativistic baseline).

    Returns
    -------
    :class:`matplotlib.figure.Figure`
    """
    _require_mpl()
    items = list(components.items())
    # Drop non-numeric entries
    items = [(k, v) for k, v in items if isinstance(v, (int, float))]
    if not items:
        raise ValueError("No numeric components to plot.")

    names = [k for k, _ in items]
    values = np.array([float(v) for _, v in items])

    if reference_key is not None:
        if reference_key not in components:
            raise KeyError(f"reference_key {reference_key!r} not in components")
        values = values - float(components[reference_key])

    if ax is None:
        fig, ax = _plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    colours = ["tab:blue" if v >= 0 else "tab:red" for v in values]
    bars = ax.bar(range(len(names)), values, color=colours)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("energy (Hartree)")
    ax.set_title(title or ("QED / correction breakdown"
                           + (f" (relative to {reference_key})"
                              if reference_key else "")))

    # Annotate each bar
    for bar, v in zip(bars, values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (0.02 * abs(values.max())),
                f"{v:+.3e}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SCF convergence
# ---------------------------------------------------------------------------

def plot_scf_convergence(
    residuals: Sequence[float],
    energies: Optional[Sequence[float]] = None,
    *,
    ax=None,
    title: Optional[str] = None,
):
    """Residual curve (and optionally energy) from an SCF run.

    Parameters
    ----------
    residuals : sequence of float
        Per-iteration residual.
    energies : sequence of float, optional
        Per-iteration total energy (plotted on a twin axis).
    """
    _require_mpl()
    residuals = np.asarray(residuals, dtype=float)
    if ax is None:
        fig, ax = _plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    iters = np.arange(1, len(residuals) + 1)
    ax.semilogy(iters, np.maximum(residuals, 1e-16), "o-",
                color="tab:blue", label="residual")
    ax.set_xlabel("SCF iteration")
    ax.set_ylabel("residual")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    if energies is not None:
        energies = np.asarray(energies, dtype=float)
        ax_t = ax.twinx()
        ax_t.plot(iters, energies, "s--", color="tab:red",
                  label="total energy")
        ax_t.set_ylabel("total energy (Hartree)", color="tab:red")

    ax.set_title(title or "SCF convergence")
    fig.tight_layout()
    return fig
