"""matplotlib rendering of scattering and spectral-synthesis results.

Follows the :mod:`atomx.viz.mpl` conventions: every function returns a
``matplotlib.figure.Figure``, accepts ``ax=`` for embedding, and never
calls ``plt.show()``.

Available renderers
-------------------
:func:`plot_cross_section`
    sigma(E) for one or more series with optional reference overlay
    (FAC / NIST / experiment) and a ratio subpanel.
:func:`plot_collision_strength`
    Omega vs E/threshold, the natural collision-strength axes.
:func:`plot_eii_channels`
    Total EII cross section with the per-subshell breakdown.
:func:`plot_dr_resonances`
    Resonance spectrum: stem plot of strength vs E_res, colored by J,
    with the Auger/radiative width balance.
:func:`plot_dr_rate`
    alpha_DR(T) on log-log axes with reference curves.
:func:`plot_spectrum_channels`
    Synthesized emission decomposed into lines / free-free / RR
    continuum, stacked so line-to-continuum ratios read directly.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

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
            "matplotlib is required for atomx.viz.scattering but is not "
            f"installed: {_MPL_IMPORT_ERROR}"
        )


_SERIES_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple",
                  "tab:brown", "tab:cyan"]


# ---------------------------------------------------------------------------
# Cross sections
# ---------------------------------------------------------------------------

def plot_cross_section(
    energies_eV,
    sigma_cm2,
    *,
    label: str = "atomx",
    reference=None,
    reference_label: str = "reference",
    threshold_eV: Optional[float] = None,
    extra_series: Optional[Mapping[str, tuple]] = None,
    ratio_panel: bool = True,
    loglog: bool = True,
    title: Optional[str] = None,
    ax=None,
):
    """sigma(E) with optional reference overlay and ratio subpanel.

    Parameters
    ----------
    energies_eV, sigma_cm2
        Primary series (e.g. the DW result).
    reference
        Optional ``(E_eV, sigma_cm2)`` tuple — FAC, NIST or experiment.
        Drawn as points; when ``ratio_panel`` is true a lower panel shows
        sigma/reference interpolated onto the reference grid.
    threshold_eV
        Ionization/excitation threshold, drawn as a dashed vertical.
    extra_series
        ``{label: (E_eV, sigma_cm2)}`` additional curves (e.g. BEB, Born).
    """
    _require_mpl()
    E = np.asarray(energies_eV, dtype=float)
    s = np.asarray(sigma_cm2, dtype=float)

    if ax is not None:
        fig = ax.figure
        ax_main, ax_ratio = ax, None
    elif reference is not None and ratio_panel:
        fig, (ax_main, ax_ratio) = _plt.subplots(
            2, 1, figsize=(8, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    else:
        fig, ax_main = _plt.subplots(figsize=(8, 4.8))
        ax_ratio = None

    ax_main.plot(E, s, color=_SERIES_COLORS[0], lw=1.6, label=label)
    if extra_series:
        for i, (lab, (Ex, sx)) in enumerate(extra_series.items()):
            ax_main.plot(np.asarray(Ex, float), np.asarray(sx, float),
                         color=_SERIES_COLORS[(i + 1) % len(_SERIES_COLORS)],
                         lw=1.1, ls="--", label=lab)
    if reference is not None:
        Er = np.asarray(reference[0], dtype=float)
        sr = np.asarray(reference[1], dtype=float)
        ax_main.plot(Er, sr, "o", color="k", ms=5, mfc="none",
                     label=reference_label)
        if ax_ratio is not None:
            si = np.interp(Er, E, s)
            ok = sr > 0
            ax_ratio.plot(Er[ok], si[ok] / sr[ok], "o-", color="k",
                          ms=4, lw=0.8)
            ax_ratio.axhline(1.0, color="0.6", lw=0.8)
            ax_ratio.set_ylabel(f"{label} / ref")
            ax_ratio.set_xlabel("electron energy (eV)")
            band = max(abs(si[ok] / sr[ok] - 1.0).max() * 1.3, 0.12)
            ax_ratio.set_ylim(1.0 - band, 1.0 + band)
    if threshold_eV:
        ax_main.axvline(float(threshold_eV), color="0.4", ls=":", lw=1.0)
        ax_main.text(float(threshold_eV), ax_main.get_ylim()[0], " threshold",
                     rotation=90, va="bottom", fontsize=8, color="0.4")
    if loglog:
        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
    if ax_ratio is None:
        ax_main.set_xlabel("electron energy (eV)")
    ax_main.set_ylabel(r"$\sigma$ (cm$^2$)")
    ax_main.set_title(title or "Cross section")
    ax_main.legend(frameon=False, fontsize=9)
    ax_main.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return fig


def plot_collision_strength(
    energies_eV,
    omega,
    threshold_eV: float,
    *,
    label: str = "atomx",
    reference=None,
    reference_label: str = "FAC",
    title: Optional[str] = None,
    ax=None,
):
    """Collision strength Omega vs E/threshold (the FAC-comparison axes)."""
    _require_mpl()
    E = np.asarray(energies_eV, dtype=float)
    om = np.asarray(omega, dtype=float)
    if ax is None:
        fig, ax = _plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    x = E / float(threshold_eV)
    ax.plot(x, om, color=_SERIES_COLORS[0], lw=1.6, label=label)
    if reference is not None:
        xr = np.asarray(reference[0], float) / float(threshold_eV)
        ax.plot(xr, np.asarray(reference[1], float), "o", color="k",
                mfc="none", ms=5, label=reference_label)
    ax.set_xscale("log")
    ax.set_xlabel(r"$E$ / threshold")
    ax.set_ylabel(r"collision strength $\Omega$")
    ax.set_title(title or "Collision strength")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return fig


def plot_eii_channels(
    eii_result,
    *,
    title: Optional[str] = None,
    ax=None,
):
    """Total EII cross section with the per-subshell channel breakdown.

    Accepts an :class:`~atomx.scattering.electron_ionization.EIIResult`.
    """
    _require_mpl()
    if ax is None:
        fig, ax = _plt.subplots(figsize=(8, 4.8))
    else:
        fig = ax.figure
    E = np.asarray(eii_result.energies_eV, dtype=float)
    ax.plot(E, eii_result.sigma_total_cm2, color="k", lw=1.8, label="total")
    for i, (lab, sig) in enumerate(sorted(
            eii_result.sigma_by_orbital.items())):
        ax.plot(E, np.asarray(sig, float),
                color=_SERIES_COLORS[i % len(_SERIES_COLORS)],
                lw=1.1, ls="--", label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("electron energy (eV)")
    ax.set_ylabel(r"$\sigma$ (cm$^2$)")
    ax.set_title(title or f"EII ({eii_result.method})")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dielectronic recombination
# ---------------------------------------------------------------------------

def plot_dr_resonances(
    resonances: Sequence,
    *,
    weight: str = "strength",
    title: Optional[str] = None,
    ax=None,
):
    """Resonance spectrum: stems at E_res, colored by J.

    ``resonances`` is any sequence of
    :class:`~atomx.scattering.autoionization.auger.AutoionizationState`.
    ``weight``: ``"strength"`` (statistically weighted branching
    g * Gamma_A Gamma_r / Gamma_tot — what DR actually sums),
    ``"auger"`` or ``"radiative"`` for the raw widths.
    """
    _require_mpl()
    if ax is None:
        fig, ax = _plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    Js = sorted({float(r.J) for r in resonances})
    cmap = _plt.get_cmap("viridis", max(len(Js), 2))
    j_color = {j: cmap(i) for i, j in enumerate(Js)}

    for r in resonances:
        ga = float(r.gamma_auger_per_s)
        ar = float(r.a_radiative_per_s)
        tot = ga + ar
        if weight == "strength":
            w = (2 * float(r.J) + 1) * (ga * ar / tot if tot > 0 else 0.0)
        elif weight == "auger":
            w = ga
        elif weight == "radiative":
            w = ar
        else:
            raise ValueError(f"unknown weight {weight!r}")
        if w <= 0:
            continue
        ax.vlines(float(r.E_resonance_eV), 0.0, w,
                  color=j_color[float(r.J)], lw=1.4, alpha=0.85)

    handles = [_plt.Line2D([0], [0], color=j_color[j], lw=2,
                           label=f"J={j:g}") for j in Js]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncols=3)
    ax.set_xlabel("resonance energy (eV)")
    ylabels = {"strength": r"$g_d\,\Gamma_A\Gamma_r/\Gamma$ (s$^{-1}$)",
               "auger": r"$\Gamma_A$ (s$^{-1}$)",
               "radiative": r"$A_r$ (s$^{-1}$)"}
    ax.set_ylabel(ylabels[weight])
    ax.set_yscale("log")
    ax.set_title(title or "DR resonance spectrum")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    return fig


def plot_dr_rate(
    rate,
    *,
    references: Optional[Mapping[str, tuple]] = None,
    title: Optional[str] = None,
    ax=None,
):
    """alpha_DR(T) on log-log axes.

    ``rate`` is a
    :class:`~atomx.scattering.dielectronic_recombination.solver.DRRateCoefficient`;
    ``references`` maps label -> (T_eV, alpha_cm3_s) for overlay (e.g.
    the clean-room FAC reconstruction).
    """
    _require_mpl()
    if ax is None:
        fig, ax = _plt.subplots(figsize=(8, 4.8))
    else:
        fig = ax.figure
    ax.plot(rate.T_eV, rate.alpha_cm3_s, color=_SERIES_COLORS[0],
            lw=1.8, label="atomx")
    if references:
        for i, (lab, (T, a)) in enumerate(references.items()):
            ax.plot(np.asarray(T, float), np.asarray(a, float), "o--",
                    color=_SERIES_COLORS[(i + 1) % len(_SERIES_COLORS)],
                    ms=4, lw=0.9, mfc="none", label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T_e$ (eV)")
    ax.set_ylabel(r"$\alpha_{DR}$ (cm$^3$ s$^{-1}$)")
    ax.set_title(title or "Dielectronic recombination rate")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spectral synthesis channels
# ---------------------------------------------------------------------------

def plot_spectrum_channels(
    total_spectrum,
    *,
    continuum_spectra: Optional[Mapping[str, object]] = None,
    log_y: bool = True,
    title: Optional[str] = None,
    ax=None,
):
    """Synthesized emission decomposed into lines and continuum channels.

    ``total_spectrum`` is the full :class:`~atomx.spectrum.Spectrum`
    (e.g. from ``crm_emission_spectrum(..., include_free_free=True,
    include_rr_continuum=True)``); ``continuum_spectra`` maps channel
    labels to Spectrum objects rendered with (and without) each channel,
    typically::

        {"free-free": spec_ff_only, "RR continuum": spec_rr_only}

    Continuum channels are drawn shaded so line-to-continuum ratios read
    directly off the figure.
    """
    _require_mpl()
    if ax is None:
        fig, ax = _plt.subplots(figsize=(9, 4.8))
    else:
        fig = ax.figure

    wl = total_spectrum.wavelength_nm
    ax.plot(wl, total_spectrum.values, color="k", lw=1.2, label="total")

    if continuum_spectra:
        fills = ["tab:orange", "tab:green", "tab:purple"]
        for i, (lab, spec) in enumerate(continuum_spectra.items()):
            ax.fill_between(spec.wavelength_nm, spec.values,
                            color=fills[i % len(fills)], alpha=0.35,
                            label=lab, lw=0)
    if log_y:
        pos = total_spectrum.values[total_spectrum.values > 0]
        if pos.size:
            ax.set_yscale("log")
            floor = pos.max() * 1e-8
            if continuum_spectra:
                # keep every continuum channel on-scale: the whole point
                # of this figure is the line-to-continuum ratio
                cmaxs = [float(np.max(sp.values))
                         for sp in continuum_spectra.values()
                         if np.max(sp.values) > 0]
                if cmaxs:
                    floor = min(cmaxs) * 1e-2
            ax.set_ylim(bottom=max(floor, pos.max() * 1e-12))
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel(total_spectrum.unit)
    ax.set_title(title or "Synthesized spectrum")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_cross_section",
    "plot_collision_strength",
    "plot_eii_channels",
    "plot_dr_resonances",
    "plot_dr_rate",
    "plot_spectrum_channels",
]
