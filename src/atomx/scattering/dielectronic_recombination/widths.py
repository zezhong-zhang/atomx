"""Ab-initio Auger / radiative width factories for the DR driver.

HYP-394 wired the ``auger_rate_fn`` / ``radiative_rate_fn`` hooks on
:func:`atomx.scattering.dielectronic_recombination.solver.dr_calculate`
and :func:`atomx.scattering.dielectronic_recombination.enumerate.build_autoionization_states`,
but left callers to construct the actual callables.  HYP-530 ships the
canonical factory that closes over a solved-atom result and produces an
``auger_rate_fn(spec, E_resonance_eV)`` adapter around
:func:`atomx.scattering.autoionization.auger_rate_csf_average`.

The DR ``DRResonanceSpec`` encodes the resonance as one core excitation
(``core_from`` → ``core_to``) plus a captured spectator electron
(``captured``).  The Auger decay that returns the resonance to the
recombining-ion ground state is the two-electron process

    |core_to, captured>  ->  |core_from; epsilon>

so the channel labels passed into ``auger_rate_csf_average`` are
``hole = core_from``, ``source = core_to``, ``source2 = captured``.
"""

from __future__ import annotations

from typing import Callable

from atomx.scattering.dielectronic_recombination.enumerate import DRResonanceSpec


def make_abinitio_auger_rate_fn(
    result,
    *,
    shooting_n_grid: int | None = None,
    kappa_max: int = 3,
) -> Callable[[DRResonanceSpec, float], float]:
    """Return an ``auger_rate_fn`` adapter around ``auger_rate_csf_average``.

    Parameters
    ----------
    result : AtomicResult
        Solved-atom result carrying the radial orbitals and B-spline
        workspace.  Must include the captured-electron orbitals at the
        expected n (e.g. add ``2p^5 3l`` reference configs to the solve
        so the 3s, 3p, 3d radial functions are well-defined).
    shooting_n_grid : int, optional
        Forwarded to ``auger_rate_csf_average``.  ``None`` (default) takes
        the new post-HYP-530 default (δ(E)-normalised Dirac shooting at
        n_grid = 400), which is the calibration validated against FAC.
    kappa_max : int, default 3
        Continuum partial-wave cutoff.  3 = s/p/d/f waves, matching the
        FAC AITable default for KLL satellite Auger.

    Returns
    -------
    Callable[[DRResonanceSpec, float], float]
        A closure suitable as ``auger_rate_fn=`` on ``dr_calculate`` /
        ``build_autoionization_states``.  Returns the Auger width in
        s⁻¹; falls back to ``0.0`` when the channel labels cannot be
        mapped onto solved-atom subshells (e.g. captured n exceeds the
        solved basis).
    """
    # Lazy import — ``auger_rate_csf_average`` lives in
    # ``autoionization.__init__`` which itself imports from this package's
    # ``solver`` submodule, so a top-level import creates a cycle.
    from atomx.scattering.autoionization import auger_rate_csf_average

    def auger_rate_fn(spec: DRResonanceSpec, energy_eV: float) -> float:
        out = auger_rate_csf_average(
            result,
            hole_label=spec.core_from.label,
            source_label=spec.core_to.label,
            source2_label=spec.captured.label,
            resonance_energy_eV=float(energy_eV),
            shooting_n_grid=shooting_n_grid,
            kappa_max=kappa_max,
        )
        return float(out["Gamma_per_s"])

    return auger_rate_fn
