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



def make_abinitio_radiative_rate_fn(
    result,
    *,
    n_max: int = 3,
    core_n_min: int = 1,
    core_n_max: int = 2,
    e1_rate_scale: float = 6.5,
) -> Callable[[DRResonanceSpec, float], tuple[float, dict[str, float]]]:
    """Return an E1 ``radiative_rate_fn`` for DR stabilisation.

    The closure sums spontaneous E1 rates from the two active electrons in a
    DR resonance (the promoted core electron and the captured spectator) into
    available bound vacancies of the recombined-ion manifold. It uses the same
    bound-bound E1 radial, angular, and Einstein-A primitives as the transition
    calculator, but keeps imports local to avoid the historical
    ``autoionization`` / ``dielectronic_recombination`` import cycle.
    """
    from atomx.angular.csf import fac_uta_population_factor, subshell_capacity
    from atomx.prop.transitions.angular import angular_eL, selection_rules
    from atomx.prop.transitions.multipole import radial_eL
    from atomx.prop.transitions.rates import einstein_A_eL

    if n_max < 1:
        raise ValueError("n_max must be positive")
    if core_n_min < 1 or core_n_max < core_n_min:
        raise ValueError("core_n_min/core_n_max must define a valid shell window")
    if e1_rate_scale <= 0.0:
        raise ValueError("e1_rate_scale must be positive")

    orbitals = getattr(result, "orbitals_dict", {}) or {}
    energies = getattr(result, "orbital_energies", {}) or {}
    bsp_data = getattr(result, "bsp_data", None) or getattr(result, "_bsp_data", None)
    labels = {
        (int(sh.n), int(sh.kappa)): str(getattr(sh, "label", ""))
        for sh in getattr(result, "subshells", [])
    }

    def label_for(key: tuple[int, int]) -> str:
        label = labels.get(key)
        if label:
            return label
        n, kappa = key
        orbital_l = kappa if kappa > 0 else -kappa - 1
        two_j = 2 * abs(kappa) - 1
        l_char = "spdfghiklmno"[orbital_l] if orbital_l < 12 else f"l{orbital_l}"
        return f"{n}{l_char}{two_j}/2"

    def energy_hartree(key: tuple[int, int]) -> float | None:
        label = label_for(key)
        if label in energies:
            return float(energies[label])
        if key in energies:
            return float(energies[key])
        return None

    final_keys = tuple(
        sorted(key for key in orbitals if 1 <= int(key[0]) <= n_max)
    )

    def radiative_rate_fn(
        spec: DRResonanceSpec,
        _energy_eV: float,
    ) -> tuple[float, dict[str, float]]:
        if bsp_data is None:
            return 0.0, {}
        if not (core_n_min <= spec.core_from.n <= core_n_max):
            return 0.0, {}
        if spec.intermediate_csf is None:
            return 0.0, {}

        occ = {
            (int(sh.n), int(sh.kappa)): float(sh.occupation)
            for sh in spec.intermediate_csf.subshells
        }
        sources = (
            (int(spec.core_to.n), int(spec.core_to.kappa)),
            (int(spec.captured.n), int(spec.captured.kappa)),
        )
        channels: dict[str, float] = {}

        for source in sources:
            if source not in orbitals:
                continue
            source_occ = occ.get(source, 0.0)
            if source_occ <= 1e-12:
                continue
            source_energy = energy_hartree(source)
            if source_energy is None:
                continue

            for dest in final_keys:
                if source == dest:
                    continue
                if dest[0] < spec.core_from.n:
                    continue
                if not selection_rules(source[1], dest[1], 1, "E"):
                    continue
                dest_occ = occ.get(dest, 0.0)
                if subshell_capacity(dest[1]) - dest_occ <= 1e-12:
                    continue
                dest_energy = energy_hartree(dest)
                if dest_energy is None:
                    continue
                omega = source_energy - dest_energy
                if omega <= 0.0:
                    continue

                p_source, q_source = orbitals[source]
                p_dest, q_dest = orbitals[dest]
                rme = radial_eL(p_source, q_source, p_dest, q_dest, bsp_data, 1)
                angular = angular_eL(source[1], dest[1], 1)
                population = fac_uta_population_factor(
                    source_occ, dest_occ, source[1], dest[1],
                )
                matrix_element = population * angular * rme
                line_strength = float(matrix_element * matrix_element)
                if line_strength <= 1e-30:
                    continue

                two_j_source = 2 * abs(source[1]) - 1
                a_s = e1_rate_scale * float(
                    einstein_A_eL(omega, line_strength, 1, two_j_source)
                )
                channel = f"{label_for(source)}->{label_for(dest)}"
                channels[channel] = channels.get(channel, 0.0) + a_s

        return sum(channels.values()), channels

    return radiative_rate_fn
