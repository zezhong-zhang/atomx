"""Ab-initio width helpers for dielectronic-recombination resonances."""

from __future__ import annotations

from typing import Callable

from atomx.angular.recoupling import lj_from_kappa
from atomx.config.subshell import l_letter
from atomx.properties.transitions.angular import angular_eL, selection_rules
from atomx.properties.transitions.multipole import radial_eL
from atomx.properties.transitions.rates import einstein_A_eL
from atomx.scattering.dielectronic_recombination.enumerate import DRResonanceSpec


def solved_result(atom):
    """Return the solved structure result attached to an Atom-like object."""

    result = getattr(atom, "_result", None)
    if result is not None:
        return result
    result = getattr(atom, "result", None)
    if result is not None:
        return result
    return None


def has_solved_result(atom) -> bool:
    """Whether ``atom`` carries a solved result usable by width factories."""

    return solved_result(atom) is not None


def _label_for_key(n: int, kappa: int) -> str:
    orbital_l, j2 = lj_from_kappa(kappa)
    return f"{n}{l_letter(orbital_l)}{j2}/2"


def _orbital_key_from_label(result, label: str):
    for key in getattr(result, "orbitals_dict", {}):
        if _label_for_key(*key) == label:
            return key
    for sh in getattr(result, "subshells", []):
        key = (int(sh.n), int(sh.kappa))
        if getattr(sh, "label", _label_for_key(*key)) == label:
            return key
    return None


def _orbital_energy_hartree(result, key, label: str) -> float | None:
    energies = getattr(result, "orbital_energies", {}) or {}
    if label in energies:
        return float(energies[label])
    if key in energies:
        return float(energies[key])
    return None


def _result_bsp_data(result):
    return getattr(result, "bsp_data", None) or getattr(result, "_bsp_data", None)


def make_abinitio_auger_rate_fn(initial_atom) -> Callable[[DRResonanceSpec, float], float]:
    """Build a DR ``auger_rate_fn`` backed by the solved structure result.

    The capture (inverse-Auger) process involves one electron pair
    (core_to, captured) and one hole in the initial core shell
    (core_from).  The channel population factor is therefore
    ``1 / g_core_from`` where ``g = 2|κ|`` is the subshell degeneracy,
    rather than the value inferred from a ground-state configuration
    (which has no hole and would give zero).
    """

    result = solved_result(initial_atom)
    if result is None:
        raise ValueError("initial_atom does not carry a solved result")

    def _capture_population_factor(hole_label: str) -> float:
        hole_key = _orbital_key_from_label(result, hole_label)
        if hole_key is None:
            return 0.0
        return 1.0 / max(1, 2 * abs(hole_key[1]))

    def rate(spec: DRResonanceSpec, resonance_energy_eV: float) -> float:
        from atomx.scattering.autoionization import auger_rate_csf_average

        row = auger_rate_csf_average(
            result,
            hole_label=spec.core_from.label,
            source_label=spec.core_to.label,
            source2_label=spec.captured.label,
            resonance_energy_eV=resonance_energy_eV,
            population_factor=_capture_population_factor(spec.core_from.label),
        )
        return float(row.get("Gamma_per_s", 0.0))

    return rate


def make_abinitio_radiative_rate_fn(
    initial_atom,
) -> Callable[[DRResonanceSpec, float], tuple[float, dict[str, float]]]:
    """Build a DR ``radiative_rate_fn`` from E1 orbital matrix elements."""

    result = solved_result(initial_atom)
    if result is None:
        raise ValueError("initial_atom does not carry a solved result")

    orbitals = getattr(result, "orbitals_dict", {}) or {}
    bsp_data = _result_bsp_data(result)

    def channel_rate(source, dest) -> float:
        if not selection_rules(source.kappa, dest.kappa, 1, "E"):
            return 0.0
        if bsp_data is None:
            return 0.0
        source_key = _orbital_key_from_label(result, source.label)
        dest_key = _orbital_key_from_label(result, dest.label)
        if source_key not in orbitals or dest_key not in orbitals:
            return 0.0

        e_source = _orbital_energy_hartree(result, source_key, source.label)
        e_dest = _orbital_energy_hartree(result, dest_key, dest.label)
        if e_source is None or e_dest is None:
            return 0.0
        omega = e_source - e_dest
        if omega <= 0.0:
            return 0.0

        P_source, Q_source = orbitals[source_key]
        P_dest, Q_dest = orbitals[dest_key]
        rme = radial_eL(P_source, Q_source, P_dest, Q_dest, bsp_data, 1,
                        kappa_i=source.kappa, kappa_f=dest.kappa, omega=-omega)
        angular = angular_eL(source.kappa, dest.kappa, 1)
        line_strength = float((angular * rme) ** 2)
        if line_strength <= 1e-30:
            return 0.0
        two_j_source = 2 * abs(source.kappa) - 1
        return float(einstein_A_eL(omega, line_strength, 1, two_j_source))

    def rate(spec: DRResonanceSpec, _resonance_energy_eV: float):
        channels: dict[str, float] = {}
        for source, dest in (
            (spec.captured, spec.core_from),
            (spec.core_to, spec.core_from),
        ):
            label = f"{source.label}->{dest.label}"
            a_s = channel_rate(source, dest)
            if a_s > 0.0:
                channels[label] = a_s
        return sum(channels.values()), channels

    return rate


__all__ = [
    "has_solved_result",
    "make_abinitio_auger_rate_fn",
    "make_abinitio_radiative_rate_fn",
    "solved_result",
]
