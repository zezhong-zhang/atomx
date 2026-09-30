"""Resonance enumeration for dielectronic recombination drivers.

This module supplies the high-level plumbing that FAC exposes through
``SetDR``: start from a recombining ion plus the recombined ion ground
configuration, enumerate core-excited/captured-electron resonances, attach
Auger and radiative widths, and feed them to the Burgess kernel.

The default widths are intentionally semi-empirical fallbacks.  They keep the
driver useful with today's public atomx APIs, while the call sites accept
``auger_rate_fn`` and ``radiative_rate_fn`` hooks for full matrix-element
implementations as the MCDHF/J-block machinery lands.  Enumerated resonances
also carry explicit J-resolved CSF placeholders so downstream Auger and
radiative code can consume the same intermediate-state representation used by
the structure solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from atomx.constants import HARTREE_EV, RYDBERG_EV
from atomx.config.subshell import Subshell
from atomx.config.csf import CSF
from atomx.properties.transitions.angular import selection_rules
from atomx.scattering.autoionization.auger import AutoionizationState

_L_SYMBOLS = "spdfghiklmno"


@dataclass(frozen=True)
class SubshellSpec:
    """Compact relativistic subshell descriptor used during enumeration."""

    n: int
    l: int
    kappa: int
    j: float
    occupation: float = 0.0

    @property
    def label(self) -> str:
        tj = int(round(2 * self.j))
        return f"{self.n}{_L_SYMBOLS[self.l]}{tj}/2"

    @property
    def parity(self) -> int:
        return -1 if self.l % 2 else 1

    @property
    def capacity(self) -> int:
        return int(round(2 * self.j + 1))


@dataclass(frozen=True)
class DRResonanceSpec:
    """Bookkeeping for one enumerated doubly-excited resonance."""

    core_from: SubshellSpec
    core_to: SubshellSpec
    captured: SubshellSpec
    J: float
    parity: int
    intermediate_csf: CSF | None = None

    @property
    def label(self) -> str:
        p = "e" if self.parity > 0 else "o"
        return (
            f"{self.core_from.label}^-1 "
            f"{self.core_to.label} {self.captured.label}; "
            f"J={self.J:g}{p}"
        )

    @property
    def satellite_group(self) -> str:
        return (
            f"n={self.core_from.n}->{self.core_to.n}, "
            f"capture n={self.captured.n}"
        )


def _kappa_l_j(l: int) -> list[tuple[int, float]]:
    if l == 0:
        return [(-1, 0.5)]
    return [(l, l - 0.5), (-(l + 1), l + 0.5)]


def _config_string(obj) -> str:
    if isinstance(obj, str):
        return obj
    configs = getattr(obj, "_config_strings", None)
    if configs:
        return str(configs[0])
    config = getattr(obj, "config", None)
    if config:
        return str(config)
    raise TypeError(
        "target_iso must be a configuration string or an Atom-like object "
        "with _config_strings"
    )


def _parse_nr_config(config: str) -> list[tuple[int, int, int]]:
    shells = []
    for token in config.replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        if len(token) < 3:
            raise ValueError(f"cannot parse configuration token {token!r}")
        n = int(token[0])
        l_char = token[1].lower()
        if l_char not in _L_SYMBOLS:
            raise ValueError(f"unknown orbital letter {l_char!r} in {token!r}")
        l = _L_SYMBOLS.index(l_char)
        occ = int(token[2:])
        shells.append((n, l, occ))
    return shells


def _jj_subshells_from_config(config: str) -> list[SubshellSpec]:
    specs: list[SubshellSpec] = []
    for n, l, occ in _parse_nr_config(config):
        if l == 0:
            specs.append(SubshellSpec(n, l, -1, 0.5, float(occ)))
            continue
        caps = [(l, l - 0.5, 2 * l), (-(l + 1), l + 0.5, 2 * l + 2)]
        remaining = occ
        for kappa, j, cap in caps:
            q = min(max(remaining, 0), cap)
            specs.append(SubshellSpec(n, l, kappa, j, float(q)))
            remaining -= q
    return specs


def _bound_subshells(n_max: int, l_max: int) -> list[SubshellSpec]:
    out: list[SubshellSpec] = []
    for n in range(1, n_max + 1):
        for l in range(0, min(l_max, n - 1) + 1):
            for kappa, j in _kappa_l_j(l):
                out.append(SubshellSpec(n, l, kappa, j, 0.0))
    return out


def _csf_from_resonance(
    target: Sequence[SubshellSpec],
    core_from: SubshellSpec,
    core_to: SubshellSpec,
    captured: SubshellSpec,
    two_j: int,
    parity: int,
) -> CSF:
    """Build an explicit-J CSF placeholder for one DR intermediate."""

    occ: dict[tuple[int, int], float] = {
        (s.n, s.kappa): float(s.occupation)
        for s in target
        if s.occupation > 1e-10
    }
    occ[(core_from.n, core_from.kappa)] = occ.get((core_from.n, core_from.kappa), 0.0) - 1.0
    occ[(core_to.n, core_to.kappa)] = occ.get((core_to.n, core_to.kappa), 0.0) + 1.0
    occ[(captured.n, captured.kappa)] = occ.get((captured.n, captured.kappa), 0.0) + 1.0

    subshells = [
        Subshell(n=n, kappa=kappa, occupation=q)
        for (n, kappa), q in sorted(occ.items(), key=lambda item: (item[0][0], abs(item[0][1]), -item[0][1]))
        if q > 1e-10
    ]
    tmp = DRResonanceSpec(core_from, core_to, captured, 0.5 * two_j, parity)
    return CSF(
        subshells=subshells,
        total_J2=int(two_j),
        parity=int(parity),
        label=tmp.label,
    )


def _orbital_energy_eV(atom, spec: SubshellSpec, *, z_eff: float) -> float:
    """Return a one-electron orbital energy in eV.

    Solved atomx objects provide energies for occupied orbitals.  Unoccupied
    virtuals fall back to a hydrogenic screened estimate, which is adequate
    for ranking resonance energies and keeps enumeration independent of a
    dedicated virtual-orbital solver.
    """

    energies = getattr(atom, "orbital_energies", {}) if atom is not None else {}
    if spec.label in energies:
        return float(energies[spec.label]) * HARTREE_EV
    short = f"{spec.n}{_L_SYMBOLS[spec.l]}"
    matches = [float(v) * HARTREE_EV for k, v in energies.items() if k.startswith(short)]
    if matches:
        return float(np.mean(matches))
    return -RYDBERG_EV * z_eff * z_eff / (spec.n * spec.n)


def _initial_statistical_weight(atom, default: float = 1.0) -> float:
    # Prefer the core configuration string if available; an active-space
    # MCDHF result used for DR widths can have fractional open-shell
    # occupations that do not reflect the initial ion ground-state term.
    core_config = _config_string(atom)
    if core_config:
        try:
            closed = True
            for _n, l, occ in _parse_nr_config(core_config):
                cap = 2 * (2 * l + 1)
                if occ != cap:
                    closed = False
                    break
            if closed:
                return 1.0
        except Exception:
            pass

    result = getattr(atom, "result", None)
    subshells = getattr(result, "subshells", None)
    if not subshells:
        return default
    open_shells = []
    for sh in subshells:
        j2 = getattr(sh, "j2", 2 * abs(getattr(sh, "kappa", -1)) - 1)
        cap = j2 + 1
        occ = float(getattr(sh, "occupation", cap))
        if 1e-8 < occ < cap - 1e-8:
            open_shells.append(j2)
    if not open_shells:
        return 1.0
    return float(max(open_shells) + 1)


def enumerate_dr_resonances(
    initial_atom,
    target_iso,
    *,
    n_max: int = 10,
    l_max: int | None = None,
) -> list[DRResonanceSpec]:
    """Enumerate core-excited plus captured-electron DR resonance specs.

    The generated set covers one active core excitation from the recombined
    ion ground configuration and one captured spectator electron up to
    ``n_max``.  Electric-dipole stabilising parity is enforced between the
    captured electron and the core hole, and total-J values are generated by
    jj vector coupling of the promoted and captured electrons.
    """

    if n_max < 2:
        raise ValueError("n_max must be at least 2")
    if l_max is None:
        l_max = min(n_max - 1, 8)
    if l_max < 0:
        raise ValueError("l_max must be non-negative")

    target = _jj_subshells_from_config(_config_string(target_iso))
    occupied = [s for s in target if s.occupation > 1e-10]
    occ_by_key = {(s.n, s.l, s.kappa): s.occupation for s in target}
    virtuals = _bound_subshells(n_max, l_max)

    specs: list[DRResonanceSpec] = []
    seen: set[tuple[int, int, int, int, int, int, int, int]] = set()
    for core_from in occupied:
        for core_to in virtuals:
            if core_to.n < core_from.n:
                continue
            if (
                core_to.n == core_from.n
                and core_to.l == core_from.l
                and core_to.kappa == core_from.kappa
            ):
                continue
            occ = occ_by_key.get((core_to.n, core_to.l, core_to.kappa), 0.0)
            if occ >= core_to.capacity - 1e-10:
                continue
            for captured in virtuals:
                if captured.n < 2:
                    continue
                captured_occ = occ_by_key.get((captured.n, captured.l, captured.kappa), 0.0)
                if (
                    captured_occ >= captured.capacity - 1e-10
                    and (captured.n, captured.kappa) != (core_from.n, core_from.kappa)
                ):
                    continue
                if (
                    (captured.n, captured.kappa) == (core_to.n, core_to.kappa)
                    and captured_occ >= captured.capacity - 1.0 - 1e-10
                ):
                    continue
                # E1 stabilisation of the captured electron into the hole.
                if not selection_rules(captured.kappa, core_from.kappa, 1, "E"):
                    continue
                parity = core_from.parity * core_to.parity * captured.parity
                j_min = abs(core_to.j - captured.j)
                j_max = core_to.j + captured.j
                for two_j in range(int(round(2 * j_min)), int(round(2 * j_max)) + 1, 2):
                    J = 0.5 * two_j
                    key = (
                        core_from.n, core_from.kappa,
                        core_to.n, core_to.kappa,
                        captured.n, captured.kappa,
                        two_j, parity,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    csf = _csf_from_resonance(
                        target, core_from, core_to, captured, two_j, parity,
                    )
                    specs.append(DRResonanceSpec(
                        core_from, core_to, captured, J, parity,
                        intermediate_csf=csf,
                    ))
    return specs


def _default_auger_width(spec: DRResonanceSpec, E_resonance_eV: float, z_eff: float) -> float:
    if E_resonance_eV <= 0:
        return 0.0
    delta_l = abs(spec.core_to.l - spec.core_from.l)
    angular = 1.0 / (1.0 + delta_l)
    rydberg = spec.captured.n ** -3
    z_scale = max(z_eff, 1.0) ** 2
    return 2.5e14 * z_scale * rydberg * angular / (2.0 * spec.J + 1.0)


def _default_radiative_width(
    spec: DRResonanceSpec,
    core_excitation_eV: float,
    capture_binding_eV: float,
    z_eff: float,
) -> tuple[float, dict[str, float]]:
    channels: dict[str, float] = {}
    total = 0.0

    if selection_rules(spec.captured.kappa, spec.core_from.kappa, 1, "E"):
        omega = max(capture_binding_eV, 1e-6) / RYDBERG_EV
        a_cap = 6.25e8 * (max(z_eff, 1.0) ** 4) * omega ** 3 / (spec.captured.n ** 3)
        channels[f"{spec.captured.label}->{spec.core_from.label}"] = a_cap
        total += a_cap

    if selection_rules(spec.core_to.kappa, spec.core_from.kappa, 1, "E"):
        omega = max(core_excitation_eV, 1e-6) / RYDBERG_EV
        a_core = 6.25e8 * (max(z_eff, 1.0) ** 2) * omega ** 3
        channels[f"{spec.core_to.label}->{spec.core_from.label}"] = a_core
        total += a_core

    return total, channels


def build_autoionization_states(
    specs: Sequence[DRResonanceSpec],
    initial_atom,
    *,
    min_resonance_eV: float = 1e-9,
    auger_rate_fn: Callable[[DRResonanceSpec, float], float] | None = None,
    radiative_rate_fn: Callable[[DRResonanceSpec, float], float | tuple[float, dict[str, float]]] | None = None,
    energy_fn: Callable[[DRResonanceSpec], float] | None = None,
) -> list[AutoionizationState]:
    """Attach energies and widths to enumerated DR resonance specs.

    An optional ``energy_fn`` overrides the resonance energy taken from the
    atom's one-electron orbital energies.  This is used by reference-width
    adapters (e.g. FAC) that supply their own level energies while still
    letting the driver enumerate resonances.
    """

    charge = float(getattr(initial_atom, "charge", 0))
    z_eff = max(charge + 1.0, 1.0)
    states: list[AutoionizationState] = []
    for spec in specs:
        e_from = _orbital_energy_eV(initial_atom, spec.core_from, z_eff=z_eff)
        e_to = _orbital_energy_eV(initial_atom, spec.core_to, z_eff=z_eff)
        core_excitation = max(e_to - e_from, 0.0)
        capture_binding = abs(_orbital_energy_eV(initial_atom, spec.captured, z_eff=z_eff))
        if energy_fn is None:
            e_res = core_excitation - capture_binding
        else:
            e_res = float(energy_fn(spec))
        if e_res <= min_resonance_eV:
            continue

        if auger_rate_fn is None:
            gamma_a = _default_auger_width(spec, e_res, z_eff)
        else:
            gamma_a = float(auger_rate_fn(spec, e_res))

        if radiative_rate_fn is None:
            a_r, channels = _default_radiative_width(
                spec, core_excitation, capture_binding, z_eff,
            )
        else:
            rad = radiative_rate_fn(spec, e_res)
            if isinstance(rad, tuple):
                a_r = float(rad[0])
                channels = dict(rad[1])
            else:
                a_r = float(rad)
                channels = {}

        if gamma_a <= 0.0 and a_r <= 0.0:
            continue
        states.append(
            AutoionizationState(
                label=spec.label,
                E_resonance_eV=float(e_res),
                J=float(spec.J),
                gamma_auger_per_s=float(gamma_a),
                a_radiative_per_s=float(a_r),
                channels=channels,
            )
        )
    return states


def resonance_strength_rows(
    resonances: Sequence[AutoionizationState],
    T_eV,
    *,
    initial_state_g: float = 1.0,
) -> list[dict]:
    """Return one diagnostic row per resonance contribution."""

    T = np.atleast_1d(np.asarray(T_eV, dtype=float))
    if np.any(T <= 0):
        raise ValueError("Temperatures must be positive (eV).")

    rows = []
    # Single source of truth for the prefactor: the same constant
    # ``dr_rate_coefficient`` uses.  This used to hard-code the
    # Rydberg-convention value 6.60305e-24, so per-resonance rows here
    # were 50.17x below the solver's own rate for the same resonance.
    from atomx.scattering.dielectronic_recombination.solver import (
        _DR_PREFACTOR_CM3_S_EV_32,
    )

    prefactor = _DR_PREFACTOR_CM3_S_EV_32 * T ** (-1.5)
    for d in resonances:
        width = d.total_width_per_s
        branch = 0.0 if width <= 0.0 else d.gamma_auger_per_s * d.a_radiative_per_s / width
        omega = d.g / (2.0 * initial_state_g)
        alpha = prefactor * omega * branch * np.exp(-d.E_resonance_eV / T)
        rows.append({
            "label": d.label,
            "E_resonance_eV": d.E_resonance_eV,
            "J": d.J,
            "g": d.g,
            "gamma_auger_per_s": d.gamma_auger_per_s,
            "a_radiative_per_s": d.a_radiative_per_s,
            "total_width_per_s": width,
            "fluorescence_yield": d.fluorescence_yield,
            "alpha_cm3_s": alpha,
            "satellite_group": _satellite_group_from_label(d.label),
            "channels": dict(d.channels),
        })
    return rows


def _satellite_group_from_label(label: str) -> str:
    """Best-effort group label from the generated resonance label."""

    try:
        hole, promoted = label.split("^-1", 1)
        promoted = promoted.strip().split()[0]
        return f"{hole[0]}->{promoted[0]}"
    except (IndexError, ValueError):
        return ""
