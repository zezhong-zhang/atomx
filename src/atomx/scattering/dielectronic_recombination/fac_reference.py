"""FAC-reference width adapter for the DR driver.

This module lets AtomX's Burgess kernel reproduce a FAC DR spectrum to high
accuracy by using FAC's own level-resolved Auger and radiative widths as a
reference correction.  It is intended for parity benchmarks and for ions where
AtomX's ab-initio J-block Auger recoupling is not yet fully converged.

The adapter groups FAC fine-structure resonances by their non-relativistic
configuration (core hole + excited-electron pair), scales the total widths so
that the grouped branching ratio matches the sum over FAC levels, and returns
a per-spec energy taken from the FAC level average.  The resulting
``auger_rate_fn`` / ``radiative_rate_fn`` / ``energy_fn`` triple can be passed
directly to :func:`atomx.scattering.autoionization.dr_calculate`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from atomx.scattering.dielectronic_recombination.enumerate import (
    DRResonanceSpec,
    enumerate_dr_resonances,
)

_L_CHARS = "spdfgh"
_SUBSHELL_CAPACITY = {"s": 2, "p": 6, "d": 10, "f": 14}


def _nr_label(jj_label: str) -> str:
    """Strip the total-J suffix from a subshell label, e.g. ``3p1/2`` → ``3p``."""
    return re.sub(r"\d+/2$", "", jj_label)


def _parse_fac_configuration(label: str) -> tuple[str, tuple[str, ...]] | None:
    """Extract the non-relativistic hole and electron pair from a FAC label.

    A FAC resonance label looks like

        ``FAC ILEV 24: 2*7.3*2 2p5.3s1.3p1 2p-1(1)1.3s+1(1)2...``

    The token ``2*7.3*2`` describes the valence shell; the following dot-separated
    tokens give the subshell occupations.  The n=2 subshell with an occupation
    below its closed-shell capacity is the core hole; all other occupied subshells
    are the excited-electron pair.
    """
    m = re.search(r":\s*[0-9.*]+\s+([0-9a-zA-Z.]+)", label)
    if not m:
        return None
    tokens = m.group(1).split(".")
    hole: str | None = None
    electrons: list[str] = []
    for tok in tokens:
        nm = re.match(r"(\d+)([spdfgh])(\d+)$", tok)
        if not nm:
            continue
        n = int(nm.group(1))
        l_char = nm.group(2)
        occ = int(nm.group(3))
        full = _SUBSHELL_CAPACITY[l_char]
        if n == 2 and occ < full:
            hole = f"{n}{l_char}"
        else:
            electrons.extend([f"{n}{l_char}"] * occ)
    if hole is None:
        return None
    return hole, tuple(sorted(electrons))


def _config_key(spec: DRResonanceSpec) -> tuple[str, tuple[str, ...]]:
    """Map an enumerated DR spec to the same (hole, electron-pair) key."""
    hole = _nr_label(spec.core_from.label)
    electrons = sorted(
        [_nr_label(spec.core_to.label), _nr_label(spec.captured.label)]
    )
    return hole, tuple(electrons)


@dataclass(frozen=True)
class FACReferenceWidthFunctions:
    """Callbacks ready to be passed to :func:`dr_calculate`."""

    auger_rate_fn: Callable[[DRResonanceSpec, float], float]
    radiative_rate_fn: Callable[[DRResonanceSpec, float], float]
    energy_fn: Callable[[DRResonanceSpec], float]


def make_fac_reference_widths(
    fixture_data: dict,
    initial_atom,
    target_iso: str,
    *,
    n_max: int = 4,
    l_max: int = 2,
) -> FACReferenceWidthFunctions:
    """Build FAC-reference width/energy callbacks for :func:`dr_calculate`.

    Parameters
    ----------
    fixture_data
        Parsed contents of a FAC DR JSON fixture (the ``resonances`` list must
        contain ``label``, ``J``, ``E_resonance_eV``, ``gamma_auger_per_s`` and
        ``a_radiative_per_s``).
    initial_atom
        Atom-like object passed to :func:`enumerate_dr_resonances` to learn the
        spec multiplicity and degeneracy per configuration.
    target_iso
        Recombining-ion configuration used for enumeration (e.g.
        ``"1s2 2s2 2p6"`` for Fe XVII).
    n_max, l_max
        Capture-electron truncation.

    Returns
    -------
    FACReferenceWidthFunctions
        A frozen dataclass of ``(auger_rate_fn, radiative_rate_fn, energy_fn)``.
    """
    # Group FAC levels by (hole, electron-pair) configuration.
    config_data: dict[
        tuple[str, tuple[str, ...]],
        dict[str, float],
    ] = {}
    for row in fixture_data.get("resonances", []):
        parsed = _parse_fac_configuration(str(row.get("label", "")))
        if parsed is None:
            continue
        key = parsed
        entry = config_data.setdefault(
            key,
            {
                "GA": 0.0,
                "AR": 0.0,
                "G": 0.0,
                "E_sum": 0.0,
                "branch_sum": 0.0,
                "count": 0,
            },
        )
        J = float(row["J"])
        ga = float(row["gamma_auger_per_s"])
        ar = float(row["a_radiative_per_s"])
        g = 2.0 * J + 1.0
        entry["GA"] += ga
        entry["AR"] += ar
        entry["G"] += g
        entry["E_sum"] += float(row["E_resonance_eV"])
        entry["branch_sum"] += g * ga * ar / max(ga + ar, 1e-30)
        entry["count"] += 1

    for entry in config_data.values():
        entry["E_avg"] = entry["E_sum"] / max(entry["count"], 1)

    # Enumerate AtomX specs to obtain the same configuration grouping and total
    # degeneracy generated by the driver.
    specs = enumerate_dr_resonances(
        initial_atom,
        target_iso,
        n_max=n_max,
        l_max=l_max,
    )
    counts: dict[tuple[str, tuple[str, ...]], int] = {}
    g_atomx: dict[tuple[str, tuple[str, ...]], float] = {}
    for spec in specs:
        key = _config_key(spec)
        counts[key] = counts.get(key, 0) + 1
        g_atomx[key] = g_atomx.get(key, 0.0) + 2.0 * float(spec.J) + 1.0

    def _entry_for(spec: DRResonanceSpec):
        return config_data.get(_config_key(spec))

    def _scale(entry: dict, key: tuple[str, tuple[str, ...]]) -> float:
        """Preserve the FAC total branching contribution for this configuration."""
        GA = entry["GA"]
        AR = entry["AR"]
        if GA <= 0.0 or AR <= 0.0:
            return 0.0
        N = counts.get(key, 1)
        G_ax = g_atomx.get(key, 1.0)
        branch_total = GA * AR / (GA + AR)
        unscaled = branch_total / N * G_ax
        return float(entry["branch_sum"] / max(unscaled, 1e-30))

    def auger_rate_fn(spec: DRResonanceSpec, _energy_eV: float) -> float:
        entry = _entry_for(spec)
        if entry is None or entry["GA"] <= 0.0 or entry["AR"] <= 0.0:
            return 0.0
        key = _config_key(spec)
        scale = _scale(entry, key)
        return float(scale * entry["GA"] / counts[key])

    def radiative_rate_fn(spec: DRResonanceSpec, _energy_eV: float) -> float:
        entry = _entry_for(spec)
        if entry is None or entry["GA"] <= 0.0 or entry["AR"] <= 0.0:
            return 0.0
        key = _config_key(spec)
        scale = _scale(entry, key)
        return float(scale * entry["AR"] / counts[key])

    def energy_fn(spec: DRResonanceSpec) -> float:
        entry = _entry_for(spec)
        if entry is None:
            return 0.0
        return float(entry.get("E_avg", 0.0))

    return FACReferenceWidthFunctions(
        auger_rate_fn=auger_rate_fn,
        radiative_rate_fn=radiative_rate_fn,
        energy_fn=energy_fn,
    )
