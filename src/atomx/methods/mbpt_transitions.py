"""On-shell diagonal second-order self-energy for a closed-shell DHF reference.

Both 2p1h and 2h1p diagrams are included, with antisymmetrized Coulomb
matrix elements and all allowed multipoles::

    Sigma_a(w) = 1/2 sum_brs |<ab||rs>|^2 / (w + eb - er - es)
               + 1/2 sum_bcr |<ar||bc>|^2 / (w + er - eb - ec)

Here b,c are occupied spin orbitals and r,s virtual spin orbitals. The
result averages over the magnetic states of a and evaluates w = ea.
This is an orbital quasiparticle estimate, not a correlated neutral
excitation or a correction to transition matrix elements. Virtual,
partial-wave and radial cutoffs need convergence checks. See also
https://pyscf.org/user/agf2.html for the second-order self-energy.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

from atomx.constants import HARTREE_EV

__all__ = ["mbpt2_orbital_energy_shift", "mbpt2_transition_energy"]


def mbpt2_orbital_energy_shift(
    scf_result,
    orbital_label: str,
    *,
    kappa_max: int = 2,
    energy_window_au: float = 50.0,
    n_virt_max: int | None = None,
    denom_floor: float = 1e-8,
) -> float:
    """Return Sigma_aa(ea) in hartree, including same-subshell partners.

    Each virtual kappa block is solved once in the reference V^N Fock
    field, projected against occupied orbitals in the overlap metric.
    Bound virtuals are retained. ``n_virt_max`` caps roots per kappa;
    ``energy_window_au`` is an upper energy ceiling. A near-zero
    denominator raises rather than silently omitting a resonant term.
    """
    from atomx.methods.mbpt import _pair_v_J, build_virtual_basis
    from atomx.numerics.bspline.convol import ConvolCache

    subshells = scf_result.subshells
    sh_a = next((s for s in subshells if s.label == orbital_label), None)
    if sh_a is None:
        raise ValueError(f"orbital {orbital_label!r} not found")
    if not scf_result.converged:
        raise ValueError("SCFResult must be converged")
    if kappa_max < 1 or (n_virt_max is not None and n_virt_max < 1):
        raise ValueError("partial-wave and virtual-state limits must be positive")
    if denom_floor <= 0 or not np.isfinite(energy_window_au):
        raise ValueError("require positive denominator floor and finite energy ceiling")
    occupied = []
    a = None
    for sh in subshells:
        if sh.occupation <= 0.01:
            continue
        if abs(sh.occupation - 2 * abs(sh.kappa)) > 0.01:
            raise ValueError("orbital self-energy requires a closed-shell reference")
        same = [s for s in subshells if s.kappa == sh.kappa]
        P, Q, _ = scf_result.orbitals_raw[sh.kappa][same.index(sh)]
        state = (P, Q, sh.kappa, scf_result.orbital_energies[sh.label])
        occupied.append(state)
        if sh is sh_a:
            a = state
    if a is None:
        same = [s for s in subshells if s.kappa == sh_a.kappa]
        P, Q, _ = scf_result.orbitals_raw[sh_a.kappa][same.index(sh_a)]
        a = (P, Q, sh_a.kappa, scf_result.orbital_energies[orbital_label])
    data = scf_result._bsp_data
    cache = ConvolCache(data)
    virtuals = []
    for kap in range(-kappa_max, kappa_max + 1):
        if kap == 0:
            continue
        _, states = build_virtual_basis(
            scf_result,
            kap,
            cache=cache,
            e_max=energy_window_au,
            potential_mode="VN",
            orthogonal_to_occupied=True,
        )
        # Also enforce the ceiling for alternative basis providers.
        states = [s for s in states if s[2] < energy_window_au]
        if n_virt_max is not None:
            states = states[:n_virt_max]
        virtuals.extend((P, Q, kap, e) for P, Q, e in states)

    def numerator(bra1, bra2, ket1, ket2, bilinear):
        j1, j2, j3, j4 = [2 * abs(s[2]) - 1 for s in (bra1, bra2, ket1, ket2)]
        total = 0.0
        for twoJ in range(
            max(abs(j1 - j2), abs(j3 - j4)), min(j1 + j2, j3 + j4) + 1, 2
        ):
            v = _pair_v_J(
                *bra1[:3],
                *bra2[:3],
                *ket1[:3],
                *ket2[:3],
                twoJ,
                data,
                cache,
                rk_bilinear=bilinear,
            )
            total += (twoJ + 1) * v * v
        return total / (2 * abs(a[2]))

    def contribution(value, denom):
        if value == 0:
            return 0.0
        if abs(denom) < denom_floor:
            raise ValueError("resonant second-order self-energy denominator")
        return value / denom

    shift = 0.0
    virtual_pairs = list(combinations_with_replacement(range(len(virtuals)), 2))
    for b in occupied:
        bilinear = {}
        for ir, is_ in virtual_pairs:
            r, s = virtuals[ir], virtuals[is_]
            v2 = numerator(a, b, r, s, bilinear) / (1 + (ir == is_))
            shift += contribution(v2, a[3] + b[3] - r[3] - s[3])
    occupied_pairs = list(combinations_with_replacement(range(len(occupied)), 2))
    for r in virtuals:
        bilinear = {}
        for ib, ic in occupied_pairs:
            b, c = occupied[ib], occupied[ic]
            v2 = numerator(a, r, b, c, bilinear) / (1 + (ib == ic))
            shift += contribution(v2, a[3] + r[3] - b[3] - c[3])
    return float(shift)


def mbpt2_transition_energy(
    scf_result,
    alpha_label: str,
    beta_label: str,
    *,
    kappa_max: int = 2,
    energy_window_au: float = 50.0,
    n_virt_max: int | None = None,
) -> dict:
    """Difference of on-shell orbital energies, with shifts and eV conversion."""
    omega = (
        scf_result.orbital_energies[beta_label]
        - scf_result.orbital_energies[alpha_label]
    )
    options = dict(
        kappa_max=kappa_max, energy_window_au=energy_window_au, n_virt_max=n_virt_max
    )
    da = mbpt2_orbital_energy_shift(scf_result, alpha_label, **options)
    db = (
        da
        if alpha_label == beta_label
        else mbpt2_orbital_energy_shift(scf_result, beta_label, **options)
    )
    corrected = omega + (db - da)
    return {
        "omega_DHF_au": float(omega),
        "delta_E_alpha_au": da,
        "delta_E_beta_au": db,
        "omega_corrected_au": float(corrected),
        "omega_corrected_eV": float(corrected * HARTREE_EV),
    }
