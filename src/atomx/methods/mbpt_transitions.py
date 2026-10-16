"""Diagonal second-order self-energy for a closed-shell DHF reference.

Both 2p1h and 2h1p diagrams are included, with antisymmetrized Coulomb
matrix elements and all allowed multipoles::

    Sigma_a(w) = 1/2 sum_brs |<ab||rs>|^2 / (w + eb - er - es)
               + 1/2 sum_bcr |<ar||bc>|^2 / (w + er - eb - ec)

Here b,c are occupied spin orbitals and r,s virtual spin orbitals. The
result averages over the magnetic states of a; the default evaluates w = ea.
This is an orbital quasiparticle estimate, not a correlated neutral
excitation or a correction to transition matrix elements. Virtual,
partial-wave and radial cutoffs need convergence checks. See also
https://pyscf.org/user/agf2.html for the second-order self-energy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from atomx.constants import HARTREE_EV

__all__ = [
    "DiagonalSelfEnergy", "DysonRoot", "build_mbpt2_self_energy",
    "mbpt2_orbital_energy_shift", "mbpt2_transition_energy",
]


@dataclass(frozen=True)
class DysonRoot:
    """DF-connected scalar root in an unchanged frozen-orbital reference."""

    energy_Ha: float
    shift_Ha: float
    residue: float
    residual_Ha: float
    bracket_Ha: tuple[float, float]
    reference_identity: str


@dataclass(frozen=True)
class DiagonalSelfEnergy:
    """Immutable real pole expansion Sigma(w)=sum W/(w-pole), W>=0.

    ``reference_identity`` must identify the caller's resolved nucleus,
    Hamiltonian, state, radial metric and retained virtual space. This is
    a snapshot, not a cache transferable to another SCF reference. Energies
    are in Ha and weights in Ha**2. Only exact zero weights are discarded.
    Scalar Dyson solving changes the energy argument, not the spinors or
    core field; it is not a full Brueckner or self-consistent GF2 method.
    """

    reference_energy_Ha: float
    poles_Ha: np.ndarray
    weights_Ha2: np.ndarray
    reference_identity: str
    denominator_floor_Ha: float = 1e-8

    def __post_init__(self):
        poles = np.asarray(self.poles_Ha)
        weights = np.asarray(self.weights_Ha2)
        if (poles.ndim != 1 or weights.shape != poles.shape
                or not np.isrealobj(poles) or not np.isrealobj(weights)
                or not np.all(np.isfinite(poles)) or not np.all(np.isfinite(weights))
                or np.any(weights < 0) or np.ndim(self.reference_energy_Ha) != 0
                or not np.isrealobj(self.reference_energy_Ha) or not np.isfinite(self.reference_energy_Ha)
                or not np.isfinite(self.denominator_floor_Ha) or self.denominator_floor_Ha <= 0
                or not isinstance(self.reference_identity, str) or not self.reference_identity.strip()):
            raise ValueError("invalid real self-energy snapshot or missing reference identity")
        keep = weights != 0
        # Immutable byte buffers prevent callers from re-enabling writes.
        for name, values in (("poles_Ha", poles[keep]), ("weights_Ha2", weights[keep])):
            values = np.frombuffer(np.asarray(values, dtype=float).tobytes(), dtype=float)
            object.__setattr__(self, name, values)

    def _denominators(self, energy_Ha):
        if np.ndim(energy_Ha) != 0 or not np.isrealobj(energy_Ha) or not np.isfinite(energy_Ha):
            raise ValueError("self-energy energy must be a finite real scalar")
        denominators = energy_Ha - self.poles_Ha
        if np.any(abs(denominators) < self.denominator_floor_Ha):
            raise ValueError("resonant second-order self-energy denominator")
        return denominators

    def evaluate(self, energy_Ha: float) -> float:
        """Evaluate the unchanged pole expansion at an explicit energy."""
        return float(np.sum(self.weights_Ha2 / self._denominators(energy_Ha)))

    def derivative(self, energy_Ha: float) -> float:
        """Return the analytic derivative -sum W/(w-pole)**2."""
        return float(-np.sum(self.weights_Ha2 / self._denominators(energy_Ha)**2))

    def solve_dyson(self, bracket_Ha: tuple[float, float]) -> DysonRoot:
        """Solve the DF-connected branch; reject any intervening weighted pole.

        The bracket must contain the DF energy and lie inside its pole-free
        interval. Since 1-Sigma'(w)>0 there is at most one root in that
        interval; these conditions select its connection to the DF state.
        The quasiparticle residue is 1/(1-Sigma'(E)).
        """
        from scipy.optimize import brentq

        lo, hi = bracket_Ha
        if (np.ndim(lo) != 0 or np.ndim(hi) != 0
                or not np.isrealobj(lo) or not np.isrealobj(hi)
                or not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi
                or not lo <= self.reference_energy_Ha <= hi):
            raise ValueError("Dyson bracket must contain the DF reference energy")
        if np.any((self.poles_Ha >= lo - self.denominator_floor_Ha)
                  & (self.poles_Ha <= hi + self.denominator_floor_Ha)):
            raise ValueError("Dyson bracket crosses a nonzero-weight self-energy pole")

        def residual(energy):
            return energy - self.reference_energy_Ha - self.evaluate(energy)

        energy = brentq(residual, lo, hi, xtol=1e-13, rtol=4*np.finfo(float).eps)
        error = residual(energy)
        residue = 1.0 / (1.0 - self.derivative(energy))
        if abs(error) > 1e-10 or not 0 < residue <= 1:
            raise RuntimeError("Dyson root failed residual or residue gate")
        return DysonRoot(float(energy), float(energy-self.reference_energy_Ha),
                         float(residue), float(error), (float(lo), float(hi)),
                         self.reference_identity)


def _capture_attachment_self_energy(
    scf_result,
    orbital_label: str,
    *,
    kappa_max: int = 2,
    energy_window_au: float = 50.0,
    n_virt_max: int | None = None,
    denom_floor: float = 1e-8,
    valence_orbital: tuple[np.ndarray, np.ndarray, int, float] | None = None,
    virtual_kappas: Sequence[int] | None = None,
    term_recorder: Callable[[dict], None] | None = None,
    evaluation_energy_au: float | None = None,
    _matrix_orbitals=None,
    _matrix_recorder=None,
    _matrix_radial_factory=None,
) -> float:
    """Return diagonal Sigma in hartree, including same-subshell partners.

    Each virtual kappa block is solved once in the reference V^N Fock
    field, projected against occupied orbitals in the overlap metric.
    Bound virtuals are retained. ``n_virt_max`` caps roots per kappa;
    ``energy_window_au`` is an upper energy ceiling. A near-zero
    denominator raises rather than silently omitting a resonant term.

    ``valence_orbital=(P,Q,kappa,ea)`` supplies an external, normalized
    orbital orthogonal to the closed core. It does not add that orbital
    to the reference density or exclude it from the virtual sum. This
    computes a monovalent attachment self-energy in the unchanged core
    field. Without it, ``orbital_label`` selects a reference orbital.

    ``virtual_kappas`` explicitly selects the virtual angular blocks.
    A complete l<=2 space is (-1,1,-2,2,-3); ``kappa_max=2`` alone omits
    d5/2. With ``term_recorder``, each evaluated diagram is reported with
    its numerator, denominator and contribution in atomic units.
    ``evaluation_energy_au`` replaces only the denominator energy argument;
    the reference energies, occupied field and all orbitals remain unchanged.
    The default returns Sigma_aa(ea). Records also retain the physical pole
    energy so the same expansion can be evaluated without rebuilding integrals.
    """
    from atomx.methods.mbpt import _pair_v_J, build_virtual_basis
    from atomx.numerics.bspline.convol import ConvolCache

    subshells = scf_result.subshells
    sh_a = next((s for s in subshells if s.label == orbital_label), None)
    if sh_a is None and valence_orbital is None:
        raise ValueError(f"orbital {orbital_label!r} not found")
    if not scf_result.converged:
        raise ValueError("SCFResult must be converged")
    if kappa_max < 1 or (n_virt_max is not None and n_virt_max < 1):
        raise ValueError("partial-wave and virtual-state limits must be positive")
    if not np.isfinite(denom_floor) or denom_floor <= 0 or not np.isfinite(energy_window_au):
        raise ValueError("require positive denominator floor and finite energy ceiling")
    kappas = (
        tuple(k for k in range(-kappa_max, kappa_max + 1) if k)
        if virtual_kappas is None else tuple(virtual_kappas)
    )
    if (not kappas or len(set(kappas)) != len(kappas)
            or any(not isinstance(k, (int, np.integer)) or k == 0 for k in kappas)):
        raise ValueError("virtual_kappas must contain distinct nonzero integers")
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
    external = ([valence_orbital] if _matrix_orbitals is None
                else list(_matrix_orbitals))
    if valence_orbital is not None:
        for target in external:
            P, Q, kap, energy = target
            data = scf_result._bsp_data
            if (not isinstance(kap, (int, np.integer)) or kap == 0
                    or np.shape(P) != (data["ns"],) or np.shape(Q) != (data["ns"],)
                    or not np.isrealobj(P) or not np.isrealobj(Q)
                    or not np.all(np.isfinite(P)) or not np.all(np.isfinite(Q))
                    or np.ndim(energy) != 0 or not np.isrealobj(energy)
                    or not np.isfinite(energy)):
                raise ValueError("invalid external valence orbital")
            from atomx.numerics.radial_workspace import (
                eval_orbital_pair,
                workspace_arrays,
            )

            _, weights = workspace_arrays(data)
            p, q = eval_orbital_pair(P, Q, data)
            if abs(float(np.sum(weights * (p*p + q*q))) - 1.0) > 1e-7:
                raise ValueError("external valence orbital must be normalized in the radial metric")
            for Pb, Qb, kb, _ in occupied:
                if kb == kap:
                    pb, qb = eval_orbital_pair(Pb, Qb, data)
                    if abs(float(np.sum(weights * (p*pb + q*qb)))) > 1e-7:
                        raise ValueError("external valence orbital must be orthogonal to the core")
        P, Q, kap, reference_energy = valence_orbital
        a = (np.asarray(P), np.asarray(Q), int(kap), float(reference_energy))
    elif a is None:
        same = [s for s in subshells if s.kappa == sh_a.kappa]
        P, Q, _ = scf_result.orbitals_raw[sh_a.kappa][same.index(sh_a)]
        a = (P, Q, sh_a.kappa, scf_result.orbital_energies[orbital_label])
    data = scf_result._bsp_data
    energy = a[3] if evaluation_energy_au is None else evaluation_energy_au
    if np.ndim(energy) != 0 or not np.isrealobj(energy) or not np.isfinite(energy):
        raise ValueError("self-energy energy must be a finite real scalar")
    cache = ConvolCache(data)
    virtuals = []
    for kap in kappas:
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

    def numerator(bra1, bra2, ket1, ket2, bilinear, radial_evaluators=None):
        j1, j2, j3, j4 = [2 * abs(s[2]) - 1 for s in (bra1, bra2, ket1, ket2)]
        total = 0.0 if _matrix_orbitals is None else np.zeros((len(external), len(external)))
        for twoJ in range(
            max(abs(j1 - j2), abs(j3 - j4)), min(j1 + j2, j3 + j4) + 1, 2
        ):
            if _matrix_orbitals is None:
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
            else:
                # Preserve signed amplitudes before contraction. Squared diagonal
                # weights cannot determine interference between external orbitals.
                values = np.array([
                    _pair_v_J(*target[:3], *bra2[:3], *ket1[:3], *ket2[:3],
                              twoJ, data, cache, rk_bilinear=bilinear[i],
                              rk_evaluator=None if radial_evaluators is None else radial_evaluators[i])
                    for i, target in enumerate(external)
                ])
                total += (twoJ + 1) * np.outer(values, values)
        return total / (2 * abs(a[2]))

    def contribution(value, denom):
        if value == 0:
            return 0.0
        if abs(denom) < denom_floor:
            raise ValueError("resonant second-order self-energy denominator")
        return value / denom

    shift = 0.0
    for b in occupied:
        bilinear = {} if _matrix_orbitals is None else [{} for _ in external]
        evaluators = (None if _matrix_radial_factory is None
                      else _matrix_radial_factory(external, b, virtuals, data, cache))
        for ir, is_ in combinations_with_replacement(range(len(virtuals)), 2):
            r, s = virtuals[ir], virtuals[is_]
            v2 = numerator(a, b, r, s, bilinear, evaluators) / (1 + (ir == is_))
            pole = r[3] + s[3] - b[3]
            denom = energy - pole
            value = contribution(v2 if _matrix_orbitals is None else v2[0, 0], denom)
            shift += value
            if term_recorder is not None:
                term_recorder(dict(diagram="2p1h", core=((b[2], b[3]),),
                                   virtual_indices=(ir, is_),
                                   virtual=((r[2], r[3]), (s[2], s[3])), numerator=v2,
                                   denominator_Ha=denom, contribution_Ha=value,
                                   pole_energy_Ha=pole))
            if _matrix_recorder is not None:
                _matrix_recorder(pole, v2, "2p1h")
    occupied_pairs = list(combinations_with_replacement(range(len(occupied)), 2))
    for ir, r in enumerate(virtuals):
        bilinear = {} if _matrix_orbitals is None else [{} for _ in external]
        evaluators = (None if _matrix_radial_factory is None
                      else _matrix_radial_factory(external, r, occupied, data, cache))
        for ib, ic in occupied_pairs:
            b, c = occupied[ib], occupied[ic]
            v2 = numerator(a, r, b, c, bilinear, evaluators) / (1 + (ib == ic))
            pole = b[3] + c[3] - r[3]
            denom = energy - pole
            value = contribution(v2 if _matrix_orbitals is None else v2[0, 0], denom)
            shift += value
            if term_recorder is not None:
                term_recorder(dict(diagram="2h1p", core=((b[2], b[3]), (c[2], c[3])),
                                   virtual_indices=(ir,), virtual=((r[2], r[3]),), numerator=v2,
                                   denominator_Ha=denom, contribution_Ha=value,
                                   pole_energy_Ha=pole))
            if _matrix_recorder is not None:
                _matrix_recorder(pole, v2, "2h1p")
    return float(shift)


def mbpt2_orbital_energy_shift(
    scf_result, orbital_label: str, *, kappa_max: int = 2,
    energy_window_au: float = 50.0, n_virt_max: int | None = None,
    denom_floor: float = 1e-8,
    valence_orbital: tuple[np.ndarray, np.ndarray, int, float] | None = None,
    virtual_kappas: Sequence[int] | None = None,
    term_recorder: Callable[[dict], None] | None = None,
    evaluation_energy_au: float | None = None,
) -> float:
    """Diagonal closed-core attachment Sigma, in Ha, from both diagrams.

    External ``(P,Q,kappa,energy)`` spinors must be normalized and core
    orthogonal. Virtuals include bound states, use the complete reference VN
    field and declare angular/energy/root cutoffs explicitly. An off-shell
    energy changes only denominators, not spinors or the core field.
    Near resonant weighted denominators raise. The shared signed pair kernel
    also supplies :func:`atomx.methods.mbpt_matrix.build_mbpt2_matrix_self_energy`.
    """
    return _capture_attachment_self_energy(
        scf_result, orbital_label, kappa_max=kappa_max,
        energy_window_au=energy_window_au, n_virt_max=n_virt_max,
        denom_floor=denom_floor, valence_orbital=valence_orbital,
        virtual_kappas=virtual_kappas, term_recorder=term_recorder,
        evaluation_energy_au=evaluation_energy_au,
    )


def build_mbpt2_self_energy(
    scf_result,
    orbital_label: str,
    *,
    reference_identity: str,
    kappa_max: int = 2,
    energy_window_au: float = 50.0,
    n_virt_max: int | None = None,
    denom_floor: float = 1e-8,
    valence_orbital: tuple[np.ndarray, np.ndarray, int, float] | None = None,
    virtual_kappas: Sequence[int] | None = None,
    term_recorder: Callable[[dict], None] | None = None,
) -> DiagonalSelfEnergy:
    """Capture both native diagrams once for off-shell or scalar Dyson use.

    Reference identity is the caller's explicit snapshot identity; it must
    include the target spinor and declared virtual-space controls. The shared
    native coupling kernel and all its metric/denominator checks are reused.
    """
    from array import array

    if not isinstance(reference_identity, str) or not reference_identity.strip():
        raise ValueError("missing reference identity")
    poles, weights = array("d"), array("d")

    def capture(term):
        if term["numerator"] != 0:
            poles.append(term["pole_energy_Ha"])
            weights.append(term["numerator"])
        if term_recorder is not None:
            term_recorder(term)

    mbpt2_orbital_energy_shift(
        scf_result, orbital_label, kappa_max=kappa_max, energy_window_au=energy_window_au,
        n_virt_max=n_virt_max, denom_floor=denom_floor, valence_orbital=valence_orbital,
        virtual_kappas=virtual_kappas, term_recorder=capture,
    )
    reference_energy = (scf_result.orbital_energies[orbital_label]
                        if valence_orbital is None else valence_orbital[3])
    return DiagonalSelfEnergy(reference_energy, np.asarray(poles), np.asarray(weights),
                              reference_identity, denom_floor)


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
