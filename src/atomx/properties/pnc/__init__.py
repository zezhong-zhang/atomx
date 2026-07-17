"""Parity non-conservation (PNC) amplitudes.

Matrix element of the weak-interaction Hamiltonian

    H_W = (G_F / (2√2)) · Q_W · γ_5 · ρ_nuc(r)

between opposite-parity atomic states.  The dominant PNC observable
in heavy alkalis (Cs, Yb, Fr) is the spin-independent ``E_PNC`` E1
amplitude

    E_PNC = Σ_n ⟨f|D|n p⟩⟨n p|H_W|i⟩/(E_i-E_np)
          + Σ_n ⟨f|H_W|n p⟩⟨n p|D|i⟩/(E_f-E_np)

via opposite-parity admixtures with intermediate p states.  That
amplitude dictates the parity-violating optical-rotation signal.

This module implements the **single-particle weak Hamiltonian**
expectation value via the radial integral

    ⟨P_a | H_W | P_b⟩  ∝  ⟨P_a | ρ_nuc(r) · (g_a · g_b ± f_a · f_b) | ⟩

with appropriate sign for s_{1/2} ↔ p_{1/2} parity-flipping pairs.
The result is dressed by core/correlation enhancements through
RPA + MBPT (planned), without which PNC results have ~30 % error.

References
----------
- M. A. Bouchiat, C. C. Bouchiat, *J. Physique* **35**, 899 (1974).
- V. A. Dzuba, V. V. Flambaum, M. G. Kozlov, *Phys. Rev. A* **66**,
  012512 (2002) — all-order PNC amplitudes for Cs.
- J. S. M. Ginges, V. V. Flambaum, *Phys. Rep.* **397**, 63 (2004).

This module provides:

- :func:`weak_hamiltonian_radial_me` — single-particle ⟨a|H_W|b⟩.
- :func:`pnc_amplitude_dirac_only` — reserved for the full sum-over-
  states PNC E1 amplitude; currently raises ``NotImplementedError``.
- :func:`weak_charge_qw` — Q_W from Z, N (default Standard Model).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from atomx.numerics.bspline.convol import ConvolCache
from atomx.numerics.bspline.evaluate import eval_P, eval_Q
from atomx.properties._pseudoscalar import pseudoscalar_angular
from atomx.properties._sos_basis import build_sos_virtual_basis
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.multipole import radial_eL


G_F_AU = 2.222516e-14  # Fermi constant in atomic units
# Q_W(p) ≈ 0.0713; Q_W(n) ≈ -0.9890 (Standard Model)
Q_W_PROTON = 0.0713
Q_W_NEUTRON = -0.9890


@dataclass
class PNCResult:
    """PNC amplitude result for a transition n₁s ↔ n₂s."""

    E_pnc_au: float = 0.0
    E_pnc_iea: float = 0.0   # in 10⁻¹¹ i e a₀ / Q_W convenience unit
    Q_W: float = 0.0
    label_initial: str = ""
    label_final: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)


def weak_charge_qw(Z: int, N: int) -> float:
    """Standard-model nuclear weak charge Q_W = Z·Q_W(p) + N·Q_W(n).

    Includes 1-loop radiative correction (Marciano-Sirlin 1990)
    via the empirical reduction:

        Q_W^{SM} ≈ Z (1 − 4 sin²θ_W) − N

    with the effective sin²θ_W = 0.2381.  Default uses the proton/
    neutron values above.
    """
    return float(Z * Q_W_PROTON + N * Q_W_NEUTRON)


def weak_hamiltonian_radial_me(
    P_a: np.ndarray,
    Q_a: np.ndarray,
    P_b: np.ndarray,
    Q_b: np.ndarray,
    bsp_data: dict,
    *,
    rho_nuc_func,
    Q_W: float,
) -> float:
    """Radial matrix element of the weak Hamiltonian between two
    Dirac orbitals.

    H_W = (G_F / (2√2)) · Q_W · γ_5 · ρ_nuc(r)

    For s_{1/2} ↔ p_{1/2} parity-flipping transitions (the only
    non-zero case for the spin-independent dominant term):

        ⟨a|H_W|b⟩ = (G_F / (2√2)) · Q_W ·
                    ∫₀^∞ ρ_nuc(r) · [P_a Q_b − Q_a P_b] dr

    with γ_5 in the Dirac representation acting as a parity-flip
    between large and small components.

    Parameters
    ----------
    rho_nuc_func
        Callable rho_nuc(r) returning the normalised nuclear charge
        density at radius r (atomic units), normalised to ∫ρ dr = 1.
        Use the Fermi or uniform models from
        :mod:`atomx.potentials.nuclear`.
    Q_W
        Nuclear weak charge.

    Returns
    -------
    float
        Matrix element in atomic units.
    """
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    Pa = eval_P(P_a, bsp_data)
    Qa = eval_Q(Q_a, bsp_data)
    Pb = eval_P(P_b, bsp_data)
    Qb = eval_Q(Q_b, bsp_data)
    rho_grid = np.vectorize(rho_nuc_func)(gr.ravel()).reshape(gr.shape)
    integrand = rho_grid * (Pa * Qb - Qa * Pb)
    return (G_F_AU / (2.0 * np.sqrt(2.0))) * Q_W * float(np.sum(grw * integrand))


def pnc_amplitude_dirac_only(
    scf_result,
    initial_label: str,
    final_label: str,
    *,
    rho_nuc_func=None,
    N: Optional[int] = None,
    Q_W: Optional[float] = None,
    n_intermediate_max: Optional[int] = None,
    e_intermediate_max: Optional[float] = None,
    potential_mode: str = "VN-1",
) -> PNCResult:
    """Dirac-only PNC E1 amplitude via V^{N-1} sum-over-states.

    Computes the two off-diagonal second-order terms

        Σ_n <f|D|n p1/2><n p1/2|H_W|i>/(E_i - E_n)
      + Σ_n <f|H_W|n p1/2><n p1/2|D|i>/(E_f - E_n).

    The initial state must be an occupied s1/2 subshell in the SCF
    reference.  If the final label is not occupied (the usual Cs
    ``6s -> 7s`` case), the final state is the lowest-energy V^{N-1}
    virtual with κ=-1, so the final and intermediate states share the
    same Fock operator and denominator convention.

    Returns
    -------
    PNCResult
    """
    t0 = time.time()

    if not scf_result.converged:
        raise ValueError("SCFResult must be converged.")
    if Q_W is None:
        if N is None:
            raise ValueError("Either N or Q_W must be supplied.")
        Q_W = weak_charge_qw(int(scf_result.Z), int(N))
    if not hasattr(scf_result, "_bsp_data") or scf_result._bsp_data is None:
        sh_by_label = {s.label: s for s in scf_result.subshells}
        _validate_s_half_label(initial_label, sh_by_label, "initial_label")
        _validate_s_half_label(final_label, sh_by_label, "final_label")
        return PNCResult(
            E_pnc_au=0.0,
            E_pnc_iea=0.0,
            Q_W=float(Q_W),
            label_initial=initial_label,
            label_final=final_label,
            provenance={
                "method": "dirac_sos_vn_minus_1",
                "status": "no_bspline_sos_basis",
                "n_intermediates": 0,
                "cpu_seconds": time.time() - t0,
            },
        )

    bsp_data = scf_result._bsp_data
    default_rho = rho_nuc_func is None
    if rho_nuc_func is None:
        rho_nuc_func = _default_contact_density(bsp_data)

    cache = ConvolCache(bsp_data)
    kappa_list, occ_basis, virt_basis = build_sos_virtual_basis(
        scf_result, cache=cache, kappa_max=1,
        n_virt_max=n_intermediate_max, e_virt_max=e_intermediate_max,
        potential_mode=potential_mode,
    )

    sh_by_label = {s.label: s for s in scf_result.subshells}
    initial = _find_occupied_endpoint(
        initial_label, -1, sh_by_label, occ_basis,
        endpoint_name="initial_label",
    )
    final = _find_endpoint(
        final_label, -1, sh_by_label, occ_basis, virt_basis,
        endpoint_name="final_label",
    )

    P_i, Q_i, eps_i, lab_i, kappa_i = initial
    P_f, Q_f, eps_f, lab_f, kappa_f = final

    total = 0.0
    term_initial = 0.0
    term_final = 0.0
    n_used = 0

    # H_W is a rank-0 odd-parity operator: only p1/2 (κ=+1) can connect
    # the s1/2 endpoints.  p3/2 is deliberately excluded by the same
    # angular helper used by the EDM SOS code.
    kappa_n = 1
    if kappa_n in kappa_list:
        ang_w_ni = pseudoscalar_angular(kappa_i, kappa_n)
        ang_w_fn = pseudoscalar_angular(kappa_n, kappa_f)
        ang_d_fn = angular_eL(kappa_f, kappa_n, 1)
        ang_d_ni = angular_eL(kappa_n, kappa_i, 1)
        if (
            abs(ang_w_ni) > 1e-15 and abs(ang_w_fn) > 1e-15
            and abs(ang_d_fn) > 1e-15 and abs(ang_d_ni) > 1e-15
        ):
            for (P_n, Q_n, eps_n) in virt_basis.get(kappa_n, []):
                den_i = eps_i - eps_n
                den_f = eps_f - eps_n
                if abs(den_i) <= 1e-12 or abs(den_f) <= 1e-12:
                    continue

                d_fn = ang_d_fn * radial_eL(P_f, Q_f, P_n, Q_n, bsp_data, L=1)
                w_ni = ang_w_ni * weak_hamiltonian_radial_me(
                    P_n, Q_n, P_i, Q_i, bsp_data,
                    rho_nuc_func=rho_nuc_func, Q_W=Q_W,
                )
                w_fn = ang_w_fn * weak_hamiltonian_radial_me(
                    P_f, Q_f, P_n, Q_n, bsp_data,
                    rho_nuc_func=rho_nuc_func, Q_W=Q_W,
                )
                d_ni = ang_d_ni * radial_eL(P_n, Q_n, P_i, Q_i, bsp_data, L=1)

                contrib_i = d_fn * w_ni / den_i
                contrib_f = w_fn * d_ni / den_f
                term_initial += contrib_i
                term_final += contrib_f
                total += contrib_i + contrib_f
                n_used += 1

    return PNCResult(
        E_pnc_au=float(total),
        E_pnc_iea=float((total / Q_W) / 1.0e-11) if Q_W != 0.0 else 0.0,
        Q_W=float(Q_W),
        label_initial=lab_i,
        label_final=lab_f,
        provenance={
            "method": "dirac_sos_vn_minus_1",
            "potential_mode": potential_mode,
            "rho_nuc_func": "narrow_gaussian_contact" if default_rho else "user",
            "intermediate_kappa": kappa_n,
            "n_intermediates": n_used,
            "n_virtuals": {k: len(v) for k, v in virt_basis.items()},
            "eps_initial": float(eps_i),
            "eps_final": float(eps_f),
            "term_initial": float(term_initial),
            "term_final": float(term_final),
            "cpu_seconds": time.time() - t0,
        },
    )


def _default_contact_density(bsp_data):
    """Return a narrow radial Gaussian normalised as ∫rho(r) dr = 1."""
    gr = np.asarray(bsp_data["gr"], dtype=float)
    positive = gr[gr > 0.0]
    sigma = max(float(np.min(positive)) * 4.0, 1.0e-8) if positive.size else 1.0e-8
    norm = 2.0 / (np.sqrt(np.pi) * sigma)

    def rho(r):
        x = float(r) / sigma
        return norm * np.exp(-(x * x))

    return rho


def _parse_label_n(label: str) -> Optional[int]:
    match = re.match(r"^\s*(\d+)", label)
    return int(match.group(1)) if match else None


def _find_occupied_endpoint(
    label: str,
    expected_kappa: int,
    sh_by_label,
    occ_basis,
    *,
    endpoint_name: str,
) -> Tuple[np.ndarray, np.ndarray, float, str, int]:
    _validate_s_half_label(label, sh_by_label, endpoint_name)
    for P, Q, eps, lab in occ_basis.get(expected_kappa, []):
        if lab == label:
            return P, Q, eps, lab, expected_kappa
    raise ValueError(f"Could not find occupied orbital coefficients for {label!r}.")


def _validate_s_half_label(label: str, sh_by_label, endpoint_name: str) -> None:
    sh = sh_by_label.get(label)
    if sh is None:
        raise ValueError(
            f"{endpoint_name} {label!r} must be an occupied SCF subshell. "
            f"Available: {list(sh_by_label.keys())}"
        )
    if sh.kappa != -1:
        raise ValueError(f"{endpoint_name} {label!r} must be s1/2 (kappa=-1).")


def _find_endpoint(
    label: str,
    expected_kappa: int,
    sh_by_label,
    occ_basis,
    virt_basis,
    *,
    endpoint_name: str,
) -> Tuple[np.ndarray, np.ndarray, float, str, int]:
    if label in sh_by_label:
        return _find_occupied_endpoint(
            label, expected_kappa, sh_by_label, occ_basis,
            endpoint_name=endpoint_name,
        )
    if not virt_basis.get(expected_kappa):
        raise ValueError(f"No V^{{N-1}} virtual s1/2 states available for {label!r}.")

    requested_n = _parse_label_n(label)
    occupied_ns = [s.n for s in sh_by_label.values() if s.kappa == expected_kappa]
    expected_n = (max(occupied_ns) + 1) if occupied_ns else None
    if requested_n is not None and expected_n is not None and requested_n != expected_n:
        raise ValueError(
            f"{endpoint_name} {label!r} is not occupied; only the lowest "
            f"V^{{N-1}} virtual s1/2 is currently exposed (expected n={expected_n})."
        )

    P, Q, eps = virt_basis[expected_kappa][0]
    return P, Q, eps, label, expected_kappa


__all__ = [
    "PNCResult",
    "pnc_amplitude_dirac_only",
    "weak_hamiltonian_radial_me",
    "weak_charge_qw",
    "G_F_AU",
    "Q_W_PROTON",
    "Q_W_NEUTRON",
]
