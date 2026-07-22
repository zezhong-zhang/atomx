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
from atomx.potentials.nuclear import fermi_params
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
    # Conventional coefficient: C = -(N/Q_W)·E_pnc_au/10⁻¹¹.
    k_pnc: Optional[float] = None
    # Diagnostic only: E_PNC/(Q_W·10⁻¹¹), not the Boulder/Dzuba C.
    E_pnc_per_qw: float = 0.0
    E_pnc_iea: float = 0.0   # legacy alias for E_pnc_per_qw, not Boulder/Dzuba C
    Q_W: float = 0.0
    N: Optional[int] = None
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
        density at radius r (atomic units), normalised in 3-D as
        4π∫r²ρ(r)dr = 1.  The weak radial matrix element itself remains
        ∫ρ(r)[P_a Q_b - Q_a P_b]dr because P and Q are radial components
        with the volume factor already absorbed.
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
    if rho_nuc_func is None and N is None:
        raise ValueError(
            "Default finite nuclear density requires N so A=Z+N is known; "
            "pass N or an explicit rho_nuc_func."
        )
    if not hasattr(scf_result, "_bsp_data") or scf_result._bsp_data is None:
        sh_by_label = {s.label: s for s in scf_result.subshells}
        _validate_s_half_label(initial_label, sh_by_label, "initial_label")
        _validate_s_half_label(final_label, sh_by_label, "final_label")
        k_pnc, per_qw, unit_reason = _pnc_unit_diagnostics(0.0, float(Q_W), N)
        return PNCResult(
            E_pnc_au=0.0,
            k_pnc=k_pnc,
            E_pnc_per_qw=per_qw,
            E_pnc_iea=per_qw,
            Q_W=float(Q_W),
            N=N,
            label_initial=initial_label,
            label_final=final_label,
            provenance={
                "method": "dirac_sos_vn_minus_1",
                "status": "no_bspline_sos_basis",
                "k_pnc_status": unit_reason,
                "n_intermediates": 0,
                "term_initial": 0.0,
                "term_final": 0.0,
                "cpu_seconds": time.time() - t0,
            },
        )

    bsp_data = scf_result._bsp_data
    default_rho = rho_nuc_func is None
    density_provenance: Dict[str, Any] = {}
    if rho_nuc_func is None:
        rho_nuc_func, density_provenance = _default_finite_nuclear_density(
            int(scf_result.Z), int(N), bsp_data,
        )

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

    k_pnc, per_qw, unit_reason = _pnc_unit_diagnostics(total, float(Q_W), N)
    return PNCResult(
        E_pnc_au=float(total),
        k_pnc=k_pnc,
        E_pnc_per_qw=per_qw,
        E_pnc_iea=per_qw,
        Q_W=float(Q_W),
        N=N,
        label_initial=lab_i,
        label_final=lab_f,
        provenance={
            "method": "dirac_sos_vn_minus_1",
            "potential_mode": potential_mode,
            "rho_nuc_func": "fermi_finite_grid_normalized" if default_rho else "user",
            "k_pnc_status": unit_reason,
            "intermediate_kappa": kappa_n,
            "n_intermediates": n_used,
            "n_virtuals": {k: len(v) for k, v in virt_basis.items()},
            "eps_initial": float(eps_i),
            "eps_final": float(eps_f),
            "term_initial": float(term_initial),
            "term_final": float(term_final),
            "cpu_seconds": time.time() - t0,
            **density_provenance,
        },
    )


def _pnc_unit_diagnostics(
    E_pnc_au: float,
    Q_W: float,
    N: Optional[int],
) -> Tuple[Optional[float], float, str]:
    """Return conventional C and per-Q_W diagnostics for E_PNC."""
    per_qw = float((E_pnc_au / Q_W) / 1.0e-11) if Q_W != 0.0 else np.nan
    if N is None:
        return None, per_qw, "k_pnc_unavailable_without_N"
    if Q_W == 0.0:
        return np.nan, per_qw, "k_pnc_unavailable_zero_Q_W"
    return float(-(int(N) / Q_W) * E_pnc_au / 1.0e-11), per_qw, "ok"


def _default_finite_nuclear_density(
    Z: int,
    N: int,
    bsp_data,
    *,
    tol: float = 1e-6,
    min_core_points: int = 6,
    min_surface_points: int = 3,
):
    """Return an A-based Fermi density normalised as 4π∫r²rho(r)dr = 1."""
    gr = np.asarray(bsp_data["gr"], dtype=float)
    grw = np.asarray(bsp_data["grw"], dtype=float)
    A = int(Z) + int(N)
    a, c = fermi_params(int(Z), A)
    resolution = _fermi_density_resolution(gr, a, c)
    if (
        resolution["density_resolution_core_points"] < min_core_points
        or resolution["density_resolution_surface_points"] < min_surface_points
    ):
        raise ValueError(
            "Finite nuclear density is under-resolved on the integration grid; "
            f"found {resolution['density_resolution_core_points']} quadrature points "
            f"within r <= c+4a and "
            f"{resolution['density_resolution_surface_points']} points in "
            f"c-2a <= r <= c+2a, but require at least {min_core_points} "
            f"and {min_surface_points}, respectively. "
            f"Fermi scale: a={a:.6e} bohr, c={c:.6e} bohr."
        )

    def shape_values(r):
        r_arr = np.asarray(r, dtype=float)
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + np.exp((r_arr - c) / a))

    shape_grid = shape_values(gr)
    norm = float(4.0 * np.pi * np.sum(grw * gr * gr * shape_grid))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(
            "Could not normalize finite nuclear density on the integration grid; "
            f"normalization was {norm!r}."
        )
    rho_grid = shape_grid / norm
    residual = float(abs(4.0 * np.pi * np.sum(grw * gr * gr * rho_grid) - 1.0))
    if not np.isfinite(residual) or residual > tol:
        raise ValueError(
            "Finite nuclear density is not resolved on the integration grid; "
            f"3-D normalization residual {residual:.3e} exceeds {tol:.1e}."
        )

    def rho(r):
        values = shape_values(r) / norm
        if np.ndim(values) == 0:
            return float(values)
        return values

    provenance = {
        "density_model": "fermi",
        "density_A": A,
        "density_N": int(N),
        "density_fermi_a_bohr": float(a),
        "density_fermi_c_bohr": float(c),
        "density_norm_3d": float(4.0 * np.pi * np.sum(grw * gr * gr * rho_grid)),
        "density_norm_residual": residual,
        **resolution,
    }
    return rho, provenance


def _fermi_density_resolution(gr: np.ndarray, a: float, c: float) -> Dict[str, Any]:
    """Resolution metrics for the Fermi core and half-density surface."""
    finite_gr = np.asarray(gr, dtype=float)
    finite_gr = finite_gr[np.isfinite(finite_gr)]
    surface_lo = max(0.0, c - 2.0 * a)
    surface_hi = c + 2.0 * a
    core_hi = c + 4.0 * a
    core_points = int(np.count_nonzero(finite_gr <= core_hi))
    surface_points = int(
        np.count_nonzero((finite_gr >= surface_lo) & (finite_gr <= surface_hi))
    )
    first_grid_point = float(np.min(finite_gr)) if finite_gr.size else np.nan
    return {
        "density_resolution_core_points": core_points,
        "density_resolution_surface_points": surface_points,
        "density_resolution_core_radius_bohr": float(core_hi),
        "density_resolution_surface_lo_bohr": float(surface_lo),
        "density_resolution_surface_hi_bohr": float(surface_hi),
        "density_resolution_first_grid_point_bohr": first_grid_point,
    }


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
