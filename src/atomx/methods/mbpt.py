"""Many-body perturbation theory on a Dirac-Hartree-Fock reference.

Currently implements **second-order MBPT** for a closed-shell DHF
reference.  The correlation energy is

    E^(2) = − Σ_{a≤b core} Σ_{r≤s virt}
                (1+δ_ab)^{-1} (1+δ_rs)^{-1}
                · Σ_J (2J+1) [V_J(abrs) − V_J(absr)]^2
                / (ε_r + ε_s − ε_a − ε_b)

with the J-coupled two-particle Coulomb matrix element

    V_J(abrs) = Σ_k (−1)^{j_b+j_r+J}
                · {j_a j_b J; j_s j_r k}
                · X^k(abrs)

and

    X^k(abrs) = ⟨κ_a‖C^k‖κ_r⟩ ⟨κ_b‖C^k‖κ_s⟩ R^k(abrs).

The reduced matrix element of the spherical tensor C^k between
relativistic spinors with quantum numbers κ_a, κ_b is

    ⟨κ_a‖C^k‖κ_b⟩ = (−1)^{j_a+1/2} √[(2j_a+1)(2j_b+1)]
                     · 3j(j_a, k, j_b; −1/2, 0, 1/2)
                     · π(l_a + k + l_b even),

where π is the parity selector; the angular sum in V_J vanishes if
the parity rule is violated for either pair.

The virtual orbitals are obtained by diagonalising the closed-shell
DHF Fock matrix for each κ in the converged B-spline basis.  This is
the "V^N-1 frozen-core" virtual basis (each virtual sees the same
average potential the occupied orbitals saw at convergence) and is
the standard choice in atomic MBPT (Johnson, Sapirstein, Blundell,
1988).  All physical eigenstates above the negative-energy continuum
are kept, including the high-lying B-spline pseudo-continuum, which
provides a near-complete representation of positive-energy states.

References
----------
- I. Lindgren & J. Morrison, *Atomic Many-Body Theory* (2nd ed.,
  Springer, 1986), Chap. 12-13.
- W. R. Johnson, *Atomic Structure Theory* (Springer, 2007), §11.4.
- W. R. Johnson, S. A. Blundell, J. Sapirstein,
  *Phys. Rev. A* **37**, 2764 (1988) — relativistic many-body in a
  finite B-spline basis.
- V. A. Dzuba, V. V. Flambaum, M. G. Kozlov,
  *Phys. Rev. A* **54**, 3948 (1996) — all-order method (extension
  beyond MBPT(2); not implemented here).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from atomx.angular.coefficients import (
    _l_from_kappa,
    _twoj_from_kappa,
    exchange_lambda,
    reduced_C,
)
from atomx.angular.wigner import wigner_6j
from atomx.constants import C_SQ
from atomx.numerics.bspline.convol import (
    ConvolCache,
    build_slater_rk_bilinear,
    compute_direct_potential_cell,
    compute_exchange_matrix_cell,
    compute_slater_rk_cell,
    compute_slater_rk_table,
    eval_slater_rk_bilinear,
)
from atomx.numerics.bspline.dirac_assembly import (
    build_dirac_matrices,
    solve_dirac,
)

# Module-level logger
logger = logging.getLogger(__name__)

# The ladder path re-evaluates the same small-integer angular factors tens
# of thousands of times per pair block; memoize at this call-site level
# (the shared numba wigner module stays uncached).
_wigner_6j = lru_cache(maxsize=None)(wigner_6j)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MBPT2Result:
    """Outcome of a closed-shell MBPT(2) calculation.

    Attributes
    ----------
    correlation_energy
        Total second-order correlation energy in hartree.
    pair_contributions
        ``{(a_label, b_label): ΔE}`` — partial sum over (r,s) for
        each occupied pair.  Useful diagnostic; pairs sum to
        ``correlation_energy``.
    n_virtuals
        ``{kappa: count}`` — number of virtual spinors used per κ.
    kappa_max
        Largest |κ| included for both occupied and virtual pairs.
    n_max
        Cap on principal-like index for the virtual basis (None →
        all B-spline eigenstates kept).
    cpu_seconds
        Wall time of the angular-radial double sum.
    provenance
        Free-form dict for the basis size, included multipoles, etc.
    """

    correlation_energy: float = 0.0
    pair_contributions: Dict[Tuple[str, str], float] = field(default_factory=dict)
    n_virtuals: Dict[int, int] = field(default_factory=dict)
    kappa_max: int = 0
    n_max: Optional[int] = None
    cpu_seconds: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Angular helpers — canonical implementations live in atomx.angular.coefficients
# (imported above).  reduced_C stays in __all__ and is test-imported;
# _l_from_kappa/_twoj_from_kappa are thin aliases used throughout this module
# and by mbpt_transitions.py.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _allowed_k_range(kappa_a: int, kappa_b: int) -> List[int]:
    """k satisfying triangle(j_a,k,j_b) ∧ parity(l_a+k+l_b even).

    (lru_cached; callers must not mutate the returned list.)
    """
    j_a = abs(kappa_a) - 0.5
    j_b = abs(kappa_b) - 0.5
    k_min = int(abs(j_a - j_b))
    k_max = int(j_a + j_b)
    l_a = _l_from_kappa(kappa_a)
    l_b = _l_from_kappa(kappa_b)
    return [k for k in range(k_min, k_max + 1) if (k + l_a + l_b) % 2 == 0]


# ---------------------------------------------------------------------------
# Closed-shell V^N Fock builder for an arbitrary κ
# ---------------------------------------------------------------------------


def _build_closed_shell_fock(
    scf_result,
    kappa: int,
    *,
    cache: Optional[ConvolCache] = None,
    potential_mode: str = "VN-1",
):
    """Assemble the closed-shell DHF Fock matrix for orbitals of κ.

    For a closed-shell core (every subshell completely filled, q_b =
    2j_b+1) the Fock operator seen by an orbital of κ is

        h_κ = h_0(κ) + Σ_b q_b' · V_d^{k=0}(b)
                     − Σ_b Σ_k q_b' · Λ^k(κ, κ_b)/[(2j_κ+1)(2j_b+1)] · K^k(b)

    Only k=0 direct multipoles survive on a closed shell.  Exchange
    runs over allowed k.

    Parameters
    ----------
    potential_mode : {"VN-1", "VN"}
        ``"VN-1"`` (default, standard for atomic MBPT, Sapirstein-
        Johnson-Blundell 1988): the virtual orbital with quantum
        number κ sees N-1 electrons.  Implemented by subtracting 1
        from the occupation of any occupied subshell whose κ matches
        the target.  Gives bound Rydberg-like virtuals.
        ``"VN"``: full N-electron mean field.  Cheaper and matches
        the SCF Fock exactly, but the virtuals are essentially
        unbound (electron-affinity-like) and produce a poor MBPT
        expansion.

    Parameters
    ----------
    scf_result
        Converged closed-shell ``SCFResult``.
    kappa
        Quantum number κ for which to build the Fock matrix.
    cache
        Reusable :class:`ConvolCache` keyed on the B-spline basis;
        share across all κ.

    Returns
    -------
    H, S
        Generalised eigenvalue pair, ready for ``solve_dirac``.
    """
    bsp_data = scf_result._bsp_data
    if bsp_data is None:
        raise ValueError(
            "SCFResult is missing _bsp_data — re-run DiracFock.solve()."
        )
    Z = scf_result.Z
    if Z is None:
        raise ValueError("SCFResult is missing Z.")

    ns_full = bsp_data["ns"]
    nuclear_potential_func = scf_result.provenance.get("nuclear_potential_func") \
        if hasattr(scf_result, "provenance") else None

    # One-electron H₀ + S
    H, S = build_dirac_matrices(ns_full, kappa, Z, bsp_data, nuclear_potential_func)

    if cache is None:
        cache = ConvolCache(bsp_data)

    tj_kappa = _twoj_from_kappa(kappa)

    # ── V^{N-1} bookkeeping ─────────────────────────────────────────
    # Subtract one electron from the OUTERMOST occupied subshell to
    # form the (N-1)-electron core potential seen by every virtual.
    # This is the standard Sapirstein-Johnson-Blundell choice (1988)
    # and matches FAC's "frozen-core" mode.
    if potential_mode == "VN-1":
        outer = max(
            scf_result.subshells,
            key=lambda s: (s.n, _l_from_kappa(s.kappa), -abs(s.kappa)),
        )
        reduce_id = id(outer)
    elif potential_mode == "VN":
        reduce_id = None
    else:
        raise ValueError(
            f"Unknown potential_mode {potential_mode!r}; "
            f"expected 'VN-1' or 'VN'."
        )

    # Loop over occupied subshells b (kappa_b grouped)
    for sh in scf_result.subshells:
        kb = sh.kappa
        tj_b = _twoj_from_kappa(kb)
        q_b = float(sh.occupation)
        if id(sh) == reduce_id:
            q_b -= 1.0
        if q_b <= 0.0:
            continue
        # Fetch (P, Q) for this subshell.  orbitals_raw maps
        # kappa → list of (P, Q, occ).  Match by principal-index order
        # within the κ list (matches DiracFock layout).
        orb_list = scf_result.orbitals_raw.get(kb, [])
        # Find this orbital among the list ordered by n
        sh_kappa = [s for s in scf_result.subshells if s.kappa == kb]
        idx_in_kappa = sh_kappa.index(sh)
        if idx_in_kappa >= len(orb_list):
            continue
        Pb, Qb, _ = orb_list[idx_in_kappa]

        # ── Direct: k=0 only on closed shell ────────────────────────
        V_PP, V_QQ = compute_direct_potential_cell(Pb, Qb, bsp_data, 0, cache=cache)
        coef_d = q_b
        nsp_p = V_PP.shape[0]
        nsq_q = V_QQ.shape[0]
        H[:nsp_p, :nsp_p] += coef_d * V_PP
        H[ns_full:ns_full + nsq_q, ns_full:ns_full + nsq_q] += coef_d * V_QQ

        # ── Exchange: sum over allowed k ────────────────────────────
        for k in _allowed_k_range(kappa, kb):
            lam = exchange_lambda(kappa, kb, k)
            if abs(lam) < 1e-15:
                continue
            # Closed-shell exchange coefficient for orbital with κ
            # from a fully filled subshell with κ_b:
            #   c^k_x = − q_b · Λ^k / [(2j_κ+1)(2j_b+1)]
            #         = − Λ^k / (2j_κ+1)   when q_b = 2j_b+1.
            coef_x = - q_b * lam / ((tj_kappa + 1) * (tj_b + 1))

            K_PP, K_QQ = compute_exchange_matrix_cell(
                Pb, Qb, Pb, Qb, bsp_data, k, cache=cache
            )
            nsp_k = K_PP.shape[0]
            nsq_k = K_QQ.shape[0]
            H[:nsp_k, :nsp_k] += coef_x * K_PP
            H[ns_full:ns_full + nsq_k, ns_full:ns_full + nsq_k] += coef_x * K_QQ

    return H, S


# ---------------------------------------------------------------------------
# Virtual-basis builder
# ---------------------------------------------------------------------------


def build_virtual_basis(
    scf_result,
    kappa: int,
    *,
    cache: Optional[ConvolCache] = None,
    e_max: Optional[float] = None,
    potential_mode: str = "VN-1",
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], List[Tuple[np.ndarray, np.ndarray, float]]]:
    """Diagonalise h_κ in the converged B-spline basis.

    Returns
    -------
    occupied : list[(P, Q, ε)]
        Eigenstates that match an occupied subshell with this κ.
        Identified by overlap ≥ 0.5 with a stored DHF orbital.
    virtuals : list[(P, Q, ε)]
        All remaining physical eigenstates (E > -2c² ≈ negative
        continuum threshold), sorted by increasing energy.  Includes
        bound + pseudo-continuum positive-energy states.
    """
    bsp_data = scf_result._bsp_data
    ns_full = bsp_data["ns"]
    nsp = bsp_data.get("nsp", ns_full)
    nsq = bsp_data.get("nsq", ns_full)

    H, S = _build_closed_shell_fock(scf_result, kappa, cache=cache,
                                     potential_mode=potential_mode)
    res = solve_dirac(H, S, ns_full, kappa, nsp=nsp, nsq=nsq)

    # Filter to physical states (above Dirac negative-energy continuum)
    physical_mask = res.eigenvalues > -1.5 * C_SQ
    if e_max is not None:
        physical_mask &= res.eigenvalues < e_max
    phys_idx = np.where(physical_mask)[0]
    phys_idx = phys_idx[np.argsort(res.eigenvalues[phys_idx])]

    # Split eigenvectors into (P, Q) per the DiracResult layout
    states: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for i in phys_idx:
        vec = res.coefficients[:, i]
        P = np.zeros(ns_full)
        Q = np.zeros(ns_full)
        P[:nsp] = vec[:nsp]
        Q[:nsq] = vec[ns_full:ns_full + nsq]
        states.append((P, Q, float(res.eigenvalues[i])))

    # Identify which states are the occupied ones — by overlap with
    # the stored DHF orbitals.
    occ_orbs = scf_result.orbitals_raw.get(kappa, [])
    # Use S overlap matrix for proper inner product
    S_full = S if "bsp_p" in bsp_data else None

    def _overlap(Pa, Qa, Pb, Qb):
        # ⟨a|b⟩ via S matrix
        a = np.concatenate([Pa, Qa])
        b = np.concatenate([Pb, Qb])
        return float(a @ (S_full @ b)) if S_full is not None else float(a @ b)

    occupied_indices = set()
    for (P_occ, Q_occ, _) in occ_orbs:
        best, best_idx = -1.0, -1
        for i, (P_v, Q_v, _) in enumerate(states):
            if i in occupied_indices:
                continue
            ov = abs(_overlap(P_occ, Q_occ, P_v, Q_v))
            if ov > best:
                best, best_idx = ov, i
        if best_idx >= 0 and best > 0.5:
            occupied_indices.add(best_idx)

    occupied = [states[i] for i in sorted(occupied_indices)]
    virtuals = [states[i] for i in range(len(states)) if i not in occupied_indices]
    return occupied, virtuals


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def mbpt2_energy(
    scf_result,
    *,
    kappa_max: int = 4,
    n_virt_max: Optional[int] = None,
    multipole_max: Optional[int] = None,
    e_virt_max: Optional[float] = None,
    verbose: bool = False,
) -> MBPT2Result:
    """Compute the second-order MBPT correlation energy on a closed-shell
    DHF reference.

    Parameters
    ----------
    scf_result
        Converged ``SCFResult`` produced by :class:`DiracFock`.  All
        subshells must be fully filled (closed shell).
    kappa_max
        Largest ``|κ|`` considered for both occupied and virtual
        spaces.  Default 4 covers s, p, d, f, g partial waves.
    n_virt_max
        Optional cap on the number of virtual states *per κ* (lowest
        ``n_virt_max`` retained).  Default ``None`` → keep every
        physical eigenstate of the B-spline matrix.
    multipole_max
        Cap on the multipole order ``k`` in the Coulomb expansion.
        Default ``None`` → use the full triangle / parity range
        (selection rules typically truncate at ``k ≲ 2·kappa_max``).
    e_virt_max
        Optional energy ceiling for virtual orbitals (hartree).  Use
        to drop the highest pseudo-continuum states which contribute
        negligibly but cost time.
    verbose
        If True, log per-pair contributions.

    Returns
    -------
    MBPT2Result
    """
    import time

    t0 = time.time()

    # ── Sanity ──────────────────────────────────────────────────────
    if not scf_result.converged:
        raise ValueError("SCFResult must be converged.")
    for sh in scf_result.subshells:
        max_occ = _twoj_from_kappa(sh.kappa) + 1
        if abs(sh.occupation - max_occ) > 0.01:
            raise ValueError(
                f"MBPT(2) requires a closed-shell reference; subshell "
                f"{sh.label} has occupation {sh.occupation} (expected "
                f"{max_occ})."
            )

    bsp_data = scf_result._bsp_data
    cache = ConvolCache(bsp_data)

    # ── Build virtual basis for every κ in scope ───────────────────
    kappa_list = []
    for k_abs in range(1, kappa_max + 2):
        # κ = -ℓ-1 (ℓ-1/2 missing) and +ℓ (ℓ+1/2 missing) pairs
        kappa_list.append(-k_abs)         # κ = -1, -2, -3, ... → ℓ = 0, 1, 2, ...
        if k_abs >= 1:
            kappa_list.append(+k_abs)     # κ = +1 (p_{1/2}), +2 (d_{3/2}), ...
    # Remove κ=+0 (doesn't exist) and trim to those with ℓ ≤ kappa_max
    kappa_list = [
        k for k in kappa_list
        if _l_from_kappa(k) <= kappa_max
    ]

    occ_basis: Dict[int, List[Tuple[np.ndarray, np.ndarray, float, str]]] = {}
    virt_basis: Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]] = {}

    for kappa in kappa_list:
        occ, virt = build_virtual_basis(scf_result, kappa, cache=cache,
                                          e_max=e_virt_max)
        # Sapirstein-Johnson-Blundell prescription: use SCF ε_a for
        # occupied orbitals (the V^{N-1} eigenvalues differ and would
        # over-shrink the energy denominator).
        kappa_subshells = [s for s in scf_result.subshells if s.kappa == kappa]
        occ_with_labels: List[Tuple[np.ndarray, np.ndarray, float, str]] = []
        for i, (P, Q, _eps_v_n1) in enumerate(occ):
            if i < len(kappa_subshells):
                sh = kappa_subshells[i]
                lab = sh.label
                eps_scf = scf_result.orbital_energies.get(lab)
                eps_use = eps_scf if eps_scf is not None else _eps_v_n1
            else:
                lab = f"occ-{kappa}-{i}"
                eps_use = _eps_v_n1
            occ_with_labels.append((P, Q, eps_use, lab))
        if n_virt_max is not None:
            virt = virt[:n_virt_max]
        occ_basis[kappa] = occ_with_labels
        virt_basis[kappa] = virt

    # ── Two-particle MBPT(2) sum ───────────────────────────────────
    E_corr = 0.0
    pair_contribs: Dict[Tuple[str, str], float] = {}

    occ_kappas = [k for k in kappa_list if occ_basis[k]]

    for ka_idx, kappa_a in enumerate(occ_kappas):
        for kappa_b in occ_kappas[ka_idx:]:
            for orb_a_idx, (Pa, Qa, eps_a, lab_a) in enumerate(occ_basis[kappa_a]):
                # When κ_a == κ_b, avoid double-counting orbitals
                if kappa_a == kappa_b:
                    inner_iter_b = enumerate(occ_basis[kappa_b][orb_a_idx:],
                                             start=orb_a_idx)
                else:
                    inner_iter_b = enumerate(occ_basis[kappa_b])
                for orb_b_idx, (Pb, Qb, eps_b, lab_b) in inner_iter_b:
                    same_ab = (kappa_a == kappa_b and orb_a_idx == orb_b_idx)
                    pair_E = _mbpt2_pair_contribution(
                        Pa, Qa, eps_a, kappa_a,
                        Pb, Qb, eps_b, kappa_b,
                        virt_basis, kappa_list,
                        bsp_data, cache,
                        multipole_max=multipole_max,
                        same_ab=same_ab,
                    )
                    E_corr += pair_E
                    pair_contribs[(lab_a, lab_b)] = pair_E
                    if verbose:
                        logger.debug(f"  ({lab_a}, {lab_b})  ΔE^(2) = {pair_E:.6e} Ha")

    elapsed = time.time() - t0

    return MBPT2Result(
        correlation_energy=E_corr,
        pair_contributions=pair_contribs,
        n_virtuals={k: len(v) for k, v in virt_basis.items()},
        kappa_max=kappa_max,
        n_max=n_virt_max,
        cpu_seconds=elapsed,
        provenance={
            "multipole_max": multipole_max,
            "e_virt_max": e_virt_max,
            "kappa_list": kappa_list,
            "Z": scf_result.Z,
            "n_subshells": len(scf_result.subshells),
        },
    )


# ---------------------------------------------------------------------------
# Pair contribution: J-coupled antisymmetrised form
# ---------------------------------------------------------------------------


def _mbpt2_pair_contribution(
    Pa, Qa, eps_a, kappa_a,
    Pb, Qb, eps_b, kappa_b,
    virt_basis: Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]],
    kappa_list: List[int],
    bsp_data,
    cache: ConvolCache,
    *,
    multipole_max: Optional[int],
    same_ab: bool,
) -> float:
    """Sum over (r,s) virtuals for one occupied pair (a,b)."""
    tj_a = _twoj_from_kappa(kappa_a)
    tj_b = _twoj_from_kappa(kappa_b)
    tJ_min = abs(tj_a - tj_b)
    tJ_max = tj_a + tj_b
    pair_E = 0.0
    norm_ab = 2.0 if same_ab else 1.0  # δ_ab factor

    # Bilinear R^k form: M^k_(ab) depends only on (a, b, k) — build once
    # per k, reuse across every (κ_r, κ_s) channel AND both direct and
    # exchange tables (exchange only swaps the virtual slots).  Turns the
    # O(n_r·n_s) scalar compute_slater_rk_cell calls per channel into one
    # GEMM against the stacked virtual coefficients.
    M_ab: Dict[int, tuple] = {}

    def _M(k):
        Mk = M_ab.get(k)
        if Mk is None:
            Mk = build_slater_rk_bilinear(Pa, Qa, Pb, Qb, bsp_data, k,
                                          cache=cache)
            M_ab[k] = Mk
        return Mk

    # Stacked virtual coefficient columns per κ (GEMM operands)
    virt_stacks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _stack(kappa):
        st = virt_stacks.get(kappa)
        if st is None:
            vl = virt_basis[kappa]
            st = (np.column_stack([v[0] for v in vl]),
                  np.column_stack([v[1] for v in vl]))
            virt_stacks[kappa] = st
        return st

    for kappa_r in kappa_list:
        # Triangle on (j_a, k, j_r) for at least one k ⇒ skip if range empty
        if not _allowed_k_range(kappa_a, kappa_r) and \
           not _allowed_k_range(kappa_b, kappa_r):
            continue
        tj_r = _twoj_from_kappa(kappa_r)

        for kappa_s in kappa_list:
            tj_s = _twoj_from_kappa(kappa_s)
            # J-pair triangle on (j_r, j_s, J) must overlap (j_a, j_b, J)
            tJ_rs_min = abs(tj_r - tj_s)
            tJ_rs_max = tj_r + tj_s
            tJ_lo = max(tJ_min, tJ_rs_min)
            tJ_hi = min(tJ_max, tJ_rs_max)
            if tJ_lo > tJ_hi:
                continue
            # Parity of total CSF: l_a+l_b = l_r+l_s (mod 2) for non-zero
            # angular factor.
            l_a = _l_from_kappa(kappa_a)
            l_b = _l_from_kappa(kappa_b)
            l_r = _l_from_kappa(kappa_r)
            l_s = _l_from_kappa(kappa_s)
            if (l_a + l_b + l_r + l_s) % 2 != 0:
                continue

            # Allowed multipole range (intersection of direct & exchange)
            k_d_list = _allowed_k_range(kappa_a, kappa_r)
            k_d_list = [k for k in k_d_list if k in _allowed_k_range(kappa_b, kappa_s)]
            k_e_list = _allowed_k_range(kappa_a, kappa_s)
            k_e_list = [k for k in k_e_list if k in _allowed_k_range(kappa_b, kappa_r)]
            if multipole_max is not None:
                k_d_list = [k for k in k_d_list if k <= multipole_max]
                k_e_list = [k for k in k_e_list if k <= multipole_max]
            if not k_d_list and not k_e_list:
                continue

            # Pre-compute reduced C^k matrix elements
            Ck_ar = {k: reduced_C(kappa_a, kappa_r, k) for k in k_d_list}
            Ck_bs = {k: reduced_C(kappa_b, kappa_s, k) for k in k_d_list}
            Ck_as = {k: reduced_C(kappa_a, kappa_s, k) for k in k_e_list}
            Ck_br = {k: reduced_C(kappa_b, kappa_r, k) for k in k_e_list}

            # Loop over virtuals (r in κ_r, s in κ_s)
            virt_r = virt_basis[kappa_r]
            virt_s = virt_basis[kappa_s]
            if not virt_r or not virt_s:
                continue
            CP_r, CQ_r = _stack(kappa_r)
            CP_s, CQ_s = _stack(kappa_s)
            # Whole-channel Slater tables: Rd[k][r, s] = R^k(a r, b s),
            # Re[k][r, s] = R^k(a s, b r).  Same Ck gates as the scalar path.
            Rd_tab = {
                k: compute_slater_rk_table(_M(k), CP_r, CQ_r, CP_s, CQ_s)
                for k in k_d_list
                if abs(Ck_ar[k]) >= 1e-15 and abs(Ck_bs[k]) >= 1e-15
            }
            Re_tab = {
                k: compute_slater_rk_table(_M(k), CP_s, CQ_s, CP_r, CQ_r).T
                for k in k_e_list
                if abs(Ck_as[k]) >= 1e-15 and abs(Ck_br[k]) >= 1e-15
            }
            for orb_r_idx, (_Pr, _Qr, eps_r) in enumerate(virt_r):
                for orb_s_idx, (_Ps, _Qs, eps_s) in enumerate(virt_s):
                    # Avoid double counting r↔s when κ_r == κ_s
                    if kappa_r == kappa_s and orb_r_idx > orb_s_idx:
                        continue
                    same_rs = (kappa_r == kappa_s and orb_r_idx == orb_s_idx)
                    norm_rs = 2.0 if same_rs else 1.0

                    delta_eps = eps_r + eps_s - eps_a - eps_b
                    if abs(delta_eps) < 1e-10:
                        continue

                    # Direct Slater integrals R^k_d(abrs)
                    Rd: Dict[int, float] = {
                        k: tab[orb_r_idx, orb_s_idx]
                        for k, tab in Rd_tab.items()
                    }
                    # Exchange Slater integrals R^k_e(absr)
                    Re: Dict[int, float] = {
                        k: tab[orb_r_idx, orb_s_idx]
                        for k, tab in Re_tab.items()
                    }

                    if not Rd and not Re:
                        continue

                    # Sum over total J (twice-J integers)
                    J_sum = 0.0
                    for tJ in range(tJ_lo, tJ_hi + 1, 2):
                        # V_J(abrs) = Σ_k (-1)^{j_b+j_r+J} {j_a j_b J; j_s j_r k} X^k(abrs)
                        VJ_d = 0.0
                        for k, Rk in Rd.items():
                            sj = _wigner_6j(tj_a, tj_b, tJ,
                                           tj_s, tj_r, 2 * k)
                            if abs(sj) < 1e-15:
                                continue
                            X_k = Ck_ar[k] * Ck_bs[k] * Rk
                            VJ_d += sj * X_k
                        # Phase (-1)^{j_b+j_r+J}
                        ph_d = (-1) ** ((tj_b + tj_r + tJ) // 2)
                        VJ_d *= ph_d

                        # V_J(absr) = Σ_k (-1)^{j_b+j_s+J} {j_a j_b J; j_r j_s k} X^k(absr)
                        VJ_e = 0.0
                        for k, Rk in Re.items():
                            sj = _wigner_6j(tj_a, tj_b, tJ,
                                           tj_r, tj_s, 2 * k)
                            if abs(sj) < 1e-15:
                                continue
                            X_k = Ck_as[k] * Ck_br[k] * Rk
                            VJ_e += sj * X_k
                        ph_e = (-1) ** ((tj_b + tj_s + tJ) // 2)
                        VJ_e *= ph_e

                        # Antisymmetrised pair amplitude — extra
                        # phase (-1)^{j_r+j_s-J} on the exchange to
                        # keep the |VJ_d − (−1)^…·VJ_e|^2 form
                        # consistent with the antisymmetric two-
                        # particle ket.
                        ph_swap = (-1) ** ((tj_r + tj_s - tJ) // 2)
                        amp = VJ_d - ph_swap * VJ_e
                        J_sum += (tJ + 1) * amp * amp

                    pair_E += -J_sum / (norm_ab * norm_rs * delta_eps)

    return pair_E


# ---------------------------------------------------------------------------
# All-order pp-ladder resummation (first all-order rung beyond MBPT(2))
# ---------------------------------------------------------------------------


def _pair_v_J(Pa, Qa, kappa_a, Pb, Qb, kappa_b,
              Pr, Qr, kappa_r, Ps, Qs, kappa_s,
              tJ, bsp_data, cache, multipole_max=None,
              rk_bilinear=None):
    """Antisymmetrised J-coupled pair matrix element <ab;J|v|rs;J>_A.

    Exactly the V_J combination used by :func:`_mbpt2_pair_contribution`
    (direct - (-1)^{j_r+j_s-J} exchange with the standard 6j recoupling),
    factored out so the SAME kernel serves (occ,occ)->(virt,virt) and
    (virt,virt)->(virt,virt) ladder blocks.  States are the UNNORMALISED
    coupled products; callers divide by sqrt((1+delta_ab)(1+delta_rs)).

    ``rk_bilinear`` — optional {k: M} dict OWNED BY THE CALLER and valid
    only for this (Pa,Qa,Pb,Qb): lazily filled with
    :func:`build_slater_rk_bilinear` matrices and reused across every
    (r, s) sharing the bra pair (direct and exchange use the same M).
    One M build costs ≈ one scalar R^k; each reuse is O(ns²).
    """

    def _rk(k, P2, Q2, P4, Q4):
        # R^k(a ·, b ·) with slot-2/slot-4 orbitals (·) varying
        if rk_bilinear is None:
            return compute_slater_rk_cell(Pa, Qa, P2, Q2, Pb, Qb, P4, Q4,
                                          bsp_data, k, cache=cache)
        Mk = rk_bilinear.get(k)
        if Mk is None:
            Mk = build_slater_rk_bilinear(Pa, Qa, Pb, Qb, bsp_data, k,
                                          cache=cache)
            rk_bilinear[k] = Mk
        return eval_slater_rk_bilinear(Mk, P2, Q2, P4, Q4)

    tj_a = _twoj_from_kappa(kappa_a)
    tj_b = _twoj_from_kappa(kappa_b)
    tj_r = _twoj_from_kappa(kappa_r)
    tj_s = _twoj_from_kappa(kappa_s)
    if tJ < max(abs(tj_a - tj_b), abs(tj_r - tj_s)):
        return 0.0
    if tJ > min(tj_a + tj_b, tj_r + tj_s):
        return 0.0
    if (_l_from_kappa(kappa_a) + _l_from_kappa(kappa_b)
            + _l_from_kappa(kappa_r) + _l_from_kappa(kappa_s)) % 2:
        return 0.0

    k_d = [k for k in _allowed_k_range(kappa_a, kappa_r)
           if k in _allowed_k_range(kappa_b, kappa_s)]
    k_e = [k for k in _allowed_k_range(kappa_a, kappa_s)
           if k in _allowed_k_range(kappa_b, kappa_r)]
    if multipole_max is not None:
        k_d = [k for k in k_d if k <= multipole_max]
        k_e = [k for k in k_e if k <= multipole_max]

    VJ_d = 0.0
    for k in k_d:
        c1 = reduced_C(kappa_a, kappa_r, k)
        c2 = reduced_C(kappa_b, kappa_s, k)
        if abs(c1) < 1e-15 or abs(c2) < 1e-15:
            continue
        sj = _wigner_6j(tj_a, tj_b, tJ, tj_s, tj_r, 2 * k)
        if abs(sj) < 1e-15:
            continue
        Rk = _rk(k, Pr, Qr, Ps, Qs)
        VJ_d += sj * c1 * c2 * Rk
    VJ_d *= (-1) ** ((tj_b + tj_r + tJ) // 2)

    VJ_e = 0.0
    for k in k_e:
        c1 = reduced_C(kappa_a, kappa_s, k)
        c2 = reduced_C(kappa_b, kappa_r, k)
        if abs(c1) < 1e-15 or abs(c2) < 1e-15:
            continue
        sj = _wigner_6j(tj_a, tj_b, tJ, tj_r, tj_s, 2 * k)
        if abs(sj) < 1e-15:
            continue
        Rk = _rk(k, Ps, Qs, Pr, Qr)
        VJ_e += sj * c1 * c2 * Rk
    VJ_e *= (-1) ** ((tj_b + tj_s + tJ) // 2)

    ph_swap = (-1) ** ((tj_r + tj_s - tJ) // 2)
    return VJ_d - ph_swap * VJ_e


def pair_ladder_energy(
    scf_result,
    *,
    kappa_max: int = 3,
    n_virt_max: Optional[int] = 8,
    multipole_max: Optional[int] = None,
    e_virt_max: Optional[float] = None,
    include_ladder: bool = True,
) -> "MBPT2Result":
    """All-order particle-particle-ladder pair correlation energy.

    For each occupied pair (a, b) and total J, the doubles space
    {|rs;J>_A} carries the pair equation

        (D + V_pp) t = -v0,      E_pair^J = (2J+1) v0^T t,

    with D the diagonal denominators eps_r + eps_s - eps_a - eps_b,
    v0[rs] = <rs;J|v|ab;J>_A and V_pp[rs,r's'] = <rs;J|v|r's';J>_A on
    NORMALISED antisymmetrised states.  Solving the linear system sums
    the pp-ladder diagram class TO ALL ORDERS exactly:

        E = -v0^T D^-1 v0  +  v0^T D^-1 V D^-1 v0  -  ...

    the first term being precisely MBPT(2) (``include_ladder=False``
    reproduces :func:`mbpt2_energy` to machine precision — tested), the
    second the third-order pp-ladder, etc.  Hole-hole and ring (core
    polarisation) classes remain at second order; the full coupled-pair
    (CCD) extension iterates the same objects with the ring recoupling
    and is the documented next rung.

    Cost scales as (n_virt_pairs)^2 per (pair, J); the defaults
    (kappa_max 3, n_virt_max 8) keep He-like systems interactive.
    """
    import time
    t0 = time.time()
    if not scf_result.converged:
        raise ValueError("SCFResult must be converged.")
    for sh in scf_result.subshells:
        max_occ = _twoj_from_kappa(sh.kappa) + 1
        if abs(sh.occupation - max_occ) > 0.01:
            raise ValueError("pair_ladder_energy requires a closed shell.")

    bsp_data = scf_result._bsp_data
    cache = ConvolCache(bsp_data)

    kappa_list = []
    for k_abs in range(1, kappa_max + 2):
        kappa_list.append(-k_abs)
        if k_abs >= 1:
            kappa_list.append(+k_abs)
    kappa_list = [k for k in kappa_list if _l_from_kappa(k) <= kappa_max]

    occ_basis: Dict[int, List[Tuple[np.ndarray, np.ndarray, float, str]]] = {}
    virt_basis: Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]] = {}
    for kappa in kappa_list:
        occ, virt = build_virtual_basis(scf_result, kappa, cache=cache,
                                         e_max=e_virt_max)
        kappa_subshells = [x for x in scf_result.subshells if x.kappa == kappa]
        occ_l = []
        for i, (P, Q, _e) in enumerate(occ):
            if i < len(kappa_subshells):
                lab = kappa_subshells[i].label
                eps = scf_result.orbital_energies.get(lab, _e)
            else:
                lab, eps = f"occ-{kappa}-{i}", _e
            occ_l.append((P, Q, float(eps), lab))
        if n_virt_max is not None:
            virt = virt[:n_virt_max]
        occ_basis[kappa] = occ_l
        virt_basis[kappa] = virt

    occ_kappas = [k for k in kappa_list if occ_basis[k]]
    E_corr = 0.0
    pair_contribs: Dict[Tuple[str, str], float] = {}

    for ka_idx, kappa_a in enumerate(occ_kappas):
        for kappa_b in occ_kappas[ka_idx:]:
            for ia, (Pa, Qa, eps_a, lab_a) in enumerate(occ_basis[kappa_a]):
                itb = (enumerate(occ_basis[kappa_b][ia:], start=ia)
                       if kappa_a == kappa_b
                       else enumerate(occ_basis[kappa_b]))
                for ib, (Pb, Qb, eps_b, lab_b) in itb:
                    same_ab = (kappa_a == kappa_b and ia == ib)
                    tj_a = _twoj_from_kappa(kappa_a)
                    tj_b = _twoj_from_kappa(kappa_b)
                    # {k: M} bilinear cache for the fixed bra pair (a, b);
                    # valid across all J blocks and (r, s) kets
                    bl_ab: Dict[int, tuple] = {}
                    pair_E = 0.0
                    for tJ in range(abs(tj_a - tj_b), tj_a + tj_b + 1, 2):
                        # enumerate the doubles basis (kr,ir) <= (ks,is)
                        basis = []
                        for kr in kappa_list:
                            for ks in kappa_list:
                                if (kr, ks) > (ks, kr):
                                    continue
                                tj_r = _twoj_from_kappa(kr)
                                tj_s = _twoj_from_kappa(ks)
                                if tJ < abs(tj_r - tj_s) or tJ > tj_r + tj_s:
                                    continue
                                if (_l_from_kappa(kappa_a)
                                        + _l_from_kappa(kappa_b)
                                        + _l_from_kappa(kr)
                                        + _l_from_kappa(ks)) % 2:
                                    continue
                                for ir, vr in enumerate(virt_basis[kr]):
                                    js0 = ir if kr == ks else 0
                                    for isx in range(js0,
                                                     len(virt_basis[ks])):
                                        vs = virt_basis[ks][isx]
                                        same_rs = (kr == ks and ir == isx)
                                        basis.append((kr, ir, ks, isx,
                                                      vr, vs, same_rs))
                        if not basis:
                            continue
                        n = len(basis)
                        v0 = np.zeros(n)
                        D = np.zeros(n)
                        na = math.sqrt(2.0) if same_ab else 1.0
                        keep = []
                        for ii, (kr, _ir, ks, _isx, vr, vs, same_rs) in enumerate(basis):
                            nr = math.sqrt(2.0) if same_rs else 1.0
                            amp = _pair_v_J(
                                Pa, Qa, kappa_a, Pb, Qb, kappa_b,
                                vr[0], vr[1], kr, vs[0], vs[1], ks,
                                tJ, bsp_data, cache, multipole_max,
                                rk_bilinear=bl_ab)
                            v0[ii] = amp / (na * nr)
                            D[ii] = vr[2] + vs[2] - eps_a - eps_b
                            # drop identically-empty rows (Pauli-forbidden
                            # same-orbital odd-J states have amp == 0 and
                            # zero ladder coupling; keep the space tight)
                            if abs(v0[ii]) > 1e-18 or not same_rs:
                                keep.append(ii)
                        keep = np.asarray(keep, dtype=int)
                        v0 = v0[keep]
                        D = D[keep]
                        basis_k = [basis[i] for i in keep]
                        n = len(basis_k)
                        if n == 0:
                            continue
                        if float(np.max(np.abs(v0))) < 1e-16:
                            continue  # Pauli/parity-empty block
                        if not include_ladder:
                            eJ = -float(np.sum(v0 * v0 / D))
                        else:
                            V = np.zeros((n, n))
                            for ii in range(n):
                                (kr, ir, ks, isx, vr, vs, same_rs) = basis_k[ii]
                                ni = math.sqrt(2.0) if same_rs else 1.0
                                # bilinear cache for the row bra (r, s);
                                # discarded per row to bound memory
                                bl_row: Dict[int, tuple] = {}
                                for jj in range(ii, n):
                                    (kc, ic, kd, idx, vc, vd, same_cd) = basis_k[jj]
                                    nj = math.sqrt(2.0) if same_cd else 1.0
                                    vel = _pair_v_J(
                                        vr[0], vr[1], kr, vs[0], vs[1], ks,
                                        vc[0], vc[1], kc, vd[0], vd[1], kd,
                                        tJ, bsp_data, cache, multipole_max,
                                        rk_bilinear=bl_row)
                                    V[ii, jj] = vel / (ni * nj)
                                    V[jj, ii] = V[ii, jj]
                            t = np.linalg.solve(np.diag(D) + V, -v0)
                            eJ = float(v0 @ t)
                        pair_E += (tJ + 1) * eJ
                    E_corr += pair_E
                    pair_contribs[(lab_a, lab_b)] = pair_E

    return MBPT2Result(
        correlation_energy=E_corr,
        pair_contributions=pair_contribs,
        n_virtuals={k: len(v) for k, v in virt_basis.items()},
        kappa_max=kappa_max,
        n_max=n_virt_max,
        cpu_seconds=time.time() - t0,
        provenance={
            "method": ("all-order pp-ladder" if include_ladder
                       else "MBPT(2) cross-check"),
            "multipole_max": multipole_max,
            "kappa_list": kappa_list,
        },
    )


__all__ = [
    "MBPT2Result",
    "build_virtual_basis",
    "mbpt2_energy",
    "pair_ladder_energy",
    "reduced_C",
]
