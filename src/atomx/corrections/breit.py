"""Legacy scalar Breit approximation retained for explicit comparison runs.

Historical audit below describes the pre-retardation CI implementation.
Current resolved CI includes complete static Gaunt plus retardation; the
DHF default is the independent ``dbsr_breit.py`` implementation. Neither
uses this legacy approximation as its default.

The Breit operator in the multipole expansion has contributions at
three ranks v = L-1, L, L+1 for each multipole order L, with 8
angular components S(1)..S(8) corresponding to different pairings
of large (P) and small (Q) components.

The four types of Breit Slater integrals:
  S_ppqq^v(a,b) = ∫∫ P_a(r1)P_b(r1) [r_<^v/r_>^{v+1}] Q_a(r2)Q_b(r2) dr
  S_pqqp^v(a,b) = ∫∫ P_a(r1)Q_b(r1) [r_<^v/r_>^{v+1}] Q_a(r2)P_b(r2) dr
  (and symmetric partners obtained by swapping P↔Q)

Reference: Grant & Pyper, J.Phys.B 9, 761 (1976); DBSR_HF hf_energy_BR.f90.

What this scalar is for, and what it is not
-------------------------------------------
This is the **closed-shell / single-CSF total**.  It is *complementary*
to the exact Gaunt operator of :mod:`atomx.physics.csf.breit_ci`, not
redundant with it:

* the operator carries the non-scalar (spin-other-orbit, spin-spin,
  orbit-orbit) content, which is what splits levels inside a
  ``(2J, parity)`` block — this scalar cannot do that by construction;
* this scalar carries the ``v != L`` retardation channels, which the
  operator omits entirely.  Its residual against DBSR_HF is +0.13 %
  (He), -0.35 % (Ne), +0.63 % (Ar) *after* the fix below, against the
  operator's +0.000005 %, +4.15 %, +6.2 % — i.e. the scalar's total is
  the better one for closed shells, the operator's is the better one
  for anything that needs level splitting.

The two share no angular kernel (DBSR ``SMU`` transcription + ``S^v``
radial integrals here, 3j-based Racah algebra + Gaunt photon multipoles
there), so their agreement against DBSR_HF is genuine independent
evidence.

The ``(2L+1)`` fix (2026-09)
----------------------------
The ``v = L`` channel used to be ~3x too large — 3.004x / 3.030x /
3.098x DBSR_HF for He / Ne / Ar.  Cause: the ``(2L+1)`` multipole
normalisation that belongs with the Racah-normalised
``C^L = sqrt(4*pi/(2L+1)) Y_L`` used by the ``S^v`` radial integrals,

    P_L(cos θ12) = (4*pi/(2L+1)) Σ_q Y_L^q(1) Y_L^{q*}(2),

while DBSR's ``SMU`` is written for the unnormalised ``Y_L`` contraction
(``SMU = -(ka+kc)*(kb+kd)/L/(L+1)``, hf_energy_BR.f90:82).  Dividing the
``v = L`` coefficients by ``(2L+1)`` gives 1.001 / 0.996 / 1.006 x
DBSR_HF.  It is not a fudge factor: the 1/(2L+1) is exactly the
denominator of the addition theorem above, and He 1s2 proves it does
real work — the 1s2 retardation is nil (the Gaunt-only operator already
reproduces DBSR_HF to 5e-9 relative there), so there was nothing for a
wrong factor to cancel.

Two further differences from DBSR remain and are *not* fixed here (they
are ~1 % effects on closed shells and are reported, not hidden): the
``v != L`` channels are ~50x too small (Ne 4.25e-05 where retardation
must supply ~6.96e-04), and ``L`` starts at 0 in ``hf_energy_BR.f90``
(line 28) and at 1 here.  ``coef(i,j,k)`` (dbsr_lib_zconfjj.f90:4098) is
replaced by the ``oa*ob/((2j_a+1)(2j_b+1))`` weight.  The residual signs
above bracket DBSR_HF, which is the signature of these two separate
small errors.

Acceptance (DBSR_HF, ``dbsr_hf an=Z term=AV mbreit=1``):
    He 1s2           6.377743e-05      <- measured 6.3857e-05  (+0.13 %)
    Ne 1s2 2s2 2p6   0.016643630840    <- measured 0.016585      (-0.35 %)
    Ar 1s2..3p6      0.132364619357    <- measured 0.13320       (+0.63 %)

Radial quadrature (2026-07 fix)
-------------------------------
The Y^v assembly uses the canonical arbitrary-grid kernel
:func:`atomx.numerics.quadrature.yk_on_grid` (trapezoidal cumulatives,
second order on any grid) on the flattened radial nodes of whichever
workspace the orbitals live on (B-spline Gauss nodes or a shooting
grid).  The previous implementation used the B-spline SCF helper
``compute_yk``, whose rectangle-rule cumulatives double-count the
diagonal cell (r' = r) — the exact O(h) bias class documented in
``numerics/quadrature.py`` from the Auger verification campaign.
(``compute_yk`` itself was later given an interpolatory partial-interval
rule on 2026-09-23, ``atomx.numerics.yk.yk_partial_matrices``; this
module's choice of ``yk_on_grid`` is unchanged, and the trapezoid bias
class below still applies to it.)  For
the strongly cancelling Breit combinations that bias inflated the total
by ~3%: Ne I (point nucleus) gave 0.05197 Ha where the dense-grid
converged value is 0.050429 Ha (cubic-spline resampling of the same
DHF orbitals onto 5k/20k/80k-point grids, all agreeing to 1e-6, and
independently reproduced at 0.05044 Ha by the ODE-shooting engine).
"""

import numpy as np

from atomx.numerics.quadrature import yk_on_grid
from atomx.numerics.radial_workspace import (
    eval_orbital_pair,
    workspace_arrays,
)


def _breit_smu_coefficients(kappa_a, kappa_b, kappa_c, kappa_d, L):
    """Compute the 8 Breit angular coefficients S(1)..S(8) for multipole L.

    Returns dict with keys v: L-1, L, L+1 → list of 8 S-coefficients.
    Based on DBSR_HF SMU function (hf_energy_BR.f90 lines 79-115).
    """
    K = kappa_c - kappa_a
    Kp = kappa_d - kappa_b
    results = {}

    # v = L: diagonal term.
    #
    # The 1/(2L+1) is the denominator of the addition theorem for the
    # Racah-normalised spherical harmonics the S^v radial integrals use,
    #     P_L(cos t12) = (4 pi/(2L+1)) sum_q Y_L^q(1) Y_L^q*(2),
    # while hf_energy_BR.f90's SMU (line 82) is written for the
    # unnormalised Y_L contraction.  Without it the whole v = L (Gaunt)
    # family came out 3.004x / 3.030x / 3.098x DBSR_HF for He / Ne / Ar;
    # with it, 1.001 / 0.996 / 1.006.  See the module docstring.
    if L > 0:
        smu = -(kappa_a + kappa_c) * (kappa_b + kappa_d) / (L * (L + 1))
        smu = smu / (2 * L + 1)
        results[L] = [smu] * 8  # all 8 components equal

    # v = L-1
    if L >= 2:
        bp = (L + 1) / (2 * (2 * L - 1))
        cp = -(L - 2) / (2 * L * (2 * L - 1))
        s = [0.0] * 8
        s[0] = (L + K) * (bp + cp * Kp)
        s[1] = (L + Kp) * (bp + cp * K)
        s[2] = (L - K) * (bp - cp * Kp)
        s[3] = (L - Kp) * (bp - cp * K)
        s[4] = -(L + K) * (bp - cp * Kp)
        s[5] = -(L - Kp) * (bp + cp * K)
        s[6] = -(L - K) * (bp + cp * Kp)
        s[7] = -(L + Kp) * (bp - cp * K)
        results[L - 1] = s

    # v = L+1
    b = L / (2 * (2 * L + 3))
    c = (L + 3) / ((2 * L + 2) * (2 * L + 3))
    il = -L - 1
    s = [0.0] * 8
    s[0] = (il + Kp) * (b + c * K)
    s[1] = (il + K) * (b + c * Kp)
    s[2] = (il - Kp) * (b - c * K)
    s[3] = (il - K) * (b - c * Kp)
    s[4] = -(il - Kp) * (b + c * K)
    s[5] = -(il + K) * (b - c * Kp)
    s[6] = -(il + Kp) * (b - c * K)
    s[7] = -(il - K) * (b + c * Kp)
    results[L + 1] = s

    return results


def compute_breit_energy(orbitals, bsp_data, Z, *, jj_configs=None):
    """Compute full Breit interaction energy.

    Includes all 8 angular components at each multipole L,
    with contributions at v = L-1, L, L+1.

    The 8 integrals correspond to the four density pairings:
    S1..S4: "direct" type — S^v(P_a P_b, Q_a Q_b) etc.
    S5..S8: "exchange" type — S^v(P_a Q_b, Q_a P_b) etc.

    Parameters
    ----------
    orbitals : list of (P, Q, occ, kappa, label) tuples
    bsp_data : dict
        B-spline workspace (``P``/``Q`` are coefficient vectors), or a
        radial-grid workspace from
        :func:`atomx.numerics.radial_workspace.make_grid_workspace`
        (``P``/``Q`` are grid arrays) — e.g. the ODE-shooting engine.
    Z : int
    jj_configs : sequence of JJConfig, optional
        Explicit ensemble of integer jj occupations. When supplied, use
        its exact two-electron pair probabilities instead of products of
        mean occupations. This fixes the particle-number and jj-partner
        correlations of the scalar estimate; it does not make the scalar
        angular kernel a resolved Breit operator.
    """
    # Breit is a two-electron operator. The configuration-average angular
    # sum below otherwise gives a spurious self-pair for a singly occupied
    # orbital (and for fractional jj occupations of one nl electron).
    # This exact particle-number constraint holds on every radial grid.
    if sum(float(orbital[2]) for orbital in orbitals) <= 1.0 + 1e-10:
        return 0.0

    gr, grw = workspace_arrays(bsp_data)
    # Flatten to 1D grid order for the arbitrary-grid Y^v kernel
    # (B-spline workspaces carry (nv, n_gauss) arrays; the flattened
    # Gauss nodes are strictly increasing).
    # This evaluation owns a fixed mesh. An immutable copy lets the radial
    # kernel reuse its bounded geometry cache across densities and ranks,
    # without assuming that the caller's workspace cannot change.
    r_flat = np.array(gr, copy=True).reshape(-1)
    r_flat.setflags(write=False)
    w_flat = np.ravel(grw)

    # Evaluate every orbital on the radial nodes once.
    evaluated = []
    for (Pa, Qa, oa, ka, la) in orbitals:
        fP, fQ = eval_orbital_pair(Pa, Qa, bsp_data)
        evaluated.append((np.ravel(fP), np.ravel(fQ), oa, ka, la))

    pair_probability = None
    if jj_configs is not None:
        from atomx.config.jj_coupling import compute_WW

        configs = list(jj_configs)
        if not configs:
            raise ValueError("Breit jj configuration ensemble is empty")
        if len(configs[0].subshells) != len(evaluated):
            raise ValueError("Breit jj configurations do not match orbitals")
        if abs(sum(c.weight for c in configs) - 1.0) > 1e-10:
            raise ValueError("Breit jj configuration weights must sum to one")
        pair_probability = np.zeros((len(evaluated), len(evaluated)))
        for i, (_, _, _, kappa_i, label_i) in enumerate(evaluated):
            gi = 2 * abs(kappa_i)
            for j, (_, _, _, kappa_j, label_j) in enumerate(evaluated):
                gj = 2 * abs(kappa_j)
                if any(c.subshells[i].label != label_i or
                       c.subshells[j].label != label_j for c in configs):
                    raise ValueError("Breit jj subshell order does not match orbitals")
                denominator = gi * gj - (gi if i == j else 0)
                pair_probability[i, j] = compute_WW(i, j, configs) / denominator

    E_Breit = 0.0

    for _i, (fa_P, fa_Q, oa, ka, _la) in enumerate(evaluated):
        tj_a = 2 * abs(ka) - 1

        for _j, (fb_P, fb_Q, ob, kb, _lb) in enumerate(evaluated):
            tj_b = 2 * abs(kb) - 1

            # Weight for average-of-configuration
            w_ab = (pair_probability[_i, _j] if pair_probability is not None
                    else oa * ob / ((tj_a + 1.0) * (tj_b + 1.0)))
            if w_ab == 0:
                continue

            # The four cross densities are L/v-independent.
            rho_PP = fa_P * fb_P
            rho_QQ = fa_Q * fb_Q
            rho_PQ = fa_P * fb_Q
            rho_QP = fa_Q * fb_P

            # Loop over multipole L
            L_max = abs(ka) + abs(kb)
            for L in range(1, L_max + 1):
                # Check angular selection rule: 3j triangle for (j_a, L, j_b)
                if tj_a + tj_b < 2 * L or abs(tj_a - tj_b) > 2 * L:
                    continue

                # Get 8 S-coefficients for each v = L-1, L, L+1
                smu_dict = _breit_smu_coefficients(ka, kb, ka, kb, L)

                for v, s_coeffs in smu_dict.items():
                    if v < 0:
                        continue
                    if all(abs(s) < 1e-15 for s in s_coeffs):
                        continue

                    # Compute the 8 Breit integrals for this (a,b,v):
                    # Direct-type densities (indices 0-3):
                    #   PP density at r1, QQ density at r2 (and swaps)
                    # Exchange-type densities (indices 4-7):
                    #   PQ density at r1, QP density at r2 (and swaps)
                    yk_QQ = yk_on_grid(rho_QQ, r_flat, v)
                    yk_PP = yk_on_grid(rho_PP, r_flat, v)
                    yk_QP = yk_on_grid(rho_QP, r_flat, v)
                    yk_PQ = yk_on_grid(rho_PQ, r_flat, v)

                    # Direct integrals (S1-S4 pattern)
                    R_ppqq = np.sum(w_flat * rho_PP * yk_QQ)  # S1, S2
                    R_qqpp = np.sum(w_flat * rho_QQ * yk_PP)  # S3, S4
                    # Exchange integrals (S5-S8 pattern)
                    R_pqqp = np.sum(w_flat * rho_PQ * yk_QP)  # S5, S6
                    R_qppq = np.sum(w_flat * rho_QP * yk_PQ)  # S7, S8

                    # Combine with angular coefficients
                    contrib = (s_coeffs[0] * R_ppqq + s_coeffs[1] * R_ppqq
                             + s_coeffs[2] * R_qqpp + s_coeffs[3] * R_qqpp
                             + s_coeffs[4] * R_pqqp + s_coeffs[5] * R_pqqp
                             + s_coeffs[6] * R_qppq + s_coeffs[7] * R_qppq)

                    # Breit Slater integrals on the native Dirac spinor
                    # (P, Q) already carry the Q/c ~ Zα suppression: the
                    # PPQQ, PQQP, etc. combinations are intrinsically
                    # O(α²) relative to Coulomb.  The DBSR_HF reference
                    # (``hf_energy_BR.f90``) accumulates ``crk·sk_ppqq``
                    # without any further c² division.  atomx's original
                    # ``/ C_SQ`` factor therefore over-suppressed Breit
                    # by another α² (Z=26 He-like gave 3.8e-5 Ha instead
                    # of ~0.7 Ha).
                    E_Breit -= w_ab * contrib

    return 0.5 * E_Breit


#: Measured worth of the fixed scalar against DBSR_HF
#: (``dbsr_hf an=Z term=AV mbreit=1``).  Kept as a constant so callers can
#: report it instead of restating it.
SCALAR_BREIT_NOTE = (
    "Breit: corrections.breit.compute_breit_energy (scalar "
    "configuration-average Grant-Pyper multipole sum, v=L with the "
    "1/(2L+1) multipole normalisation).  DBSR_HF agreement: He "
    "6.3857e-05 vs 0.000063777432919 (+0.13 %), Ne 0.016585 vs "
    "0.016643630840490 (-0.35 %), Ar 0.13320 vs 0.132364619357042 "
    "(+0.63 %).  The residual is the v!=L retardation channels, which "
    "are still ~50x too small; the exact Gaunt operator "
    "(physics/csf/breit_ci.py) omits retardation entirely and sits "
    "4-6 % above DBSR_HF at Ne/Ar, so the two are complementary."
)


def scalar_breit_provenance() -> dict:
    """Provenance dict for a run whose applied Breit is this scalar."""
    return {
        "mode": "scalar",
        "operator": "none",
        "scalar_shift_applied": True,
        "externally_validated": False,
        "validation": SCALAR_BREIT_NOTE,
        "note": SCALAR_BREIT_NOTE,
    }
