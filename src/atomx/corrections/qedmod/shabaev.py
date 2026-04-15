"""Shabaev–Tupitsyn–Yerokhin model QED operator (local part).

Reference
---------
V. M. Shabaev, I. I. Tupitsyn, V. A. Yerokhin,
    "Model operator approach to the Lamb shift calculations in
    relativistic many-electron atoms",
    *Phys. Rev. A* **88**, 012513 (2013).
V. M. Shabaev, I. I. Tupitsyn, V. A. Yerokhin,
    "QEDMOD: Fortran program for calculating the model Lamb-shift
    operator", *Comput. Phys. Commun.* **189**, 175 (2015) — v7
    source at ``qedmod_v7/``.

Implementation source
---------------------
The local-operator form below is derived from QEDMOD v7
``f/potgen/se_pot.f``, subroutine ``local_se_pot``:

    vloc(i) = dexp(-r(i)*cl) * r(i)           ! line 189
    de      = sint(vloc, P, Q, P, Q, r, v)    ! overlap integral
    vloc0   = se_loc(il) / de                 ! amplitude A_κ

That is, the local potential kernel is ``r · exp(−r·c)`` (not just
``exp(−r·c)``), the overlap integral weighting is the Dirac density
``P² + Q²`` on an ``(r, dr)`` measure (no extra r factor), and the
amplitude is calibrated against the tabulated ``se_loc`` value for
the lowest-n reference of each κ.

Paper eqs. (13–14) define the structure; the explicit factor of r in
the kernel is a consequence of the Dirac-Coulomb reduction applied
inside QEDMOD.

Mohr-vs-Shabaev normalisation
-----------------------------
The Fortran ``FSE_dat`` (``f/potgen/fse_dat.f``) stores tabulated
``F_ik(Zα)`` values with the leading Bethe-logarithm term
``−(8/3)·ln(Z/c)`` **subtracted off** for smooth interpolation
(``fse_dat.f`` lines 33–50), then added back after interpolation.
This explains why Shabaev's raw tabulated ``F`` values differ from
Mohr's by a factor of ≈ 2.4 for s-states at Z=10; the Mohr
tabulation keeps the log *inside* ``F``.

Because Mohr's tabulation is what atomx already ships, we calibrate
``A_κ(Z)`` directly against it.  This is algebraically equivalent
to going through Shabaev's ``F_ik + log correction`` chain.

What this backend delivers
--------------------------
The ``"shabaev_local"`` backend implements the LOCAL piece of the
Shabaev model operator:

    V_{loc,κ}^{SE}(r) = A_κ(Z) · r · exp(−r·c)                       (1)

with ``c = 1/α = 137.036`` (speed of light in atomic units) and
``A_κ(Z)`` calibrated by

    ⟨ψ_{ref(κ)} | V_{loc,κ}^{SE} | ψ_{ref(κ)}⟩
                    =  E_SE^Mohr(n_ref, κ, Z)                       (2)

where ``ref(κ)`` is the lowest-n hydrogenic state with that κ
(1s₁/₂ for κ=−1, 2p₁/₂ for κ=+1, 2p₃/₂ for κ=−2) and
``E_SE^Mohr`` is the tabulated Mohr (1982/1992) self-energy.

For SCF orbital ``a`` the backend returns

    ⟨a | V_{loc,κ}^{SE} | a⟩
        = A_κ(Z) · ∫ [P_a²(r) + Q_a²(r)] · r · exp(−r·c) dr         (3)

Compared to the pure Mohr backend this is a systematic improvement
for any orbital that shares κ with the Mohr reference but differs
in principal quantum number or screening: the n⁻³ Mohr fallback is
replaced by the actual short-range density overlap on the SCF
wavefunction.

Deferred
--------
* Non-local projector term ``Σ |φ_i⟩ B_{ik} ⟨φ_k|`` — paper eq. (15)
  — needed to push agreement below 1 % against reference Mohr/
  Yerokhin data for n ≥ 3 states.  Port of
  ``f/potgen/se_pot.f:nonlocal_se_pot`` + ``fse_dat.f`` full tables
  is a multi-session follow-up.
* κ = ±2 reference integrals for hydrogenic 3d (not yet tabulated
  analytically here; κ = ±3 falls back to zero for now).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from atomx.bspline.evaluate import eval_orbital_on_grid
from atomx.constants import ALPHA, C
from atomx.corrections.qedmod.base import SEBackend, SEResult, register_backend
from atomx.corrections.qedmod.mohr import _interpolate_se_f


# ---------------------------------------------------------------------------
# Non-relativistic hydrogenic ⟨r · exp(-r·c)⟩ analytic formulas
# ---------------------------------------------------------------------------

def _hydrogenic_r_exp_integral(n: int, l: int, Z: int) -> float:
    """Analytic non-rel ⟨r · exp(-r·c)⟩ for hydrogenic state (n, l) at Z.

    Computes  I(n, l, Z) = ∫₀^∞ P_{n,l}²(r) · r · exp(−c·r) dr
    with ``c = 1/α``.

    Covers the lowest-n reference for every Mohr-tabulated κ:
    1s, 2s, 2p.  Relativistic (P, Q) corrections are O((Zα)²) and
    are absorbed into the semi-empirical calibration against Mohr.
    """
    c = 1.0 / ALPHA

    if n == 1 and l == 0:
        # P_{1s}(r) = 2 Z^{3/2} r exp(-Zr)
        # ⟨r · exp(-cr)⟩ = ∫ 4 Z³ r³ exp(-(2Z+c)r) dr
        #                = 4 Z³ · 6 / (2Z+c)⁴ = 24 Z³ / (2Z+c)⁴
        denom = 2.0 * Z + c
        return 24.0 * Z ** 3 / denom ** 4

    if n == 2 and l == 0:
        # P_{2s}² = (Z³/2) r² (1 - Zr/2)² exp(-Zr)
        # ⟨r · exp(-cr)⟩ = (Z³/2) ∫ r³ (1 - Zr + Z²r²/4) exp(-β r) dr
        # with β = Z + c.
        # ∫r³·e^(-βr)dr = 6/β⁴
        # ∫r⁴·e^(-βr)dr = 24/β⁵
        # ∫r⁵·e^(-βr)dr = 120/β⁶
        beta = Z + c
        integrand = (6.0 / beta ** 4
                     - 24.0 * Z / beta ** 5
                     + 30.0 * Z ** 2 / beta ** 6)
        return 0.5 * Z ** 3 * integrand

    if n == 2 and l == 1:
        # P_{2p}²(r) = (Z⁵/24) r⁴ exp(-Zr)
        # ⟨r · exp(-cr)⟩ = (Z⁵/24) · ∫ r⁵ · exp(-β r) dr
        #                = (Z⁵/24) · 120/β⁶
        #                = 5 Z⁵ / β⁶
        beta = Z + c
        return 5.0 * Z ** 5 / beta ** 6

    if n == 3 and l == 2:
        # P_{3d}(r)  = (4/(81√30)) Z^{7/2} r³ exp(-Zr/3)
        # P_{3d}²    = (16/(81² · 30)) Z⁷ r⁶ exp(-2Zr/3)
        #            = (16/196830) Z⁷ r⁶ exp(-2Zr/3)
        # ⟨r · exp(-cr)⟩ = (16/196830) Z⁷ · ∫ r⁷ exp(-β r) dr   with β = 2Z/3 + c
        #                = (16/196830) Z⁷ · 7!/β⁸
        #                = (16 × 5040 / 196830) · Z⁷/β⁸
        #                = (80640 / 196830) · Z⁷/β⁸
        beta = (2.0 * Z) / 3.0 + c
        return (80640.0 / 196830.0) * Z ** 7 / beta ** 8

    if n == 3 and l == 1:
        # P_{3p}(r) = (4/(81√6)) Z^{3/2} (6 − Zr) (Zr) exp(-Zr/3)
        # P_{3p}²   = (16/(81²·6)) Z³ (6 − Zr)² (Zr)² exp(-2Zr/3)
        #           = (16/39366) Z⁵ r² (6 − Zr)² exp(-2Zr/3)
        # ⟨r · exp(-cr)⟩ = (16/39366) Z⁵ ∫ r³ (6 − Zr)² exp(-β r) dr
        #   with β = 2Z/3 + c,
        #   (6−Zr)² = 36 − 12Zr + Z²r²
        #   ∫r³ exp(-βr) dr = 6/β⁴
        #   ∫r⁴ exp(-βr) dr = 24/β⁵
        #   ∫r⁵ exp(-βr) dr = 120/β⁶
        beta = (2.0 * Z) / 3.0 + c
        moment = (36.0 * 6.0 / beta ** 4
                  - 12.0 * Z * 24.0 / beta ** 5
                  + Z ** 2 * 120.0 / beta ** 6)
        return (16.0 / 39366.0) * Z ** 5 * moment

    if n == 3 and l == 0:
        # P_{3s}(r) = (2/(81√3)) Z^{3/2} (27 − 18 Zr + 2 (Zr)²) exp(-Zr/3)
        # P_{3s}²   = (4/(81²·3)) Z³ (27 − 18Zr + 2 Z²r²)² exp(-2Zr/3)
        #           = (4/19683) Z³ (…)² exp(-2Zr/3)
        # Expand (27 − 18Zr + 2 Z²r²)² = 729 − 972 Zr + 432 (Zr)²
        #                                − 72 (Zr)³ + 4 (Zr)⁴.
        # ⟨r · exp(-cr)⟩ = (4/19683) Z³ ∫ r [729 − 972 Zr + …] · r² · exp(-β r) dr
        #   = (4/19683) Z³ ∫ [729 r³ − 972 Zr⁴ + 432 Z²r⁵
        #                       − 72 Z³ r⁶ + 4 Z⁴ r⁷] exp(-βr) dr
        beta = (2.0 * Z) / 3.0 + c
        moment = (
            729.0 * 6.0 / beta ** 4
            - 972.0 * Z * 24.0 / beta ** 5
            + 432.0 * Z ** 2 * 120.0 / beta ** 6
            - 72.0 * Z ** 3 * 720.0 / beta ** 7
            + 4.0 * Z ** 4 * 5040.0 / beta ** 8
        )
        return (4.0 / 19683.0) * Z ** 3 * moment

    # Higher n not implemented — caller falls back to 0
    return 0.0


def _reference_state_for_kappa(kappa: int) -> Tuple[int, int] | None:
    """Lowest-n hydrogenic reference (n, l) for κ.

    Returns ``None`` for κ without a Mohr reference + analytic integral.
    """
    if kappa == -1:
        return (1, 0)   # 1s₁/₂
    if kappa == +1:
        return (2, 1)   # 2p₁/₂ (non-rel shared with 2p)
    if kappa == -2:
        return (2, 1)   # 2p₃/₂ (non-rel shared with 2p)
    if kappa == +2:
        return (3, 2)   # 3d₃/₂ (non-rel shared with 3d)
    if kappa == -3:
        return (3, 2)   # 3d₅/₂ (non-rel shared with 3d)
    return None


@lru_cache(maxsize=128)
def _hydrogenic_r_exp_integral_dirac_cached(n: int, kappa: int, Z: int) -> float:
    """Cache the Dirac ab-initio reference integral per (n, κ, Z).

    Computes ``∫ [P² + Q²] · r · exp(−r·c) dr`` using a hydrogenic
    Dirac solve on the B-spline basis (first-principles, includes all
    (Zα)² relativistic corrections to the reference density).
    """
    from atomx.dirac.hydrogenic_wf import hydrogenic_r_exp_integral_dirac
    try:
        return hydrogenic_r_exp_integral_dirac(n, kappa, Z)
    except Exception:
        # Numerical solve failed (e.g. Z too high for bound state) —
        # fall back to non-rel analytic.
        l = kappa if kappa > 0 else -kappa - 1
        return _hydrogenic_r_exp_integral(n, l, Z)


def _amplitude_A_kappa(
    Z: int, kappa: int,
    *, use_dirac_reference: bool = True,
) -> float:
    """Calibrated Shabaev amplitude ``A_κ(Z)`` per eq. (2).

    Implements the Fortran logic of QEDMOD ``local_se_pot``:

        vloc(r) = r · exp(−r·c)
        de      = ∫ [P_ref² + Q_ref²] · vloc(r) dr
        A_κ     = E_SE^Mohr(ref) / de

    By default uses the **Dirac ab-initio** reference density, which
    includes all (Zα)² corrections relative to the non-rel analytic
    form.  For heavy atoms this can differ from the non-rel value by
    ~10 % (1s at Z=92) to ~100 % (2p₁/₂ at Z=92).  Set
    ``use_dirac_reference=False`` to recover the faster non-rel form.

    Returns 0.0 for κ without a reference.
    """
    ref = _reference_state_for_kappa(kappa)
    if ref is None:
        return 0.0
    n_ref, l_ref = ref

    # Mohr SE of the reference, per electron, in Hartree.
    # E_SE = (α/π)(Zα)⁴/n³ · F · mc² = α³ Z⁴ F / (π n³)   (atomic units)
    F = _interpolate_se_f(Z, n_ref, kappa)
    E_ref = (ALPHA ** 3) * (Z ** 4) * F / (np.pi * n_ref ** 3)

    if use_dirac_reference:
        I_ref = _hydrogenic_r_exp_integral_dirac_cached(n_ref, kappa, Z)
    else:
        I_ref = _hydrogenic_r_exp_integral(n_ref, l_ref, Z)

    if I_ref <= 0:
        return 0.0
    return E_ref / I_ref


# ---------------------------------------------------------------------------
# Shabaev-local backend (working)
# ---------------------------------------------------------------------------

@register_backend
class ShabaevLocalBackend(SEBackend):
    """Local part of the Shabaev-Tupitsyn-Yerokhin model operator.

    Handles orbitals with κ ∈ {−1, +1, −2} (s₁/₂, p₁/₂, p₃/₂).
    Higher-|κ| orbitals return zero with a quiet skip; callers
    wanting d-orbital SE should use ``"mohr"`` until the non-local
    term is ported.
    """

    name = "shabaev_local"
    description = (
        "Shabaev-Tupitsyn-Yerokhin (PRA 88, 012513, 2013) LOCAL model "
        "operator V_{loc,κ}(r) = A_κ · r · exp(-r/λ_C) with A_κ "
        "Mohr-calibrated (via QEDMOD v7 se_pot.f logic).  "
        "Non-local projector term deferred."
    )

    def compute(self, orbitals, bsp_data, Z) -> SEResult:
        if bsp_data is None:
            raise ValueError(
                "shabaev_local backend requires bsp_data to evaluate "
                "⟨a|r·exp(-r/λ_C)|a⟩ on the radial grid."
            )

        gr = bsp_data["gr"]
        grw = bsp_data["grw"]
        nv = bsp_data["nv"]
        ksp = bsp_data.get("ksp", bsp_data["ks"])
        ksq = bsp_data.get("ksq", bsp_data["ks"])
        nsp = bsp_data.get("nsp", bsp_data["ns"])
        nsq = bsp_data.get("nsq", bsp_data["ns"])
        bsp_p = bsp_data["bsp_p"]
        bsp_q = bsp_data.get("bsp_q", bsp_p)

        # V_loc kernel:  r · exp(−r·c),  c = 1/α
        r_exp_kernel = gr * np.exp(-C * gr)

        A_cache: Dict[int, float] = {}
        per_orbital: Dict[str, float] = {}
        E_SE = 0.0

        for P, Q, occ, kappa, label in orbitals:
            if np.max(np.abs(P)) < 1e-12:
                per_orbital[label] = 0.0
                continue

            if kappa not in A_cache:
                A_cache[kappa] = _amplitude_A_kappa(Z, kappa)
            A_kap = A_cache[kappa]

            if A_kap == 0.0:
                per_orbital[label] = 0.0
                continue

            fP = eval_orbital_on_grid(P, bsp_p, nv, ksp, nsp)
            fQ = eval_orbital_on_grid(Q, bsp_q, nv, ksq, nsq)
            rho = fP * fP + fQ * fQ

            I_a = float(np.sum(grw * rho * r_exp_kernel))
            E_a = A_kap * I_a
            per_orbital[label] = E_a
            E_SE += occ * E_a

        return SEResult(
            E_SE=E_SE, per_orbital=per_orbital, method=self.name,
        )


# ---------------------------------------------------------------------------
# Stub for the full (local + non-local) Shabaev operator
# ---------------------------------------------------------------------------

@register_backend
class ShabaevBackend(SEBackend):
    """Full Shabaev-Tupitsyn-Yerokhin model operator (local + non-local).

    *Not yet implemented* — reserves the ``"shabaev"`` name and documents
    the target paper.  Use ``"shabaev_local"`` for the working local-
    only approximation, or ``"mohr"`` for the default tabulated backend.
    """

    name = "shabaev"
    description = (
        "Shabaev-Tupitsyn-Yerokhin model QED operator (PRA 88, 012513, "
        "2013; QEDMOD CPC 189, 175, 2015) — full non-local form not "
        "yet implemented (use 'shabaev_local' for the local-only "
        "approximation)."
    )

    def compute(self, orbitals, bsp_data, Z) -> SEResult:
        raise NotImplementedError(
            "Full non-local Shabaev operator is not yet implemented. "
            "The non-local projector term requires hydrogenic Dirac-"
            "Coulomb Green's functions on the B-spline basis and a "
            "port of  f/potgen/se_pot.f:nonlocal_se_pot  +  fse_dat.f "
            "tables.  Use backend 'shabaev_local' for the local-only "
            "approximation, 'mohr' (default) for tabulated F(Zα), or "
            "'flambaum_ginges' for the electric radiative potential."
        )
