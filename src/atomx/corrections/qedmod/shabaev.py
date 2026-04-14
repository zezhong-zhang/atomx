"""Shabaev–Tupitsyn–Yerokhin model QED operator (QEDMOD).

This backend will implement the model self-energy operator introduced in

    V. M. Shabaev, I. I. Tupitsyn, V. A. Yerokhin,
    "Model operator approach to the Lamb shift calculations in
    relativistic many-electron atoms",
    Phys. Rev. A 88, 012513 (2013).  arXiv:1305.3578.

and its companion Fortran package

    V. M. Shabaev, I. I. Tupitsyn, V. A. Yerokhin,
    "QEDMOD: Fortran program for calculating the model Lamb-shift
    operator", Comput. Phys. Commun. 189, 175 (2015).

The paper provides exact tabulated data:

    Table I   — F_{ik}(αZ) for ns states (Z = 10 … 120, n,n′ = 1..5)
    Table II  — np₁/₂
    Table III — np₃/₂
    Table IV  — nd₃/₂
    Table V   — nd₅/₂

Structure of the operator (paper eqs. 13–18)
--------------------------------------------
The full one-electron model SE operator is

    h^SE  =  V_loc^SE(r)  +  Σ_{i,k} |φ_i⟩ B_{ik} ⟨φ_k|                  (15)

The local part is κ-dependent via a projector (eq. 12–14):

    V_loc^SE(r)  =  Σ_κ V_{loc,κ}^SE(r) · P_κ                            (13)
    V_{loc,κ}^SE(r)  =  A_κ · exp(−r/λ_C)                                (14)

where λ_C = α (Compton wavelength in atomic units) and the amplitude
``A_κ`` is chosen so that the matrix element of V_loc^SE between two
hydrogenic orbitals of the *lowest* energy at given κ reproduces the
exact SE correction for that orbital.

The non-local matrix ``B_{ik}`` is determined from the linear system
(eq. 17):

    B_{ik}  =  Σ_{j,l} (D⁻¹)_{ij} [ ½(Σ_{jl}+Σ_{lj}) − V_{loc}^SE ]_{jl} (D⁻¹)_{lk}

with D_{ik} = ⟨φ_i | ψ_k⟩ the overlap between the auxiliary basis {φ_i}
and the hydrogenic reference wavefunctions {ψ_k}, and Σ_{ij} computed
from the tabulated F_{ik}(αZ) via

    Σ_{ik} = (α/π) · (αZ)⁴ / (n_i n_k)^{3/2} · F_{ik}(αZ) · mc²          (24)

The auxiliary basis {φ_i} is chosen as hydrogenic wavefunctions cut off
at short range (eq. 20), localising the non-local part so that it only
affects low-lying states where SE is significant.

Implementation roadmap
----------------------
1. Tabulate F_{ik}(αZ) from paper Tables I–V for κ ∈ {−1, +1, −2, +2, −3}
   and Z = 10 … 120 in steps of 5.  Provide linear interpolation in Z
   and fall back to hydrogenic ∝ 1/n³ scaling outside the table.
2. Fit or tabulate the κ-dependent amplitude A_κ (QEDMOD source has
   these pre-computed).
3. Build D matrix on-the-fly from SCF orbitals + hydrogenic references
   using :mod:`atomx.dirac.hydrogenic`.
4. Invert to get B_{ik}.
5. Return ⟨a | h^SE | a⟩ = ⟨a | V_loc | a⟩ + ⟨a | Σ|φ⟩B⟨φ| | a⟩.

This is a multi-file, multi-iteration port of the QEDMOD Fortran.
Each table above contains 20 Z values × 9 (n,n′) pairs = 180 floats,
so ≈ 900 numbers across κ = ns / np₁/₂ / np₃/₂ / nd₃/₂ / nd₅/₂.
Accurate but involved; kept as a stub for now.

For expedient users, :mod:`atomx.corrections.qedmod.flambaum_ginges`
offers a simpler local radiative potential that captures a large
fraction of the Shabaev operator's effect at 5–10 % accuracy.
"""

from __future__ import annotations

from atomx.corrections.qedmod.base import SEBackend, SEResult, register_backend


@register_backend
class ShabaevBackend(SEBackend):
    """Shabaev–Tupitsyn–Yerokhin model QED operator — **stub**.

    Reserves the ``"shabaev"`` name in the backend registry and
    documents the target 2013 paper (PRA 88, 012513) and its tabulated
    F_ik(αZ) data.  The port of QEDMOD's parameter tables and non-local
    matrix construction is planned for a follow-up iteration.

    Users who need the bare Dirac-Fock + partial QED today:

        * ``se_method="mohr"``            — tabulated F(Zα), 1s/2s/2p only
        * ``se_method="welton"``          — closed-form, any (n, κ=±1)
        * ``se_method="flambaum_ginges"`` — electric radiative potential
    """

    name = "shabaev"
    description = (
        "Shabaev-Tupitsyn-Yerokhin model QED operator (PRA 88, 012513, "
        "2013; QEDMOD CPC 189, 175, 2015) — not yet implemented."
    )

    def compute(self, orbitals, bsp_data, Z) -> SEResult:
        raise NotImplementedError(
            "Shabaev-Tupitsyn-Yerokhin model QED operator is not yet "
            "implemented. Use backend 'mohr' (default), 'welton', or "
            "'flambaum_ginges' for now. Target reference: "
            "Shabaev, Tupitsyn, Yerokhin, PRA 88, 012513 (2013); "
            "QEDMOD, Comput. Phys. Commun. 189, 175 (2015)."
        )
