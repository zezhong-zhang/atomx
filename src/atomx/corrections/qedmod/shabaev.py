"""Shabaev–Tupitsyn–Yerokhin model operator (stub).

The Shabaev–Tupitsyn–Yerokhin (STY) model operator is the current state
of the art for the one-electron QED self-energy in many-electron atoms.
Its full form is local + non-local:

    Σ_SE(r, r') = V_loc(r) δ(r − r') + Σ_{i,j} |φ_i(r)⟩ B_{ij} ⟨φ_j(r')|

where the local potential ``V_loc(r)`` reproduces the short-range behaviour
of the exact SE, the basis ``{φ_i}`` runs over a set of hydrogenic
reference functions (typically ``n = 1..5`` for each κ), and the matrix
``B_{ij}`` is fixed so that the diagonal hydrogenic SE values agree with
the exact tabulation of Mohr / Yerokhin / Jentschura.

Key references
--------------
V. M. Shabaev, I. I. Tupitsyn, V. A. Yerokhin,
    "Model operator approach to the Lamb shift calculations in
    relativistic many-electron atoms",
    Phys. Rev. A 88, 012513 (2013).  [arXiv:1305.3578]
V. M. Shabaev, I. I. Tupitsyn, V. A. Yerokhin,
    "QEDMOD: Fortran program for calculating the model Lamb-shift
    operator", Comput. Phys. Commun. 189, 175 (2015).
V. A. Yerokhin, V. M. Shabaev,
    "Lamb shift of n=1 and n=2 states of hydrogen-like atoms,
    1 ≤ Z ≤ 110", J. Phys. Chem. Ref. Data 44, 033103 (2015).

Accuracy: sub-percent agreement with exact two-loop SE for every bound
orbital of every atom up to Z = 120.  This is the operator to use for
precision spectroscopy; it is also suitable for self-consistent QED
(plugging ``V_loc`` directly into the DHF Fock matrix).

Implementation plan
-------------------
    1. Port the parameterised ``V_loc`` from the QEDMOD Fortran sources
       (s, p, d, f shells; Z = 1 … 120 tabulation).
    2. Build the non-local matrix ``B_{ij}`` from hydrogenic references
       computed by ``atomx.dirac.hydrogenic``.
    3. Implement matrix element assembly against the SCF orbitals.
    4. Expose ``V_loc`` separately so the SCF solver can add it to the
       Fock matrix for self-consistent QED (the "scqed" SCF mode).
"""

from __future__ import annotations

from atomx.corrections.qedmod.base import SEBackend, SEResult, register_backend


@register_backend
class ShabaevBackend(SEBackend):
    """Shabaev–Tupitsyn–Yerokhin model operator (QEDMOD) — **stub**.

    Reserves the ``"shabaev"`` name in the backend registry and documents
    the target papers.  The QEDMOD parameter tables and assembly code will
    be filled in in a follow-up iteration.
    """

    name = "shabaev"
    description = (
        "Shabaev–Tupitsyn–Yerokhin model operator (PRA 88, 012513, 2013; "
        "QEDMOD, CPC 189, 175, 2015) — not yet implemented."
    )

    def compute(self, orbitals, bsp_data, Z) -> SEResult:
        raise NotImplementedError(
            "Shabaev–Tupitsyn–Yerokhin model operator is not yet implemented. "
            "Use backend 'mohr' (default) or 'welton' for now. "
            "Target reference: Shabaev, Tupitsyn, Yerokhin, "
            "Phys. Rev. A 88, 012513 (2013); QEDMOD, "
            "Comput. Phys. Commun. 189, 175 (2015)."
        )
