"""Native theory-ladder parameters for the embedding impurity (design E1.5).

The impurity's Slater multipoles and spin-orbit coupling come from atomx's
own Dirac-Fock machinery instead of hardcoded tables:

- F2/F4(3d,3d): ``compute_slater_fk`` on the DF 3d B-spline orbitals,
  averaged over the jj partners (which agree to <1% at this Z);
- zeta_3d: (2/3) of the DF 3d jj orbital-energy splitting;
- provenance records the theory level, basis and nuclear model.

Acceptance anchors (tests/test_embedding_native.py): the DF values land
within 1% of the Cowan free-ion set (12.234 / 7.598 eV) — validating the
radial pipeline against an independent tabulation — and the E1 solver
driven by DF parameters reproduces the exact Racah closed forms with
B = F2/49, C = 5*F4/49 recomputed from the DF values.

Honest scope: the PARAMETERS are now theory-ladder native; the eigenproblem
ASSEMBLY still runs in the LS |m,s> basis (standard practice for
Cowan-parameterized cluster models). Re-routing the assembly through the
relativistic CSF machinery remains the deeper E1.5+ item.

Kappa bookkeeping (bit me once): kappa(3d3/2) = 2, kappa(3d5/2) = -3
(-(l+1) for j = l+1/2); kappa = -2 is 2p3/2. Always key orbitals by
(n, kappa) through ``orbitals_by_subshell()``.
"""
from __future__ import annotations

HARTREE_EV = 27.211386245988

# Cowan free-ion tabulation for cross-validation (independent source)
COWAN_NI2_FREE_ION = {"F2": 12.234, "F4": 7.598}


def dirac_fock_3d_params(
    *, Z: int = 28, charge: int = 2, max_iter: int = 60
) -> dict:
    """Dirac-Fock Slater/SOC parameters for an open 3d impurity shell."""
    from atomx import Atom, Basis
    from atomx.methods.dirac_fock import DiracFock
    from atomx.numerics.bspline.slater_yk import compute_slater_fk
    from atomx.types import NuclearModel

    config = "1s2 2s2 2p6 3s2 3p6 3d8"
    atom = Atom(
        Z, config=config, charge=charge, engine="bspline",
        basis=Basis(ns=120, ks=11, r_max=60.0),
    )
    state = atom.solve(DiracFock(nuclear=NuclearModel.POINT), max_iter=max_iter)
    res = state.result
    if not res.converged:
        raise RuntimeError("Dirac-Fock SCF did not converge")

    by = res.orbitals_by_subshell()
    P3, Q3, _ = by[(3, 2)]    # 3d3/2
    P5, Q5, _ = by[(3, -3)]   # 3d5/2
    bsp = res._bsp_data

    f2 = [
        compute_slater_fk(P, Q, P, Q, bsp, 2) * HARTREE_EV
        for P, Q in ((P3, Q3), (P5, Q5))
    ]
    f4 = [
        compute_slater_fk(P, Q, P, Q, bsp, 4) * HARTREE_EV
        for P, Q in ((P3, Q3), (P5, Q5))
    ]
    e = res.orbital_energies
    zeta = (2.0 / 3.0) * (e["3d5/2"] - e["3d3/2"]) * HARTREE_EV
    return {
        "F2_eV": float(sum(f2) / 2.0),
        "F4_eV": float(sum(f4) / 2.0),
        "F2_jj_spread_eV": float(abs(f2[0] - f2[1])),
        "zeta_3d_eV": float(zeta),
        "slater_eV": {"F2": float(sum(f2) / 2.0), "F4": float(sum(f4) / 2.0)},
        "provenance": {
            "model": "atomx:DiracFock(point-nucleus,bspline)",
            "config": config,
            "charge": charge,
            "theory_level": "Dirac-Hartree-Fock",
            "converged": True,
            "note": (
                "free-ion values; solid-state screening is a separate "
                "convention (benchmark screened set: F2=7.9521, F4=4.9387)"
            ),
        },
    }
