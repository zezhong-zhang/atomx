"""End-to-end regression tests for M1 and E2 transition rates.

Validates the full ``rates × multipole × angular`` pipeline for the
magnetic-dipole (L=1) and electric-quadrupole (L=2) cases against
published hydrogenic reference values.

Covers HYP-144 — closes the M1 / E2 verification gate in
``benchmarks/REPORT.md §7.5``.

References
----------
* Drake, G. W. F. (1971), *Phys. Rev. A* **3**, 908 — H-like
  ``2s_{1/2} -> 1s_{1/2}`` M1 rate
  ``A(M1) ≈ 2.5e-6 · Z^10 s^-1`` (leading-order Z-scaling, accurate
  to a few % up through Z ~ 30; deviates by ~5–10 % at Z ~ 50 from
  relativistic / retardation corrections beyond the leading term).
* Garstang, R. H. (1957), *Mon. Not. R. Astron. Soc.* **117**, 393 /
  NIST ASD — H I ``3d_{5/2} -> 1s_{1/2}`` E2 rate
  ``A(E2) = 5.96e2 s^-1``, scales as Z^6 for the leading hydrogenic
  contribution.

Bug context: pre-fix the M1 pipeline carried a stray /c in ``radial_mL``
plus a missing /(2L+1) in ``angular_mL``, producing rates that were
~ c^2 / (2L+1)^2 = ~ 2000× too small at L=1. The fix here matches the
Grant (2007) Eq. 8.4.17 / FAC ``TRRateMultipole`` convention.
"""

import pytest

from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import (
    build_dirac_matrices, solve_dirac,
)
from atomx.prop.transitions.multipole import radial_eL, radial_mL
from atomx.prop.transitions.angular import angular_eL, angular_mL
from atomx.prop.transitions.rates import einstein_A_eL, einstein_A_mL


def _solve_hlike(Z, kappa, n_idx, ns=80, ks=9, r_max=None, kb=True):
    """Solve the Dirac equation for a hydrogenic ion and return the
    ``n_idx``-th bound (P, Q) spinor plus the bspline grid data."""
    if r_max is None:
        r_max = max(60.0 / Z, 5.0)
    t = make_knots(ns, ks, 0.005 / Z, r_max)
    data = prepare_bspline_data(t, ns, ks, kinetic_balance=kb)
    H, S = build_dirac_matrices(data["ns"], kappa, Z, data)
    res = solve_dirac(
        H, S, data["ns"], kappa,
        nsp=data.get("nsp", data["ns"]),
        nsq=data.get("nsq", data["ns"]),
    )
    P, Q, E = res.get_orbital(n_idx)
    return P, Q, E, data


def _hlike_A_M1_2s_1s(Z, ns=80, r_max=None):
    """Full pipeline: A(M1, H-like 2s_{1/2} -> 1s_{1/2}) at given Z."""
    P_2s, Q_2s, E_2s, data = _solve_hlike(Z, -1, 1, ns=ns, r_max=r_max)
    P_1s, Q_1s, E_1s, _ = _solve_hlike(Z, -1, 0, ns=ns, r_max=r_max)
    omega = E_2s - E_1s
    R = radial_mL(P_2s, Q_2s, P_1s, Q_1s, data, L=1)
    ang = angular_mL(-1, -1, L=1)
    S = (ang * R) ** 2
    return einstein_A_mL(omega, S, L=1, tj_i=1)


def _hlike_A_E2_3d52_1s(Z, ns=100, r_max=None):
    """Full pipeline: A(E2, H-like 3d_{5/2} -> 1s_{1/2}) at given Z."""
    if r_max is None:
        r_max = max(120.0 / Z, 8.0)
    P_3d, Q_3d, E_3d, data = _solve_hlike(Z, -3, 0, ns=ns, r_max=r_max)
    P_1s, Q_1s, E_1s, _ = _solve_hlike(Z, -1, 0, ns=ns, r_max=r_max)
    omega = E_3d - E_1s
    R = radial_eL(P_3d, Q_3d, P_1s, Q_1s, data, L=2)
    ang = angular_eL(-3, -1, L=2)
    S = (ang * R) ** 2
    return einstein_A_eL(omega, S, L=2, tj_i=5)


# ----------------------------------------------------------------------
# M1: H-like 2s_{1/2} -> 1s_{1/2}, Drake (1971) reference  A = 2.5e-6 Z^10
# ----------------------------------------------------------------------

@pytest.mark.parametrize("Z,A_ref", [
    (1, 2.5e-6 * 1**10),
    (2, 2.5e-6 * 2**10),
    (10, 2.5e-6 * 10**10),
])
def test_M1_hlike_2s_to_1s_drake(Z, A_ref):
    """Hydrogenic M1 ``2s_{1/2} -> 1s_{1/2}`` matches Drake's leading
    ``2.5e-6 Z^10 s^-1`` to within 1 % across Z = 1, 2, 10.

    This pins both the magnetic prefactor and the
    ``(kappa_i+kappa_f)/(2L+1)`` algebraic factor — pre-fix the rate was
    ~ 2000× too small at all Z.
    """
    A = _hlike_A_M1_2s_1s(Z)
    assert A == pytest.approx(A_ref, rel=0.01), (
        f"M1 Z={Z}: atomx={A:.4e}, Drake={A_ref:.4e}, "
        f"ratio={A/A_ref:.4f}"
    )


def test_M1_hlike_2s_to_1s_Z26_moderate_relativistic():
    """At Z = 26 (Fe XXVI) the leading ``2.5e-6 Z^10`` Drake scaling
    starts to drift from the exact relativistic rate by a few %; we
    accept 5 % here as the documented retardation envelope.  The
    ``Z = 1, 2, 10`` tests above pin the prefactor; this one keeps an
    eye on the moderate-Z regime."""
    A = _hlike_A_M1_2s_1s(26)
    A_ref = 2.5e-6 * 26**10
    assert A == pytest.approx(A_ref, rel=0.05), (
        f"M1 Z=26: atomx={A:.4e}, Drake-Z^10={A_ref:.4e}, "
        f"ratio={A/A_ref:.4f}"
    )


# ----------------------------------------------------------------------
# E2: H-like 3d_{5/2} -> 1s_{1/2}, Garstang (1957) reference
#   A = 5.96e2 s^-1 at Z = 1, scaling as Z^6.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("Z,A_ref", [
    (1, 5.96e2 * 1**6),
    (10, 5.96e2 * 10**6),
    (26, 5.96e2 * 26**6),
])
def test_E2_hlike_3d52_to_1s_garstang(Z, A_ref):
    """Hydrogenic E2 ``3d_{5/2} -> 1s_{1/2}`` matches the Garstang /
    NIST hydrogenic value ``5.96e2 Z^6 s^-1`` to within 1 % across
    Z = 1, 10, 26.

    This pins the existing electric-quadrupole prefactor
    (Grant Eq. 8.4.10 at L=2) and confirms the E2 path is unaffected
    by the M1 fix."""
    A = _hlike_A_E2_3d52_1s(Z)
    assert A == pytest.approx(A_ref, rel=0.01), (
        f"E2 Z={Z}: atomx={A:.4e}, Garstang Z^6={A_ref:.4e}, "
        f"ratio={A/A_ref:.4f}"
    )


# ----------------------------------------------------------------------
# Sanity: E1 baseline must not regress (the L=1 fix in REPORT §1.1).
# ----------------------------------------------------------------------

def test_E1_baseline_h_2p_to_1s_regression():
    """Closed-form check that the E1 H 2p_{1/2} -> 1s_{1/2} rate
    using the canonical line strength S = 1.110 still returns
    6.265e8 s^-1.  Guards against accidental coupling of the M1 fix
    into the E1 path."""
    A = einstein_A_eL(omega=3.0/8.0, line_strength=1.110, L=1, tj_i=1)
    assert A == pytest.approx(6.265e8, rel=2e-3)
