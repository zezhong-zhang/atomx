"""Crystal-field operator in the m-scheme sparse CI.

V_CF = sum_{k,q} A^q_k r^k C^q_k(rhat),   C^q_k = sqrt(4pi/(2k+1)) Y_{kq}

is a ONE-body operator.  In the m-scheme spinor basis its element

    <n kappa_a m_a| V_CF |n' kappa_b m_b>
        = sum_{k,q} B_kq * <n kappa_a| r^k |n' kappa_b>
          * (-1)^{j_a - m_a} (j_a k j_b; -m_a q m_b) <kappa_a||C^k||kappa_b>

connects (kappa, m) -> (kappa, m - q): q != 0 terms MIX different m of
the same kappa — this is why the assembler needs the separate m-mixing
one-body channel (``one_body_m_delta`` in sparse_ci) rather than a
widening of the (kappa, m) partner classes (the base one-body value is
m-blind and would be wrongly re-emitted for every m pair).

CONVENTION: ``bkq`` values are Wybourne-style crystal-field parameters
A^q_k expressed in cm^-1 (1 au = 219474.63 cm^-1); the r^k expectation
in bohr^k is applied INSIDE the operator, so B_kq are radial-free.
Values may be complex (general site); a Hermitian set satisfies
B_{k,-q} = (-1)^q conj(B_{kq}).  Provenance: the B_kq supplied here are
INPUTS (DFT-IAM decomposition, literature, or :func:`point_charge_bkq`
for validation) — this module never invents them.

Physics gates (recorded lesson: sign/magnitude gate BEFORE anchor runs;
see tests/test_crystal_field.py):
  G1  radial k=0 == 1 for normalised orbitals (fixes the P=rR measure)
  G2  Hermiticity: cf(ia, ib) == conj(cf(ib, ia))
  G3  per-subshell trace of the CF diagonal == 0 for every k > 0
      (C^k is traceless over a full m set)
  G4  digit-exact vs direct numerical angular quadrature of the spinor
      spherical harmonics (independent path: CG-built Omega, no 3j of
      the operator route)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

from atomx.angular.csf_hamiltonian import _m2_values, twoj
from atomx.angular.coefficients import reduced_cl_j2
from atomx.angular.wigner import wigner_3j

__all__ = ["build_cf_operator", "point_charge_bkq", "rpow_grid"]

CM_PER_HARTREE = 219474.63
HARTREE_PER_CM = 1.0 / CM_PER_HARTREE


def _l_of_kappa(kappa: int) -> int:
    return kappa if kappa > 0 else -kappa - 1


def rpow_grid(orbitals_dict, bsp_data):
    """Radial moments <a| r^k |b> on the workspace Gauss grid.

    The B-spline P, Q follow the r*R convention (measure dr, cf. the
    Slater cell integrator), so
        <a| r^k |b> = sum_i w_i (P_a P_b + Q_a Q_b)(r_i) r_i^k
    computed from the coefficient representation via the workspace's
    on-grid basis evaluation.  Returns a cached callable.
    """
    import numpy as np

    from atomx.methods.dirac_fock import _eval_bspline_on_grid

    r = np.asarray(bsp_data["gr"]).ravel()
    w = np.asarray(bsp_data["grw"]).ravel()

    @lru_cache(maxsize=None)
    def grid_of(n_kappa):
        P, Q = orbitals_dict[n_kappa]
        Pg = _eval_bspline_on_grid(P, bsp_data, "p")
        Qg = _eval_bspline_on_grid(Q, bsp_data, "q")
        return np.asarray(Pg).ravel(), np.asarray(Qg).ravel()

    @lru_cache(maxsize=None)
    def rpow(a_key, b_key, k):
        Pa, Qa = grid_of(a_key)
        Pb, Qb = grid_of(b_key)
        return float(np.sum(w * (Pa * Pb + Qa * Qb) * r ** k))

    return rpow


def build_cf_operator(orbitals_dict, bsp_data, Z, ns, bkq, radial=None):
    """One-body CF element callable over spinor index pairs.

    Parameters
    ----------
    bkq : dict {(k, q): B_kq in cm^-1}
        k = 2, 4, 6 for f electrons; q spans -k..k as supplied.  Values
        may be complex; Hermiticity requires B_{k,-q} = (-1)^q B_{kq}*.
    radial : optional callable ((na,ka), (nb,kb), k) -> <a| r^k |b> in
        bohr^k; when None, integrated on the workspace Gauss grid.

    Returns a callable cf(ia, ib) -> element in HARTREE (complex when
    any B is complex) between spinor indices of the shared m-scheme
    spinor layout ((n, kappa, 2m) tuples); consumers adapt it to their
    own index tables via labels.
    """
    import numpy as np

    ext = sorted(orbitals_dict)
    spinors = [(n, k, m2) for (n, k) in ext for m2 in _m2_values(twoj(k))]
    index = {s: i for i, s in enumerate(spinors)}

    rpow = radial if radial is not None else rpow_grid(orbitals_dict,
                                                       bsp_data)

    @lru_cache(maxsize=None)
    def norm_of(n_kappa):
        if bsp_data is None:
            return 1.0  # synthetic/angular tests with a custom radial
        from atomx.physics.csf.matrix import _norm_sq
        P, Q = orbitals_dict[n_kappa]
        return _norm_sq(P, Q, n_kappa[1], Z, bsp_data, ns)

    @lru_cache(maxsize=None)
    def radial_of(a_key, b_key, k):
        """Symmetrically renormalised <a| r^k |b>."""
        n2a, n2b = norm_of(a_key), norm_of(b_key)
        return rpow(a_key, b_key, k) / np.sqrt(n2a * n2b)

    @lru_cache(maxsize=None)
    def angular(kappa_a, m2a, kappa_b, m2b, k):
        """<kappa_a m_a| C^k_q |kappa_b m_b>, q fixed by m_a - m_b = q."""
        q2 = m2a - m2b
        if q2 % 2:
            return 0.0
        # parity: l_a + k + l_b must be even (same-l for CF in practice)
        if (_l_of_kappa(kappa_a) + _l_of_kappa(kappa_b) + k) % 2:
            return 0.0
        # GRASP/FAC reduced element (reduced_cl_j2): the legacy reduced_C
        # phase differs by (-1)^{j_a+j_b+k} — harmless in the CI's
        # two-vertex products, WRONG for a one-body operator (G4 gate)
        red = reduced_cl_j2(twoj(kappa_a), 2 * k, twoj(kappa_b))
        if red == 0.0:
            return 0.0
        tja, tjb = twoj(kappa_a), twoj(kappa_b)
        t3 = wigner_3j(tja, 2 * k, tjb, -m2a, q2, m2b)
        if t3 == 0.0:
            return 0.0
        phase = -1.0 if ((tja - m2a) // 2) % 2 else 1.0
        return phase * t3 * red

    def cf(ia, ib):
        a, b = spinors[ia], spinors[ib]
        if a[0] != b[0]:
            # off-diagonal n mixing deferred (needs the same-n
            # leading term first; two-body relaxation follow-up)
            return 0.0
        q2 = a[2] - b[2]
        if q2 % 2:
            return 0.0
        q = q2 // 2
        total = 0.0
        for (k, kq), bval in bkq.items():
            if kq != q:
                continue        # B lookup keyed by the tensor component
            ang = angular(a[1], a[2], b[1], b[2], k)
            if ang == 0.0:
                continue
            rad = radial_of((a[0], a[1]), (b[0], b[1]), k)
            total += bval * HARTREE_PER_CM * rad * ang
        return total

    cf.spinors = spinors
    cf.index = index
    cf.radial_of = radial_of
    cf.angular = angular
    return cf


def point_charge_bkq(charges_coords, r_site, kmax=6):
    """A^q_k of a point-charge lattice at a site (validation input).

    Multipole expansion of sum_i q_i / |R_i - r| for r inside the shell:

        A^q_k = sqrt(4pi/(2k+1)) sum_i q_i conj(Y_kq(R_i)) / R_i^{k+1}

    charges_coords : list of (q_charge, (x, y, z)) in atomic units;
    r_site : the impurity site position.  Returns {(k, q): B_kq} with
    complex values in cm^-1 (the B convention of
    :func:`build_cf_operator`; Hermiticity holds by construction for
    real charges).
    """
    import cmath
    from math import factorial, pi, sqrt

    import numpy as np

    def Y(k, q, theta, phi):
        """Complex spherical harmonic Y_kq.

        P_k^q via the power-basis q-th derivative of P_k (validation
        grade; cross-checked against sympy in the test suite).
        """
        if q < 0:
            mq = -q
            pref = ((-1) ** mq) * factorial(k - mq) / factorial(k + mq)
            return pref * Y(k, mq, theta, phi)
        x = np.cos(theta)
        pk = np.polynomial.legendre.leg2poly(
            np.polynomial.legendre.Legendre.basis(k).coef)
        dp = np.polynomial.Polynomial(pk).deriv(q)
        pkq = ((-1.0) ** q) * (1.0 - x * x) ** (q / 2.0) * float(dp(x))
        norm = sqrt((2 * k + 1) / (4 * pi)
                    * factorial(k - q) / factorial(k + q))
        return norm * pkq * cmath.exp(1j * q * phi)

    bkq: Dict[Tuple[int, int], complex] = {}
    for (qc, (x, y, z)) in charges_coords:
        dx, dy, dz = x - r_site[0], y - r_site[1], z - r_site[2]
        R = np.sqrt(dx * dx + dy * dy + dz * dz)
        if R < 1e-8:
            continue
        theta = np.arccos(dz / R)
        phi = np.arctan2(dy, dx)
        for k in range(2, kmax + 1, 2):
            for q in range(-k, k + 1):
                val = (qc * np.conj(Y(k, q, theta, phi))
                       / R ** (k + 1))
                val = val * sqrt(4 * pi / (2 * k + 1))
                bkq[(k, q)] = bkq.get((k, q), 0.0) + val
    return {kk: v * CM_PER_HARTREE for kk, v in bkq.items()}
