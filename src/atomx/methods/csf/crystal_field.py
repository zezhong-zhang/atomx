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

from atomx.angular.coefficients import reduced_cl_j2
from atomx.angular.csf_hamiltonian import _m2_values, twoj
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


def build_cf_operator(orbitals_dict, bsp_data, Z, ns, bkq, radial=None,
                      hermiticity="warn"):
    """One-body CF element callable over spinor index pairs.

    Parameters
    ----------
    bkq : dict {(k, q): B_kq in cm^-1}
        k = 2, 4, 6 for f electrons; q spans -k..k as supplied.  Values
        may be complex; Hermiticity requires B_{k,-q} = (-1)^q B_{kq}*.
        UNIT HAZARD: these are RADIAL-FREE A_kq.  Literature Wybourne
        B_kq usually INCLUDE <r^k> (B = A <r^k>) — divide it out before
        feeding them here, or r^k is applied twice.
    radial : optional callable ((na,ka), (nb,kb), k) -> <a| r^k |b> in
        bohr^k; when None, integrated on the workspace Gauss grid.
    hermiticity : "warn" (default) / "strict" / "off"
        Build-time check of the B set: a partner B_{k,-q} that is
        missing (common q>0-only input habit) or violates
        B_{k,-q} = (-1)^q B_{kq}* warns in "warn" mode and raises in
        "strict" — a non-Hermitian set silently produces a non-Hermitian
        H otherwise.

    Returns a callable cf(ia, ib) -> element in HARTREE (complex when
    any B is complex) between spinor indices of the shared m-scheme
    spinor layout ((n, kappa, 2m) tuples); consumers adapt it to their
    own index tables via labels.  CI wiring additionally needs a REAL
    set (float() of complex raises) — complex sites wait on the tesseral
    conversion follow-up.
    """
    import warnings

    import numpy as np

    for (k, q) in bkq:
        if abs(q) > k:
            raise ValueError(f"B entry ({k},{q}): |q| > k is invalid")
    if hermiticity != "off":
        for (k, q), bval in bkq.items():
            partner = bkq.get((k, -q))
            expect = (-1) ** q * np.conj(bval)
            if partner is None:
                if q != 0:
                    msg = f"B set lacks ({k},{-q}): H will be non-Hermitian"
                    if hermiticity == "strict":
                        raise ValueError(msg)
                    warnings.warn(msg, stacklevel=2)
            elif abs(partner - expect) > 1e-9 * max(1.0, abs(expect)):
                msg = (f"B({k},{-q}) != (-1)^q conj(B({k},{q})): "
                       "H will be non-Hermitian")
                if hermiticity == "strict":
                    raise ValueError(msg)
                warnings.warn(msg, stacklevel=2)

    ext = sorted(orbitals_dict)
    spinors = [(n, k, m2) for (n, k) in ext for m2 in _m2_values(twoj(k))]
    index = {s: i for i, s in enumerate(spinors)}

    rpow = radial if radial is not None else rpow_grid(orbitals_dict,
                                                       bsp_data)

    @lru_cache(maxsize=None)
    def norm_of(n_kappa):
        # same quadrature as rpow (single consistent Gauss-grid measure;
        # dividing by an external _norm_sq mixes two quadrature rules)
        return rpow(n_kappa, n_kappa, 0)

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


def _Y_kq(k, q, theta, phi):
    """Complex spherical harmonic Y_kq (Condon-Shortley), module-level
    for direct test access.  P_k^q via the power-basis q-th derivative
    of P_k; the q<0 branch is Y_{k,-q} = (-1)^q conj(Y_{k,q}) — NO
    factorial ratio (the normalisation carries it; keeping both was the
    accidental-cancellation bug the review caught)."""
    import cmath
    from math import factorial, pi, sqrt

    import numpy as np

    if q < 0:
        return ((-1) ** (-q)) * np.conj(_Y_kq(k, -q, theta, phi))
    x = np.cos(theta)
    pk = np.polynomial.legendre.leg2poly(
        np.polynomial.legendre.Legendre.basis(k).coef)
    dp = np.polynomial.Polynomial(pk).deriv(q)
    pkq = ((-1.0) ** q) * (1.0 - x * x) ** (q / 2.0) * float(dp(x))
    norm = sqrt((2 * k + 1) / (4 * pi)
                * factorial(k - q) / factorial(k + q))
    return norm * pkq * cmath.exp(1j * q * phi)


def point_charge_bkq(charges_coords, r_site, kmax=6, r_inside=None):
    """A^q_k of a point-charge lattice at a site (validation input).

    ELECTRON Hamiltonian convention: the crystal-field operator acting
    on the ELECTRON is V = -phi with phi = sum_i q_i/|R_i - r|, so

        A^q_k = -sqrt(4pi/(2k+1)) sum_i q_i conj(Y_kq(R_i)) / R_i^{k+1}

    (the overall MINUS is the electron charge; entering physical ionic
    charges q_i > 0 gives electron B values directly).

    charges_coords : list of (q_charge, (x, y, z)) in atomic units;
    r_site : the impurity site position;
    r_inside : REQUIRED bound on the CF shell radius (bohr).  The
        r^k/R^{k+1} expansion converges only for r < R_min — a charge
        inside the shell silently produces divergent/garbage B, so this
        guard refuses to run without an explicit shell estimate.

    Returns {(k, q): B_kq} with complex values in cm^-1 (the B
    convention of :func:`build_cf_operator`; Hermiticity holds by
    construction for real charges).  Closed-form check (gated in the
    test suite): one axial charge q at distance R gives exactly
    A_k^0 = -q/R^{k+1} for every even k (because
    sqrt(4pi/(2k+1)) Y_k0 = P_k and P_k(1) = 1).
    """
    from math import pi, sqrt

    import numpy as np

    if r_inside is None:
        raise ValueError(
            "r_inside (bohr, an upper bound on the CF shell radius, "
            "e.g. ~1 for 4f) is required: the multipole expansion "
            "diverges for charges inside the shell")

    ykq = _Y_kq

    bkq: Dict[Tuple[int, int], complex] = {}
    for (qc, (x, y, z)) in charges_coords:
        dx, dy, dz = x - r_site[0], y - r_site[1], z - r_site[2]
        R = np.sqrt(dx * dx + dy * dy + dz * dz)
        if R < 1e-8:
            continue
        if r_inside >= R:
            raise ValueError(
                f"charge at R={R:.3f} bohr <= r_inside={r_inside}: "
                "multipole expansion invalid (charge inside the shell)")
        theta = np.arccos(dz / R)
        phi = np.arctan2(dy, dx)
        for k in range(2, kmax + 1, 2):
            for q in range(-k, k + 1):
                val = -(qc * np.conj(ykq(k, q, theta, phi))
                        / R ** (k + 1))
                val = val * sqrt(4 * pi / (2 * k + 1))
                bkq[(k, q)] = bkq.get((k, q), 0.0) + val
    return {kk: v * CM_PER_HARTREE for kk, v in bkq.items()}
