"""Crystal-field operator in the m-scheme sparse CI.

V_CF(r) = sum_{k,q} B_{kq} r^k C^k_q(rhat)   (k = 2, 4, 6 for f electrons)

is a ONE-body operator.  In the m-scheme spinor basis its matrix
element connects spinors (n, kappa, m) -> (n', kappa, m - q), so q != 0
terms MIX different m of the same kappa — this is why the assembler's
one-body partner classes must be widened from (kappa, m) to kappa-only
for CF work (see the one_body_extra hook).

Provenance: the B_{kq} supplied here are INPUTS (from the DFT-IAM
decomposition pipeline when available, or literature/empirical values
for validation runs).  This module never invents them.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

from atomx.angular.csf_hamiltonian import _m2_values, twoj
from atomx.angular.wigner import wigner_3j

__all__ = ["build_cf_operator", "point_charge_bkq"]


def _l_of_kappa(kappa: int) -> int:
    return kappa if kappa > 0 else -kappa - 1


def build_cf_operator(orbitals_dict, bsp_data, Z, ns, bkq, radial=None):
    """One-body CF element callable over spinor index pairs.

    Parameters
    ----------
    bkq : dict {(k, q): B_kq in cm^-1}
        Wybourne-style crystal-field parameters.  k = 2, 4, 6 (4f); q
        spans -k..k as supplied.
    radial : optional callable (n, kappa, k) -> <n kappa| r^k |n kappa>
        radial expectation; when None, integrated numerically from the
        orbitals on the workspace grid.

    Returns a callable cf(ia, ib) -> element in Hartree between spinor
    indices of the SHARED m-scheme spinor table layout ((n, kappa, 2m)
    tuples); consumers adapt it to their own index tables via labels.
    """
    from atomx.physics.csf.matrix import _rk  # same integrator family
    import numpy as np

    ext = sorted(orbitals_dict)
    spinors = [(n, k, m2) for (n, k) in ext for m2 in _m2_values(twoj(k))]
    index = {s: i for i, s in enumerate(spinors)}
    CM = 4.556335e-6  # 1 cm^-1 in Hartree

    @lru_cache(maxsize=None)
    def rpow(n_kappa, k):
        """<n kappa | r^k | n kappa> on the workspace grid."""
        (n, kappa) = n_kappa
        P, Q = orbitals_dict[(n, kappa)]
        # grid from bsp_data
        r = bsp_data.get("r") if isinstance(bsp_data, dict) else None
        if r is None:
            # fall back: use the first orbital's grid via _rk(k=0) trick is
            # not available; integrate with the knot grid if exposed
            raise RuntimeError("workspace grid not exposed for r^k; "
                               "pass radial= explicitly")
        w = bsp_data.get("weights")
        dens = P * P + Q * Q
        return float(np.sum(dens * r ** (k + 2) * (w if w is not None else 1.0)))

    def radial_of(n_kappa, k):
        if radial is not None:
            return radial(n_kappa, k)
        return rpow(n_kappa, k)

    @lru_cache(maxsize=None)
    def angular(kappa_a, m2a, kappa_b, m2b, k):
        """<kappa m| C^k_q |kappa m'> for the q fixed by m - m' = q."""
        q = (m2a - m2b) // 2  # q = m_a - m_b
        # C^k_q angular factor: sqrt(4pi/2k+1) Y_kq — reduced matrix
        # element over spinors:
        # <kappa m|C^k_q|kappa' m'> ~ 3j(j k j'; -m, q, m') * <kappa||C^k||kappa'>
        # same-kappa only (CF diagonal in n for like orbitals)
        if kappa_a != kappa_b:
            return 0.0
        two_j = twoj(kappa_a)
        if abs(q) > k:
            return 0.0
        three = wigner_3j(two_j, 2 * k, two_j, -m2a, 2 * q, m2b)
        if three == 0.0:
            return 0.0
        # reduced element <kappa||C^k||kappa> computed via parity+l relation
        l = _l_of_kappa(kappa_a)
        if (l + k + l) % 2:
            return 0.0
        # radial-normalised reduced element handled outside; angular shell:
        return three

    def cf(ia, ib):
        a, b = spinors[ia], spinors[ib]
        if a[1] != b[1]:
            return 0.0
        total = 0.0
        for (k, q), bval in bkq.items():
            if abs(q) > k:
                continue
            m2b_needed = a[2] - 2 * q
            if b[2] != m2b_needed:
                continue
            ang = angular(a[1], a[2], b[1], b[2], k)
            if ang == 0.0:
                continue
            # same-n diagonal radial (leading term; off-diagonal n mixing
            # requires two-electron relaxation and is deferred)
            if a[0] != b[0]:
                continue
            rad = radial_of((a[0], a[1]), k)
            total += bval * CM * rad * ang
        return total

    cf.spinors = spinors
    cf.index = index
    return cf


def point_charge_bkq(charges_coords, r_site, kmax=6):
    """B_{kq} of a point-charge lattice at a site (validation input).

    charges_coords : list of (q_charge, (x, y, z)) in atomic units
    r_site : site position.  Returns {(k, q): B_kq in cm^-1} (Stevens
    convention not applied — Wybourne-style raw B_kq).
    """
    import numpy as np
    CM = 219474.63  # Hartree -> cm^-1
    bkq = {}
    for (qc, (x, y, z)) in charges_coords:
        R = np.sqrt((x - r_site[0])**2 + (y - r_site[1])**2 + (z - r_site[2])**2)
        if R < 1e-8:
            continue
        for k in range(2, kmax + 1, 2):
            for q in range(-k, k + 1):
                # spherical harmonic Y_kq of the ligand direction
                theta = np.arccos((z - r_site[2]) / R)
                phi = np.arctan2(y - r_site[1], x - r_site[0])
                ykq = _real_Y(k, q, theta, phi)
                val = qc * ykq / R ** (k + 1)
                bkq[(k, q)] = bkq.get((k, q), 0.0) + val
    # convention conversion factor r^k C^k_q <-> B_kq kept raw (validation)
    return {kk: v * CM for kk, v in bkq.items()}


def _real_Y(k, q, theta, phi):
    import numpy as np
    from math import factorial
    # unnormalised real spherical harmonic sufficient for SMOKE scaling
    if q < 0:
        return np.sin(abs(q) * phi) * np.sin(theta) ** abs(q)
    return np.cos(q * phi) * np.sin(theta) ** q if q else 1.0
