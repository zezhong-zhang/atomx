"""QED corrections: vacuum polarization (Uehling) and self-energy.

Both computed perturbatively from converged DHF orbitals.

Vacuum polarization
-------------------
``V_VP(r) = -(2Zα)/(3π) K_1(2cr) / r`` — Uehling potential for a point
nucleus.  Higher-order vacuum-polarization (Wichmann–Kroll, Källén–Sabry)
will be added alongside this; see the roadmap.

Self-energy
-----------
Dispatched through :mod:`atomx.corrections.qedmod`, which supports
multiple interchangeable backends:

    * ``"mohr"`` (default) — tabulated Mohr F(Zα), the previous atomx
      implementation.  Matches earlier results bit-for-bit.
    * ``"welton"`` — closed-form Welton estimate using |ψ(0)|².  Extends
      coverage to every (n, κ = ±1) orbital at ~25 % accuracy.
    * ``"flambaum_ginges"`` — local radiative potential (stub).
    * ``"shabaev"`` — Shabaev–Tupitsyn–Yerokhin QEDMOD (stub).

See :mod:`atomx.corrections.qedmod` for details and references.
"""

import numpy as np
from scipy.special import k1 as bessel_k1

from atomx.constants import ALPHA, C
from atomx.corrections.qedmod import get_backend

# Re-export for backwards compatibility; external consumers that imported
# these names from ``atomx.corrections.qed`` continue to work.
from atomx.corrections.qedmod.mohr import (  # noqa: F401
    _SE_TABLE,
    _interpolate_se_f,
    _overlap_ratio,
)


# ---------------------------------------------------------------------------
# Vacuum Polarization (Uehling potential, 2nd order)
# ---------------------------------------------------------------------------

def uehling_potential(r, Z):
    """Uehling vacuum polarization potential V_VP(r) for a point nucleus.

    V_VP(r) = -(2Zα)/(3π) * (1/r) * K_1(2cr)

    where c = 1/α ≈ 137.036 and K_1 is the modified Bessel function.

    Parameters
    ----------
    r : float or ndarray
        Radial distance in Bohr.
    Z : int
        Nuclear charge.

    Returns
    -------
    V : float or ndarray
        VP potential in Hartree.
    """
    x = 2.0 * C * np.asarray(r, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    prefactor = -2.0 * Z * ALPHA / (3.0 * np.pi)

    if r_arr.ndim == 0:
        if x < 0.01:
            return prefactor / (r_arr * x) if r_arr > 0 else 0.0
        return prefactor / r_arr * bessel_k1(float(x))
    else:
        V = np.zeros_like(r_arr)
        large = x > 0.01
        V[large] = prefactor / r_arr[large] * bessel_k1(x[large])
        small = (~large) & (r_arr > 0)
        if np.any(small):
            V[small] = prefactor / (r_arr[small] * x[small])
        return V


def compute_vp_energy(orbitals, bsp_data, Z):
    """Compute vacuum polarization energy correction.

    E_VP = Σ_a q_a * <a | V_VP | a>
         = Σ_a q_a * ∫ [P_a(r)^2 + Q_a(r)^2] * V_VP(r) dr

    Parameters
    ----------
    orbitals : list of (P, Q, occ, kappa, label) tuples
    bsp_data : dict
    Z : int

    Returns
    -------
    E_VP : float (Hartree)
    """
    from atomx.bspline.evaluate import eval_orbital_on_grid

    nv = bsp_data["nv"]
    ns = bsp_data["ns"]
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    nsp = bsp_data.get("nsp", ns)
    nsq = bsp_data.get("nsq", ns)
    bsp_p = bsp_data["bsp_p"]
    bsp_q = bsp_data.get("bsp_q", bsp_p)

    V_vp = uehling_potential(gr, Z)

    E_VP = 0.0
    for P, Q, occ, kappa, label in orbitals:
        fP = eval_orbital_on_grid(P, bsp_p, nv, ksp, nsp)
        fQ = eval_orbital_on_grid(Q, bsp_q, nv, ksq, nsq)
        rho = fP * fP + fQ * fQ
        E_VP += occ * np.sum(grw * rho * V_vp)

    return E_VP


# ---------------------------------------------------------------------------
# Self-Energy (dispatched through the qedmod backend registry)
# ---------------------------------------------------------------------------

DEFAULT_SE_METHOD = "mohr"


def compute_se_energy(orbitals, Z, bsp_data=None, *, method=DEFAULT_SE_METHOD):
    """Compute the self-energy correction through a pluggable backend.

    Parameters
    ----------
    orbitals : list of (P, Q, occ, kappa, label) tuples
    Z : int
    bsp_data : dict, optional
        B-spline workspace.  Required by backends that evaluate orbitals
        on the radial grid (mohr overlap ratio, welton, ...).
    method : str
        Backend name.  One of :func:`atomx.corrections.qedmod.available_backends`.
        Default ``"mohr"`` reproduces the behaviour of atomx ≤ 0.2.

    Returns
    -------
    float
        Total self-energy in Hartree (summed over orbitals and weighted
        by occupation).

    See Also
    --------
    atomx.corrections.qedmod : detailed backend documentation.
    """
    backend = get_backend(method)
    result = backend.compute(orbitals, bsp_data, Z)
    return result.E_SE


# ---------------------------------------------------------------------------
# Combined QED corrections
# ---------------------------------------------------------------------------

def compute_qed_corrections(orbitals, bsp_data, Z, *, se_method=DEFAULT_SE_METHOD):
    """Compute all QED corrections (VP + SE).

    Parameters
    ----------
    orbitals : list of (P, Q, occ, kappa, label) tuples
    bsp_data : dict
    Z : int
    se_method : str, keyword-only
        Self-energy backend.  Default ``"mohr"`` (atomx ≤ 0.2 behaviour).
        See :mod:`atomx.corrections.qedmod` for alternatives.

    Returns
    -------
    dict with keys: ``E_VP``, ``E_SE``, ``E_QED_total``, ``se_method``.
    """
    E_VP = compute_vp_energy(orbitals, bsp_data, Z)
    E_SE = compute_se_energy(orbitals, Z, bsp_data=bsp_data, method=se_method)

    return {
        "E_VP": E_VP,
        "E_SE": E_SE,
        "E_QED_total": E_VP + E_SE,
        "se_method": se_method,
    }
