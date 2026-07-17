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

    * ``"auto"`` (default) — Mohr/MSS for Z <= 36, Indelicato for Z >= 37.
    * ``"mohr"`` / ``"mss"`` — tabulated Mohr F(Zα), the previous atomx
      implementation.  Matches earlier results bit-for-bit.
    * ``"indelicato"`` — FAC/Indelicato-Mohr high-Z K/L-shell table.
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


#: Dense auxiliary log-grid used to integrate nuclear-range kernels
#: against B-spline orbitals.  25k points over 9 decades ≈ 2800 points
#: per decade — trapezoid error < 1e-6 relative on the Uehling folding
#: (checked against a 400k-point evaluation).
_DENSE_KERNEL_POINTS = 25_000
_DENSE_KERNEL_R_MIN = 1.0e-9


def _short_range_expectation(orbitals, bsp_data, V_of_r):
    """Σ_a q_a ⟨a|V|a⟩ for a nuclear-range local potential ``V(r)``.

    The Uehling kernel ``K₁(2cr)/r`` lives on the Compton scale
    ``r ≲ 1/(2c) ≈ 0.004`` Bohr and diverges like ``1/r²`` toward the
    origin; the SCF basis' own Gauss nodes under-resolve it whenever the
    innermost knots are wide on that scale (He physics grid: 8e-3
    relative error on E_VP).  For B-spline workspaces we therefore
    evaluate the spline expansion EXACTLY on a dense auxiliary log grid
    and integrate there.  Radial-grid (shooting) workspaces keep their
    native ``sum(f · difr)`` sum — Salvat's SGRID is exponentially dense
    at the origin and self-converged to ~1e-7 on this integral.
    """
    from atomx.corrections.workspace import (
        eval_orbital_pair, is_grid_workspace, workspace_arrays)

    gr, grw = workspace_arrays(bsp_data)

    if is_grid_workspace(bsp_data) or "t" not in bsp_data:
        # Grid flavour — or a legacy B-spline workspace without its knot
        # vector, where the Gauss-node sum is the only option.
        V = V_of_r(gr)
        E = 0.0
        for P, Q, occ, kappa, label in orbitals:
            fP, fQ = eval_orbital_pair(P, Q, bsp_data)
            E += occ * float(np.sum(grw * (fP * fP + fQ * fQ) * V))
        return E

    # B-spline flavour: exact spline evaluation on a dense log grid.
    from scipy.interpolate import BSpline

    t = np.asarray(bsp_data["t"], dtype=float)
    tp = np.asarray(bsp_data.get("tp", t), dtype=float)
    ks = bsp_data["ks"]
    ksp = bsp_data.get("ksp", ks)
    ksq = bsp_data.get("ksq", ks)
    nsp = bsp_data.get("nsp", bsp_data["ns"])
    nsq = bsp_data.get("nsq", bsp_data["ns"])

    r = np.geomspace(_DENSE_KERNEL_R_MIN, float(t[-1]),
                     _DENSE_KERNEL_POINTS)
    w = np.empty_like(r)
    w[1:-1] = 0.5 * (r[2:] - r[:-2])
    w[0] = 0.5 * (r[1] - r[0])
    w[-1] = 0.5 * (r[-1] - r[-2])
    V = V_of_r(r)

    E = 0.0
    for P, Q, occ, kappa, label in orbitals:
        splP = BSpline(tp, np.asarray(P[:nsp], dtype=float), ksp - 1,
                       extrapolate=False)
        splQ = BSpline(t, np.asarray(Q[:nsq], dtype=float), ksq - 1,
                       extrapolate=False)
        fP = np.nan_to_num(splP(r))
        fQ = np.nan_to_num(splQ(r))
        E += occ * float(np.sum(w * (fP * fP + fQ * fQ) * V))
    return E


def compute_vp_energy(orbitals, bsp_data, Z):
    """Compute vacuum polarization energy correction.

    E_VP = Σ_a q_a * <a | V_VP | a>
         = Σ_a q_a * ∫ [P_a(r)^2 + Q_a(r)^2] * V_VP(r) dr

    Parameters
    ----------
    orbitals : list of (P, Q, occ, kappa, label) tuples
    bsp_data : dict
        B-spline workspace, or a radial-grid workspace from
        :func:`atomx.corrections.workspace.make_grid_workspace`
        (in which case ``P``/``Q`` are grid arrays).
    Z : int

    Returns
    -------
    E_VP : float (Hartree)
    """
    return _short_range_expectation(
        orbitals, bsp_data, lambda r: uehling_potential(r, Z))


# ---------------------------------------------------------------------------
# Self-Energy (dispatched through the qedmod backend registry)
# ---------------------------------------------------------------------------

DEFAULT_SE_METHOD = "auto"


def resolve_se_method(method: str, Z: int) -> str:
    """Resolve user-facing QED backend aliases and the high-Z auto policy."""
    if method == "auto":
        return "mohr" if Z <= 36 else "indelicato"
    if method == "mss":
        return "mohr"
    return method


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
    resolved_method = resolve_se_method(method, Z)
    backend = get_backend(resolved_method)
    result = backend.compute(orbitals, bsp_data, Z)
    return result.E_SE


# ---------------------------------------------------------------------------
# Combined QED corrections
# ---------------------------------------------------------------------------

def compute_wk_vp_energy(orbitals, bsp_data, Z):
    """Wichmann–Kroll higher-order vacuum-polarization energy contribution.

    Uses the simplified closed-form approximation of Flambaum–Ginges
    (PRA 72, 052115, 2005), eq. 12:

        V_WK(r) = +(4α/9π) · (Z/r) · 0.092 Z²α² / (1 + (1.62·c·r)⁴)

    (Sign in atomx potential-energy convention: WK is opposite to Uehling,
    hence *positive* here — it is a small repulsive correction that
    partially cancels Uehling at short range for heavy Z.)

    Returns
    -------
    float
        E_WK = Σ_a q_a · ⟨a | V_WK | a⟩  in Hartree.  Typically ≪ E_VP
        (Uehling); it is a ~1 % correction for Z > 80 and negligible for
        light atoms.
    """
    from atomx.corrections.qedmod.flambaum_ginges import V_wichmann_kroll

    return _short_range_expectation(
        orbitals, bsp_data, lambda r: V_wichmann_kroll(r, Z))


def compute_qed_corrections(
    orbitals, bsp_data, Z,
    *,
    se_method=DEFAULT_SE_METHOD,
    include_wichmann_kroll=False,
):
    """Compute all QED corrections (VP + SE), optionally with higher-order VP.

    Parameters
    ----------
    orbitals : list of (P, Q, occ, kappa, label) tuples
    bsp_data : dict
    Z : int
    se_method : str, keyword-only
        Self-energy backend.  Default ``"mohr"`` (atomx ≤ 0.2 behaviour).
        See :mod:`atomx.corrections.qedmod` for alternatives.
    include_wichmann_kroll : bool, keyword-only
        When ``True``, add the simplified Wichmann–Kroll higher-order VP
        term (Flambaum–Ginges eq. 12) to the VP budget and return it as
        the extra key ``E_WK`` in the result dict.  Default ``False``.

    Returns
    -------
    dict with keys: ``E_VP``, ``E_SE``, ``E_QED_total``, ``se_method``,
    and ``E_WK`` when ``include_wichmann_kroll=True``.
    """
    E_VP = compute_vp_energy(orbitals, bsp_data, Z)
    resolved_se_method = resolve_se_method(se_method, Z)
    E_SE = compute_se_energy(orbitals, Z, bsp_data=bsp_data, method=resolved_se_method)

    result = {
        "E_VP": E_VP,
        "E_SE": E_SE,
        "se_method": resolved_se_method,
        "qed_backend": se_method,
    }

    total = E_VP + E_SE
    if include_wichmann_kroll:
        E_WK = compute_wk_vp_energy(orbitals, bsp_data, Z)
        result["E_WK"] = E_WK
        total += E_WK

    result["E_QED_total"] = total
    return result
