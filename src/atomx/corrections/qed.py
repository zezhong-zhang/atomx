"""QED corrections: vacuum polarization (Uehling) and self-energy.

Both computed perturbatively from converged DHF orbitals.

Vacuum polarization
-------------------
``V_VP(r) = -(2Zα)/(3π) I(2cr) / r`` with the Uehling kernel

    ``I(x) = ∫_1^∞ e^{-xt} (1 + 1/(2t²)) √(t²-1)/t² dt``

for a point nucleus.  ``I`` is **not** ``K_1``: K_1 is only the large-x
asymptote up to an O(1) factor, and using it inflates ``V_VP`` by a
factor rising to 160 at small x (see :func:`_uehling_kernel`).  The
kernel is tabulated once by adaptive quadrature and evaluated by a
spline in ``log I``; relative accuracy is verified against
``scipy.integrate.quad`` in ``tests/test_qed_uehling.py``.
Higher-order vacuum-polarization (Wichmann–Kroll, Källén–Sabry) is
handled by :func:`wichmann_kroll_potential`; see the roadmap.

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

from functools import lru_cache

import numpy as np
from scipy.special import k1 as bessel_k1

from atomx.constants import ALPHA, C
from atomx.corrections.qedmod import get_backend

# ---------------------------------------------------------------------------
# Vacuum Polarization (Uehling potential, 2nd order)
# ---------------------------------------------------------------------------

#: Upper end of the tabulated Uehling kernel.  ``I(80) ≈ 4e-37`` and the
#: prefactor supplies another ``1/r``, so everything beyond this is far
#: below double-precision relevance for any atomic orbital; the
#: large-``x`` branch below is used instead of the table there.
_UEHLING_X_MAX = 80.0

#: Lower end of the tabulated kernel.  ``x = 2 c r``, so this corresponds
#: to ``r ≈ 4e-11`` Bohr — eight decades below the innermost knot of any
#: grid the package builds.
_UEHLING_X_MIN = 1.0e-10


def _uehling_integrand(t, x):
    """Uehling kernel integrand ``e^{-xt}(1 + 1/(2t²))√(t²-1)/t²``."""
    return np.exp(-x * t) * (1.0 + 0.5 / (t * t)) * np.sqrt(t * t - 1.0) / (t * t)


def _uehling_I_exact(x):
    """Reference ``I(x)`` by adaptive quadrature (used to build the table)."""
    from scipy.integrate import quad

    value, _err = quad(
        _uehling_integrand, 1.0, np.inf, args=(float(x),),
        epsabs=1e-300, epsrel=1e-13, limit=200,
    )
    return value


@lru_cache(maxsize=1)
def _uehling_kernel_spline():
    """Cached cubic spline of ``log I(x) + x`` against ``log x``.

    ``I`` falls from ``≈ 23`` at ``x = 1e-10`` to ``≈ 4e-37`` at
    ``x = 80``, i.e. over 37 decades.  Splining ``log I`` directly would
    need a very fine grid at large ``x`` (there ``log I ≈ −x``, unit
    slope); factoring out the exponential leaves
    ``g(u) = log I(e^u) + e^u``, which varies only from ``≈ 3.1`` to
    ``≈ −31`` across the whole range and is smooth in ``u = log x``, so
    ~900 nodes give near machine precision everywhere.

    Building the table costs ~0.5 s once per process.
    """
    from scipy.interpolate import CubicSpline

    log_x = np.linspace(np.log(_UEHLING_X_MIN), np.log(_UEHLING_X_MAX), 900)
    xs = np.exp(log_x)
    vals = np.array([_uehling_I_exact(x) for x in xs])
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0.0):
        raise RuntimeError("Uehling kernel table failed to build")
    return CubicSpline(log_x, np.log(vals) + xs)


def _uehling_kernel(x):
    """Uehling kernel ``I(x)`` for ``x = 2cr`` (scalar or ndarray).

    ``I`` is the Bickley–Naylor combination
    ``(1 + x²/12)K_0 − x(5/6 + x²/12)K_1 + x(3/4 + x²/12)Ki_1``; here it
    is evaluated from a spline table validated against
    ``scipy.integrate.quad`` (see ``tests/test_qed_uehling.py``).

    Below ``x = _UEHLING_X_MIN`` the kernel is extrapolated from the
    table's innermost node; ``I`` varies only logarithmically there and
    the ``1/r`` prefactor drives the potential to zero, so the
    extrapolation error is physically irrelevant.  Above
    ``x = _UEHLING_X_MAX`` the leading large-``x`` behaviour
    ``I(x) → (3/2)K_1(x)/x`` is used (``I < 1e-37`` there).
    """
    x_arr = np.asarray(x, dtype=float)
    scalar = x_arr.ndim == 0
    xf = np.atleast_1d(x_arr).astype(float)

    out = np.zeros_like(xf)
    big = xf > _UEHLING_X_MAX
    mid = ~big

    if np.any(mid):
        spline = _uehling_kernel_spline()
        xm = np.clip(xf[mid], _UEHLING_X_MIN, _UEHLING_X_MAX)
        out[mid] = np.exp(spline(np.log(xm)) - xm)
    if np.any(big):
        xb = xf[big]
        out[big] = 1.5 * bessel_k1(xb) / xb

    return float(out[0]) if scalar else out

    return float(out[0]) if scalar else out


def uehling_potential(r, Z):
    """Uehling vacuum polarization potential V_VP(r) for a point nucleus.

    ``V_VP(r) = -(2Zα)/(3π) · I(2cr) / r``

    where ``c = 1/α ≈ 137.036`` and ``I`` is the Uehling kernel
    (see :func:`_uehling_kernel`).

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

    Notes
    -----
    The kernel is *positive*, so ``V_VP`` is attractive (it increases the
    binding).  Its range is the (reduced) Compton scale
    ``1/(2c) ≈ 0.0036`` Bohr; beyond ``r ≳ 0.3`` Bohr the potential is
    below ``1e-30`` Ha.
    """
    r_arr = np.asarray(r, dtype=float)
    prefactor = -2.0 * Z * ALPHA / (3.0 * np.pi)
    x = 2.0 * C * r_arr

    with np.errstate(divide="ignore", invalid="ignore"):
        V = prefactor * _uehling_kernel(x) / r_arr

    if r_arr.ndim == 0:
        return float(V) if r_arr > 0 else 0.0
    V = np.asarray(V, dtype=float)
    V[r_arr <= 0] = 0.0
    return V


#: Dense auxiliary log-grid used to integrate nuclear-range kernels
#: against B-spline orbitals.  25k points over 9 decades ≈ 2800 points
#: per decade — trapezoid error < 1e-6 relative on the Uehling folding
#: (checked against a 400k-point evaluation).
_DENSE_KERNEL_POINTS = 25_000
_DENSE_KERNEL_R_MIN = 1.0e-9


def short_range_radial_matrix(orbitals, bsp_data, V_of_r, *, diagonal_only=False):
    """Full <a|V|b> matrix for ``(P,Q,kappa)`` orbitals and local V(r).

    The Uehling kernel ``I(2cr)/r`` lives on the Compton scale
    ``r ≲ 1/(2c) ≈ 0.004`` Bohr and behaves like ``log(r)/r`` toward the
    origin; the SCF basis' own Gauss nodes under-resolve it whenever the
    innermost knots are wide on that scale (He physics grid: 8e-3
    relative error on E_VP).  For B-spline workspaces we therefore
    evaluate the spline expansion EXACTLY on a dense auxiliary log grid
    and integrate there.  Radial-grid (shooting) workspaces keep their
    native ``sum(f · difr)`` sum — Salvat's SGRID is exponentially dense
    at the origin and self-converged to ~1e-7 on this integral.
    """
    from atomx.corrections.workspace import (
        eval_orbital_pair,
        is_grid_workspace,
        workspace_arrays,
    )

    if not orbitals:
        return np.zeros(0) if diagonal_only else np.zeros((0, 0))
    gr, grw = workspace_arrays(bsp_data)
    if is_grid_workspace(bsp_data) or "t" not in bsp_data:
        values = []
        for P, Q, _kappa in orbitals:
            fP, fQ = eval_orbital_pair(P, Q, bsp_data)
            values.append((fP.ravel(), fQ.ravel()))
        return _local_radial_contraction(
            orbitals, values, (grw * V_of_r(gr)).ravel(), diagonal_only)

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
    values = []
    for P, Q, _kappa in orbitals:
        splP = BSpline(tp, np.asarray(P[:nsp], dtype=float), ksp - 1,
                       extrapolate=False)
        splQ = BSpline(t, np.asarray(Q[:nsq], dtype=float), ksq - 1,
                       extrapolate=False)
        fP = np.nan_to_num(splP(r))
        fQ = np.nan_to_num(splQ(r))
        values.append((fP, fQ))
    return _local_radial_contraction(orbitals, values, w * V_of_r(r), diagonal_only)


def _local_radial_contraction(orbitals, values, weights, diagonal_only):
    p = np.asarray([pq[0] for pq in values])
    q = np.asarray([pq[1] for pq in values])
    if diagonal_only:
        return np.sum(weights * (p*p + q*q), axis=1)
    matrix = (p * weights) @ p.T + (q * weights) @ q.T
    kappas = np.asarray([orb[2] for orb in orbitals])
    matrix[kappas[:, None] != kappas[None, :]] = 0.0
    return 0.5 * (matrix + matrix.T)


def _short_range_expectation(orbitals, bsp_data, V_of_r):
    """Occupation-weighted diagonal for a determinant or incoherent ensemble."""
    diagonal = short_range_radial_matrix(
        [(p, q, k) for p, q, _, k, _ in orbitals], bsp_data, V_of_r, diagonal_only=True)
    return float(np.dot([occ for _, _, occ, _, _ in orbitals], diagonal))


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


def self_energy_radial_matrix(orbitals, Z, bsp_data, *, method=DEFAULT_SE_METHOD):
    """Matrix for (P,Q,kappa,label) orbitals; diagonal fallback for table models."""
    backend = get_backend(resolve_se_method(method, Z))
    matrix_builder = getattr(backend, "radial_matrix", None)
    if callable(matrix_builder):
        return matrix_builder([(p, q, k) for p, q, k, _ in orbitals], bsp_data, Z)
    diagonal = [backend.compute([(p, q, 1., k, label)], bsp_data, Z).E_SE
                for p, q, k, label in orbitals]
    return np.diag(diagonal)


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
    include_vp=True,
):
    """Compute all QED corrections (VP + SE), optionally with higher-order VP.

    Parameters
    ----------
    orbitals : list of (P, Q, occ, kappa, label) tuples
    bsp_data : dict
    Z : int
    se_method : str, keyword-only
        Self-energy backend.  Default ``"mohr"`` (atomx ≤ 0.2 behaviour).
        See :mod:`atomx.corrections.qedmod` for alternatives. ``None``
        disables self-energy.
    include_vp : bool, keyword-only
        Include the Uehling term (default True), independently of SE/WK.
    include_wichmann_kroll : bool, keyword-only
        When ``True``, add the simplified Wichmann–Kroll higher-order VP
        term (Flambaum–Ginges eq. 12) to the VP budget and return it as
        the extra key ``E_WK`` in the result dict.  Default ``False``.

    Returns
    -------
    dict with ``E_QED_total`` and only the enabled component keys:
    ``E_VP``, ``E_SE``/``se_method``, and ``E_WK``.
    """
    result = {}
    total = 0.0
    if include_vp:
        result["E_VP"] = compute_vp_energy(orbitals, bsp_data, Z)
        total += result["E_VP"]
    if se_method is not None:
        resolved_se_method = resolve_se_method(se_method, Z)
        result["E_SE"] = compute_se_energy(
            orbitals, Z, bsp_data=bsp_data, method=resolved_se_method)
        result["se_method"] = resolved_se_method
        result["qed_backend"] = se_method
        total += result["E_SE"]
    if include_wichmann_kroll:
        E_WK = compute_wk_vp_energy(orbitals, bsp_data, Z)
        result["E_WK"] = E_WK
        total += E_WK

    result["E_QED_total"] = total
    return result
