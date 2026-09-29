"""Continuum Dirac wavefunctions for photoionization cross-sections.

For photoionization, we need the continuum wavefunction at energy E > 0
(electron energy above the ionization threshold). The continuum state
satisfies the same Dirac equation but with positive energy, and is
normalized to ``delta(E-E')`` (not L^2 normalizable).

In the B-spline basis, the continuum solution is obtained by solving
the Dirac equation at a specified positive energy, then extracting
the phase shift from the asymptotic form of the wavefunction.

For photoionization cross-section σ(ω):
  σ(ω) = (4π²αω/3) Σ_{κ_f} |<f|r|i>|² × (2j_f+1)

where |f> is the continuum state, |i> the bound orbital, and the
matrix element involves the dipole operator.
"""

from collections import OrderedDict
from hashlib import blake2b

import numpy as np
from scipy import linalg

from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices



def continuum_energy_normalization_factor(energy, bsp_data, equation=None):
    """Box-normalized pseudo-state -> delta-energy normalization factor.

    ``solve_continuum`` diagonalizes the Dirac equation in a finite radial box,
    so the returned eigenvectors have unit box norm.  Scattering amplitudes
    require continuum waves normalized to ``delta(E-E')``.  Discrete box
    states obey ``<n|m> = delta_nm`` and continuum states obey
    ``<E|E'> = delta(E-E')``; therefore ``|E_n> ~= sqrt(dn/dE) |n>``.
    For a large free hard-wall box this density is

        rho(E) ~= R / (pi * k),

    so this helper returns the analytic fallback ``sqrt(rho)``.  Coulomb and
    finite-box phase shifts change ``dn/dE`` appreciably near threshold;
    ``solve_continuum(..., energy_normalized=True)`` therefore uses the local
    pseudo-spectrum spacing instead of this asymptotic fallback.

    ``equation`` selects the dispersion relation for ``k``: the Dirac
    pseudo-spectrum oscillates with the relativistic
    ``k = sqrt(2E(1 + alpha^2 E/2))``, so that is the default here; pass
    :class:`atomx.physics.Schrodinger` for a non-relativistic box
    (``k = sqrt(2E)``).
    """
    if energy <= 0:
        return 0.0
    if equation is None:
        from atomx.physics import Dirac

        equation = Dirac()
    R_box = float(bsp_data["t"][-1])
    k = float(equation.continuum_wave_number(float(energy)))
    rho = R_box / (np.pi * max(k, 1e-30))
    return float(np.sqrt(rho))


def _continuum_energy_normalization_from_spectrum(
    positive_energies,
    idx,
    fallback_density=None,
):
    """Return ``sqrt(dn/dE)`` from a bounded local continuum stencil.

    A single adjacent spacing is too fragile near spectrum boundaries or
    unresolved pseudo-level clusters: it can turn a numerical near-degeneracy
    into an arbitrarily large continuum amplitude.  We therefore estimate the
    level density from the widest available five-state local stencil centered on
    ``idx``.  Duplicate, unordered, or too-sparse spectra fail closed unless an
    analytic fallback density is supplied by the caller.  When supplied, the
    fallback is also an upper bound on the local density; for the B-spline
    continuum this is the free hard-wall density ``R/(pi*k)``.
    """
    energies = np.asarray(positive_energies, dtype=float)
    n = len(energies)
    if idx < 0 or idx >= n:
        raise IndexError("continuum spectrum index out of range")
    has_fallback = (
        fallback_density is not None
        and np.isfinite(fallback_density)
        and fallback_density > 0.0
    )
    if n < 3:
        if has_fallback:
            return float(np.sqrt(fallback_density))
        raise ValueError("at least three positive-energy states are required")
    if np.any(np.diff(energies) <= 0.0):
        raise ValueError("positive continuum spectrum must be strictly ordered")

    start = max(0, min(idx - 2, n - 5))
    stop = min(n, max(idx + 3, start + 5))
    start = max(0, stop - 5)
    stencil = energies[start:stop]
    span = float(stencil[-1] - stencil[0])
    scale = max(float(abs(stencil[-1])), float(abs(stencil[0])), 1.0)
    if not np.isfinite(span) or span <= np.sqrt(np.finfo(float).eps) * scale:
        if has_fallback:
            return float(np.sqrt(fallback_density))
        raise ValueError("continuum spectrum stencil is numerically unresolved")

    density = float((len(stencil) - 1) / span)
    if has_fallback:
        density = min(density, float(fallback_density))
    return float(np.sqrt(density))


def _continuum_spectrum(kappa, Z, bsp_data, V_scf_PP, V_scf_QQ):
    """Energy-independent eigensystem, owned by the radial workspace.

    Content fingerprints include every input to Hamiltonian assembly, so
    in-place potential or grid edits invalidate it. At most eight partial
    waves/potential combinations survive, and deleting the workspace frees
    its spectra. Returned arrays are internal read-only cache data.
    """
    digest = blake2b(digest_size=20)
    for key in ("ns", "nv", "ks", "ksp", "ksq", "nsp", "nsq", "t",
                "gr", "grw", "bsp_p", "bsp_q", "bsp_p_d", "bsp_q_d"):
        value = bsp_data.get(key)
        digest.update(key.encode())
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode())
    geometry = digest.digest()
    for value in (V_scf_PP, V_scf_QQ):
        if value is None:
            digest.update(b"None")
        else:
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
    key = (int(kappa), float(Z), digest.digest())
    cache = bsp_data.setdefault("_continuum_spectra", OrderedDict())
    if bsp_data.get("_continuum_geometry") != geometry:
        cache.clear()
        bsp_data["_continuum_geometry"] = geometry
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    ns = bsp_data["ns"]
    ms = 2 * ns

    H, S = build_dirac_matrices(ns, kappa, Z, bsp_data)

    # Add SCF potential if provided
    if V_scf_PP is not None:
        H[:ns, :ns] += V_scf_PP
    if V_scf_QQ is not None:
        H[ns:, ns:] += V_scf_QQ

    # Solve full eigenvalue problem (including both bound and continuum)
    nb = 1  # boundary removal
    keep_p = list(range(nb, ns - nb))
    keep_q = list(range(ns + nb, ms - nb))
    keep = keep_p + keep_q

    H_red = H[np.ix_(keep, keep)]
    S_red = S[np.ix_(keep, keep)]

    eigenvalues, eigenvectors = linalg.eigh(H_red, S_red)

    positive = eigenvalues > 0.0
    energies = eigenvalues[positive]
    vectors = eigenvectors[:, positive]
    if len(energies) == 0:
        raise ValueError("no positive-energy continuum pseudo-states found")
    energies.flags.writeable = vectors.flags.writeable = False
    result = (energies, vectors, keep_p, keep_q)
    cache[key] = result
    while len(cache) > 8:
        cache.popitem(last=False)
    return result


def solve_continuum(kappa, energy, Z, bsp_data, V_scf_PP=None, V_scf_QQ=None,
                    energy_normalized=False):
    """Find the continuum wavefunction at given positive energy.

    Solves the Dirac equation (H - E*S)c = 0 where E > 0 is the
    electron kinetic energy. The solution closest to the target energy
    is extracted from the discretized eigenvalue spectrum.

    Parameters
    ----------
    kappa : int
        Angular momentum quantum number of the continuum electron.
    energy : float
        Kinetic energy of the continuum electron in Hartree (> 0).
    Z : int
        Nuclear charge (or effective charge of the residual ion).
    bsp_data : dict
        B-spline data.
    V_scf_PP, V_scf_QQ : ndarray, optional
        Self-consistent potential from bound electrons (Hartree + exchange).
        If None, uses bare nuclear potential.
    energy_normalized : bool, optional
        If true, multiply the box-normalized pseudo-state by the local
        pseudo-spectrum ``sqrt(dn/dE)`` factor so it approximates delta-energy
        normalization.  Scattering and Auger matrix elements should set this.

    Returns
    -------
    P : ndarray, shape (ns,)
        Large component B-spline coefficients (box-normalized).
    Q : ndarray, shape (ns,)
        Small component B-spline coefficients.
    E_actual : float
        Nearest discretized eigenvalue to target energy.
    """
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("continuum energy must be finite and positive")
    ns = bsp_data["ns"]
    positive_evals, positive_evecs, keep_p, keep_q = _continuum_spectrum(
        kappa, Z, bsp_data, V_scf_PP, V_scf_QQ)

    # Find closest positive-energy pseudo-state to the target continuum energy.
    idx = int(np.argmin(np.abs(positive_evals - energy)))
    E_actual = positive_evals[idx]

    # Map back to full vectors
    vec_red = positive_evecs[:, idx]
    P = np.zeros(ns)
    Q = np.zeros(ns)
    for ii, k in enumerate(keep_p):
        P[k] = vec_red[ii]
    for ii, k in enumerate(keep_q):
        Q[k - ns] = vec_red[len(keep_p) + ii]

    if energy_normalized:
        fallback_density = continuum_energy_normalization_factor(
            float(E_actual), bsp_data) ** 2
        norm = _continuum_energy_normalization_from_spectrum(
            positive_evals, idx, fallback_density=fallback_density)
        P *= norm
        Q *= norm

    return P, Q, E_actual


def dipole_matrix_element(P_bound, Q_bound, P_cont, Q_cont, bsp_data):
    """Compute the electric dipole matrix element <f|r|i>.

    For photoionization, the dipole operator in length gauge is r.
    The radial matrix element is:
        <f|r|i> = ∫ [P_f(r) * P_i(r) + Q_f(r) * Q_i(r)] * r dr

    Parameters
    ----------
    P_bound, Q_bound : ndarray
        Bound orbital B-spline coefficients.
    P_cont, Q_cont : ndarray
        Continuum orbital B-spline coefficients.
    bsp_data : dict

    Returns
    -------
    float
        Dipole matrix element.
    """
    # Use the kinetic-balance-aware helpers so P uses (bsp_p, ksp) and
    # Q uses (bsp_q, ksq) — required when ksp ≠ ksq (RKB default).
    from atomx.numerics.bspline.evaluate import eval_P, eval_Q
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]

    fi_P = eval_P(P_bound, bsp_data)
    fi_Q = eval_Q(Q_bound, bsp_data)
    ff_P = eval_P(P_cont, bsp_data)
    ff_Q = eval_Q(Q_cont, bsp_data)

    # <f|r|i> = ∫ (P_f P_i + Q_f Q_i) r dr
    integrand = (ff_P * fi_P + ff_Q * fi_Q) * gr
    return np.sum(grw * integrand)
