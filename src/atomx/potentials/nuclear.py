"""Nuclear charge distribution models (DBSR_HF convention).

Three models available:
  "point"   — V(r) = -Z/r  (singular at origin)
  "fermi"   — Fermi distribution ρ(r) = ρ₀/(1+exp((r-c)/a))  (DBSR_HF default)
  "uniform" — Uniform sphere of equivalent RMS radius

DBSR_HF uses Fermi distribution by default. The Fermi parameters are
shared between grid (knots.py) and potential via `fermi_params()`.
"""

import numpy as np

from atomx.constants import BOHR_RADIUS_FM

# ── Tabulated RMS charge radii (Angeli & Marinova, 2013) ────────────
# Values from DBSR_HF atoms_par module (fm). Empirical formula used for Z>10.

_RRMS_TABLE_FM = {
    1: 0.8783,    # H
    2: 1.6755,    # He
    3: 2.5890,    # Li
    4: 2.5190,    # Be
    5: 2.4277,    # B
    6: 2.4702,    # C
    7: 2.5582,    # N
    8: 2.6991,    # O
    9: 2.8976,    # F
    10: 3.0055,   # Ne
}

# Default mass numbers (DBSR_HF atoms_par)
_MASS_TABLE = {
    1: 1, 2: 4, 3: 7, 4: 9, 5: 11, 6: 12, 7: 14, 8: 16, 9: 19, 10: 20,
}


def _get_rrms_fm(Z, A=None):
    """RMS charge radius in fm, using tabulated values for light elements."""
    if Z in _RRMS_TABLE_FM:
        return _RRMS_TABLE_FM[Z]
    if A is None:
        A = _MASS_TABLE.get(Z, int(round(2.5 * max(Z, 1))))
    rrms = 0.836 * A ** (1.0 / 3.0) + 0.570
    return rrms


def _get_mass(Z, A=None):
    """Mass number, using tabulated values for light elements."""
    if A is not None:
        return A
    return _MASS_TABLE.get(Z, int(round(2.5 * max(Z, 1))))


def nuclear_profile(Z, nuclear="fermi", A=None):
    """Describe the actual charge model separately from the recoil mass.

    A selects an empirical radius, not an isotope measurement. Light-element
    Fermi models retain their element reference radius and historical clamp.
    """
    if A is not None and (isinstance(A, bool) or int(A) != A or A < Z):
        raise ValueError("nuclear mass number must be an integer A >= Z")
    if nuclear == "point":
        return dict(model="point", radius_mass_number=None,
                    requested_mass_number=A, rms_radius_fm=0.0,
                    radius_source="point_nucleus")
    if nuclear not in {"fermi", "uniform"}:
        raise ValueError(f"Unknown nuclear model: {nuclear!r}")
    if nuclear == "uniform":
        mass_number = A if A is not None else int(round(2.5 * max(Z, 1)))
        reference = 0.836 * mass_number ** (1.0 / 3.0) + 0.570
    else:
        mass_number = _get_mass(Z, A)
        reference = _get_rrms_fm(Z, mass_number)
    effective = max(2.0, reference) if nuclear == "fermi" else reference
    profile = dict(model=nuclear, radius_mass_number=mass_number,
                   requested_mass_number=A, rms_radius_fm=effective,
                   reference_rms_radius_fm=reference,
                   radius_source=("element_reference" if nuclear == "fermi" and Z in _RRMS_TABLE_FM
                                  else "empirical_0.836_A_cuberoot_plus_0.570_fm"),
                   radius_clamped=effective != reference)
    if nuclear == "fermi":
        a, c = fermi_params(Z, A)
        profile.update(diffuseness_bohr=float(a), half_density_radius_bohr=float(c))
    return profile


# ── Shared Fermi parameters (used by both grid and potential) ────────

def fermi_params(Z, A=None):
    """Compute DBSR_HF Fermi nuclear parameters (a, c) in atomic units.

    Matches DBSR_HF: Read_nuclear -> t_fermi, a_fermi, GETCPR(rrms, a, c).
    All intermediate calculations in fm (as in DBSR_HF), convert to bohr at end.

    Parameters
    ----------
    Z : int
        Nuclear charge.
    A : int, optional
        Mass number. Default: tabulated or round(2.5 * Z).

    Returns
    -------
    a_au : float
        Skin thickness in Bohr.
    c_au : float
        Half-density radius in Bohr.
    """
    A = _get_mass(Z, A)

    # DBSR_HF defaults: t_fermi=2.30 fm -> a = t/(4*ln3)  in fm
    t_fermi_fm = 2.30
    a_fm = t_fermi_fm / (4.0 * np.log(3.0))  # = 0.52350... fm

    # RMS charge radius (fm): tabulated for Z<=10, empirical for Z>10
    rrms_fm = _get_rrms_fm(Z, A)

    # DBSR_HF clamps rrms >= 2.0 fm for Fermi model (GRASP compatibility)
    if rrms_fm < 2.0:
        rrms_fm = 2.0

    # c from rrms and a via bisection (DBSR_HF GETCPR, works in fm)
    c_fm = _getcpr(rrms_fm, a_fm)

    # Convert fm -> bohr
    fm_per_bohr = BOHR_RADIUS_FM
    a_au = a_fm / fm_per_bohr
    c_au = c_fm / fm_per_bohr

    return a_au, c_au


def fermi_grid_step(Z, hi=0.25):
    """Initial grid step for Fermi nuclear model (shared with knots.py).

    hs = 4*log(3) * a_fermi * hi

    Parameters
    ----------
    Z : int
    hi : float
        DBSR_HF grid parameter (default 0.25).

    Returns
    -------
    hs : float
        Initial step in Bohr.
    """
    a_au, _ = fermi_params(Z)
    ts = 4.0 * np.log(3.0) * a_au
    return ts * hi


def _skfun(k, x):
    """Polylogarithm-like function from DBSR_HF.

    S_k(x) = Σ_{n=1}^∞ (-1)^n exp(n*x) / n^k
    """
    if x < -500.0:
        return 0.0
    eps = 10.0 * np.finfo(float).eps
    base = -np.exp(x)
    dnum = base
    result = base
    n = 1
    while True:
        n += 1
        dnum *= base
        delta = dnum / float(n) ** k
        result += delta
        if abs(delta) <= eps * abs(result):
            break
        if n > 1000:
            break
    return result


def _estrms(a_par, c_par):
    """Estimate RMS radius from Fermi parameters (DBSR_HF ESTRMS).

    Based on Parpia & Mohanty, Phys Rev A 46, 3735 (1992).
    """
    A = a_par / c_par
    P = np.pi * A
    X = -c_par / a_par
    DN = 1.0 + 10.0/3.0 * P**2 + 7.0/3.0 * P**4 - 120.0 * A**5 * _skfun(5, X)
    DD = 1.0 + P**2 - 6.0 * A**3 * _skfun(3, X)
    return c_par * np.sqrt(3.0 * DN / 5.0 / DD)


def _getcpr(rrms, a_par):
    """Compute Fermi c-parameter from RMS radius and a-parameter.

    Bisection method matching DBSR_HF GETCPR subroutine exactly.
    """
    accy = 2.0 * np.finfo(float).eps

    # Bracket c with lower and upper limits
    cpmin = rrms
    while _estrms(a_par, cpmin) > rrms:
        cpmin *= 0.5

    cpmax = rrms
    while _estrms(a_par, cpmax) < rrms:
        cpmax *= 2.0

    # Bisection
    for _ in range(200):
        cptry = 0.5 * (cpmax + cpmin)
        rmstry = _estrms(a_par, cptry)
        if rmstry > rrms:
            cpmax = cptry
        else:
            cpmin = cptry
        if (cpmax - cpmin) / (cpmax + cpmin) <= accy:
            break
        if abs(rmstry - rrms) / rrms <= accy:
            break

    return cptry


# ── Nuclear potential functions ──────────────────────────────────────

def make_nuclear_potential(Z, nuclear="fermi", A=None):
    """Create nuclear potential function V(r).

    Parameters
    ----------
    Z : int
        Nuclear charge.
    nuclear : str
        "point", "fermi", or "uniform".
    A : int, optional
        Mass number for fermi/uniform models.

    Returns
    -------
    V_func : callable or None
        V(r) function. None means use default -Z/r (point nucleus).
    """
    if nuclear == "point":
        return None  # integrals.py handles -Z/r natively
    elif nuclear == "fermi":
        return _make_fermi_potential(Z, A)
    elif nuclear == "uniform":
        return _make_uniform_potential(Z, A)
    else:
        raise ValueError(f"Unknown nuclear model: {nuclear!r}. "
                         f"Use 'point', 'fermi', or 'uniform'.")


def _make_fermi_potential(Z, A=None):
    """Analytical Fermi nuclear potential (Parpia & Mohanty, PRA 46, 3735, 1992).

    Exact closed-form solution of Poisson's equation for the Fermi density:
        rho(r) = rho_0 / (1 + exp((r - c) / a))

    Uses polylogarithm functions S_k for machine-precision evaluation.
    This is the same formula as DBSR_HF's V_nuclear subroutine.
    """
    a, c = fermi_params(Z, A)

    # Pre-compute constants (all in atomic units)
    ac = a / c
    ac2 = ac * ac
    ac3 = ac * ac2
    p = np.pi ** 2 * ac2
    s = 6.0 * ac3 * _skfun(3, -c / a)   # 6*(a/c)^3 * S_3(-c/a)
    N = 1.0 + p - s                       # normalization

    def V_func(r):
        if r <= 0.0:
            # V(0): limit as r->0
            return -Z * (3.0 + p) / (2.0 * N * c)

        rc = r / c

        if r <= c:
            ra = (r - c) / a
            s2 = ac2 * _skfun(2, ra)
            s3 = ac3 * _skfun(3, ra)
            V = (3.0 - rc * rc + p) / 2.0 - 3.0 * s2 - s / rc + 6.0 / rc * s3
            return -Z * V / (N * c)
        else:
            ra = (c - r) / a
            s2 = ac2 * _skfun(2, ra)
            s3 = ac3 * _skfun(3, ra)
            V = N + 6.0 * s3 + 3.0 * rc * s2
            return -Z / r * (V / N)

    return V_func


def _make_uniform_potential(Z, A=None):
    """Uniform charged sphere potential.

    V(r) = -Z/(2R) * (3 - r²/R²)  for r ≤ R
    V(r) = -Z/r                     for r > R
    """
    if A is None:
        A = int(round(2.5 * max(Z, 1)))
    rrms_fm = 0.836 * A ** (1.0 / 3.0) + 0.570
    R_rms_au = rrms_fm / BOHR_RADIUS_FM
    R = np.sqrt(5.0 / 3.0) * R_rms_au  # equivalent uniform sphere radius

    def V_func(r):
        if r <= R:
            return -Z / (2.0 * R) * (3.0 - (r / R) ** 2)
        return -Z / r

    return V_func

