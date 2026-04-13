"""B-spline knot sequence generation.

Supports exponential and DBSR_HF-style semi-exponential grids.
"""

import numpy as np
import numba as nb


def make_knots(ns: int, ks: int, r_min: float, r_max: float,
               *, mode: str = "exponential") -> np.ndarray:
    """Build a B-spline knot sequence with boundary multiplicities.

    Parameters
    ----------
    ns : int
        Number of B-splines (basis functions).
    ks : int
        Order of B-splines (polynomial degree = ks - 1).
    r_min : float
        First interior knot (smallest nonzero radius).
    r_max : float
        Boundary radius (box size).

    Returns
    -------
    t : ndarray, shape (ns + ks,)
    """
    if ns < ks + 1:
        raise ValueError(f"Need ns >= ks + 1, got ns={ns}, ks={ks}")
    if r_min <= 0 or r_max <= r_min:
        raise ValueError(f"Need 0 < r_min < r_max, got {r_min}, {r_max}")

    return make_exponential_knots(ns, ks, r_min, r_max)


def make_exponential_knots(ns: int, ks: int, r_min: float,
                           r_max: float) -> np.ndarray:
    """Pure exponential knot sequence (DBSR_HF grid_type=2)."""
    nv = ns - ks + 1
    he = np.exp(np.log(r_max / r_min) / (nv - 1))

    t = np.zeros(ns + ks)
    t[ks] = r_min
    for i in range(ks + 1, ns + 1):
        t[i] = t[i - 1] * he
    t[ns:ns + ks] = r_max

    return t


def make_dbsr_knots(ks: int, Z: int, r_max: float = 50.0,
                    hi: float = 0.25, he: float = 0.25,
                    h_max: float = 1.0,
                    nuclear: str = "fermi") -> tuple:
    """DBSR_HF mkgrid_01 semi-exponential grid.

    Exactly reproduces the DBSR_HF grid with ns≈113 for typical atoms.
    The key: Fermi nucleus gives very small initial step (~5e-6 Bohr),
    requiring ~50 exponential steps to reach the hmax region.

    Parameters
    ----------
    ks : int
        B-spline order (default 9).
    Z : int
        Nuclear charge.
    r_max : float
        Box radius (default 50 Bohr).
    hi, he, h_max : float
        Grid parameters (DBSR_HF defaults: 0.25, 0.25, 1.0).
    nuclear : str
        "fermi" (default, matches DBSR_HF) or "point".

    Returns
    -------
    t : ndarray, shape (ns + ks,)
    ns : int
    """
    # Initial step depends on nuclear model (DBSR_HF mkgrid_01 lines 679-691)
    if nuclear == "fermi":
        A = int(round(2.5 * max(Z, 1)))
        a_fermi_fm = 0.5  # Fermi skin thickness in fm
        bohr_fm = 5.29177e4  # Bohr in fm
        a_fermi = a_fermi_fm / bohr_fm
        ts = 4.0 * np.log(3.0) * a_fermi
        hs = ts * hi
    else:
        hs = hi / max(Z, 1)

    ht = 1.0 + he

    # Phase 1: ml = ks uniform intervals of size hs
    ml = ks
    breakpoints = [hs * i for i in range(1, ml + 1)]

    # Phase 2: multiplicative growth r_next = r_prev * (1+he)
    r = breakpoints[-1]
    while r < r_max:
        r_next = r * ht
        if h_max > 0 and (r_next - r) > h_max:
            break
        r = r_next
        if r >= r_max:
            break
        breakpoints.append(r)

    # Phase 3: constant step h_max
    if h_max > 0 and r < r_max:
        while r < r_max:
            r += h_max
            if r >= r_max:
                break
            breakpoints.append(r)

    # Extend to tmax >= rmax
    if not breakpoints or breakpoints[-1] < r_max:
        breakpoints.append(r_max)

    tmax = breakpoints[-1]
    nv = len(breakpoints)
    ns = nv + ks - 1

    t = np.zeros(ns + ks)
    for i, bp in enumerate(breakpoints):
        t[ks + i] = bp
    t[ns:ns + ks] = tmax

    return t, ns


def make_semi_exponential_knots(ns_target: int, ks: int, Z: int,
                                r_max: float, *,
                                hi: float = 0.25,
                                he: float = 0.25,
                                h_max: float = 0.0) -> np.ndarray:
    """Semi-exponential knot sequence (legacy interface)."""
    t, ns = make_dbsr_knots(ks, Z, r_max, hi, he, h_max, nuclear="point")
    return t
