"""B-spline knot sequence generation.

Unified physics-based grid: semi-exponential with Z-dependent r_min,
exponential growth, and constant-step outer region. Works for all Z.
"""

import numpy as np


def make_knots(ns: int, ks: int, r_min: float, r_max: float,
               *, mode: str = "exponential") -> np.ndarray:
    """Pure exponential knot sequence (legacy interface)."""
    if ns < ks + 1:
        raise ValueError(f"Need ns >= ks + 1, got ns={ns}, ks={ks}")
    if r_min <= 0 or r_max <= r_min:
        raise ValueError(f"Need 0 < r_min < r_max, got {r_min}, {r_max}")
    return make_exponential_knots(ns, ks, r_min, r_max)


def make_exponential_knots(ns: int, ks: int, r_min: float,
                           r_max: float) -> np.ndarray:
    """Pure exponential (grid_type=2): given ns, compute he."""
    nv = ns - ks + 1
    he = np.exp(np.log(r_max / r_min) / (nv - 1))
    t = np.zeros(ns + ks)
    t[ks] = r_min
    for i in range(ks + 1, ns + 1):
        t[i] = t[i - 1] * he
    t[ns:ns + ks] = r_max
    return t


def make_physics_knots(ks: int, Z: int, ns: int = 113,
                       r_max: float = None, nuclear: str = "point",
                       hi: float = None, he: float = None,
                       hmax: float = None, ml: int = 1) -> tuple:
    """DBSR_HF-style semi-exponential grid (mkgrid_01).

    Three-region grid:
      1. ml equal-spaced intervals of step hs  (nuclear region)
      2. Exponential growth with factor ht = 1 + he  (inner/mid region)
      3. Constant step hmax  (outer region, if needed)

    Default parameters tuned for DBSR_HF-quality accuracy:
      nuclear="point":  hi=0.025, he=0.125, hmax=1.0  (~100-125 B-splines)
      nuclear="fermi":  hi=0.25,  he=0.25,  hmax=1.0  (~112 B-splines, DBSR_HF standard)

    For point nucleus: hs = hi / Z
    For Fermi nucleus: hs = 4*log(3) * a_fermi * hi  (very small r_min)

    Parameters
    ----------
    ks : int
        B-spline order.
    Z : int
        Nuclear charge.
    ns : int
        Ignored — ns is determined by the grid construction.
    r_max : float or None
        Box radius. Default: 50.0 (DBSR_HF default).
    nuclear : str
        "point" (default) or "fermi" (DBSR_HF standard).
    hi, he, hmax : float or None
        Grid parameters. None = use defaults for the nuclear model.
    ml : int
        Number of initial equal-spaced intervals (default: 1).

    Returns
    -------
    t : ndarray
    ns : int (determined by grid)
    """
    if r_max is None:
        r_max = 50.0  # DBSR_HF default

    # Select defaults based on nuclear model
    # DBSR_HF mkgrid_01 uses the same defaults for all nuclear models:
    #   hi=0.25, he=0.25, hmax=1.0
    if hi is None: hi = 0.25
    if he is None: he = 0.25
    if hmax is None: hmax = 1.0

    # Initial step depends on nuclear model (DBSR_HF mkgrid_01)
    if nuclear == "fermi":
        from atomx.nuclear import fermi_grid_step
        hs = fermi_grid_step(Z, hi=hi)
    else:
        # Point nucleus: hs = hi / Z
        hs = hi / max(Z, 1)

    # Build breakpoints following DBSR_HF mkgrid_01
    breakpoints = []
    # Region 1: ml equal-spaced intervals
    for i in range(1, ml + 1):
        breakpoints.append(hs * i)

    # Region 2: exponential growth (ht = 1 + he)
    ti = breakpoints[-1]
    ht = 1.0 + he
    while True:
        tj = ti * ht
        if (tj - ti) > hmax:
            break
        if tj > r_max:
            break
        breakpoints.append(tj)
        ti = tj

    # Region 3: constant step hmax
    if ti < r_max:
        while True:
            ti_next = ti + hmax
            if ti_next >= r_max:
                break
            breakpoints.append(ti_next)
            ti = ti_next

    # Final boundary
    tmax = breakpoints[-1]
    if tmax < r_max:
        tmax = r_max

    nv = len(breakpoints)
    ns_out = nv + ks - 1

    # Build knot sequence: ks copies of 0, then breakpoints, then ks copies of tmax
    t = np.zeros(ns_out + ks)
    for i, bp in enumerate(breakpoints):
        t[ks + i] = bp
    t[ns_out:ns_out + ks] = tmax

    return t, ns_out


def make_dbsr_knots(ks: int, Z: int, r_max: float = 50.0,
                    hi: float = 0.25, he: float = 0.25,
                    h_max: float = 1.0,
                    nuclear: str = "fermi") -> tuple:
    """DBSR_HF mkgrid_01 semi-exponential grid (legacy interface).

    Uses Fermi nucleus parameter for initial step (very small r_min).
    Kept for backward compatibility.
    """
    if nuclear == "fermi":
        A = int(round(2.5 * max(Z, 1)))
        a_fermi_fm = 0.5
        bohr_fm = 5.29177e4
        a_fermi = a_fermi_fm / bohr_fm
        ts = 4.0 * np.log(3.0) * a_fermi
        hs = ts * hi
    else:
        hs = hi / max(Z, 1)

    ht = 1.0 + he
    ml = 1
    breakpoints = [hs * i for i in range(1, ml + 1)]

    r = breakpoints[-1]
    while r < r_max:
        r_next = r * ht
        if h_max > 0 and (r_next - r) > h_max:
            break
        r = r_next
        if r >= r_max:
            break
        breakpoints.append(r)

    if h_max > 0 and r < r_max:
        while r < r_max:
            r += h_max
            if r >= r_max:
                break
            breakpoints.append(r)

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


def make_semi_exponential_knots(ns_target, ks, Z, r_max, *,
                                hi=0.25, he=0.25, h_max=0.0):
    """Legacy interface."""
    t, ns = make_physics_knots(ks, Z, r_max, h_max, he)
    return t
