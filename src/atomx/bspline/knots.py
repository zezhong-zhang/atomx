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
                       r_max: float = None) -> tuple:
    """Unified physics-based grid for all elements (pure exponential).

    r_min = 0.05/Z : resolves 1s orbital peak (10% of orbital scale)
    r_max = adapted : captures outermost orbital tail
    he = computed  : from (ns, r_min, r_max) to optimally fill the range

    The pure exponential distributes points proportionally to r,
    matching the density gradient of ALL atomic shells simultaneously.
    No wasted points in regions with negligible density.

    Parameters
    ----------
    ks : int
        B-spline order.
    Z : int
        Nuclear charge.
    ns : int
        Number of B-splines (default 113, DBSR_HF standard).
    r_max : float or None
        Box radius. Default: adapted to outermost shell.

    Returns
    -------
    t : ndarray
    ns : int (same as input)
    """
    # Physics-derived r_min: 10% of 1s orbital scale
    r_min = 0.05 / max(Z, 1)

    # Physics-derived r_max: capture outermost orbital tail
    if r_max is None:
        # Outermost shell: n_max²/Z_eff, with tail extending to ~10x
        n_max = {1:1, 2:1}.get(Z)
        if n_max is None:
            if Z <= 2: n_max = 1
            elif Z <= 10: n_max = 2
            elif Z <= 18: n_max = 3
            elif Z <= 36: n_max = 4
            elif Z <= 54: n_max = 5
            else: n_max = 6
        r_outer = n_max ** 2 / max(Z - 10, 1)  # rough Thomas-Fermi
        r_max = max(20.0, min(50.0, 10.0 * r_outer))

    t = make_exponential_knots(ns, ks, r_min, r_max)
    return t, ns


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
