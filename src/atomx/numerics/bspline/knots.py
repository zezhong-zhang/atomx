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

    Default parameters (DBSR_HF mkgrid_01): hi=0.25, he=0.25, hmax=1.0
    for every nuclear model, except he=0.0625 for Z <= 2 (the
    eleven-digit hydrogen gate; see inline comment).

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
    if he is None:
        # Light atoms span a larger relative range between the nuclear
        # scale (1/Z) and the diffuse valence tail; the default
        # mid-region density leaves the point-nucleus H 1s eigenvalue
        # 2e-11 off the Sommerfeld value where the book's preface gate
        # (tests/test_book_preface.py) demands 1e-11.  Quadrupling the
        # exponential density was marginal on newer Python/NumPy
        # combinations; he=0.05 gives headroom while keeping heavier
        # DBSR_HF-validated grids unchanged.
        he = 0.05 if Z <= 2 else 0.25
    if hmax is None: hmax = 1.0

    # Initial step depends on nuclear model (DBSR_HF mkgrid_01)
    if nuclear == "fermi":
        from atomx.potentials.nuclear import fermi_grid_step
        hs = fermi_grid_step(Z, hi=hi)
    else:
        # Point nucleus: hs = hi / Z
        hs = hi / max(Z, 1)

    # Build breakpoints following DBSR_HF mkgrid_01 exactly.
    # ns = ks + nv - 1, where nv = number of breakpoints (intervals).
    # The knot vector t has ns+ks entries: ks zeros, then nv breakpoints,
    # then ks-1 copies of tmax.

    ns_count = ks + ml  # initial: ks boundary + ml uniform intervals
    breakpoints = []
    for i in range(1, ml + 1):
        breakpoints.append(hs * i)

    # Region 2: exponential growth (ht = 1 + he)
    ti = breakpoints[-1]
    ht = 1.0 + he
    while True:
        tj = ti * ht
        if (tj - ti) > hmax:
            break
        ns_count += 1
        ti = tj
        if ti > r_max:
            ns_count -= 1  # DBSR_HF: drop last if > rmax
            break
        breakpoints.append(ti)

    # Region 3: constant step hmax (until reaching rmax)
    if ti < r_max:
        while True:
            ti_next = ti + hmax
            if ti_next >= r_max:
                break
            ns_count += 1
            ti = ti_next
            breakpoints.append(ti)

    # tmax = last breakpoint + hmax (DBSR_HF overshoots rmax)
    tmax = breakpoints[-1] + hmax
    breakpoints.append(tmax)  # include tmax as the final breakpoint

    # nv = number of intervals = len(breakpoints)
    # ns = nv + ks - 1  (standard B-spline relation)
    nv = len(breakpoints)
    ns_out = nv + ks - 1

    # Knot vector: ks zeros, nv breakpoints, ks-1 copies of tmax
    t = np.zeros(ns_out + ks)
    for i, bp in enumerate(breakpoints):
        t[ks + i] = bp
    t[ns_out:ns_out + ks] = tmax

    return t, ns_out


def make_continuum_knots(
    ks: int,
    Z: int,
    k_max_au: float,
    *,
    r_max: float = 50.0,
    points_per_wavelength: int = 4,
) -> tuple[np.ndarray, int]:
    """Continuum-oriented semi-exponential knot sequence.

    The inner region resolves the shortest de Broglie wavelength
    ``2*pi/k_max_au`` with roughly ``points_per_wavelength`` points, then
    hands off to the standard physics grid spacing outside ``min(5/Z, 0.5)``.
    """
    if ks < 2:
        raise ValueError("ks must be >= 2")
    if Z <= 0:
        raise ValueError("Z must be positive")
    if k_max_au <= 0.0:
        raise ValueError("k_max_au must be positive")
    if r_max <= 0.0:
        raise ValueError("r_max must be positive")
    if points_per_wavelength <= 0:
        raise ValueError("points_per_wavelength must be positive")

    lambda_min = 2.0 * np.pi / float(k_max_au)
    r_inner = min(5.0 / float(Z), 0.5)
    h_inner = min(0.25 / float(Z), lambda_min / float(points_per_wavelength))

    if k_max_au <= 2.0:
        return make_physics_knots(ks=ks, Z=Z, r_max=r_max)

    breakpoints: list[float] = []
    r = h_inner
    while r <= min(r_inner, r_max):
        breakpoints.append(r)
        r += h_inner

    if not breakpoints:
        breakpoints.append(min(h_inner, r_max))

    # Continue with exponential growth capped by the standard 1 a0 outer step.
    step_cap = 1.0
    growth = 1.25
    r = breakpoints[-1]
    step = max(h_inner, 1.0e-8)
    while r < r_max:
        step = min(step * growth, step_cap)
        r_next = r + step
        if r_next >= r_max:
            break
        breakpoints.append(r_next)
        r = r_next

    if breakpoints[-1] < r_max:
        breakpoints.append(r_max)

    tmax = breakpoints[-1]
    nv = len(breakpoints)
    ns_out = nv + ks - 1
    t = np.zeros(ns_out + ks)
    for i, bp in enumerate(breakpoints):
        t[ks + i] = bp
    t[ns_out:ns_out + ks] = tmax
    return t, ns_out
