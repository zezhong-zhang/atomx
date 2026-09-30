"""Real radial Helmholtz multipoles on ordered Gauss cells.

The Green function is cos(k |r-s|)/|r-s|. Racah-normalized angular
multipoles factor into regular/irregular spherical Bessel functions on
the two radial triangles. Partial cell integrals preserve this structure:
work and storage grow linearly with the number of cells.
"""

import numpy as np
from scipy.special import factorial2, spherical_jn, spherical_yn

from atomx.numerics.yk import yk_partial_matrices


def _scaled_factors(order, x):
    """Return J, Y, (J-1)/x² and (Y-1)/x², including x=0.

    J=(2L+1)!! j_L(x)/x^L, Y=-x^(L+1)y_L(x)/(2L-1)!!.
    Subtracted series prevent cancellation in the static-subtracted Green
    function. Both series are even; the irregular factor is regular here.
    """
    x = np.abs(np.asarray(x, dtype=float))
    j, y, dj, dy = (np.empty_like(x) for _ in range(4))
    small = x < .5
    xx = x[small]**2
    tj = np.full_like(xx, -1 / (2 * (2 * order + 3)))
    ty = np.full_like(xx, 1 / (2 * (2 * order - 1)))
    sj, sy = tj.copy(), ty.copy()
    for n in range(2, 13):
        tj *= -xx / (2 * n * (2 * order + 2 * n + 1))
        ty *= xx / (2 * n * (2 * order + 1 - 2 * n))
        sj += tj
        sy += ty
    dj[small], dy[small] = sj, sy
    j[small], y[small] = 1 + xx * sj, 1 + xx * sy
    large = ~small
    z = x[large]
    j[large] = factorial2(2 * order + 1) * spherical_jn(order, z) / z**order
    denominator = 1.0 if order == 0 else factorial2(2 * order - 1)
    y[large] = -z**(order + 1) * spherical_yn(order, z) / denominator
    dj[large] = (j[large] - 1) / z**2
    dy[large] = (y[large] - 1) / z**2
    return j, y, dj, dy


def _triangle_potential(rho, weights, lower, upper):
    """Contract lower(r_<) upper(r_>) without a dense radial matrix."""
    out_matrix, in_matrix = yk_partial_matrices(rho.shape[1])
    f, g = weights * rho * lower, weights * rho * upper
    ft, gt = f.sum(axis=1), g.sum(axis=1)
    outward = (np.cumsum(ft) - ft)[:, None] + f @ out_matrix.T
    inward = (np.cumsum(gt[::-1])[::-1] - gt)[:, None] + g @ in_matrix.T
    return upper * outward + lower * inward


class HelmholtzKernel:
    """Reusable geometry for one multipole and wave number on Gauss cells.

    Different transition densities share these Bessel factors. A caller
    may cache kernels within one workspace; there is no process-wide cache
    retaining radial grids or orbital data.
    """

    def __init__(self, r, weights, order, wave_number):
        if int(order) != order or order < 0:
            raise ValueError("multipole order must be a nonnegative integer")
        if not np.isfinite(wave_number):
            raise ValueError("wave_number must be finite")
        r, weights = (np.asarray(a, dtype=float) for a in (r, weights))
        if r.ndim != 2 or r.shape != weights.shape:
            raise ValueError("radii and weights need matching Gauss-cell shapes")
        if np.any(r <= 0):
            raise ValueError("Helmholtz Gauss nodes must have positive radii")
        order = int(order)
        self.k = abs(float(wave_number))
        self.weights = weights
        j, y, dj, dy = _scaled_factors(order, self.k * r)
        lower, upper = r**order, r**(-order - 1)
        self.regular = lower * j
        self.irregular = upper * y
        self.lower = lower
        self.regular_difference = lower * r**2 * dj
        self.irregular_difference = upper * r**2 * dy

    def potential(self, rho, *, component="full"):
        """Contract a density with full, subtracted, or retardation kernel."""
        rho = np.asarray(rho, dtype=float)
        if rho.shape != self.weights.shape:
            raise ValueError("density and kernel need matching Gauss-cell shapes")
        if component == "full":
            return _triangle_potential(rho, self.weights, self.regular, self.irregular)
        if component not in {"difference", "retardation"}:
            raise ValueError("unknown Helmholtz component")
        # (J_lo Y_hi - 1)/k² = r_lo² dJ_lo Y_hi + r_hi² dY_hi.
        # Retain finite-frequency terms without subtracting nearly equal
        # separately integrated kernels.
        divided_difference = (
            _triangle_potential(rho, self.weights, self.regular_difference, self.irregular)
            + _triangle_potential(rho, self.weights, self.lower, self.irregular_difference))
        if component == "difference":
            return self.k**2 * divided_difference
        return -2 * divided_difference


def helmholtz_potential(rho, r, weights, order, wave_number, *, component="full"):
    """Contract a real Helmholtz multipole with a radial density.

    ``full`` uses cos(k R)/R; ``difference`` uses (cos(k R)-1)/R;
    ``retardation`` uses -2(cos(k R)-1)/(k² R), tending to R at zero.
    All use Racah-normalized angular multipoles. ``r`` and ``weights``
    are two-dimensional ordered Gauss-cell arrays with positive radii.
    The wave number is in inverse Bohr, not an energy in Hartree.
    For multiple densities reuse a :class:`HelmholtzKernel` instance.
    """
    return HelmholtzKernel(r, weights, order, wave_number).potential(rho, component=component)
