"""B-spline basis functions, radial grids, and integral assembly.

Follows the DBSR_HF approach (Zatsarinny & Froese Fischer, CPC 2016).

Submodules
----------
basis : B-spline evaluation (Cox-de Boor) and bsp_data preparation.
knots : Radial grid (knot sequence) generation.
integrals : One-electron matrix elements (overlap, nuclear, coupling).
convol : Two-electron Slater integrals via cell-based convolution.
evaluate : Orbital evaluation and Y^k potentials on the grid.
"""

from atomx.numerics.bspline.basis import (
    bsplvb,
    bsplvd,
    evaluate_bsplines_on_grid,
    setup_gauss_points,
)
from atomx.numerics.bspline.integrals import (
    coupling_deriv_matrix,
    coupling_matrix,
    nuclear_matrix,
    overlap_matrix,
)
from atomx.numerics.bspline.knots import make_exponential_knots, make_knots
