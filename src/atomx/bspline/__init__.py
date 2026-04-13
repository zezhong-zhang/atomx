"""B-spline basis and integrals for relativistic atomic structure calculations.

Follows the DBSR_HF approach (Zatsarinny & Froese Fischer, CPC 2016).
"""

from atomx.bspline.knots import make_knots, make_exponential_knots
from atomx.bspline.basis import (
    bsplvb,
    bsplvd,
    setup_gauss_points,
    evaluate_bsplines_on_grid,
)
from atomx.bspline.integrals import (
    overlap_matrix,
    nuclear_matrix,
    coupling_matrix,
    coupling_deriv_matrix,
)
