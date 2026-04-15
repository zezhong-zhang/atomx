"""M0 tests: B-spline basis and integrals."""

import numpy as np
import pytest
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.basis import bsplvb, find_span, setup_gauss_points, evaluate_bsplines_on_grid
from atomx.numerics.bspline.integrals import overlap_matrix


class TestKnots:
    def test_exponential_knot_boundaries(self):
        """First ks knots = 0, last ks knots = r_max."""
        ns, ks = 40, 9
        t = make_knots(ns, ks, r_min=0.001, r_max=50.0)
        assert t.shape == (ns + ks,)
        assert np.all(t[:ks] == 0.0)
        assert np.all(t[ns:] == 50.0)

    def test_knot_monotonicity(self):
        """Knots must be non-decreasing."""
        t = make_knots(50, 9, r_min=0.001, r_max=100.0)
        assert np.all(np.diff(t) >= 0)

    def test_interior_knots_positive(self):
        """Interior knots must be strictly positive."""
        ns, ks = 40, 9
        t = make_knots(ns, ks, r_min=0.001, r_max=50.0)
        interior = t[ks:ns]
        assert np.all(interior > 0)


class TestBasis:
    def test_partition_of_unity(self):
        """B-splines sum to 1 at any interior point."""
        ns, ks = 30, 7
        t = make_knots(ns, ks, r_min=0.01, r_max=20.0)

        # Test at several points
        for x in [0.05, 0.5, 5.0, 10.0, 19.0]:
            left = find_span(t, ns, ks, x)
            b = bsplvb(t, ks, x, left)
            assert np.sum(b) == pytest.approx(1.0, abs=1e-14), \
                f"Partition of unity failed at x={x}: sum={np.sum(b)}"

    def test_nonnegativity(self):
        """B-splines are non-negative."""
        ns, ks = 30, 7
        t = make_knots(ns, ks, r_min=0.01, r_max=20.0)

        for x in np.linspace(0.01, 19.9, 100):
            left = find_span(t, ns, ks, x)
            b = bsplvb(t, ks, x, left)
            assert np.all(b >= -1e-15), f"Negative B-spline at x={x}"


class TestOverlap:
    def test_overlap_symmetry(self):
        """Overlap matrix must be symmetric."""
        ns, ks = 30, 7
        t = make_knots(ns, ks, r_min=0.01, r_max=20.0)
        gr, grw = setup_gauss_points(t, ns, ks)
        bsp = evaluate_bsplines_on_grid(t, ns, ks, gr, nderiv=1)
        bsp_val = bsp[:, :, :, 0]
        nv = ns - ks + 1
        S = overlap_matrix(nv, ks, ns, bsp_val, grw)
        np.testing.assert_allclose(S, S.T, atol=1e-14)

    def test_overlap_positive_definite(self):
        """Overlap matrix must be positive definite."""
        ns, ks = 30, 7
        t = make_knots(ns, ks, r_min=0.01, r_max=20.0)
        gr, grw = setup_gauss_points(t, ns, ks)
        bsp = evaluate_bsplines_on_grid(t, ns, ks, gr, nderiv=1)
        bsp_val = bsp[:, :, :, 0]
        nv = ns - ks + 1
        S = overlap_matrix(nv, ks, ns, bsp_val, grw)
        # Check eigenvalues > 0 (excluding boundary splines)
        eigvals = np.linalg.eigvalsh(S[1:-1, 1:-1])
        assert np.all(eigvals > 0), f"Non-positive eigenvalue: {eigvals.min()}"
