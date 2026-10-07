"""Small complete 4f² gates; measured ion transfers live in the slow campaign.

These fixed, normalized radial functions test algorithms and conventions,
not a stationary free-ion model or experimental agreement. All 14 f spinors
and their complete M sectors are retained (91 determinants in total).
"""

from itertools import combinations

import numpy as np
import pytest

from atomx.methods.csf.slater_scaling import (
    fit_slater_scaling,
    free_ion_levels,
    make_slater_delta,
    multiset_subtract,
)
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.knots import make_knots
from atomx.physics.csf import sparse_ci as SC


@pytest.fixture(scope="module")
def small_f2():
    ns, ks = 24, 7
    knots = make_knots(ns, ks, 1e-6, 30.0)
    data = prepare_bspline_data(knots, ns, ks)
    # A fixed f-like expansion, shared between the two j partners.
    p = np.zeros(ns)
    p[16:23] = (0.1, 0.3, 0.7, 1.0, 0.8, 0.4, 0.1)
    values = np.zeros_like(data["gr"])
    for iv in range(data["nv"]):
        values[iv] = data["bsp_p"][iv] @ p[iv:iv + ks]
    p /= np.sqrt(np.sum(data["grw"] * values**2))
    od = {(4, k): (p.copy(), np.zeros(ns)) for k in (3, -4)}
    patterns = [((4, 3, 2),), ((4, 3, 1), (4, -4, 1)), ((4, -4, 2),)]
    return od, data, patterns


def test_scaling_is_linear_antisymmetric_and_hermitian(small_f2):
    od, data, patterns = small_f2
    masks, spinors = SC.enumerate_determinants(patterns, od, 0)
    scales = {2: -0.2, 4: -0.1, 6: -0.05}
    delta = make_slater_delta(od, data, scales)
    individual = {k: make_slater_delta(od, data, {k: s}) for k, s in scales.items()}
    pairs = list(combinations(range(len(spinors)), 2))
    nonzero = off_diagonal = 0
    for a, b in pairs:
        for c, d in pairs:
            value = delta(a, b, c, d, spinors)
            assert value == pytest.approx(sum(f(a, b, c, d, spinors)
                                             for f in individual.values()), abs=2e-15)
            assert value == pytest.approx(-delta(b, a, c, d, spinors), abs=2e-15)
            assert value == pytest.approx(-delta(a, b, d, c, spinors), abs=2e-15)
            assert value == pytest.approx(delta(c, d, a, b, spinors), abs=2e-15)
            nonzero += abs(value) > 1e-12
            off_diagonal += (a, b) != (c, d) and abs(value) > 1e-12
    assert nonzero > 10 and off_diagonal > 10
    H = SC.assemble_sparse_hamiltonian(
        masks, spinors, od, data, 1, data["ns"],
        two_body_delta=lambda a, b, c, d: delta(a, b, c, d, spinors))
    np.testing.assert_allclose(H.toarray(), H.toarray().T, atol=1e-13, rtol=0)
    assert make_slater_delta(od, data, {2: 0.0}) is None


def test_complete_j_ladder_and_known_parameter_recovery(small_f2):
    od, data, patterns = small_f2
    target = (-0.2, -0.1, -0.05)

    def levels(scale):
        return free_ion_levels(od, data, 1, data["ns"], patterns, scale=scale, cm=1.0)

    reference = levels(target)
    # Full 4f² space: choose(14,2) states, recovered with 2J+1 weights.
    assert sum(round(2*j + 1)*len(es) for j, es in reference.items()) == 91
    def residuals(scale):
        computed = levels(scale)
        assert computed.keys() == reference.keys()
        return np.concatenate([computed[j] - reference[j] for j in sorted(reference)])

    fitted, history, diagnostics = fit_slater_scaling(residuals, tol=1e-12)
    np.testing.assert_allclose(fitted, target, atol=1e-8, rtol=0)
    assert history[-1] < 1e-10
    assert np.isfinite(diagnostics["cond"])


def test_multiset_subtraction_preserves_degenerate_copies():
    np.testing.assert_array_equal(multiset_subtract([1, 1, 2, 3], [1, 3]), [1, 2])
    with pytest.raises(ValueError, match="high sector larger"):
        multiset_subtract([1], [1, 1])
