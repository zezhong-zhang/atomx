"""Block validation must retain full Gram semantics for cross-block leakage."""
from types import SimpleNamespace

import numpy as np
import pytest

from atomx.physics.csf.levels import ci_level_blocks


def case(mixing):
    csfs = [SimpleNamespace(total_J2=0,parity=1),SimpleNamespace(total_J2=2,parity=-1)]
    ci = SimpleNamespace(state_energies=np.array([-2.,-1.]),mixing_coefficients=np.array(mixing,dtype=float))
    return csfs,ci


def test_small_nonorthogonal_cross_block_leakage_cannot_hide_behind_probability_purity():
    csfs,ci = case([[1.,1e-8],[0.,1.]])
    # Leakage probability 1e-16 passes symmetry tolerance; the Gram defect
    # 1e-8 still violates orthogonality and must take the full fallback.
    with pytest.raises(ValueError,match='orthonormal'):
        ci_level_blocks(csfs,ci)


def test_orthogonal_small_cross_block_rotation_keeps_original_tolerance_semantics():
    a = 1e-5
    csfs,ci = case([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    e,blocks = ci_level_blocks(csfs,ci)
    np.testing.assert_array_equal(blocks[0,1],[0])
    np.testing.assert_array_equal(blocks[2,-1],[1])


def test_tiny_nonzero_coefficients_and_zero_columns_are_not_discarded():
    csfs,ci = case([[1.,1e-80],[0.,1.]])
    before = ci.mixing_coefficients.tobytes()
    ci_level_blocks(csfs,ci)
    assert before==ci.mixing_coefficients.tobytes()
    ci.mixing_coefficients[:,1] = 0
    with pytest.raises(ValueError,match='orthonormal'):
        ci_level_blocks(csfs,ci)


def test_exact_block_support_still_requires_each_block_to_be_orthonormal():
    csfs,ci = case([[1.01,0.],[0.,1.]])
    with pytest.raises(ValueError,match='orthonormal'):
        ci_level_blocks(csfs,ci)
