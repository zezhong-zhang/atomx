"""Scaffold migrations must still reject missing physical inputs.

Implemented limited entry points require explicit aligned physical inputs.
Actual numerical CCSD/CCSDT gates live in test_finite_ccsd.py and
test_finite_ccsdt.py; unsupported higher ranks and FSCC remain explicit errors.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# methods/ scaffolds
# ---------------------------------------------------------------------------


def test_finite_coupled_cluster_requires_explicit_physical_inputs():
    from atomx.methods import coupled_cluster

    assert hasattr(coupled_cluster, "CoupledClusterResult")
    # Finite nonlinear CCSD/CCSDT require aligned H/masks/reference.
    # Missing physical inputs still raise before any numerical work.
    with pytest.raises(TypeError):
        coupled_cluster.compute()




# ---------------------------------------------------------------------------
# scattering/ scaffolds
# ---------------------------------------------------------------------------


def test_r_matrix_scaffold_import_and_raises():
    from atomx.scattering import r_matrix

    assert hasattr(r_matrix, "RMatrixResult")
    assert hasattr(r_matrix, "compute")
    assert callable(r_matrix.compute)
    # compute() now aliases compute_multichannel (phase 2 landed);
    # calling with no args raises TypeError, not NotImplementedError.
    with pytest.raises(TypeError):
        r_matrix.compute()
