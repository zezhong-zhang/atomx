"""Costly partial-space/temporary/cached-metric rejection contracts."""

from unittest.mock import patch

import monovalent_norm_control as norm
import numpy as np
import pytest

from atomx.methods.core_sd import plan_core_sd
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.reduced_monovalent_sd import plan_reduced_monovalent_sd
from atomx.numerics.bspline.coulomb_features import PolynomialCellCoulombFeatures


@pytest.fixture(scope="module")
def plans():
    radial = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    identity = PolynomialCellCoulombFeatures(radial._bsp_data).identity
    cp = plan_core_sd(((1, -1),), ((2, -1), (3, -1)), identity)
    vp = plan_reduced_monovalent_sd(cp, ((2, -1),))
    return cp, vp


def test_norm_budget_rejects_before_any_vector_decode(plans):
    cp, vp = plans
    with (
        patch.object(norm, "decode", side_effect=AssertionError("late guard")),
        pytest.raises(MemoryError, match="scratch"),
    ):
        norm.norm_components(cp, vp, None, None, (2, -1), memory_bytes=1)


def test_incomplete_vector_rejects_before_radial_unpack(plans):
    cp, _ = plans
    with pytest.raises(ValueError, match="complete"):
        norm.decode(cp, np.zeros(cp["raw_amplitudes"] - 1))


def test_complex_vector_never_silently_projects_to_real(plans):
    cp, _ = plans
    with pytest.raises(ValueError, match="real FP64"):
        norm.decode(cp, np.zeros(cp["raw_amplitudes"], dtype=complex))


def test_boolean_projection_rejects_before_decode(plans):
    cp, vp = plans
    with (
        patch.object(norm, "decode", side_effect=AssertionError("late projection")),
        pytest.raises(ValueError, match="projection"),
    ):
        norm.norm_components(cp, vp, None, None, (2, -1), twoM=True)


def test_finite_scan_bounds_temporary_and_checks_last_tile():
    size = 1024**2 + 3
    plan = dict(raw_amplitudes=size)
    vector = np.zeros(size)
    vector[-1] = np.nan
    original = np.isfinite
    observed = []

    def checked(values):
        observed.append(values.size)
        if values.size > 1024**2:
            raise AssertionError("full-vector boolean allocation")
        return original(values)

    with (
        patch.object(norm.np, "isfinite", checked),
        pytest.raises(ValueError, match="finite"),
    ):
        norm.decode(plan, vector)
    assert sum(observed) == size


def test_cached_angular_metric_cannot_be_mutated():
    for gram in norm.grams(-1, -1, -1, -1, (0, 1), (0, 1), "none", 0):
        with pytest.raises(ValueError, match="read-only"):
            gram.flat[0] = 0
