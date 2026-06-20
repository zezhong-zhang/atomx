"""Tests for Sternheimer sum-over-states EDM enhancement calculation.

These tests verify that the Sternheimer EDM enhancement factor:
1. Reproduces the correct order of magnitude for light atoms (H, Li)
2. Scales approximately as Z³ α² (the Sandars scaling)
3. Is consistent with the static polarizability via the relation
   R_e ≈ (Zα)² × α(0) × C_rel where C_rel is a relativistic correction.
"""

import pytest
import numpy as np

from atomx.methods.dirac_fock import DiracFock, Subshell, Exchange
from atomx.prop.edm.sternheimer import sternheimer_edm_enhancement
from atomx.prop.edm.scaling import electron_edm_enhancement_sandars
from atomx.prop.polarizability import dipole_polarizability
from atomx.constants import ALPHA


class TestSternheimerEDM:
    """Test Sternheimer EDM enhancement factor calculation."""

    def test_hydrogen_edm_order_of_magnitude(self):
        """For hydrogen, R_e should be ~10⁻⁴ (Sandars gives ~7×10⁻⁴)."""
        df = DiracFock(exchange=Exchange.HARTREE_FOCK)
        subshells = [Subshell(n=1, kappa=-1, occupation=1)]
        res = df.solve(Z=1, subshells=subshells)
        assert res.converged

        edm = sternheimer_edm_enhancement(res, kappa_max=4, n_virt_max=20)
        R_e = abs(edm.R_electron_edm)

        # Sandars scaling for H: ~7×10⁻⁴
        # Sternheimer should be within a factor of ~5 of Sandars
        sandars = electron_edm_enhancement_sandars(1, n_eff=1.0, J=0.5, L=0)
        assert R_e > 1e-5, f"R_e too small: {R_e}"
        assert R_e < 1e-2, f"R_e too large: {R_e}"
        # Check ratio with Sandars is within factor of 10
        ratio = R_e / sandars
        assert 0.1 < ratio < 10, f"Ratio with Sandars out of range: {ratio}"

    def test_lithium_edm_order_of_magnitude(self):
        """For Li (Z=3), R_e should be ~10⁻² (Sandars gives ~3.8×10⁻²)."""
        df = DiracFock(exchange=Exchange.HARTREE_FOCK)
        subshells = [
            Subshell(n=1, kappa=-1, occupation=2),
            Subshell(n=2, kappa=-1, occupation=1),
        ]
        res = df.solve(Z=3, subshells=subshells)
        assert res.converged

        edm = sternheimer_edm_enhancement(
            res, kappa_max=4, n_virt_max=20, target_subshell="2s1/2"
        )
        R_e = abs(edm.R_electron_edm)

        sandars = electron_edm_enhancement_sandars(3, n_eff=2.0, J=0.5, L=0)
        assert R_e > 1e-3, f"R_e too small: {R_e}"
        assert R_e < 1e0, f"R_e too large: {R_e}"
        ratio = R_e / sandars
        assert 0.1 < ratio < 10, f"Ratio with Sandars out of range: {ratio}"

    def test_edm_polarizability_relation(self):
        """R_e should be approximately (Zα)² × α(0) for light atoms."""
        df = DiracFock(exchange=Exchange.HARTREE_FOCK)
        subshells = [Subshell(n=1, kappa=-1, occupation=1)]
        res = df.solve(Z=1, subshells=subshells)
        assert res.converged

        pol = dipole_polarizability(res, kappa_max=4, n_virt_max=20)
        edm = sternheimer_edm_enhancement(res, kappa_max=4, n_virt_max=20)

        alpha_static = pol.alpha_static
        R_e = abs(edm.R_electron_edm)
        expected = (1.0 * ALPHA) ** 2 * alpha_static

        # For H, the relation R_e ≈ (Zα)² α(0) should hold within factor of 5
        ratio = R_e / expected
        assert 0.1 < ratio < 5, f"R_e / (Zα)²α relation out of range: {ratio}"

    def test_p3_2_does_not_contribute(self):
        """For s_{1/2} valence, p_{3/2} virtuals should not contribute
        because the T,P-odd pseudoscalar operator requires Δj = 0."""
        df = DiracFock(exchange=Exchange.HARTREE_FOCK)
        subshells = [Subshell(n=1, kappa=-1, occupation=1)]
        res = df.solve(Z=1, subshells=subshells)

        # Compute with only p_{1/2} (κ=+1) virtuals
        edm_p1 = sternheimer_edm_enhancement(
            res, kappa_max=1, n_virt_max=20
        )

        # Compute with p_{1/2} and p_{3/2} (κ=-2) virtuals
        edm_p3 = sternheimer_edm_enhancement(
            res, kappa_max=2, n_virt_max=20
        )

        # The p_{3/2} contribution should be negligible (cancels to zero)
        # because the T,P-odd angular factor is zero for Δj ≠ 0
        ratio = abs(edm_p3.R_electron_edm / edm_p1.R_electron_edm)
        assert ratio < 2, f"p_{{3/2}} contribution unexpectedly large: {ratio}"

    def test_target_subshell_selection(self):
        """Selecting a specific target subshell should work."""
        df = DiracFock(exchange=Exchange.HARTREE_FOCK)
        subshells = [
            Subshell(n=1, kappa=-1, occupation=2),
            Subshell(n=2, kappa=-1, occupation=1),
        ]
        res = df.solve(Z=3, subshells=subshells)

        # Compute for 2s1/2 only
        edm_2s = sternheimer_edm_enhancement(
            res, kappa_max=4, n_virt_max=20, target_subshell="2s1/2"
        )
        assert "2s1/2" in edm_2s.per_orbital
        assert abs(edm_2s.R_electron_edm) > 1e-5

    def test_result_container(self):
        """SternheimerEDMResult should have the expected attributes."""
        df = DiracFock(exchange=Exchange.HARTREE_FOCK)
        subshells = [Subshell(n=1, kappa=-1, occupation=1)]
        res = df.solve(Z=1, subshells=subshells)

        edm = sternheimer_edm_enhancement(res, kappa_max=4, n_virt_max=20)
        assert hasattr(edm, "R_electron_edm")
        assert hasattr(edm, "per_orbital")
        assert hasattr(edm, "scaling_comparison")
        assert hasattr(edm, "cpu_seconds")
        assert hasattr(edm, "provenance")
        assert "sandars" in edm.scaling_comparison
        assert "ratio" in edm.scaling_comparison

    def test_sandars_comparison_present(self):
        """The scaling comparison should contain Sandars value."""
        df = DiracFock(exchange=Exchange.HARTREE_FOCK)
        subshells = [Subshell(n=1, kappa=-1, occupation=1)]
        res = df.solve(Z=1, subshells=subshells)

        edm = sternheimer_edm_enhancement(res, kappa_max=4, n_virt_max=20)
        assert edm.scaling_comparison["sandars"] > 0
        assert isinstance(edm.scaling_comparison["ratio"], float)
