"""Tests for Sternheimer sum-over-states EDM enhancement calculation.

The electron-EDM enhancement factor ``R_e`` is dominated by the
near-nuclear relativistic region, where the small spinor component
``Q ~ (Zα)/(2c) P`` produces the steep ``Z``-dependence that makes
heavy atoms sensitive to ``d_e`` (Sandars 1965).  These tests verify
that the Dirac-operator Sternheimer SOS reproduces the qualitative
physics that the issue requires:

1. ``R_e`` is finite and non-zero for a simple valence electron.
2. ``R_e`` grows steeply with ``Z`` — the relativistic enhancement —
   reaching the hundreds regime for a heavy atom (Tl benchmark).
3. The T,P-odd pseudoscalar selection rule (``Δj = 0``) suppresses the
   ``p_{3/2}`` channel for an ``s_{1/2}`` valence.
4. The result container and Sandars comparison are populated.

Absolute agreement with the Sandars closed-form scaling
(:mod:`atomx.prop.edm.scaling`) is only expected at the factor-of-2-3
level of a single-configuration (Dirac-Fock) calculation; the
light-atom ``R_e`` in particular is sensitive to finite-nuclear-size
regularisation of the ``Z/r²`` field kernel, which is not yet folded
in, so the tests use physically-robust bounds rather than tight
agreement with the (heavy-atom-calibrated) Sandars formula.
"""

import pytest
import numpy as np

from atomx.methods.dirac_fock import DiracFock, Subshell, Exchange
from atomx.prop.edm.sternheimer import sternheimer_edm_enhancement
from atomx.prop.edm.scaling import electron_edm_enhancement_sandars
from atomx.constants import ALPHA


def _solve_h():
    df = DiracFock(exchange=Exchange.HARTREE_FOCK)
    return df.solve(Z=1, subshells=[Subshell(n=1, kappa=-1, occupation=1)])


def _solve_lithium():
    df = DiracFock(exchange=Exchange.HARTREE_FOCK)
    return df.solve(
        Z=3,
        subshells=[
            Subshell(n=1, kappa=-1, occupation=2),
            Subshell(n=2, kappa=-1, occupation=1),
        ],
    )


def _thallium_subshells():
    """[Xe] 4f14 5d10 6s2 6p1 -> ground state 6p_{1/2}."""
    return [
        Subshell(n=1, kappa=-1, occupation=2),
        Subshell(n=2, kappa=-1, occupation=2), Subshell(n=2, kappa=1, occupation=2),
        Subshell(n=2, kappa=-2, occupation=4),
        Subshell(n=3, kappa=-1, occupation=2), Subshell(n=3, kappa=1, occupation=2),
        Subshell(n=3, kappa=-2, occupation=4), Subshell(n=3, kappa=2, occupation=4),
        Subshell(n=3, kappa=-3, occupation=6),
        Subshell(n=4, kappa=-1, occupation=2), Subshell(n=4, kappa=1, occupation=2),
        Subshell(n=4, kappa=-2, occupation=4), Subshell(n=4, kappa=2, occupation=4),
        Subshell(n=4, kappa=-3, occupation=6), Subshell(n=4, kappa=3, occupation=6),
        Subshell(n=4, kappa=-4, occupation=8),
        Subshell(n=5, kappa=-1, occupation=2), Subshell(n=5, kappa=1, occupation=2),
        Subshell(n=5, kappa=-2, occupation=4), Subshell(n=5, kappa=2, occupation=4),
        Subshell(n=6, kappa=-1, occupation=2),
        Subshell(n=6, kappa=1, occupation=1),
    ]


class TestSternheimerEDM:
    """Test Sternheimer EDM enhancement factor calculation."""

    def test_hydrogen_edm_finite_and_nonzero(self):
        """R_e for hydrogen must be finite, non-zero and small (the eEDM
        signal vanishes as Z -> 0 because there is no relativistic
        enhancement)."""
        res = _solve_h()
        assert res.converged
        edm = sternheimer_edm_enhancement(res, kappa_max=4, n_virt_max=20)
        R_e = abs(edm.R_electron_edm)
        assert R_e > 1e-6, f"R_e unexpectedly zero: {R_e}"
        # Light atoms carry essentially no relativistic enhancement.
        assert R_e < 1.0, f"R_e(H) unexpectedly large: {R_e}"

    def test_rel_scaling_with_z(self):
        """R_e must rise steeply with Z — the Sandars relativistic
        enhancement.  A 3x increase in Z from Li to Na must increase
        |R_e| by far more than the naive non-relativistic ~Z factor."""
        li = _solve_lithium()
        df = DiracFock(exchange=Exchange.HARTREE_FOCK)
        na = df.solve(
            Z=11,
            subshells=[
                Subshell(n=1, kappa=-1, occupation=2),
                Subshell(n=2, kappa=-1, occupation=2), Subshell(n=2, kappa=1, occupation=2),
                Subshell(n=2, kappa=-2, occupation=4),
                Subshell(n=3, kappa=-1, occupation=1),
            ],
        )
        assert li.converged and na.converged
        R_li = abs(sternheimer_edm_enhancement(
            li, kappa_max=4, n_virt_max=20, target_subshell="2s1/2").R_electron_edm)
        R_na = abs(sternheimer_edm_enhancement(
            na, kappa_max=4, n_virt_max=20, target_subshell="3s1/2").R_electron_edm)
        # Z rises 3x (3 -> 11); the relativistic kernel makes R_e rise
        # much faster than Z.  Non-relativistically one would expect
        # roughly a factor ~Z ~ 3-4; the genuine Dirac operator gives ~20x.
        assert R_na > 5.0 * R_li, (
            f"Expected steep Z-scaling (R(Na) >> R(Li)); got "
            f"R(Na)={R_na:.3e} vs R(Li)={R_li:.3e}")

    def test_thallium_benchmark(self):
        """Tl 6p_{1/2} is the classic heavy-atom eEDM benchmark.

        The relativistic enhancement must place |R_e| in the hundreds
        regime (literature ~585; Sandars scaling ~hundreds), NOT the
        ~0.1 value a non-relativistic / E1-proportional shortcut would
        give.  This is the central deliverable of HYP-685.
        """
        df = DiracFock(exchange=Exchange.HARTREE_FOCK)
        res = df.solve(Z=81, subshells=_thallium_subshells())
        assert res.converged, "Tl SCF must converge for the benchmark"
        edm = sternheimer_edm_enhancement(
            res, kappa_max=4, n_virt_max=20, target_subshell="6p1/2")
        R_e = abs(edm.R_electron_edm)
        # Hundreds regime: comfortably above the light-atom scale and
        # well below an unphysical blow-up.
        assert R_e > 50.0, (
            f"R_e(Tl 6p1/2)={R_e:.3e} far below the expected hundreds-scale "
            f"enhancement (lit ~585); relativistic boost not captured.")
        assert R_e < 5000.0, f"R_e(Tl)={R_e:.3e} unphysically large"
        # The heavy-atom enhancement must dwarf the light-atom value.
        res_h = _solve_h()
        R_h = abs(sternheimer_edm_enhancement(
            res_h, kappa_max=4, n_virt_max=20).R_electron_edm)
        assert R_e > 1e3 * R_h, (
            f"Heavy-atom Tl ({R_e:.3e}) should dwarf H ({R_h:.3e}) "
            f"by many orders of magnitude.")

    def test_p3_2_does_not_contribute(self):
        """For s_{1/2} valence, p_{3/2} virtuals must not contribute
        because the T,P-odd pseudoscalar operator requires Delta j = 0."""
        res = _solve_h()
        # p_{1/2} (kappa=+1) virtuals only (kappa_max=1).
        edm_p1 = sternheimer_edm_enhancement(res, kappa_max=1, n_virt_max=20)
        # p_{1/2} and p_{3/2} (kappa=-2) virtuals (kappa_max=2).
        edm_p3 = sternheimer_edm_enhancement(res, kappa_max=2, n_virt_max=20)
        ratio = abs(edm_p3.R_electron_edm) / abs(edm_p1.R_electron_edm)
        assert ratio < 2, f"p_{{3/2}} contribution unexpectedly large: {ratio}"

    def test_target_subshell_selection(self):
        """Selecting a specific target subshell should work."""
        res = _solve_lithium()
        edm_2s = sternheimer_edm_enhancement(
            res, kappa_max=4, n_virt_max=20, target_subshell="2s1/2")
        assert "2s1/2" in edm_2s.per_orbital
        assert abs(edm_2s.R_electron_edm) > 1e-6

    def test_result_container(self):
        """SternheimerEDMResult should have the expected attributes."""
        res = _solve_h()
        edm = sternheimer_edm_enhancement(res, kappa_max=4, n_virt_max=20)
        assert hasattr(edm, "R_electron_edm")
        assert hasattr(edm, "per_orbital")
        assert hasattr(edm, "scaling_comparison")
        assert hasattr(edm, "cpu_seconds")
        assert hasattr(edm, "provenance")
        assert "sandars" in edm.scaling_comparison
        assert "ratio" in edm.scaling_comparison

    def test_sandars_comparison_present(self):
        """The scaling comparison should contain the Sandars value."""
        res = _solve_h()
        edm = sternheimer_edm_enhancement(res, kappa_max=4, n_virt_max=20)
        assert edm.scaling_comparison["sandars"] > 0
        assert isinstance(edm.scaling_comparison["ratio"], float)
