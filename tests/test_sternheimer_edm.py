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
(:mod:`atomx.properties.edm.scaling`) is NOT expected for light atoms:
the closed form ``R_e ≈ 10 α² Z³ n*/(J(J+1)(2L+1))`` is a heavy-atom
scaling law, and for H 1s the measured ratio is 42.85
(``R_e = 0.0304260`` from the SOS vs ``7.10018e-4`` from the closed
form), not the factor-of-2-3 the module docstring used to claim.  See
``test_sandars_ratio_measured_for_h``.

The Tl benchmark (item 2) is **ungated**: the Tl SCF reaches its fixed
point (residual 1.877e-07 at iteration 11, 53x below ``tol=1e-5``) but
``DiracFock.solve`` downgrades ``converged`` to False because the
same-kappa overlap diagnostic measures 1.702e-03 against its 1e-03
threshold — the solver omits the off-diagonal Lagrange multipliers for
orbitals sharing κ (review finding H4).  ``sternheimer_edm_enhancement``
therefore raises ``ValueError("SCFResult must be converged")`` and no
``R_e(Tl)`` number exists to gate.  Both Tl tests below are
``xfail(strict=True)`` so the missing flagship number stays loud, and
``test_thallium_blocker_is_orthogonality_not_the_residual`` pins the
cause executably.
"""

import functools

import pytest

from atomx.methods.dirac_fock import (
    _SAME_KAPPA_OVERLAP_TOL,
    DiracFock,
    Exchange,
    Subshell,
)
from atomx.properties.edm.sternheimer import sternheimer_edm_enhancement


@functools.lru_cache(maxsize=1)
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


@functools.lru_cache(maxsize=1)
def _solve_thallium():
    """Tl SCF shared by the benchmark, its smoke variant and the blocker test.

    One solve (~13 s) instead of three; the result is read-only here.
    """
    df = DiracFock(exchange=Exchange.HARTREE_FOCK)
    return df.solve(Z=81, subshells=_thallium_subshells())


class TestSternheimerEDM:
    """Test Sternheimer EDM enhancement factor calculation."""

    @pytest.mark.snapshot
    def test_hydrogen_edm_finite_and_nonzero(self):
        """R_e for hydrogen: measured 0.0304260 (κmax=4, n_virt=20).

        The previous gate was ``1e-6 < R_e < 1`` — a 10⁶-wide window that
        could not distinguish this from any other light-atom value.  The
        physical content kept here is the sign/magnitude structure
        (small, positive, non-zero: no relativistic enhancement at Z=1);
        the *number* is atomx's own (no external H R_e reference is
        available at this order), hence ``snapshot``.
        """
        res = _solve_h()
        assert res.converged
        edm = sternheimer_edm_enhancement(res, kappa_max=4, n_virt_max=20)
        R_e = abs(edm.R_electron_edm)
        assert R_e > 1e-6, f"R_e unexpectedly zero: {R_e}"
        # Light atoms carry essentially no relativistic enhancement.
        assert R_e < 1.0, f"R_e(H) unexpectedly large: {R_e}"
        assert pytest.approx(0.0304260, rel=0.05) == R_e, (
            f"R_e(H 1s) = {R_e:.6e}; measured 0.0304260 at kappa_max=4, "
            f"n_virt_max=20")

    @pytest.mark.slow
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

    @pytest.mark.slow
    @pytest.mark.xfail(strict=True, reason=(
        "R_e(Tl 6p1/2) is UNGATED.  The Tl SCF reaches its fixed point — "
        "measured residual 1.877e-07 at iteration 11, i.e. 53x BELOW "
        "tol=1e-5, unchanged at max_iter=120 — but DiracFock.solve "
        "downgrades converged to False because the same-kappa overlap "
        "diagnostic measures max_same_kappa_overlap = 1.702e-03 against "
        "the 1e-03 threshold in src/atomx/methods/dirac_fock.py (the SCF "
        "omits the off-diagonal Lagrange multipliers for orbitals sharing "
        "kappa, review finding H4).  sternheimer_edm_enhancement then "
        "raises ValueError('SCFResult must be converged'), so no "
        "R_e(Tl) number exists at all: the 'central deliverable of "
        "HYP-685' (file docstring, literature ~585) has no gate.  The "
        "audit's suggested fix (relax the SCF tolerance) would not help: "
        "the residual is already 53x inside it.  max_iter=120, "
        "mode='orbital', damping=0.3 and use_physics_grid=False all give "
        "the same 1.702e-03 overlap.  This xfail is loud by design: the "
        "moment the solver enforces same-kappa orthogonality the test "
        "XPASSes and strict mode fails."))
    def test_thallium_benchmark(self):
        """Tl 6p_{1/2} is the classic heavy-atom eEDM benchmark.

        The relativistic enhancement must place |R_e| in the hundreds
        regime (literature ~585; Sandars scaling ~hundreds), NOT the
        ~0.1 value a non-relativistic / E1-proportional shortcut would
        give.  This is the central deliverable of HYP-685.
        """
        res = _solve_thallium()
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
        R_h = abs(sternheimer_edm_enhancement(
            _solve_h(), kappa_max=4, n_virt_max=20).R_electron_edm)
        assert R_e > 1e3 * R_h, (
            f"Heavy-atom Tl ({R_e:.3e}) should dwarf H ({R_h:.3e}) "
            f"by many orders of magnitude.")

    def test_thallium_blocker_is_orthogonality_not_the_residual(self):
        """The Tl gate is blocked by the same-kappa overlap, not the SCF.

        Executable record of the xfail reasons above: if anyone fixes the
        Lagrange-multiplier omission this test fails loudly and the two
        xfails must be re-examined (and will XPASS anyway).
        """
        res = _solve_thallium()
        # Numerically converged: the orbital-change residual is far inside
        # the solver tolerance ...
        assert res.residual < 1e-5, (
            f"Tl SCF residual {res.residual:.3e} is not below tol=1e-5")
        # ... and yet `converged` is False, solely because of the
        # same-kappa orthogonality diagnostic.
        assert not res.converged
        overlap = res.energy_components["max_same_kappa_overlap"]
        assert overlap > _SAME_KAPPA_OVERLAP_TOL, (
            f"same-kappa overlap {overlap:.3e} no longer exceeds "
            f"{_SAME_KAPPA_OVERLAP_TOL:.0e}; re-check the Tl xfails")
        # ... so the public EDM entry point refuses to produce a number.
        with pytest.raises(ValueError, match="must be converged"):
            sternheimer_edm_enhancement(
                res, kappa_max=4, n_virt_max=20, target_subshell="6p1/2")

    def test_p3_2_does_not_contribute(self):
        """For s_{1/2} valence, p_{3/2} virtuals must not contribute
        because the T,P-odd pseudoscalar operator requires Delta j = 0."""
        from atomx.properties.edm.sternheimer import _tp_odd_angular
        # Direct angular-factor test: s_{1/2} (kappa=-1) -> p_{3/2} (kappa=-2)
        # has j_a=1/2, j_r=3/2 -> Delta j != 0, so the pseudoscalar operator
        # must vanish.
        assert _tp_odd_angular(-1, -2) == 0.0, (
            "p_{3/2} angular factor for s_{1/2} valence should be exactly zero")
        # End-to-end: with kappa_max large enough to include p_{3/2}, the
        # total R_e must still be non-zero (p_{1/2} contributes) and finite.
        res = _solve_h()
        edm = sternheimer_edm_enhancement(res, kappa_max=4, n_virt_max=20)
        R_e = abs(edm.R_electron_edm)
        assert R_e > 1e-6, f"R_e unexpectedly zero: {R_e}"

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

    @pytest.mark.snapshot
    def test_sandars_comparison_present(self):
        """The Sandars comparison is a *number*, not just a float.

        Measured for H 1s at kappa_max=4, n_virt_max=20:
        ``R_e(SOS) = 0.0304260``, ``sandars = 7.10018e-4``,
        ``ratio = 42.85``.  The old assertion (``isinstance(ratio, float)``)
        could not see that this is 43x, not the "factor-of-2-3" the module
        docstring used to claim — and ``sandars`` here is the closed form
        ``10 α² Z³ n*/(J(J+1)(2L+1))`` evaluated at Z=1, where that
        heavy-atom scaling law is not expected to hold.  The pinned ratio
        is atomx's own (snapshot); the *physics* statement that the two
        models disagree at Z=1 by more than 5x is asserted separately.
        """
        res = _solve_h()
        edm = sternheimer_edm_enhancement(res, kappa_max=4, n_virt_max=20)
        assert edm.scaling_comparison["sandars"] > 0
        ratio = edm.scaling_comparison["ratio"]
        assert isinstance(ratio, float)
        assert ratio > 5.0, (
            f"Sandars/SOS ratio for H is {ratio:.2f}; the module docstring's "
            f"factor-of-2-3 claim would need ratio in [2, 3]")
        assert pytest.approx(42.85, rel=0.05) == ratio, (
            f"H 1s R_e(SOS)/R_e(Sandars) = {ratio:.4f} (measured 42.85)")

    def test_value_error_on_bad_subshell(self):
        """Unknown target_subshell must raise ValueError, not silently return zero."""
        res = _solve_h()
        with pytest.raises(ValueError, match="Unknown target_subshell"):
            sternheimer_edm_enhancement(res, kappa_max=4, n_virt_max=20, target_subshell="bad")

    def test_tp_odd_angular_selection_rules(self):
        """Direct unit tests for the T,P-odd angular factor — instant, no SCF."""
        from atomx.properties.edm.sternheimer import _tp_odd_angular
        # s_{1/2} (kappa=-1) -> p_{3/2} (kappa=-2): Delta j != 0, must vanish
        assert _tp_odd_angular(-1, -2) == 0.0
        # p_{1/2} (kappa=+1) -> p_{3/2} (kappa=-2): Delta j != 0, must vanish
        assert _tp_odd_angular(+1, -2) == 0.0
        # s_{1/2} (kappa=-1) -> p_{1/2} (kappa=+1): Delta j = 0, must be non-zero
        assert abs(_tp_odd_angular(-1, +1)) > 0.0
        # s_{1/2} -> d_{3/2} (kappa=+2): Delta j != 0, must vanish
        assert _tp_odd_angular(-1, +2) == 0.0

    @pytest.mark.slow
    def test_fast_smoke_z_scaling(self):
        """Fast smoke variant of Z-scaling test with reduced virtual basis."""
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
        # Reduced n_virt_max for speed; still enough to see the boost.
        R_li = abs(sternheimer_edm_enhancement(
            li, kappa_max=4, n_virt_max=8, target_subshell="2s1/2").R_electron_edm)
        R_na = abs(sternheimer_edm_enhancement(
            na, kappa_max=4, n_virt_max=8, target_subshell="3s1/2").R_electron_edm)
        assert R_na > 3.0 * R_li, (
            f"Fast smoke: expected Z-scaling boost; got "
            f"R(Na)={R_na:.3e} vs R(Li)={R_li:.3e}")

    @pytest.mark.slow
    @pytest.mark.xfail(strict=True, reason=(
        "Same blocker as test_thallium_benchmark: the Tl SCF fixed point "
        "has residual 1.877e-07 (53x inside tol=1e-5) but "
        "max_same_kappa_overlap = 1.702e-03 > the 1e-03 threshold, so "
        "converged is False and sternheimer_edm_enhancement raises "
        "ValueError('SCFResult must be converged') — there is no fast-smoke "
        "R_e(Tl) either.  See test_thallium_blocker_is_orthogonality_not_"
        "the_residual."))
    def test_fast_smoke_thallium(self):
        """Fast smoke variant of Tl benchmark with reduced virtual basis."""
        res = _solve_thallium()
        assert res.converged
        edm = sternheimer_edm_enhancement(
            res, kappa_max=4, n_virt_max=8, target_subshell="6p1/2")
        R_e = abs(edm.R_electron_edm)
        # With reduced virtuals the result is rougher but must still be
        # in the hundreds regime, not ~0.1.
        assert R_e > 50.0, (
            f"Fast smoke Tl R_e={R_e:.3e} below hundreds; "
            f"relativistic boost not captured.")
        assert R_e < 5000.0, f"Fast smoke Tl R_e={R_e:.3e} unphysically large"
