"""Tests for radiative transition properties.

Validates:
- Angular selection rules
- Radial matrix elements
- E1 A-coefficient for H 2p->1s (NIST value: 6.2649e8 s^-1) — the gate
  ``docs/VERIFICATION.md:29`` names for this file
- Single-config transitions via Atom API

(The gauge consistency this module used to claim is gated in
``tests/test_gauge_invariance.py`` and
``tests/test_velocity_gauge_full_order.py``.)
"""

from types import SimpleNamespace

import numpy as np
import pytest

from atomx.methods.hydrogenic_reference import hydrogenic_dirac_wavefunction
from atomx.properties.transitions.angular import (
    allowed_final_kappas,
    angular_eL,
    selection_rules,
)
from atomx.properties.transitions.multipole import radial_eL
from atomx.properties.transitions.rates import (
    _double_factorial,
    einstein_A_eL,
    lifetime,
    oscillator_strength,
)

#: NIST ASD / standard Ly-alpha A-coefficient: A(H 2p -> 1s) = 6.2649e8 s^-1.
LY_ALPHA_A_NIST_PER_S = 6.2649e8
#: The value this file used to pass with (HYP-144 pre-fix defect, commit
#: 621fe99): the *lower* state degeneracy 2j_l+1 was used in place of
#: 2j_u+1, giving 2.0000x the correct rate.  Named here so the regression
#: is detectably excluded rather than merely implied by the NIST band.
LY_ALPHA_A_PREFIX_BUG_PER_S = 1.253677e9

# ===== Angular module tests =====

class TestSelectionRules:
    """Test E1/M1/E2 selection rules."""

    def test_e1_s_to_p_allowed(self):
        """E1: s1/2 -> p1/2 (kappa=-1 -> kappa=1) should be allowed."""
        assert selection_rules(-1, 1, 1, 'E') is True

    def test_e1_s_to_p32_allowed(self):
        """E1: s1/2 -> p3/2 (kappa=-1 -> kappa=-2) should be allowed."""
        assert selection_rules(-1, -2, 1, 'E') is True

    def test_e1_s_to_s_forbidden(self):
        """E1: s1/2 -> s1/2 (kappa=-1 -> kappa=-1) should be forbidden."""
        assert selection_rules(-1, -1, 1, 'E') is False

    def test_e1_p_to_d_allowed(self):
        """E1: p3/2 -> d5/2 (kappa=-2 -> kappa=-3) should be allowed."""
        assert selection_rules(-2, -3, 1, 'E') is True

    def test_m1_s_to_s_allowed(self):
        """M1: s1/2 -> s1/2 (kappa=-1 -> kappa=-1): l=0+0+1=1 odd -> allowed by parity."""
        # But triangle: |j_i - j_f| = 0 <= L=1 <= j_i + j_f = 1 -> allowed
        # However same state -> no energy difference in practice, but selection rules allow it
        assert selection_rules(-1, -1, 1, 'M') is True

    def test_m1_p12_to_p32(self):
        """M1: p1/2 -> p3/2 (kappa=1 -> kappa=-2): l=1+1+1=3 odd -> allowed."""
        assert selection_rules(1, -2, 1, 'M') is True

    def test_e2_s_to_d_allowed(self):
        """E2: s1/2 -> d3/2 (kappa=-1 -> kappa=2) should be allowed."""
        assert selection_rules(-1, 2, 2, 'E') is True

    def test_e2_s_to_d52_allowed(self):
        """E2: s1/2 -> d5/2 (kappa=-1 -> kappa=-3) should be allowed."""
        assert selection_rules(-1, -3, 2, 'E') is True

    # --- Higher multipoles E3 / M3 ------------------------------------

    def test_e3_s_to_f_allowed(self):
        """E3: s1/2 -> f5/2 (kappa=-1 -> kappa=3) should be allowed
        (parity: l_i=0, l_f=3; 0+3+3=6 even → E parity OK; triangle
        |1/2 - 5/2|=2 ≤ 3 ≤ 3 ✓)."""
        assert selection_rules(-1, 3, 3, 'E') is True

    def test_e3_s_to_p_forbidden_by_triangle(self):
        """E3: s1/2 -> p1/2 fails the triangle inequality
        (|1/2−1/2|=0, 1/2+1/2=1 < 3)."""
        assert selection_rules(-1, 1, 3, 'E') is False

    def test_m3_p_to_f_allowed(self):
        """M3: p3/2 -> f7/2 (kappa=-2 -> kappa=-4) should be allowed
        (parity: l_i=1, l_f=3; 1+3+3=7 odd → M parity OK)."""
        assert selection_rules(-2, -4, 3, 'M') is True

    def test_m3_s_to_s_forbidden_by_triangle(self):
        """M3: s1/2 -> s1/2 fails the TRIANGLE rule, not parity.

        Parity is satisfied for a magnetic multipole (l_i+l_f+L = 0+0+3
        is odd), but the angular momenta cannot couple: |1/2 − 1/2| = 0
        and j_i + j_f = 1 < L = 3.  (This test used to be named
        ``test_m3_s_to_d_forbidden_by_parity`` while its own comment said
        "Actually this is allowed by both" — s1/2 → d5/2 IS allowed for
        M3, and is asserted below.)
        """
        assert selection_rules(-1, -1, 3, 'M') is False
        # The allowed M3 case the old name described: s1/2 → d5/2
        # (parity 0+2+3 odd, triangle 2 ≤ 3 ≤ 3).
        assert selection_rules(-1, -3, 3, 'M') is True

    def test_angular_e3_nonzero_for_allowed(self):
        """Angular factor for allowed E3 should be nonzero."""
        ang = angular_eL(-1, 3, 3)   # s1/2 -> f5/2
        assert abs(ang) > 1e-10


class TestAllowedKappas:
    """Test generation of allowed final kappa values."""

    def test_e1_from_s12(self):
        """E1 from s1/2 (kappa=-1): should give p1/2 and p3/2."""
        kappas = allowed_final_kappas(-1, 1, 'E')
        assert 1 in kappas     # p1/2
        assert -2 in kappas    # p3/2

    def test_e1_from_p32(self):
        """E1 from p3/2 (kappa=-2): should give s1/2 and d3/2, d5/2."""
        kappas = allowed_final_kappas(-2, 1, 'E')
        assert -1 in kappas    # s1/2
        assert 2 in kappas     # d3/2
        assert -3 in kappas    # d5/2


class TestAngularFactors:
    """Test angular coupling coefficients."""

    def test_angular_eL_nonzero_for_allowed(self):
        """Angular factor should be nonzero for allowed E1 transitions."""
        ang = angular_eL(-1, 1, 1)  # s1/2 -> p1/2
        assert abs(ang) > 1e-10

    def test_angular_eL_zero_for_forbidden(self):
        """Angular factor should be zero for forbidden E1 transitions."""
        ang = angular_eL(-1, -1, 1)  # s1/2 -> s1/2
        assert abs(ang) < 1e-15


def test_closed_shell_e1_particle_hole_obeys_one_electron_sum_rule(monkeypatch):
    """A filled shell has unit total strength per occupied magnetic spinor."""
    from atomx.methods.dirac_fock import Subshell
    from atomx.config.csf import CSF
    from atomx.properties.transitions import calculator as calc_mod
    from atomx.properties.transitions.calculator import TransitionCalculator

    lower = CSF([
        Subshell(1, -1, 2.0),
        Subshell(2, -1, 2.0),
        Subshell(2, 1, 2.0),
        Subshell(2, -2, 4.0),
    ], total_J2=0, parity=1, label="closed [J=0/2]")
    upper = CSF([
        Subshell(1, -1, 2.0),
        Subshell(2, -1, 2.0),
        Subshell(2, 1, 2.0),
        Subshell(2, -2, 3.0),
        Subshell(3, 2, 1.0),
    ], total_J2=2, parity=-1, label="2p3/2^-1 3d3/2 [J=2/2]")

    monkeypatch.setattr(calc_mod, "radial_eL", lambda *_args, **_kwargs: 1.0)
    atom = SimpleNamespace(converged=True, Z=26)
    calc = TransitionCalculator(atom)
    orbitals = {
        (2, -2): (np.ones(2), np.zeros(2)),
        (3, 2): (np.ones(2), np.zeros(2)),
    }

    me = calc._csf_transition_me(
        upper, lower, orbitals, {}, 1, "E", "length", omega=1.0,
    )

    # Summing |<j_f m_f|C^1_q|j_i m_i>|² over the filled initial shell
    # and all final m,q gives |<j_f||C^1||j_i>|², without a sqrt(2).
    from atomx.angular.coefficients import reduced_C
    assert me ** 2 == pytest.approx(reduced_C(2, -2, 1) ** 2)


# ===== Rates module tests =====

class TestRates:
    """Test rate formulas."""

    def test_double_factorial(self):
        """Check double factorial: (2*1+1)!! = 3!! = 3."""
        assert _double_factorial(3) == 3  # 1*3 = 3
        assert _double_factorial(5) == 15  # 1*3*5

    def test_lifetime_single_channel(self):
        """Lifetime of single decay channel."""
        tau = lifetime([1e8])
        assert abs(tau - 1e-8) < 1e-15

    def test_lifetime_multiple_channels(self):
        """Lifetime with two equal channels = half the single-channel lifetime."""
        tau = lifetime([1e8, 1e8])
        assert abs(tau - 0.5e-8) < 1e-15

    def test_oscillator_strength_positive(self):
        """Oscillator strength should be positive for allowed transitions."""
        f, gf = oscillator_strength(0.5, 1.0, 1, 1, 1)
        assert f > 0
        assert gf > 0


# ===== NIST-anchored E1 gate (the one VERIFICATION.md:29 names) =====

class TestHydrogenicLyAlpha:
    """H 2p -> 1s E1 A-coefficient against the NIST/standard value.

    ``docs/VERIFICATION.md:29`` names this file as the gate for the
    "Hydrogen 2p->1s A coefficient".  Before this class the file's
    strongest A-rate assertion was ``A > 0`` (``test_neon_transitions``),
    which the pre-fix buggy rate 1.253677e9 s^-1 also satisfied.
    """

    _NS = 80
    _KS = 9

    def _ly_alpha_A(self, kappa_up: int) -> tuple[float, float]:
        """Return (A [s^-1], omega [Ha]) for H 2p_{kappa_up} -> 1s_{1/2}."""
        P_1s, Q_1s, bsp, E_1s = hydrogenic_dirac_wavefunction(
            1, -1, 1, ns=self._NS, ks=self._KS)
        P_2p, Q_2p, _, E_2p = hydrogenic_dirac_wavefunction(
            2, kappa_up, 1, ns=self._NS, ks=self._KS)
        omega = abs(E_2p - E_1s)
        rme = radial_eL(P_2p, Q_2p, P_1s, Q_1s, bsp, 1)
        ang = angular_eL(kappa_up, -1, 1)
        return einstein_A_eL(omega, (ang * rme) ** 2, 1, 2 * abs(kappa_up) - 1), omega

    @pytest.mark.parametrize("kappa_up", [1, -2], ids=["2p1/2", "2p3/2"])
    def test_h_2p_to_1s_matches_nist(self, kappa_up):
        """Both fine-structure components agree with NIST to 0.2 %.

        Measured: A(2p1/2) = 6.268313e8, A(2p3/2) = 6.268384e8 s^-1 at
        ns=80/ks=9 — 5.6e-4 above the NIST 6.2649e8 (the residual is the
        finite-nuclear-mass / radiative correction absent from the
        point-nucleus Dirac value, not a numerical error).
        """
        A, _ = self._ly_alpha_A(kappa_up)
        assert pytest.approx(LY_ALPHA_A_NIST_PER_S, rel=2e-3) == A, (
            f"H 2p_{kappa_up} -> 1s A = {A:.6e} s^-1 vs NIST "
            f"{LY_ALPHA_A_NIST_PER_S:.4e}")
        # The pre-fix value must stay detectably excluded (it is 2.00x NIST).
        assert abs(A / LY_ALPHA_A_PREFIX_BUG_PER_S - 1.0) > 0.5, (
            f"A = {A:.6e} s^-1 is within 50 % of the pre-fix buggy rate "
            f"{LY_ALPHA_A_PREFIX_BUG_PER_S:.6e} (wrong state degeneracy)")

    def test_ly_alpha_components_are_degenerate(self):
        """The two Ly-alpha components have equal A to the O(alpha^2) level.

        A = (4/3) alpha^3 omega^3 S/(2j_u+1) with S_3/2 = 2 S_1/2 and
        omega_3/2 = omega_1/2 to fine structure, so A_3/2/A_1/2 = 1 exactly
        (measured 1.0000114, the omega^3 fine-structure splitting).  The
        pre-fix bug used 2j_l+1 for the 2p3/2 row and gave ratio 2.0000.
        """
        A_12, w_12 = self._ly_alpha_A(1)
        A_32, w_32 = self._ly_alpha_A(-2)
        assert pytest.approx(1.0, rel=1e-4) == A_32 / A_12, (
            f"A(2p3/2)/A(2p1/2) = {A_32 / A_12:.6f} (pre-fix bug: 2.0000)")
        # The residual is exactly the omega^3 fine-structure factor.
        assert pytest.approx((w_32 / w_12) ** 3, rel=1e-4) == A_32 / A_12


# ===== Integration test with Atom =====

class TestAtomTransitions:
    """Integration test: compute transitions through Atom API."""

    def test_he_has_transitions(self):
        """Helium with 2 configs should produce E1 transitions."""
        from atomx import Atom
        he = Atom(Z=2, config=["1s2", "1s1 2p1"])
        he.solve(max_iter=40, tol=1e-4)
        if not he.converged:
            pytest.skip("He MCDHF did not converge")
        transitions = he.transitions(multipoles=['E1'])
        # Should find at least one transition
        assert len(transitions) > 0

    def test_hydrogen_e1_basic(self):
        """Hydrogen should produce orbital-to-orbital transitions."""
        from atomx import Atom
        # H atom: single 1s electron — default engine (shooting); the
        # transition machinery consumes grid orbitals through the
        # numerics.radial_workspace seam (see test_transitions_engine_uniform).
        h = Atom(Z=1, config="1s1")
        h.solve(max_iter=40)
        if not h.converged:
            pytest.skip("H did not converge")
        # Single-config H has only 1s orbital, no transitions possible
        # between occupied orbitals (only 1 orbital)
        transitions = h.transitions()
        # With only 1s, there are no other occupied orbitals to transition to
        assert len(transitions) == 0

    def test_neon_transitions(self):
        """Neon should have E1 transitions between subshells."""
        from atomx import Atom
        # Default engine (shooting) — cross-engine rate agreement is
        # gated in test_transitions_engine_uniform.py.
        ne = Atom(Z=10)
        ne.solve(max_iter=50, tol=1e-4)
        if not ne.converged:
            pytest.skip("Ne did not converge")
        transitions = ne.transitions(multipoles=['E1'])
        # Ne has 1s, 2s, 2p1/2, 2p3/2 subshells
        # E1 allowed: 2s -> 2p and 1s -> 2p (2p -> 1s emission, etc.)
        assert len(transitions) > 0
        # All A-rates should be positive
        for tr in transitions:
            assert tr.A_rate_per_s > 0
            assert tr.wavelength_angstrom > 0
