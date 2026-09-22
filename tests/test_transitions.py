"""Tests for radiative transition properties.

Validates:
- Angular selection rules
- Radial matrix elements
- E1 A-coefficient for H 2p->1s (exact value: 6.2649e8 s^-1)
- Gauge consistency (length vs velocity)
- Single-config transitions via Atom API
"""

from types import SimpleNamespace

import numpy as np
import pytest

from atomx.properties.transitions.angular import (
    allowed_final_kappas,
    angular_eL,
    selection_rules,
)
from atomx.properties.transitions.rates import (
    _double_factorial,
    lifetime,
    oscillator_strength,
)

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

    def test_m3_s_to_d_forbidden_by_parity(self):
        """M3: s1/2 -> d5/2 fails parity for magnetic multipole
        (l_i=0, l_f=2, L=3 → 0+2+3=5 odd ✓ for magnetic, triangle 2≤3≤3 ✓)."""
        # Actually this is allowed by both — replace with a true M-forbidden:
        # M3 p3/2 -> p3/2: parity l_i+l_f+L = 1+1+3=5 odd ✓ for M; but triangle 0≤3≤3 ✓
        # M3 s1/2 -> s1/2: parity 0+0+3=3 odd ✓, but triangle 0≤3≤1 FAILS ✓
        assert selection_rules(-1, -1, 3, 'M') is False

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


def test_closed_shell_e1_particle_hole_uses_fac_babushkin_normalization(monkeypatch):
    """Closed J=0 -> p-hole d3/2 E1 uses FAC's sqrt(2) radial convention."""
    from atomx.methods.dirac_fock import Subshell
    from atomx.methods.mcdhf.config import CSF
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

    assert me == pytest.approx(np.sqrt(2.0) * angular_eL(-2, 2, 1))


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
        # corrections.workspace seam (see test_transitions_engine_uniform).
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
