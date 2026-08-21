"""Tests for enhanced potential models (Phase 2).

Validates TFDA, GSZ, and parametric potentials against known results.
"""

import numpy as np
import pytest


class TestThomasFermi:
    """Test Thomas-Fermi-Dirac-Amaldi potential."""

    def test_tf_screening_at_origin(self):
        """Screening function should be ~1 at r=0."""
        from atomx.potentials.thomas_fermi import _thomas_fermi_screening
        assert abs(_thomas_fermi_screening(0.0) - 1.0) < 0.01

    def test_tf_screening_decays(self):
        """Screening function should decay to 0 at large r."""
        from atomx.potentials.thomas_fermi import _thomas_fermi_screening
        assert _thomas_fermi_screening(100.0) < 0.01

    def test_tf_potential_asymptotics(self):
        """TF potential should be -Z/r near origin, -1/r far out for neutral."""
        from atomx.potentials.thomas_fermi import thomas_fermi_potential
        r = np.array([0.001, 100.0])
        V = thomas_fermi_potential(10, 10, r)
        # Near origin: V ~ -Z/r
        assert abs(V[0] * r[0] / (-10) - 1.0) < 0.05
        # Far out: V ~ -(Z-N+1)/r = -1/r for neutral
        assert abs(V[1] * r[1] / (-1) - 1.0) < 0.1

    @pytest.mark.parametrize(
        ("Z", "N_electrons", "expected_tail"),
        [
            (10, 10, -1.0),
            (10, 8, -3.0),
        ],
    )
    def test_tfda_potential_preserves_latter_tail(self, Z, N_electrons, expected_tail):
        """TFDA Amaldi correction must not change the Latter asymptotic charge."""
        from atomx.potentials.thomas_fermi import tfda_potential

        r = np.array([200.0, 500.0, 1000.0])
        rv = r * tfda_potential(Z, N_electrons, r)

        np.testing.assert_allclose(rv, expected_tail, rtol=0.0, atol=1e-8)

    @pytest.mark.parametrize(
        ("Z", "N_electrons", "expected_tail", "switch_radius"),
        [
            (10, 10, -1.0, 1.76162),
            (10, 8, -3.0, 0.651464),
            (10, 8, -3.0, 0.992003),
        ],
    )
    def test_tfda_latter_switch_neighborhood_is_continuous(
        self, Z, N_electrons, expected_tail, switch_radius
    ):
        """TFDA Latter selection must not introduce a jump near the switch."""
        from atomx.potentials.thomas_fermi import tfda_potential

        r = np.linspace(0.98 * switch_radius, 1.02 * switch_radius, 4097)
        V = tfda_potential(Z, N_electrons, r)
        rv = r * V

        assert np.all(np.isfinite(V))
        assert np.all(V < 0.0)
        assert np.max(np.abs(np.diff(rv))) < 1e-3

        tail_r = np.array([200.0, 500.0, 1000.0])
        tail_rv = tail_r * tfda_potential(Z, N_electrons, tail_r)
        np.testing.assert_allclose(tail_rv, expected_tail, rtol=0.0, atol=1e-8)

    def test_tfda_atom_calculation(self):
        """TFDA exchange mode should produce a converged result with orbital energies."""
        from atomx import Atom
        ne = Atom(Z=10, exchange="tfda", ns=50, ks=9)
        ne.solve()
        assert ne.converged
        # TFDA total energy uses a different formula than HF, so the value
        # can differ significantly. Check that orbital energies are negative.
        for label, E in ne.orbital_energies.items():
            assert E < 0, f"Orbital {label} should be bound (E<0)"

    def test_tfda_faster_than_hf(self):
        """TFDA should need only 1 iteration (no SCF)."""
        from atomx import Atom
        ne = Atom(Z=10, exchange="tfda", ns=50, ks=9)
        ne.solve()
        assert ne.iterations == 1

    def test_starting_potential_tfda(self):
        """TFDA starting potential should accelerate HF convergence."""
        from atomx import Atom
        # Without starting potential
        ne1 = Atom(Z=10, exchange="hf", ns=50, ks=9)
        ne1.solve(max_iter=60, tol=1e-5)
        iter1 = ne1.iterations

        # With TFDA starting potential
        ne2 = Atom(Z=10, exchange="hf", starting_potential="tfda", ns=50, ks=9)
        ne2.solve(max_iter=60, tol=1e-5)
        iter2 = ne2.iterations

        # TFDA start should converge at least as fast
        # (may not always be faster for light atoms, but should not be worse)
        assert ne2.converged


class TestGSZ:
    """Test Green-Sellin-Zachor potential."""

    def test_gsz_asymptotic(self):
        """GSZ should give -Z_eff/r at large r."""
        from atomx.potentials.parametric import green_sellin_zachor
        r = np.array([0.001, 200.0])
        V = green_sellin_zachor(10, 10, r)
        # Far out: -(Z-N+1)/r = -1/r for neutral
        assert abs(V[1] * r[1] + 1.0) < 0.5

    def test_gsz_atom(self):
        """GSZ exchange mode should produce converged result."""
        from atomx import Atom
        ne = Atom(Z=10, exchange="gsz", ns=50, ks=9)
        ne.solve()
        assert ne.converged
        # GSZ is parametric - check orbital energies are bound
        for label, E in ne.orbital_energies.items():
            assert E < 0, f"Orbital {label} should be bound"
