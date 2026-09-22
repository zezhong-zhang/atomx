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

    @pytest.mark.parametrize("N_electrons", [0.5, 1.0])
    def test_tfda_sub_one_electron_cases_are_bare_nuclear(self, N_electrons):
        """TFDA must not add screening for one-electron and sub-one-electron ions."""
        from atomx.potentials.thomas_fermi import tfda_potential

        r = np.geomspace(1e-6, 1e3, 128)
        V = tfda_potential(10, N_electrons, r)

        assert np.all(np.isfinite(V))
        np.testing.assert_allclose(V, -10.0 / r, rtol=0.0, atol=0.0)

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
        from atomx import Atom, Basis
        ne = Atom(Z=10, exchange="tfda", basis=Basis(ns=50, ks=9))
        ne.solve()
        assert ne.converged
        # TFDA total energy uses a different formula than HF, so the value
        # can differ significantly. Check that orbital energies are negative.
        for label, E in ne.orbital_energies.items():
            assert E < 0, f"Orbital {label} should be bound (E<0)"

    def test_tfda_faster_than_hf(self):
        """TFDA should need only 1 iteration (no SCF)."""
        from atomx import Atom, Basis
        ne = Atom(Z=10, exchange="tfda", basis=Basis(ns=50, ks=9))
        ne.solve()
        assert ne.iterations == 1

    def test_tfda_starting_potential_does_not_change_the_answer(self):
        """Whatever the start does, the converged fixed point must be the same.

        Measured 2026-09-22: E = -128.6299783505843 Ha bit-identical with
        ``starting_potential=None`` and ``starting_potential="tfda"``.
        """
        from atomx import Atom, Basis
        ne1 = Atom(Z=10, exchange="hf", basis=Basis(ns=50, ks=9))
        ne1.solve(max_iter=60, tol=1e-5)
        ne2 = Atom(
                   Z=10,
                   exchange="hf",
                   basis=Basis(starting_potential="tfda", ns=50, ks=9),
              )
        ne2.solve(max_iter=60, tol=1e-5)
        assert ne2.converged
        assert ne2.total_energy == ne1.total_energy

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The TFDA starting potential is silently IGNORED on the "
            "Atom/Basis route: Ne HF converges in 12 iterations and to the "
            "bit-identical total energy -128.6299783505843 Ha with "
            "starting_potential=None, 'tfda' and 'gsz' alike, so it cannot "
            "accelerate anything.  Named cause: DFSolver consumes "
            "starting_potential only in its first hydrogenic eigensolve "
            "(dirac_fock.py:1726-1777), and the MCDHF/Atom route skips that "
            "branch because it warm-starts from seed_orbitals; the option is "
            "live only for a directly constructed DFSolver.  The test that "
            "used to live here asserted `<=` and passed on 12 vs 12."
        ),
    )
    def test_tfda_starting_potential_accelerates_hf_convergence(self):
        """TFDA starting potential should *strictly* accelerate HF convergence."""
        from atomx import Atom, Basis
        ne1 = Atom(Z=10, exchange="hf", basis=Basis(ns=50, ks=9))
        ne1.solve(max_iter=60, tol=1e-5)
        ne2 = Atom(
                   Z=10,
                   exchange="hf",
                   basis=Basis(starting_potential="tfda", ns=50, ks=9),
              )
        ne2.solve(max_iter=60, tol=1e-5)
        assert ne2.iterations < ne1.iterations, (
            f"tfda start converged in {ne2.iterations} iterations vs "
            f"{ne1.iterations} for the default start"
        )


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
        from atomx import Atom, Basis
        ne = Atom(Z=10, exchange="gsz", basis=Basis(ns=50, ks=9))
        ne.solve()
        assert ne.converged
        # GSZ is parametric - check orbital energies are bound
        for label, E in ne.orbital_energies.items():
            assert E < 0, f"Orbital {label} should be bound"
