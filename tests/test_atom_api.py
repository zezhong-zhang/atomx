"""Tests for the high-level Atom API."""

import pytest
import numpy as np
from atomx import Atom


class TestAtomAPI:
    def test_helium_simple(self):
        """Simplest use case: Atom(Z=2).solve()"""
        he = Atom(Z=2)
        he.solve()
        assert he.converged
        assert he.total_energy == pytest.approx(-2.86, abs=0.1)

    def test_neon_default_config(self):
        """Ne uses built-in ground state config."""
        ne = Atom(Z=10)
        ne.solve()
        assert ne.converged
        assert ne.total_energy == pytest.approx(-128.5, abs=2.0)

    def test_custom_config(self):
        """Custom configuration string."""
        li = Atom(Z=3, config="1s2 2s1")
        li.solve()
        assert li.converged
        assert li.total_energy < 0

    def test_ion(self):
        """Ionized atom: He+ (Z=2, charge=1)."""
        he_plus = Atom(Z=2, charge=1)
        he_plus.solve()
        assert he_plus.converged
        # He+ is hydrogenic, E ≈ -Z^2/2 = -2.0 (1 electron, no e-e interaction)
        assert he_plus.total_energy == pytest.approx(-2.0, abs=0.5)

    def test_with_qed(self):
        """QED corrections included."""
        he = Atom(Z=2, qed=True)
        he.solve()
        assert "E_VP" in he.energy_components
        assert "E_SE" in he.energy_components

    def test_mcdhf(self):
        """Multi-configuration calculation produces J-resolved levels.

        Be ``["1s2 2s2", "1s2 2p2"]``:
          - ``1s2 2s2`` → 1 closed-shell CSF (J=0).
          - ``1s2 2p2`` → 5 J-resolved CSFs (jj-splits (0,2), (1,1), (2,0)
            with Pauli-allowed J multiplets {0,2}, {1,2}, {0}).
        Public ``Atom(configs=[...])`` now uses the J-resolved
        enumeration and block-diagonalises CI by (J, parity).
        """
        be = Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"])
        be.solve()
        assert be.converged
        # 1 + 5 = 6 J-resolved CSFs
        assert len(be.state_energies) == 6
        # Ground state must be lowest and well-separated
        assert be.state_energies[0] < be.state_energies[1]
        # The MCDHFResult exposes J-resolved labels
        twoJ = be._result.state_J2
        par = be._result.state_parity
        assert len(twoJ) == 6
        assert twoJ[0] == 0  # ground state J=0 (1s2 2s2 closed shell)
        # All these CSFs are even parity (l=0,1; q even on 2s², l*q even
        # on 2p² since l=1 q=2 gives parity +1)
        assert all(p == 1 for p in par)

    def test_chaining(self):
        """solve() returns self for chaining."""
        E = Atom(Z=2).solve().total_energy
        assert E < 0

    def test_repr(self):
        he = Atom(Z=2)
        assert "He" in repr(he)
        assert "unsolved" in repr(he)
        he.solve()
        assert "solved" in repr(he)

    def test_orbital_energies_dict(self):
        """orbital_energies returns dict with string keys."""
        ne = Atom(Z=10)
        ne.solve()
        oe = ne.orbital_energies
        assert isinstance(oe, dict)
        assert "1s1/2" in oe
        assert oe["1s1/2"] < oe["2s1/2"]  # 1s more bound

    def test_ev_conversion(self):
        he = Atom(Z=2).solve()
        assert he.total_energy_eV == pytest.approx(he.total_energy * 27.21, abs=1.0)

    def test_iron(self):
        """Fe (Z=26) — key for spectroscopy. Should converge."""
        fe = Atom(Z=26, ns=60, ks=9)
        fe.solve(tol=1e-4)
        assert fe.converged
        assert fe.total_energy < 0
        # Fe DHF total energy: ~-1263 Hartree (Desclaux 1973)
        assert fe.total_energy < -1000

    def test_argon(self):
        """Ar (Z=18) — noble gas, should converge easily."""
        ar = Atom(Z=18)
        ar.solve()
        assert ar.converged
        assert ar.total_energy < -450  # Ar DHF ~-528 Ha (basis-limited)
