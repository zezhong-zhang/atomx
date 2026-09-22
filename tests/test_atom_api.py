"""Tests for the high-level Atom API.

Where a closed form or an external code exists, the gates below use it rather
than a wide band: the He II 1s energy is the Sommerfeld value
``C^2 (gamma - 1)``, and the He/Ne/Ar/Fe total energies are the live DBSR_HF
Coulomb energies quoted in ``benchmarks/EXTERNAL_VERIFICATION.md`` (the code
this package is modelled on).  All of them are taken with ``corrections="none"``
so the comparison is not mixed with the Breit/QED terms — whose scalar is
currently 3.03x the DBSR_HF value, see
``tests/test_nist_levels.py::test_breit_scalar_matches_dbsr_hf``.
"""

import numpy as np
import pytest

from atomx import Atom, Basis
from atomx.constants import ALPHA, HARTREE_EV, C


class TestAtomAPI:
    def test_helium_simple(self):
        """Simplest use case: Atom(Z=2).solve()"""
        he = Atom(Z=2, corrections="none")
        he.solve()
        assert he.converged
        # DBSR_HF (live ``dbsr_hf an=2 term=AV mbreit=0``,
        # benchmarks/EXTERNAL_VERIFICATION.md section 1): He Coulomb
        # -2.86181332 Ha.  Measured here: -2.8618133209580385 (3e-10 Ha).
        assert he.total_energy == pytest.approx(-2.86181332, abs=1e-6)

    def test_neon_default_config(self):
        """Ne uses built-in ground state config."""
        ne = Atom(Z=10, corrections="none")
        ne.solve(tol=1e-11, max_iter=60)
        assert ne.converged
        # DBSR_HF Ne Coulomb -128.69192580 Ha (same source as above);
        # measured -128.69192580015647 at tol=1e-11.  The default tol=1e-5
        # stops 8e-6 Ha higher, which is why the SCF is converged here.
        assert ne.total_energy == pytest.approx(-128.69192580, abs=1e-6)

    def test_custom_config(self):
        """Custom configuration string, bracketed by two independent bounds."""
        li = Atom(Z=3, config="1s2 2s1")
        li.solve()
        assert li.converged
        # Lower bound (variational): the non-interacting hydrogenic sum for
        # 1s2 2s1 at Z=3 is 2*(-9/2) + (-9/8) = -10.125 Ha, and the
        # electron-electron repulsion is positive, so E > -10.125.
        # Upper bound: the non-relativistic Hartree-Fock limit -7.432727 Ha
        # (Clementi & Roetti, At. Data Nucl. Data Tables 14, 177 (1974)),
        # which the ~6e-5 relativistic lowering measured here sits below.
        assert -10.125 < li.total_energy < -7.432727 + 1e-3

    def test_ion(self):
        """Ionized atom: He+ (Z=2, charge=1) is a one-electron ion."""
        he_plus = Atom(Z=2, charge=1, corrections="none")
        he_plus.solve()
        assert he_plus.converged
        # Exact hydrogenic Dirac 1s: E = C^2 (gamma - 1) with
        # gamma = sqrt(kappa^2 - (Z alpha)^2), |kappa| = 1 for 1s1/2.
        gamma = np.sqrt(1.0 - (2 * ALPHA) ** 2)
        exact = C ** 2 * (gamma - 1.0)
        # Measured -2.000106498786554 vs -2.0001065140528396 exact, i.e. a
        # relative error of 7.6e-9 (the gate below is 13x that).
        assert abs(he_plus.total_energy / exact - 1.0) < 1e-7

    def test_with_qed(self):
        """QED corrections included."""
        he = Atom(Z=2)
        he.solve()
        assert "E_VP" in he.energy_components
        assert "E_SE" in he.energy_components

    def test_mcdhf(self):
        """Multi-configuration calculation produces J-resolved levels.

        Be ``["1s2 2s2", "1s2 2p2"]``:
          - ``1s2 2s2`` → 1 closed-shell CSF (J=0).
          - ``1s2 2p2`` → 5 J-resolved CSFs (jj-splits (0,2), (1,1), (2,0)
            with Pauli-allowed J multiplets {0,2}, {1,2}, {0}).
        Public ``Atom(config=[...])`` now uses the J-resolved
        enumeration and block-diagonalises CI by (J, parity).
        """
        be = Atom(Z=4, config=["1s2 2s2", "1s2 2p2"])
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
        """solve() returns self for chaining (the identity the API promises)."""
        atom = Atom(Z=2, corrections="none")
        solved = atom.solve()
        assert solved is atom
        assert solved.total_energy == atom.total_energy
        assert atom.converged

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
        """The Ha -> eV conversion uses the single library constant."""
        he = Atom(Z=2).solve()
        assert he.total_energy_eV == pytest.approx(
            he.total_energy * HARTREE_EV, rel=1e-12)

    def test_iron(self):
        """Fe (Z=26) — key for spectroscopy. Should converge."""
        fe = Atom(Z=26, basis=Basis(ns=60, ks=9), corrections="none")
        fe.solve(tol=1e-4)
        assert fe.converged
        # Live DBSR_HF ``an=26 term=AV mbreit=0`` gives the Coulomb energy
        # -1271.414194275546834 Ha.  The 0.5 Ha window is the *basis*
        # truncation (ns=60 of 113 splines) measured here as -1271.5441; it
        # replaces the previous ``< -1000`` (271 Ha of slack).
        assert fe.total_energy == pytest.approx(-1271.414194275546834, abs=0.5)

    def test_argon(self):
        """Ar (Z=18) — noble gas, should converge easily."""
        ar = Atom(Z=18, corrections="none")
        ar.solve(tol=1e-11, max_iter=60)
        assert ar.converged
        # DBSR_HF Ar Coulomb -528.68377257 Ha
        # (benchmarks/EXTERNAL_VERIFICATION.md section 1); measured
        # -528.6837144 — the 5.8e-5 gap is the shooting engine's default
        # 1000-point grid, not the SCF (it does not move with tol).
        assert ar.total_energy == pytest.approx(-528.68377257, abs=5e-4)
