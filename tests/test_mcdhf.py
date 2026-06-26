"""M5 tests: MCDHF (Multi-Configuration Dirac-Hartree-Fock).

Tests CSF parsing, CI matrix construction, and MCDHF solver.
"""

import numpy as np
import pytest
from atomx.methods.mcdhf.config import (
    CSF,
    enumerate_j_resolved_csfs,
    make_csf_list,
    _parse_config_string,
    get_shared_orbital_basis,
)
from atomx.methods.mcdhf.solver import MCDHFSolver, MCDHFResult
from atomx.methods.dirac_fock import Subshell
from atomx.constants import HARTREE_CM1


class TestCSFParsing:
    def test_parse_helium(self):
        """Parse '1s2' → one subshell."""
        shells = _parse_config_string("1s2")
        assert len(shells) == 1
        assert shells[0].n == 1
        assert shells[0].kappa == -1
        assert shells[0].occupation == 2

    def test_parse_neon(self):
        """Parse '1s2 2s2 2p6' → 1s, 2s, 2p1/2, 2p3/2."""
        shells = _parse_config_string("1s2 2s2 2p6")
        assert sum(sh.occupation for sh in shells) == 10

    def test_parse_excited(self):
        """Parse excited config '1s2 2s1 2p6 3s1'."""
        shells = _parse_config_string("1s2 2s1 2p6 3s1")
        assert sum(sh.occupation for sh in shells) == 10

    def test_make_csf_list(self):
        csfs = make_csf_list(10, ["1s2 2s2 2p6", "1s2 2s1 2p6 3s1"])
        assert len(csfs) == 2
        assert csfs[0].n_electrons == 10
        assert csfs[1].n_electrons == 10

    def test_config_diff(self):
        """Two configs that differ by 1 electron (2s→3s)."""
        csfs = make_csf_list(10, ["1s2 2s2 2p6", "1s2 2s1 2p6 3s1"])
        diffs = csfs[0].differs_from(csfs[1])
        assert len(diffs) >= 2  # 2s occupation differs, 3s appears

    def test_n_electron_diff(self):
        csfs = make_csf_list(10, ["1s2 2s2 2p6", "1s2 2s1 2p6 3s1"])
        assert csfs[0].n_electron_diff(csfs[1]) == 1

    def test_shared_orbital_basis(self):
        csfs = make_csf_list(4, ["1s2 2s2", "1s2 2s1 2p1"])
        basis = get_shared_orbital_basis(csfs)
        # Should have 1s, 2s, and at least one 2p orbital
        n_kappa_set = {(n, k) for n, k in basis}
        assert (1, -1) in n_kappa_set  # 1s
        assert (2, -1) in n_kappa_set  # 2s


class TestMCDHFSolver:
    """Test MCDHF solver with simple cases."""

    def test_single_config_matches_dhf(self):
        """MCDHF with 1 CSF should match DHF."""
        csfs = make_csf_list(2, ["1s2"])
        solver = MCDHFSolver(Z=2, csfs=csfs, ns=40, ks=9)
        result = solver.solve(max_iter=5, scf_tol=1e-4, ci_tol=1e-6)
        # Single CSF → ground energy = DHF energy
        assert result.n_states == 1
        assert result.ground_energy == pytest.approx(-2.86, abs=0.1)

    def test_two_config_be(self):
        """Be with 2 CSFs: 1s2 2s2 + 1s2 2p2 (2s-2p near-degeneracy)."""
        csfs = make_csf_list(4, ["1s2 2s2", "1s2 2p2"])
        solver = MCDHFSolver(Z=4, csfs=csfs, ns=40, ks=9)
        result = solver.solve(max_iter=10, scf_tol=1e-4, ci_tol=1e-6)
        assert result.n_states == 2
        # Ground state should be lower than either diagonal element
        assert result.state_energies[0] < result.state_energies[1]
        # Both states should be bound
        assert result.ground_energy < 0


    def test_per_block_orbital_sets_are_populated_when_enabled(self):
        """Opt-in DODS/EOL scaffold produces block-keyed orbital sets."""
        csfs = []
        for cfg in ["1s2", "1s1 2p1"]:
            csfs.extend(enumerate_j_resolved_csfs(cfg))
        solver = MCDHFSolver(
            Z=6,
            csfs=csfs,
            ns=24,
            ks=7,
            use_physics_grid=False,
            per_block_orbital_sets=True,
        )
        result = solver.solve(max_iter=3, scf_tol=1e-3, ci_tol=1e-5, scf_max_iter=8)
        assert result.per_block_orbitals
        assert all(isinstance(block, tuple) and len(block) == 2 for block in result.per_block_orbitals)
        assert all(result.per_block_orbitals[block] for block in result.per_block_orbitals)

    def test_excitation_energies(self):
        """Excitation energies should be positive."""
        csfs = make_csf_list(4, ["1s2 2s2", "1s2 2p2"])
        solver = MCDHFSolver(Z=4, csfs=csfs, ns=40, ks=9)
        result = solver.solve(max_iter=10, scf_tol=1e-4, ci_tol=1e-6)
        de = result.excitation_energies(unit="cm-1")
        assert de[0] == 0.0  # ground state
        if len(de) > 1:
            assert de[1] > 0  # excited states above ground

    def test_mixing_coefficients(self):
        """Mixing coefficients should be normalized."""
        csfs = make_csf_list(4, ["1s2 2s2", "1s2 2p2"])
        solver = MCDHFSolver(Z=4, csfs=csfs, ns=40, ks=9)
        result = solver.solve(max_iter=10, scf_tol=1e-4, ci_tol=1e-6)
        for state in range(result.n_states):
            norm = np.sum(result.mixing_coefficients[:, state] ** 2)
            assert norm == pytest.approx(1.0, abs=1e-10)


def test_fac_frozen_radial_optimization_keeps_virtual_occupations_tiny():
    from atomx.methods.mcdhf.config import enumerate_j_resolved_csfs

    csfs = []
    for cfg in ["1s2 2s2 2p6", "1s2 2s2 2p5 3d1"]:
        csfs.extend(enumerate_j_resolved_csfs(cfg))

    solver = MCDHFSolver(
        Z=26,
        csfs=csfs,
        ns=35,
        ks=7,
        use_physics_grid=False,
        radial_optimization="fac_frozen",
    )
    subshells = solver._fac_frozen_subshells()
    occ = {(sh.n, sh.kappa): sh.occupation for sh in subshells}

    assert occ[(2, 1)] == 2
    assert occ[(2, -2)] == 4
    assert occ[(3, 2)] == 0.001
    assert occ[(3, -3)] == 0.001


def test_fac_frozen_local_radial_optimization_solves_with_frozen_occupations():
    from atomx.methods.mcdhf.config import enumerate_j_resolved_csfs

    csfs = []
    for cfg in ["1s2", "1s1 2p1"]:
        csfs.extend(enumerate_j_resolved_csfs(cfg))

    solver = MCDHFSolver(
        Z=6,
        csfs=csfs,
        ns=28,
        ks=7,
        use_physics_grid=False,
        radial_optimization="fac_frozen_local",
    )
    result = solver.solve(max_iter=4, scf_tol=1e-3, ci_tol=1e-5, scf_max_iter=8)

    assert result.converged
    assert result.orbitals_dict
