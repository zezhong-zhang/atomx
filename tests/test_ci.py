"""Pure-CI diagonalisation on frozen DHF orbitals.

This exercises :func:`atomx.methods.ci.ci_solve` directly (no MCDHF
orbital-opt loop).  It is the canonical regression for the Phase-6
split: the first MCDHF macro-iteration builds CI on the *starting*
DHF orbitals and diagonalises, so a fresh ``ci_solve`` on those same
orbitals must produce the same state energies bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx.methods.ci import CIResult, ci_solve
from atomx.methods.dirac_fock import DFSolver
from atomx.methods.mcdhf import MCDHFSolver, make_csf_list
from atomx.methods.mcdhf.config import get_shared_orbital_basis


def _he_csfs():
    return make_csf_list(Z=2, config_strings=["1s2"])


def test_ci_solve_returns_ciresult_and_ground_energy_is_minimum():
    csfs = _he_csfs()
    solver = MCDHFSolver(Z=2, csfs=csfs, ns=50, ks=7)

    # Run only the first macro iteration manually so we have frozen DHF
    # orbitals + matching bsp_data.
    avg = solver._averaged_subshells()
    dhf = DFSolver(Z=2, subshells=avg, ns=solver.ns, ks=solver.ks, exchange="hf")
    dhf_result = dhf.solve(max_iter=40, tol=1e-5)

    orbitals_dict = {}
    kappa_counter = {}
    for sh in avg:
        idx = kappa_counter.get(sh.kappa, 0)
        kappa_counter[sh.kappa] = idx + 1
        orb_list = dhf_result.orbitals_raw.get(sh.kappa, [])
        if idx < len(orb_list):
            P, Q, _ = orb_list[idx]
            orbitals_dict[(sh.n, sh.kappa)] = (P, Q)

    basis = get_shared_orbital_basis(csfs)
    ci = ci_solve(csfs, orbitals_dict, basis, dhf.bsp_data, Z=2, ns=dhf.ns)

    assert isinstance(ci, CIResult)
    assert ci.state_energies.shape == (len(csfs),)
    assert ci.mixing_coefficients.shape == (len(csfs), len(csfs))
    # Ground state is the minimum eigenvalue.
    assert ci.ground_energy == pytest.approx(float(ci.state_energies.min()), rel=1e-12)
    # Provenance records the CSF count.
    assert ci.provenance["n_csf"] == len(csfs)


def test_ci_solve_matches_first_mcdhf_macroiteration():
    """ci_solve on frozen DHF orbitals reproduces the state energies that
    MCDHFSolver computes inside its first macro-iteration."""
    csfs = _he_csfs()
    # Disable the DBSR-HF physics knots grid that MCDHFSolver installs by
    # default — the standalone DFSolver below uses the default knot grid,
    # so both code paths must share the same basis for the equivalence
    # check to be meaningful (otherwise the DHF energies — and hence the
    # 1×1 CI energies for a single CSF — differ by ~5 mHa).
    solver = MCDHFSolver(Z=2, csfs=csfs, ns=50, ks=7, use_physics_grid=False)

    # Run the full MCDHF (single CSF, so zero correlation — converges
    # in one iteration but still produces a well-defined CI result).
    mcdhf_result = solver.solve(max_iter=3, scf_tol=1e-5, ci_tol=1e-8)

    # Now reproduce the CI part standalone.
    avg = solver._averaged_subshells(weights=np.ones(len(csfs)) / len(csfs))
    dhf = DFSolver(Z=2, subshells=avg, ns=solver.ns, ks=solver.ks, exchange="hf")
    dhf_result = dhf.solve(max_iter=40, tol=1e-5)
    orbitals_dict = {}
    kappa_counter = {}
    for sh in avg:
        idx = kappa_counter.get(sh.kappa, 0)
        kappa_counter[sh.kappa] = idx + 1
        orb_list = dhf_result.orbitals_raw.get(sh.kappa, [])
        if idx < len(orb_list):
            P, Q, _ = orb_list[idx]
            orbitals_dict[(sh.n, sh.kappa)] = (P, Q)

    basis = get_shared_orbital_basis(csfs)
    ci = ci_solve(csfs, orbitals_dict, basis, dhf.bsp_data, Z=2, ns=dhf.ns)

    # Ground energies from ci_solve and MCDHF must agree to full tolerance.
    assert ci.ground_energy == pytest.approx(mcdhf_result.ground_energy, rel=1e-10, abs=1e-12)
