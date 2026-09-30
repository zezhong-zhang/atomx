"""Analytic invariants of the CI+MBPT Σ⁽¹⁾ injection.

Matrix-level facts on a small, fast Be CI space (["1s2 2s2", "1s2 2p2"]):

1. Common-mode: every CSF occupies 1s exactly twice, so a Σ⁽¹⁾ shift of
   the 1s orbital must shift every diagonal H[r,r] by exactly 2δ, leave
   off-diagonal elements untouched, and leave all excitation energies
   invariant.
2. Occupation contrast: a 2s shift moves the 2s-occupied states only
   (diagonal shift 2δ for "1s2 2s2", 0 for "1s2 2p2"), so the
   ground→excited gap moves by exactly -2δ.
3. ci_mbpt_solve provenance: Σ⁽¹⁾-only runs carry mbpt_provenance
   recording sigma2 as not implemented, and asking for sigma2 raises.
"""
from __future__ import annotations

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs_many, get_shared_orbital_basis
from atomx.methods.ci_mbpt import ci_mbpt_solve
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.physics.csf.matrix import build_ci_matrix

CONFIGS = ["1s2 2s2", "1s2 2p2"]


def _be_workspace():
    """Solve Be with a tiny-occupation 2p virtual so all three orbitals exist."""
    dhf = DFSolver(
        Z=4,
        subshells=[Subshell(n=1, kappa=-1, occupation=2.0),
                   Subshell(n=2, kappa=-1, occupation=2.0),
                   Subshell(n=2, kappa=1, occupation=1e-6),
                   Subshell(n=2, kappa=-2, occupation=1e-6)],
        ns=40, ks=7,
    )
    res = dhf.solve()
    od = {}
    for (n, kappa), (P, Q, _occ) in res.orbitals_by_subshell().items():
        od[(n, kappa)] = (P, Q)
    return dhf, od


@pytest.fixture(scope="module")
def be_space():
    dhf, od = _be_workspace()
    csfs = enumerate_jj_csfs_many(CONFIGS)
    return dhf, od, csfs


def _basis(csfs):
    return get_shared_orbital_basis(csfs)


def _matrix(dhf, od, csfs, shifts):
    return build_ci_matrix(csfs, od, _basis(csfs),
                           dhf.bsp_data, 4, dhf.ns,
                           one_body_shifts=shifts)


def test_sigma1_common_mode_is_exact_2delta(be_space):
    dhf, od, csfs = be_space
    delta = 1e-3
    H0 = _matrix(dhf, od, csfs, None)
    H1 = _matrix(dhf, od, csfs, {(1, -1): delta})
    shift = np.diag(H1) - np.diag(H0)
    assert np.allclose(shift, 2.0 * delta, rtol=0, atol=1e-12)
    off = H1 - H0
    np.fill_diagonal(off, 0.0)
    assert np.max(np.abs(off)) < 1e-14
    # Excitation spectrum invariant under a common-mode shift.
    w0 = np.sort(np.linalg.eigvalsh(H0))
    w1 = np.sort(np.linalg.eigvalsh(H1))
    assert np.allclose(w1 - w0, 2.0 * delta, atol=1e-12)


def test_sigma1_occupation_contrast_moves_gap(be_space):
    dhf, od, csfs = be_space
    delta = 5e-4
    H0 = _matrix(dhf, od, csfs, None)
    H1 = _matrix(dhf, od, csfs, {(2, -1): delta})
    d = np.diag(H1) - np.diag(H0)
    # The "1s2 2s2" CSF carries two 2s electrons; "1s2 2p2" carries none.
    assert np.isclose(np.max(d), 2.0 * delta, atol=1e-12)
    assert np.isclose(np.min(d), 0.0, atol=1e-14)


def test_ci_mbpt_provenance_and_sigma2_guard(be_space):
    dhf, od, csfs = be_space
    res = ci_mbpt_solve(csfs, od, _basis(csfs),
                        dhf.bsp_data, 4, dhf.ns, sigma1={(2, -1): 1e-4})
    assert res.mbpt_provenance["sigma2"] == "not_implemented"
    assert res.mbpt_provenance["sigma1"] == {(2, -1): 1e-4}
    with pytest.raises(NotImplementedError):
        ci_mbpt_solve(csfs, od, _basis(csfs),
                      dhf.bsp_data, 4, dhf.ns, sigma2={"anything": 1})
