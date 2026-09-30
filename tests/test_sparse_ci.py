"""Sparse M-scheme CI vs the exact engine and the m-scheme oracle.

Three independent assemblies of the same Dirac-Coulomb Hamiltonian on the
same orbital/radial kernels must give the same spectrum:

1. ``build_ci_matrix`` — jj-CSF blocks + determinant projection (exact);
2. the m-scheme oracle (``benchmarks/mscheme_ci_reference``);
3. ``sparse_ci.sparse_ci_spectrum`` — sparse determinant CSR + eigsh.

Be ["1s2 2s2", "1s2 2p2"] (even parity): the 2M=0 determinant sector spans
exactly the union of the J = 0, 2, 4 CSF blocks, so the three sorted
spectra must coincide to tight tolerance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent
_BENCH = _ROOT.parent / "benchmarks"
for _p in (str(_ROOT), str(_BENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atomx.config.csf import enumerate_jj_csfs_many, get_shared_orbital_basis
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.physics.csf.matrix import build_ci_matrix
from atomx.physics.csf.sparse_ci import (
    enumerate_determinants,
    patterns_from_csfs,
    sparse_ci_spectrum,
)

CONFIGS = ["1s2 2s2", "1s2 2p2"]


def _be_space():
    dhf = DFSolver(
        Z=4,
        subshells=[Subshell(n=1, kappa=-1, occupation=2.0),
                   Subshell(n=2, kappa=-1, occupation=2.0),
                   Subshell(n=2, kappa=1, occupation=1e-6),
                   Subshell(n=2, kappa=-2, occupation=1e-6)],
        ns=40, ks=7,
    )
    res = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o) in
          res.orbitals_by_subshell().items()}
    return dhf, od


@pytest.fixture(scope="module")
def be():
    dhf, od = _be_space()
    csfs = enumerate_jj_csfs_many(CONFIGS)
    return dhf, od, csfs


def test_m0_sector_dimension(be):
    """2M=0 dets: 2s2 (1), 2p1/2^2 (1), 2p1/2 2p3/2 (2), 2p3/2^2 (2) = 6,
    matching the J-resolved state count: J=0 x3 + J=1 x1 + J=2 x2 — every
    J manifold contributes exactly one M=0 member."""
    dhf, od, csfs = be
    masks, spinors = enumerate_determinants(patterns_from_csfs(csfs), od, 0)
    assert len(spinors) == 10  # 1s(2) + 2s(2) + 2p1/2(2) + 2p3/2(4)
    assert len(masks) == 6


def test_sparse_spectrum_matches_exact_csf_blocks(be):
    dhf, od, csfs = be
    w_sparse, n_dets = sparse_ci_spectrum(od, dhf.bsp_data, 4, dhf.ns,
                                          patterns_from_csfs(csfs),
                                          m2_total=0, n_states=100)
    # Exact engine: union over every (2J, parity) CSF block.
    from atomx.physics.csf.matrix import _group_csf_indices
    basis = get_shared_orbital_basis(csfs)
    w_exact = []
    for _key, idx in _group_csf_indices(csfs):
        sub = [csfs[i] for i in idx]
        H = build_ci_matrix(sub, od, basis, dhf.bsp_data, 4, dhf.ns)
        w_exact.extend(np.linalg.eigvalsh(np.asarray(H)))
    w_exact = np.sort(w_exact)
    assert len(w_exact) == len(w_sparse), (len(w_exact), len(w_sparse))
    assert np.allclose(w_sparse, w_exact, atol=1e-10)


def test_sparse_matches_mscheme_oracle(be):
    dhf, od, csfs = be
    from mscheme_ci_reference import patterns_from_configs, mscheme_levels
    patterns = patterns_from_configs(CONFIGS)
    w_oracle, n_oracle = mscheme_levels(od, dhf.bsp_data, 4, dhf.ns,
                                        patterns, M2=0)
    w_sparse, n_sparse = sparse_ci_spectrum(od, dhf.bsp_data, 4, dhf.ns,
                                            patterns_from_csfs(csfs),
                                            m2_total=0, n_states=100)
    assert n_sparse == n_oracle
    assert np.allclose(w_sparse, w_oracle, atol=1e-10)
