"""Frozen-orbital solver in the converged exchange='gu' DFS potential.

Validates that atomx's local-DFS machinery reproduces FAC's optimized
central potential *and* its excited-orbital eigenvalues: FAC solves every
orbital (occupied or not) in the single potential from OptimizeRadial;
``dfs_frozen_orbital`` does the same against the converged ``_rv`` that
the shooting solver now exports.

Reference values from FAC 1.1.5 ``WaveFuncTable`` after
``OptimizeRadial(['rec'])`` for Fe XVII (rec = 1s2 2s2 2p6):

    orbital   FAC energy (eV)     (Ha)
    3s1/2     -5.29415263e2      -19.458
    3p1/2     -4.96964752e2      -18.265
    3d5/2     -4.47317880e2      -16.440

and the occupied set from ``GetPotential``:
    1s -278.49, 2s -49.72, 2p1/2 -45.83, 2p3/2 -45.35 (Ha).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.numerics.shooting.solver import dfs_frozen_orbital  # noqa: E402


@pytest.fixture(scope="module")
def fe17_dfs():
    a = Atom(Z=26, charge=16, config="1s2 2s2 2p6", engine="shooting", exchange="gu")
    a.solve()
    return a._result


def test_occupied_energies_match_fac(fe17_dfs):
    ref = {"1s1/2": -278.49, "2s1/2": -49.72, "2p1/2": -45.83,
           "2p3/2": -45.35}
    for lab, e_fac in ref.items():
        assert fe17_dfs.orbital_energies[lab] == pytest.approx(e_fac,
                                                               rel=5e-3)


def test_result_exports_potential_and_waves(fe17_dfs):
    assert fe17_dfs._rv is not None
    assert fe17_dfs._pa is not None and fe17_dfs._pa.shape[0] == 4
    # converged potential asymptote -> -(Z - N + 1)/r = -17/r
    r = fe17_dfs._grid
    z_asym = -fe17_dfs._rv[-1]
    assert z_asym == pytest.approx(17.0, abs=0.2)
    assert r[-1] > 20.0


@pytest.mark.parametrize("n,kappa,guess,e_fac_ha", [
    (3, -1, -19.4, -529.415263 / 27.211386),
    (3, 1, -18.2, -496.964752 / 27.211386),
    (3, -3, -16.4, -447.317880 / 27.211386),
])
def test_frozen_virtuals_match_fac(fe17_dfs, n, kappa, guess, e_fac_ha):
    P, Q, e = dfs_frozen_orbital(fe17_dfs, n, kappa, guess)
    # 0.5%: atomx's Gu-alpha_xc DFS vs FAC's — same model, independent codes.
    assert e == pytest.approx(e_fac_ha, rel=5e-3)
    # slag6-normalized
    import numpy as np
    difr = fe17_dfs._difr
    norm = float(np.sum((P ** 2 + Q ** 2) * difr))
    assert norm == pytest.approx(1.0, rel=1e-6)
