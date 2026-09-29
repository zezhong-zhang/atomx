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
from atomx.numerics.shooting.local import dfs_frozen_orbital  # noqa: E402


@pytest.fixture(scope="module")
def fe17_dfs():
    a = Atom(Z=26, charge=16, config="1s2 2s2 2p6", engine="shooting", exchange="gu")
    a.solve()
    return a._result


def test_occupied_energies_match_fac(fe17_dfs):
    """FAC 1.1.5 WaveFuncTable values, gated per orbital by the measurement.

    Measured relative differences (2026-09-22): 1s 1.310e-4, 2s 1.758e-3,
    2p1/2 1.521e-3, 2p3/2 1.572e-3.  The single rel=5e-3 gate this replaces
    had a 38x margin on the 1s (which is the orbital the model pins best);
    each gate below is ~3x its own measurement.
    """
    ref = {"1s1/2": (-278.49, 5.0e-4), "2s1/2": (-49.72, 5.0e-3),
           "2p1/2": (-45.83, 5.0e-3), "2p3/2": (-45.35, 5.0e-3)}
    for lab, (e_fac, rel_gate) in ref.items():
        got = fe17_dfs.orbital_energies[lab]
        assert got == pytest.approx(e_fac, rel=rel_gate), (
            f"{lab}: atomx={got!r} fac={e_fac!r} "
            f"rel={abs(got-e_fac)/abs(e_fac):.3e} gate={rel_gate:.0e}"
        )


def test_result_exports_potential_and_waves(fe17_dfs):
    assert fe17_dfs._rv is not None
    assert fe17_dfs._pa is not None and fe17_dfs._pa.shape[0] == 4
    # converged potential asymptote -> -(Z - N + 1)/r = -17/r
    r = fe17_dfs._grid
    z_asym = -fe17_dfs._rv[-1]
    # The Latter cap makes this exact, not approximate: measured
    # z_asym = 17.000000000000 (the old abs=0.2 band admitted 16.8..17.2).
    assert z_asym == pytest.approx(17.0, abs=1.0e-9)
    assert r[-1] > 20.0


@pytest.mark.parametrize("n,kappa,guess,e_fac_ha,rel_gate", [
    # Gates tiered by the measured atomx-vs-FAC difference: 9.10e-4, 1.389e-3,
    # 3.576e-3 relative.  The 3d5/2 gap is the *outermost* virtual, i.e. the
    # one most sensitive to the model difference (FAC's OptimizeRadial
    # relaxes the potential for the 'rec' configuration; atomx solves the
    # virtual in the frozen converged _rv), so it keeps a 1.4x margin while
    # 3s/3p are gated at ~2x their own measurement.  This is a model
    # difference, not numerical noise.
    (3, -1, -19.4, -529.415263 / 27.211386, 2.0e-3),
    (3, 1, -18.2, -496.964752 / 27.211386, 3.0e-3),
    (3, -3, -16.4, -447.317880 / 27.211386, 5.0e-3),
])
def test_frozen_virtuals_match_fac(fe17_dfs, n, kappa, guess, e_fac_ha,
                                   rel_gate):
    P, Q, e = dfs_frozen_orbital(fe17_dfs, n, kappa, guess)
    assert e == pytest.approx(e_fac_ha, rel=rel_gate), (
        f"n={n} kappa={kappa}: atomx={e!r} fac={e_fac_ha!r} "
        f"rel={abs(e-e_fac_ha)/abs(e_fac_ha):.3e} gate={rel_gate:.0e}"
    )
    # slag6-normalized
    import numpy as np
    difr = fe17_dfs._difr
    norm = float(np.sum((P ** 2 + Q ** 2) * difr))
    assert norm == pytest.approx(1.0, rel=1e-6)
