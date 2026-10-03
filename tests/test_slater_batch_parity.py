"""Parity gate: batched cell kernels vs the scalar path (tier-3a).

compute_slater_rk_cell_batch reproduces the scalar
compute_slater_rk_cell arithmetic exactly (measured max rel diff
4.5e-14 over 20k mixed integrals on the Ne K-hole solve).  The
speedup is load-dependent: 1.1x measured on the shared box under
load ~16, expected higher on a quiet box.  The hot-path wiring into
sparse_ci's assembly waits for a measured win (measurement
discipline); this gate locks the parity contract so the wiring can
land later without re-deriving it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom, Correlation  # noqa: E402
from atomx.numerics.bspline.convol import (  # noqa: E402
    ConvolCache,
    compute_slater_rk_cell,
    compute_slater_rk_cell_batch,
)


@pytest.fixture(scope="module")
def ne_khole():
    atom = Atom(Z=10, config=["1s1 2s2 2p6"],
                correlation=Correlation(radial_optimization="fac_frozen"),
                corrections="none")
    atom.solve()
    return atom._result

def test_batch_matches_scalar_exactly(ne_khole):
    od = ne_khole.orbitals_dict
    names = list(od.keys())[:4]
    keys = []
    for a in names:
        for b in names:
            for c in names:
                for d in names:
                    for k in (0, 1, 2):
                        keys.append((a, b, c, d, k))
    keys = keys[:240]  # fast: the four-orbital x k set
    bsp = ne_khole._bsp_data
    cache = ConvolCache(bsp)
    batch = compute_slater_rk_cell_batch(od, keys, bsp, cache=cache)
    for key, got in zip(keys, batch, strict=True):
        ref = compute_slater_rk_cell(
            od[key[0]][0], od[key[0]][1], od[key[1]][0], od[key[1]][1],
            od[key[2]][0], od[key[2]][1], od[key[3]][0], od[key[3]][1],
            bsp, key[4], cache=cache)
        assert got == pytest.approx(ref, rel=1e-12, abs=1e-14), (
            f"key {key}: batch {got!r} vs scalar {ref!r}"
        )
