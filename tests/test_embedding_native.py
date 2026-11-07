"""E1.5 acceptance: theory-ladder-native parameters feed the E1 solver.

Anchors:
1. DF F2/F4 within 1% of the Cowan free-ion tabulation (independent
   tabulation validates the radial pipeline);
2. internal jj consistency (3d3/2 vs 3d5/2 Slater integrals agree <1%);
3. zeta_3d from the jj splitting in the free-ion range;
4. the E1 solver driven by DF parameters reproduces the exact Racah
   closed forms 1D=5B+C, 1G=12B+2C/9, 3P=15B-5C/3, 1S=22B+3C with
   B, C recomputed from the DF values — the anchors travel with the
   parameters, not with a fixed table.
"""
import numpy as np
import pytest

from atomx.embedding import ImpuritySpec, solve_atomic_limit
from atomx.embedding.native_params import (
    COWAN_NI2_FREE_ION,
    dirac_fock_3d_params,
)


@pytest.fixture(scope="module")
def native():
    return dirac_fock_3d_params()


def test_df_slater_matches_cowan_free_ion(native):
    assert native["F2_eV"] == pytest.approx(COWAN_NI2_FREE_ION["F2"], rel=0.01)
    assert native["F4_eV"] == pytest.approx(COWAN_NI2_FREE_ION["F4"], rel=0.01)


def test_jj_partners_agree(native):
    assert native["F2_jj_spread_eV"] < 0.01 * native["F2_eV"]


def test_zeta_in_free_ion_range(native):
    assert 0.08 < native["zeta_3d_eV"] < 0.30


def test_racah_ladder_with_df_parameters(native):
    spec = ImpuritySpec(
        n_electrons=8,
        slater_eV={"F0": 0.409231746031746,
                   "F2": native["F2_eV"], "F4": native["F4_eV"]},
        zeta_d_eV=0.0,
    )
    sol = solve_atomic_limit(spec)
    ev = np.sort(sol["N=8"]["evals"])
    clusters = []
    for e in ev:
        if clusters and abs(e - clusters[-1][0]) < 1e-7:
            clusters[-1] = (clusters[-1][0], clusters[-1][1] + 1)
        else:
            clusters.append((float(e), 1))
    spacings = np.sort([c[0] - clusters[0][0] for c in clusters[1:]])
    B = native["F2_eV"] / 49.0
    C = 5.0 * native["F4_eV"] / 49.0
    predicted = sorted([5 * B + C, 12 * B + 2 * C / 9, 15 * B - 5 * C / 3,
                        22 * B + 3 * C])
    assert np.allclose(spacings, predicted, atol=1e-8)
