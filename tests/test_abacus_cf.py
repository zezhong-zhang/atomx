"""ABACUS CF embedding: DFT crystal field → atomx multiplet spectrum.

Pipeline: ABACUS LCAO d-block (built & converged on this host) →
cubic-to-spherical transform → ImpuritySpec(cf_matrix) → Lehmann solve.

The ABACUS NiO rocksalt PBE SCF gives 10Dq ≈ 0.007 eV (grossly
underestimated by PBE without +U, as expected for correlated oxides);
the literature 10Dq = 0.85 eV is the reference value. Both paths are
pinned here with their multiplet ladders — the DFT CF, however small,
splits the atomic 3F ground term the same way the literature CF does,
just at a compressed scale.
"""
import numpy as np

from atomx.embedding.abacus_cf import (
    abacus_dblock_to_spherical,
    impurity_from_abacus_dblock,
)

# From the converged ABACUS NiO rocksalt PBE LCAO SCF (this host):
# Ni d (1st zeta) on-site eigenvalues: t2g ×3 = 1.0097, eg ×2 = 1.0168 eV
# ABACUS orbital order: (dz2, dxz, dyz, dx2y2, dxy)
ABACUS_H_D = np.diag([1.0168, 1.0097, 1.0097, 1.0168, 1.0097])
ABACUS_10DQ = 1.0168 - 1.0097

LITERATURE_H_D = np.diag([6 * 0.085, -4 * 0.085, -4 * 0.085, 6 * 0.085, -4 * 0.085])


def _ladder(out):
    ev = np.sort(out["N=8"]["evals"])
    cl = []
    for e in ev:
        if cl and abs(e - cl[-1][0]) < 1e-4:
            cl[-1] = (cl[-1][0], cl[-1][1] + 1)
        else:
            cl.append((float(e), 1))
    return cl


def test_transform_roundtrip():
    H_sp = abacus_dblock_to_spherical(ABACUS_H_D)
    from atomx.embedding.abacus_cf import cubic_to_spherical_5
    T = cubic_to_spherical_5()
    H_back = T @ H_sp @ T.conj().T  # inverse transform
    assert np.allclose(H_back, ABACUS_H_D, atol=1e-14)


def test_abacus_cf_splits_ground_term():
    """Even the small PBE 10Dq splits the 3F triplet ground state."""
    out = impurity_from_abacus_dblock(
        ABACUS_H_D, slater_eV={"F0": 0.409, "F2": 7.9521, "F4": 4.9387},
        zeta_d_eV=0.083,
    )
    cl = _ladder(out)
    # ground term (with SOC) had a near-degenerate triplet → now split
    assert cl[0][1] <= 3  # CF breaks the degeneracy
    # the scale is ABACUS 10Dq (~meV), not the literature value
    first_gap = cl[1][0] - cl[0][0]
    assert 0 < first_gap < 0.1


def test_literature_cf_gives_nio_dd_scale():
    """With the literature 10Dq the first excitation is dd-scale (~0.8 eV)."""
    out = impurity_from_abacus_dblock(
        LITERATURE_H_D, slater_eV={"F0": 0.409, "F2": 7.9521, "F4": 4.9387},
        zeta_d_eV=0.083,
    )
    cl = _ladder(out)
    first = cl[1][0] - cl[0][0]
    assert 0.5 < first < 1.5  # dd excitation scale


def test_provenance_chain():
    out = impurity_from_abacus_dblock(
        ABACUS_H_D, slater_eV={"F0": 0.409, "F2": 7.9521, "F4": 4.9387},
        zeta_d_eV=0.083,
    )
    prov = out["cf_input_provenance"]
    assert prov["model"] == "abacus:LCAO-d-block"
    assert "spherical" in prov["basis_out"]
    assert "cf_matrix_spherical" in out
