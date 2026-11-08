"""DFT+U crystal field from ABACUS: NiO AFM rocksalt (U=5.0 eV).

The ABACUS DFT+U SCF (converged on this host, 8 ranks, bandgap 0.16/2.23 eV)
gives Ni(up) d eigenvalues {t2g: 0.6511×3, eg: 0.7152/0.7314×2} —
10Dq ≈ 0.08 eV, a 10x improvement over pure PBE (0.007 eV). Feeding this
into the atomx impurity produces a multiplet ladder whose ground-term
splitting now carries the DFT+U scale.
"""
import numpy as np

from atomx.embedding.abacus_cf import (
    impurity_from_abacus_dblock,
)

# From the converged ABACUS DFT+U NiO AFM SCF (this host):
DFTU_H_D = np.array([
    [0.6513, -0.0015, -0.0015, 0., -0.003],
    [-0.0015, 0.7259, -0.0053, -0.0026, 0.0053],
    [-0.0015, -0.0053, 0.7259, 0.0026, 0.0053],
    [0., -0.0026, 0.0026, 0.6513, 0.],
    [-0.003, 0.0053, 0.0053, 0., 0.7259],
])

DFTU_10DQ = 0.7152 - 0.6511  # ~0.064 eV from eigenvalue spread
PBE_10DQ = 0.0071
LITERATURE_10DQ = 0.85


def _ladder(out):
    ev = np.sort(out["N=8"]["evals"])
    cl = []
    for e in ev:
        if cl and abs(e - cl[-1][0]) < 1e-4:
            cl[-1] = (cl[-1][0], cl[-1][1] + 1)
        else:
            cl.append((float(e), 1))
    return cl


def test_dftu_10dq_improves_over_pbe():
    assert DFTU_10DQ > 5 * PBE_10DQ  # +U opens the CF significantly
    assert DFTU_10DQ < LITERATURE_10DQ  # but still below experimental


def test_dftu_cf_embeds_into_multiplet():
    """The DFT+U CF (0.064 eV scale) splits the atomic ground term."""
    out = impurity_from_abacus_dblock(
        DFTU_H_D, slater_eV={"F0": 0.409, "F2": 7.9521, "F4": 4.9387},
        zeta_d_eV=0.083,
    )
    cl = _ladder(out)
    # Ground term no longer has a high-degeneracy cluster
    assert cl[0][1] <= 3
    # First excitation at the CF+SOC scale (tens of meV)
    first = cl[1][0] - cl[0][0]
    assert 0.005 < first < 0.1
    # dd excitations at ~0.2 eV scale (multiplet, not CF)
    dd = cl[6][0] - cl[0][0]
    assert dd > 0.15


def test_cf_and_multiplet_scales_separate():
    """CF splitting (meV) and multiplet splitting (~eV) are cleanly
    separable in the spectrum — the small CF perturbs, not destroys."""
    out = impurity_from_abacus_dblock(
        DFTU_H_D, slater_eV={"F0": 0.409, "F2": 7.9521, "F4": 4.9387},
        zeta_d_eV=0.083,
    )
    cl = _ladder(out)
    spacings = [c[0] - cl[0][0] for c in cl[1:]]
    # below 0.1 eV: CF+SOC fine structure
    low = [s for s in spacings if s < 0.1]
    # above 0.15 eV: multiplet (dd)
    high = [s for s in spacings if s > 0.15]
    assert len(low) >= 3  # CF splits the ground manifold
    assert len(high) >= 5  # multiplet ladder survives
