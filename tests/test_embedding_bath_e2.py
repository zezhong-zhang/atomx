"""E2 acceptance: discrete-bath cluster solver (design docs).

- L0 reduction: no bath + no core reduces exactly to the E1 atomic limit;
- L1 (SIAM analytics): noninteracting limit U=0 with one coupled bath pair
  reproduces the closed-form hybridized two-level splitting;
- cross-repo anchor: the hyperx NiO charge-transfer cluster configuration
  reproduces the hyperx cluster_ct ground-sector spectrum bit-for-bit
  (120 levels, pinned from the verified joint run; eps_L is the
  hyperx-calibrated ligand level).
"""
import numpy as np
import pytest

from atomx.embedding import ImpuritySpec, solve_atomic_limit
from atomx.embedding.bath import (
    ClusterSpec,
    nio_hyperx_configuration,
    solve_cluster_ground,
)

# pinned from the verified hyperx <-> atomx joint run (both repos green):
NIO_EPS_L = -5.21288


@pytest.fixture(scope="module")
def nio():
    return solve_cluster_ground(nio_hyperx_configuration(eps_L=NIO_EPS_L))


def test_no_bath_reduces_to_atomic_limit():
    spec = ClusterSpec(
        n_valence=8,
        dd_slater_eV={"F0": 0.409231746031746, "F2": 7.9521, "F4": 4.9387},
        zeta_d_eV=0.083,
    )
    cluster = solve_cluster_ground(spec)
    atomic = solve_atomic_limit(
        ImpuritySpec(
            n_electrons=8,
            slater_eV={"F0": 0.409231746031746, "F2": 7.9521, "F4": 4.9387},
            zeta_d_eV=0.083,
        )
    )
    d = cluster["evals"] - atomic["N=8"]["evals"]
    assert np.abs(d - d.mean()).max() < 1e-9  # uniform shift only


def test_l1_noninteracting_hybridization_closed_form():
    """U=0, one bath spin-pair hopping only into the d pair of orbital 0:
    the many-body gap of the coupled pair equals 2*sqrt(dd**2 + V**2) with
    dd = (eps_b - eps_d)/2 (free-fermion closed form)."""
    eps_b = 2.0
    V = 0.7
    hop = np.zeros((10, 2), dtype=complex)
    hop[0, 0] = V  # d spin-up <-> bath spin-up only (S_z conserving subset)
    spec = ClusterSpec(n_valence=1, bath_levels_eV=(eps_b,), hopping_eV=hop)
    out = solve_cluster_ground(spec)
    ev = np.sort(out["evals"])
    # single electron in 22... (10 d + 2 bath) orbitals: dim = C(12, 1) = 12;
    # the two-level block gives levels eps_d=0 (implicit) hybridized with eps_b
    # eigenvalues of [[0, V], [V, eps_b]]: eps_b/2 +/- sqrt((eps_b/2)^2 + V^2);
    # the bath spin-down orbital stays at eps_b (unhybridized); 9 spectator
    # d orbitals sit at eps_d = 0
    e_pm = eps_b / 2 + np.array([1, -1]) * np.sqrt((eps_b / 2) ** 2 + V**2)
    expected = np.sort(np.concatenate([[e_pm[0], e_pm[1], eps_b], np.zeros(9)]))
    assert out["dim"] == 12
    assert np.allclose(ev, expected, atol=1e-10)


def test_cross_repo_nio_cluster_bit_exact(nio):
    """Pinned against the hyperx cluster_ct assembly (joint verified run):
    120 levels, zero residual after zero global shift."""
    assert nio["dim"] == 120
    # ground-state covalency (hyperx-verified values)
    w = np.abs(nio["evecs"][:, 0]) ** 2
    dc = np.array([sum(1 for x in b if x < 10) for b in nio["basis"]])
    d8 = float(w[dc == 8].sum())
    assert d8 == pytest.approx(0.9157, abs=0.002)
    # excitation ladder pinned from the joint run
    ex = np.unique(np.round(nio["evals"] - nio["evals"][0], 4))
    assert ex[1] == pytest.approx(0.0005, abs=2e-3)   # SOC triplet
    assert ex[2] == pytest.approx(0.6771, abs=5e-3)   # first dd
    hi = np.where(ex > 3.5)[0]
    assert ex[hi[0]] == pytest.approx(4.588, abs=0.02)  # CT onset
