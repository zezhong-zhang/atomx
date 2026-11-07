"""E3 v1: one-shot first-principles spectral product.

Anchors (each stage already jointly verified; this test pins the CHAIN):
- the screened parameters reproduce the CPC-2019 benchmark values within
  the parameter round-trip tolerance (F2 ~ 7.95, zeta ~ 0.083);
- the payload carries complete provenance (DF free-ion values, screening
  convention, solver level) — nothing borrowed, nothing silent;
- the spectral function obeys the unit-weight sum rule and the N-sector
  ladder matches the benchmark multiplet ladder within the screening
  round-trip tolerance (~1%).
"""
import numpy as np
import pytest

from atomx.embedding.one_shot import one_shot_spectral


@pytest.fixture(scope="module")
def out():
    return one_shot_spectral(omega_eV=np.linspace(-15, 5, 600))


def test_screened_params_reproduce_benchmark(out):
    p = out["params_used"]
    assert p["F2"] == pytest.approx(7.9521, rel=0.01)
    assert p["F4"] == pytest.approx(4.9387, rel=0.01)
    assert p["zeta_3d"] == pytest.approx(0.083, rel=0.02)


def test_payload_provenance_complete(out):
    prov = out["payload"]["provenance"]
    assert prov["model"] == "atomx:one-shot(DF->screen->Lehmann)"
    assert set(prov["df_free_ion_eV"]) == {"F2", "F4", "zeta_3d"}
    assert prov["screening"]["dd"] == pytest.approx(0.6498, abs=1e-4)
    assert "convention" in prov["screening"]


def test_spectral_sum_rule(out):
    A = out["spectral_function"]
    omega = np.asarray(out["payload"]["omega_eV"])
    w = A.sum(axis=1) * (omega[1] - omega[0])
    assert np.allclose(w, 1.0, atol=0.02)


def test_ladder_matches_benchmark_zeta_off():
    """Parameter round-trip: with zeta off, the one-shot dd parameters
    reproduce the no-SOC Racah benchmark ladder (1% parameters -> <1%
    ladder by linearity)."""
    out = one_shot_spectral(
        omega_eV=np.linspace(-15, 5, 10), screening_zeta=0.0
    )
    ev = np.sort(out["evals_N"])
    cl = []
    for e in ev:
        if cl and abs(e - cl[-1][0]) < 1e-7:
            cl[-1] = (cl[-1][0], cl[-1][1] + 1)
        else:
            cl.append((float(e), 1))
    assert sorted(c[1] for c in cl) == sorted([21, 9, 9, 5, 1])  # fingerprint
    sp = np.sort([c[0] - cl[0][0] for c in cl[1:]])
    bench = np.array([1.315388, 1.594401, 2.059442, 5.082178])
    assert np.allclose(sp, bench, rtol=0.01)


def test_fine_structure_present(out):
    """With the screened zeta (0.083 eV) the ground term splits — the
    3F_J fine structure that the diagonal-only-SOC bug once faked."""
    ev = np.sort(out["evals_N"])
    ex = np.unique(np.round(ev - ev[0], 4))
    low = ex[(ex > 0.0) & (ex < 0.4)]
    assert len(low) >= 2  # J-split members below 0.4 eV
