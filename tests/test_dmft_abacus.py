"""ABACUS H(R) -> Hubbard-I DMFT: exact anchors + NiO regression.

Anchors (independent of the DMFT code path being right by construction):
  A1 pole expansion: G_at(z) == 1/(z - eps - Sigma(z)) at arbitrary complex z (exact algebra);
  A2 non-interacting limit: U = J = 0 -> Sigma == 0 identically and the DMFT loop returns
     the PBE chemical potential and d occupation (DMFT reduces to DFT);
  A3 FLL double counting closed form;
  A4 NiO paramagnetic regression: PBE is metallic, DFT+DMFT(U=8, J=1) opens a 3.5-3.9 eV
     charge-transfer gap (exp. 3.8-4.3 eV); gap varies < 0.5 eV for U in [6, 9] (CT regime).
"""

import os

import numpy as np
import pytest

from atomx.embedding import dmft

NIO = "/home/zzhang/data/dmft/nio_pm"
ORB = "/home/zzhang/code/abacus-develop/tests/PP_ORB"
have_nio = os.path.exists(
    f"{NIO}/OUT.NiO_PM/data-HR-sparse_SPIN0.csr"
) and os.path.isdir(ORB)


def test_A3_edc_fll_closed_form():
    assert dmft.edc_fll(8.0, 8.0, 1.0) == pytest.approx(8.0 * 7.5 - 1.0 * 3.5)
    s = dmft.slater_from_UJ(8.0, 1.0)
    assert s["F0"] == 8.0 and s["F4"] / s["F2"] == pytest.approx(0.625)
    assert (s["F2"] + s["F4"]) / 14.0 == pytest.approx(1.0)


def test_A1_pole_expansion_exact():
    eps = np.array([1.3, 0.0, 0.0, 1.3, 0.0]) - 50.0
    hub = dmft.solve_hubbard_I(
        eps, dmft.slater_from_UJ(8.0, 1.0), mu=8 * 8.0 - 50.0 - 4.0
    )
    z = np.array([0.37 + 0.05j, -3.1 + 0.2j, 7.3 + 0.1j, 40 + 3j]) - 10.0
    for a in range(5):
        p, r = hub.poles[a]
        assert r.sum() == pytest.approx(
            1.0, abs=1e-8
        )  # poles with weight < 1e-10 are pruned
        G = (r[None, :] / (z[:, None] - p[None, :])).sum(1)
        zt, R = hub.sigma_poles[a]
        assert (R > 0).all()  # causal: positive residues
        Sig = hub.sigma_inf[a] + (R[None, :] / (z[:, None] - zt[None, :])).sum(1)
        assert np.allclose(G, 1.0 / (z - eps[a] - Sig), atol=1e-9)


@pytest.fixture(scope="module")
def nio():
    if not have_nio:
        pytest.skip("ABACUS NiO paramagnetic run not present")
    from atomx.embedding import abacus_hr as ah

    HR = ah.read_csr(f"{NIO}/OUT.NiO_PM/data-HR-sparse_SPIN0.csr")
    SR = ah.read_csr(f"{NIO}/OUT.NiO_PM/data-SR-sparse_SPIN0.csr")
    lay = ah.parse_layout(f"{NIO}/STRU", ORB)
    Hl, _ = ah.lowdin_hk(HR, SR, ah.kmesh(6))
    W, mu, nd, _ = ah.natural_d_projector(Hl, lay, 0, 24.0)
    return Hl, W, mu, nd


def _gap(res):
    E, V, nb = res["E"], res["V"], res["nb"]
    wph = (np.abs(V[:, :nb, :]) ** 2).sum(1)
    e = np.sort(E[wph > 0.02])
    d = np.diff(e)
    mu = res["mu"]
    i = np.argmax(np.where((e[:-1] < mu + 6) & (e[1:] > mu - 6), d, 0))
    return float(e[i + 1] - e[i])


def test_A2_noninteracting_limit_reduces_to_dft(nio):
    Hl, W, mu, nd = nio
    res = dmft.dmft_loop(Hl, W, 24.0, 0.0, 0.0, mu, nd, maxit=5, log=lambda *a: None)
    hub = res["hub"]
    # Sigma(z) == 0: no poles with weight and Sigma_inf == 0
    for a in range(5):
        zt, R = hub.sigma_poles[a]
        assert (len(R) == 0) or np.all(R < 1e-9)
        assert abs(hub.sigma_inf[a]) < 1e-9
    assert res["n_d"] == pytest.approx(nd, abs=2e-3)


def test_A4_nio_charge_transfer_gap(nio):
    Hl, W, mu, nd = nio
    gaps = {}
    for U in (6.0, 8.0, 9.0):
        res = dmft.dmft_loop(Hl, W, 24.0, U, 1.0, mu, nd, maxit=40, log=lambda *a: None)
        assert res["hub"].N0 == 8
        gaps[U] = _gap(res)
    assert 3.5 < gaps[8.0] < 3.9
    assert max(gaps.values()) - min(gaps.values()) < 0.5
