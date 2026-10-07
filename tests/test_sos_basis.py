from __future__ import annotations

import pytest

from atomx.angular.recoupling import l_of_kappa
from atomx.methods.dirac_fock import DiracFock, Exchange, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.numerics.bspline.convol import ConvolCache
from atomx.properties._sos_basis import build_sos_virtual_basis


@pytest.fixture(scope="module")
def lithium_result():
    df = DiracFock(exchange=Exchange.HARTREE_FOCK)
    res = df.solve(
        Z=3,
        subshells=[
            Subshell(n=1, kappa=-1, occupation=2),
            Subshell(n=2, kappa=-1, occupation=1),
        ],
    )
    assert res.converged
    return res


def test_kappa_list_contents(lithium_result):
    cache = ConvolCache(lithium_result._bsp_data)

    kappa_list_1, occ_basis_1, virt_basis_1 = build_sos_virtual_basis(
        lithium_result, cache=cache, kappa_max=1, n_virt_max=1
    )
    assert kappa_list_1 == [-1, +1, -2]
    assert len(kappa_list_1) == len(set(kappa_list_1))
    assert all(l_of_kappa(kappa) <= 1 for kappa in kappa_list_1)
    assert set(occ_basis_1) == set(kappa_list_1)
    assert set(virt_basis_1) == set(kappa_list_1)

    kappa_list_2, _, _ = build_sos_virtual_basis(
        lithium_result, cache=cache, kappa_max=2, n_virt_max=1
    )
    assert kappa_list_2 == [-1, +1, -2, +2, -3]
    assert all(l_of_kappa(kappa) <= 2 for kappa in kappa_list_2)


def test_n_virt_max_truncation(lithium_result):
    cache = ConvolCache(lithium_result._bsp_data)
    kappa_list, _, virt_basis = build_sos_virtual_basis(
        lithium_result, cache=cache, kappa_max=1, n_virt_max=2
    )
    assert all(len(virt_basis[kappa]) <= 2 for kappa in kappa_list)

    unbounded_kappa_list, _, unbounded_virt_basis = build_sos_virtual_basis(
        lithium_result, cache=cache, kappa_max=1, n_virt_max=None
    )
    assert unbounded_kappa_list == kappa_list
    for kappa in unbounded_kappa_list:
        _, raw_virt = build_virtual_basis(lithium_result, kappa, cache=cache)
        assert len(unbounded_virt_basis[kappa]) == len(raw_virt)


def test_occupied_labels_and_scf_energies(lithium_result):
    cache = ConvolCache(lithium_result._bsp_data)
    _, occ_basis, _ = build_sos_virtual_basis(
        lithium_result, cache=cache, kappa_max=1, n_virt_max=1
    )

    s_occ = occ_basis[-1]
    assert [entry[3] for entry in s_occ[:2]] == ["1s1/2", "2s1/2"]
    for entry in s_occ[:2]:
        assert len(entry) == 4
        _P, _Q, eps, label = entry
        assert eps == pytest.approx(lithium_result.orbital_energies[label])


def test_basis_key_invariants(lithium_result):
    cache = ConvolCache(lithium_result._bsp_data)
    kappa_list, occ_basis, virt_basis = build_sos_virtual_basis(
        lithium_result, cache=cache, kappa_max=2, n_virt_max=1
    )

    assert set(occ_basis) == set(kappa_list)
    assert set(virt_basis) == set(kappa_list)
