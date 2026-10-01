"""Nonzero self-energy gates against explicit spin-orbital sums."""

from itertools import product
from unittest.mock import patch

import pytest
from test_review_regressions import _coulomb

from atomx.constants import HARTREE_EV
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt_transitions import (
    mbpt2_orbital_energy_shift,
    mbpt2_transition_energy,
)


@pytest.fixture(scope="module")
def helium():
    return DFSolver(
        Z=4, subshells=[Subshell(n=1, kappa=-1, occupation=2.0)], ns=30, ks=7
    ).solve()


def test_same_subshell_self_energy_matches_spin_orbital_sum(helium):
    from atomx.angular.csf_hamiltonian import _m2_values, twoj
    from atomx.methods.mbpt import build_virtual_basis

    occupied = [(1, -1)]
    od = {key: (P, Q) for key, (P, Q, q) in helium.orbitals_by_subshell().items()}
    eps = {(1, -1): helium.orbital_energies["1s1/2"]}
    blocks = {}
    for k in (-1, 1):
        _, states = build_virtual_basis(
            helium, k, e_max=50.0, orthogonal_to_occupied=True
        )
        blocks[k] = states[:2]
        for i, (P, Q, e) in enumerate(blocks[k]):
            od[(100 + i, k)] = (P, Q)
            eps[(100 + i, k)] = e
    virt = [key for key in od if key not in occupied]

    def spin(keys):
        return [(*key, m) for key in keys for m in _m2_values(twoj(key[1]))]

    core, vspin = spin(occupied), spin(virt)
    g = _coulomb(od, helium._bsp_data)
    reference = 0.0
    for a in core:
        for b, r, s in product(core, vspin, vspin):
            anti = g(a, b, r, s) - g(a, b, s, r)
            reference += (
                0.5
                * anti**2
                / (eps[a[:2]] + eps[b[:2]] - eps[r[:2]] - eps[s[:2]])
                / len(core)
            )
        for b, c, r in product(core, core, vspin):
            anti = g(a, r, b, c) - g(a, r, c, b)
            reference += (
                0.5
                * anti**2
                / (eps[a[:2]] + eps[r[:2]] - eps[b[:2]] - eps[c[:2]])
                / len(core)
            )

    def basis(scf, kappa, **kwargs):
        assert kwargs["potential_mode"] == "VN"
        assert kwargs["orthogonal_to_occupied"]
        return [], blocks[kappa]

    with patch("atomx.methods.mbpt.build_virtual_basis", side_effect=basis) as build:
        actual = mbpt2_orbital_energy_shift(helium, "1s1/2", kappa_max=1, n_virt_max=2)
    assert abs(reference) > 1e-6
    assert actual == pytest.approx(reference, rel=1e-11, abs=1e-13)
    assert build.call_count == 2  # One eigensystem per kappa, independent of r/s loops.
    assert any(e < 0 for _, _, e in blocks[-1])  # Bound virtuals contribute.


def test_invalid_orbital_label(helium):
    with pytest.raises(ValueError, match="not found"):
        mbpt2_orbital_energy_shift(helium, "9z9/2")


def test_virtual_basis_errors_propagate(helium):
    with (
        patch(
            "atomx.methods.mbpt.build_virtual_basis",
            side_effect=RuntimeError("eigensolver failed"),
        ),
        pytest.raises(RuntimeError, match="eigensolver failed"),
    ):
        mbpt2_orbital_energy_shift(helium, "1s1/2")


def test_self_transition_zero_with_nonzero_equal_shifts(helium):
    result = mbpt2_transition_energy(
        helium, "1s1/2", "1s1/2", kappa_max=1, n_virt_max=2
    )
    assert abs(result["delta_E_alpha_au"]) > 1e-6
    assert result["delta_E_alpha_au"] == result["delta_E_beta_au"]
    assert result["omega_DHF_au"] == result["omega_corrected_au"] == 0.0


def test_nonzero_transition_difference_and_units(helium):
    from types import SimpleNamespace

    fake = SimpleNamespace(orbital_energies={"a": -1.0, "b": -0.2})
    with patch(
        "atomx.methods.mbpt_transitions.mbpt2_orbital_energy_shift",
        side_effect=[0.03, -0.01],
    ):
        result = mbpt2_transition_energy(fake, "a", "b")
    assert result["omega_corrected_au"] == pytest.approx(0.76)
    assert result["omega_corrected_eV"] == pytest.approx(0.76 * HARTREE_EV)
