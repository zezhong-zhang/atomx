"""DBSR_HF Breit source translation and particle-number checks."""

import pytest

from atomx import Atom
from atomx.config.jj_coupling import compute_WW, generate_jj_average
from atomx.corrections.adapters import _dirac_orbital_payload
from atomx.corrections.dbsr_breit import (
    _pair_moments,
    _smu,
    compute_dbsr_breit_energy,
)


@pytest.mark.parametrize(
    ("z", "configuration", "reference"),
    [(1, "1s1", 0.0),
     (2, "1s2", 0.000063777432919),
     (10, "1s2 2s2 2p6", 0.016643630840490),
     (18, "1s2 2s2 2p6 3s2 3p6", 0.132364619357042)],
)
def test_dbsr_hf_breit_closed_shell_reference(z, configuration, reference):
    """Against DBSR_HF ``term=AV mbreit=1`` recorded Breit energies."""
    atom = Atom(z, config=configuration, engine="shooting",
                corrections="none").solve()
    orbitals, workspace = _dirac_orbital_payload(atom.result)
    actual = compute_dbsr_breit_energy(orbitals, workspace)
    assert actual == pytest.approx(reference, rel=5e-4, abs=1e-12)


def test_dbsr_hf_open_shell_uses_generalized_occupation_products():
    shells, configurations = generate_jj_average("2p2")
    orbitals = [(None, None, sh.occupation, sh.kappa, sh.label)
                for sh in shells]
    moments = _pair_moments(orbitals, configurations)
    for i in range(len(shells)):
        for j in range(len(shells)):
            assert moments[i, j] == pytest.approx(
                compute_WW(i, j, configurations), abs=1e-14)
    assert moments[0, 1] != pytest.approx(
        shells[0].occupation * shells[1].occupation)
    with pytest.raises(ValueError, match="explicit configuration ensemble"):
        _pair_moments(orbitals, None)


def test_dbsr_hf_smu_selection_and_gaunt_coefficients():
    # This is the L=v=1 channel in hf_energy_BR.f90::SMU.
    assert not any(_smu(-1, -1, -1, -1, 0, 1))
    assert all(value == -2 for value in _smu(-1, -1, -1, -1, 1, 1))
