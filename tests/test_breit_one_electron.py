"""Particle-number checks for the scalar Breit correction."""

import pytest

from atomx import Atom, Basis
from atomx.corrections.breit import compute_breit_energy
from atomx.methods.dirac_fock import DiracFock


@pytest.mark.parametrize("occupations", [[1.0], [1 / 3, 2 / 3]])
def test_scalar_breit_has_no_one_electron_self_pair(occupations):
    orbitals = [(None, None, occ, -1, f"orbital-{i}")
                for i, occ in enumerate(occupations)]
    assert compute_breit_energy(orbitals, None, 1) == 0.0


@pytest.mark.parametrize("z,configuration", [(1, "1s1"), (2, "1s1")])
def test_dhf_one_electron_breit_component_is_zero(z, configuration):
    result = Atom(Z=z, config=configuration, engine="bspline",
                  basis=Basis(r_max=50, grid_hmax=1)).solve(
                      DiracFock(breit=True, qed=False), max_iter=30)
    assert result.converged
    assert result.energy_components["E_Breit"] == 0.0
