import pytest

from atomx import Atom
from atomx.methods.dirac_fock import DiracFock
from atomx.potentials.nuclear import nuclear_profile
from atomx.types import NuclearModel


def test_profile_preserves_and_identifies_radius_approximations():
    legacy = nuclear_profile(18)
    selected = nuclear_profile(18, A=40)
    assert legacy['radius_mass_number'] == 45
    assert selected['radius_mass_number'] == 40
    assert selected['rms_radius_fm'] < legacy['rms_radius_fm']
    assert selected['radius_source'].startswith('empirical')
    assert nuclear_profile(1)['radius_clamped']
    assert nuclear_profile(1)['rms_radius_fm'] == 2.0
    with pytest.raises(ValueError, match='A >= Z'):
        nuclear_profile(18, A=12)


def test_selected_mass_number_reaches_finite_nuclear_hamiltonian():
    # A larger nucleus weakens the attraction of a penetrating 1s electron.
    energies = []
    for a in (40, 45):
        result = Atom(Z=18, config='1s1', engine='bspline').solve(
            DiracFock(nuclear_mass_number=a), tol=1e-9)
        assert result.converged
        assert result.energy_components['_nuclear_profile']['radius_mass_number'] == a
        energies.append(result.total_energy)
    assert energies[1] > energies[0]
    point = []
    for a in (40, 45):
        result = Atom(Z=18, config='1s1', engine='bspline').solve(
            DiracFock(nuclear=NuclearModel.POINT, nuclear_mass_number=a))
        point.append(result.total_energy)
    assert point[0] == pytest.approx(point[1], abs=1e-12)
