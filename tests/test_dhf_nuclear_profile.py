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


@pytest.mark.parametrize('nuclear',['point','fermi'])
def test_neon_energy_and_recoil_converge_between_radial_engines(nuclear):
    """Resolve the Fermi surface and converge orbitals for first-order properties."""
    from atomx.methods.dirac_fock import DFSolver, neon_subshells
    from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
    from atomx.properties.isotope_shift import compute_sms_abinitio
    from atomx.radial_state import native_radial_state
    bs = DFSolver(10,neon_subshells(),ns=90,r_max=75.,nuclear=nuclear).solve(
        tol=1e-9,max_iter=100)
    grid = ShootingHFSolver(10,neon_subshells(),n_grid=2000,r_max=75.,nuclear=nuclear).solve(
        tol=1e-10,orbital_tolerance=1e-9,max_iter=100)
    assert bs.converged and grid.converged
    assert bs.total_energy == pytest.approx(grid.total_energy,abs=1e-9)
    coefficients=[]
    for result in (bs,grid):
        state=native_radial_state(result)
        coefficients.append(compute_sms_abinitio(list(state.occupied),state.correction_workspace(),Z=10).K_SMS_au)
    assert coefficients[0] == pytest.approx(coefficients[1],rel=1e-9)
    assert grid.solver_diagnostics['undamped_orbital_update_norm'] <= 1e-9
