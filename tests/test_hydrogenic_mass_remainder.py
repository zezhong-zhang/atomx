"""Finite-mass energy identity and correction-stack integration."""
from types import SimpleNamespace

import pytest

from atomx import Atom
from atomx.corrections.adapters import HydrogenicMassRemainderCorrection
from atomx.corrections.recoil import hydrogenic_mass_remainder
from atomx.corrections.stack import CorrectionStack
from atomx.methods.dirac_fock import DiracFock


@pytest.mark.parametrize('z,n,mass', [(1, 1, 1836.152673426), (1, 2, 1836.152673426),
                                      (20, 3, 72000.), (2, 1, 1.)])
def test_exact_two_body_schrodinger_mass_scaling(z, n, mass):
    infinite = -.5*(z/n)**2
    reduced_mass = mass/(mass+1.)
    # This is the exact two-body Coulomb spectrum in center-of-mass coordinates.
    exact = -.5*reduced_mass*(z/n)**2
    first_order = -infinite/mass
    assert infinite+first_order+hydrogenic_mass_remainder(z, n, mass) == pytest.approx(
        exact, abs=1e-14)


@pytest.mark.parametrize('mass', [0., -1., float('nan'), float('inf')])
def test_invalid_mass_rejected(mass):
    with pytest.raises(ValueError):
        hydrogenic_mass_remainder(1, 1, mass)


def test_stack_requires_first_order_operator_and_mass():
    with pytest.raises(ValueError):
        CorrectionStack(include_hydrogenic_mass_remainder=True)
    with pytest.raises(ValueError):
        CorrectionStack(nuclear_mass_me=2000., nms_mode='virial',
                        include_hydrogenic_mass_remainder=True)


def test_multielectron_and_correlation_labels_are_not_treated_as_hydrogenic():
    adapter = HydrogenicMassRemainderCorrection(2000.)
    helium = SimpleNamespace(Z=2, subshells=[SimpleNamespace(n=1, occupation=2.)])
    result = adapter.evaluate(helium)
    assert result.energy_shift == 0.
    assert not result.provenance['applied']
    correlated = SimpleNamespace(subshells=helium.subshells, one_body_density_matrix=lambda: None)
    with pytest.raises(ValueError):
        adapter.evaluate(correlated)
    mixed = SimpleNamespace(Z=1, subshells=[SimpleNamespace(n=1, occupation=.5),
                                          SimpleNamespace(n=2, occupation=.5)])
    with pytest.raises(ValueError):
        adapter.evaluate(mixed)


def test_mcdhf_rejects_requested_remainder_instead_of_ignoring_it():
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.mcdhf.solver import MCDHFSolver
    with pytest.raises(NotImplementedError, match='principal-shell'):
        MCDHFSolver(Z=1, csfs=enumerate_jj_csfs('1s1'), correction_stack=CorrectionStack(
            nuclear_mass_me=2000., include_hydrogenic_mass_remainder=True))


def test_dhf_transition_receives_remainder_once_and_preserves_nms():
    mass = 1836.152673426
    gaps = []
    for enabled in (False, True):
        energies = []
        for n in (1, 2):
            result = Atom(Z=1, config=f'{n}s1', engine='bspline').solve(DiracFock(
                nuclear_mass_me=mass, include_hydrogenic_mass_remainder=enabled)).result
            assert result.converged
            assert result.energy_components['nms'] > 0
            if enabled:
                assert result.energy_components['hydrogenic_mass_remainder'] == pytest.approx(
                    hydrogenic_mass_remainder(1, n, mass), rel=1e-12)
                assert result.solver_diagnostics['_hydrogenic_mass_remainder_provenance']['first_order_subtracted']
            energies.append(result.total_energy)
        gaps.append(energies[1]-energies[0])
    assert gaps[1]-gaps[0] == pytest.approx(.375/(mass*(mass+1)), rel=1e-8)
