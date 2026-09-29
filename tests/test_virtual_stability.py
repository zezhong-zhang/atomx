"""Finite virtual trials preserve the physical variational problem."""
import numpy as np

from atomx import Atom
from atomx.methods.mcdhf.config import enumerate_jj_csfs
from atomx.methods.mcdhf.variational import (
    CorrelatedOrbitalFunctional,
    optimize_orbitals,
)
from atomx.methods.mcdhf.virtual_stability import populate_weak_virtuals
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.potentials.nuclear import make_nuclear_potential


def test_compact_empty_f_trials_lower_original_energy_without_projector_leakage(monkeypatch):
    base = Atom(Z=2, config='1s2', engine='bspline', corrections='none').solve().result
    csfs = [c for cfg in ('1s2', '4f2') for c in enumerate_jj_csfs(cfg)
            if c.total_J2 == 0 and c.parity == 1]
    functional = CorrelatedOrbitalFunctional(csfs, base._bsp_data, 2,
                                           make_nuclear_potential(2, 'fermi'))
    orbitals, projectors = {}, {}
    for key in functional.keys:
        k = key[1]
        spectrum = solve_dirac(functional.one_body[k], functional.metric[k], functional.ns, k,
                               nsp=functional.data.get('nsp'), nsq=functional.data.get('nsq'))
        projectors[k] = spectrum.coefficients
        trial = spectrum.coefficients[:, 0]
        orbitals[key] = (trial[:functional.ns], trial[functional.ns:])
    initial, _ = functional.evaluate(orbitals, gradient=False)
    corrected, ci, report = populate_weak_virtuals(
        functional, orbitals, occupation_threshold=1e-4, strengths=(1.,))
    checked, _ = functional.evaluate(corrected, gradient=False)
    assert checked < initial-1e-6
    assert abs(checked-ci.ground_energy) < 1e-12
    assert any(t['accepted'] for t in report['trials'])
    for key, waves in corrected.items():
        v = np.concatenate(waves)
        metric, projector = functional.metric[key[1]], projectors[key[1]]
        assert abs(v@metric@v-1.) < 1e-12
        assert np.linalg.norm(v-projector@(projector.T@metric@v)) < 1e-10

    progress = []
    _, optimized_ci, diagnostics = optimize_orbitals(
        functional, corrected, max_iter=2, tolerance=1e-12,
        occupation_floor=1e-4,
        progress_callback=lambda iteration, orbitals, ci, status: progress.append(
            (iteration, ci.ground_energy, status['energy_ha'])))
    assert progress[0][0] == 0
    assert progress[-1][0] == diagnostics['iterations']
    assert abs(progress[-1][1]-optimized_ci.ground_energy) < 1e-12
    assert all(abs(energy-status_energy) < 1e-12 for _, energy, status_energy in progress)
    assert optimized_ci.ground_energy <= checked
    assert diagnostics['termination_reason'] == 'maximum_iterations'

    # Force a failed secant search, then let the steepest-descent fallback
    # use the real helium Hamiltonian. It must lower the same CI energy.
    original_evaluate = functional.evaluate
    trial_calls = 0

    def reject_first_search(*args, **kwargs):
        nonlocal trial_calls
        if kwargs.get('gradient') is False:
            trial_calls += 1
            if trial_calls <= 28:
                raise FloatingPointError('injected invalid line-search trial')
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(functional, 'evaluate', reject_first_search)
    _, fallback_ci, fallback = optimize_orbitals(
        functional, corrected, max_iter=1, tolerance=1e-12, occupation_floor=1e-4)
    assert fallback['line_search_fallbacks'] == 1
    assert fallback['termination_reason'] == 'maximum_iterations'
    assert fallback_ci.ground_energy < checked

    def reject_all_trials(*args, **kwargs):
        if kwargs.get('gradient') is False:
            raise FloatingPointError('injected invalid line-search trial')
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(functional, 'evaluate', reject_all_trials)
    _, failed_ci, failure = optimize_orbitals(
        functional, corrected, max_iter=1, tolerance=1e-12, occupation_floor=1e-4)
    assert not failure['converged']
    assert failure['termination_reason'] == 'line_search_failed'
    assert failure['line_search_fallbacks'] == 1
    assert abs(failed_ci.ground_energy-checked) < 1e-12
