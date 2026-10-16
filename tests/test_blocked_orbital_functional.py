"""Full blocked objective, signed gradient and immutable native trial admission."""

from unittest.mock import patch

import numpy as np
import pytest
from test_ci_blocks import native_space as native_fixture

from atomx.methods.csf.blocked_functional import (
    BlockedCSFOrbitalFunctional,
    BlockedOrbitalTrial,
)
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.orbital_optimization import optimize_orbitals
from atomx.physics.csf.state_average import CIStateAverage

native_space = native_fixture


def test_complete_matrix_objective_and_all_radial_gradients(native_space):
    core, od, csfs = native_space
    target = CIStateAverage(((0, 1, 0, 2.), (2, -1, 0, 1.)))
    ordinary = CSFOrbitalFunctional(csfs, core._bsp_data, 4,
                         nuclear_potential=core._nuclear_potential, state_average=target)
    blocked = BlockedCSFOrbitalFunctional(csfs, core._bsp_data, 4, state_average=target,
                    reference_keys=((2, -1),), nuclear_potential=core._nuclear_potential)
    e, gradient, _ = ordinary.evaluate(od)
    actual, derivative, trial = blocked.evaluate(od)
    assert actual == pytest.approx(e, abs=1e-12)
    assert trial.full_H_residual_Ha < 1e-11
    for key in blocked.keys:
        np.testing.assert_allclose(derivative[key], gradient[key], atol=2e-10, rtol=1e-12)
    assert blocked.evaluate(od, gradient=False, ci_result=trial) == (actual, trial)
    _, replay, _ = blocked.evaluate(od, ci_result=trial)
    for key in replay:
        np.testing.assert_array_equal(replay[key], derivative[key])


def test_trial_ownership_point_binding_and_model_identity(native_space):
    core, od, csfs = native_space
    target = CIStateAverage(((0, 1, 0, 1.),))
    functional = BlockedCSFOrbitalFunctional(csfs, core._bsp_data, 4,
        state_average=target, reference_keys=((2, -1),), nuclear_potential=core._nuclear_potential)
    _, trial = functional.evaluate(od, gradient=False)
    for values in (trial.coefficients, trial.energies_Ha, trial.weights):
        with pytest.raises(ValueError):
            values.flags.writeable = True
    changed = dict(od)
    changed[2, -1] = tuple(-x for x in od[2, -1])
    with pytest.raises(ValueError, match='reference mismatch'):
        functional.evaluate(changed, ci_result=trial)
    other = BlockedCSFOrbitalFunctional(csfs, core._bsp_data, 4,
        state_average=CIStateAverage(((0, 1, 1, 1.),)),reference_keys=((2, -1),),
        nuclear_potential=core._nuclear_potential)
    with pytest.raises(ValueError, match='different physical objective'):
        other.evaluate(od, ci_result=trial)
    with pytest.raises(TypeError):
        BlockedOrbitalTrial()


def test_caps_roots_and_degenerate_cluster_reject(native_space):
    core, od, csfs = native_space
    target = CIStateAverage(((0, 1, 0, 1.),))
    with pytest.raises(MemoryError):
        BlockedCSFOrbitalFunctional(csfs,core._bsp_data,4,state_average=target,
                                   reference_keys=((2,-1),),memory_bytes=1)
    with pytest.raises(ValueError,match='root outside'):
        BlockedCSFOrbitalFunctional(csfs,core._bsp_data,4,
            state_average=CIStateAverage(((0,1,999,1.),)),reference_keys=((2,-1),))
    functional=BlockedCSFOrbitalFunctional(csfs,core._bsp_data,4,state_average=target,
                         reference_keys=((2,-1),),nuclear_potential=core._nuclear_potential)
    with patch('atomx.methods.csf.blocked_functional.SpinorCIHamiltonian.block_matrix',
               side_effect=lambda csfs,**kw:np.eye(len(csfs))), pytest.raises(ValueError,match='degenerate'):
        functional.evaluate(od,gradient=False)


def test_constrained_optimizer_consumes_owned_blocked_trials(native_space):
    core, od, csfs = native_space
    functional = BlockedCSFOrbitalFunctional(csfs, core._bsp_data, 4,
        state_average=CIStateAverage(((0,1,0,1.),(2,-1,0,1.))),reference_keys=((2,-1),),
        nuclear_potential=core._nuclear_potential)
    waves, trial, diagnostics = optimize_orbitals(functional, od, max_iter=2,
                    tolerance=1e-8,energy_tolerance=1e-10)
    assert len(diagnostics['energy_history']) >= 2
    assert diagnostics['energy_history'][-1] < diagnostics['energy_history'][0]
    energy, derivative, checked = functional.evaluate(waves,ci_result=trial)
    assert energy == diagnostics['energy_history'][-1]
    assert checked.full_H_residual_Ha < 1e-10
    assert all(np.isfinite(v).all() for v in derivative.values())
    for k in {key[1] for key in functional.keys}:
        x=np.column_stack([np.concatenate(waves[key]) for key in functional.keys if key[1]==k])
        np.testing.assert_allclose(x.T@functional.metric[k]@x,np.eye(x.shape[1]),atol=1e-12,rtol=0)


@pytest.mark.parametrize('phase', [1., -1.])
def test_ground_variational_change_matches_independent_full_rayleigh(native_space, phase):
    from atomx.physics.csf.spinor_ci import SpinorCIHamiltonian

    core, original, csfs = native_space
    reference = dict(original)
    reference[2, -1] = tuple(phase*v for v in reference[2, -1])
    functional = BlockedCSFOrbitalFunctional(csfs, core._bsp_data, 4,
        state_average=CIStateAverage(((0, 1, 0, 2.), (2, -1, 0, 1.))),
        reference_keys=((2, -1),), nuclear_potential=core._nuclear_potential)
    e0, old = functional.evaluate(reference, gradient=False)
    rng = np.random.default_rng(414)
    from atomx.numerics.bspline.dirac_assembly import solve_dirac
    positive = {k: solve_dirac(functional.one_body[k], functional.metric[k], functional.ns, k).coefficients
                for k in functional.metric}
    directions = {key: positive[key[1]]@rng.normal(size=positive[key[1]].shape[1])
                  for key in functional.keys}
    for step in (-.001, .001):
        trial = {}
        for kappa in {key[1] for key in functional.keys}:
            keys = [key for key in functional.keys if key[1] == kappa]
            c = np.column_stack([np.concatenate(reference[key]) for key in keys])
            d = np.column_stack([directions[key] for key in keys])
            s = functional.metric[kappa]
            d -= c@(.5*(c.T@s@d+d.T@s@c))
            t = c+step*d
            values, vectors = np.linalg.eigh(t.T@s@t)
            t = t@((vectors/np.sqrt(values))@vectors.T)
            for i, key in enumerate(keys):
                trial[key] = (t[:functional.ns, i], t[functional.ns:, i])
        increments = {key: tuple(t-r for t, r in zip(trial[key], reference[key], strict=True))
                      for key in functional.keys}
        e1, new = functional.evaluate(trial, gradient=False)
        difference = functional.ground_variational_upper_change(reference, trial, old, new,
                                                                 increments=increments)
        builder = SpinorCIHamiltonian(csfs, trial, core._bsp_data, 4, functional.ns,
                    reference_keys=((2, -1),), nuclear_potential=core._nuclear_potential)
        rayleigh = 0.
        diagonal = 0.
        for row in functional.plan['blocks']:
            sector = row['J2'], row['parity']
            columns = [i for i, state in enumerate(old.state_ids) if state[:2] == sector]
            if not columns:
                continue
            h = builder.block_matrix([csfs[i] for i in row['indices']], max_determinants=80000)
            for i in columns:
                v = old.coefficients[row['indices'], i]
                rayleigh += old.weights[i]*float(v@h@v/(v@v))
                diagonal += old.weights[i]*float(v*v@np.diag(h)/(v@v))
        assert difference == pytest.approx(rayleigh-e0, abs=1e-9)
        assert e1-e0 <= difference+1e-9
        assert abs(diagonal-rayleigh) > 1e-5  # Coherent CI response matters.
        assert functional.ground_variational_upper_change(reference, trial, old, new) == pytest.approx(
            difference, abs=1e-12)
        bad = {key: tuple(2*x for x in value) for key, value in increments.items()}
        assert functional.ground_variational_upper_change(reference, trial, old, new,
                                                          increments=bad) is None
        with pytest.raises(ValueError, match='reference mismatch'):
            functional.ground_variational_upper_change(trial, reference, old, new)


def test_ground_variational_change_rejects_excited_objectives(native_space):
    core, od, csfs = native_space
    functional = BlockedCSFOrbitalFunctional(csfs, core._bsp_data, 4,
        state_average=CIStateAverage(((0, 1, 1, 1.),)), reference_keys=((2, -1),),
        nuclear_potential=core._nuclear_potential)
    _, trial = functional.evaluate(od, gradient=False)
    assert functional.ground_variational_upper_change(od, od, trial, trial) is None
    with pytest.raises(ValueError, match='different physical objective'):
        functional.ground_variational_upper_change(od, od, object(), trial)


def test_ground_variational_change_rejects_nonphysical_boundary_extension(native_space):
    core, original, csfs = native_space
    od = {key: tuple(v.copy() for v in pair) for key, pair in original.items()}
    od[2, -1][0][0] = 1e-6
    functional = BlockedCSFOrbitalFunctional(csfs, core._bsp_data, 4,
        state_average=CIStateAverage(((0, 1, 0, 1.),)), reference_keys=((2, -1),),
        nuclear_potential=core._nuclear_potential)
    _, trial = functional.evaluate(od, gradient=False)
    assert functional.ground_variational_upper_change(od, od, trial, trial) is None


def test_blocked_gradient_is_reproducible_across_python_hash_seeds(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = '''
import runpy,sys,numpy as np
from atomx.methods.csf.blocked_functional import BlockedCSFOrbitalFunctional
from atomx.physics.csf.state_average import CIStateAverage
core,od,csfs=runpy.run_path('tests/test_ci_blocks.py')['native_space'].__wrapped__()
f=BlockedCSFOrbitalFunctional(csfs,core._bsp_data,4,
 state_average=CIStateAverage(((0,1,0,2.),(2,-1,0,1.))),reference_keys=((2,-1),),
 nuclear_potential=core._nuclear_potential)
e,g,ci=f.evaluate(od)
np.savez(sys.argv[1],energy=e,coefficients=ci.coefficients,
 **{f'n{k[0]}_k{k[1]}':v for k,v in g.items()})
'''
    paths = []
    for seed in ('1', '2'):
        path = tmp_path/f'seed{seed}.npz'
        environment = dict(os.environ, PYTHONHASHSEED=seed,
                           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        subprocess.run([sys.executable, '-c', script, str(path)],
                       cwd=Path(__file__).resolve().parents[1], env=environment,
                       check=True, capture_output=True, timeout=45)
        paths.append(path)
    with np.load(paths[0]) as a, np.load(paths[1]) as b:
        assert a.files == b.files
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
