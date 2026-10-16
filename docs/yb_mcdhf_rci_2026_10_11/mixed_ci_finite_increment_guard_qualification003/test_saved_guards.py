"""Sealed successor: saved medium-step derivatives and genuine-owner refusals.

One native baseline H is built, without a CI solve or gradient. All thirteen
previous finite endpoints remain saved evidence; no radial increment is rerun.
"""
import copy
import json
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.ci import CIResult
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.knots import make_knots
from atomx.physics.csf.hamiltonian_snapshot import build_ci_hamiltonian_snapshot
from atomx.physics.csf.state_average import CIStateAverage
from atomx.physics.csf.tracking import CIBasisTracking

BASE = Path(__file__).resolve().parent
PRIOR = BASE.parent / 'mixed_ci_finite_increment_implementation'
CONFIGS = ('1s2', '1s1 2s1', '1s1 3s1', '2s2', '2s1 3s1', '3s2',
           '2p2', '2p1 3p1', '3p2')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def nuclear(radius):
    return -2 / radius


class Calls:
    """Forward-only calls, with zero budgets for prohibited physical work."""
    limits = dict(H_build=1, CI_solve=0, gradient=0, one_body_assembly=6,
                  nuclear_Dirac_solve=0, polar_retraction=0, physical_overlap=2,
                  tracking_SVD=0, SciPy_eigh=0, NumPy_eigh=0, optimizer=0,
                  radial_increment=0)

    def __init__(self, output):
        self.output = output
        self.started = {key: 0 for key in self.limits}
        self.completed = dict(self.started)
        self.one_body_outputs = []
        self.flush()

    def flush(self):
        write_json(self.output / 'calls.json', dict(started=self.started,
                   completed=self.completed, limits=self.limits))

    def wrap(self, key, original):
        def counted(*args, **kwargs):
            if self.started[key] >= self.limits[key]:
                self.flush()
                raise RuntimeError(f'prospective {key} cap exceeded')
            self.started[key] += 1
            self.flush()
            result = original(*args, **kwargs)
            self.completed[key] += 1
            self.flush()
            return result
        return counted

    def install(self, patch):
        from scipy import linalg

        from atomx.methods import ci
        from atomx.methods.csf import variational
        from atomx.numerics.bspline import dirac_assembly, orbital_optimization
        from atomx.physics.csf import matrix, radial_increment, tracking

        patch.setattr(matrix, 'build_ci_matrix', self.wrap('H_build', matrix.build_ci_matrix))
        solve = ci.ci_solve
        for module in (ci, variational):
            patch.setattr(module, 'ci_solve', self.wrap('CI_solve', solve))
        patch.setattr(CSFOrbitalFunctional, 'gradient_from_weights',
                      self.wrap('gradient', CSFOrbitalFunctional.gradient_from_weights))
        one_body = dirac_assembly.build_dirac_matrices
        counted_one_body = self.wrap('one_body_assembly', one_body)
        def recorded_one_body(*args, **kwargs):
            h, metric = counted_one_body(*args, **kwargs)
            self.one_body_outputs.append((int(args[1]), h.copy(), metric.copy()))
            return h, metric
        for module in (dirac_assembly, variational, radial_increment, matrix):
            patch.setattr(module, 'build_dirac_matrices', recorded_one_body)
        patch.setattr(dirac_assembly, 'solve_dirac',
                      self.wrap('nuclear_Dirac_solve', dirac_assembly.solve_dirac))
        patch.setattr(orbital_optimization, '_polar_retraction',
                      self.wrap('polar_retraction', orbital_optimization._polar_retraction))
        patch.setattr(orbital_optimization, 'optimize_orbitals',
                      self.wrap('optimizer', orbital_optimization.optimize_orbitals))
        patch.setattr(tracking, 'ci_subspace_overlaps',
                      self.wrap('physical_overlap', tracking.ci_subspace_overlaps))
        patch.setattr(np.linalg, 'svd', self.wrap('tracking_SVD', np.linalg.svd))
        patch.setattr(np.linalg, 'eigh', self.wrap('NumPy_eigh', np.linalg.eigh))
        patch.setattr(linalg, 'eigh', self.wrap('SciPy_eigh', linalg.eigh))
        patch.setattr(radial_increment, 'bspline_dc_radial_increment',
                      self.wrap('radial_increment', radial_increment.bspline_dc_radial_increment))


@pytest.fixture(scope='module')
def saved_problem():
    output_name = os.environ.get('ATOMX_MIXED_GUARD_OUTPUT')
    if not output_name:
        pytest.fail('successor requires the explicit root-sealed driver')
    output = Path(output_name)
    calls = Calls(output)
    patch = pytest.MonkeyPatch()
    calls.install(patch)
    try:
        assert np.__version__ == '2.2.6'
        geometry = np.load(PRIOR / 'actual/geometry_inputs.npz', allow_pickle=False)
        keys = tuple(tuple(int(v) for v in key) for key in geometry['orbital_keys'])
        waves = {key: tuple(v.copy() for v in pair)
                 for key, pair in zip(keys, geometry['waves_0'], strict=True)}
        increments = {key: tuple(v.copy() for v in pair)
                      for key, pair in zip(keys, geometry['increments_0'], strict=True)}
        assert all(np.count_nonzero(v) == 0 for pair in increments.values() for v in pair)
        data = prepare_bspline_data(make_knots(35, 9, .005, 30), 35, 9)
        csfs = [c for config in CONFIGS for c in enumerate_jj_csfs(config)]
        target = CIStateAverage(((0, 1, 0, 1), (0, 1, 1, 3)))
        f = CSFOrbitalFunctional(csfs, data, 2, nuclear, state_average=target,
                                 retain_hamiltonian=True)
        assert len(csfs) == 29 and tuple(f.keys) == keys
        assert len(calls.one_body_outputs) == 3
        for k, native_raw, native_metric in calls.one_body_outputs:
            assert np.array_equal(native_raw, geometry[f'h_{k}'])
            assert np.array_equal(native_metric, geometry[f'S_{k}'])
            assert native_raw.tobytes() == geometry[f'h_{k}'].tobytes()
            assert native_metric.tobytes() == geometry[f'S_{k}'].tobytes()
        for k in sorted(f.metric):
            assert np.array_equal(f.metric[k], geometry[f'S_{k}'])
            raw = geometry[f'h_{k}']
            mapped = np.tril(raw) + np.tril(raw, -1).T
            assert np.array_equal(f.one_body[k], mapped)
            assert f.one_body[k].tobytes() == mapped.tobytes()
        owner = build_ci_hamiltonian_snapshot(csfs, waves, keys, data, 2, 35,
            nuclear_potential=nuclear, convol_cache=f.cache, max_determinants=f.max_determinants)
        baseline = np.load(PRIOR / 'actual/CI_00.npz', allow_pickle=False)
        assert len(calls.one_body_outputs) == 6
        raw_assets = {}
        for index, (k, native_raw, native_metric) in enumerate(calls.one_body_outputs):
            assert np.array_equal(native_raw, geometry[f'h_{k}'])
            assert np.array_equal(native_metric, geometry[f'S_{k}'])
            assert native_raw.tobytes() == geometry[f'h_{k}'].tobytes()
            assert native_metric.tobytes() == geometry[f'S_{k}'].tobytes()
            raw_assets[f'raw_h_{index}_k{k}'] = native_raw
            raw_assets[f'S_{index}_k{k}'] = native_metric
        np.savez_compressed(output / 'captured_native_raw_hS.npz', **raw_assets)
        assert np.array_equal(owner.matrix_Ha, baseline['H'])
        diagnostics = json.loads((PRIOR / 'actual/finite_diagnostics.json').read_text())
        header = diagnostics[0]['diagnostic']
        assert owner.input_sha256 == header['endpoint_input_sha256'][0]
        assert owner.matrix_sha256 == header['endpoint_H_sha256'][0]
        saved_controls = []
        for index in range(13):
            point = np.load(PRIOR / 'actual' / f'CI_{index:02}.npz', allow_pickle=False)
            H, C, energies = point['H'], point['mixing'], point['energies']
            assert H.shape == C.shape == (29, 29)
            residual = float(np.max(np.linalg.norm(H @ C - C * energies, axis=0)))
            gram = float(np.max(np.abs(C.T @ C - np.eye(29))))
            assert residual <= 1e-8 and gram <= 1e-10
            saved_controls.append(dict(index=index, residual=residual, Gram=gram))
        ci = CIResult(baseline['energies'].copy(), baseline['mixing'].copy(),
                      hamiltonian_snapshot=owner)
        assert ci.hamiltonian_for(csfs, waves, keys, data, 2, 35,
            nuclear_potential=nuclear, max_determinants=f.max_determinants) is owner.matrix_Ha
        f.level_tracking = CIBasisTracking(target, csfs, ci, waves, f.metric)
        equal = CIStateAverage(((0, 1, 0, 1), (0, 1, 1, 1)))
        rank_tracker = CIBasisTracking(equal, csfs, ci, waves, f.metric)
        assert any(len(group) == 2 for group, _ in rank_tracker._groups)
        legacy_saved = np.load(PRIOR / 'actual/CI_01.npz', allow_pickle=False)
        legacy_ci = CIResult(legacy_saved['energies'].copy(), legacy_saved['mixing'].copy())
        legacy_waves = {key: tuple(v.copy() for v in pair)
                        for key, pair in zip(keys, geometry['waves_1'], strict=True)}
        write_json(output / 'baseline_owner_validation.json', dict(
            genuine_public_producer=True, H_bitwise_saved_baseline=True,
            all6_native_raw_h_S_bitwise_saved=True,
            functional_H_bitwise_lower_completed_saved_raw=True,
            eigenpairs_attached_after_independent_validation=True,
            owner_header=owner.identity_payload(), input_sha256=owner.input_sha256,
            matrix_sha256=owner.matrix_sha256, all13saved_eigenpair_controls=saved_controls,
            orbital_keys=keys, CSFs=len(csfs), setup_calls=calls.completed))
        yield dict(output=output, calls=calls, f=f, ci=ci, waves=waves,
                   increments=increments, geometry=geometry, keys=keys,
                   diagnostics=diagnostics, rank_tracker=rank_tracker, equal=equal,
                   legacy_ci=legacy_ci, legacy_waves=legacy_waves,
                   gradient=np.load(PRIOR / 'actual/candidate_gradients.npz',
                                    allow_pickle=False)['baseline'])
    finally:
        calls.flush()
        patch.undo()


@pytest.mark.parametrize('direction', (0, 1))
def test_saved_medium_step_original_derivative_gate(saved_problem, direction):
    p = saved_problem
    before = dict(p['calls'].started)
    derivative = 0.
    for k in (-2, -1, 1):
        tangent = p['geometry'][f'P_{k}'] @ p['geometry'][f'eta{direction}_{k}']
        active = [key for key in p['keys'] if key[1] == k]
        for column, key in enumerate(active):
            derivative += float(p['gradient'][p['keys'].index(key)] @ tangent[:, column])
    by_label = {row['label']: row['diagnostic'] for row in p['diagnostics']}
    scales = []
    for h in (1e-4, 1e-6, 1e-8):
        measured = (by_label[f'd{direction}_{h}']['rayleigh_change']
                    - by_label[f'd{direction}_{-h}']['rayleigh_change']) / (2 * h)
        gate = max(1e-8, 1e-5 * abs(derivative))
        scales.append(dict(h=h, derivative=measured, error=measured-derivative,
                           unchanged_gate=gate, gate_passed=abs(measured-derivative) <= gate))
    write_json(p['output'] / f'saved_direction{direction}_scales.json', dict(
        gradient_derivative=derivative, scales=scales, selected_h=1e-6,
        original_coarse_failure_preserved=True, new_radial_evaluation=False))
    assert scales[0]['gate_passed'] is False
    assert scales[1]['derivative'] == pytest.approx(derivative, rel=1e-5, abs=1e-8)
    assert p['calls'].started == before


@pytest.mark.parametrize('change', ('float32', 'float128', 'complex', 'missing',
    'basis_reorder', 'parity', 'keys', 'target', 'rank', 'subclass', 'no_owner',
    'subset', 'operator', 'eigen_residual', 'ci_float32', 'ci_float128'))
def test_unsupported_scope_rejects_without_physical_calls(saved_problem, change):
    p = saved_problem
    f, original_ci = p['f'], p['ci']
    consumer, waves, ci = copy.copy(f), dict(p['waves']), copy.copy(original_ci)
    if change in ('float32', 'float128', 'complex'):
        dtype = {'float32': np.float32, 'float128': np.longdouble,
                 'complex': np.complex128}[change]
        waves = {key: tuple(v.astype(dtype) for v in pair) for key, pair in waves.items()}
    elif change == 'missing':
        waves.pop(next(iter(waves)))
    elif change == 'basis_reorder':
        consumer.csfs = list(reversed(f.csfs))
    elif change == 'parity':
        consumer.csfs = copy.deepcopy(f.csfs)
        consumer.csfs[0].parity *= -1
    elif change == 'keys':
        consumer.keys = tuple(reversed(f.keys))
    elif change == 'target':
        consumer.state_average = CIStateAverage(((0, 1, 0, 3), (0, 1, 1, 1)))
    elif change == 'rank':
        consumer.state_average, consumer.level_tracking = p['equal'], p['rank_tracker']
    elif change == 'subclass':
        class Unsupported(CSFOrbitalFunctional):
            pass
        consumer.__class__ = Unsupported
    elif change == 'no_owner':
        ci.hamiltonian_snapshot = None
    elif change == 'subset':
        ci.state_energies = ci.state_energies[:-1]
        ci.mixing_coefficients = ci.mixing_coefficients[:, :-1]
    elif change in ('ci_float32', 'ci_float128'):
        ci.mixing_coefficients = ci.mixing_coefficients.astype(
            np.float32 if change == 'ci_float32' else np.longdouble)
    elif change == 'eigen_residual':
        ci.state_energies = ci.state_energies.copy()
        ci.state_energies[0] += 2e-8
    before = dict(p['calls'].started)
    if change == 'operator':
        # Genuine public coverage refusal plus the old malformed-descriptor
        # negative case. The deliberate shell below is NEVER admitted as a
        # saved-H fixture; it exists only to verify early rejection.
        with pytest.raises(ValueError, match='Dirac-Coulomb only'):
            ci.hamiltonian_for(f.csfs, waves, f.keys, f.data, f.Z, f.ns,
                nuclear_potential=nuclear, include_breit=True)
        with pytest.raises(TypeError):
            consumer.mixed_ci_energy_difference(p['waves'], waves, original_ci, ci,
                increments=p['increments'], include_breit=True)
        owner = original_ci.hamiltonian_snapshot
        malformed = object.__new__(type(owner))
        for field in fields(owner):
            object.__setattr__(malformed, field.name, getattr(owner, field.name))
        object.__setattr__(malformed, 'hamiltonian_terms', ('dirac-coulomb', 'static-breit'))
        ci.hamiltonian_snapshot = malformed
        with pytest.raises(ValueError, match='complete native DC owners'):
            consumer.mixed_ci_energy_difference(p['waves'], waves, original_ci, ci,
                                                increments=p['increments'])
    else:
        with pytest.raises((ValueError, TypeError)):
            consumer.mixed_ci_energy_difference(p['waves'], waves, original_ci, ci,
                                                increments=p['increments'])
    assert p['calls'].started == before


def test_legacy_fixed_density_and_constrained_mixed_guards(saved_problem):
    p, before = saved_problem, dict(saved_problem['calls'].started)
    f, ci, waves = p['f'], p['ci'], p['waves']
    assert f.fixed_density_energy_difference(waves, p['legacy_waves'], ci, p['legacy_ci']) is None
    assert f.constrained_energy_difference(waves, p['legacy_waves'], ci, p['legacy_ci']) is None
    assert p['calls'].started == before
