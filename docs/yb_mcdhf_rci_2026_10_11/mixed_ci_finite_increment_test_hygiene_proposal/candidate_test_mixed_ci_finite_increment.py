"""Complete finite mixed-CI diagnostics; sealed numerical qualification only.

The shared finite fixture is built once: thirteen complete CI points, five
frozen ordinary H comparisons and three signed gradients. No optimizer call.
The independent Decimal oracle interprets the native assets without Atomx
kernels, retaining all phases, off-diagonal terms and endpoint norm corrections.
"""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.orbital_optimization import _polar_retraction
from atomx.numerics.ci_energy_increment import real_rank_one_response
from atomx.physics.csf.levels import ci_level_blocks
from atomx.physics.csf.state_average import CIStateAverage
from atomx.physics.csf.tracking import CIBasisTracking

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / 'docs/yb_mcdhf_rci_2026_10_11/mixed_ci_finite_increment_implementation'
CONFIGS = ('1s2', '1s1 2s1', '1s1 3s1', '2s2', '2s1 3s1', '3s2',
           '2p2', '2p1 3p1', '3p2')
SCALES = (1e-4, -1e-4, 1e-6, -1e-6, 1e-8, -1e-8)


def load_oracle():
    spec = importlib.util.spec_from_file_location('independent_decimal_oracle', EVIDENCE / 'decimal_oracle.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def nuclear(radius):
    return -2 / radius


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Calls:
    """Forward-only accounting; checks before work and preserves partial calls."""
    limits = dict(H_build=13, CI_solve=13, gradient=2, one_body_assembly=48,
                  nuclear_Dirac_solve=3, polar_retraction=36, physical_overlap=41,
                  tracking_SVD=80, SciPy_eigh=64, optimizer=0)

    def __init__(self, output):
        self.output = output
        self.started = {k: 0 for k in self.limits}
        self.completed = dict(self.started)
        self.flush()

    def flush(self):
        write_json(self.output / 'candidate_calls.json', dict(started=self.started,
                   completed=self.completed, limits=self.limits))

    def wrap(self, key, original):
        def counted(*args, **kwargs):
            if self.started[key] >= self.limits[key]:
                raise RuntimeError(f'prospective {key} cap exceeded')
            self.started[key] += 1
            self.flush()
            result = original(*args, **kwargs)
            self.completed[key] += 1
            self.flush()
            return result
        return counted

    def install(self, monkeypatch):
        from atomx.methods.csf import variational
        from atomx.numerics.bspline import dirac_assembly
        from atomx.physics.csf import matrix, radial_increment, tracking
        monkeypatch.setattr(matrix, 'build_ci_matrix', self.wrap('H_build', matrix.build_ci_matrix))
        monkeypatch.setattr(variational, 'ci_solve', self.wrap('CI_solve', variational.ci_solve))
        monkeypatch.setattr(CSFOrbitalFunctional, 'gradient_from_weights',
                            self.wrap('gradient', CSFOrbitalFunctional.gradient_from_weights))
        ordinary_one_body = dirac_assembly.build_dirac_matrices
        for module in (dirac_assembly, variational, radial_increment, matrix):
            monkeypatch.setattr(module, 'build_dirac_matrices', self.wrap('one_body_assembly', ordinary_one_body))
        monkeypatch.setattr(tracking, 'ci_subspace_overlaps',
                            self.wrap('physical_overlap', tracking.ci_subspace_overlaps))
        monkeypatch.setattr(np.linalg, 'svd', self.wrap('tracking_SVD', np.linalg.svd))
        from scipy import linalg
        monkeypatch.setattr(linalg, 'eigh', self.wrap('SciPy_eigh', linalg.eigh))


@pytest.fixture(scope='module')
def finite_problem():
    output_name = os.environ.get('ATOMX_MIXED_INCREMENT_OUTPUT')
    if not output_name:
        pytest.skip('live finite fixture requires root sealed resource/call execution driver')
    output = Path(output_name)
    output.mkdir(parents=True, exist_ok=True)
    monkeypatch = pytest.MonkeyPatch()
    calls = Calls(output)
    calls.install(monkeypatch)
    try:
        ns, ks = 35, 9
        data = prepare_bspline_data(make_knots(ns, ks, .005, 30), ns, ks)
        csfs = [c for config in CONFIGS for c in enumerate_jj_csfs(config)]
        target = CIStateAverage(((0, 1, 0, 1), (0, 1, 1, 3)))
        f = CSFOrbitalFunctional(csfs, data, 2, nuclear, state_average=target, retain_hamiltonian=True)
        # All setup work is counted separately from complete CI/H/g operations.
        from atomx.numerics.bspline import dirac_assembly
        one_body = {k: dirac_assembly.build_dirac_matrices(ns, k, 2, data, nuclear)
                    for k in sorted(f.metric)}
        layout, base_waves, assets = [], {}, {}
        for k in sorted(f.metric):
            eigen = calls.wrap('nuclear_Dirac_solve', solve_dirac)(f.one_body[k], f.metric[k], ns, k)
            P = eigen.coefficients.copy()
            keys = tuple(key for key in f.keys if key[1] == k)
            x = np.zeros((P.shape[1], len(keys)))
            ell = k if k > 0 else -k-1
            for col, (n, _) in enumerate(keys):
                x[n-ell-1+int(k > 0), col] = 1.
            layout.append((k, keys, P, x))
            assets[f'P_{k}'], assets[f'x0_{k}'] = P, x
            coefficients = P @ x
            for col, key in enumerate(keys):
                base_waves[key] = coefficients[:ns, col].copy(), coefficients[ns:, col].copy()
            assets[f'h_{k}'], assets[f'S_{k}'] = one_body[k]
        raw = np.load(EVIDENCE / 'raw_directions.npz', allow_pickle=False)
        directions = []
        for direction, seed in enumerate((413711, 413712)):
            rng = np.random.Generator(np.random.PCG64(seed))
            eta = []
            for k, _, _P, x in layout:
                z = rng.standard_normal(x.shape)
                assert np.array_equal(z, raw[f'z{direction}_{k}'])
                value = z-x @ ((x.T @ z + z.T @ x)*.5)
                eta.append(value)
            norm = np.sqrt(sum(float(e.ravel() @ e.ravel()) for e in eta))
            eta = [e/norm for e in eta]
            for (k, _, _, _), e in zip(layout, eta, strict=True):
                assets[f'eta{direction}_{k}'] = e
            directions.append(eta)
        points = [dict(label='base', waves=base_waves,
                       increments={key: tuple(np.zeros_like(v) for v in pair)
                                   for key, pair in base_waves.items()})]
        for direction, eta in enumerate(directions):
            for step in SCALES:
                waves, increments = {}, {}
                for (k, keys, P, x), e in zip(layout, eta, strict=True):
                    trial_x = calls.wrap('polar_retraction', _polar_retraction)(x+step*e)
                    trial, change = P @ trial_x, P @ (trial_x-x)
                    assets[f'x_{direction}_{step}_{k}'] = trial_x
                    for col, key in enumerate(keys):
                        waves[key] = trial[:ns, col].copy(), trial[ns:, col].copy()
                        increments[key] = change[:ns, col].copy(), change[ns:, col].copy()
                points.append(dict(label=f'd{direction}_{step}', direction=direction,
                                   step=step, waves=waves, increments=increments))
        keys = tuple(f.keys)
        for index, point in enumerate(points):
            assets[f'waves_{index}'] = np.array([point['waves'][k] for k in keys])
            assets[f'increments_{index}'] = np.array([point['increments'][k] for k in keys])
        assets['orbital_keys'] = np.array(keys, dtype=np.int64)
        np.savez_compressed(output / 'geometry_inputs.npz', **assets)
        # This irreversible input ledger is written before the first H/CI/g.
        write_json(output / 'generated_input_identity.json', dict(
            setup_before_first_H_CI_gradient=True,
            generated_arrays_sha256=sha(output / 'geometry_inputs.npz'),
            raw_seed_arrays_sha256=sha(EVIDENCE / 'raw_directions.npz'),
            NumPy_version=np.__version__, configs=CONFIGS, targets=target.states,
            csf_count=len(csfs), radial_keys=keys, data_keys=sorted(data),
            test_source_sha256=sha(Path(__file__)), oracle_sha256=sha(EVIDENCE / 'decimal_oracle.py'),
            setup_calls=dict(calls.completed)))
        for index, point in enumerate(points):
            energy, ci = f.evaluate(point['waves'], gradient=False)
            point.update(energy=energy, ci=ci)
            if index == 0:
                f.level_tracking = CIBasisTracking(target, csfs, ci, base_waves, f.metric)
            np.savez_compressed(output / f'CI_{index:02}.npz',
                H=ci.hamiltonian_for(csfs, point['waves'], keys, data, 2, ns, nuclear_potential=nuclear),
                energies=ci.state_energies, mixing=ci.mixing_coefficients)
        _, gradient, _ = f.evaluate(base_waves, ci_result=points[0]['ci'])
        _, endpoint_gradient, _ = f.evaluate(points[1]['waves'], ci_result=points[1]['ci'])
        np.savez_compressed(output / 'candidate_gradients.npz',
            baseline=np.array([gradient[k] for k in keys]), endpoint=np.array([endpoint_gradient[k] for k in keys]))
        # Independent old source imports happen in a fresh process and cannot
        # accidentally reuse candidate module globals or newly changed kernels.
        subprocess.run([sys.executable, str(EVIDENCE / 'frozen_comparator.py'), str(output)],
                       check=True, cwd=REPO, env={**os.environ, 'PYTHONPATH': str(
                           REPO / 'docs/yb_mcdhf_rci_2026_10_11/checkpoint_hamiltonian_reuse_implementation/source002')})
        oracle_module = load_oracle()
        structures, banks = [], {}
        orders = (data.get('ksp', ks), data.get('ksq', ks))
        for structure, projection, columns in f.radial_density.blocks:
            table = structure.table.tocsr()
            rows = tuple(tuple((int(table.indices[pos]), float(table.data[pos]))
                               for pos in range(table.indptr[row], table.indptr[row+1]))
                         for row in range(table.shape[0]))
            projected = projection.toarray() if hasattr(projection, 'toarray') else np.asarray(projection)
            structures.append((structure.keys, projected, structure.pairs.tolist(), rows,
                               tuple(columns), structure.radial_symmetry))
            for radial_key in structure.keys:
                if radial_key[0] == 'r':
                    for left, a in enumerate(('p', 'q')):
                        for right, b in enumerate(('p', 'q')):
                            bank_key = radial_key[1], left, right
                            if bank_key not in banks:
                                banks[bank_key] = f.cache.get_rkb_t(radial_key[1], a+b+a+b)[0].copy()
        # Binding complete captured assets also permits saved-array replay.
        oracle_assets = {}
        for (rank, left, right), bank in banks.items():
            oracle_assets[f'bank_{rank}_{left}_{right}'] = bank
        for block_index, block in enumerate(structures):
            _, projection, pairs, rows, columns, symmetry = block
            oracle_assets[f'projection_{block_index}'] = projection
            oracle_assets[f'pairs_{block_index}'] = np.array(pairs, dtype=np.int64)
            oracle_assets[f'columns_{block_index}'] = np.array(columns, dtype=np.int64)
        np.savez_compressed(output / 'oracle_assets.npz', **oracle_assets)
        write_json(output / 'oracle_signed_tables.json', dict(
            blocks=[dict(keys=b[0], rows=b[3], columns=b[4], radial_symmetry=b[5]) for b in structures],
            oracle_assets_sha256=sha(output / 'oracle_assets.npz'),
            geometry_inputs_sha256=sha(output / 'geometry_inputs.npz'),
            filtering='none; all table coefficients and all P/Q banks retained'))
        operator = oracle_module.DecimalDCRadialOperator(ns=ns, orders=orders, one_body=one_body,
                   banks=banks, blocks=structures, dimension=len(csfs))
        operator.add_point('base', base_waves)
        Hdecimal0 = operator.matrix('base')
        diagnostics = []
        for index, point in enumerate(points):
            point['diagnostic'] = f.mixed_ci_energy_difference(
                base_waves, point['waves'], points[0]['ci'], point['ci'], increments=point['increments'])
            if index:
                operator.add_point(f'ideal{index}', base_waves, point['increments'])
                operator.add_point(f'actual{index}', point['waves'])
                ideal, actual = operator.matrix(f'ideal{index}'), operator.matrix(f'actual{index}')
            else:
                ideal = actual = Hdecimal0
            targets = []
            _, blocks = ci_level_blocks(csfs, point['ci'], require_full=True)
            for target_row in point['diagnostic']['targets']:
                j, p, root = target_row['selected_trial']
                targets.append((target_row['weight'], point['ci'].mixing_coefficients[:, blocks[j, p][root]]))
            point['oracle_explicit'] = operator.contracted_difference(Hdecimal0, ideal, targets)
            point['oracle_reconstruction'] = operator.contracted_difference(ideal, actual, targets)
            point['oracle_complete_radial'] = operator.contracted_difference(Hdecimal0, actual, targets)
            point['oracle_matrix'] = actual
            diagnostics.append(dict(label=point['label'], diagnostic=point['diagnostic'],
                 oracle_explicit=str(point['oracle_explicit']),
                 oracle_reconstruction=str(point['oracle_reconstruction']),
                 oracle_complete_radial=str(point['oracle_complete_radial'])))
            write_json(output / 'finite_diagnostics.json', diagnostics)
        yield dict(f=f, points=points, gradient=gradient, layout=layout, directions=directions,
                   oracle=oracle_module, output=output, calls=calls)
    finally:
        calls.flush()
        monkeypatch.undo()


@pytest.mark.parametrize('offset', (0., 1e8))
@pytest.mark.parametrize('scale', (1., 2.3, .71))
@pytest.mark.parametrize('phase', (-1, 1))
def test_independent_finite_response_normalization_phase_residual_and_common_offset(offset, scale, phase):
    oracle = load_oracle()
    H0 = np.array([[2., .3], [.3, 3.]]) + offset*np.eye(2)
    H1 = np.array([[2.+4e-8, .3+8e-8], [.3+8e-8, 3.-3e-8]]) + offset*np.eye(2)
    c0, c1 = phase*scale*np.array([.6, .8]), scale*np.array([.61, .795])
    l0, l1 = 2.75+offset, 2.74999998+offset
    actual = real_rank_one_response(H0, H1, c0, c1, l0, l1, phase=phase)
    independent = oracle.response_oracle(H0, H1, c0, c1, l0, l1, phase)
    tolerance = 1e-13 if offset == 0 else 1e-11
    for name in ('reported_terms', 'rayleigh_terms', 'quotient_minus_reported'):
        assert np.max(np.abs(np.array(getattr(actual, name))-np.array(independent[name], dtype=float))) <= tolerance
    assert abs(actual.reported_response-float(independent['reported_response'])) <= tolerance
    assert abs(actual.rayleigh_response-float(independent['rayleigh_response'])) <= tolerance
    assert abs(float(independent['radial'])+actual.reported_response-float(independent['reported_delta'])) <= tolerance
    assert abs(float(independent['radial'])+actual.rayleigh_response-float(independent['quotient_delta'])) <= tolerance
    reverse = real_rank_one_response(H1, H0, c1, c0, l1, l0, phase=phase)
    reverse_oracle = oracle.response_oracle(H1, H0, c1, c0, l1, l0, phase)
    assert abs(float(reverse_oracle['radial'])+reverse.reported_response
               +float(independent['reported_delta'])) <= tolerance
    assert abs(float(reverse_oracle['radial'])+reverse.rayleigh_response
               +float(independent['quotient_delta'])) <= tolerance
    assert np.array_equal(c0, phase*scale*np.array([.6, .8]))


@pytest.mark.parametrize('index', range(13))
def test_complete_signed_native_increment_against_independent_decimal(finite_problem, index):
    point = finite_problem['points'][index]
    result = point['diagnostic']
    assert result['negative_signed_density_terms'] > 0
    assert result['optimizer_adapter'] is False and result['certified_error_bound'] is False
    assert abs(result['explicit_radial']-float(point['oracle_explicit'])) <= 1e-13
    assert abs(result['reconstruction_radial']-float(point['oracle_reconstruction'])) <= 1e-13
    assert abs(result['explicit_radial']+result['reconstruction_radial']
               -float(point['oracle_complete_radial'])) <= 1e-13
    # Physical finite total identity includes independently calculated CI
    # response against actual retained H (including finite eigen residuals).
    f, baseline = finite_problem['f'], finite_problem['points'][0]
    H0 = baseline['ci'].hamiltonian_snapshot.matrix_Ha
    H1 = point['ci'].hamiltonian_snapshot.matrix_Ha
    _, blocks0 = ci_level_blocks(f.csfs, baseline['ci'], require_full=True)
    _, blocks1 = ci_level_blocks(f.csfs, point['ci'], require_full=True)
    reported, rayleigh = 0., 0.
    for target in result['targets']:
        j0, p0, r0 = target['selected_reference']
        j1, p1, r1 = target['selected_trial']
        col0, col1 = blocks0[j0, p0][r0], blocks1[j1, p1][r1]
        oracle = finite_problem['oracle'].response_oracle(H0, H1,
             baseline['ci'].mixing_coefficients[:, col0], point['ci'].mixing_coefficients[:, col1],
             baseline['ci'].state_energies[col0], point['ci'].state_energies[col1], target['phase'])
        reported += target['weight']*float(oracle['reported_response'])
        rayleigh += target['weight']*float(oracle['rayleigh_response'])
    oracle_reported = float(point['oracle_complete_radial'])+reported
    oracle_rayleigh = float(point['oracle_complete_radial'])+rayleigh
    assert abs(result['reported_change']-oracle_reported) <= 1e-12
    assert abs(result['rayleigh_change']-oracle_rayleigh) <= 1e-12
    if abs(oracle_rayleigh) > 1e-13:
        assert abs(result['rayleigh_change']-oracle_rayleigh) <= .1*abs(oracle_rayleigh)
    if index == 0:
        assert result['explicit_radial'] == 0.
        assert result['reconstruction_radial'] == 0.
    with localcontext() as context:
        context.prec = 80
        matrix_error = max(abs(Decimal.from_float(float(value))-exact)
             for row, exact_row in zip(H1, point['oracle_matrix'], strict=True)
             for value, exact in zip(row, exact_row, strict=True))
    assert float(matrix_error) <= 1e-8


def test_frozen_default_H_and_complete_signed_gradient_parity(finite_problem):
    output = finite_problem['output']
    old = np.load(output / 'frozen_comparators.npz', allow_pickle=False)
    for index in (0, 1, 2, 7, 8):
        new = np.load(output / f'CI_{index:02}.npz', allow_pickle=False)['H']
        assert np.array_equal(new, old[f'H_{index}'])
    current = np.load(output / 'candidate_gradients.npz', allow_pickle=False)['baseline']
    assert np.array_equal(current, old['baseline_gradient'])
    counters = json.loads((output / 'frozen_calls.json').read_text())
    assert counters['completed'] == dict(H_build=5, CI_solve=0, gradient=1, one_body_assembly=6,
                                         physical_overlap=2, tracking_SVD=2)
    for key, maximum in Calls.limits.items():
        if key != 'SciPy_eigh':
            assert finite_problem['calls'].completed[key] == maximum
    assert finite_problem['calls'].completed['SciPy_eigh'] <= Calls.limits['SciPy_eigh']


def test_signed_central_directional_derivative_in_same_constrained_space(finite_problem):
    problem = finite_problem
    for direction in range(2):
        derivative = 0.
        for (_, keys, P, _), eta in zip(problem['layout'], problem['directions'][direction], strict=True):
            tangent = P @ eta
            derivative += sum(float(problem['gradient'][key] @ tangent[:, col])
                              for col, key in enumerate(keys))
        plus, minus = problem['points'][3+direction*6:5+direction*6]
        measured = (plus['diagnostic']['rayleigh_change']-minus['diagnostic']['rayleigh_change'])/(2e-6)
        assert measured == pytest.approx(derivative, rel=1e-5, abs=1e-8)


@pytest.mark.parametrize('change', ('float32', 'float128', 'complex', 'missing', 'basis_reorder',
    'parity', 'keys', 'target', 'rank', 'subclass', 'no_owner', 'subset', 'operator',
    'eigen_residual', 'ci_float32', 'ci_float128'))
def test_unsupported_scope_stale_density_or_owner_rejects_without_calls(finite_problem, change):
    problem = finite_problem
    f, base = problem['f'], problem['points'][0]
    consumer, waves, ci, increments = copy.copy(f), dict(base['waves']), copy.copy(base['ci']), base['increments']
    if change in ('float32', 'float128', 'complex'):
        dtype = {'float32': np.float32, 'float128': np.longdouble, 'complex': np.complex128}[change]
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
        consumer.level_tracking = copy.copy(f.level_tracking)
        groups = consumer.level_tracking._groups
        object.__setattr__(consumer.level_tracking, '_groups', ((groups[0][0]+groups[1][0], .5),))
    elif change == 'subclass':
        class Unsupported(CSFOrbitalFunctional):
            pass
        consumer.__class__ = Unsupported
    elif change == 'no_owner':
        ci.hamiltonian_snapshot = None
    elif change == 'subset':
        ci.state_energies = ci.state_energies[:-1]
        ci.mixing_coefficients = ci.mixing_coefficients[:, :-1]
    elif change == 'operator':
        from dataclasses import fields
        original_owner = ci.hamiltonian_snapshot
        ci.hamiltonian_snapshot = object.__new__(type(original_owner))
        for field in fields(original_owner):
            object.__setattr__(ci.hamiltonian_snapshot, field.name, getattr(original_owner, field.name))
        object.__setattr__(ci.hamiltonian_snapshot, 'hamiltonian_terms', ('dirac-coulomb', 'static-breit'))
    elif change in ('ci_float32', 'ci_float128'):
        ci.mixing_coefficients = ci.mixing_coefficients.astype(
            np.float32 if change == 'ci_float32' else np.longdouble)
    elif change == 'eigen_residual':
        ci.state_energies = ci.state_energies.copy()
        ci.state_energies[0] += 2e-8
    before = dict(problem['calls'].started)
    with pytest.raises((ValueError, TypeError)):
        consumer.mixed_ci_energy_difference(base['waves'], waves, base['ci'], ci, increments=increments)
    assert problem['calls'].started == before


def test_legacy_fixed_density_and_constrained_mixed_guards_remain(finite_problem):
    f, points = finite_problem['f'], finite_problem['points']
    calls = dict(finite_problem['calls'].started)
    assert f.fixed_density_energy_difference(points[0]['waves'], points[1]['waves'],
                                            points[0]['ci'], points[1]['ci']) is None
    assert f.constrained_energy_difference(points[0]['waves'], points[1]['waves'],
                                          points[0]['ci'], points[1]['ci']) is None
    assert finite_problem['calls'].started == calls
