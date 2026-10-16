"""Prepared full531 same-point chain panel. Root seal and qualification required."""
from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import importlib.util
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def plain(value):
    """Normalize scalar metadata; numerical arrays belong only in NPZ."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError('JSON metadata requires string keys')
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if type(value).__module__.startswith('numpy'):
        if type(value).__name__ == 'ndarray' or getattr(value, 'shape', None) != ():
            raise TypeError('Numerical arrays must be archived in NPZ')
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unsupported metadata scalar: ' + type(value).__name__)


def atomic_json(path, value):
    temporary = Path(str(path) + '.part')
    with temporary.open('w') as stream:
        json.dump(plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def host(contract, cpus, cpu):
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:')) / 2**20
    loads = os.getloadavg()
    caps = contract['resources']
    return dict(available_cpus=cpus, chosen_cpu=cpu, load_average=list(loads),
                load1_per_available_cpu=loads[0] / len(cpus), host_available_GiB=available,
                admitted=len(cpus) >= 3 and cpu in cpus and available >= caps['minimum_host_available_GiB']
                and loads[0] / len(cpus) <= caps['max_load1_per_available_CPU'])


def input_checks(contract):
    checks = []
    for name, expected in contract['inputs_sha256'].items():
        checks.append(dict(name='input:' + name, passed=sha(ROOT / name) == expected))
    identity = load(ROOT / contract['candidate_source_identity_path'])
    source = ROOT / contract['candidate_frozen_source_path']
    actual = {'src/' + str(p.relative_to(source)): sha(p) for p in sorted((source / 'atomx').rglob('*.py'))}
    checks.extend((dict(name='source354 exact', passed=actual == identity['all_atomx_python_sha256'] and len(actual) == 354),
                   dict(name='source354 canonical SHA', passed=digest(actual) ==
                        identity['source_map_sha256'] == contract['candidate_source_map_sha256'])))
    return checks


def require_checks(checks):
    failed = [row['name'] for row in checks if not row['passed']]
    if failed:
        raise ValueError('Panel input/gate failure: ' + ', '.join(failed))


def execution_admission(contract, approval):
    # These authorization guards run before loading the scientific package/helper.
    required = dict(execution_authorized_by_root=True, paired_chain_performance_authorized=True,
                    complete531_qualification_required=True, exclusive_resource_window_authorized=True,
                    root_physics_diagnostic_completed=True, driver_sha256=sha(__file__),
                    preparation_contract_sha256=sha(HERE / 'contract.json'),
                    candidate_source_map_sha256=contract['candidate_source_map_sha256'],
                    max_public_H_builds=21, max_CI_solves=14, max_full_gradients=14, max_optimizer_calls=0)
    if any(approval.get(key) != value or (type(value) is bool and type(approval.get(key)) is not bool)
           for key, value in required.items()):
        raise ValueError('Missing exact root panel authorization')
    for key in ('qualification_result', 'qualification_verification', 'qualification_driver', 'qualification_contract'):
        item = contract['prerequisites'][key]
        path = ROOT / item['path']
        if approval.get(key + '_sha256') != sha(path):
            raise ValueError('Missing exact prerequisite binding: ' + key)
    result = load(ROOT / contract['prerequisites']['qualification_result']['path'])
    proof = load(ROOT / contract['prerequisites']['qualification_verification']['path'])
    if (result.get('qualification_numeric_passed') is not True or not result.get('checks')
            or not all(row.get('passed') is True for row in result['checks'])
            or proof.get('all_passed') is not True or not proof.get('checks')
            or not all(row.get('passed') is True for row in proof['checks'])
            or not proof.get('inputs_sha256')
            or result.get('candidate_source_map_sha256') != contract['candidate_source_map_sha256']
            or result.get('point_arrays_sha256') != contract['fixed_conditions']['point_arrays_sha256']
            or result.get('original_metric_sha256') != contract['fixed_conditions']['original_metric_sha256']
            or result.get('optimizer_calls') != 0 or result.get('original_stationarity_passed') is not False):
        raise ValueError('Full531 qualification is not a valid same-point predecessor')
    for name, expected in proof['inputs_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Independent qualification proof input changed: ' + name)
    proof_inputs = proof['inputs_sha256']
    for key in ('qualification_result', 'qualification_contract', 'qualification_driver'):
        name = contract['prerequisites'][key]['path']
        if proof_inputs.get(name) != approval[key + '_sha256']:
            raise ValueError('Independent qualification proof omitted exact predecessor: ' + key)
    qualification = load(ROOT / contract['prerequisites']['qualification_contract']['path'])
    for key, path in (('arrays_sha256', 'arrays.npz'), ('owner_header_sha256', 'owner_header.json'),
                      ('signed_RDM_sha256', 'signed_public_RDM.json')):
        if result.get(key) != sha(ROOT / Path(contract['prerequisites']['qualification_result']['path']).parent / path):
            raise ValueError('Qualified scientific output changed: ' + path)
    if (result.get('contract_sha256') != approval['qualification_contract_sha256']
            or result.get('driver_sha256') != approval['qualification_driver_sha256']
            or qualification['candidate_source_map_required_sha256'] != contract['candidate_source_map_sha256']
            or result.get('actual_calls') != {
                'started': dict(H_build=3, CI_solve=2, gradient=2),
                'completed': dict(H_build=3, CI_solve=2, gradient=2)}):
        raise ValueError('Qualification source/consumer manifest changed')
    return qualification


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    contract = load(HERE / 'contract.json')
    checks = input_checks(contract)
    driver = ROOT / contract['prerequisites']['qualification_driver']['path']
    if driver.exists():
        tree = ast.parse(driver.read_text())
        helper = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == 'prepare_point'), None)
        checks.append(dict(name='reusable pure point setup exact AST', passed=helper is not None and
            hashlib.sha256(ast.dump(helper, include_attributes=False).encode()).hexdigest() ==
            contract['qualification_prepare_point_AST_sha256']))
    checks.append(dict(name='preparation imports no Atomx', passed=not any(
        name == 'atomx' or name.startswith('atomx.') for name in sys.modules)))
    require_checks(checks)
    cpus = sorted(os.sched_getaffinity(0))
    cpu = cpus[-3] if len(cpus) >= 3 else None
    proof = dict(preparation_only=True, checks=checks, all_passed=True, check_count=len(checks),
                 contract_sha256=sha(HERE / 'contract.json'), driver_sha256=sha(__file__),
                 source_map_sha256=contract['candidate_source_map_sha256'],
                 qualification_result_available=(ROOT / contract['prerequisites']['qualification_result']['path']).exists(),
                 runtime_root_seal_pending=True, host=host(contract, cpus, cpu),
                 H_builds=0, CI_solves=0, gradients=0, optimizer_calls=0)
    atomic_json(HERE / 'preparation_verification.json', proof)
    print(json.dumps({key: value for key, value in proof.items() if key != 'checks'}, indent=2))


def worker(kind, execution_path, output_dir):
    contract, approval = load(HERE / 'contract.json'), load(execution_path)
    qualification_contract = execution_admission(contract, approval)
    require_checks(input_checks(contract))
    if output_dir.resolve() != (HERE / 'actual' / kind).resolve() or any(output_dir.glob('*.npz')):
        raise ValueError('Worker must use a new declared panel output; preserve old artifacts')
    started = time.perf_counter()
    cpu, cpus = approval['chosen_cpu'], approval['available_cpus']
    admission = host(contract, cpus, cpu)
    if not admission['admitted']:
        raise RuntimeError('Worker host guard failed before scientific setup')
    os.sched_setaffinity(0, {cpu})
    limit = int(contract['resources']['address_space_cap_GiB'] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    source = ROOT / contract['candidate_frozen_source_path']
    sys.path.insert(0, str(source))
    import numpy as np

    import atomx.methods.csf.variational as functional_module
    import atomx.physics.csf.matrix as matrix_module
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.physics.csf.effective_one_body import CIReferenceBinding

    helper = import_file('qualified_point_helper', ROOT / contract['prerequisites']['qualification_driver']['path'])
    canon = import_file('original_target_metadata', ROOT / contract['canonical_target_helper_path'])
    setup_construction = dict(calls=0, completed=0, seconds=0.)
    raw_init = CSFOrbitalFunctional.__init__

    @functools.wraps(raw_init)
    def timed_init(self, *args, **kwargs):
        setup_construction['calls'] += 1
        clock = time.perf_counter()
        try:
            result = raw_init(self, *args, **kwargs)
            setup_construction['completed'] += 1
            return result
        finally:
            setup_construction['seconds'] += time.perf_counter() - clock

    CSFOrbitalFunctional.__init__ = timed_init
    try:
        point = helper.prepare_point(qualification_contract)
    finally:
        CSFOrbitalFunctional.__init__ = raw_init
    require_checks(point['setup_checks'])
    require_checks([dict(name='exact four declared setup functionals', passed=
        setup_construction['calls'] == setup_construction['completed'] == 4)])
    setup_seconds = time.perf_counter() - started
    cases = ([('warmup', 0, 'ordinary'), ('warmup', 0, 'retained')] +
             [('measured', pair, mode) for pair in range(1, 6) for mode in
              (('ordinary', 'retained') if pair % 2 else ('retained', 'ordinary'))]
             if kind == 'warm' else [('cold', 0, kind)])
    counters = {'started': dict(H=0, CI=0, gradient=0), 'completed': dict(H=0, CI=0, gradient=0)}
    caps = dict(H=18 if kind == 'warm' else 2 if kind == 'ordinary' else 1,
                CI=12 if kind == 'warm' else 1, gradient=12 if kind == 'warm' else 1)
    binding = {'capture_calls': 0, 'validate_calls': 0, 'outer_seconds': 0., 'depth': 0}
    raw_build, raw_ci, raw_gradient = matrix_module.build_ci_matrix, functional_module.ci_solve, CSFOrbitalFunctional.gradient_from_weights
    raw_capture, raw_validate = CIReferenceBinding.__dict__['capture'], CIReferenceBinding.__dict__['validate']

    def counted(name, function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            if counters['started'][name] >= caps[name]:
                raise RuntimeError('Declared physical call budget exhausted: ' + name)
            counters['started'][name] += 1
            result = function(*args, **kwargs)
            counters['completed'][name] += 1
            return result
        return call

    def bound(name, function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            binding[name + '_calls'] += 1
            outer, clock = binding['depth'] == 0, time.perf_counter()
            binding['depth'] += 1
            try:
                return function(*args, **kwargs)
            finally:
                binding['depth'] -= 1
                if outer:
                    binding['outer_seconds'] += time.perf_counter() - clock
        return call

    matrix_module.build_ci_matrix = counted('H', raw_build)
    functional_module.ci_solve = counted('CI', raw_ci)
    CSFOrbitalFunctional.gradient_from_weights = counted('gradient', raw_gradient)
    CIReferenceBinding.capture = classmethod(bound('capture', raw_capture.__func__))
    CIReferenceBinding.validate = bound('validate', raw_validate)
    records, physics_passed, controlled = [], True, True
    gates = contract['acceptance']
    current_case = None
    qualified_rdm = load(ROOT / Path(contract['prerequisites']['qualification_result']['path']).parent /
                         'signed_public_RDM.json')

    def journal(phase):
        atomic_json(output_dir / 'science_progress.json', dict(phase=phase, current_case=current_case,
            counters=counters, records=records, binding=binding,
            qualification_result_sha256=approval['qualification_result_sha256'],
            source_map_sha256=contract['candidate_source_map_sha256']))

    def close(checks, name, value, cap):
        value = float(value)
        checks.append(dict(name=name, error=value, limit=cap, passed=math.isfinite(value) and value <= cap))

    def numerical_ci(checks, prefix, h, ci):
        shape_ok = (h.shape == (531, 531) and ci.state_energies.shape == (531,)
                    and ci.mixing_coefficients.shape == (531, 531)
                    and all(value.dtype == np.dtype('float64') and np.all(np.isfinite(value))
                            for value in (h, ci.state_energies, ci.mixing_coefficients)))
        checks.append(dict(name=prefix + '/complete finite float64', passed=bool(shape_ok)))
        require_checks(checks)
        close(checks, prefix + '/complete eigenpair residual', np.max(np.linalg.norm(
            h @ ci.mixing_coefficients - ci.mixing_coefficients * ci.state_energies, axis=0)), gates['CI_residual_Ha'])
        close(checks, prefix + '/complete CI Gram', np.max(abs(
            ci.mixing_coefficients.T @ ci.mixing_coefficients - np.eye(531))), gates['CI_orthogonality'])

    symmetry = [(csf.total_J2, csf.parity) for csf in point['csfs']]
    cross = np.array([[left != right for right in symmetry] for left in symmetry])

    try:
        for category, pair, mode in cases:
            current_case = dict(category=category, pair=pair, mode=mode)
            before_host = host(contract, cpus, cpu)
            # Resource failure at a case boundary stops further physics. Completed
            # earlier cases remain observations and cannot support controlled speedup.
            if not before_host['admitted']:
                controlled = False
                break
            before = dict(counters['completed'])
            binding_before = dict(binding)
            f, waves = point[mode], point['waves']
            case_start = time.perf_counter()
            identity_start = case_start
            require_checks(input_checks(contract))
            identity_seconds = time.perf_counter() - identity_start
            journal('before_energy_trial')
            clock = time.perf_counter()
            trial_energy, trial_ci = f.evaluate(waves, gradient=False)
            energy_seconds = time.perf_counter() - clock
            journal('after_energy_trial_before_accepted_gradient')
            clock = time.perf_counter()
            checks = []
            # The independently qualified full same-point H is a numerical anchor,
            # never injected into ci_solve or represented as a physical owner.
            numerical_ci(checks, 'trial before gradient', point['archive']['matrix_Ha'], trial_ci)
            require_checks(checks)
            before_gradient_validation_seconds = time.perf_counter() - clock
            journal('after_trial_eigenpair_gate_before_accepted_gradient')
            clock = time.perf_counter()
            energy, radial_gradient, ci = f.evaluate(waves, ci_result=trial_ci)
            gradient_seconds = time.perf_counter() - clock
            journal('after_full_signed_gradient_before_callback')
            clock = time.perf_counter()
            weights, identities = f.level_tracking.resolve(point['csfs'], ci, waves, f.metric)
            if mode == 'retained':
                h = ci.hamiltonian_for(point['csfs'], waves, f.keys, point['data'], 70, 123,
                    nuclear_potential=point['nuclear'], max_determinants=2000)
            else:
                if ci.hamiltonian_snapshot is not None:
                    raise ValueError('Ordinary result unexpectedly retained H')
                h = matrix_module.build_ci_matrix(point['csfs'], waves, f.keys, point['data'], 70, 123,
                    nuclear_potential=point['nuclear'], convol_cache=f.cache, max_determinants=2000)
            callback_seconds = time.perf_counter() - clock
            journal('after_callback_H_before_complete_validation')
            clock = time.perf_counter()
            numerical_ci(checks, 'actual callback H', h, ci)
            checks.append(dict(name='accepted same CI object', passed=ci is trial_ci))
            close(checks, 'H hermiticity', np.max(abs(h - h.T)), gates['matrix_hermiticity_Ha'])
            checks.append(dict(name='actual H bitwise equal full frozen320 H', passed=
                np.array_equal(h, point['archive']['matrix_Ha'])))
            close(checks, 'complete cross symmetry', np.max(abs(h[cross])), gates['cross_symmetry_zero_Ha'])
            close(checks, 'all roots replay', np.max(abs(ci.state_energies - point['archive']['energies_Ha'])), gates['all_roots_vs_frozen320_max_Ha'])
            close(checks, 'trial accepted EO', abs(energy - trial_energy), 0)
            checks.append(dict(name='complete original physical targets', passed=
                canon.canonical_physical_target_groups(identities) ==
                canon.canonical_physical_target_groups(point['checkpoint']['physical_tracking']) and
                np.count_nonzero(weights) == 5 and np.array_equal(weights[weights > 0], np.repeat(.2, 5))))
            checks.append(dict(name='complete public signed RDM equals independent live qualification', passed=
                plain([dict(key=key, value=float(value)) for key, value in f.last_weights.items()]) ==
                qualified_rdm['terms']))
            g, pg = point['original_gradients'](radial_gradient)
            close(checks, 'complete projected g', np.max(abs(g - point['archive']['original_projected_gradient'])), 1e-9)
            close(checks, 'complete original pg', np.max(abs(pg - point['archive']['original_preconditioned_gradient'])), 1e-10)
            residual, indicator = float(np.linalg.norm(pg)), .5 * abs(float(g @ pg))
            close(checks, 'original residual matches independent320', abs(residual -
                point['endpoint_proof']['actual_original_metric_orbital_residual']), 1e-10 + 1e-6 * residual)
            close(checks, 'original indicator matches independent320', abs(indicator -
                point['endpoint_proof']['actual_original_metric_energy_indicator_Ha']), 1e-12 + 1e-6 * indicator)
            checks.append(dict(name='original stationarity FAIL preserved', passed=
                residual > gates['original_orbital_tolerance'] and indicator < gates['original_energy_tolerance_Ha']
                and point['endpoint_proof']['original_stationarity_gates_passed'] is False))
            for i, row in enumerate(point['layout']):
                keys = [tuple(key) for key in row['keys']]
                c = np.column_stack([np.concatenate(waves[key]) for key in keys])
                s, projector = f.metric[keys[0][1]], point['metric'][f'projector_{i}']
                close(checks, f'radial Gram:{i}', np.max(abs(c.T @ s @ c - np.eye(len(keys)))), gates['radial_Gram_error'])
                outside = c - projector @ (projector.T @ s @ c)
                close(checks, f'original no-pair:{i}', np.sqrt(max(0., np.max(np.sum(outside * (s @ outside), axis=0)))), gates['projector_leakage_norm'])
            expected_calls = dict(H=1 if mode == 'retained' else 2, CI=1, gradient=1)
            actual_calls = {name: counters['completed'][name] - before[name] for name in before}
            checks.append(dict(name='exact same-chain physical calls', passed=actual_calls == expected_calls))
            if not all(check['passed'] for check in checks):
                physics_passed = False
            validation_seconds = time.perf_counter() - clock
            journal('after_complete_validation_before_archive')
            # Exactly the same complete numeric archive/schema/codec/fsync/readback
            # for both modes. Mixing readback is local to its own H; pair comparison
            # never requires saved column identity inside degeneracies.
            clock = time.perf_counter()
            archive = {name: point['archive'][name] for name in point['archive'] if
                       name in ('x', 'P', 'Q', 'orbital_keys', 'inverse_diagonal') or
                       name.startswith(('projector_', 'inverse_blocks_'))}
            keys = tuple(sorted(radial_gradient))
            archive.update(matrix_Ha=h, energies_Ha=ci.state_energies, mixing=ci.mixing_coefficients,
                original_projected_gradient=g, original_preconditioned_gradient=pg,
                radial_gradient=np.array([radial_gradient[key] for key in keys]),
                radial_gradient_keys=np.array(keys), state_weights=weights,
                signed_RDM_values=np.array(list(f.last_weights.values())))
            rdm_keys = [repr(key) for key in f.last_weights]
            stem = f'{category}_{pair:02d}_{mode}'
            output = output_dir / (stem + '.npz')
            temporary = output.with_suffix('.npz.part')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **archive)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(output)
            with np.load(output, allow_pickle=False) as replay:
                readback = set(replay.files) == set(archive) and all(np.array_equal(replay[name], value) for name, value in archive.items())
            checks.append(dict(name='complete own-archive readback', passed=readback))
            matrix_bytes_sha = hashlib.sha256(repr((h.shape, h.dtype.str)).encode() + h.tobytes(order='C')).hexdigest()
            manifest = dict(mode=mode, category=category, pair=pair, arrays_sha256=sha(output),
                complete_array_keys=list(archive), signed_RDM_keys=rdm_keys, physical_tracking=identities,
                source_map_sha256=contract['candidate_source_map_sha256'],
                original_metric_sha256=point['metric_sha256'], epoch_identity=point['epoch_identity'],
                owner_input_sha256=ci.hamiltonian_snapshot.input_sha256 if mode == 'retained' else None,
                owner_matrix_sha256=matrix_bytes_sha, qualification_result_sha256=approval['qualification_result_sha256'],
                original_residual=residual, original_indicator_Ha=indicator,
                original_stationarity_passed=False, checks=checks)
            atomic_json(output.with_suffix('.json'), manifest)
            archive_seconds = time.perf_counter() - clock
            chain_seconds = time.perf_counter() - case_start
            after_host = host(contract, cpus, cpu)
            controlled &= before_host['admitted'] and after_host['admitted']
            record = dict(mode=mode, category=category, pair=pair, path=str(output.relative_to(HERE)),
                manifest_sha256=sha(output.with_suffix('.json')), arrays_sha256=sha(output),
                whole_chain_seconds=chain_seconds, identity_seconds=identity_seconds,
                energy_trial_seconds=energy_seconds, accepted_gradient_seconds=gradient_seconds,
                before_gradient_full_eigenpair_validation_seconds=before_gradient_validation_seconds,
                callback_tracking_and_H_seconds=callback_seconds, validation_seconds=validation_seconds,
                full_archive_write_fsync_readback_seconds=archive_seconds, actual_calls=actual_calls,
                binding_capture_calls=binding['capture_calls'] - binding_before['capture_calls'],
                binding_validate_calls=binding['validate_calls'] - binding_before['validate_calls'],
                binding_seconds_subscope=binding['outer_seconds'] - binding_before['outer_seconds'],
                binding_subscope_overlaps_stages=True, checks_passed=all(row['passed'] for row in checks),
                before_host=before_host, after_host=after_host)
            records.append(record)
            atomic_json(output_dir / 'progress.json', dict(records=records, counters=counters,
                setup_seconds=setup_seconds, physics_passed=physics_passed, controlled=controlled))
            journal('complete_case_archive')
            if not record['checks_passed'] or not after_host['admitted']:
                physics_passed &= record['checks_passed']
                break
        result = dict(records=records, counters=counters, setup_seconds=setup_seconds,
            setup_functional_construction=setup_construction,
            worker_seconds=time.perf_counter() - started, peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            physics_passed=physics_passed and all(row['checks_passed'] for row in records),
            complete_declared_cases=len(records) == len(cases), controlled=controlled,
            candidate_source_map_sha256=contract['candidate_source_map_sha256'],
            qualification_result_sha256=approval['qualification_result_sha256'], optimizer_calls=0)
        result['actual_runtime_caps_passed'] = (sorted(os.sched_getaffinity(0)) == [cpu]
            and result['peak_RSS_MiB'] < contract['resources']['address_space_cap_GiB'] * 1024
            and all(os.environ[name] == '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')))
        atomic_json(output_dir / 'result.json', result)
    except BaseException as error:
        atomic_json(output_dir / 'failure.json', dict(exception=type(error).__name__, message=str(error),
            current_case=current_case, counters=counters, binding=binding, records=records,
            source_map_sha256=contract['candidate_source_map_sha256'], optimizer_calls=0))
        raise
    finally:
        matrix_module.build_ci_matrix, functional_module.ci_solve = raw_build, raw_ci
        CSFOrbitalFunctional.gradient_from_weights = raw_gradient
        CIReferenceBinding.capture, CIReferenceBinding.validate = raw_capture, raw_validate


def panel(execution_path):
    started = time.perf_counter()
    contract, approval = load(HERE / 'contract.json'), load(execution_path)
    execution_admission(contract, approval)
    require_checks(input_checks(contract))
    if (approval['chosen_cpu'] not in os.sched_getaffinity(0)
            or approval['available_cpus'] != sorted(os.sched_getaffinity(0))):
        raise ValueError('Root available CPU identity differs from actual panel affinity')
    os.sched_setaffinity(0, {approval['chosen_cpu']})
    limit = int(contract['resources']['address_space_cap_GiB'] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    if (HERE / 'actual').exists():
        raise FileExistsError('Do not replace a previous actual panel; prepare a successor')
    actual = HERE / 'actual'
    actual.mkdir()
    children, failure = [], None
    for kind in ('ordinary', 'retained', 'warm'):
        output = actual / kind
        output.mkdir()
        env = dict(os.environ)
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
            env[name] = '1'
        cache = output / 'numba_cache'
        cache.mkdir()
        env['NUMBA_CACHE_DIR'] = str(cache)
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--kind', kind,
                   '--execution-contract', str(execution_path.resolve()), '--output-dir', str(output)]
        before = host(contract, approval['available_cpus'], approval['chosen_cpu'])
        if not before['admitted']:
            failure = 'host_admission_failed_before_worker'
            break
        remaining = contract['resources']['whole_panel_wall_cap_s'] - (time.perf_counter() - started)
        if remaining <= 0:
            failure = 'whole_panel_wall_cap'
            break
        clock = time.perf_counter()
        with (output / 'execution.log').open('w') as stream:
            try:
                child = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=remaining, check=False)
                code = child.returncode
            except subprocess.TimeoutExpired:
                code, failure = None, 'whole_panel_wall_cap'
        elapsed = time.perf_counter() - clock
        row = dict(kind=kind, command=command, process_cold_elapsed_seconds=elapsed if kind != 'warm' else None,
                   observed_worker_process_seconds=elapsed, exit_code=code, before_host=before,
                   after_host=host(contract, approval['available_cpus'], approval['chosen_cpu']))
        children.append(row)
        atomic_json(actual / 'execution_progress.json', dict(children=children, failure=failure))
        if code != 0:
            failure = failure or 'worker_failed'
            break
    summarize(contract, approval, actual, children, failure, time.perf_counter() - started, execution_path)


def summarize(contract, approval, actual, children, failure, elapsed, execution_path):
    import numpy as np

    workers = {row['kind']: load(actual / row['kind'] / 'result.json') for row in children
               if (actual / row['kind'] / 'result.json').exists()}
    checks = []
    warm = workers.get('warm', {})
    pairs = []
    for pair in range(1, 6):
        rows = {r['mode']: r for r in warm.get('records', []) if r['category'] == 'measured' and r['pair'] == pair}
        if set(rows) != {'ordinary', 'retained'}:
            continue
        paths = {mode: HERE / row['path'] for mode, row in rows.items()}
        headers = {mode: load(path.with_suffix('.json')) for mode, path in paths.items()}
        with np.load(paths['ordinary'], allow_pickle=False) as a, np.load(paths['retained'], allow_pickle=False) as b:
            # Compare full physical arrays/weights/densities/gradients; mixing
            # already has its own independent full H residual and Gram gates.
            same = set(a.files) == set(b.files) and all(np.array_equal(a[name], b[name]) for name in a.files if name != 'mixing')
        checks.append(dict(name=f'pair{pair} complete same physics', passed=same and
                           headers['ordinary']['signed_RDM_keys'] == headers['retained']['signed_RDM_keys']))
        pairs.append(dict(pair=pair, ordinary_seconds=rows['ordinary']['whole_chain_seconds'],
                          retained_seconds=rows['retained']['whole_chain_seconds'],
                          ordinary_over_retained=rows['ordinary']['whole_chain_seconds'] / rows['retained']['whole_chain_seconds']))
    sums = {name: sum(w['counters']['completed'][name] for w in workers.values()) for name in ('H', 'CI', 'gradient')}
    checks.append(dict(name='all14 declared chains exact budgets', passed=sums == dict(H=21, CI=14, gradient=14)))
    checks.append(dict(name='all worker runtime caps', passed=all(
        w.get('actual_runtime_caps_passed') is True for w in workers.values()) and
        elapsed < contract['resources']['whole_panel_wall_cap_s']))
    numeric = failure is None and len(workers) == 3 and len(pairs) == 5 and all(
        w['physics_passed'] and w['complete_declared_cases'] for w in workers.values()) and all(row['passed'] for row in checks)
    controlled = numeric and all(w['controlled'] for w in workers.values()) and all(
        row['before_host']['admitted'] and row['after_host']['admitted'] for row in children)
    ratios = [row['ordinary_over_retained'] for row in pairs]
    median = statistics.median(ratios) if ratios else None
    speedup = controlled and all(value > 1.0 for value in ratios) and median >= contract['performance_decision']['minimum_median_ratio']
    result = dict(scope=contract['scope'], candidate_source_map_sha256=contract['candidate_source_map_sha256'],
        contract_sha256=sha(HERE / 'contract.json'), execution_contract_sha256=sha(execution_path),
        prerequisite_qualification_sha256=approval['qualification_result_sha256'],
        actual_calls=sums, children=children, warm_pairs=pairs, median_warm_ratio=median,
        warm_ratio_range=[min(ratios), max(ratios)] if ratios else None, checks=checks,
        scientific_parity_passed=numeric, controlled_admission_passed=controlled,
        controlled_warm_chain_speedup_supported=speedup, stopping_condition=failure,
        observed_panel_seconds=elapsed, optimizer_calls=0, whole_optimizer_speedup_claim=False,
        GPU_speedup_claim=False, original_stationarity_passed=False,
        cold_scope='one fresh process per mode with its own initially empty Numba cache; OS filesystem caches not flushed; cold observations only')
    atomic_json(actual / 'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('children', 'checks')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'panel', 'worker'))
    parser.add_argument('--execution-contract', type=Path)
    parser.add_argument('--kind', choices=('ordinary', 'retained', 'warm'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.execution_contract is None:
        raise ValueError('Scientific runtime requires a root execution contract')
    elif args.command == 'worker':
        if args.kind is None or args.output_dir is None:
            raise ValueError('Worker requires the declared mode/output')
        worker(args.kind, args.execution_contract, args.output_dir)
    else:
        panel(args.execution_contract)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        # Preserve previous completed case archives/progress on any hard stop.
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        raise
