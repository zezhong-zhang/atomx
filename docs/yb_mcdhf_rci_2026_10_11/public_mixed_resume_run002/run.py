"""Prepare or run one explicitly authorized public mixed-CI resume epoch.

The default command only reads archives and reports resources. Execution is
separately bound to a root-owned contract and frozen public source snapshot.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import resource
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = 'docs/yb_mcdhf_rci_2026_10_11/'
START = time.monotonic()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def plain(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    with temporary.open('w') as stream:
        stream.write(json.dumps(plain(value), indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def host_resources(caps, *, accounting_cpus=None):
    mem = {k: int(v.split()[0]) for k, v in
           (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    affinity = sorted(os.sched_getaffinity(0))
    cpus = len(affinity)
    quota_file = Path('/sys/fs/cgroup/cpu.max')
    quota = quota_file.read_text().split() if quota_file.exists() else None
    if quota and quota[0] != 'max':
        cpus = min(cpus, int(quota[0]) / int(quota[1]))
    if accounting_cpus is not None:
        # Keep launch host accounting after this job pins itself to one CPU.
        cpus = accounting_cpus
    available = mem['MemAvailable'] / 1024**2
    memory_max = Path('/sys/fs/cgroup/memory.max')
    memory_current = Path('/sys/fs/cgroup/memory.current')
    if memory_max.exists() and memory_current.exists() and memory_max.read_text().strip() != 'max':
        available = min(available, (int(memory_max.read_text()) - int(memory_current.read_text())) / 1024**3)
    load_average = os.getloadavg()
    ratio = load_average[0] / cpus
    return {'observed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'load_average': load_average, 'available_CPUs': cpus, 'affinity': affinity,
            'load1_per_available_CPU': ratio, 'host_available_GiB': available,
            'admitted': ratio <= caps['max_load1_per_available_CPU']
            and available >= caps['minimum_host_available_GiB'],
            'purpose': 'Resource feasibility; never a controlled timing certificate'}


def source_map(directory):
    return {str(p.relative_to(directory.parent)): sha(p)
            for p in sorted(directory.rglob('*.py'))}


def interface_status(path):
    tree = ast.parse(path.read_text())
    functions = {x.name: x for x in tree.body if isinstance(x, ast.FunctionDef)}
    optimizer = functions['optimize_orbitals']
    keywords = {x.arg for x in optimizer.args.kwonlyargs}
    required = {'resume_state', 'objective_identity', 'allow_objective_change',
                'checkpoint_callback', 'max_wall_time', 'return_state'}
    state_present = any(isinstance(x, ast.ClassDef) and x.name == 'OrbitalOptimizationState' for x in tree.body)
    return {'typed_state_declared': state_present, 'required_keywords': sorted(required),
            'missing_keywords': sorted(required - keywords), 'available': state_present and not (required - keywords)}


def matched_reference_status(contract, *, require_frozen=False):
    declaration = contract['matched_recovered_point_reference']
    directory = ROOT / declaration['directory']
    ready = declaration['status'] == 'frozen_verified_same_recovered_point'
    declared = declaration.get('inputs_sha256', {})
    expected_files = {str(directory.relative_to(ROOT) / name) for name in declaration['files']}
    if ready and (set(declared) != expected_files or any(contract['inputs_sha256'].get(path) != expected
            or sha(ROOT / path) != expected for path, expected in declared.items())):
        raise ValueError('Same recovered-point reference is not exactly input/source-bound')
    if require_frozen and not ready:
        raise ValueError('Matched original005-source recovered-point reference pending; no solve or seal allowed')
    return {'ready': ready, 'status': declaration['status'], 'directory': declaration['directory'],
            'frozen_reference_files_verified': len(declared) if ready else 0}


def prospective_preflight(contract):
    for path, expected in contract['inputs_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Frozen scientific input changed: ' + path)
    caps = contract['resources']
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    # Read-only enumeration uses original source; no functional/CI is created.
    frozen = ROOT / contract['fixed_conditions']['frozen_source_root']
    original_source = load(ROOT / (PREFIX + 'native_run/nuclear_eol005_projected_increment/source_identity.json'))
    for name, expected in original_source['all_atomx_python_sha256'].items():
        if sha(frozen / Path(name).relative_to('src')) != expected:
            raise ValueError('Frozen old353 reference source changed: ' + name)
    sys.path.insert(0, str(frozen))
    import numpy as np

    from atomx.angular.csf_hamiltonian import csf_signature, pattern_m_counts
    from atomx.config.csf import enumerate_jj_csfs

    enumeration = load(ROOT / (PREFIX + 'balanced_cv_replay/basis_identity.json'))
    csfs = [c for row in enumeration['configs'] for c in enumerate_jj_csfs(row['config'])]
    if len(csfs) != 531 or {c.n_electrons for c in csfs} != {69}:
        raise ValueError('Complete declared physical model changed')
    blocks = {}
    for index, c in enumerate(csfs):
        saved, sig = enumeration['CSFs'][index], csf_signature(c)
        if (saved['twoJ'] != c.total_J2 or saved['parity'] != c.parity
                or saved['subshells'] != plain(sig.subshells) or saved['coupling'] != plain(sig.coupling)):
            raise ValueError('Signed CSF order changed')
        blocks.setdefault((c.total_J2, c.parity), []).append(c)
    targets = load(ROOT / (PREFIX + 'balanced_cv_replay/stage_4f5p5s/target_binding.json'))['targets']
    selected = {(t['twoJ'], t['parity']) for t in targets}
    table = []
    for (two_j, parity), members in sorted(blocks.items()):
        patterns = {csf_signature(c).pattern for c in members}
        raw = sum(dict(pattern_m_counts(p)).get(two_j, 0) for p in patterns)
        table.append({'twoJ': two_j, 'parity': parity, 'CSFs': len(members), 'raw_M_equals_J': raw})
    overlap_pairs = sum(r['raw_M_equals_J']**2 for r in table if (r['twoJ'], r['parity']) in selected)
    if max(r['raw_M_equals_J'] for r in table) > caps['max_raw_determinants_per_J_parity']:
        raise MemoryError('Raw determinant cap exceeded; do not shrink model')
    if overlap_pairs > caps['max_determinant_overlap_pairs']:
        raise MemoryError('Physical overlap work cap exceeded; do not shrink model')
    recovered = np.load(ROOT / (PREFIX + 'native_run/nuclear_eol005_projected_increment/stationarity_replay002/recovered_coordinates.npz'))
    if recovered['x'].shape != (2856,) or recovered['x'].dtype != np.float64 or not np.isfinite(recovered['x']).all():
        raise ValueError('Original005 recovered coordinate identity failed')
    old_metric = np.load(ROOT / (PREFIX + 'native_run/nuclear_eol004_stable_delta_trace/failed_coordinates.npz'))
    layout = load(ROOT / (PREFIX + 'native_run/nuclear_eol004_stable_delta_trace/failed_coordinate_layout.json'))
    metric_digest = hashlib.sha256()
    for i, row in enumerate(layout):
        metric_digest.update(repr(([tuple(k) for k in row['keys']], tuple(row['shape']))).encode())
        metric_digest.update(np.ascontiguousarray(old_metric[f'projector_{i}']).tobytes())
    metric_digest.update(old_metric['inverse_diagonal'].tobytes())
    for i in range(len(layout)):
        for block in old_metric[f'inverse_blocks_{i}']:
            metric_digest.update(np.ascontiguousarray(block).tobytes())
    if metric_digest.hexdigest() != contract['fixed_conditions']['original_projector_and_preconditioner_metric_sha256']:
        raise ValueError('Original complete metric changed')
    current_sources = source_map(ROOT / 'src/atomx')
    actual_runtime_source_map_sha256 = digest({'src/' + name: value for name, value in current_sources.items()})
    if actual_runtime_source_map_sha256 != contract['scientific_source_map_required_sha256']:
        raise ValueError('Prepared successor must retain the exact scientific source of the failed launch')
    answer = {'preparation_only': True, 'functional_evaluations': 0, 'CI_solves': 0, 'optimizer_calls': 0,
              'matched_recovered_point_reference': matched_reference_status(contract),
              'contract_sha256': sha(HERE / 'contract.json'), 'CSFs': 531, 'radial_orbitals': 24,
              'blocks': table, 'max_raw_M_equals_J': max(r['raw_M_equals_J'] for r in table),
              'selected_overlap_pair_work_upper': overlap_pairs, 'coordinates_shape': [2856],
              'original_metric_sha256': metric_digest.hexdigest(), 'resources': host_resources(caps),
              'public_interface': interface_status(ROOT / 'src/atomx/numerics/bspline/orbital_optimization.py'),
              'planning_current_source_map_sha256': digest(current_sources),
              'planning_runtime_source_map_sha256': actual_runtime_source_map_sha256,
              'planning_current_sources_sha256': current_sources,
              'execution_source_epoch_frozen': False, 'input_hashes_verified': len(contract['inputs_sha256'])}
    answer['old353_reference_source_files_verified'] = len(original_source['all_atomx_python_sha256'])
    atomic_json(HERE / 'preflight.json', answer)
    return answer


def seal_source():
    if (HERE / 'source_identity.json').exists() or (HERE / 'source').exists():
        raise FileExistsError('Never replace an existing source epoch')
    contract = load(HERE / 'contract.json')
    matched_reference_status(contract, require_frozen=True)
    status = interface_status(ROOT / 'src/atomx/numerics/bspline/orbital_optimization.py')
    if not status['available']:
        raise RuntimeError('Public resume API not available; no source epoch created')
    original = load(ROOT / (PREFIX + 'native_run/nuclear_eol005_projected_increment/source_identity.json'))['all_atomx_python_sha256']
    current = {'src/' + name: value for name, value in source_map(ROOT / 'src/atomx').items()}
    if digest(current) != contract['scientific_source_map_required_sha256']:
        raise ValueError('Gradient point-identity correction must not change scientific source')
    shutil.copytree(ROOT / 'src/atomx', HERE / 'source/atomx', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbc', '*.nbi'))
    copied = {'src/' + name: value for name, value in source_map(HERE / 'source/atomx').items()}
    if copied != current:
        raise ValueError('Source changed during epoch copy; stop without solving')
    identity = {'source_map_sha256': digest(current), 'all_atomx_python_sha256': current,
                'old353_source_map_sha256': '9dff33cc24acecff9e821c4f9b836e02efca0d615291d6f92ff5a7014a726c3f',
                'changed_from_old353': [k for k in current if k in original and current[k] != original[k]],
                'added_since_old353': sorted(current.keys() - original.keys()),
                'deleted_since_old353': sorted(original.keys() - current.keys()),
                'public_interface': status, 'driver_sha256': sha(Path(__file__)),
                'preparation_contract_sha256': sha(HERE / 'contract.json'), 'solve_started': False}
    atomic_json(HERE / 'source_identity.json', identity)
    return identity


class BudgetStop(RuntimeError):
    """A resource stop preserves the most recent fully accepted checkpoint."""


def execute(execution_contract):
    approval = load(execution_contract)
    contract = load(HERE / 'contract.json')
    matched_reference_status(contract, require_frozen=True)
    source = load(HERE / 'source_identity.json')
    if (approval.get('execution_authorized_by_root') is not True
            or approval.get('preparation_contract_sha256') != sha(HERE / 'contract.json')
            or approval.get('driver_sha256') != sha(Path(__file__))
            or approval.get('source_identity_sha256') != sha(HERE / 'source_identity.json')):
        raise ValueError('Missing exact root authorization/new-source contract; no solver started')
    for relative, expected in contract['inputs_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen scientific input changed before execution')
    for path, expected in source['all_atomx_python_sha256'].items():
        if sha(HERE / 'source' / Path(path).relative_to('src')) != expected:
            raise ValueError('Frozen execution source changed')
    caps, gates = contract['resources'], contract['acceptance']
    admission = host_resources(caps)
    atomic_json(HERE / 'execution_resource_admission.json', admission)
    if not admission['admitted']:
        raise BudgetStop('Launch host resource guard failed; no solve started')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    os.sched_setaffinity(0, {admission['affinity'][-1]})
    byte_cap = int(caps['address_space_cap_GiB'] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (byte_cap, byte_cap))

    def interrupted(*_):
        raise BudgetStop('Hard890s wall limit; latest accepted checkpoint remains authoritative')

    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(caps['wall_cap_s'])
    sys.path.insert(0, str(HERE / 'source'))
    import numpy as np

    from atomx.angular.csf_hamiltonian import csf_signature
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.ci import CIResult
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.numerics.bspline.orbital_optimization import (
        OrbitalOptimizationState,
        optimize_orbitals,
    )
    from atomx.physics.csf.matrix import build_ci_matrix
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.physics.csf.tracking import CIBasisTracking
    from atomx.potentials.nuclear import make_nuclear_potential

    native = ROOT / (PREFIX + 'native_run/nuclear_eol005_projected_increment')
    replay = ROOT / (PREFIX + 'balanced_cv_replay')
    original = ROOT / (PREFIX + 'native_run/nuclear_eol004_stable_delta_trace')
    enumeration = load(replay / 'basis_identity.json')
    csfs = [c for row in enumeration['configs'] for c in enumerate_jj_csfs(row['config'])]
    if len(csfs) != 531 or {c.n_electrons for c in csfs} != {69}:
        raise ValueError('New-source complete physical basis changed')
    for csf, record in zip(csfs, enumeration['CSFs'], strict=True):
        signature = csf_signature(csf)
        if (signature.total_J2 != record['twoJ'] or csf.parity != record['parity']
                or plain(signature.subshells) != record['subshells']
                or plain(signature.coupling) != record['coupling']):
            raise ValueError('New-source signed CSF order changed')
    bare = [c for row in enumeration['configs'] if row['family'] == 'bare' for c in enumerate_jj_csfs(row['config'])]
    targets = load(replay / 'stage_4f5p5s/target_binding.json')['targets']
    selectors = tuple((t['twoJ'], t['parity'], t['models']['DC']['local_root'], 1.) for t in targets)
    average = CIStateAverage(selectors)
    radial = np.load(native / 'orbitals_workspace.npz')
    physical_reference_waves = {tuple(k): (p.copy(), q.copy()) for k, p, q in
                                zip(radial['orbital_keys'], radial['P'], radial['Q'], strict=True)}
    data = {n[6:]: radial[n].item() if radial[n].ndim == 0 else radial[n]
            for n in radial.files if n.startswith('basis_')}
    metric = np.load(original / 'failed_coordinates.npz')
    recovered = np.load(native / 'stationarity_replay002/recovered_coordinates.npz')
    reference_directory = ROOT / contract['matched_recovered_point_reference']['directory']
    reference_result = load(reference_directory / 'result.json')
    reference_verification = load(reference_directory / 'verification.json')
    reference = np.load(reference_directory / 'arrays.npz')
    if (reference_verification.get('all_passed') is not True
            or reference_verification['strict_g_gate'] != gates['bare_same_recovered_point_projected_gradient_max']
            or reference_verification['strict_pg_gate'] != gates['bare_same_recovered_point_preconditioned_gradient_max']
            or reference_verification['result_sha256'] != sha(reference_directory / 'result.json')
            or reference_verification['contract_sha256'] != sha(reference_directory / 'contract.json')
            or reference_verification['arrays_sha256'] != sha(reference_directory / 'arrays.npz')
            or reference_result['original005_source_map_sha256'] != contract['fixed_conditions']['source_map_sha256']
            or reference_result['recovered_coordinate_archive_sha256'] != sha(native / 'stationarity_replay002/recovered_coordinates.npz')
            or reference_result['original_layout_sha256'] != sha(original / 'failed_coordinate_layout.json')
            or reference_result['point_identity'] != 'direct_original_projector_times_archived_recovered_x'
            or reference_result['arrays_sha256'] != sha(reference_directory / 'arrays.npz')):
        raise ValueError('Matched reference source/point/proof identity failed before physical evaluation')
    reference_model = reference_result['physical_model']
    if (reference_model['Z'] != 70 or reference_model['electrons'] != 69
            or reference_model['bare_CSFS'] != 5 or reference_model['ns'] != 123
            or reference_model['radial_orbitals'] != 24 or reference_model['nucleus'] != 'Fermi A175'
            or reference_model['operator'] != 'bare5 no-pair DC'
            or set(tuple(row) for row in reference_model['selectors']) != set(selectors)
            or reference_model['state_weights'] != [.2] * 5):
        raise ValueError('Matched bare reference changed Hamiltonian/nucleus/targets/weights')
    rows = load(original / 'failed_coordinate_layout.json')
    layout = tuple((tuple(tuple(k) for k in row['keys']), metric[f'projector_{i}'].copy(),
                    tuple(row['shape']), slice(*row['slice'])) for i, row in enumerate(rows))
    inverse_blocks = tuple(tuple(b.copy() for b in metric[f'inverse_blocks_{i}']) for i in range(len(rows)))
    nuclear = make_nuclear_potential(70, 'fermi', A=175)
    raw_functional = CSFOrbitalFunctional(csfs, data, 70, nuclear, state_average=average, max_determinants=2000)
    bare_functional = CSFOrbitalFunctional(bare, data, 70, nuclear, state_average=average,
                                         cache=raw_functional.cache, max_determinants=2000)
    target_identity = [{'label': t['label'], 'twoJ': t['twoJ'], 'parity': t['parity'],
                        'root': t['models']['DC']['local_root'], 'weight': .2} for t in targets]
    common = {'Z': 70, 'electrons': 69, 'operator': 'full declared no-pair Dirac-Coulomb',
              'nucleus': 'native005 Fermi A175', 'radial_archive_sha256': sha(native / 'orbitals_workspace.npz'),
              'target_identity': target_identity}
    old_payload = {**common, 'CSFs': [c for c in enumeration['CSFs'] if c['family'] == 'bare'],
                   'original_source_map_sha256': source['old353_source_map_sha256']}
    new_payload = {**common, 'CSFs': enumeration['CSFs'], 'source_map_sha256': source['source_map_sha256'],
                   'physical_reference_CI_sha256': sha(replay / 'stage_4f5p5s/DC_arrays.npz'),
                   'physical_reference_binding_sha256': sha(replay / 'stage_4f5p5s/target_binding.json')}
    old_identity, new_identity = digest(old_payload), digest(new_payload)
    atomic_json(HERE / 'objective_identity.json', {'old': old_payload, 'new': new_payload,
                                                  'old_sha256': old_identity, 'new_sha256': new_identity})
    initial = OrbitalOptimizationState.from_arrays(
        raw_functional, layout=layout, coordinates=recovered['x'].copy(),
        inverse_diagonal=metric['inverse_diagonal'].copy(), inverse_blocks=inverse_blocks,
        projector_source='nuclear', objective_identity=old_identity, occupation_floor=.05)
    expected_metric = contract['fixed_conditions']['original_projector_and_preconditioner_metric_sha256']
    if initial.residual_metric_sha256 != expected_metric:
        raise ValueError('Public typed state rebuilt or changed original metric')
    if (reference_result['original_metric_sha256'] != expected_metric
            or reference['x'].dtype != np.float64 or not np.array_equal(reference['x'], recovered['x'])
            or reference['g'].shape != (2856,) or reference['g'].dtype != np.float64
            or reference['preconditioned_gradient'].shape != (2856,)
            or reference['preconditioned_gradient'].dtype != np.float64
            or not np.isfinite(reference['g']).all() or not np.isfinite(reference['preconditioned_gradient']).all()):
        raise ValueError('Matched reference original metric/coordinate/gradient arrays changed')
    orbitals = {}
    for keys, projector, shape, where in initial.layout:
        c = projector @ initial.coordinates[where].reshape(shape)
        for col, key in enumerate(keys):
            orbitals[key] = (c[:123, col].copy(), c[123:, col].copy())
    reference_waves = {tuple(key): (p, q) for key, p, q in
                       zip(reference['orbital_keys'], reference['P'], reference['Q'], strict=True)}
    if (reference['orbital_keys'].shape != (24, 2)
            or reference['P'].shape != (24, 123) or reference['Q'].shape != (24, 123)
            or reference['P'].dtype != np.float64 or reference['Q'].dtype != np.float64
            or set(reference_waves) != set(orbitals) or len(reference_waves) != 24
            or any(not np.array_equal(a, b) for key in orbitals
                   for a, b in zip(orbitals[key], reference_waves[key], strict=True))):
        raise ValueError('Reference gradients were not evaluated on exact same direct original-P*x waves')
    saved = np.load(replay / 'stage_4f5p5s/DC_arrays.npz')
    tracking = CIBasisTracking(average, csfs, CIResult(saved['energies_Ha'], saved['mixing']), physical_reference_waves,
                              raw_functional.metric, max_determinant_pairs=1_000_000)
    raw_functional = CSFOrbitalFunctional(csfs, data, 70, nuclear, state_average=average,
        level_tracking=tracking, cache=raw_functional.cache, max_determinants=2000)

    def original_metrics(gradient):
        g = np.zeros(2856)
        for keys, projector, shape, where in initial.layout:
            y = initial.coordinates[where].reshape(shape)
            block = projector.T @ np.column_stack([gradient[k] for k in keys])
            multiplier = y.T @ block
            g[where] = (block - y @ (.5 * (multiplier + multiplier.T))).ravel()
        preconditioned = np.zeros_like(g)
        for (_, _, shape, where), blocks in zip(initial.layout, initial.inverse_blocks, strict=True):
            y, b = initial.coordinates[where].reshape(shape), g[where].reshape(shape)
            z = np.column_stack([p @ b[:, col] for col, p in enumerate(blocks)])
            multiplier = y.T @ z
            preconditioned[where] = (z - y @ (.5 * (multiplier + multiplier.T))).ravel()
        return float(np.linalg.norm(preconditioned)), .5 * abs(float(g @ preconditioned)), g, preconditioned

    # Independent original-geometry replay is mandatory before the new call.
    baseline_gradient_calls = 0
    baseline_gradient_calls += 1
    bare_energy, bare_gradient, _ = bare_functional.evaluate(orbitals)
    bare_residual, bare_indicator, bare_g, bare_preconditioned = original_metrics(bare_gradient)
    if not (bare_residual < 1e-7 and bare_indicator < 1e-8):
        raise ValueError('New source does not retain original005 stationarity in original metric')
    baseline_gradient_calls += 1
    energy, gradient, ci = raw_functional.evaluate(orbitals)
    h = build_ci_matrix(csfs, orbitals, raw_functional.keys, data, 70, 123, nuclear_potential=nuclear,
                        convol_cache=raw_functional.cache, max_determinants=2000)
    residual, indicator, mixed_g, mixed_preconditioned = original_metrics(gradient)
    del mixed_g, mixed_preconditioned
    old_probe = load(ROOT / (PREFIX + 'mixed_ci_stationarity_probe/result.json'))['baseline']
    controls = {'baseline_gradient_calls': baseline_gradient_calls,
                'total_run_full_gradient_calls_before_optimizer': baseline_gradient_calls,
                'planned_max_total_run_full_gradient_calls': caps['max_total_run_full_gradient_calls'],
                'bare_EO_energy_Ha': bare_energy, 'bare_original_residual': bare_residual,
                'bare_original_energy_indicator_Ha': bare_indicator, 'mixed_EO_energy_Ha': energy,
                'mixed_original_residual': residual, 'mixed_original_energy_indicator_Ha': indicator,
                'matrix_replay_error_Ha': float(np.max(abs(h - saved['matrix_Ha']))),
                'all_CI_energy_replay_error_Ha': float(np.max(abs(ci.state_energies - saved['energies_Ha']))),
                'mixed_residual_replay_error': abs(residual - old_probe['original_metric_orbital_residual']),
                'original_x_minus_saved_physical_wave_max': max(float(np.max(abs(a - b)))
                    for key in orbitals for a, b in zip(orbitals[key], physical_reference_waves[key], strict=True)),
                'original_x_minus_old_wide_probe_EO_Ha': energy - old_probe['EOL_objective_Ha'],
                'old_wide_probe_residual': old_probe['original_metric_orbital_residual'],
                'old_probe_preserved_unchanged': True,
                'matched_reference_result_sha256': sha(reference_directory / 'result.json'),
                'matched_reference_arrays_sha256': sha(reference_directory / 'arrays.npz'),
                'same_recovered_x_and_direct_unpacked_waves_exact': True,
                'bare_same_recovered_point_gradient_replay_max': float(np.max(abs(bare_g - reference['g']))),
                'bare_same_recovered_point_preconditioned_gradient_replay_max': float(np.max(abs(bare_preconditioned
                    - reference['preconditioned_gradient']))),
                'different_point_original_physical_wave_gradient_difference_max': float(np.max(abs(bare_g - recovered['g']))),
                'different_point_original_physical_wave_preconditioned_gradient_difference_max': float(np.max(abs(
                    bare_preconditioned - recovered['preconditioned_gradient']))),
                'predecessor_failed_different_point_assertion_preserved': contract['predecessor_failure'],
                'metric_sha256': expected_metric}
    atomic_json(HERE / 'new_source_baseline_controls.json', controls)
    if (controls['matrix_replay_error_Ha'] > gates['baseline_matrix_replay_Ha']
            or controls['all_CI_energy_replay_error_Ha'] > gates['baseline_target_energy_replay_Ha']
            or controls['bare_same_recovered_point_gradient_replay_max']
                > gates['bare_same_recovered_point_projected_gradient_max']
            or controls['bare_same_recovered_point_preconditioned_gradient_replay_max']
                > gates['bare_same_recovered_point_preconditioned_gradient_max']
            or controls['mixed_residual_replay_error'] > 1e-8 + 1e-6 * residual):
        raise ValueError('New-source matched baseline failed; no optimization started')

    class BudgetedFunctional:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.gradient_calls = 0
            self.energy_calls = 0

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def evaluate(self, waves, **kwargs):
            if time.monotonic() - START > caps['accepted_boundary_soft_stop_s']:
                raise BudgetStop('Evaluation-boundary soft wall limit')
            if kwargs.get('gradient', True):
                if self.gradient_calls >= caps['max_optimizer_full_gradient_calls']:
                    raise BudgetStop('Optimizer full-gradient call budget reached; two baseline calls separate')
                self.gradient_calls += 1
            else:
                self.energy_calls += 1
                if self.energy_calls > caps['max_energy_trials']:
                    raise BudgetStop('Energy trial budget reached')
            return self.wrapped.evaluate(waves, **kwargs)

    functional = BudgetedFunctional(raw_functional)
    checkpoints = []

    def checkpoint(iteration, state, waves, current_ci, diagnostics):
        if state.residual_metric_sha256 != expected_metric or state.objective_identity != new_identity:
            raise ValueError('Accepted checkpoint changed metric/objective')
        weights, identities = tracking.resolve(csfs, current_ci, waves, functional.metric)
        if np.count_nonzero(weights) != 5 or not np.array_equal(weights[weights > 0], np.repeat(.2, 5)):
            raise ValueError('Accepted checkpoint changed target weights')
        h = build_ci_matrix(csfs, waves, functional.keys, data, 70, 123, nuclear_potential=nuclear,
                            convol_cache=raw_functional.cache, max_determinants=2000)
        numeric = {'CI_residual_Ha': float(np.max(np.linalg.norm(h @ current_ci.mixing_coefficients
            - current_ci.mixing_coefficients * current_ci.state_energies, axis=0))),
            'CI_orthogonality': float(np.max(abs(current_ci.mixing_coefficients.T
            @ current_ci.mixing_coefficients - np.eye(531)))),
            'matrix_hermiticity_Ha': float(np.max(abs(h - h.T)))}
        for name in numeric:
            if numeric[name] > gates[name]:
                raise ValueError('Accepted checkpoint numeric gate failed: ' + name)
        gram, leakage = 0., 0.
        for keys, projector, _, _ in state.layout:
            c = np.column_stack([np.concatenate(waves[k]) for k in keys])
            s = raw_functional.metric[keys[0][1]]
            gram = max(gram, float(np.max(abs(c.T @ s @ c - np.eye(len(keys))))))
            outside = c - projector @ (projector.T @ s @ c)
            leakage = max(leakage, float(np.sqrt(max(0., np.max(np.sum(outside * (s @ outside), axis=0))))))
        numeric.update(radial_Gram_error=gram, projector_leakage_norm=leakage)
        if gram > gates['radial_Gram_error'] or leakage > gates['projector_leakage_norm']:
            raise ValueError('Accepted checkpoint radial/no-pair gate failed')
        path = HERE / 'checkpoints'
        path.mkdir(exist_ok=True)
        stem = f'accepted_{iteration:04d}'
        arrays = {'x': state.coordinates, 'inverse_diagonal': state.inverse_diagonal,
                  'energies_Ha': current_ci.state_energies, 'mixing': current_ci.mixing_coefficients,
                  'matrix_Ha': h,
                  'orbital_keys': np.array(sorted(waves)), 'P': np.array([waves[k][0] for k in sorted(waves)]),
                  'Q': np.array([waves[k][1] for k in sorted(waves)])}
        for i, (_, p, _, _) in enumerate(state.layout):
            arrays[f'projector_{i}'] = p
            arrays[f'inverse_blocks_{i}'] = np.array(state.inverse_blocks[i])
        temporary = path / (stem + '.npz.part')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        destination = path / (stem + '.npz')
        temporary.replace(destination)
        public_diagnostics = {k: v for k, v in diagnostics.items() if k != 'optimizer_state'}
        manifest = {'iteration': iteration, 'accepted_complete_point': True, 'array_sha256': sha(destination),
                    'objective_sha256': new_identity, 'original_metric_sha256': expected_metric,
                    'geometry_sha256': state.geometry_sha256, 'physical_tracking': identities,
                    'numerical_checks': numeric,
                    'diagnostics': public_diagnostics, 'source_identity_sha256': sha(HERE / 'source_identity.json'),
                    'preparation_contract_sha256': sha(HERE / 'contract.json'),
                    'frozen_scientific_inputs_sha256': contract['inputs_sha256'],
                    'execution_contract_sha256': sha(execution_contract), 'elapsed_s': time.monotonic() - START,
                    'baseline_gradient_calls': baseline_gradient_calls,
                    'optimizer_full_gradient_calls': functional.gradient_calls,
                    'total_run_full_gradient_calls': baseline_gradient_calls + functional.gradient_calls,
                    'stationarity_passed': diagnostics['orbital_residual'] < 1e-7
                    and diagnostics['estimated_energy_error'] < 1e-8,
                    'production_certified': False, 'experimental_gate_passed': False,
                    'static_Breit_spectrum_changed': False}
        atomic_json(path / (stem + '.json'), manifest)
        checkpoints.append({'path': 'checkpoints/' + stem + '.json', 'sha256': sha(path / (stem + '.json'))})
        atomic_json(HERE / 'latest_checkpoint.json', checkpoints[-1])
        print(json.dumps({'accepted_iteration': iteration, 'orbital_residual': diagnostics['orbital_residual'],
                          'estimated_energy_error': diagnostics['estimated_energy_error'],
                          'elapsed_s': manifest['elapsed_s']}), flush=True)
        host = host_resources(caps, accounting_cpus=admission['available_CPUs'])
        if not host['admitted']:
            atomic_json(HERE / 'accepted_resource_stop.json', {'host': host, 'latest_checkpoint': checkpoints[-1]})
            raise BudgetStop('Accepted-boundary host resource guard failed after checkpoint')

    try:
        # Exactly one public solve; no consumer-side orbital or line-search loop.
        final_waves, final_ci, diagnostics = optimize_orbitals(
            functional, orbitals, resume_state=initial, objective_identity=new_identity, allow_objective_change=True,
            checkpoint_callback=checkpoint,
            max_wall_time=max(0., caps['accepted_boundary_soft_stop_s'] - (time.monotonic() - START)),
            return_state=True, max_iter=caps['max_iterations'], tolerance=1e-7, energy_tolerance=1e-8,
            projector_source='nuclear', orbital_preconditioner='spectral', occupation_floor=.05,
            history_size=10, energy_difference=None, linear_energy_increments=False)
        del final_waves, final_ci
        result = {'status': 'public_optimizer_returned', 'diagnostics': {k: v for k, v in diagnostics.items()
                                                                     if k != 'optimizer_state'},
                  'actual_full_public_optimizer_call': True, 'checkpoints': checkpoints,
                  'gradient_calls': functional.gradient_calls, 'energy_trial_calls': functional.energy_calls,
                  'gradient_calls_scope': 'optimizer functional only; separate baseline calls reported below',
                  'baseline_gradient_calls': baseline_gradient_calls,
                  'optimizer_full_gradient_calls': functional.gradient_calls,
                  'total_run_full_gradient_calls': baseline_gradient_calls + functional.gradient_calls,
                  'optimizer_reports_converged': diagnostics['converged'],
                  'expanded_MCDHF_stationarity_certified': False,
                  'independent_saved_endpoint_verification_pending': True,
                  'production_certified': False, 'experimental_gate_passed': False,
                  'highest_level_reached': False, 'static_Breit_spectrum_changed': False}
        atomic_json(HERE / 'result_unverified.json', result)
    except BaseException as error:
        atomic_json(HERE / 'result_partial.json', {'status': 'stopped_or_failed_preserve_last_checkpoint',
                    'error': str(error), 'traceback': traceback.format_exc(), 'checkpoints': checkpoints,
                    'baseline_gradient_calls': baseline_gradient_calls,
                    'optimizer_full_gradient_calls': functional.gradient_calls,
                    'total_run_full_gradient_calls': baseline_gradient_calls + functional.gradient_calls,
                    'expanded_MCDHF_stationarity_passed': False, 'production_certified': False,
                    'experimental_gate_passed': False, 'static_Breit_spectrum_changed': False})
        raise
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'seal-source', 'solve'), default='preflight', nargs='?')
    parser.add_argument('--execution-contract', type=Path)
    args = parser.parse_args()
    if args.mode == 'preflight':
        answer = prospective_preflight(load(HERE / 'contract.json'))
        print(json.dumps({k: answer[k] for k in ('preparation_only', 'CSFs', 'max_raw_M_equals_J',
                                                'selected_overlap_pair_work_upper', 'resources', 'public_interface')}))
    elif args.mode == 'seal-source':
        print(json.dumps({'source_map_sha256': seal_source()['source_map_sha256'], 'solve_started': False}))
    else:
        if args.execution_contract is None:
            parser.error('solve requires an exact root execution contract; preparation never authorizes execution')
        try:
            execute(args.execution_contract)
        except BaseException as error:
            if not (HERE / 'result_partial.json').exists():
                atomic_json(HERE / 'execution_failure_pre_solve.json', {'status': 'admission_or_baseline_failed',
                    'error': str(error), 'traceback': traceback.format_exc(), 'optimizer_endpoint_certified': False,
                    'latest_complete_checkpoint': load(HERE / 'latest_checkpoint.json')
                    if (HERE / 'latest_checkpoint.json').exists() else None})
            raise


if __name__ == '__main__':
    main()
