"""Prepare or run one explicitly authorized public mixed-CI resume epoch.

The default command only reads archives and reports resources. Execution is
separately bound to a root-owned contract and frozen public source snapshot.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import numbers
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


def canonical_physical_target_groups(groups):
    """Compare complete target spans/weights across runtime tuples and JSON lists.

    Group and within-span enumeration order are not physical identities. Retain
    every UID and weight, reject invalid/repeated UIDs, and ignore only measured
    overlap diagnostics, which legitimately change at a new orbital point.
    """
    canonical = []
    references, selections = set(), set()
    for group in groups:
        spans = []
        for field, seen in (('reference_levels', references), ('selected_levels', selections)):
            levels = []
            for level in group[field]:
                if (not isinstance(level, (tuple, list)) or len(level) != 3
                        or any(not isinstance(v, numbers.Integral) or isinstance(v, bool) for v in level)):
                    raise ValueError('Invalid complete physical level UID')
                uid = tuple(int(v) for v in level)
                if uid[0] < 0 or uid[1] not in (-1, 1) or uid[2] < 0 or uid in seen:
                    raise ValueError('Invalid/repeated complete physical level UID')
                seen.add(uid)
                levels.append(uid)
            if not levels:
                raise ValueError('Empty physical target span')
            spans.append(tuple(sorted(levels)))
        weight = group['per_level_weight']
        if (len(spans[0]) != len(spans[1]) or not isinstance(weight, numbers.Real)
                or isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0):
            raise ValueError('Invalid physical target rank/weight')
        canonical.append((spans[0], spans[1], float(weight)))
    if not canonical:
        raise ValueError('Empty physical target groups')
    return tuple(sorted(canonical))


def start_checkpoint_status(contract):
    """Check exact complete160 point, objective and independent original-metric replay."""
    point = contract['start_checkpoint']
    manifest = load(ROOT / point['manifest'])
    parent = load(ROOT / point['parent_contract'])
    verification = load(ROOT / point['independent_verification'])
    objective = load(ROOT / point['parent_objective_identity'])
    if (manifest.get('accepted_complete_point') is not True
            or manifest['iteration'] != point['reference_iteration']
            or manifest['array_sha256'] != sha(ROOT / point['arrays'])
            or manifest['source_identity_sha256'] != sha(ROOT / point['parent_source_identity'])
            or manifest['preparation_contract_sha256'] != sha(ROOT / point['parent_contract'])
            or manifest['execution_contract_sha256'] != sha(ROOT / point['parent_execution_contract'])
            or manifest['frozen_scientific_inputs_sha256'] != parent['inputs_sha256']
            or manifest['objective_sha256'] != point['same_objective_sha256']
            or manifest['diagnostics']['objective_identity'] != point['same_objective_sha256']
            or digest(objective['new']) != point['same_objective_sha256']
            or objective['new_sha256'] != point['same_objective_sha256']
            or verification.get('all_identity_replay_checks_passed') is not True
            or not all(c['passed'] for c in verification['checks'])
            or verification['checks_count'] != len(verification['checks'])
            or verification['new_source_identity_sha256'] != sha(ROOT / point['parent_source_identity'])
            or verification['preparation_contract_sha256'] != sha(ROOT / point['parent_contract'])
            or verification['execution_contract_sha256'] != sha(ROOT / point['parent_execution_contract'])
            or verification['objective_identity_file_sha256'] != sha(ROOT / point['parent_objective_identity'])
            or verification['accepted_checkpoint_sha256'] != sha(ROOT / point['manifest'])
            or verification['accepted_arrays_sha256'] != sha(ROOT / point['arrays'])
            or verification['independently_rebuilt_objective_sha256'] != point['same_objective_sha256']
            or verification['independently_rebuilt_objective_payload'] != objective['new']
            or verification['actual_original_metric_orbital_residual'] != point['starting_residual']
            or verification['actual_original_metric_energy_indicator_Ha'] != point['starting_energy_indicator_Ha']
            or verification['original_stationarity_gates_passed'] != point['starting_original_stationarity_passed']
            or manifest['original_metric_sha256'] != contract['fixed_conditions']['original_projector_and_preconditioner_metric_sha256']
            or manifest['geometry_sha256'] != contract['fixed_conditions']['same_geometry_sha256']):
        raise ValueError('Complete starting160 checkpoint/provenance/objective/original-metric replay changed')
    return {'ready': True, 'iteration': manifest['iteration'], 'accepted_checkpoint_sha256': sha(ROOT / point['manifest']),
            'accepted_arrays_sha256': sha(ROOT / point['arrays']),
            'independent_verification_sha256': sha(ROOT / point['independent_verification']),
            'same_objective_sha256': point['same_objective_sha256'],
            'original_stationarity_passed': verification['original_stationarity_gates_passed'],
            'starting_original_residual': verification['actual_original_metric_orbital_residual'],
            'starting_original_indicator_Ha': verification['actual_original_metric_energy_indicator_Ha']}


def prospective_preflight(contract):
    for path, expected in contract['inputs_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Frozen scientific input changed: ' + path)
    caps = contract['resources']
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    # Read-only same10d481 source enumeration; no functional/CI is created.
    start_status = start_checkpoint_status(contract)
    frozen = ROOT / contract['fixed_conditions']['frozen_source_root']
    original_source = load(ROOT / contract['fixed_conditions']['frozen_source_identity_file'])
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
    recovered = np.load(ROOT / contract['start_checkpoint']['arrays'])
    if recovered['x'].shape != (2856,) or recovered['x'].dtype != np.float64 or not np.isfinite(recovered['x']).all():
        raise ValueError('Exact accepted160 coordinate identity failed')
    old_metric = np.load(ROOT / (PREFIX + 'native_run/nuclear_eol004_stable_delta_trace/failed_coordinates.npz'))
    layout = load(ROOT / (PREFIX + 'native_run/nuclear_eol004_stable_delta_trace/failed_coordinate_layout.json'))
    metric_digest = hashlib.sha256()
    for i, row in enumerate(layout):
        metric_digest.update(repr(([tuple(k) for k in row['keys']], tuple(row['shape']))).encode())
        metric_digest.update(np.ascontiguousarray(old_metric[f'projector_{i}']).tobytes())
        if not np.array_equal(old_metric[f'projector_{i}'], recovered[f'projector_{i}']):
            raise ValueError('Accepted160 projector differs from original004')
    metric_digest.update(old_metric['inverse_diagonal'].tobytes())
    if not np.array_equal(old_metric['inverse_diagonal'], recovered['inverse_diagonal']):
        raise ValueError('Accepted160 inverse diagonal differs from original004')
    for i in range(len(layout)):
        if not np.array_equal(old_metric[f'inverse_blocks_{i}'], recovered[f'inverse_blocks_{i}']):
            raise ValueError('Accepted160 inverse blocks differ from original004')
        for block in old_metric[f'inverse_blocks_{i}']:
            metric_digest.update(np.ascontiguousarray(block).tobytes())
    if metric_digest.hexdigest() != contract['fixed_conditions']['original_projector_and_preconditioner_metric_sha256']:
        raise ValueError('Original complete metric changed')
    keys = [tuple(k) for k in recovered['orbital_keys']]
    if (len(keys) != 24 or len(set(keys)) != 24 or recovered['orbital_keys'].shape != (24, 2)
            or any(recovered[name].shape != (24, 123) or recovered[name].dtype != np.float64
                   or not np.isfinite(recovered[name]).all() for name in ('P', 'Q'))):
        raise ValueError('Accepted160 complete float64 radial archive changed')
    waves = {key: np.concatenate((p, q)) for key, p, q in zip(keys, recovered['P'], recovered['Q'], strict=True)}
    for i, row in enumerate(layout):
        values = old_metric[f'projector_{i}'] @ recovered['x'][slice(*row['slice'])].reshape(row['shape'])
        if any(not np.array_equal(values[:, col], waves[tuple(key)]) for col, key in enumerate(row['keys'])):
            raise ValueError('Readonly accepted160 complete physical waves are not exact original P*x')
    current_sources = source_map(ROOT / 'src/atomx')
    actual_runtime_source_map_sha256 = digest({'src/' + name: value for name, value in current_sources.items()})
    if actual_runtime_source_map_sha256 != contract['scientific_source_map_required_sha256']:
        raise ValueError('Prepared continuation must retain exact parent scientific source')
    answer = {'preparation_only': True, 'functional_evaluations': 0, 'CI_solves': 0, 'optimizer_calls': 0,
              'starting_full531_checkpoint': start_status,
              'contract_sha256': sha(HERE / 'contract.json'), 'CSFs': 531, 'radial_orbitals': 24,
              'blocks': table, 'max_raw_M_equals_J': max(r['raw_M_equals_J'] for r in table),
              'selected_overlap_pair_work_upper': overlap_pairs, 'coordinates_shape': [2856],
              'original_metric_sha256': metric_digest.hexdigest(), 'resources': host_resources(caps),
              'public_interface': interface_status(ROOT / 'src/atomx/numerics/bspline/orbital_optimization.py'),
              'planning_current_source_map_sha256': digest(current_sources),
              'planning_runtime_source_map_sha256': actual_runtime_source_map_sha256,
              'planning_current_sources_sha256': current_sources,
              'execution_source_epoch_frozen': False, 'input_hashes_verified': len(contract['inputs_sha256'])}
    answer['starting353_source_files_verified'] = len(original_source['all_atomx_python_sha256'])
    atomic_json(HERE / 'preflight.json', answer)
    return answer


def seal_source():
    if (HERE / 'source_identity.json').exists() or (HERE / 'source').exists():
        raise FileExistsError('Never replace an existing source epoch')
    contract = load(HERE / 'contract.json')
    start_checkpoint_status(contract)
    status = interface_status(ROOT / 'src/atomx/numerics/bspline/orbital_optimization.py')
    if not status['available']:
        raise RuntimeError('Public resume API not available; no source epoch created')
    original = load(ROOT / contract['fixed_conditions']['frozen_source_identity_file'])['all_atomx_python_sha256']
    current = {'src/' + name: value for name, value in source_map(ROOT / 'src/atomx').items()}
    if digest(current) != contract['scientific_source_map_required_sha256']:
        raise ValueError('Same-objective continuation must not change scientific source')
    shutil.copytree(ROOT / 'src/atomx', HERE / 'source/atomx', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbc', '*.nbi'))
    copied = {'src/' + name: value for name, value in source_map(HERE / 'source/atomx').items()}
    if copied != current:
        raise ValueError('Source changed during epoch copy; stop without solving')
    identity = {'source_map_sha256': digest(current), 'all_atomx_python_sha256': current,
                'parent_source_map_sha256': contract['scientific_source_map_required_sha256'],
                'changed_from_parent353': [k for k in current if k in original and current[k] != original[k]],
                'added_since_parent353': sorted(current.keys() - original.keys()),
                'deleted_since_parent353': sorted(original.keys() - current.keys()),
                'public_interface': status, 'driver_sha256': sha(Path(__file__)),
                'preparation_contract_sha256': sha(HERE / 'contract.json'), 'solve_started': False}
    atomic_json(HERE / 'source_identity.json', identity)
    return identity


class BudgetStop(RuntimeError):
    """A resource stop preserves the most recent fully accepted checkpoint."""


def execute(execution_contract):
    approval = load(execution_contract)
    contract = load(HERE / 'contract.json')
    start_checkpoint_status(contract)
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
    actual_records = []
    for config_index, row in enumerate(enumeration['configs']):
        for csf in enumerate_jj_csfs(row['config']):
            signature = csf_signature(csf)
            actual_records.append({'index': len(actual_records), 'config_index': config_index,
                'family': row['family'], 'label': csf.label, 'twoJ': csf.total_J2, 'parity': csf.parity,
                'subshells': [list(s) for s in signature.subshells],
                'coupling': [list(c) for c in signature.coupling],
                'canonical_pattern': [list(p) for p in signature.pattern]})
    if actual_records != enumeration['CSFs']:
        raise ValueError('Actual complete signed CSF order/labels/coupling changed')
    targets = load(replay / 'stage_4f5p5s/target_binding.json')['targets']
    selectors = tuple((t['twoJ'], t['parity'], t['models']['DC']['local_root'], 1.) for t in targets)
    average = CIStateAverage(selectors)
    radial = np.load(native / 'orbitals_workspace.npz')
    physical_reference_waves = {tuple(k): (p.copy(), q.copy()) for k, p, q in
                                zip(radial['orbital_keys'], radial['P'], radial['Q'], strict=True)}
    data = {n[6:]: radial[n].item() if radial[n].ndim == 0 else radial[n]
            for n in radial.files if n.startswith('basis_')}
    metric = np.load(original / 'failed_coordinates.npz')
    recovered = np.load(ROOT / contract['start_checkpoint']['arrays'])
    entry_manifest = load(ROOT / contract['start_checkpoint']['manifest'])
    entry_verification = load(ROOT / contract['start_checkpoint']['independent_verification'])
    from verify_endpoint import saved_eigenpair_checks

    saved_checks = saved_eigenpair_checks(recovered, gates, 531)
    atomic_json(HERE / 'starting_saved_eigenpair_controls.json', {'checks': saved_checks,
        'actual_start_manifest_sha256': sha(ROOT / contract['start_checkpoint']['manifest']),
        'full_gradients': 0, 'physical_H_builds': 0, 'optimizer_calls': 0})
    if not all(c['passed'] for c in saved_checks):
        raise ValueError('Complete saved160 eigenpairs failed before creating functional')
    rows = load(original / 'failed_coordinate_layout.json')
    layout = tuple((tuple(tuple(k) for k in row['keys']), metric[f'projector_{i}'].copy(),
                    tuple(row['shape']), slice(*row['slice'])) for i, row in enumerate(rows))
    inverse_blocks = tuple(tuple(b.copy() for b in metric[f'inverse_blocks_{i}']) for i in range(len(rows)))
    nuclear = make_nuclear_potential(70, 'fermi', A=175)
    raw_functional = CSFOrbitalFunctional(csfs, data, 70, nuclear, state_average=average, max_determinants=2000)
    target_identity = [{'label': t['label'], 'twoJ': t['twoJ'], 'parity': t['parity'],
                        'root': t['models']['DC']['local_root'], 'weight': .2} for t in targets]
    common = {'Z': 70, 'electrons': 69, 'operator': 'full declared no-pair Dirac-Coulomb',
              'nucleus': 'native005 Fermi A175', 'radial_archive_sha256': sha(native / 'orbitals_workspace.npz'),
              'target_identity': target_identity}
    new_payload = {**common, 'CSFs': actual_records, 'source_map_sha256': source['source_map_sha256'],
                   'physical_reference_CI_sha256': sha(replay / 'stage_4f5p5s/DC_arrays.npz'),
                   'physical_reference_binding_sha256': sha(replay / 'stage_4f5p5s/target_binding.json')}
    new_identity = digest(new_payload)
    parent_objective = load(ROOT / contract['start_checkpoint']['parent_objective_identity'])
    if (new_identity != contract['fixed_conditions']['same_objective_sha256']
            or new_payload != parent_objective['new'] or new_identity != entry_manifest['objective_sha256']):
        raise ValueError('Continuation changed full531 physical objective or frozen reference')
    atomic_json(HERE / 'objective_identity.json', {'new': new_payload, 'new_sha256': new_identity,
        'same_objective_as_parent': True, 'parent_objective_identity_sha256':
        sha(ROOT / contract['start_checkpoint']['parent_objective_identity'])})
    initial = OrbitalOptimizationState.from_arrays(
        raw_functional, layout=layout, coordinates=recovered['x'].copy(),
        inverse_diagonal=metric['inverse_diagonal'].copy(), inverse_blocks=inverse_blocks,
        projector_source='nuclear', objective_identity=new_identity, occupation_floor=.05)
    expected_metric = contract['fixed_conditions']['original_projector_and_preconditioner_metric_sha256']
    if initial.residual_metric_sha256 != expected_metric:
        raise ValueError('Public typed state rebuilt or changed original metric')
    if (recovered['x'].shape != (2856,) or recovered['x'].dtype != np.float64
            or not np.isfinite(recovered['x']).all()
            or not np.array_equal(initial.coordinates, recovered['x'])
            or initial.geometry_sha256 != entry_manifest['geometry_sha256']
            or not np.array_equal(initial.inverse_diagonal, recovered['inverse_diagonal'])
            or any(not np.array_equal(p, recovered[f'projector_{i}'])
                   or not np.array_equal(np.asarray(initial.inverse_blocks[i]), recovered[f'inverse_blocks_{i}'])
                   for i, (_, p, _, _) in enumerate(initial.layout))):
        raise ValueError('Full original-metric accepted160 coordinates/geometry changed')
    orbitals = {}
    for keys, projector, shape, where in initial.layout:
        c = projector @ initial.coordinates[where].reshape(shape)
        for col, key in enumerate(keys):
            orbitals[key] = (c[:123, col].copy(), c[123:, col].copy())
    reference_waves = {tuple(key): (p, q) for key, p, q in
                       zip(recovered['orbital_keys'], recovered['P'], recovered['Q'], strict=True)}
    if (recovered['orbital_keys'].shape != (24, 2)
            or recovered['P'].shape != (24, 123) or recovered['Q'].shape != (24, 123)
            or recovered['P'].dtype != np.float64 or recovered['Q'].dtype != np.float64
            or set(reference_waves) != set(orbitals) or len(reference_waves) != 24
            or any(not np.array_equal(a, b) for key in orbitals
                   for a, b in zip(orbitals[key], reference_waves[key], strict=True))):
        raise ValueError('Accepted160 saved waves do not equal exact direct original-P*x')
    saved = np.load(replay / 'stage_4f5p5s/DC_arrays.npz')
    tracking = CIBasisTracking(average, csfs, CIResult(saved['energies_Ha'], saved['mixing']), physical_reference_waves,
                              raw_functional.metric, max_determinant_pairs=1_000_000)
    raw_functional = CSFOrbitalFunctional(csfs, data, 70, nuclear, state_average=average,
        level_tracking=tracking, cache=raw_functional.cache, max_determinants=2000)

    def original_metrics(gradient, coordinates=None):
        coordinates = initial.coordinates if coordinates is None else coordinates
        g = np.zeros(2856)
        for keys, projector, shape, where in initial.layout:
            y = coordinates[where].reshape(shape)
            block = projector.T @ np.column_stack([gradient[k] for k in keys])
            multiplier = y.T @ block
            g[where] = (block - y @ (.5 * (multiplier + multiplier.T))).ravel()
        preconditioned = np.zeros_like(g)
        for (_, _, shape, where), blocks in zip(initial.layout, initial.inverse_blocks, strict=True):
            y, b = coordinates[where].reshape(shape), g[where].reshape(shape)
            z = np.column_stack([p @ b[:, col] for col, p in enumerate(blocks)])
            multiplier = y.T @ z
            preconditioned[where] = (z - y @ (.5 * (multiplier + multiplier.T))).ravel()
        return float(np.linalg.norm(preconditioned)), .5 * abs(float(g @ preconditioned)), g, preconditioned

    # Two full531 gradients at exactly the same160 point, with no bare005 gate.
    baseline_gradient_calls = 1
    energy, gradient, ci = raw_functional.evaluate(orbitals)
    h = build_ci_matrix(csfs, orbitals, raw_functional.keys, data, 70, 123, nuclear_potential=nuclear,
                        convol_cache=raw_functional.cache, max_determinants=2000)
    residual, indicator, mixed_g, mixed_preconditioned = original_metrics(gradient)
    baseline_gradient_calls += 1
    saved_energy, saved_gradient, _ = raw_functional.evaluate(orbitals,
        ci_result=CIResult(recovered['energies_Ha'], recovered['mixing']))
    saved_residual, saved_indicator, saved_g, saved_preconditioned = original_metrics(saved_gradient)
    weights, identities = tracking.resolve(csfs, ci, orbitals, raw_functional.metric)
    target_identities = canonical_physical_target_groups(identities)
    previous_target_identities = canonical_physical_target_groups(entry_manifest['physical_tracking'])
    if (target_identities != previous_target_identities or np.count_nonzero(weights) != 5
            or not np.array_equal(weights[weights > 0], np.repeat(.2, 5))):
        raise ValueError('Samepoint full531 entry changed physical target identities/weights')
    fresh_numeric = {'CI_residual_Ha': float(np.max(np.linalg.norm(h @ ci.mixing_coefficients
        - ci.mixing_coefficients * ci.state_energies, axis=0))),
        'CI_orthogonality': float(np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients - np.eye(531)))),
        'matrix_hermiticity_Ha': float(np.max(abs(h - h.T)))}
    if any(value > gates[name] for name, value in fresh_numeric.items()):
        raise ValueError('Samepoint full531 fresh complete eigenpair/Gram/H gate failed before optimizer')
    controls = {'baseline_gradient_calls': baseline_gradient_calls,
        'total_run_full_gradient_calls_before_optimizer': baseline_gradient_calls,
        'planned_max_total_run_full_gradient_calls': caps['max_total_run_full_gradient_calls'],
        'same_start160_manifest_sha256': sha(ROOT / contract['start_checkpoint']['manifest']),
        'same_start160_arrays_sha256': sha(ROOT / contract['start_checkpoint']['arrays']),
        'same_independent160_replay_sha256': sha(ROOT / contract['start_checkpoint']['independent_verification']),
        'same_authoritative_x_direct_waves_exact': True, 'same_objective_sha256': new_identity,
        'mixed_EO_energy_Ha': energy, 'mixed_original_residual': residual,
        'mixed_original_energy_indicator_Ha': indicator,
        'matrix_replay_error_Ha': float(np.max(abs(h - recovered['matrix_Ha']))),
        'all_CI_energy_replay_error_Ha': float(np.max(abs(ci.state_energies - recovered['energies_Ha']))),
        'samepoint_full531_projected_gradient_error': float(np.max(abs(mixed_g - saved_g))),
        'samepoint_full531_preconditioned_gradient_error': float(np.max(abs(mixed_preconditioned - saved_preconditioned))),
        'samepoint_saved_CI_EOL_error_Ha': abs(energy - saved_energy),
        'samepoint_saved_CI_residual_error': abs(residual - saved_residual),
        'samepoint_saved_CI_indicator_error_Ha': abs(indicator - saved_indicator),
        'entry_independent160_residual_error': abs(residual - entry_verification['actual_original_metric_orbital_residual']),
        'entry_independent160_indicator_error_Ha': abs(indicator - entry_verification['actual_original_metric_energy_indicator_Ha']),
        'entry_reported_EO_error_Ha': abs(energy - entry_manifest['diagnostics']['energy_ha']),
        'full_fresh_numeric': fresh_numeric, 'physical_tracking': identities,
        'metric_sha256': expected_metric, 'starting_stationarity_passed': residual < 1e-7 and indicator < 1e-8,
        'nonstationary_start_permitted_not_certified': True, 'bare005_gradient_evaluations': 0}
    atomic_json(HERE / 'new_source_baseline_controls.json', controls)
    if (controls['matrix_replay_error_Ha'] > gates['baseline_matrix_replay_Ha']
            or controls['all_CI_energy_replay_error_Ha'] > gates['baseline_target_energy_replay_Ha']
            or controls['entry_reported_EO_error_Ha'] > gates['baseline_target_energy_replay_Ha']
            or controls['samepoint_saved_CI_EOL_error_Ha'] > gates['baseline_target_energy_replay_Ha']
            or controls['samepoint_full531_projected_gradient_error'] > gates['full531_same_start_projected_gradient_replay_max']
            or controls['samepoint_full531_preconditioned_gradient_error'] > gates['full531_same_start_preconditioned_gradient_replay_max']
            or controls['entry_independent160_residual_error'] > gates['entry_residual_replay_absolute']
                + gates['entry_residual_replay_relative'] * residual
            or controls['entry_independent160_indicator_error_Ha'] > gates['entry_indicator_replay_absolute_Ha']
                + gates['entry_indicator_replay_relative'] * indicator):
        raise ValueError('Samepoint full531 entry controls failed; no optimization started')

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
        _, _, accepted_g, accepted_pg = original_metrics(functional.last_gradient, state.coordinates)
        arrays = {'original_projected_gradient': accepted_g, 'original_preconditioned_gradient': accepted_pg,
                  'x': state.coordinates, 'inverse_diagonal': state.inverse_diagonal,
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
        manifest = {'iteration': iteration, 'parent_accepted_iteration': 160, 'accepted_complete_point': True, 'array_sha256': sha(destination),
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
            functional, orbitals, resume_state=initial, objective_identity=new_identity, allow_objective_change=False,
            checkpoint_callback=checkpoint,
            max_wall_time=max(0., caps['accepted_boundary_soft_stop_s'] - (time.monotonic() - START)),
            return_state=True, max_iter=caps['max_iterations'], tolerance=1e-7, energy_tolerance=1e-8,
            projector_source='nuclear', orbital_preconditioner='spectral', occupation_floor=.05,
            history_size=10, energy_difference=None, linear_energy_increments=False)
        del final_waves, final_ci
        result = {'status': 'public_optimizer_returned', 'diagnostics': {k: v for k, v in diagnostics.items()
                                                                     if k != 'optimizer_state'},
                  'actual_full_public_optimizer_call': True, 'same_objective_as_parent': True,
                  'fresh_secant_history_declared': True, 'parent_accepted_iteration': 160, 'checkpoints': checkpoints,
                  'gradient_calls': functional.gradient_calls, 'energy_trial_calls': functional.energy_calls,
                  'gradient_calls_scope': 'optimizer functional only; two full531samepoint baseline calls separate; no bare005 gradient',
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
