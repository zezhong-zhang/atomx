"""Two prospective bare5 gradients at two explicitly different saved points."""
import hashlib
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[2]
BASE = ROOT / 'docs/yb_mcdhf_rci_2026_10_11'
START = time.monotonic()
CALLS = 0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(name, value):
    destination = OWN / name
    temporary = destination.with_name(destination.name + '.part')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(destination)


def host(caps):
    mem = {k: int(v.split()[0]) for k, v in
           (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    affinity = sorted(os.sched_getaffinity(0))
    cpus = len(affinity)
    quota = Path('/sys/fs/cgroup/cpu.max')
    q = quota.read_text().split() if quota.exists() else ['max']
    if q[0] != 'max':
        cpus = min(cpus, int(q[0]) / int(q[1]))
    available = mem['MemAvailable'] / 1024**2
    mx, current = Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory.current')
    if mx.exists() and current.exists() and mx.read_text().strip() != 'max':
        available = min(available, (int(mx.read_text()) - int(current.read_text())) / 1024**3)
    loads = os.getloadavg()
    return {'affinity': affinity, 'CPUs': cpus, 'available_memory_GiB': available,
            'load_average': loads, 'load1_per_available_CPU': loads[0] / cpus,
            'admitted': available >= caps['minimum_host_available_GiB']
            and loads[0] / cpus <= caps['max_load1_per_available_CPU']}


def run():
    global CALLS
    if (OWN / 'arrays.npz').exists() or (OWN / 'result.json').exists():
        raise FileExistsError('Never replace a completed scientific reference')
    contract = load(OWN / 'contract.json')
    caps = contract['resources']
    admission = host(caps)
    admission.update(contract_sha256=sha(OWN / 'contract.json'), driver_sha256=sha(Path(__file__)))
    save('execution_resource_admission.json', admission)
    if not admission['admitted']:
        raise RuntimeError('Resource admission failed before import/evaluation')
    for relative, expected in contract['inputs_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen scientific input changed: ' + relative)
    original = load(BASE / 'native_run/nuclear_eol005_projected_increment/source_identity.json')
    frozen = Path(contract['old_source_root'])
    assert original['source_map_sha256'] == contract['old_source_map_sha256']
    for relative, expected in original['all_atomx_python_sha256'].items():
        if sha(frozen / Path(relative).relative_to('src')) != expected:
            raise ValueError('Frozen original source changed: ' + relative)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['NUMBA_CACHE_DIR'] = str(OWN / 'compiler_cache')
    os.sched_setaffinity(0, {admission['affinity'][-1]})
    byte_cap = int(caps['address_space_GiB'] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (byte_cap, byte_cap))

    def stop(*_):
        raise TimeoutError('Contract hard60s cap; retain any completed point')

    signal.signal(signal.SIGALRM, stop)
    signal.alarm(caps['hard_wall_s'])
    assert not any(n == 'atomx' or n.startswith('atomx.') for n in sys.modules)
    sys.path.insert(0, str(frozen))
    import numpy as np

    from atomx.angular.csf_hamiltonian import csf_signature
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.potentials.nuclear import make_nuclear_potential

    native = BASE / 'native_run/nuclear_eol005_projected_increment'
    metric_dir = BASE / 'native_run/nuclear_eol004_stable_delta_trace'
    identity = load(BASE / 'balanced_cv_replay/basis_identity.json')
    bare = [c for row in identity['configs'] if row['family'] == 'bare'
            for c in enumerate_jj_csfs(row['config'])]
    records = [row for row in identity['CSFs'] if row['family'] == 'bare']
    assert len(bare) == 5 and {c.n_electrons for c in bare} == {69}
    for csf, row in zip(bare, records, strict=True):
        sig = csf_signature(csf)
        assert csf.total_J2 == row['twoJ'] and csf.parity == row['parity']
        assert [list(x) for x in sig.subshells] == row['subshells']
        assert json.loads(json.dumps(sig.coupling)) == row['coupling']
    targets = load(BASE / 'balanced_cv_replay/stage_4f5p5s/target_binding.json')['targets']
    selectors = tuple((t['twoJ'], t['parity'], t['models']['DC']['local_root'], 1.) for t in targets)
    assert selectors == ((1, 1, 0, 1.), (3, 1, 0, 1.), (5, 1, 0, 1.), (1, -1, 0, 1.), (3, -1, 0, 1.))
    average = CIStateAverage(selectors)
    workspace = np.load(native / 'orbitals_workspace.npz')
    data = {n[6:]: workspace[n].item() if workspace[n].ndim == 0 else workspace[n]
            for n in workspace.files if n.startswith('basis_')}
    metric = np.load(metric_dir / 'failed_coordinates.npz')
    recovered = np.load(native / 'stationarity_replay002/recovered_coordinates.npz')
    x = recovered['x'].copy()
    assert x.shape == (2856,) and x.dtype == np.float64
    rows = load(metric_dir / 'failed_coordinate_layout.json')
    functional = CSFOrbitalFunctional(bare, data, 70, make_nuclear_potential(70, 'fermi', A=175),
                                     state_average=average, max_determinants=2000)
    assert functional.ns == 123 and len(functional.keys) == 24
    layout = [(tuple(tuple(k) for k in row['keys']), metric[f'projector_{i}'],
               tuple(row['shape']), slice(*row['slice'])) for i, row in enumerate(rows)]
    digest = hashlib.sha256()
    for keys, p, shape, _ in layout:
        digest.update(repr((list(keys), shape)).encode())
        digest.update(np.ascontiguousarray(p).tobytes())
    digest.update(metric['inverse_diagonal'].tobytes())
    for i in range(len(layout)):
        for block in metric[f'inverse_blocks_{i}']:
            digest.update(np.ascontiguousarray(block).tobytes())
    metric_sha = digest.hexdigest()
    assert metric_sha == contract['fixed']['original_metric_sha256']
    physical = {tuple(k): (p.copy(), q.copy()) for k, p, q in
                zip(workspace['orbital_keys'], workspace['P'], workspace['Q'], strict=True)}
    actual = {}
    for keys, p, shape, where in layout:
        coefficients = p @ x[where].reshape(shape)
        for col, key in enumerate(keys):
            actual[key] = coefficients[:123, col].copy(), coefficients[123:, col].copy()
    assert set(actual) == set(physical) == set(functional.keys)
    coefficients_difference = max(float(np.max(abs(a - b))) for key in actual
                                  for a, b in zip(actual[key], physical[key], strict=True))
    all_arrays = {'x': x, 'inverse_diagonal': metric['inverse_diagonal']}
    for i, (keys, projector, _, _) in enumerate(layout):
        all_arrays[f'projector_{i}'] = projector
        all_arrays[f'inverse_blocks_{i}'] = metric[f'inverse_blocks_{i}']
        all_arrays[f'metric_{i}'] = functional.metric[keys[0][1]]
        all_arrays[f'one_body_{i}'] = functional.one_body[keys[0][1]]

    def evaluate(waves, file):
        global CALLS
        if CALLS >= 2:
            raise RuntimeError('No third gradient is permitted')
        CALLS += 1
        energy, gradient, ci = functional.evaluate(waves)
        g, pg = np.zeros_like(x), np.zeros_like(x)
        for keys, p, shape, where in layout:
            y = x[where].reshape(shape)
            derivative = p.T @ np.column_stack([gradient[k] for k in keys])
            multiplier = y.T @ derivative
            g[where] = (derivative - y @ (.5 * (multiplier + multiplier.T))).ravel()
        for i, (_, _, shape, where) in enumerate(layout):
            y, b = x[where].reshape(shape), g[where].reshape(shape)
            z = np.column_stack([p @ b[:, col] for col, p in enumerate(metric[f'inverse_blocks_{i}'])])
            multiplier = y.T @ z
            pg[where] = (z - y @ (.5 * (multiplier + multiplier.T))).ravel()
        payload = dict(all_arrays, g=g, preconditioned_gradient=pg, orbital_keys=np.array(functional.keys),
                       P=np.array([waves[k][0] for k in functional.keys]),
                       Q=np.array([waves[k][1] for k in functional.keys]),
                       radial_gradient=np.array([gradient[k] for k in functional.keys]),
                       ci_energies_Ha=ci.state_energies, ci_mixing=ci.mixing_coefficients,
                       EOL_energy_Ha=np.array(energy), state_weights=functional.last_state_weights)
        assert np.array_equal(payload['state_weights'], np.full(5, .2))
        np.savez_compressed(OWN / file, **payload)
        radial_gram = max(float(np.linalg.norm(
            np.column_stack([np.concatenate(waves[k]) for k in keys]).T @ functional.metric[keys[0][1]]
            @ np.column_stack([np.concatenate(waves[k]) for k in keys]) - np.eye(len(keys)), ord=np.inf))
            for keys, _, _, _ in layout)
        return {'EOL_energy_Ha': float(energy), 'original_metric_orbital_residual': float(np.linalg.norm(pg)),
                'original_metric_energy_indicator_Ha': .5 * abs(float(g @ pg)),
                'original_stationarity_passed': np.linalg.norm(pg) < 1e-7 and .5 * abs(float(g @ pg)) < 1e-8,
                'radial_S_Gram_defect': radial_gram,
                'g_minus_archived_recovered_g_max': float(np.max(abs(g - recovered['g']))),
                'pg_minus_archived_recovered_pg_max': float(np.max(abs(pg - recovered['preconditioned_gradient']))),
                'arrays_sha256': sha(OWN / file)}, payload

    physical_result, physical_payload = evaluate(physical, 'physical_point_arrays.npz')
    save('physical_point_completed.json', physical_result)
    actual_result, actual_payload = evaluate(actual, 'arrays.npz')
    archived_gradients = np.load(native / 'ci_arrays.npz')
    radial_error = max(float(np.max(abs(physical_payload['radial_gradient'][functional.keys.index(tuple(k))] - g)))
                       for k, g in zip(archived_gradients['gradient_keys'], archived_gradients['gradient'], strict=True))
    stopped = load(BASE / 'public_mixed_resume_run/new_source_baseline_controls.json')
    loaded = {}
    for name, module in sys.modules.items():
        if name == 'atomx' or name.startswith('atomx.'):
            file = getattr(module, '__file__', None)
            if file is None:
                raise RuntimeError('Unidentified atomx module: ' + name)
            path = Path(file).resolve()
            path.relative_to(frozen.resolve())
            loaded[name] = {'file': str(path), 'sha256': sha(path)}
    for relative, expected in contract['inputs_sha256'].items():
        assert sha(ROOT / relative) == expected
    for relative, expected in original['all_atomx_python_sha256'].items():
        assert sha(frozen / Path(relative).relative_to('src')) == expected
    comparisons = {
        'original_physical_radial_gradient_minus_original005_archive_max': radial_error,
        'old_source_actual_minus_stopped_new_EO_Ha': actual_result['EOL_energy_Ha'] - stopped['bare_EO_energy_Ha'],
        'old_source_actual_minus_stopped_new_original_residual': actual_result['original_metric_orbital_residual'] - stopped['bare_original_residual'],
        'old_source_actual_minus_stopped_new_original_indicator_Ha': actual_result['original_metric_energy_indicator_Ha'] - stopped['bare_original_energy_indicator_Ha'],
        'old_source_point_g_difference_max_minus_stopped_rawg_difference_max': actual_result['g_minus_archived_recovered_g_max'] - stopped['bare_recovered_gradient_replay_max'],
        'old_source_point_pg_difference_max_minus_stopped_pg_difference_max': actual_result['pg_minus_archived_recovered_pg_max'] - stopped['bare_recovered_preconditioned_gradient_replay_max'],
        'complete_old_current_elementwise_g_comparison_obtained': False,
        'reason': 'Stopped process retained scalar baseline controls, not its full g array; successor must compare this source-bound full reference elementwise.'}
    finding = (physical_result['g_minus_archived_recovered_g_max'] <= 1e-9
               and physical_result['pg_minus_archived_recovered_pg_max'] <= 1e-10
               and radial_error <= 1e-10
               and actual_result['g_minus_archived_recovered_g_max'] > 1e-9
               and abs(comparisons['old_source_point_g_difference_max_minus_stopped_rawg_difference_max']) <= 1e-9
               and abs(comparisons['old_source_actual_minus_stopped_new_EO_Ha']) <= 1e-9
               and abs(comparisons['old_source_actual_minus_stopped_new_original_residual']) <= 1e-10)
    save('result.json', {'status': 'completed_two_old_source_bare_gradients',
        'original005_source_map_sha256': original['source_map_sha256'],
        'recovered_coordinate_archive_sha256': sha(native / 'stationarity_replay002/recovered_coordinates.npz'),
        'original_metric_sha256': metric_sha,
        'original_layout_sha256': sha(metric_dir / 'failed_coordinate_layout.json'),
        'point_identity': 'direct_original_projector_times_archived_recovered_x',
        'x_raw_bytes_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
        'contract_sha256': sha(OWN / 'contract.json'), 'driver_sha256': sha(Path(__file__)),
        'arrays_sha256': sha(OWN / 'arrays.npz'), 'physical_point_arrays_sha256': sha(OWN / 'physical_point_arrays.npz'),
        'physical_model': {'bare_CSFS': 5, 'Z': 70, 'electrons': 69, 'radial_orbitals': 24, 'ns': 123,
                           'nucleus': 'Fermi A175', 'operator': 'bare5 no-pair DC', 'selectors': selectors,
                           'state_weights': [.2] * 5},
        'physical_original005': physical_result, 'actual_x_unpack_originalpoint': actual_result,
        'physical_to_actual_coefficient_max': coefficients_difference,
        'comparisons': comparisons,
        'different_radial_point_explanation_supported': bool(finding),
        'finding': 'Archived recovered g belongs to original005 saved physical waves; it is not the gradient evaluated on directP@recoveredx. Strict matched-point gates are retained for the new full reference.',
        'scientific_inputs_unchanged': True, 'original_source_files_unchanged': 353,
        'loaded_atomx_modules': loaded, 'full_bare_gradient_calls': CALLS, 'optimizer_calls': 0,
        'seconds_uncontrolled': time.monotonic() - START,
        'observed_peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'enforced_address_space_GiB': 3.95, 'enforced_affinity': sorted(os.sched_getaffinity(0)),
        'performance_claim': False, 'new_expanded_model_or_experimental_certification': False})
    signal.alarm(0)
    print(json.dumps({'point_explanation_supported': bool(finding), 'physical': physical_result,
                      'actual': actual_result, 'comparisons': comparisons, 'gradient_calls': CALLS}, indent=2))


if __name__ == '__main__':
    try:
        run()
    except BaseException as error:
        save('failure.json', {'type': type(error).__name__, 'message': str(error),
                              'traceback': traceback.format_exc(), 'full_gradient_calls': CALLS,
                              'seconds': time.monotonic() - START, 'optimizer_calls': 0})
        raise
