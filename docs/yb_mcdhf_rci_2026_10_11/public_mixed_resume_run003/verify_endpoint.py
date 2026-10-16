"""Independent saved-accepted-point replay; never run the optimizer.

This script is prepared but must only be run after root authorizes the new
source epoch and a real public-resume checkpoint exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = 'docs/yb_mcdhf_rci_2026_10_11/'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def checkpoint_bindings(checkpoint, *, source_sha256, contract_sha256, execution_sha256, inputs):
    """Bind the saved point itself, independently of a valid latest pointer."""
    return {
        'accepted_complete_point': checkpoint.get('accepted_complete_point') is True,
        'checkpoint_source_identity': checkpoint.get('source_identity_sha256') == source_sha256,
        'checkpoint_preparation_contract': checkpoint.get('preparation_contract_sha256') == contract_sha256,
        'checkpoint_execution_contract': checkpoint.get('execution_contract_sha256') == execution_sha256,
        'checkpoint_frozen_scientific_inputs': checkpoint.get('frozen_scientific_inputs_sha256') == inputs,
    }


def rebuilt_objective_payload(records, targets, *, radial_sha256, source_map_sha256,
                              reference_ci_sha256, reference_binding_sha256):
    """Rebuild declared physics from actual enumerated signed CSFs, not a saved hash."""
    return {
        'Z': 70, 'electrons': 69, 'operator': 'full declared no-pair Dirac-Coulomb',
        'nucleus': 'native005 Fermi A175', 'radial_archive_sha256': radial_sha256,
        'target_identity': [{'label': target['label'], 'twoJ': target['twoJ'],
                             'parity': target['parity'], 'root': target['models']['DC']['local_root'],
                             'weight': .2} for target in targets],
        'CSFs': records, 'source_map_sha256': source_map_sha256,
        'physical_reference_CI_sha256': reference_ci_sha256,
        'physical_reference_binding_sha256': reference_binding_sha256,
    }


def saved_eigenpair_checks(archive, gates, dimension):
    """Validate saved amplitudes against saved H, allowing rotations in degenerate spaces."""
    import numpy as np

    arrays = {name: archive[name] for name in ('matrix_Ha', 'energies_Ha', 'mixing')}
    shapes = {'matrix_Ha': (dimension, dimension), 'energies_Ha': (dimension,),
              'mixing': (dimension, dimension)}
    checks = []
    for name, array in arrays.items():
        valid = (array.shape == shapes[name] and array.dtype == np.dtype('float64')
                 and np.isfinite(array).all())
        checks.append({'name': 'saved_' + name + '/complete_finite_float64',
                       'error': int(not valid), 'limit': 0, 'passed': bool(valid)})
    if not all(check['passed'] for check in checks):
        return checks
    h, energies, mixing = arrays['matrix_Ha'], arrays['energies_Ha'], arrays['mixing']
    errors = {
        'saved_full_eigenpair_residual_Ha': (float(np.max(np.linalg.norm(
            h @ mixing - mixing * energies, axis=0))), gates['CI_residual_Ha']),
        'saved_full_signed_CI_Gram': (float(np.max(abs(mixing.T @ mixing
            - np.eye(dimension)))), gates['CI_orthogonality']),
        'saved_H_Hermiticity_Ha': (float(np.max(abs(h - h.T))), gates['matrix_hermiticity_Ha']),
    }
    checks.extend({'name': name, 'error': error, 'limit': gate,
                   'passed': bool(np.isfinite(error) and error <= gate)}
                  for name, (error, gate) in errors.items())
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-contract', type=Path, required=True)
    args = parser.parse_args()
    contract = load(HERE / 'contract.json')
    approval = load(args.execution_contract)
    if (approval.get('execution_authorized_by_root') is not True
            or approval.get('independent_endpoint_replay_authorized') is not True
            or approval.get('verifier_sha256') != sha(Path(__file__))
            or approval.get('driver_sha256') != sha(HERE / 'run.py')
            or approval.get('preparation_contract_sha256') != sha(HERE / 'contract.json')
            or approval.get('source_identity_sha256') != sha(HERE / 'source_identity.json')):
        raise ValueError('Independent endpoint replay lacks exact source-bound root contract')
    import run as prepared_driver

    prepared_driver.start_checkpoint_status(contract)
    pointer = load(HERE / 'latest_checkpoint.json')
    checkpoint_path = HERE / pointer['path']
    if sha(checkpoint_path) != pointer['sha256']:
        raise ValueError('Last complete accepted checkpoint pointer changed')
    checkpoint = load(checkpoint_path)
    bindings = checkpoint_bindings(checkpoint,
        source_sha256=sha(HERE / 'source_identity.json'),
        contract_sha256=sha(HERE / 'contract.json'), execution_sha256=sha(args.execution_contract),
        inputs=contract['inputs_sha256'])
    if not all(bindings.values()):
        raise ValueError('Accepted checkpoint provenance failed: ' + ', '.join(
            name for name, passed in bindings.items() if not passed))
    array_path = checkpoint_path.with_suffix('.npz')
    if sha(array_path) != checkpoint['array_sha256']:
        raise ValueError('Last complete accepted checkpoint arrays changed')
    caps, gates = contract['resources'], contract['acceptance']
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    # A separate one-gradient bounded replay needs its own actual admission.
    import run as driver

    admission = driver.host_resources(caps)
    if not admission['admitted']:
        raise RuntimeError('Independent replay host-resource guard failed')
    os.sched_setaffinity(0, {admission['affinity'][-1]})
    limit = int(caps['address_space_cap_GiB'] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    def timeout(*_):
        raise TimeoutError('Independent saved-point replay exceeded890s cap')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(890)
    started = time.monotonic()
    checks = []

    def check(name, error, gate):
        value = float(error)
        checks.append({'name': name, 'error': value, 'limit': gate,
                       'passed': bool(math.isfinite(value) and value <= gate)})

    for name, passed in bindings.items():
        check(name, not passed, 0)
    source = load(HERE / 'source_identity.json')
    actual_source_map = {'src/' + str(path.relative_to(HERE / 'source')): sha(path)
                         for path in sorted((HERE / 'source/atomx').rglob('*.py'))}
    check('complete_actual_frozen_source_map', actual_source_map != source['all_atomx_python_sha256'], 0)
    check('canonical_actual_frozen_source_map_SHA', canonical_sha(actual_source_map)
          != source['source_map_sha256'], 0)
    check('same_scientific_source_as_verified160_parent', canonical_sha(actual_source_map)
          != contract['scientific_source_map_required_sha256'], 0)
    for path, expected in source['all_atomx_python_sha256'].items():
        check('frozen_new_source:' + path, sha(HERE / 'source' / Path(path).relative_to('src')) != expected, 0)
    for path, expected in contract['inputs_sha256'].items():
        check('immutable_input:' + path, sha(ROOT / path) != expected, 0)
    if not all(c['passed'] for c in checks):
        driver.atomic_json(HERE / 'independent_replay_input_failure.json', {'checks': checks, 'gradient_evaluations': 0})
        raise ValueError('Independent replay source/input identity failed before computation')
    sys.path.insert(0, str(HERE / 'source'))
    import numpy as np

    from atomx.angular.csf_hamiltonian import csf_signature
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.constants import HARTREE_CM1
    from atomx.methods.ci import CIResult
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.numerics.bspline.orbital_optimization import OrbitalOptimizationState
    from atomx.physics.csf.levels import ci_level_blocks
    from atomx.physics.csf.matrix import build_ci_matrix
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.physics.csf.tracking import CIBasisTracking
    from atomx.potentials.nuclear import make_nuclear_potential

    native = ROOT / (PREFIX + 'native_run/nuclear_eol005_projected_increment')
    replay = ROOT / (PREFIX + 'balanced_cv_replay')
    archive = np.load(array_path)
    checks.extend(saved_eigenpair_checks(archive, gates, 531))
    if not all(c['passed'] for c in checks):
        driver.atomic_json(HERE / 'independent_replay_archive_failure.json',
                           {'checks': checks, 'gradient_evaluations': 0})
        raise ValueError('Saved complete eigenpairs failed before fresh CI/gradient evaluation')
    radial = np.load(native / 'orbitals_workspace.npz')
    data = {n[6:]: radial[n].item() if radial[n].ndim == 0 else radial[n]
            for n in radial.files if n.startswith('basis_')}
    waves = {tuple(k): (p.copy(), q.copy()) for k, p, q in zip(archive['orbital_keys'], archive['P'], archive['Q'], strict=True)}
    original_waves = {tuple(k): (p.copy(), q.copy()) for k, p, q in zip(radial['orbital_keys'], radial['P'], radial['Q'], strict=True)}
    enumeration = load(replay / 'basis_identity.json')
    csfs, actual_records = [], []
    for config_index, row in enumerate(enumeration['configs']):
        for csf in enumerate_jj_csfs(row['config']):
            signature = csf_signature(csf)
            actual_records.append({'index': len(csfs), 'config_index': config_index, 'family': row['family'],
                'label': csf.label, 'twoJ': csf.total_J2, 'parity': csf.parity,
                'subshells': [list(s) for s in signature.subshells],
                'coupling': [list(c) for c in signature.coupling],
                'canonical_pattern': [list(p) for p in signature.pattern]})
            csfs.append(csf)
    if len(csfs) != 531 or {c.n_electrons for c in csfs} != {69}:
        raise ValueError('Complete declared mixed basis changed')
    for actual, saved in zip(actual_records, enumeration['CSFs'], strict=True):
        check(f'signed_complete_CSF_order:{actual["index"]}', actual != saved, 0)
    targets = load(replay / 'stage_4f5p5s/target_binding.json')['targets']
    rebuilt_payload = rebuilt_objective_payload(actual_records, targets,
        radial_sha256=sha(native / 'orbitals_workspace.npz'), source_map_sha256=canonical_sha(actual_source_map),
        reference_ci_sha256=sha(replay / 'stage_4f5p5s/DC_arrays.npz'),
        reference_binding_sha256=sha(replay / 'stage_4f5p5s/target_binding.json'))
    rebuilt_objective_sha256 = canonical_sha(rebuilt_payload)
    saved_objective = load(HERE / 'objective_identity.json')
    check('independently_rebuilt_objective_payload', rebuilt_payload != saved_objective['new'], 0)
    check('independently_rebuilt_objective_SHA', rebuilt_objective_sha256 != saved_objective['new_sha256'], 0)
    check('saved_objective_payload_canonical_SHA', canonical_sha(saved_objective['new'])
          != saved_objective['new_sha256'], 0)
    check('checkpoint_actual_objective_SHA', checkpoint['objective_sha256'] != rebuilt_objective_sha256, 0)
    check('checkpoint_diagnostics_actual_objective_SHA', checkpoint['diagnostics']['objective_identity']
          != rebuilt_objective_sha256, 0)
    if not all(c['passed'] for c in checks):
        driver.atomic_json(HERE / 'independent_replay_objective_failure.json',
                           {'checks': checks, 'gradient_evaluations': 0})
        raise ValueError('Rebuilt actual objective identity failed before fresh CI/gradient evaluation')
    selectors = tuple((t['twoJ'], t['parity'], t['models']['DC']['local_root'], 1.) for t in targets)
    average = CIStateAverage(selectors)
    nuclear = make_nuclear_potential(70, 'fermi', A=175)
    provisional = CSFOrbitalFunctional(csfs, data, 70, nuclear, state_average=average, max_determinants=2000)
    baseline = np.load(replay / 'stage_4f5p5s/DC_arrays.npz')
    tracking = CIBasisTracking(average, csfs, CIResult(baseline['energies_Ha'], baseline['mixing']),
                              original_waves, provisional.metric, max_determinant_pairs=1_000_000)
    functional = CSFOrbitalFunctional(csfs, data, 70, nuclear, state_average=average,
                                      level_tracking=tracking, cache=provisional.cache, max_determinants=2000)
    # A new full CI solve and one true full radial gradient, not saved diagnostics.
    energy, gradient, ci = functional.evaluate(waves)
    h = build_ci_matrix(csfs, waves, functional.keys, data, 70, 123, nuclear_potential=nuclear,
                        convol_cache=functional.cache, max_determinants=2000)
    check('complete_H_replay_Ha', np.max(abs(h - archive['matrix_Ha'])), gates['baseline_matrix_replay_Ha'])
    check('all_root_energy_replay_Ha', np.max(abs(ci.state_energies - archive['energies_Ha'])), gates['baseline_target_energy_replay_Ha'])
    check('full_CI_residual_Ha', np.max(np.linalg.norm(h @ ci.mixing_coefficients
          - ci.mixing_coefficients * ci.state_energies, axis=0)), gates['CI_residual_Ha'])
    check('full_signed_CI_Gram', np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients - np.eye(531))), gates['CI_orthogonality'])
    check('H_Hermiticity_Ha', np.max(abs(h - h.T)), gates['matrix_hermiticity_Ha'])
    symmetry = [(c.total_J2, c.parity) for c in csfs]
    cross = np.array([[a != b for b in symmetry] for a in symmetry])
    check('cross_symmetry_zero_Ha', np.max(abs(h[cross])), gates['cross_symmetry_zero_Ha'])
    weights, identities = tracking.resolve(csfs, ci, waves, functional.metric)
    check('five_equal_fixed_physical_targets', np.count_nonzero(weights) != 5
          or not np.array_equal(weights[weights > 0], np.repeat(.2, 5)), 0)
    check('fresh_saved_samepoint_complete_physical_target_groups',
          driver.canonical_physical_target_groups(identities)
          != driver.canonical_physical_target_groups(checkpoint['physical_tracking']), 0)
    rows = load(ROOT / (PREFIX + 'native_run/nuclear_eol004_stable_delta_trace/failed_coordinate_layout.json'))
    old_metric = np.load(ROOT / (PREFIX + 'native_run/nuclear_eol004_stable_delta_trace/failed_coordinates.npz'))
    layout = tuple((tuple(tuple(k) for k in row['keys']), old_metric[f'projector_{i}'].copy(),
                    tuple(row['shape']), slice(*row['slice'])) for i, row in enumerate(rows))
    original_blocks = tuple(tuple(b.copy() for b in old_metric[f'inverse_blocks_{i}']) for i in range(len(rows)))
    endpoint_state = OrbitalOptimizationState.from_arrays(functional, layout=layout, coordinates=archive['x'],
        inverse_diagonal=old_metric['inverse_diagonal'], inverse_blocks=original_blocks,
        projector_source='nuclear', objective_identity=rebuilt_objective_sha256, occupation_floor=.05)
    check('public_geometry_fingerprint_replay', endpoint_state.geometry_sha256 != checkpoint['geometry_sha256'], 0)
    metric_digest = hashlib.sha256()
    x, g, preconditioned = archive['x'], np.zeros(2856), np.zeros(2856)
    for i, row in enumerate(rows):
        keys, shape, where = [tuple(k) for k in row['keys']], tuple(row['shape']), slice(*row['slice'])
        projector = old_metric[f'projector_{i}']
        metric_digest.update(repr((keys, shape)).encode())
        metric_digest.update(np.ascontiguousarray(projector).tobytes())
        check(f'unchanged_original_projector:{i}', np.max(abs(projector - archive[f'projector_{i}'])), 0)
        c = np.column_stack([np.concatenate(waves[k]) for k in keys])
        check(f'authoritative_x_unpack:{i}', np.max(abs(c - projector @ x[where].reshape(shape))), 0)
        s = functional.metric[keys[0][1]]
        check(f'original_S_Gram:{i}', np.max(abs(c.T @ s @ c - np.eye(len(keys)))), gates['radial_Gram_error'])
        outside = c - projector @ (projector.T @ s @ c)
        check(f'original_no_pair_leakage:{i}', np.sqrt(max(0., np.max(np.sum(outside * (s @ outside), axis=0)))), gates['projector_leakage_norm'])
        y = x[where].reshape(shape)
        block = projector.T @ np.column_stack([gradient[k] for k in keys])
        multiplier = y.T @ block
        g[where] = (block - y @ (.5 * (multiplier + multiplier.T))).ravel()
    metric_digest.update(old_metric['inverse_diagonal'].tobytes())
    check('unchanged_original_inverse_diagonal', np.max(abs(archive['inverse_diagonal'] - old_metric['inverse_diagonal'])), 0)
    for i, row in enumerate(rows):
        shape, where = tuple(row['shape']), slice(*row['slice'])
        blocks = old_metric[f'inverse_blocks_{i}']
        check(f'unchanged_original_inverse_blocks:{i}', np.max(abs(archive[f'inverse_blocks_{i}'] - blocks)), 0)
        for block in blocks:
            metric_digest.update(np.ascontiguousarray(block).tobytes())
        y, b = x[where].reshape(shape), g[where].reshape(shape)
        z = np.column_stack([p @ b[:, col] for col, p in enumerate(blocks)])
        multiplier = y.T @ z
        preconditioned[where] = (z - y @ (.5 * (multiplier + multiplier.T))).ravel()
    expected_metric = contract['fixed_conditions']['original_projector_and_preconditioner_metric_sha256']
    check('complete_original_metric_SHA', metric_digest.hexdigest() != expected_metric
          or checkpoint['original_metric_sha256'] != expected_metric, 0)
    check('saved_samepoint_complete_projected_gradient', np.max(abs(g - archive['original_projected_gradient'])),
          gates['full531_same_start_projected_gradient_replay_max'])
    check('saved_samepoint_complete_preconditioned_gradient',
          np.max(abs(preconditioned - archive['original_preconditioned_gradient'])),
          gates['full531_same_start_preconditioned_gradient_replay_max'])
    orbital_residual = float(np.linalg.norm(preconditioned))
    energy_indicator = .5 * abs(float(g @ preconditioned))
    check('reported_residual_independent_replay', abs(orbital_residual - checkpoint['diagnostics']['orbital_residual']),
          1e-10 + 1e-6 * orbital_residual)
    check('reported_indicator_independent_replay', abs(energy_indicator - checkpoint['diagnostics']['estimated_energy_error']),
          1e-12 + 1e-6 * energy_indicator)
    check('reported_EO_energy_independent_replay_Ha', abs(energy - checkpoint['diagnostics']['energy_ha']), 5e-9)
    _, columns = ci_level_blocks(csfs, ci, require_full=True)
    energies = {}
    for group in identities:
        for j, parity, local_root in group['selected_levels']:
            column = columns[j, parity][local_root]
            label = next(t['label'] for t in targets if t['twoJ'] == j and t['parity'] == parity)
            energies[label] = float(ci.state_energies[column])
    lower = energies['6s_1/2']
    previous = {r['label']: r for r in load(ROOT / (PREFIX + 'mixed_ci_stationarity_probe/card_evidence.json'))['rows_diagnostic_DC']}
    previous160 = {r['label']: r for r in load(ROOT / contract['start_checkpoint']['independent_verification'])['DC_lower_upper_rows']}
    differences = []
    for label, upper in energies.items():
        ref = previous[label]
        transition = (upper - lower) * HARTREE_CM1
        delta_upper = upper - ref['baseline_energy_Ha']
        delta_lower = lower - previous['6s_1/2']['baseline_energy_Ha']
        delta_transition = transition - ref['baseline_transition_cm1']
        check(label + '/same_lower_upper_delta_cm1', abs(delta_transition
              - (delta_upper - delta_lower) * HARTREE_CM1), 1e-7)
        previous_upper_change = upper - previous160[label]['energy_Ha']
        previous_lower_change = lower - previous160['6s_1/2']['energy_Ha']
        previous_transition_change = transition - previous160[label]['excitation_cm1']
        check(label + '/same_parent160_lower_upper_delta_cm1', abs(previous_transition_change
              - (previous_upper_change - previous_lower_change) * HARTREE_CM1), 1e-7)
        differences.append({'label': label, 'energy_Ha': upper, 'excitation_cm1': transition,
            'absolute_upper_delta_from_fixed531_Ha': delta_upper,
            'same_lower_delta_from_fixed531_Ha': delta_lower,
            'same_lower_upper_transition_delta_cm1': delta_transition,
            'absolute_upper_delta_from_parent160_Ha': previous_upper_change,
            'same_lower_delta_from_parent160_Ha': previous_lower_change,
            'same_lower_upper_transition_delta_from_parent160_cm1': previous_transition_change,
            'NIST_nominal_cm1': ref['NIST_nominal_cm1'], 'nominal_gap_cm1': transition - ref['NIST_nominal_cm1'],
            'nominal_1percent_pass': None if label == '6s_1/2'
            else abs((transition - ref['NIST_nominal_cm1']) / ref['NIST_nominal_cm1']) < .01})
    integrity = all(c['passed'] for c in checks)
    stationarity = orbital_residual < 1e-7 and energy_indicator < 1e-8
    result = {'all_identity_replay_checks_passed': integrity, 'checks_count': len(checks), 'checks': checks,
              'actual_original_metric_orbital_residual': orbital_residual,
              'actual_original_metric_energy_indicator_Ha': energy_indicator,
              'original_stationarity_gates_passed': stationarity,
              'declared531_DC_numeric_stationarity_certified': integrity and stationarity,
              'production_certified': False, 'highest_level_reached': False, 'experimental_gate_passed': False,
              'static_Breit_spectrum_changed': False, 'physical_tracking': identities, 'DC_lower_upper_rows': differences,
              'new_source_identity_sha256': sha(HERE / 'source_identity.json'),
              'preparation_contract_sha256': sha(HERE / 'contract.json'),
              'execution_contract_sha256': sha(args.execution_contract),
              'objective_identity_file_sha256': sha(HERE / 'objective_identity.json'),
              'independently_rebuilt_objective_sha256': rebuilt_objective_sha256,
              'independently_rebuilt_objective_payload': rebuilt_payload,
              'starting_parent160_reference_sha256': {p: contract['inputs_sha256'][p] for p in
                  contract['start_checkpoint'].values() if isinstance(p, str) and p in contract['inputs_sha256']},
              'saved_eigenpairs_checked_without_saved_column_equality': True,
              'accepted_checkpoint_sha256': sha(checkpoint_path), 'accepted_arrays_sha256': sha(array_path),
              'verifier_sha256': sha(Path(__file__)), 'resource_admission': admission,
              'seconds_uncontrolled': time.monotonic() - started,
              'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              'new_full_gradient_evaluations': 1, 'optimizer_calls': 0,
              'timing_is_not_controlled_performance': True}
    driver.atomic_json(HERE / 'independent_endpoint_verification.json', result)
    signal.alarm(0)
    print(json.dumps({'integrity_passed': integrity, 'stationarity_passed': stationarity,
                      'residual': orbital_residual, 'energy_indicator': energy_indicator}))
    if not integrity:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
