"""Independent saved-accepted-point replay; never run the optimizer.

This script is prepared but must only be run after root authorizes the new
source epoch and a real public-resume checkpoint exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-contract', type=Path, required=True)
    args = parser.parse_args()
    contract = load(HERE / 'contract.json')
    approval = load(args.execution_contract)
    if (approval.get('execution_authorized_by_root') is not True
            or approval.get('independent_endpoint_replay_authorized') is not True
            or approval.get('verifier_sha256') != sha(Path(__file__))
            or approval.get('preparation_contract_sha256') != sha(HERE / 'contract.json')
            or approval.get('source_identity_sha256') != sha(HERE / 'source_identity.json')):
        raise ValueError('Independent endpoint replay lacks exact source-bound root contract')
    pointer = load(HERE / 'latest_checkpoint.json')
    checkpoint_path = HERE / pointer['path']
    if sha(checkpoint_path) != pointer['sha256']:
        raise ValueError('Last complete accepted checkpoint pointer changed')
    checkpoint = load(checkpoint_path)
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

    checks = []

    def check(name, error, gate):
        value = float(error)
        checks.append({'name': name, 'error': value, 'limit': gate,
                       'passed': bool(np.isfinite(value) and value <= gate)})

    source = load(HERE / 'source_identity.json')
    for path, expected in source['all_atomx_python_sha256'].items():
        check('frozen_new_source:' + path, sha(HERE / 'source' / Path(path).relative_to('src')) != expected, 0)
    for path, expected in contract['inputs_sha256'].items():
        check('immutable_input:' + path, sha(ROOT / path) != expected, 0)
    if not all(c['passed'] for c in checks):
        driver.atomic_json(HERE / 'independent_replay_input_failure.json', {'checks': checks, 'gradient_evaluations': 0})
        raise ValueError('Independent replay source/input identity failed before computation')
    native = ROOT / (PREFIX + 'native_run/nuclear_eol005_projected_increment')
    replay = ROOT / (PREFIX + 'balanced_cv_replay')
    archive = np.load(array_path)
    radial = np.load(native / 'orbitals_workspace.npz')
    data = {n[6:]: radial[n].item() if radial[n].ndim == 0 else radial[n]
            for n in radial.files if n.startswith('basis_')}
    waves = {tuple(k): (p.copy(), q.copy()) for k, p, q in zip(archive['orbital_keys'], archive['P'], archive['Q'], strict=True)}
    original_waves = {tuple(k): (p.copy(), q.copy()) for k, p, q in zip(radial['orbital_keys'], radial['P'], radial['Q'], strict=True)}
    enumeration = load(replay / 'basis_identity.json')
    csfs = [c for row in enumeration['configs'] for c in enumerate_jj_csfs(row['config'])]
    if len(csfs) != 531 or {c.n_electrons for c in csfs} != {69}:
        raise ValueError('Complete declared mixed basis changed')
    for c, r in zip(csfs, enumeration['CSFs'], strict=True):
        sig = csf_signature(c)
        check(f'signed_CSF_order:{r["index"]}', sig.total_J2 != r['twoJ'] or c.parity != r['parity']
              or driver.plain(sig.subshells) != r['subshells'] or driver.plain(sig.coupling) != r['coupling'], 0)
    targets = load(replay / 'stage_4f5p5s/target_binding.json')['targets']
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
    rows = load(ROOT / (PREFIX + 'native_run/nuclear_eol004_stable_delta_trace/failed_coordinate_layout.json'))
    old_metric = np.load(ROOT / (PREFIX + 'native_run/nuclear_eol004_stable_delta_trace/failed_coordinates.npz'))
    layout = tuple((tuple(tuple(k) for k in row['keys']), old_metric[f'projector_{i}'].copy(),
                    tuple(row['shape']), slice(*row['slice'])) for i, row in enumerate(rows))
    original_blocks = tuple(tuple(b.copy() for b in old_metric[f'inverse_blocks_{i}']) for i in range(len(rows)))
    endpoint_state = OrbitalOptimizationState.from_arrays(functional, layout=layout, coordinates=archive['x'],
        inverse_diagonal=old_metric['inverse_diagonal'], inverse_blocks=original_blocks,
        projector_source='nuclear', objective_identity=checkpoint['objective_sha256'], occupation_floor=.05)
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
    differences = []
    for label, upper in energies.items():
        ref = previous[label]
        transition = (upper - lower) * HARTREE_CM1
        delta_upper = upper - ref['baseline_energy_Ha']
        delta_lower = lower - previous['6s_1/2']['baseline_energy_Ha']
        delta_transition = transition - ref['baseline_transition_cm1']
        check(label + '/same_lower_upper_delta_cm1', abs(delta_transition
              - (delta_upper - delta_lower) * HARTREE_CM1), 1e-7)
        differences.append({'label': label, 'energy_Ha': upper, 'excitation_cm1': transition,
            'absolute_upper_delta_from_fixed531_Ha': delta_upper,
            'same_lower_delta_from_fixed531_Ha': delta_lower,
            'same_lower_upper_transition_delta_cm1': delta_transition,
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
