"""Independent failed-screened saved-array algebra; no Atomx/spectral calls."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    contract = load(HERE / 'contract.json')
    for name, expected in contract['bound_inputs_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Saved failed input changed: ' + name)
    if (HERE / 'verification.json').exists():
        raise FileExistsError('Preserve completed failure audit')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 3:
        raise RuntimeError('Require >=3original affinity CPUs')
    os.sched_setaffinity(0, {affinity[-2]})
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))

    def timeout(*_):
        raise TimeoutError('90s saved failure audit cap')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(90)
    import numpy as np

    checks, scalar_rows, root_rows = [], [], []

    def check(name, error, limit=0):
        error = float(error)
        passed = bool(np.isfinite(error) and error <= limit)
        checks.append(dict(name=name, error=error, limit=limit, passed=passed))
        if not passed:
            raise ValueError('Independent saved failure audit: ' + name)

    try:
        invocation = load(RUN / 'contract.json')
        failure = load(RUN / 'failure.json')
        closure = load(RUN / 'root_execution.json')
        check('actual_failed_foreground_exit', closure['exit_code'] == 0 or closure['outer_wall_cap_stopped'])
        check('source_epoch_unchanged', failure['source_map_sha256'] != invocation['fixed_conditions']['source_map_sha256'])
        check('exact_frozen_driver_contract', failure['driver_sha256'] != sha(RUN / 'run.py')
              or failure['contract_sha256'] != sha(RUN / 'contract.json'))
        check('exact_root_binding', failure['execution_binding_sha256'] != sha(RUN / 'execution_binding.json'))
        check('source353_science_only', invocation['fixed_conditions']['full_atomx_source_count'] != 353)
        source = load(ROOT / invocation['input_roles']['source_identity']['path'])
        source_root = ROOT / invocation['fixed_conditions']['frozen_source_root']
        for name, expected in source['all_atomx_python_sha256'].items():
            check('whole353/' + name, sha(source_root / 'atomx' / name.removeprefix('src/atomx/')) != expected)
        failed = [r for r in failure['checks'] if not r['passed']]
        check('one_original_failed_gate', len(failed) != 1 or failed[0]['name'] != contract['original_gate']['name'])
        check('original_numeric_failure_preserved', failure['scientific_success_claim'] is not False
              or failed[0]['limit'] != 1e-8 or failed[0]['error'] <= 1e-8)
        expected_counts = dict(initial_one_body_matrix_attempts=7, completed_initial_one_body_matrices=7,
                               auxiliary_JK_builder_attempts=10, completed_auxiliary_JK_builders=10,
                               auxiliary_angular_attempts=2, completed_auxiliary_angular_precomputations=2,
                               closed_core_scalar_attempts=1, completed_closed_core_scalar=1,
                               scalar_radial_callback_attempts=3584, scalar_Slater_attempts=3584,
                               completed_scalar_Slater=3584, virtual_control_hS_consumed=1,
                               complement_SVD_attempts=1, completed_complement_SVD=1,
                               full_gvd_attempts=1, completed_full_gvd=1)
        for name, count in failure['calls'].items():
            check('actual_call/' + name, count != expected_counts.get(name, 0))
        check('no_CI_response_artifacts', any((RUN / n).exists() for n in
              ('result.json', 'baseline_CI.npz', 'balanced_virtual_CI.npz', 'baseline_H.npz', 'balanced_virtual_H.npz', 'canonical29_orbitals.npz')))
        for row in failure['completed_evidence']:
            check('saved_stage/' + row['path'], sha(RUN / row['path']) != row['sha256'])
        roles = {key: ROOT / row['path'] for key, row in invocation['input_roles'].items()}
        checkpoint = np.load(roles['checkpoint_arrays'], allow_pickle=False)
        point = load(roles['checkpoint_manifest'])
        geometry = np.load(RUN / 'native_hS_original_geometry.npz', allow_pickle=False)
        old_keys = list(map(tuple, checkpoint['orbital_keys']))
        layout = load(roles['original_layout'])
        for name, original in (('old24_keys', 'orbital_keys'), ('old24_P', 'P'), ('old24_Q', 'Q'),
                               ('original_x', 'x'), ('inverse_diagonal', 'inverse_diagonal')):
            check('original_metric_point_bytes/' + name, not np.array_equal(geometry[name], checkpoint[original]))
        for i in range(len(layout)):
            for name in ('projector_', 'inverse_blocks_'):
                check('original_layout_bytes/' + name + str(i), not np.array_equal(geometry[name + str(i)], checkpoint[name + str(i)]))
        check('original_stationarity_fail_preserved', point['diagnostics']['orbital_residual'] <= 1e-7)
        field_inputs = np.load(RUN / 'auxiliary_field_inputs.npz', allow_pickle=False)
        ordered = list(map(tuple, field_inputs['ordered_keys']))
        core = {tuple(row['key']): row['occupation'] for row in invocation['fixed_conditions']['new_virtuals']['closed_core']}
        new_keys = [tuple(row) for row in invocation['fixed_conditions']['new_virtuals']['new_keys']]
        check('closed19_68', len(core) != 19 or sum(core.values()) != 68 or any(q != 2 * abs(k[1]) for k, q in core.items()))
        check('core_target_order', ordered != sorted(set(core) | set(new_keys)))
        c = np.array([[core.get(key, 0) for key in ordered] for _ in ordered], dtype=float)
        check('zero_target_explicit_C_D', not np.array_equal(c, field_inputs['raw_coefficients'])
              or not np.array_equal(field_inputs['normalization'], np.ones(24)))
        check('scale3_C_D', not np.array_equal(field_inputs['scaled_coefficients'], 3 * c)
              or not np.array_equal(field_inputs['scaled_normalization'], np.repeat(3., 24)))
        for i, key in enumerate(ordered):
            expected = (checkpoint['P'][old_keys.index(key)], checkpoint['Q'][old_keys.index(key)]) if key in core else (np.zeros(123), np.zeros(123))
            check('actual_closed_source_PQ/' + str(key), not (np.array_equal(field_inputs['source_P'][i], expected[0])
                  and np.array_equal(field_inputs['source_Q'][i], expected[1])))
        angular = np.load(RUN / 'auxiliary_angular_coefficients.npz', allow_pickle=False)
        for name in ('direct', 'exchange'):
            check('angular_row_scale/' + name, np.max(abs(angular['scaled_' + name] - 3 * angular['primary_' + name])), 1e-12)
        scalar_report = load(RUN / 'auxiliary_scalar_control.json')
        check('scalar80_inputs', scalar_report['exact_ordered_bilinears'] != 80
              or scalar_report['actual_public_Slater_calls'] != 3584 or not scalar_report['zero_bare_one_body'])
        raw = np.load(RUN.parent / 'rank_vs_virtual_discriminator/raw_native_virtuals.npz', allow_pickle=False)
        jk_fields = {}
        for j, key in enumerate(new_keys):
            fields = np.load(RUN / ('virtual_' + str(j) + '_auxiliary_Fock.npz'), allow_pickle=False)
            pp, qq, pq, qp = [fields[n] for n in ('pp', 'qq', 'pq', 'qp')]
            jk = np.block([[pp, pq], [qp, qq]])
            check('full_native_JK_assembly/' + str(key), not np.array_equal(jk, fields['JK_Ha']))
            nuclear = geometry['h_' + str(key[1])]
            check('nuclear_once/' + str(key), not np.array_equal(nuclear, fields['nuclear_h_Ha'])
                  or not np.array_equal(fields['auxiliary_Fock_Ha'], nuclear + jk))
            check('JK_quadrants_Hermitian/' + str(key), np.max(abs(jk - jk.T)), 1e-10)
            for name in ('pp', 'qq', 'pq', 'qp'):
                check('scaled_field/' + name + '/' + str(key), np.max(abs(fields[name] - fields['scaled_' + name])), 5e-9)
            jk_fields[key[1]] = jk
        valence = {-3: (5, -3), 2: (5, 2), -2: (6, -2), -1: (6, -1), 1: (6, 1)}
        for kappa, jk in sorted(jk_fields.items()):
            a = np.load(RUN / ('auxiliary_scalar_quadrants_' + str(kappa) + '.npz'), allow_pickle=False)
            keys = list(map(tuple, a['probe_keys']))
            v = a['probe_vectors']
            new_j = next(i for i, key in enumerate(new_keys) if key[1] == kappa)
            old_j = old_keys.index(valence[kappa])
            original_vectors = (np.concatenate((checkpoint['P'][old_j], np.zeros(123))),
                                np.concatenate((np.zeros(123), checkpoint['Q'][old_j])),
                                np.concatenate((raw['P'][new_j], np.zeros(123))),
                                np.concatenate((np.zeros(123), raw['Q'][new_j])))
            check('unscaled_signed_component_probes/' + str(kappa), not np.array_equal(v, np.column_stack(original_vectors)))
            check('probes_disjoint_physical_core/' + str(kappa), bool(set(keys) & set(core)))
            dense = v.T @ jk @ v
            check('whole16_scalar_bilinear_replay/' + str(kappa), np.max(abs(dense - a['dense_native_JK_Ha'])), 1e-14)
            scalar = a['scalar_closed_core_JK_Ha']
            for quadrant, rr, cc in (('PP', [0, 2], [0, 2]), ('QQ', [1, 3], [1, 3]),
                                    ('PQ', [0, 2], [1, 3]), ('QP', [1, 3], [0, 2])):
                error = float(np.max(abs(dense[np.ix_(rr, cc)] - scalar[np.ix_(rr, cc)])))
                magnitude = float(np.max(abs(scalar[np.ix_(rr, cc)])))
                report = next(r for r in scalar_report['rows'] if r['kappa'] == kappa and r['quadrant'] == quadrant)
                check('saved_scalar_error_replay/' + quadrant + '/' + str(kappa), abs(error - report['error_Ha']), 1e-14)
                check('saved_scalar_magnitude_replay/' + quadrant + '/' + str(kappa), abs(magnitude - report['magnitude_Ha']), 1e-14)
                check('saved_scalar_absolute/' + quadrant + '/' + str(kappa), error, 5e-9)
                if quadrant in ('PQ', 'QP'):
                    check('nonzero_cross_anchor/' + quadrant + '/' + str(kappa), magnitude <= 1e-7)
                    check('cross_relative/' + quadrant + '/' + str(kappa), error / magnitude, 1e-7)
                scalar_rows.append(dict(kappa=kappa, quadrant=quadrant, error_Ha=error, magnitude_Ha=magnitude))
        report = load(RUN / 'virtual_construction_controls.json')[0]
        a = np.load(RUN / report['path'], allow_pickle=False)
        pre = np.load(RUN / 'virtual_0_native_complement.npz', allow_pickle=False)
        gvd = np.load(RUN / 'virtual_0_full_gvd_vectors.npz', allow_pickle=False)
        key = tuple(report['key'])
        check('only_attempted_minus3', key != new_keys[0] or len(load(RUN / 'virtual_construction_controls.json')) != 1)
        h, s, b = [a[n] for n in ('h', 's', 'basis')]
        check('actual_auxiliary_operator_identity', not np.array_equal(h, np.load(RUN / 'virtual_0_auxiliary_Fock.npz')['auxiliary_Fock_Ha']))
        check('actual_nuclear_metric_identity', not np.array_equal(s, geometry['s_-3']))
        for n in ('original_P', 'frozen', 'complement', 'basis', 'projected_h', 'projected_s'):
            check('preflight_before_spectrum_bytes/' + n, not np.array_equal(a[n], pre[n]))
        check('gvd_raw_coefficients_unchanged', not np.array_equal(a['full_coefficients'], gvd['full_coefficients']))
        p = a['original_P']
        same_kappa = [i for i, k in enumerate(old_keys) if k[1] == -3]
        frozen = np.column_stack([np.concatenate((checkpoint['P'][i], checkpoint['Q'][i])) for i in same_kappa])
        check('frozen_old_wave_bytes', not np.array_equal(frozen, a['frozen']))
        row = next(i for i, r in enumerate(layout) if r['keys'][0][1] == -3)
        check('original_P_bytes', not np.array_equal(p, checkpoint['projector_' + str(row)]))
        check('complement_map', np.max(abs(b - p @ a['complement'])), 1e-14)
        hld, sld, bld = [x.astype(np.longdouble) for x in (h, s, b)]
        cld = a['full_coefficients'].astype(np.longdouble)
        vld = a['normalized_physical_vectors'].astype(np.longdouble)
        hv, sv = hld @ vld, sld @ vld
        norms = np.sum(vld * sv, axis=0)
        rq = np.sum(vld * hv, axis=0) / norms
        constrained = bld.T @ (hv - sv * rq)
        absolute = np.sqrt(np.sum(constrained**2, axis=0))
        relative = absolute / np.maximum(1., np.maximum(abs(rq), np.sqrt(np.sum((bld.T @ hv)**2, axis=0))))
        check('saved_full_Rayleigh_replay', np.max(abs(rq - a['actual_Rayleigh_Ha'])), 1e-9)
        check('saved_relative_residual_replay', np.max(abs(relative - a['full_constrained_residual_relative'])), 1e-12)
        check('saved_whole_Gram_replay', np.max(abs(vld.T @ sv - a['actual_full_SGram'])), 1e-12)
        check('whole_Gram_original_gate', np.max(abs(vld.T @ sv - np.eye(len(rq)))), 1e-10)
        check('failed_max_relative_reproduced', abs(float(np.max(relative)) - failed[0]['error']), 1e-12)
        check('failed_max_remains_above_original_gate', np.max(relative) <= 1e-8)
        projection_ld, metric_ld = bld.T @ hld @ bld, bld.T @ sld @ bld
        stored_h, stored_s = a['projected_h'].astype(np.longdouble), a['projected_s'].astype(np.longdouble)
        delta_h, delta_s = projection_ld - stored_h, metric_ld - stored_s
        beta = a['normalization_factors'].astype(np.longdouble)
        coefficients = cld / beta
        exactly_mapped = bld @ coefficients
        map_error = vld - exactly_mapped
        lower_h = np.tril(stored_h) + np.tril(stored_h, -1).T
        solver_term = lower_h @ coefficients - (stored_s @ coefficients) * rq
        projection_term = (projection_ld - lower_h) @ coefficients - (delta_s @ coefficients) * rq
        mapping_term = bld.T @ (hld @ map_error - (sld @ map_error) * rq)
        decomposition = solver_term + projection_term + mapping_term
        check('finite_residual_decomposition_closure_Ha', np.max(abs(decomposition - constrained)), 1e-9)
        coordinate_norms = np.sum(cld * (stored_s @ cld), axis=0)
        coordinate_rq = np.sum(cld * (stored_h @ cld), axis=0) / coordinate_norms
        coordinate_res = stored_h @ cld - (stored_s @ cld) * coordinate_rq
        lower_rq = np.sum(cld * (lower_h @ cld), axis=0) / coordinate_norms
        lower_res = lower_h @ cld - (stored_s @ cld) * lower_rq
        lower_reported_res = lower_h @ cld - (stored_s @ cld) * a['reported_energies_Ha']
        mapped_norms = np.sum(exactly_mapped * (sld @ exactly_mapped), axis=0)
        mapped_rq = np.sum(exactly_mapped * (hld @ exactly_mapped), axis=0) / mapped_norms
        mapped_res = bld.T @ (hld @ exactly_mapped - (sld @ exactly_mapped) * mapped_rq)
        for i in range(len(rq)):
            root_rows.append(dict(root=i, reported_Ha=float(a['reported_energies_Ha'][i]), physical_Rayleigh_Ha=float(rq[i]),
                  lower_triangle_Rayleigh_Ha=float(lower_rq[i]),
                  lower_triangle_vector_residual_absolute_Ha=float(np.sqrt(np.sum(lower_res[:, i]**2))),
                  lower_triangle_reported_residual_absolute_Ha=float(np.sqrt(np.sum(lower_reported_res[:, i]**2))),
                  coordinate_Rayleigh_Ha=float(coordinate_rq[i]), longdouble_mapped_Rayleigh_Ha=float(mapped_rq[i]),
                  physical_constrained_absolute_Ha=float(absolute[i]), physical_constrained_relative=float(relative[i]),
                  original_relative_gate_passed=bool(relative[i] <= 1e-8),
                  coordinate_residual_absolute_Ha=float(np.sqrt(np.sum(coordinate_res[:, i]**2))),
                  longdouble_map_residual_absolute_Ha=float(np.sqrt(np.sum(mapped_res[:, i]**2))),
                  stored_solver_term_norm_Ha=float(np.sqrt(np.sum(solver_term[:, i]**2))),
                  projection_rounding_term_norm_Ha=float(np.sqrt(np.sum(projection_term[:, i]**2))),
                  physical_mapping_term_norm_Ha=float(np.sqrt(np.sum(mapping_term[:, i]**2)))))
        s_distance = float(np.max(np.sum(abs(stored_s - np.eye(len(stored_s))), axis=1)))
        condition_bound = (1 + s_distance) / (1 - s_distance) if s_distance < 1 else None
        norms_diagnostic = dict(projected_H_inf_Ha=float(np.max(np.sum(abs(stored_h), axis=1))),
              projected_H_asymmetry_Ha=float(np.max(abs(stored_h - stored_h.T))),
              longdouble_projected_H_asymmetry_Ha=float(np.max(abs(projection_ld - projection_ld.T))),
              projected_H_float_vs_longdouble_max_Ha=float(np.max(abs(delta_h))),
              projected_H_float_vs_longdouble_inf_Ha=float(np.max(np.sum(abs(delta_h), axis=1))),
              projected_lower_vs_longdouble_max_Ha=float(np.max(abs(projection_ld - lower_h))),
              projected_lower_vs_longdouble_inf_Ha=float(np.max(np.sum(abs(projection_ld - lower_h), axis=1))),
              saved_lowest_actual_Rayleigh_gap_Ha=float(rq[1] - rq[0]),
              projected_S_distance_from_I_inf=s_distance, projected_S_condition_inf_upper_bound=condition_bound,
              float64_epsilon_times_H_inf_diagnostic_Ha=float(np.finfo(float).eps * np.max(np.sum(abs(stored_h), axis=1))),
              original_map_float_vs_longdouble_max=float(np.max(abs(map_error))),
              maximum_relative_failed_root=int(np.argmax(relative)),
              full_root_count=len(rq), original_relative_failed_roots=[i for i, r in enumerate(relative) if r > 1e-8],
              reported_eigenvalues_are_not_verified_bounds=True,
              epsilon_times_norm_is_not_rigorous_error_bound=True)
        with (HERE / 'diagnostic_arrays.npz').open('xb') as stream:
            np.savez_compressed(stream, longdouble_projected_h=projection_ld, longdouble_projected_s=metric_ld,
                 effective_lower_projected_h=lower_h, lower_triangle_residual=lower_res,
                 lower_triangle_reported_residual=lower_reported_res, stored_operator_residual_term=solver_term, projection_rounding_term=projection_term,
                 physical_mapping_term=mapping_term, actual_constrained_residual=constrained,
                 coordinate_residual=coordinate_res, longdouble_mapped_residual=mapped_res,
                 actual_Rayleigh_Ha=rq, coordinate_Rayleigh_Ha=coordinate_rq, longdouble_mapped_Rayleigh_Ha=mapped_rq)
        check('no_Atomx_import', any(n == 'atomx' or n.startswith('atomx.') for n in sys.modules))
        proof = dict(all_independent_saved_evidence_checks_passed=True, checks_count=len(checks), checks=checks,
              original_scientific_invocation_passed=False, original_failed_gate_remains_failed=True,
              new_H_CI_Dirac_virtual_gradient_optimizer_eigensolves=0, Atomx_imports=0,
              contract_sha256=sha(HERE / 'contract.json'), audit_driver_sha256=sha(__file__),
              bound_inputs_sha256=contract['bound_inputs_sha256'], rootwise_diagnostics=root_rows,
              scalar_saved_quadrant_controls=scalar_rows, projection_and_conditioning=norms_diagnostic,
              diagnostic_arrays_sha256=sha(HERE / 'diagnostic_arrays.npz'),
              original_stationarity_passed=False, experimental_gate_passed=False,
              screened_CI_transition_or_RMS_response_available=False,
              scope='Saved valid fields/scalar anchors plus preserved failed first virtual full-spectrum residual; no new solves/operators/CI. No screened physical spectral response or operator-completeness claim.')
        (HERE / 'verification.json').write_text(json.dumps(proof, indent=2) + '\n')
        print(json.dumps(dict(checks=len(checks), all_saved_evidence_checks_passed=True,
                             original_invocation_passed=False, diagnostics=norms_diagnostic, lowest=root_rows[0])))
    except Exception as error:
        (HERE / 'audit_failure.json').write_text(json.dumps(dict(error=repr(error), checks=checks,
              contract_sha256=sha(HERE / 'contract.json'), audit_driver_sha256=sha(__file__),
              new_scientific_calls=0), indent=2) + '\n')
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
