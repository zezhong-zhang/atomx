"""Independent stdlib scalar, preservation and renderer-seam checks; no browser."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    c = load(HERE / 'contract.json')
    site = Path(c['site'])
    before, after = load(HERE / 'original_overlay.json'), load(site / c['overlay_path'])
    old = load(HERE / 'old_models.json')
    rows = []

    def check(name, condition):
        rows.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)

    check('exact_old21_models84rows', len(old) == 21 and sum(len(m['rows']) for m in old) == 84)
    check('all_old_overlay_models_preserved', after['models'][:-2] == before['models'])
    check('all_old_five_runs277_records_preserved', after['runs'] == before['runs'] and
          sum(len(run['trajectory']) for run in after['runs']) == 277)
    check('all_old_source36_asset_records_preserved', after['source_files'][:-14] == before['source_files'])
    check('total23models92rows', len(after['models']) - len(before['models']) == 2 and
          after['total_models'] == 23 and after['total_rows'] == 92 and
          sum(len(m['rows']) for m in after['models'][-2:] + old) == 92)
    check('point_counts_preserved262_identity209_common', after['distinct_saved_point_identities'] == 262 and
          after['common_lineage_distinct_points'] == 209 and after['per_run_point_records'] == 277)
    preimage = load(HERE / 'prechange_manifest.json')['files']
    changed = [name for name, meta in preimage.items() if sha(site / name) != meta['sha256']]
    check('only_page_overlay_changed', changed == sorted([c['page_path'], c['overlay_path']]))
    check('all6019gzip_unchanged', sum(n.endswith('.gz') for n in preimage) == 6019)
    check('hosting_owner_private_bytes_unchanged', sha(site / '.openai/hosting.json') == preimage['.openai/hosting.json']['sha256'])
    check('bound21source_inputs_exact', all(sha(ROOT / p) == expected for p, expected in c['inputs_sha256'].items()))
    for asset in after['source_files'][-14:]:
        check('source_asset_exact/' + asset['asset'], sha(site / 'dist' / asset['asset']) == asset['sha256'] == sha(ROOT / asset['path']))
    before_keys = set(before)
    changed_keys = {k for k in before_keys if after[k] != before[k]}
    check('only_declared_existing_overlay_keys_changed', changed_keys == {'models', 'source_files', 'total_models', 'total_rows', 'endpoint_selection'})
    template = next(m for m in before['models'] if m['id'] == 'correlated29_run005_endpoint208_DC')
    refs = {r['label']: r for r in template['rows']}
    new = after['models'][-2:]
    nuclear = load(ROOT / new[0]['source'])
    screened = load(ROOT / new[1]['source'])
    measured = [nuclear['same_point_responses']['complete5f'], screened['models']['complete5f']]
    line_error = moment_error = rms_error = 0.
    for model, original in zip(new, measured, strict=True):
        check('predeclared_id_no_default/' + model['id'], model['id'] in c['model_ids'] and model['is_current'] is False)
        check('finite_space1595_31_parent3789/' + model['id'], model['rci_csf_count'] == 1595 and
              model['radial_orbital_count'] == 31 and model['declared_full_model_csf_count'] == 3789 and
              model['other_parent_CSFs_not_computed'] == 2194 and not model['full_parent_computed'])
        check('originalP_UID_weights_not_new_metric_resume/' + model['id'], model['source_map_sha256'] == c['source_map_sha256'] and
              model['seed_metric_sha256'] == template['metric_sha256'] and model['state_average_weights'] == [.2] * 5 and
              not model['original_tracker_rebased'] and not model['no_pair_projector_changed'] and not model['new_space_is_old_metric_resume'])
        check('fixed31_stationarity_not_measured/' + model['id'], model['orbital_residual'] is None and
              model['energy_indicator_Ha'] is None and not model['stationarity_evaluated'] and not model['expanded_MCDHF_executed'] and
              not model['stationarity_passed'] and not model['experimental_gate_passed'])
        by = {t['label']: t for t in model['targets']}
        source = {t['label']: t for t in original['rows']}
        lower = by['6s_1/2']['energy_Ha']
        check('all5_saved_signed_states_identical/' + model['id'], len(by) == 5 and all(
            t['energy_Ha'] == source[label]['energy_Ha'] and t['UID'] == c['fixed']['UIDs'][label] and
            t['selected_level'] == source[label]['selected_level'] and t['NIST_uncertainty_cm1'] is None and
            t['tracking']['principal_overlap_squared'] == source[label]['principal_overlap_squared'] for label, t in by.items()))
        for row in model['rows']:
            t, ref = by[row['label']], refs[row['label']]
            check('NIST_unique_UID_null_sigma/' + model['id'] + '/' + row['label'],
                  all(row[k] == ref[k] for k in ('level_id', 'reference_zero_level_id', 'reference_cm1', 'uncertainty_cm1', 'matching')) and
                  row['uncertainty_cm1'] is None and row['matching'] == 'unique_nominal_branch')
            line_error = max(line_error, abs((t['energy_Ha'] - lower) * after['Hartree_cm1'] - row['calculated_cm1']))
            check('same_lower_upper_delta_and_gap/' + model['id'] + '/' + row['label'],
                  row['source_energy_Ha'] == t['energy_Ha'] and row['calculated_cm1'] == t['excitation_cm1'] and
                  row['error_cm1'] == row['calculated_cm1'] - row['reference_cm1'] and
                  row['energy_delta_from_previous_Ha'] == t['absolute_delta_upper_Ha'] and
                  row['same_lower_delta_Ha'] == t['absolute_delta_lower_Ha'] and
                  row['transition_delta_from_previous_cm1'] == t['transition_delta_upper_minus_lower_cm1'] and not row['goal_pass'])
        for moment in model['multiplets']:
            labels, weights = (('5d_3/2', '5d_5/2'), (4, 6)) if moment['label'] == '5d' else (('6p_1/2', '6p_3/2'), (2, 4))
            center = sum(by[label]['excitation_cm1'] * w for label, w in zip(labels, weights, strict=True)) / sum(weights)
            split = by[labels[1]]['excitation_cm1'] - by[labels[0]]['excitation_cm1']
            moment_error = max(moment_error, abs(center - moment['centroid_cm1']), abs(split - moment['splitting_cm1']))
        rms_error = max(rms_error, abs(math.sqrt(sum(r['error_cm1']**2 for r in model['rows']) / 4) - model['nominal_gap_RMS_cm1']))
        check('EOL_equal_five_weights/' + model['id'], abs(sum(t['energy_Ha'] * .2 for t in by.values()) - model['EOL_Ha']) <= 2e-11)
    check('all_upper_lower_transition_arithmetic', line_error <= 1e-8 and moment_error <= 1e-9 and rms_error <= 1e-9)
    check('nuclear_original14gateFAIL_not_upgraded', new[0]['original_execution_exit_code'] == 1 and
          new[0]['qualification']['original_execution_passed'] is False and nuclear['original_expected14gate_passed'] is False)
    check('independent389_finite_qualification_separate', nuclear['saved_array_qualification_passed'] and
          nuclear['checks_count'] == 389 and new[0]['qualification']['independent_numeric_checks'] == 389)
    check('screened941_runtime_not_independent_replay', screened['all_original_numeric_identity_gates_passed'] and
          len(screened['checks']) == 941 and new[1]['qualification']['independent_saved_array_replay_performed'] is False)
    fd = after['directional_qualification']
    check('FD7053_one_direction_two_h', fd['native_controls'] == 7053 and fd['direction_count'] == 1 and
          [d['h'] for d in fd['directions']] == [1e-5, 5e-6] and all(abs(d['error']) < d['original_derivative_gate'] for d in fd['directions']))
    check('FD_not_stationarity_uniform_bound_or31gradient', not fd['stationarity_passed'] and
          not fd['uniform_finite_remainder_bound'] and not fd['global_Hessian_bound'] and not fd['expanded31_gradient_qualified'])
    page = (site / c['page_path']).read_text()
    fragment = (HERE / 'fixed208_summary.js').read_text()
    restored = page.replace(fragment + '\n', '', 1).replace(
        "\nupdate={...update,models:update.models.filter(m=>m.model_kind==='nonstationary_correlated29_mcdhf_endpoint')};", '', 1).replace(
        'correlatedWarning+fixed208Summary(payload.correlated_update)+correlatedSummary(payload.correlated_update)',
        'correlatedWarning+correlatedSummary(payload.correlated_update)', 1)
    check('renderer_only_exact_three_declared_seams', restored == (HERE / 'original_page.html').read_text())
    check('FD_and_full5state_fields_visible', all(s in fragment for s in
          ('fd.native_controls', '一条', '两个步长', 'fixed.flatMap(m=>m.targets', 'NIST_uncertainty_cm1', '末尾计数门14期望/7实际失败')))
    inline = page.split('<script>', 1)[1].split('</script>', 1)[0]
    (HERE / 'page_inline.js').write_text(inline)
    report = {'all_static_saved_JSON_and_preservation_checks_passed': True, 'checks_count': len(rows), 'checks': rows,
              'models': 23, 'rows': 92, 'old_models_preserved': 21, 'old_rows_preserved': 84,
              'unchanged_gzip': 6019, 'unchanged_trajectory_records': 277, 'unchanged_saved_identities': 262,
              'max_transition_error_cm1': line_error, 'max_multiplet_error_cm1': moment_error, 'max_RMS_error_cm1': rms_error,
              'overlay_sha256': sha(site / c['overlay_path']), 'page_sha256': sha(site / c['page_path']),
              'contract_sha256': sha(HERE / 'contract.json'), 'verifier_sha256': sha(__file__),
              'NumPy_Atomx_NPZ_science_browser_API_calls': 0, 'publication_performed': False,
              'browser_qualification_pending_root': True}
    (HERE / 'data_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print({k: v for k, v in report.items() if k != 'checks'})


if __name__ == '__main__':
    main()
