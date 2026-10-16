"""Prepare an isolated Atlas v10 delta from closed native31 JSON; stdlib only."""
import argparse
import copy
import hashlib
import json
import math
import os
import resource
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LABELS = ('6s_1/2', '5d_3/2', '5d_5/2', '6p_1/2', '6p_3/2')
KIND = 'nonstationary_correlated31_mcdhf_endpoint'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate_inputs(contract):
    """Check JSON/scalar identities without reading or hashing physical arrays."""
    for relative, expected in contract['inputs_sha256'].items():
        assert Path(relative).suffix in ('.json', '.md'), relative
        assert sha(ROOT / relative) == expected, relative
    run_root = ROOT / contract['run_root']
    actual = load(run_root / 'actual/result.json')
    closed = load(run_root / 'closure001/completion.json')
    receipt = load(run_root / 'root_execution.json')
    scientific = load(run_root / 'contract.json')
    assert receipt['status'] == 'foreground_closed' and receipt['exit_code'] == 0
    assert not receipt['timed_out'] and not receipt['current_source_adopted']
    assert actual['all_completed_controls_passed'] and len(actual['checks']) == 1582
    assert all(c['passed'] for c in actual['checks']) and actual['error'] is None
    assert closed['all_scalar_completion_controls_passed'] and len(closed['checks']) == 77
    assert all(c['passed'] for c in closed['checks']) and not closed['failed_live_controls']
    assert closed['actual_new_qualified_steps'] == closed['finite_stored_H_acceptance_count'] == 16
    assert not closed['unqualified_candidates'] and not closed['independent_H_C_gradient_replay']
    assert not actual['current_source_adopted'] and actual['no_GPU'] and actual['no_energy_callback']
    assert not actual['experimental_certified'] and not actual['physical_completeness_certified']
    fixed = scientific['fixed']
    for key, expected in contract['fixed'].items():
        assert fixed[key] == expected, key
    assert actual['source_map_sha256'] == scientific['source']['map_sha256'] == contract['source_map_sha256']
    assert actual['completed']['H'] == actual['completed']['CI'] == 23
    assert actual['completed']['gradient_from_weights'] == 51 and actual['completed']['optimizer'] == 1
    assert [p['iteration'] for p in actual['trajectory']] == list(range(17))
    assert [p['iteration'] for p in closed['trajectory']] == list(range(17))
    points = []
    for recorded, closure_point in zip(actual['trajectory'], closed['trajectory'], strict=True):
        manifest = load(ROOT / recorded['manifest'])
        assert sha(ROOT / recorded['manifest']) == recorded['manifest_sha256'] == closure_point['manifest_sha256']
        assert manifest['array_sha256'] == recorded['array_sha256'] == closure_point['array_sha256']
        assert manifest['qualified'] and manifest['iteration'] == recorded['iteration']
        assert manifest['source_map_sha256'] == contract['source_map_sha256']
        assert manifest['metric_sha256'] == fixed['new_metric_sha256']
        assert manifest['geometry_sha256'] == fixed['new_geometry_sha256']
        assert manifest['objective_identity'] == fixed['objective_identity']
        assert len(manifest['owner']['CSFs']) == 1595 and len(manifest['owner']['orbital_keys']) == 31
        assert manifest['owner']['hamiltonian_terms'] == ['dirac-coulomb']
        assert manifest['rows'] == closure_point['rows'] and manifest['multiplets'] == closure_point['multiplets']
        assert manifest['diagnostics'] == recorded['diagnostics']
        assert not manifest['original_stationarity_passed'] and not closure_point['original_stationarity_passed']
        assert manifest['diagnostics']['orbital_tolerance'] == 1e-7
        assert manifest['diagnostics']['energy_tolerance'] == 1e-8
        assert manifest['diagnostics']['orbital_residual'] > 1e-7
        assert manifest['diagnostics']['estimated_energy_error'] > 1e-8
        assert manifest['no_physical_completeness_or_experimental_certificate']
        assert [(r['label'], r['UID']) for r in manifest['rows']] == [tuple(x) for x in fixed['original_UIDs']]
        assert all(r['weight'] == .2 and r['selected_level'] == r['UID'] for r in manifest['rows'])
        assert manifest['nominal_four_gap_RMS_cm1'] == recorded['nominal_four_gap_RMS_cm1'] == closure_point['RMS_cm1']
        if recorded['iteration']:
            assert manifest['exact_acceptance']['qualified']
            assert closure_point['finite_stored_H_acceptance_qualified']
            assert manifest['previous_qualified'] == actual['trajectory'][recorded['iteration'] - 1]['manifest']
        else:
            assert manifest['previous_qualified'] is None and manifest['exact_acceptance'] is None
        points.append(manifest)
    assert actual['last_qualified_checkpoint'] == actual['trajectory'][-1]['manifest']
    assert actual['optimizer_diagnostics']['termination_reason'] == 'maximum_iterations'
    assert not actual['optimizer_diagnostics']['converged']
    assert points[-1]['nominal_four_gap_RMS_cm1'] == 1283.7699121117464
    assert points[-1]['diagnostics']['orbital_residual'] == .68254770188387
    assert points[-1]['diagnostics']['estimated_energy_error'] == .6018594158445788
    return actual, closed, receipt, points


def native31(before, contract, actual, closed, receipt, points):
    """New model and identity domain; old29 model/run objects stay byte equivalent."""
    template = next(m for m in before['models'] if m['id'] == 'correlated29_run005_endpoint208_DC')
    references = {r['label']: r for r in template['rows']}
    first_targets = {t['label']: t for t in points[0]['rows']}
    trajectory = []
    prefix = contract['evidence_prefix'].removeprefix('dist/') + '/'
    for manifest, recorded, closure_point in zip(points, actual['trajectory'], closed['trajectory'], strict=True):
        targets = copy.deepcopy(manifest['rows'])
        lower = targets[0]['energy_Ha']
        first_lower = first_targets['6s_1/2']['energy_Ha']
        for t in targets:
            ref = references.get(t['label'])
            assert abs((t['energy_Ha'] - lower) * before['Hartree_cm1'] - t['excitation_cm1']) <= 1e-8
            assert t['nominal_gap_cm1'] == t['excitation_cm1'] - t['NIST_nominal_cm1']
            if ref:
                assert t['NIST_nominal_cm1'] == ref['reference_cm1']
                assert t['UID'][:2] == [ref['two_j'], ref['parity']]
                assert ref['uncertainty_cm1'] is None and ref['matching'] == 'unique_nominal_branch'
            t.update(state_average_weight=.2, NIST_uncertainty_cm1=None,
                     physical_reference_overlap_squared=None,
                     physical_reference_overlap_scope='Not exported in the saved native31 checkpoint JSON; original UID is retained.',
                     delta_upper_from_initial_Ha=t['energy_Ha'] - first_targets[t['label']]['energy_Ha'],
                     delta_lower_from_initial_Ha=lower - first_lower,
                     transition_shift_from_initial_cm1=t['excitation_cm1'] - first_targets[t['label']]['excitation_cm1'])
        moments = copy.deepcopy(template['multiplets'])
        by = {t['label']: t for t in targets}
        for m in moments:
            saved = next(x for x in manifest['multiplets'] if x['label'] == m['label'])
            lo, hi = (('5d_3/2', '5d_5/2') if m['label'] == '5d' else ('6p_1/2', '6p_3/2'))
            weights = m['degeneracy_weights']
            center = (weights[0] * by[lo]['excitation_cm1'] + weights[1] * by[hi]['excitation_cm1']) / sum(weights)
            split = by[hi]['excitation_cm1'] - by[lo]['excitation_cm1']
            assert abs(center - saved['centroid_cm1']) <= 1e-9
            assert abs(split - saved['splitting_cm1']) <= 1e-9
            m.update(saved, centroid_gap_cm1=saved['centroid_cm1'] - m['reference_centroid_cm1'],
                     splitting_gap_cm1=saved['splitting_cm1'] - m['reference_splitting_cm1'])
        rms = math.sqrt(sum(t['nominal_gap_cm1'] ** 2 for t in targets[1:]) / 4)
        assert abs(rms - manifest['nominal_four_gap_RMS_cm1']) <= 1e-9
        assert abs(sum(t['energy_Ha'] * .2 for t in targets) - manifest['diagnostics']['energy_ha']) <= 2e-11
        identity = ['native31_saved_manifest_v1', manifest['source_map_sha256'], manifest['metric_sha256'],
                    manifest['geometry_sha256'], manifest['objective_identity'], recorded['manifest_sha256'],
                    manifest['array_sha256'], canonical_sha(manifest['owner']), canonical_sha(manifest['exact_point'])]
        d = manifest['diagnostics']
        trajectory.append(dict(iteration=manifest['iteration'], native31_local_step=manifest['iteration'],
            global_accepted_step=None, global_step_scope='New 3689 metric run: local0–16; never old29 global208–224.',
            reported_EOL_Ha=d['energy_ha'], orbital_residual=d['orbital_residual'],
            energy_indicator_Ha=d['estimated_energy_error'], nominal_four_gap_RMS_cm1=manifest['nominal_four_gap_RMS_cm1'],
            targets=targets, multiplets=moments, source_manifest=recorded['manifest'],
            source_manifest_sha256=recorded['manifest_sha256'], source_arrays_sha256=manifest['array_sha256'],
            source_manifest_asset=prefix + f"checkpoints/accepted_{manifest['iteration']:04d}.json",
            saved_point_identity=identity, saved_point_identity_schema='native31_saved_manifest_v1',
            array_identity_basis='Original saved JSON array_sha256; no array opened or rehashed in Atlas.',
            source_map_sha256=manifest['source_map_sha256'], metric_sha256=manifest['metric_sha256'],
            geometry_sha256=manifest['geometry_sha256'], objective_identity=manifest['objective_identity'],
            owner_metadata_sha256=identity[-2], exact_point_ledger_sha256=identity[-1],
            fullsigned_metrics=copy.deepcopy(manifest['fullsigned_metrics']), stationarity_passed=False,
            experimental_certification=False, physical_completeness_certified=False,
            exact_stored_H_acceptance_qualified=closure_point['finite_stored_H_acceptance_qualified']))
    assert len({tuple(p['saved_point_identity']) for p in trajectory}) == 17
    first, last = trajectory[0], trajectory[-1]
    by = {t['label']: t for t in last['targets']}
    rows = copy.deepcopy(template['rows'])
    for row in rows:
        t = by[row['label']]
        for k in ('energy_delta_from_previous_Ha', 'same_lower_delta_Ha', 'transition_delta_from_previous_cm1', 'transition_delta_from_bare_cm1'):
            row.pop(k, None)
        row.update(calculated_cm1=t['excitation_cm1'], error_cm1=t['nominal_gap_cm1'],
            error_percent=100 * t['nominal_gap_cm1'] / row['reference_cm1'],
            goal_pass=abs(t['nominal_gap_cm1'] / row['reference_cm1']) < .01,
            source_energy_Ha=t['energy_Ha'], source_column=t['column'], exact_J_local_root=t['selected_level'][2],
            source_physical_UID=dict(twoJ=t['UID'][0], parity=t['UID'][1], local_root=t['UID'][2]),
            physical_reference_overlap_squared=None, physical_reference_overlap_scope=t['physical_reference_overlap_scope'],
            delta_upper_from_initial_Ha=t['delta_upper_from_initial_Ha'],
            delta_lower_from_initial_Ha=t['delta_lower_from_initial_Ha'],
            transition_shift_from_initial_cm1=t['transition_shift_from_initial_cm1'])
    assert all(not r['goal_pass'] for r in rows)
    model = dict(id=contract['model_id'], title='原生31轨道MCDHF · 完整1595目标CSF · literal16末态',
        model_kind=KIND, is_current=False, saved_run='native31_f043_relaxation_preparation001/accepted_0016',
        operator='DC', rci_csf_count=1595, target_sector_csf_count=1595, declared_full_model_csf_count=3789,
        other_parent_CSFs_not_computed=2194, full_parent_computed=False, symmetry_block_count=5,
        radial_orbital_count=31, active_coordinate_count=3689, electron_count=69, accepted_steps=16,
        native31_terminal_local_step=16, global_initial_step=None, global_accepted_steps=None,
        all31_allowed_to_relax=True, state_average_weights=[.2] * 5, original_tracker_rebased=False,
        no_pair_projector_changed=False, expanded_MCDHF_executed=True, stationarity_evaluated=True,
        orbital_residual=last['orbital_residual'], energy_indicator_Ha=last['energy_indicator_Ha'],
        orbital_tolerance=1e-7, energy_indicator_tolerance_Ha=1e-8, stationarity_passed=False,
        experimental_gate_passed=False, physical_completeness_certified=False, production_certified=False,
        highest_level_reached=False, current_source_adopted=False, GPU_adopted=False,
        source_map_sha256=contract['source_map_sha256'], metric_sha256=last['metric_sha256'],
        geometry_sha256=last['geometry_sha256'], objective_sha256=last['objective_identity'],
        objective_identity=last['objective_identity'], initial_state_sha256=contract['fixed']['new_state_sha256'],
        point_manifest_sha256=last['source_manifest_sha256'], point_arrays_sha256=last['source_arrays_sha256'],
        point_identity=last['saved_point_identity'], point_identity_schema=last['saved_point_identity_schema'],
        ground=by['6s_1/2'], targets=last['targets'], rows=rows, multiplets=last['multiplets'],
        EOL_Ha=last['reported_EOL_Ha'], initial_EOL_Ha=first['reported_EOL_Ha'], EOL_delta_Ha=closed['EO_delta_Ha'],
        nominal_gap_RMS_cm1=last['nominal_four_gap_RMS_cm1'], initial_nominal_gap_RMS_cm1=first['nominal_four_gap_RMS_cm1'],
        initial_orbital_residual=first['orbital_residual'], elapsed_uncontrolled_s=receipt['elapsed_uncontrolled_s'],
        peak_RSS_MiB=actual['peak_RSS_MiB'], runtime_controls=1582, completion_metadata_checks=77,
        completed_calls=copy.deepcopy(actual['completed']), finite_stored_H_acceptance_count=16,
        termination_reason='maximum_iterations', source=contract['run_root'] + '/actual/result.json',
        sha256=contract['inputs_sha256'][contract['run_root'] + '/actual/result.json'],
        source_asset=prefix + 'actual/result.json', completion_asset=prefix + 'closure001/completion.json',
        trajectory_asset=prefix + 'trajectory.json', receipt_asset=prefix + 'root_execution.json',
        stationarity_note='全31径向轨道共同优化16步；原3689度量残差和能量指示量两门均FAIL。1582运行控制、77保存标量闭环与16精确有限H接受门通过，不是独立H/CI/梯度重放或导数认证。',
        scope='历史原生Atomx f043，69电子，原七核no-pair P和五physical UID等权0.2；31轨道、完整五J/宇称目标扇区1595 CSF、3689坐标度量。3789父CSF仅枚举，其余2194未计算。',
        boundary_note='取字面最后qualified local16，完整保留local0–16；不按NIST或最小RMS选点。每个上、下态来自同点同DC算符，未拼Breit/QED/反冲或经验拟合。能量指示量不是谱误差界；未认证物理完备、实验精度、当前源码或GPU，也未授权自动续算。保存点JSON未导出参考重叠，空值保留。')
    run = dict(model_id=model['id'], native31_local_step_scope=True, terminal_iteration=16,
               global_initial_step=None, global_terminal_step=None, trajectory=trajectory,
               receipt=copy.deepcopy(receipt), identity_schema='native31_saved_manifest_v1')
    return model, run


def page_delta(page, block):
    seams = {
        'function correlatedSummary(update){': block + '\nfunction correlatedSummary(update){',
        'correlatedWarning+fixed208Summary(payload.correlated_update)':
            'correlatedWarning+native31Summary(payload.correlated_update)+fixed208Summary(payload.correlated_update)',
        "const status=m.model_kind==='nonstationary_correlated29_mcdhf_endpoint'?":
            "const status=m.model_kind==='nonstationary_correlated31_mcdhf_endpoint'?'16步31轨道MCDHF · 原驻值未过':m.model_kind==='nonstationary_correlated29_mcdhf_endpoint'?",
        "'fixed_nonstationary_virtual_rci','nonstationary_correlated29_mcdhf_endpoint'].includes":
            "'fixed_nonstationary_virtual_rci','nonstationary_correlated29_mcdhf_endpoint','nonstationary_correlated31_mcdhf_endpoint'].includes",
        '${correlatedShifts(m)}<p class="source">': '${correlatedShifts(m)}${native31Shifts(m)}<p class="source">',
        '<h3>原生29轨道相关MCDHF：已闭合末态与完整轨迹</h3>':
            '<h3>原生29与31轨道相关MCDHF：分别闭合的末态与完整轨迹</h3>',
    }
    for old, new in seams.items():
        assert page.count(old) == 1, old
        page = page.replace(old, new)
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args()
    c = load(HERE / 'contract.json')
    b = load(args.binding)
    assert b['status'] == 'root_authorized_isolated_JSON_build_after_performance_closure'
    assert b['contract_sha256'] == sha(HERE / 'contract.json')
    assert b['builder_sha256'] == sha(__file__)
    assert sha(ROOT / b['performance_closure_receipt']) == b['performance_closure_receipt_sha256']
    assert b['performance_foreground_closed'] is True and b['no_API_browser_or_science'] is True
    assert b['output'] == str(HERE / 'generated_delta001')
    assert b['cpu'] in os.sched_getaffinity(0)
    os.sched_setaffinity(0, {b['cpu']})
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    for name, expected in c['authoring_sources_sha256'].items():
        assert sha(HERE / name) == expected, name
    old_manifest = load(ROOT / c['published_manifest'])
    site = Path(c['site'])
    assert sha(site / '.openai/hosting.json') == c['hosting_sha256']
    for relative, item in old_manifest['files'].items():
        assert sha(site / relative) == item['sha256'], relative
    assert sum(p.endswith('.json.gz') for p in old_manifest['files']) == 6019
    before = load(HERE / 'original_overlay.json')
    assert before['total_models'] == 23 and before['total_rows'] == 92
    actual, closed, receipt, points = validate_inputs(c)
    model, run = native31(before, c, actual, closed, receipt, points)
    overlay = copy.deepcopy(before)
    overlay['models'].append(model)
    overlay['runs'].append(run)
    output = Path(b['output'])
    output.mkdir(exist_ok=False)
    evidence = output / c['evidence_prefix']
    assets = []
    for relative, destination in c['evidence_files'].items():
        target = evidence / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        assets.append(dict(path=relative, asset=str(target.relative_to(output / 'dist')), sha256=sha(target)))
    trajectory_path = evidence / 'trajectory.json'
    write(trajectory_path, dict(model_id=model['id'], identity_schema=run['identity_schema'],
        trajectory=run['trajectory'], stationarity_passed=False, experimental_certification=False,
        point_selection='Literal last qualified local16; all17 chronological points preserved.'))
    assets.append(dict(path=str(trajectory_path.relative_to(ROOT)),
                       asset=str(trajectory_path.relative_to(output / 'dist')), sha256=sha(trajectory_path)))
    overlay['source_files'].extend(assets)
    overlay.update(total_models=24, total_rows=96, previous_v9_models_preserved=23, previous_v9_rows_preserved=92,
        native31_model_id=model['id'], native31_trajectory_records=17, native31_five_state_records=85,
        native31_distinct_saved_manifest_identities=17, native31_identity_schema=run['identity_schema'],
        total_trajectory_record_count_including_native31=294, total_five_state_records_including_native31=1470,
        legacy262_saved_point_identities_preserved=True,
        endpoint_selection=before['endpoint_selection'] + ' Native31 adds literal last qualified local16 and all17 points; no best-gap selection.')
    assert overlay['models'][:-1] == before['models'] and overlay['runs'][:-1] == before['runs']
    assert overlay['source_files'][:-len(assets)] == before['source_files']
    write(output / c['overlay_path'], overlay)
    page = page_delta((HERE / 'original_page.html').read_text(), (HERE / 'native31_summary.js').read_text())
    (output / c['page_path']).write_text(page)
    delta = {str(p.relative_to(output)): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted((output / 'dist').rglob('*')) if p.is_file()}
    write(output / 'prepared.json', dict(status='isolated_v10_delta_prepared_no_site_mutation',
        contract_sha256=sha(HERE / 'contract.json'), binding_sha256=sha(args.binding),
        models=24, rows=96, old_models=23, old_rows=92, old_runs=5, old_records=277,
        new_runs=1, new_records=17, total_records=294, old_identity_domain_count=262,
        new_manifest_identity_domain_count=17, new_domain_is_not_array_replay=True,
        all_old_manifest_files_verified=True, unchanged_gzip=6019, delta_files=delta,
        source_only_calls=dict(Atomx=0, NumPy=0, NPZ=0, science=0, browser=0, API=0),
        root_site_application_browser_and_publication_pending=True))
    print(json.dumps(dict(status='isolated_v10_delta_prepared', models=24, rows=96,
                          delta_files=len(delta), prepared_sha256=sha(output / 'prepared.json'))))


if __name__ == '__main__':
    main()
