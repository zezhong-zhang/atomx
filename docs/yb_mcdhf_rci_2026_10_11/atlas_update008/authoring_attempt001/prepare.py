"""Saved-JSON fixed-RCI overlay; reuse atlas007 utilities, no physical imports."""
import copy
import gzip
import importlib.util
import math
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
UTILITY = HERE.with_name('atlas_update007') / 'prepare.py'
spec = importlib.util.spec_from_file_location('atlas007_json_utilities', UTILITY)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
sha, load, write, LABELS = shared.sha, shared.load, shared.write, shared.LABELS
BASE = 'docs/yb_mcdhf_rci_2026_10_11/'
NUCLEAR = BASE + 'complete5f_fixed208_import_repair_preparation/'
QUAL = NUCLEAR + 'completion_counter_audit/saved_array_qualification_preparation/output_serialization_repair/'
SCREENED = BASE + 'screened_f_next_step208/execution001/'
FD = BASE + 'stationarity_next_step208/instrumentation_repair001/'


def archive(site, contract, files):
    evidence = site / contract['evidence_prefix']
    evidence.mkdir(parents=True, exist_ok=False)
    assets = []
    for folder, paths in files.items():
        (evidence / folder).mkdir()
        for relative in paths:
            source = ROOT / relative
            filename = ('actual-' if '/actual/' in relative else '') + source.name
            target = evidence / folder / filename
            assert not target.exists()
            shutil.copyfile(source, target)
            assets.append({'path': relative, 'asset': str(target.relative_to(site / 'dist')), 'sha256': sha(target)})
    return assets


def fixed_model(name, measured, template, contract, asset_map, receipt, qualification):
    by = {t['label']: copy.deepcopy(t) for t in measured['rows']}
    assert set(by) == set(LABELS)
    old_targets = {t['label']: t for t in template['targets']}
    refs = {r['label']: r for r in template['rows']}
    ground = by['6s_1/2']['energy_Ha']
    rows, targets = [], []
    for label in LABELS:
        t = by[label]
        assert t['reference_UID'] == contract['fixed']['UIDs'][label] == old_targets[label]['UID']
        assert t['selected_level'][:2] == t['reference_UID'][:2] and t['NIST_uncertainty_cm1'] is None
        assert abs((t['energy_Ha'] - ground) * contract['Hartree_cm1'] - t['excitation_cm1']) <= 1e-8
        assert t['nominal_gap_cm1'] == t['excitation_cm1'] - t['NIST_nominal_cm1']
        t.update(UID=t['reference_UID'], state_average_weight=.2,
                 energy_delta_from_previous_Ha=t['absolute_delta_upper_Ha'],
                 tracking={'reference_levels': [t['reference_UID']], 'selected_levels': [t['selected_level']],
                           'per_level_weight': .2, 'principal_overlap_squared': t['principal_overlap_squared']})
        targets.append(t)
        if label == '6s_1/2':
            continue
        row = copy.deepcopy(refs[label])
        assert row['reference_cm1'] == t['NIST_nominal_cm1'] and row['uncertainty_cm1'] is None
        for key in ('delta_upper_from_initial_Ha', 'delta_lower_from_initial_Ha', 'transition_shift_from_initial_cm1'):
            row.pop(key, None)
        row.update(calculated_cm1=t['excitation_cm1'], error_cm1=t['nominal_gap_cm1'],
                   error_percent=100 * t['nominal_gap_cm1'] / row['reference_cm1'],
                   goal_pass=abs(t['nominal_gap_cm1'] / row['reference_cm1']) < .01,
                   source_energy_Ha=t['energy_Ha'], source_column=t['column'], exact_J_local_root=t['selected_level'][2],
                   physical_reference_overlap_squared=t['principal_overlap_squared'][0],
                   energy_delta_from_previous_Ha=t['absolute_delta_upper_Ha'],
                   same_lower_delta_Ha=t['absolute_delta_lower_Ha'],
                   transition_delta_from_previous_cm1=t['transition_delta_upper_minus_lower_cm1'],
                   transition_delta_from_bare_cm1=None)
        rows.append(row)
    moments = copy.deepcopy(template['multiplets'])
    for moment in moments:
        saved = next(m for m in measured['multiplets'] if m['label'] == moment['label'])
        moment.update(saved, centroid_gap_cm1=saved['centroid_cm1'] - moment['reference_centroid_cm1'],
                      splitting_gap_cm1=saved['splitting_cm1'] - moment['reference_splitting_cm1'])
    is_nuclear = name == 'nuclear'
    source = QUAL + 'actual/verification.json' if is_nuclear else SCREENED + 'actual/result.json'
    completion = QUAL + 'completion.json' if is_nuclear else SCREENED + 'completion.json'
    detail = ('原科学执行末尾14次h/S期望与实际7次不符，原exit1和原门FAIL保留。'
              '另一次389项保存数组资格通过，认证有限H/CI/正交与原P，不改写原运行成功。'
              if is_nuclear else '941项原运行控制通过，32项独立有符号标量矩阵锚通过。'
              '完成元数据闭环；没有独立新增数组重放或扩展轨道梯度。')
    model = dict(id=contract['model_ids'][0 if is_nuclear else 1],
        title='固定208点 · 完整5f双jj · ' + ('核场径向候选' if is_nuclear else '闭核68屏蔽径向候选') + ' · DC RCI',
        model_kind='fixed_orbital_rci', is_current=False, saved_run=(NUCLEAR if is_nuclear else SCREENED).rstrip('/').split('/')[-1] + '/complete5f',
        operator='DC', rci_csf_count=1595, target_sector_csf_count=1595, declared_full_model_csf_count=3789,
        other_parent_CSFs_not_computed=2194, full_parent_computed=False, symmetry_block_count=5,
        radial_orbital_count=31, old_radial_orbital_count=29, active_coordinate_count=None,
        expanded_metric_constructed=False, new_space_is_old_metric_resume=False, electron_count=69,
        state_average_weights=[.2] * 5, original_tracker_rebased=False, no_pair_projector_changed=False,
        source_map_sha256=contract['source_map_sha256'], seed_metric_sha256=template['metric_sha256'],
        seed_geometry_sha256=template['geometry_sha256'], seed_objective_sha256=template['objective_sha256'],
        seed_point_manifest_sha256=template['point_manifest_sha256'], seed_point_arrays_sha256=template['point_arrays_sha256'],
        seed_orbital_residual=template['orbital_residual'], seed_energy_indicator_Ha=template['energy_indicator_Ha'],
        seed_global_step=208, orbital_residual=None, energy_indicator_Ha=None, stationarity_passed=False,
        stationarity_evaluated=False, expanded_MCDHF_executed=False, experimental_gate_passed=False,
        physical_completeness_certified=False, highest_level_reached=False, production_certified=False,
        ground=targets[0], targets=targets, rows=rows, multiplets=moments,
        previous_stage='同一literal208原29轨道/1046-CSF DC基线',
        EOL_Ha=measured['EOL_reported_Ha'], nominal_gap_RMS_cm1=measured['nominal_four_gap_RMS_cm1'],
        elapsed_uncontrolled_s=receipt['elapsed_uncontrolled_seconds'], original_execution_exit_code=receipt['exit_code'],
        qualification=qualification, source=source, sha256=sha(ROOT / source), source_asset=asset_map[source],
        completion_asset=asset_map[completion],
        scope='原生MCDHF→RCI主线的固定点角通道筛选：同literal208旧29波、原七核no-pair P、原五UID及等权0.2；'
              '加入预声明的两个f径向候选，仅完整五目标J/宇称扇区1595 CSF做DC RCI。3789父CSF只枚举，其他2194未计算。',
        stationarity_note='新增31轨道没有做MCDHF优化，未计算扩展驻值残差或能量指示量。原29/3451种子驻值两门仍FAIL；'
                          '不将种子残差冒充新空间测量。' + detail,
        boundary_note=detail + ' 每条谱线用同空间上态−6s下态；两个种子预声明，不按NIST选优或作为默认胜者。'
                      '闭核68 Fock是径向生成辅助场，不是相关MCDHF Euler。未加旧点Breit/QED/反冲；有限响应不是径向收敛或缺项误差界。')
    if is_nuclear:
        model['verification_asset'] = asset_map[source]
    assert all(not r['goal_pass'] for r in rows)
    assert abs(sum(t['energy_Ha'] * .2 for t in targets) - model['EOL_Ha']) < 2e-11
    assert abs(math.sqrt(sum(r['error_cm1']**2 for r in rows) / 4) - model['nominal_gap_RMS_cm1']) < 1e-9
    return model


def main():
    c = load(HERE / 'contract.json')
    site = Path(c['site'])
    assert c['status'] == 'prospective_before_local_site_edits' and c['local_only']
    for filename, expected in c['authoring_sources_sha256'].items():
        assert sha(HERE / filename) == expected, filename
    assert sha(UTILITY) == c['shared_utilities_sha256']
    assert sha(HERE / 'prechange_manifest.json') == c['prechange_manifest_sha256']
    for relative, item in load(HERE / 'prechange_manifest.json')['files'].items():
        assert sha(site / relative) == item['sha256'], relative
    for relative, expected in c['inputs_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
    before = load(HERE / 'original_overlay.json')
    assert sha(HERE / 'original_overlay.json') == c['old_overlay_sha256']
    template = next(m for m in before['models'] if m['id'] == 'correlated29_run005_endpoint208_DC')
    nist = shared.json.loads(gzip.decompress((site / 'dist/ions/70-1.json.gz').read_bytes()))
    for ref in template['rows']:
        hit = [l for l in nist['levels'] if l['level_id'] == ref['level_id']]
        assert len(hit) == 1 and hit[0]['energy_cm1'] == ref['reference_cm1']
        assert hit[0]['uncertainty_cm1'] is ref['uncertainty_cm1'] is None
    nuclear = load(ROOT / QUAL / 'actual/verification.json')
    screened = load(ROOT / SCREENED / 'actual/result.json')
    closed = load(ROOT / SCREENED / 'completion.json')
    support = load(ROOT / SCREENED / 'actual/radial_support.json')
    fd = load(ROOT / FD / 'actual/result.json')
    assert nuclear['saved_array_qualification_passed'] and nuclear['checks_count'] == 389
    assert not nuclear['original_expected14gate_passed'] and nuclear['original_failed_execution_preserved']
    assert screened['all_original_numeric_identity_gates_passed'] and len(screened['checks']) == 941
    assert closed['all_completion_metadata_checks_passed'] and closed['total_actual_live_controls'] == 941
    assert fd['all_declared_directional_controls_passed'] and len(fd['controls']) == 7053
    assert screened['source_map_sha256'] == closed['source_map_sha256'] == fd['source_map_sha256'] == c['source_map_sha256']
    assert [d['h'] for d in fd['directions']] == [1e-5, 5e-6] and fd['isolated_direction_only']
    for measured in (nuclear['same_point_responses']['baseline'], screened['models']['baseline']):
        assert {t['label']: t['energy_Ha'] for t in measured['rows']} == {t['label']: t['energy_Ha'] for t in template['targets']}
    assets = archive(site, c, {'nuclear': [NUCLEAR + 'root_execution.json', NUCLEAR + 'actual/failure.json',
        NUCLEAR + 'contract.json', QUAL + 'actual/verification.json', QUAL + 'completion.json'],
        'screened': [SCREENED + p for p in ('actual/result.json', 'actual/radial_support.json', 'completion.json', 'closure_handoff.json', 'root_execution.json', 'contract.json')],
        'FD': [FD + p for p in ('actual/result.json', 'root_execution.json', 'contract.json')]})
    amap = {a['path']: a['asset'] for a in assets}
    c = dict(c, Hartree_cm1=before['Hartree_cm1'])
    models = [fixed_model('nuclear', nuclear['same_point_responses']['complete5f'], template, c, amap,
                         load(ROOT / NUCLEAR / 'root_execution.json'),
                         {'original_execution_passed': False, 'original_expected14gate_passed': False,
                          'independent_saved_array_qualification_passed': True, 'independent_numeric_checks': 389}),
              fixed_model('screened', screened['models']['complete5f'], template, c, amap,
                         load(ROOT / SCREENED / 'root_execution.json'),
                         {'original_execution_passed': True, 'native_numeric_checks': 941,
                          'independent_saved_array_replay_performed': False, 'scalar_signed_anchors': 32})]
    for model, field in zip(models, ('nuclear_failed_run_diagnostic_mean_r_a0', 'mean_r_a0'), strict=True):
        model['radial_seed_support'] = [{'key': r['key'], 'mean_r_a0': r[field], 'adequacy_bound': False}
                                       for r in support['rows'] if r['key'] in [[5, 3], [5, -4]]]
        assert len(model['radial_seed_support']) == 2
    overlay = copy.deepcopy(before)
    overlay['models'].extend(models)
    overlay['source_files'].extend(assets)
    overlay.update(total_models=23, total_rows=92, previous_v8_models_preserved=21, previous_v8_rows_preserved=84,
        endpoint_selection=before['endpoint_selection'] + ' Two fixed literal208 f seeds are shown together; no NIST winner/default.',
        fixed208_model_ids=[m['id'] for m in models], fixed208_models_are_not_MCDHF_optimized=True,
        directional_qualification={'name': 'literal208原29/3451：一条切向、两步长FD资格',
            'native_controls': 7053, 'direction_count': 1, 'direction_definition': '-pg / ||pg|| at original3451 metric',
            'directions': fd['directions'], 'completed_calls': fd['completed'], 'source_map_sha256': fd['source_map_sha256'],
            'source_asset': amap[FD + 'actual/result.json'], 'receipt_asset': amap[FD + 'root_execution.json'],
            'qualified_direction_only': True, 'uniform_finite_remainder_bound': False, 'global_Hessian_bound': False,
            'stationarity_passed': False, 'expanded31_gradient_qualified': False, 'production_reported_objective_changed': False})
    assert overlay['models'][:-2] == before['models'] and overlay['runs'] == before['runs']
    write(site / c['overlay_path'], overlay)
    page = (HERE / 'original_page.html').read_text()
    seam = 'function correlatedSummary(update){\nif(!update)return \x27\x27;'
    assert page.count(seam) == 1
    page = page.replace(seam, seam + "\nupdate={...update,models:update.models.filter(m=>m.model_kind==='nonstationary_correlated29_mcdhf_endpoint')};")
    call = 'correlatedWarning+correlatedSummary(payload.correlated_update)'
    assert page.count(call) == 1
    page = page.replace(call, 'correlatedWarning+fixed208Summary(payload.correlated_update)+correlatedSummary(payload.correlated_update)')
    block = (HERE / 'fixed208_summary.js').read_text()
    assert page.count('function correlatedSummary(update){') == 1
    page = page.replace('function correlatedSummary(update){', block + '\nfunction correlatedSummary(update){')
    (site / c['page_path']).write_text(page)
    write(HERE / 'prepared.json', {'all_saved_JSON_preparation_checks_passed': True,
        'contract_sha256': sha(HERE / 'contract.json'), 'builder_sha256': sha(__file__),
        'shared_utilities_sha256': sha(UTILITY), 'models': 23, 'rows': 92,
        'old_models_preserved': 21, 'old_rows_preserved': 84, 'trajectory_records_unchanged': 277,
        'new_science_calls': 0, 'NPZ_reads': 0, 'publication_performed': False,
        'overlay_sha256': sha(site / c['overlay_path']), 'page_sha256': sha(site / c['page_path']),
        'new_assets_sha256': {str(p.relative_to(site)): sha(p) for p in sorted((site / c['evidence_prefix']).rglob('*')) if p.is_file()}})
    print({'models': 23, 'rows': 92, 'old_trajectory_records': 277, 'science_calls': 0})


if __name__ == '__main__':
    main()
