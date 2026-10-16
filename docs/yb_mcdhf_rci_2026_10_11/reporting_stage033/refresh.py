"""SOURCE-only stage033; root-sealed integration consumes closed separate-scope evidence."""
import argparse
import copy
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).absolute().parent
R = next(p for p in B.parents if (p / 'pyproject.toml').is_file())
T = B.parent
H = R / 'docs/atomx_hourly_progress_2026_10_10'
spec = importlib.util.spec_from_file_location('stage033_existing_metadata_utilities', T / 'reporting_stage030/refresh.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
read, sha, bound, once = shared.read, shared.sha, shared.bound, shared.once


def load_closed_GPU(binding):
    """Closed JSON only: no early status reads and no NPZ access or fabricated PASS."""
    if not isinstance(binding, dict) or set(binding) != {'result', 'receipt', 'closure'}:
        raise ValueError('Actual closed GPU successor roles still deferred')
    for row in binding.values():
        if (not isinstance(row, dict) or set(row) != {'path', 'sha256'}
                or not row['path'].startswith('docs/yb_mcdhf_rci_2026_10_11/gpu_mcdhf_next_step208/qualification_representation_repair001/')
                or not row['path'].endswith('.json')):
            raise ValueError('Exact actual successor JSON roles required')
    receipt = read(R / binding['receipt']['path'])
    if receipt.get('status') != 'foreground_closed':
        raise ValueError('Do not read science result before actual foreground closes')
    for role, row in binding.items():
        if sha(R / row['path']) != row['sha256']:
            raise ValueError('Actual closed GPU role changed: ' + role)
    result, closure = read(R / binding['result']['path']), read(R / binding['closure']['path'])
    assert closure['status'] == 'CLOSED_JSON_METADATA_ONLY_NO_NUMERICAL_REPLAY'
    assert closure['all_metadata_checks_passed'] and all(c['passed'] for c in closure['checks'])
    assert closure['actual_result'] == binding['result'] and closure['actual_receipt'] == binding['receipt']
    assert closure['NPZ_reads_or_rehashes'] == closure['new_physical_calls'] == 0
    if receipt['exit_code'] not in (0, 1) or result.get('all_declared_controls_passed') != (receipt['exit_code'] == 0):
        raise ValueError('Actual successor exit/status must be retained exactly')
    qualified = receipt['exit_code'] == 0
    if qualified:
        assert all(c['passed'] for c in result['checks'])
        assert set(result['backends']) == {'NumBa_CPU1', 'NumBa_CPU4', 'CUDA'}
        for name, count in (('H', 1), ('CI', 1), ('public_full_gradient', 1), ('private_full_gradient', 3)):
            assert result['calls'][name]['completed'] == count
        assert not result['full_H_CI_on_GPU'] and not result['controlled_speedup_qualified']
    else:
        assert result['status'] == 'failed_partial'
    return result, receipt, closure, qualified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--contract-sha256')
    parser.add_argument('--completion-binding', type=Path)
    options = parser.parse_args()
    if not options.execute:
        print('SOURCE-only stage033 prepared; closed successor/root review still required, no registry writes.')
        return
    contract = read(B / 'contract.json')
    assert options.contract_sha256 == sha(B / 'contract.json')
    assert sha(Path(__file__)) == contract['driver_sha256']
    admission = read(options.completion_binding)
    assert admission['execution_authorized_by_root'] is True
    assert admission['driver_sha256'] == sha(Path(__file__)) and admission['contract_sha256'] == sha(B / 'contract.json')
    assert admission['registry_before_sha256'] == contract['registry_before_sha256']
    assert sha(H / 'progress.json') == contract['registry_before_sha256']
    for name, expected in contract['inputs'].items():
        assert sha(R / name) == expected, name
    assert not any((B / n).exists() for n in contract['exclusive_outputs'])
    assert not (T / 'results_stage033.json').exists()
    gpu, gpu_receipt, gpu_closure, qualified = load_closed_GPU(admission.get('closed_GPU_successor'))
    assert admission['closed_GPU_successor'] == contract['closed_GPU_successor']
    old = read(R / contract['roles']['base_result'])
    old_panel = read(R / contract['roles']['base_panel'])
    result, panel = copy.deepcopy(old), copy.deepcopy(old_panel)
    fd = read(R / contract['roles']['FD_actual'])
    fd_receipt = read(R / contract['roles']['FD_receipt'])
    fd_closure = read(R / contract['roles']['FD_closure'])
    assert fd_receipt['status'] == 'foreground_closed' and fd_receipt['exit_code'] == 0
    assert fd['all_declared_controls_passed'] and len(fd['checks']) == 730 and all(c['passed'] for c in fd['checks'])
    assert fd['single_direction_finite_measurement_only'] and not fd['full31_space_certificate']
    assert not fd['original_stationarity_passed'] and not fd['expanded_MCDHF_optimized']
    assert set(fd['derivative_domains']) == {'1e-05', '5e-06'}
    for row in fd['derivative_domains'].values():
        assert row['ordinary_reported']['passed']
        assert all(v['passed'] for v in row['independent_centered'])
    assert fd_closure['actual_live_controls_count'] == fd_closure['actual_live_controls_passed'] == 730
    assert fd_closure['metadata_checks_count'] == len(fd_closure['metadata_checks']) == 81
    assert fd_closure['all_completion_metadata_checks_passed'] and all(c['passed'] for c in fd_closure['metadata_checks'])
    assert not fd_closure['independent_numerical_array_replay_performed']
    owner_metadata = {}
    for family, passed, failed, exit_code in (('primary', 63, 0, 0), ('integration', 26, 0, 0), ('mixed', 42, 2, 1)):
        receipt_path = R / contract['roles'][family + '_receipt']
        receipt = read(receipt_path)
        assert receipt['status'] == 'foreground_closed' and receipt['exit_code'] == exit_code
        log_path = R / contract['roles'][family + '_log']
        assert sha(log_path) == receipt['log_sha256']
        text = log_path.read_text()
        assert re.search(r'\b' + str(passed) + r' passed\b', text)
        if failed:
            assert re.search(r'\b' + str(failed) + r' failed\b', text)
        owner_metadata[family] = dict(receipt=bound(receipt_path), log=bound(log_path), passed=passed,
                                      failed=failed, entire_suite_passed=exit_code == 0, no_MCDHF_production_certification=True)
    calls = read(R / contract['roles']['mixed_calls'])
    assert calls['completed'] == contract['mixed_expected_completed_calls']
    hourly = read(R / contract['roles']['hour28_service'])
    hourly_capture = read(R / contract['roles']['hour28_original_capture'])
    hourly_authorization = read(R / contract['roles']['hour28_authorization'])
    assert hourly['status'] == 'completed' and len(hourly['jobs']) == len(hourly['job_ids']) == 4
    assert all(job['status'] == 'completed' for job in hourly['jobs'])
    assert all('yb' not in job_id.lower() and 'mcdhf' not in job_id.lower() for job_id in hourly['job_ids'])
    assert not hourly_capture['passed'] and hourly_capture['actual_readback_calls_by_root'] == 1
    assert hourly_authorization['passed'] and hourly_authorization['new_send_calls'] == 0
    original_gpu = read(R / contract['roles']['original_GPU_actual'])
    original_gpu_receipt = read(R / contract['roles']['original_GPU_receipt'])
    assert original_gpu_receipt['status'] == 'foreground_closed' and original_gpu_receipt['exit_code'] == 1
    assert original_gpu['status'] == 'failed_partial' and original_gpu['failure'] == 'saved_actual_owner_header'
    assert len(original_gpu['checks']) == 197 and sum(c['passed'] for c in original_gpu['checks']) == 196
    assert original_gpu['calls']['public_full_gradient']['completed'] == 0 and original_gpu['calls']['CUDA_context']['completed'] == 0
    GPU_note = ('GPU私有三后端同点完整梯度一致性已通过；无受控加速认证或生产采用。' if qualified else
                'GPU私有资格后继仍未通过，实际失败与未执行项已保留；无GPU或加速认证。')
    metadata_note = ('网页v9已发布23模型92行，仍为私有访问。31轨道原生约束组合方向两尺度导数门通过，'
                     '仅一个方向；驻值和实验门仍FAIL。共享保留H接口基础和集成回归通过，但混合CI小夹具仍有失败，待定位。'
                     + GPU_note + '原GPU身份表示比较失败保留；不把新3689指标与旧3451比较改善。'
                     '04HKT小时服务已完成四项自动诊断，Yb MCDHF自动作业为0；旧回读格式失败保留。')
    old_note = contract['stage032_metadata_note']
    for obj in (result, panel):
        assert obj['card_summary'].endswith(old_note)
        obj['card_summary'] = obj['card_summary'][:-len(old_note)] + metadata_note
    panel['inputs'].update({relative: bound(R / relative) for relative in contract['roles'].values()})
    for row in admission['closed_GPU_successor'].values():
        panel['inputs'][row['path']] = copy.deepcopy(row)
    for binding in panel['inputs'].values():
        assert isinstance(binding, dict) and {'path', 'sha256'} <= binding.keys()
        assert isinstance(binding['path'], str) and isinstance(binding['sha256'], str)
        assert sha(R / binding['path']) == binding['sha256']
    assert all(panel[k] == v for k, v in old_panel.items() if k not in ('card_summary', 'inputs'))
    result['historical_stage032_card_summary'] = old['card_summary']
    result['historical_stage032_selection_reason'] = old['selection_reason']
    result['historical_stage032_next_execution_status'] = old['next_execution_status']
    result['selection_reason'] = ('先定位当前共享混合CI小夹具的源epoch与粗步长失败，再审阅31同点优化准入。'
        '私有f043单方向导数和CPU/CUDA径向梯度一致性已资格，驻值/实验仍FAIL；不据此采用当前共享代码。'
        '04HKT四项自动诊断已完成，Yb MCDHF自动作业为0；历史零自动作业仅保留在旧记录。')
    result['next_execution_status'] = 'Current shared mixed fixture FAIL under audit; no31 optimizer/adoption without root same-point review; privatef043 kernel qualification only.'
    result['historical_stage032_fullsigned31_scope'] = copy.deepcopy(old['correlated31_fullsigned_gradient_discriminator'])
    result['correlated31_fullsigned_gradient_discriminator']['next_combined4f_f_rotation_FD'] = 'actual two h PASS for one combined direction; no full31-space certificate'
    result['correlated31_combined_directional_qualification'] = {
        'actual': bound(R / contract['roles']['FD_actual']), 'receipt': bound(R / contract['roles']['FD_receipt']),
        'closure': bound(R / contract['roles']['FD_closure']), 'actual_live_controls_PASS': 730,
        'metadata_closure_controls_PASS': 81, 'derivative_domains': fd['derivative_domains'],
        'source_map_sha256': fd['source_map_sha256'], 'metric_sha256': fd['new_metric_sha256'],
        'single_combined_direction_only': True, 'fullspace_or_stationarity_or_experiment_certificate': False}
    result['shared_retained_transverse_owner_regression'] = {
        'actual_suites': owner_metadata, 'all_regressions_passed': False,
        'mixed_completed_calls': calls['completed'], 'mixed_failures': contract['mixed_failures'],
        'private_f043_native31_FD_is_separate_epoch_and_space': True, 'current31_optimizer_or_production_adoption_authorized': False}
    result['private31_batch_actual'] = {
        'original_failure': dict(actual=bound(R / contract['roles']['original_GPU_actual']),
                                 receipt=bound(R / contract['roles']['original_GPU_receipt']), scope='1H1CI0g0GPU owner metadata failure'),
        'successor_roles': admission['closed_GPU_successor'], 'actual_successor_status': gpu['status'],
        'complete_raw_kernel_consistency_qualified': qualified, 'source_map_sha256': contract['private_f043_source_map_sha256'],
        'actual_calls': gpu['calls'], 'elapsed_uncontrolled_s': gpu_receipt['elapsed_uncontrolled_s'],
        'whole_H_CI_GPU': False, 'controlled_speedup': False, 'current_shared_adoption_or_optimizer_qualified': False,
        'closure_metadata_only_not_independent_array_replay': True}
    result['hour28_actual_service'] = {'service': bound(R / contract['roles']['hour28_service']),
        'original_capture_FAIL': bound(R / contract['roles']['hour28_original_capture']),
        'existing_nine_operation_format_authorization': bound(R / contract['roles']['hour28_authorization']),
        'completed_automatic_probe_ids': hourly['job_ids'], 'Yb_MCDHF_automatic_jobs': 0,
        'no_global_zero_scientific_job_claim': True, 'this_metadata_routine_new_science_calls': 0,
        'no_resend_or_readback_retry': True}
    result['created_at'] = datetime.now(timezone.utc).isoformat()
    for role, relative in contract['roles'].items():
        result['inputs']['stage033_' + role] = bound(R / relative)
    assert all(isinstance(v, dict) and {'path', 'sha256'} <= v.keys() for v in result['inputs'].values())
    allowed = {'created_at', 'card_summary', 'inputs', 'current_native_card_evidence', 'correlated31_fullsigned_gradient_discriminator', 'selection_reason', 'next_execution_status'}
    assert all(result[k] == v for k, v in old.items() if k not in allowed)
    record = read(H / 'progress.json')
    once(B / 'registry_before.json', record)
    once(B / 'current_card_evidence.json', panel)
    result['inputs']['current_mcdhf'] = bound(B / 'current_card_evidence.json')
    result['current_native_card_evidence'] = bound(B / 'current_card_evidence.json')
    once(T / 'results_stage033.json', result)
    record.update(physics_budget_result=str((T / 'results_stage033.json').relative_to(R)),
                  yb_mcdhf_result=str((B / 'current_card_evidence.json').relative_to(R)), updated_at=result['created_at'])
    record['tasks'][0].update(status='31组合方向两尺度导数门通过；驻值/实验仍FAIL',
                            detail='原f043空间单方向资格；当前共享混合CI夹具仍失败，先定位源/数值差。')
    record['tasks'][5].update(status='私有完整31梯度后端一致性通过；未性能认证' if qualified else '私有GPU后继资格失败；证据保留',
                            detail='完整有符号同点比较，HCI/角结构/指标仍CPU；无生产采用或受控加速声明。')
    sys.path.insert(0, str(H))
    import sync_card
    card, citations = sync_card.render(record, datetime.now(timezone.utc))
    assert card['schema'] == '2.0'
    content = '\n'.join(v.get('content', '') for v in card['body']['elements'])
    assert all(text in content for text in ('两尺度导数门通过', '混合CI小夹具仍有失败', '实验门仍FAIL'))
    once(B / 'preview_card.json', card)
    once(B / 'preview_citations.json', citations)
    assert sha(H / 'progress.json') == contract['registry_before_sha256']
    temporary = H / 'progress.stage033.tmp'
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(H / 'progress.json')
    planner_spec = importlib.util.spec_from_file_location('stage033_existing_planner', H / 'parallel_planner.py')
    planner = importlib.util.module_from_spec(planner_spec)
    planner_spec.loader.exec_module(planner)
    decision = {'status': 'implementation_frontier', 'jobs': [], 'frontier': [], 'evidence': []}
    planner.annotate_physics_priority(R, H, decision)
    assert decision['direction'] == result['card_summary']
    once(B / 'integration_proof.json', dict(all_passed=True, result=bound(T / 'results_stage033.json'),
        panel=bound(B / 'current_card_evidence.json'), registry=bound(H / 'progress.json'),
        all_original_stage032_spectrum_stationarity_FAIL_values_preserved=True,
        private_f043_directional_and_shared_mixed_fixture_domains_separate=True,
        original_GPU_failure_preserved=True, actual_closed_GPU_successor_only=True,
        new_science_NPZ_API_card_calls=0, hourly_timer_unchanged=True))
    print(json.dumps(dict(status='metadata_stage033_integrated', private31_backend_consistency=qualified)))


if __name__ == '__main__':
    main()
