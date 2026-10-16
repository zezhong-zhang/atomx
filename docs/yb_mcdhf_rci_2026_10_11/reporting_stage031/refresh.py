"""Root-only bounded metadata refresh; preserve stage030 scientific fields."""
import argparse
import copy
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parent
T = B.parent
R = T.parents[1]
UTILITY = T / 'reporting_stage030/refresh.py'
spec = importlib.util.spec_from_file_location('stage030_metadata_utilities', UTILITY)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
read, sha, bound, once = shared.read, shared.sha, shared.bound, shared.once
H = R / 'docs/atomx_hourly_progress_2026_10_10'


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--execute', action='store_true')
    args.add_argument('--contract-sha256')
    options = args.parse_args()
    if not options.execute:
        print('SOURCE-only prepared; root exact contract review required, no writes.')
        return
    contract = read(B / 'contract.json')
    assert options.contract_sha256 == sha(B / 'contract.json'), 'Changed exact root-reviewed contract'
    assert sha(Path(__file__)) == contract['driver_sha256']
    registry = H / 'progress.json'
    assert sha(registry) == contract['registry_before_sha256'], 'Registry changed; refuse stale integration'
    for name, expected in contract['inputs'].items():
        assert sha(R / name) == expected, name
    assert not any((B / n).exists() for n in contract['exclusive_outputs'])
    original = read(T / 'results_stage030.json')
    original_panel = read(T / 'reporting_stage030/current_card_evidence.json')
    result, panel = copy.deepcopy(original), copy.deepcopy(original_panel)
    publication = read(R / contract['roles']['publication'])
    gpu = read(R / contract['roles']['GPU_preparation'])
    gradient = read(R / contract['roles']['gradient31_actual'])
    receipt = read(R / contract['roles']['gradient31_receipt'])
    assert publication['all_publication_completion_metadata_checks_passed'] and publication['version_number'] == 9
    assert gpu['all_preparation_checks_passed'] and all(v == 0 for v in gpu['actual_preparation_calls'].values())
    assert gradient['all_declared_controls_passed'] and len(gradient['checks']) == 438
    assert all(c['passed'] for c in gradient['checks']) and receipt['status'] == 'foreground_closed' and receipt['exit_code'] == 0
    assert gradient['full_gradient_is_HF_partial'] and not gradient['finite_CI_response_qualified']
    assert not gradient['expanded_MCDHF_optimized'] and gradient['no_old3451_residual_improvement_claim']
    g = gradient['action_metrics']['full_normalized']
    assert g['metric_residual'] == .005162180923375838 and g['indicator_Ha'] == 9.089562518880953e-5
    assert g['strict_thresholds'] == [1e-7, 1e-8] and not g['energy_or_spectrum_error_bound']
    metadata_note = ('网页v9已发布23模型92行，原21模型84行与277轨迹保留，仍为私有访问。'
        'GPU私有batch仅源码/descriptor准备，0数值或CUDA执行，无加速认证。'
        '31轨道完整有符号HF-partial梯度实际438控制PASS；31未优化。新3689诊断度量残差'
        '0.00516218>1e−7、g·pg/2指示量9.08956e−5 Ha>1e−8，均FAIL，指示量不是能级误差界。'
        '新度量不能与旧3451残差直接比较改善；finite-CI-response导数及4f/f组合旋转FD仍待资格。')
    panel['card_summary'] += '\n' + metadata_note
    for relative in contract['roles'].values():
        panel['inputs'][relative] = contract['inputs'][relative]
    assert all(panel[k] == v for k, v in original_panel.items() if k not in ('card_summary', 'inputs'))
    old_tail = '网页v8已有21模型84行。'
    assert result['card_summary'].count(old_tail) == 1
    result['card_summary'] = result['card_summary'].replace(old_tail, '') + metadata_note
    result['created_at'] = datetime.now(timezone.utc).isoformat()
    result['inputs']['historical_current_mcdhf_stage030'] = original['inputs']['current_mcdhf']
    result['historical_atlas_current_publication_stage030'] = copy.deepcopy(original['atlas_current_publication'])
    result['atlas_current_publication'] = {'proof': bound(R / contract['roles']['publication']), 'version': 9,
        'models': 23, 'rows': 92, 'records': 277, 'URL': publication['URL'], 'same_original_FAIL_gates': True}
    result['GPU_private_source_preparation'] = {'proof': bound(R / contract['roles']['GPU_preparation']),
        'source_checks_PASS': 63, 'descriptor_checks_PASS': 36, 'documentation_checks_PASS': 36,
        'actual_numerical_GPU_calls': 0, 'numerically_qualified': False, 'speedup_certified': False}
    result['correlated31_fullsigned_gradient_discriminator'] = {
        'actual': bound(R / contract['roles']['gradient31_actual']), 'receipt': bound(R / contract['roles']['gradient31_receipt']),
        'actual_runtime_controls_PASS': 438, 'completed_calls': gradient['completed'],
        'source_map_sha256': gradient['source_map_sha256'], 'new3689_metric_sha256': gradient['new_metric_sha256'],
        'new_geometry_sha256': gradient['new_geometry_sha256'], 'full_normalized_metrics': g,
        'full_gradient_is_HF_partial': True, 'finite_CI_response_qualified': False,
        'expanded31_MCDHF_optimized': False, 'original_stationarity_passed': False, 'experimental_gate_passed': False,
        'no_old3451_residual_improvement_claim': True, 'canonical_and_recovered_are_separate_points': True,
        'max_canonical_vs_public_wave_delta': max(row['canonical_vs_public_max'] for row in gradient['recovery']),
        'max_recovered_vs_canonical_transition_delta_cm1': max(abs(row['delta_transition_cm1']) for row in gradient['points']['recovered']['rows']),
        'next_combined4f_f_rotation_FD': 'pending; no actual finite-response qualification in this report'}
    for role, relative in contract['roles'].items():
        result['inputs']['stage031_' + role] = bound(R / relative)
    allowed = {'created_at', 'inputs', 'card_summary', 'atlas_current_publication', 'current_native_card_evidence'}
    assert all(result[k] == v for k, v in original.items() if k not in allowed)
    record = read(registry)
    once(B / 'registry_before.json', record)
    once(B / 'current_card_evidence.json', panel)
    result['inputs']['current_mcdhf'] = bound(B / 'current_card_evidence.json')
    result['current_native_card_evidence'] = bound(B / 'current_card_evidence.json')
    once(T / 'results_stage031.json', result)
    record.update(physics_budget_result=str((T / 'results_stage031.json').relative_to(R)),
                  yb_mcdhf_result=str((B / 'current_card_evidence.json').relative_to(R)), updated_at=result['created_at'])
    record['tasks'][4].update(status='远程v9已发布23模型92行；私有访问',
        detail='原21模型84行、277轨迹和6019NIST载荷保留。新增两组固定31/1595 RCI未优化；原驻值与实验门FAIL。')
    record['spectra'][0]['url'] = publication['URL']
    record['atlas']['url'] = publication['URL']
    record['atlas']['scope'] += ' v9共23模型92行，核场与屏蔽f固定31结果及原208FD资格分开。'
    sys.path.insert(0, str(H))
    import sync_card
    card, citations = sync_card.render(record, datetime.now(timezone.utc))
    assert card['schema'] == '2.0'
    content = '\n'.join(v.get('content', '') for v in card['body']['elements'])
    assert all(text in content for text in ['23模型92行', '438控制PASS', '0.00516218', '0数值或CUDA执行', '不是能级误差界'])
    once(B / 'preview_card.json', card)
    once(B / 'preview_citations.json', citations)
    assert sha(registry) == contract['registry_before_sha256'], 'Concurrent registry change; refuse'
    temporary = registry.with_suffix('.stage031.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(registry)
    planner_spec = importlib.util.spec_from_file_location('stage031_planner', H / 'parallel_planner.py')
    planner = importlib.util.module_from_spec(planner_spec)
    planner_spec.loader.exec_module(planner)
    decision = {'status': 'implementation_frontier', 'jobs': [], 'frontier': [], 'evidence': []}
    planner.annotate_physics_priority(R, H, decision)
    assert decision['direction'] == result['card_summary']
    once(B / 'integration_proof.json', {'all_passed': True, 'result': bound(T / 'results_stage031.json'),
        'panel': bound(B / 'current_card_evidence.json'), 'registry': bound(registry),
        'all_scientific_stage030_fields_preserved': True, 'direct_new_citation_count': 4,
        'render_elements': len(card['body']['elements']), 'render_citations': len(citations),
        'new_science_arrays_API_cards': 0, 'original_stationarity_experimental_FAIL_preserved': True})
    print(json.dumps({'status': 'metadata_stage031_integrated', 'elements': len(card['body']['elements']), 'citations': len(citations)}))


if __name__ == '__main__':
    main()
