"""Publish closed observations and preserved failures; metadata only."""
import copy
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parent
TOP = B.parent
R = TOP.parents[1]
H = R / 'docs/atomx_hourly_progress_2026_10_10'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(path):
    return {'path': str(path.relative_to(R)), 'sha256': sha(path)}


def once(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    c = read(B / 'contract.json')
    registry = H / 'progress.json'
    assert sha(__file__) == c['driver_sha256']
    assert sha(registry) == c['registry_before_sha256']
    checks = []

    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})
        assert condition, name

    for rel, digest in c['inputs'].items():
        check('frozen/' + rel, sha(R / rel) == digest)
    previous = read(TOP / 'results_stage025.json')
    record = read(registry)
    old_panel = read(R / record['yb_mcdhf_result'])
    f = TOP / 'complete5f_fixed208_import_repair_preparation'
    failure = read(f / 'actual/failure.json')
    receipt = read(f / 'root_execution.json')
    baseline = read(f / 'actual/baseline_same_point_evidence.json')
    extension = read(f / 'actual/complete5f_same_point_evidence.json')
    failed = [x for x in failure['checks'] if not x['passed']]
    check('5f_original_failure_preserved', receipt['exit_code'] == 1 and
          failure['status'] == 'failed_preserved_no_retry' and
          [x['name'] for x in failed] == ['expected_completed_CI_internal_hS'])
    check('5f_actual_finite_work', failure['completed']['H'] == 2 and
          failure['completed']['CI'] == 2 and failure['completed']['CI_internal_hS'] == 7 and
          failure['completed']['full_scipy_eigh'] == 12 and
          failure['completed']['gradient'] == failure['completed']['optimizer'] == 0)
    check('5f_not_promoted', not extension['original_stationarity_passed'] and
          not extension['expanded_stationarity_evaluated'] and
          not extension['nominal_four_1percent_goal_passed'])
    warm = TOP / 'correlated29_radial_reuse_qualification003_preparation/warm_benchmark_preparation'
    performance = read(warm / 'completion.json')
    check('warm_science_pass_load_fail', performance['closure_all_passed'] and
          performance['all_science_gates_passed'] and not performance['all_host_gates_passed'] and
          not performance['scoped_warm_improvement_certified'])
    atlas = TOP / 'atlas_update007'
    deployment = read(atlas / 'deployment_result.json')
    check('atlas_actual_publish_succeeded', read(atlas / 'publication_completion.json')['all_passed'] and
          deployment['status'] == 'succeeded')
    result = copy.deepcopy(previous)
    panel = copy.deepcopy(old_panel)
    now = datetime.now(timezone.utc)
    result['created_at'] = now.isoformat()
    for rel, digest in c['inputs'].items():
        result['inputs'][rel] = {'path': rel, 'sha256': digest}
    summary = ('原生MCDHF→RCI主线继续，驻值与实验门仍FAIL。固定208新增完整5f双jj空间：'
        '两次RCI已产生1046/1595完整目标块，中间四线位移约−6.40/−3.18/+8.18/+9.10cm⁻¹；'
        '最终h/S调用计数门7vs预计14失败，原整轮保持失败，谱线只作待独立审核的诊断。'
        '新f平均半径约0.30a0，原4f约0.75a0，不能据小位移排除完整f关联。'
        '优先审计实际约束导数/驻值机制、同点屏蔽f种子及计数缓存；不盲增步数或按NIST选解。'
        'CPU暖态配对科学门全部PASS，观测比1.18872但负载门FAIL，未认证加速/GPU/并行效率。'
        '网页v8已发布21模型84行与完整双分支，57浏览器检查PASS。最高水平未达。')
    result['card_summary'] = summary
    result['selection_reason'] = ('有限核内局域f候选仅改变3–9cm⁻¹，不能解释千cm⁻¹差距或排除f径向不足；'
        '原驻值FAIL和未独立核验的同208导数/预条件器优先判别。原计数失败不改门、不自动重跑。')
    result['next_execution_status'] = ('同208驻值/导数与屏蔽f种子两方向源-only准备，须逐项独立准入；'
        '已闭合本小时root原生计算；服务原01:00引用拒绝0作业保留，不冒充原生迭代成功。')
    result['complete5f_fixed208_failed_diagnostic'] = {
        'receipt': bound(f / 'root_execution.json'), 'failure': bound(f / 'actual/failure.json'),
        'baseline': bound(f / 'actual/baseline_same_point_evidence.json'),
        'extension': bound(f / 'actual/complete5f_same_point_evidence.json'),
        'radial_support': bound(f / 'actual/radial_support.json'),
        'run_status': 'failed_original_final_count_gate', 'all_scientific_completion_gates_passed': False,
        'independent_saved_array_qualification_completed': False,
        'completed_calls': failure['completed'], 'recorded_controls_passed': len(failure['checks']) - len(failed),
        'recorded_controls_failed': failed, 'baseline_RMS_cm1': baseline['nominal_four_gap_RMS_cm1'],
        'candidate_RMS_cm1': extension['nominal_four_gap_RMS_cm1'], 'diagnostic_rows': extension['rows'],
        'no_physical_completeness_attribution': True, 'no_stationarity_promotion': True}
    result['warm_performance_actual'] = {k: performance[k] for k in
        ('status', 'closure_checks_count', 'actual_cumulative_recorded_gate_rows_PASS',
         'actual_counter_totals', 'all_science_gates_passed', 'all_host_gates_passed',
         'scoped_warm_improvement_certified', 'median_primary_ratio_observational', 'host_failures')}
    result['warm_performance_actual']['completion'] = bound(warm / 'completion.json')
    result['atlas_published_v8'] = {'deployment': bound(atlas / 'deployment_result.json'),
        'completion': bound(atlas / 'publication_completion.json'), 'models': 21, 'rows': 84,
        'records': 277, 'distinct_saved_point_identities': 262, 'actual_browser_checks': 57,
        'url': deployment['url'] + '/#Yb:1:models', 'audience': 'existing_owner_private_preserved'}
    table = ['|谱线|固定208原空间|增加5f候选|位移cm⁻¹|', '|---|---:|---:|---:|']
    base_rows = {x['label']: x for x in baseline['rows']}
    for row in extension['rows']:
        if row['NIST_nominal_cm1']:
            table.append(f"|{row['label']}|{base_rows[row['label']]['excitation_cm1']:.2f}|"
                         f"{row['excitation_cm1']:.2f}|{row['transition_delta_upper_minus_lower_cm1']:+.2f}|")
    panel['card_summary'] += ('\n5f中间诊断：原计数门FAIL，未独立数组资格；固定轨道非新增MCDHF驻值解。\n'
                              + '\n'.join(table) + '\n' + result['selection_reason'])
    panel['complete5f_failed_diagnostic'] = result['complete5f_fixed208_failed_diagnostic']
    for rel, digest in c['inputs'].items():
        panel['inputs'][rel] = {'path': rel, 'sha256': digest}
    check('current_five_state_and_metric_unchanged', panel['rows'] == old_panel['rows'] and
          panel['stationarity'] == old_panel['stationarity'] and not panel['production_certified'])
    once(B / 'current_card_evidence.json', panel)
    result['inputs']['historical_current_mcdhf_stage024'] = previous['inputs']['current_mcdhf']
    result['inputs']['current_mcdhf'] = bound(B / 'current_card_evidence.json')
    result['current_native_card_evidence'] = bound(B / 'current_card_evidence.json')
    for role, binding in result['inputs'].items():
        check('result_identity/' + role, sha(R / binding['path']) == binding['sha256'])
    once(TOP / 'results_stage026.json', result)
    record['physics_budget_result'] = str((TOP / 'results_stage026.json').relative_to(R))
    record['yb_mcdhf_result'] = str((B / 'current_card_evidence.json').relative_to(R))
    record['updated_at'] = result['created_at']
    record['tasks'][0].update(status='驻值FAIL；5f诊断计数门FAIL', detail=summary)
    record['tasks'][1].update(status='优先物理量级与约束导数', detail=result['selection_reason'])
    record['tasks'][5].update(status='暖态观测1.18872；负载FAIL无认证', detail=result['next_execution_status'])
    record['physics_next_setups'][1] = result['selection_reason']
    record['physics_next_setups'][5] = result['next_execution_status']
    once(B / 'registry_before.json', read(registry))
    (B / 'planner_scope').mkdir()
    once(B / 'planner_scope/progress.json', record)
    sys.path.insert(0, str(H))
    spec = importlib.util.spec_from_file_location('stage026_planner', H / 'parallel_planner.py')
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    decision = {'status': 'implementation_frontier', 'jobs': [], 'frontier': [], 'evidence': []}
    planner.annotate_physics_priority(R, B / 'planner_scope', decision)
    check('public_planner_proposed_current_binding', decision['direction'] == summary and
          decision['evidence'][-1]['sha256'] == sha(TOP / 'results_stage026.json'))
    import sync_card
    card, citations = sync_card.render(record, now)
    check('public_card_actual_render', card['schema'] == '2.0' and len(citations) > 0)
    once(B / 'preview_card.json', card)
    once(B / 'preview_citations.json', citations)
    assert sha(registry) == c['registry_before_sha256']
    temporary = registry.with_suffix('.stage026.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(registry)
    decision = {'status': 'implementation_frontier', 'jobs': [], 'frontier': [], 'evidence': []}
    planner.annotate_physics_priority(R, H, decision)
    check('public_planner_live_binding', decision['direction'] == summary)
    once(B / 'integration_proof.json', {'all_passed': True, 'checks': checks, 'checks_count': len(checks),
        'driver': bound(Path(__file__)), 'contract': bound(B / 'contract.json'),
        'result': bound(TOP / 'results_stage026.json'), 'panel': bound(B / 'current_card_evidence.json'),
        'registry': bound(registry), 'new_science_API_cards': 0,
        'original_failed_science_and_host_gates_preserved': True})
    print(json.dumps({'checks_passed': len(checks), 'result': bound(TOP / 'results_stage026.json'),
                      'registry': bound(registry), 'card_elements': len(card['body']['elements']),
                      'citations': len(citations)}))


if __name__ == '__main__':
    main()
