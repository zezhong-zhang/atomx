"""Append closed numerical diagnoses and actual publication; no science or delivery."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from atomx_report_utils import bound, load, once, sha

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
REPO = TOP.parents[1]
REGISTRY = REPO / 'docs/atomx_hourly_progress_2026_10_10/progress.json'


def main():
    contract = load(HERE / 'contract.json')
    assert sha(Path(__file__)) == contract['driver_sha256']
    assert sha(REGISTRY) == contract['registry_before_sha256']
    checks = []

    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError(name)

    for rel, digest in contract['inputs'].items():
        check('input/' + rel, sha(REPO / rel) == digest)
    previous = load(TOP / 'results_stage022.json')
    qual = TOP / 'correlated29_radial_reuse_qualification003_preparation'
    q, qclose, comparison = (load(qual / p) for p in
                            ('actual/result.json', 'completion.json', 'actual/comparison.json'))
    check('complete_new_centered_epoch', q['all_equivalence_gates_passed']
          and qclose['new_epoch_equivalence_passed'] and comparison['all_passed']
          and len(comparison['checks']) == 1126 and all(c['passed'] for c in comparison['checks']))
    check('old_route_failure_preserved', qclose['original_route002_qualification'] == 'FAIL preserved')
    gpg = TOP / 'correlated29_saved_gpg_decomposition_preparation'
    g, gclose = load(gpg / 'actual/result.json'), load(gpg / 'completion.json')
    check('two_saved_points_only', g['all_declared_controls_passed']
          and len(g['controls']) == 149 and all(c['passed'] for c in g['controls'])
          and g['array_reads_completed'] == 2 and g['scientific_calls'] == 0
          and gclose['all_completion_checks_passed'])
    check('same80_208_metric', [p['global_accepted_step'] for p in g['points']] == [80, 208]
          and all(p['metric_sha256'] == previous['stationarity']['residual_metric_sha256']
                  and not p['stationarity_passed'] for p in g['points']))
    pub = load(TOP / 'atlas_update006/root_publication.json')
    check('actual_v7_publication', pub['deployment_status'] == 'succeeded'
          and pub['version_number'] == 7 and pub['models'] == 19 and pub['rows'] == 76
          and pub['distinct_global_points'] == 209 and pub['audience_unchanged'])
    manual, hourly = (load(TOP / p / 'delivery_proof.json') for p in ('delivery_stage022', 'delivery_hour24'))
    check('actual_saved_deliveries', manual['passed'] and hourly['passed']
          and manual['cycle'] == 'mcdhf-stage022' and hourly['cycle'] == '2026-10-11T16'
          and manual['actual_readback_calls_by_root'] == hourly['actual_readback_calls_by_root'] == 1)
    result = copy.deepcopy(previous)
    result['created_at'] = datetime.now(timezone.utc).isoformat()
    for rel, digest in contract['inputs'].items():
        result['inputs'][rel] = {'path': rel, 'sha256': digest}
    result['historical_stage022_primary_view'] = {k: previous[k] for k in
                                                ('card_summary', 'selection_reason', 'next_execution_status')}
    result['centered_radial_route_qualification003'] = {
        'result': bound(qual / 'actual/result.json'), 'completion': bound(qual / 'completion.json'),
        'root_execution': bound(qual / 'root_execution.json'),
        'gate_counts': [406, 1294, 241, 1126], 'calls_H_CI_gradient_hS_eigh': [2, 2, 2, 28, 17],
        'worst_complete_cluster_projector_difference': max(c['error'] for c in comparison['checks']
                                                         if c['name'].startswith('signed_cluster_')),
        'scope': 'Private same-operator/1046-root new centered source epoch; old FAIL preserved. '
                 'No optimizer, shared adoption, experimental accuracy or speedup certified by this run.'}
    terminal = g['points'][-1]
    shares = {group: sum(r['residual_squared_coordinate_share'] for r in terminal['per_orbital_columns']
                         if r['group'] == group) for group in terminal['group_order']}
    result['saved_gpg_decomposition'] = {
        'result': bound(gpg / 'actual/result.json'), 'completion': bound(gpg / 'completion.json'),
        'controls': 149, 'source_map_sha256': terminal['source_map_sha256'],
        'metric_sha256': terminal['metric_sha256'], 'point_steps': [80, 208],
        'terminal_coordinate_residual_squared_shares': shares,
        'terminal_TMT_signed_work_matrix': terminal['TMT_group_signed_work_matrix'],
        'terminal_TMT_residual_cross_matrix': terminal['TMT_group_residual_cross_matrix'],
        'core_dominance_supported': False, 'physical_error_bound': False,
        'scope': 'Complete original-metric columns and full tangent TMT raw-mask cross terms; '
                 'coordinate shares are distinct from independently projected group drivers.'}
    result['latest_hourly_delivery'] = {'proof': bound(TOP / 'delivery_hour24/delivery_proof.json'),
                                        'message_id': hourly['message_id'], 'actual_message_link': hourly['actual_message_link'],
                                        'scope': 'Original00HKT card reported stage021/global80 before005 closure; '
                                                 'service job=null/jobs=[], no new native launch from that cycle.'}
    result['latest_manual_delivery'] = {'proof': bound(TOP / 'delivery_stage022/delivery_proof.json'),
                                        'message_id': manual['message_id'], 'actual_message_link': manual['actual_message_link'],
                                        'scope': 'Stage022/current208 actual card, followed by separately bound closed003 qualification.'}
    result['atlas_current_publication'] = {'proof': bound(TOP / 'atlas_update006/root_publication.json'),
                                           'version': 7, 'models': 19, 'rows': 76, 'distinct_global_points': 209,
                                           'url': pub['url'], 'same_original_FAIL_gates': True}
    result['selection_reason'] = ('149门两点梯度诊断：208的残差平方new5占53.38%、bare5占25.67%、core19占20.95%；'
                                  'p3/2完整κ块主导，三组驱动力强交叉抵消。先在同208/原3451 metric/同物理源下'
                                  '做history10与40各32步有界比较，40仍未装满；保留耦合和全部接受/驻值门。'
                                  '其后分辨共同EOL、虚f/g与缺失相关类别、同点RCI修正，不以NIST拟合或选择端点。')
    result['next_execution_status'] = ('最小匹配history10/40两路各32步源码准备中，精确root seal后执行；'
                                      '新的CI centered/helper及径向复用完整资格已PASS，共享采用与受控性能比较另记账。')
    result['card_summary'] = ('Atomx原生MCDHF→RCI：29轨道/1046目标扇区累计208步；残差9.55e−4、指标1.46e−6Ha，'
                              '原两驻值门仍FAIL，四线RMS2054.32cm⁻¹，实验门仍FAIL。149门梯度分解定位p3/2及新增虚轨道，'
                              '全signed交叉抵消保留；下一同点比较history10/40各32步。完整CI径向复用新epoch的'
                              '406/1294/241/1126门通过，旧FAIL保留，cold时钟不称加速。网页v7已发布19模型76行/209独立点，'
                              '可下载全部轨迹与来源。最高水平尚未达到，卡片后继续有界gap驱动迭代。')
    for key in ('computed_run', 'stationarity', 'correlated29_optimizer_run005'):
        check('native208_unchanged/' + key, result[key] == previous[key])
    registry = load(REGISTRY)
    registry['updated_at'] = result['created_at']
    registry['physics_budget_result'] = str((TOP / 'results_stage023.json').relative_to(REPO))
    registry['tasks'][0].update(status=result['expanded29_optimizer_status'], detail=result['card_summary'])
    registry['tasks'][1].update(status='固定模型先驻值，按量级分辨物理缺口', detail=result['selection_reason'])
    registry['tasks'][5].update(status='完整新CI径向复用资格PASS；采用与受控时序分开', detail=result['next_execution_status'])
    registry['physics_next_setups'][1] = result['selection_reason']
    registry['physics_next_setups'][5] = result['next_execution_status']
    once(HERE / 'registry_before.json', load(REGISTRY))
    once(TOP / 'results_stage023.json', result)
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    once(HERE / 'integration_proof.json', {'all_passed': True, 'checks': checks,
         'checks_count': len(checks), 'new_science_calls': 0, 'API_calls': 0, 'cards_sent': 0,
         'result': bound(TOP / 'results_stage023.json'), 'registry': bound(REGISTRY),
         'driver': bound(Path(__file__)), 'contract': bound(HERE / 'contract.json')})
    print(json.dumps({'checks': len(checks), 'all_passed': True,
                      'result': bound(TOP / 'results_stage023.json'), 'registry': bound(REGISTRY)}))


if __name__ == '__main__':
    main()
