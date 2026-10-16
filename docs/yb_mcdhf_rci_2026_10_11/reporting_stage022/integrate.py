"""Saved closed native continuation report; scientific/source epochs remain separate."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
REPO = HERE.parents[2]
REGISTRY = REPO / 'docs/atomx_hourly_progress_2026_10_10/progress.json'
OPT = TOP / 'correlated29_optimization_run005_preparation'
SHIFT = TOP / 'correlated29_radial_reuse_qualification002_preparation/saved_common_shift_diagnosis_preparation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def bound(path):
    return {'path': str(path.relative_to(REPO)), 'sha256': sha(path)}


def once(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    contract = load(HERE / 'contract.json')
    assert sha(Path(__file__)) == contract['driver_sha256']
    assert sha(REGISTRY) == contract['registry_before_sha256']
    assert sha(TOP / 'results_stage021.json') == contract['previous_sha256']
    checks = []

    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError(name)

    for rel, digest in contract['inputs'].items():
        check('source/' + rel, sha(REPO / rel) == digest)
    previous = load(TOP / 'results_stage021.json')
    actual, execution, closure = (load(OPT / x) for x in
        ('actual/result.json', 'root_execution.json', 'completion.json'))
    check('foreground_completed', execution['status'] == 'foreground_closed'
          and execution['exit_code'] == 0 and actual['error'] is None)
    check('all_runtime_controls', len(actual['controls']) == 8233
          and all(c['passed'] for c in actual['controls']))
    check('closure_and_all_acceptances', closure['all_completion_metadata_checks_passed']
          and closure['all_qualified_finite_stored_H_acceptances_qualified'])
    check('same_native_model_epoch', actual['source_map_sha256']
          == previous['correlated29_optimizer_run004']['source_map_sha256'])
    diag = actual['optimizer_final_diagnostics']
    check('original_residual_metric', diag['residual_metric_sha256']
          == previous['stationarity']['residual_metric_sha256'])
    check('calls_and_steps', actual['attempted_calls'] == actual['completed_calls']
          and all(actual['completed_calls'][k] == v for k, v in
                  {'H': 187, 'CI': 187, 'gradient': 129, 'optimizer': 1,
                   'validation_Dirac': 7, 'initial_hS': 7, 'CI_internal_hS': 7}.items())
          and actual['total_accepted_steps'] == 208 and len(actual['trajectory']) == 129)
    check('failed_original_physics_gates', not actual['stationarity_passed']
          and not actual['experimental_certification'] and not actual['physical_completeness_certified'])
    points = []
    for row in actual['trajectory']:
        path = REPO / row['manifest_path']
        check('point_hash_' + str(row['iteration']), sha(path) == row['manifest_sha256'])
        point = load(path)
        check('accepted_' + str(row['iteration']), point['checkpoint_qualified']
              and (row['iteration'] == 0 or point['exact_acceptance']['qualified']))
        points.append({**row, 'targets': point['targets'], 'multiplets': point['multiplets']})
    check('terminal_not_gap_selected', actual['terminal_point_selected_without_gap_optimization']
          and closure['terminal_checkpoint'] == points[-1]['manifest_path'])
    shift, shift_execution = load(SHIFT / 'actual/result.json'), load(SHIFT / 'root_execution.json')
    check('bounded_saved_shift', shift_execution['exit_code'] == 0
          and shift['all_diagnostic_gates_passed'] and len(shift['checks']) == 1066
          and all(c['passed'] for c in shift['checks'])
          and shift['counts']['scipy_eigh_completed'] == 10
          and shift['counts']['H'] == shift['counts']['gradient'] == 0
          and not shift['route_qualification_passed'])
    result = copy.deepcopy(previous)
    result['historical_stage021_primary_view'] = {k: previous[k] for k in
        ('computed_run', 'stationarity', 'correlation_scope', 'numerical_certification_scope',
         'card_summary', 'next_execution_status', 'selection_reason')}
    result['created_at'] = datetime.now(timezone.utc).isoformat()
    result['inputs']['previous_stage021'] = bound(TOP / 'results_stage021.json')
    for rel, digest in contract['inputs'].items():
        result['inputs'][rel] = {'path': rel, 'sha256': digest}
    result['computed_run'] = 'Atomx native f043 correlated29/1046 DC, global208 accepted updates'
    result['stationarity'] = {k: diag[k] for k in ('converged', 'orbital_residual',
        'orbital_tolerance', 'estimated_energy_error', 'energy_tolerance',
        'residual_metric_sha256', 'termination_reason')}
    result['stationarity'].update(iterations=208, same_metric_local_updates=128,
        scope='Unchanged original3451 metric and frozen source/P/UID/EOL; secant history cleared by original public resume interface.')
    result['numerical_certification_scope'] = ('Run005 completed128 new/global208 updates,129 qualified points/8233 recorded controls/exact stored-H acceptances. Original residual/indicator and experiment gates FAIL. This does not certify model completeness or performance; previous source epochs and failures remain distinct.')
    result['correlated29_optimizer_run005'] = {'result': bound(OPT / 'actual/result.json'),
        'execution': bound(OPT / 'root_execution.json'), 'completion': bound(OPT / 'completion.json'),
        'source_map_sha256': actual['source_map_sha256'], 'actual_calls': actual['completed_calls'],
        'controls_passed': 8233, 'new_accepted_steps': 128, 'total_accepted_steps': 208,
        'points': points, 'optimizer_final_diagnostics': diag,
        'exact_guard_budget': actual['exact_guard_budget'],
        'elapsed_uncontrolled_seconds': execution['elapsed_uncontrolled_seconds'],
        'stationarity_passed': False, 'physical_completeness_certified': False,
        'experimental_certification': False, 'speedup_certified': False}
    result['saved_common_shift_diagnosis'] = {'result': bound(SHIFT / 'actual/result.json'),
        'execution': bound(SHIFT / 'root_execution.json'), 'checks_passed': 1066,
        'scipy_eigh_calls': 10, 'physical_H_CI_construction_gradient_calls': 0,
        'full_root_cluster_max_projector_error': max(p['projector_max_abs'] for p in shift['shifted_complete_cluster_spans']),
        'column296_projector_error': shift['column296_shifted_cluster']['projector_max_abs'],
        'scope': 'Fixed common alpha applied only to saved-H complete1046 eigensolves; all original cluster/residual/Gram gates kept. Original route002FAIL not upgraded; private new source qualification003 pending.'}
    result['expanded29_optimizer_status'] = '原生29/1046累计208步；新增128步全部接受门通过，原两驻值门及实验门仍FAIL。'
    result['selection_reason'] = ('本轮同原metric残差0.00264137→0.000955327，指标1.45944e−6Ha；四线RMS2105.10→2054.32cm−1。先按保存g/pg和原投影分解core/valence/newvirtual驱动力及交叉项，再选择预条件或分阶段弛豫。收敛后再分别判别共同EOL折衷、平衡缺失相关类别、同点RCI修正；不盲加另一批步数，不拟合NIST。')
    result['next_execution_status'] = '下一为已声明末态global208与起点80的最小saved-gpg诊断；带新ci.py源码epoch的径向复用完整资格003独立准备，均等exact资源准入。'
    result['card_summary'] = ('Atomx原生MCDHF→RCI：29轨道/1046目标扇区累计208步；本轮128更新、8233控制和全部128 stored-H接受门通过，187H/187CI/129g，前台858.41s（非性能认证）。原metric残差9.55e−4、指标1.46e−6Ha，两驻值门仍FAIL；四线RMS2105.10→2054.32cm⁻¹，实验门仍FAIL。下一先分解保存梯度定位慢收敛，不盲加步数。保存H的完整1046态常数平移诊断1066门通过，原径向路由FAIL保留，新源码完整资格未运行。网页v6仍是18模型72行，最新网页末态80及历史完整保留；208尚未发布到网页。卡片后开启有界gap驱动迭代。')
    mutable = set(contract['mutable_fields'])
    check('historical_fields_preserved', all(result[k] == value for k, value in previous.items() if k not in mutable))
    registry = load(REGISTRY)
    registry['updated_at'] = result['created_at']
    registry['physics_budget_result'] = str((TOP / 'results_stage022.json').relative_to(REPO))
    registry['tasks'][0].update(status=result['expanded29_optimizer_status'], detail=result['card_summary'])
    registry['tasks'][1].update(status='先诊断优化自由度与条件数，再分层补物理', detail=result['selection_reason'])
    registry['tasks'][5].update(status='保存H平移诊断PASS；新完整route资格待执行', detail='1066门保持原值，原route002向量FAIL不改；cold时钟不称加速。')
    registry['physics_next_setups'][1] = result['selection_reason']
    registry['physics_next_setups'][5] = result['next_execution_status']
    once(HERE / 'registry_before.json', load(REGISTRY))
    once(TOP / 'results_stage022.json', result)
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    proof = {'checks_count': len(checks), 'checks': checks, 'all_passed': True,
        'new_physical_calls': 0, 'cards_sent': 0, 'site_changes': 0,
        'result': bound(TOP / 'results_stage022.json'), 'registry': bound(REGISTRY),
        'driver': bound(Path(__file__)), 'contract': bound(HERE / 'contract.json')}
    once(HERE / 'integration_proof.json', proof)
    print(json.dumps({k: proof[k] for k in ('checks_count', 'all_passed', 'result', 'registry')}))


if __name__ == '__main__':
    main()
