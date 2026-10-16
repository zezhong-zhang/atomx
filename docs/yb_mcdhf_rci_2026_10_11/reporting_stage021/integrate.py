"""Saved-only current native MCDHF report; preserve historical physical scopes."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
REPO = HERE.parents[2]
REGISTRY = REPO / 'docs/atomx_hourly_progress_2026_10_10/progress.json'
OPT = TOP / 'correlated29_optimization_run004_preparation'
PERF = TOP / 'correlated29_radial_reuse_qualification002_preparation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def bound(path):
    return {'path': str(path.relative_to(REPO)), 'sha256': sha(path)}


def once(path, payload):
    with path.open('x') as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    contract = load(HERE / 'contract.json')
    assert sha(Path(__file__)) == contract['driver_sha256']
    assert sha(REGISTRY) == contract['before_registry_sha256']
    assert sha(TOP / 'results_stage020.json') == contract['previous_stage_sha256']
    checks = []

    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError('Reporting gate failed: ' + name)

    for name, digest in contract['evidence_sha256'].items():
        check('evidence/' + name, sha(REPO / name) == digest)
    previous = load(TOP / 'results_stage020.json')
    actual, closed, completion = (load(OPT / n) for n in
                                  ('actual/result.json', 'root_execution.json', 'completion.json'))
    check('actual_closed_same_source', closed['status'] == 'foreground_closed'
          and closed['exit_code'] == 0 and actual['error'] is None
          and actual['source_map_sha256'] == previous['correlated29_optimizer_run003']['source_map_sha256'])
    check('all_recorded_controls', len(actual['controls']) == 4201
          and all(row['passed'] for row in actual['controls'])
          and completion['all_completion_metadata_checks_passed']
          and completion['all_qualified_finite_stored_H_acceptances_qualified'])
    check('same_original_metric', actual['optimizer_final_diagnostics']['residual_metric_sha256']
          == previous['correlated29_optimizer_run003']['optimizer_final_diagnostics']['residual_metric_sha256'])
    check('actual_producer_scope', actual['attempted_calls'] == actual['completed_calls']
          and all(actual['completed_calls'][k] == v for k, v in
                  {'H': 98, 'CI': 98, 'gradient': 65, 'optimizer': 1, 'validation_Dirac': 7,
                   'initial_hS': 7, 'CI_internal_hS': 7}.items()))
    check('original_gates_fail_no_highest_claim', not actual['stationarity_passed']
          and not actual['experimental_certification'] and not actual['physical_completeness_certified'])
    check('actual64_global80', len(actual['trajectory']) == 65
          and actual['total_accepted_steps'] == 80 and completion['actual_new_accepted_steps'] == 64)
    points = []
    for row in actual['trajectory']:
        path = REPO / row['manifest_path']
        check('point_identity_' + str(row['iteration']), sha(path) == row['manifest_sha256'])
        point = load(path)
        check('point_gates_' + str(row['iteration']), point['checkpoint_qualified']
              and (row['iteration'] == 0 or point['exact_acceptance']['qualified']))
        points.append({**row, 'targets': point['targets'], 'multiplets': point['multiplets']})
    terminal = points[-1]
    check('terminal_not_selected_by_gap', actual['terminal_point_selected_without_gap_optimization']
          and completion['terminal_checkpoint'] == terminal['manifest_path'])
    pf, pe = load(PERF / 'completion.json'), load(PERF / 'root_execution.json')
    failure = load(PERF / 'actual/comparison_failed.json')
    bad = [row for row in failure['checks'] if not row['passed']]
    check('route_failure_unchanged', pe['exit_code'] == 1 and pf['status'] == 'closed_first_failure_retained'
          and bad == [{'name': 'signed_cluster_[296]', 'error': 1.0443566289541906e-08,
                       'limit': 1e-08, 'passed': False}]
          and not pf['shared_source_adoption_authorized'])
    delivery = load(TOP / 'delivery_stage020/delivery_proof.json')
    check('actual_card_native_iteration', delivery['passed']
          and delivery['same_card_root_native_iteration']['new_accepted_steps'] == 64)
    publication = load(TOP / 'atlas_update005/root_publication.json')
    check('published_current_and_history', publication['deployment_status'] == 'succeeded'
          and publication['models'] == 18 and publication['rows'] == 72
          and publication['audience_unchanged'] and publication['science_calls'] == 0)
    result = copy.deepcopy(previous)
    result['historical_stage020_primary_view'] = {k: copy.deepcopy(previous[k]) for k in
        ('computed_run', 'stationarity', 'numerical_certification_scope', 'correlation_scope')}
    result['historical_stage020_card_summary'] = previous['card_summary']
    result['historical_website_v5'] = previous['website_publication']
    result['historical_latest_hourly_delivery'] = previous['latest_hourly_delivery']
    result['created_at'] = datetime.now(timezone.utc).isoformat()
    result['inputs']['previous_stage020'] = bound(TOP / 'results_stage020.json')
    for name, digest in contract['evidence_sha256'].items():
        result['inputs'][name.removeprefix(str(TOP.relative_to(REPO)) + '/')] = {'path': name, 'sha256': digest}
    diag = actual['optimizer_final_diagnostics']
    result['computed_run'] = 'native corrected f043 correlated29/1046 DC, global80 accepted updates'
    result['stationarity'] = {k: diag[k] for k in ('converged', 'orbital_residual', 'orbital_tolerance',
        'estimated_energy_error', 'energy_tolerance', 'residual_metric_sha256', 'termination_reason')}
    result['stationarity'].update(iterations=80, same_metric_local_updates=64,
        numerical_scope='Original3451 metric, fixed nuclear positive span/P/inverse and five UID weights0.2; old531 metric archived separately.')
    result['correlation_scope'] = ('Current jointly optimized native model:29 radial orbitals, complete1046 five-target sectors of2163 NR parent CSFs,69 electrons; original4f/5p/5s core-hole families and screened virtual pool. Not full core-double/high-L/radial/all-order correlation completeness. Same no-pair DC/FermiA175; no Breit/QED/recoil inserted into this new point.')
    result['numerical_certification_scope'] = ('Current f043 native MCDHF real80 updates; run00465 checkpoints/64 updates4201 retained-state controls and exact saved-H acceptances pass, same original3451 metric. Residual/indicator and experiment gates fail; no uniform derivative/model-completeness or end-to-end speed certificate. Historical353/356/old531/conditional controls and failures remain independently recorded.')
    result['correlated29_optimizer_run004'] = {'status': 'closed_64_updates_global80_not_stationary',
        'result': bound(OPT / 'actual/result.json'), 'execution': bound(OPT / 'root_execution.json'),
        'completion': bound(OPT / 'completion.json'), 'source_map_sha256': actual['source_map_sha256'],
        'controls_passed': 4201, 'actual_calls': actual['completed_calls'], 'points': points,
        'optimizer_final_diagnostics': diag, 'exact_stored_H_guard_budget': actual['exact_guard_budget'],
        'elapsed_uncontrolled_seconds': closed['elapsed_uncontrolled_seconds'],
        'peak_RSS_MiB': actual['peak_RSS_MiB'], 'stationarity_passed': False,
        'experimental_certification': False, 'physical_completeness_certified': False,
        'energy_callback_used': False, 'terminal_selected_without_gap_optimization': True}
    result['radial_reuse_successor002_failure'] = {'completion': bound(PERF / 'completion.json'),
        'execution': bound(PERF / 'root_execution.json'), 'comparison_failure': bound(PERF / 'actual/comparison_failed.json'),
        'first_failed_gate': bad[0], 'actual_gate_summary': pf['actual_gate_summary'],
        'actual_calls': pf['actual_calls'], 'stages_uncontrolled_seconds': pf['actual_stages_uncontrolled_seconds'],
        'shared_source_applied': False, 'speedup_certified': False,
        'scope': 'Analytic/whole-H/energy/fullg gates pass; whole CI cluster parity fails, later gates unexecuted. Source-only diagnostic may follow; cannot adopt or widen1e-8 tolerance.'}
    result['latest_hourly_delivery'] = bound(TOP / 'delivery_hour23/delivery_proof.json')
    result['latest_manual_delivery'] = bound(TOP / 'delivery_stage020/delivery_proof.json')
    result['website_publication'] = publication
    result['expanded29_optimizer_status'] = '原生29/1046累计80步；全部64续算接受门通过，两驻值门仍失败。'
    result['selection_reason'] = ('原3451metric残差0.040910→0.00264137、指标1.17488e−5Ha，四线RMS2203.862→2105.099cm−1，仍高于最初2048.365。继续同空间从末态求驻值，先区分未收敛与物理遗漏。H径向复用002因非目标cluster296投影1.044e−8>1e−8未合入；不靠放宽门或NIST拟合制造通过。')
    result['next_execution_status'] = '下一同源同空间128步续算独立准备；保持原目标/metric/physical gates；本报告零新数值。'
    result['card_summary'] = ('Atomx原生MCDHF→RCI：29轨道/1046完整目标扇区累计80步；本轮64续算4201控制与全部64 stored-H接受门通过，98H/98CI/65梯度，前台411.94s（含核验归档，非加速认证）。原metric残差0.040910→0.00264137、指标1.17488e−5Ha，两驻值门仍FAIL。四线RMS2203.86→2105.10cm⁻¹；末态D3/D5/P1/P3 gap +3043.56/+2553.75/−1012.51/−956.91，未达到实验门或最高水平。径向复用002的完整CI向量门首FAIL保留、未合入不放宽。远程周期表已发布18模型72行，两新末态及完整轨迹、原NIST/Ne/Eu历史保留。下一从原末态同模型续算，卡片成功后开启有界迭代。')
    mutable = set(contract['mutable_report_fields'])
    check('historical_scientific_fields_unchanged', all(result[k] == value for k, value in previous.items() if k not in mutable))
    check('original_failed_stationarity_preserved', result['historical_stage020_primary_view']['stationarity'] == previous['stationarity'])
    registry = load(REGISTRY)
    registry['updated_at'] = result['created_at']
    registry['physics_budget_result'] = str((TOP / 'results_stage021.json').relative_to(REPO))
    registry['tasks'][0].update(status='原生29/1046累计80步；驻值及实验门仍FAIL', detail=result['card_summary'])
    registry['tasks'][1].update(status='先同空间求驻值；物理遗漏仍分层测量', detail=result['selection_reason'])
    registry['tasks'][4].update(status='远程周期表新版18模型72行已发布', detail='保留16历史模型与全部6019 NIST gzip；追加原生16/80步末态及完整轨迹。')
    registry['tasks'][5].update(status='径向复用002完整CI向量门FAIL；未合入', detail=result['selection_reason'])
    registry['physics_next_setups'][1] = result['selection_reason']
    registry['physics_next_setups'][5] = result['next_execution_status']
    with (HERE / 'before_registry.json').open('xb') as stream:
        stream.write(REGISTRY.read_bytes())
    once(TOP / 'results_stage021.json', result)
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    proof = {'result': bound(TOP / 'results_stage021.json'), 'registry': bound(REGISTRY),
        'driver': bound(Path(__file__)), 'contract': bound(HERE / 'contract.json'),
        'checks_count': len(checks), 'checks': checks, 'all_passed': True,
        'new_science_calls': 0, 'cards_sent': 0, 'websites_changed': 0}
    once(HERE / 'integration_proof.json', proof)
    print(json.dumps({k: proof[k] for k in ('result', 'registry', 'checks_count', 'all_passed')}))


if __name__ == '__main__':
    main()
