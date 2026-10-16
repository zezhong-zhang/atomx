"""Metadata successor repairing current-result consistency; no science/API."""
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


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parent = TOP / 'results_stage024.json'
    registry = H / 'progress.json'
    assert sha(parent) == '7d160c91e76beb7dfc97c1222bf85d519edea772c6547f630021f0ae214f4e34'
    assert sha(registry) == '4d1859ced2ff74646fd68a7b91b713f5bf3a9554298d202842f6d28d4d5144ec'
    assert not (TOP / 'results_stage025.json').exists()
    (B / 'registry_before_repair.json').write_bytes(registry.read_bytes())
    result, record = read(parent), read(registry)
    panel = R / record['yb_mcdhf_result']
    assert read(panel)['method'] == 'MCDHF -> RCI'
    old = result['inputs']['current_mcdhf']
    assert old['path'] != record['yb_mcdhf_result']
    result['inputs']['historical_current_mcdhf_stage006'] = old
    result['inputs']['current_mcdhf'] = {'path': record['yb_mcdhf_result'], 'sha256': sha(panel)}
    delivery = TOP / 'delivery_hour25/delivery_proof.json'
    delivered = read(delivery)
    assert delivered['passed'] and delivered['service_iteration']['status'] == 'waiting_for_evidence_repair'
    assert delivered['service_iteration']['automatic_jobs'] == 0
    result['inputs']['hour25_actual_delivery'] = {'path': str(delivery.relative_to(R)), 'sha256': sha(delivery)}
    checks = []
    for role, binding in result['inputs'].items():
        passed = sha(R / binding['path']) == binding['sha256']
        checks.append({'role': role, 'passed': passed})
        assert passed, role
    result['created_at'] = datetime.now(timezone.utc).isoformat()
    result['reporting_pointer_repair'] = {
        'parent_stage_sha256': sha(parent), 'actual_failed_hourly_delivery': delivered['message_id'],
        'original_failure_retained': True, 'old_binding': old,
        'new_current_binding': result['inputs']['current_mcdhf'], 'new_scientific_calls': 0,
        'reason': 'Stage024 changed current card result but retained stage006 inputs.current_mcdhf; automatic planner correctly refused. Both pointers now bind the same preserved stage024 evidence.'}
    result['latest_hourly_delivery'] = {'path': str(delivery.relative_to(R)), 'sha256': sha(delivery),
        'source_stage': 24, 'service_status': 'waiting_for_evidence_repair',
        'actual_native_iteration_started_by_service': False}
    result['card_summary'] += ' 小时卡片的当前结果引用已同步修复；01:00那次自动迭代拒绝原样保留，未记作已运行。后续原生计算须独立准入。'
    write(TOP / 'results_stage025.json', result)
    record['physics_budget_result'] = str((TOP / 'results_stage025.json').relative_to(R))
    write(registry, record)
    sys.path.insert(0, str(H))
    specification = importlib.util.spec_from_file_location('current_parallel_planner', H / 'parallel_planner.py')
    planner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(planner)
    decision = {'status': 'implementation_frontier', 'jobs': [], 'frontier': [], 'evidence': []}
    planner.annotate_physics_priority(R, H, decision)
    assert decision['evidence'][-1]['sha256'] == sha(TOP / 'results_stage025.json')
    assert decision['direction'] == result['card_summary']
    write(B / 'integration_proof.json', {'all_passed': True, 'input_hash_checks': checks,
          'checks_count': len(checks) + 2, 'actual_public_planner_annotation_passed': True,
          'current_result_pointer_consistent': True, 'results_sha256': sha(TOP / 'results_stage025.json'),
          'registry_sha256': sha(registry), 'driver_sha256': sha(__file__),
          'new_API_calls': 0, 'new_scientific_calls': 0, 'historical_service_refusal_preserved': True})
    print(json.dumps({'all_passed': True, 'checks': len(checks) + 2,
                      'result_sha256': sha(TOP / 'results_stage025.json'), 'registry_sha256': sha(registry)}))


if __name__ == '__main__':
    main()
