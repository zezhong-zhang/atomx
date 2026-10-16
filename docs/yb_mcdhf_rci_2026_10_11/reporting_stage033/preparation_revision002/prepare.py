"""Only stdlib source/closed-JSON preparation; never import scientific/registry drivers."""
import ast
import copy
import hashlib
import json
from pathlib import Path

B = Path(__file__).absolute().parent
R = next(p for p in B.parents if (p / 'pyproject.toml').is_file())
T = B.parent

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def bound(p):
    return dict(path=str(p.relative_to(R)), sha256=sha(p))

def once(p, d):
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')

checks = []
def check(name, value):
    checks.append(dict(name=name, passed=bool(value)))
    assert value, name

prefix = 'docs/yb_mcdhf_rci_2026_10_11/'
fd = prefix + 'screened_f_next_step208/constrained_fd_proposal001/representation_repair001/'
owner = prefix + 'rci_retained_transverse_implementation001/'
gpu = prefix + 'gpu_mcdhf_next_step208/'
roles = {
    'base_result': prefix + 'results_stage032.json',
    'base_panel': prefix + 'reporting_stage032/current_card_evidence.json',
    'base_proof': prefix + 'reporting_stage032/integration_proof.json',
    'base_driver': prefix + 'reporting_stage032/refresh.py',
    'utility': prefix + 'reporting_stage030/refresh.py',
    'renderer': 'docs/atomx_hourly_progress_2026_10_10/sync_card.py',
    'planner': 'docs/atomx_hourly_progress_2026_10_10/parallel_planner.py',
    'FD_actual': fd + 'actual/result.json', 'FD_domains': fd + 'actual/derivative_domains.json',
    'FD_receipt': fd + 'root_execution.json', 'FD_closure': fd + 'completion.json',
    'primary_receipt': owner + 'primary001/root_execution.json', 'primary_log': owner + 'primary001/foreground.log',
    'integration_receipt': owner + 'integration001/root_execution.json', 'integration_log': owner + 'integration001/foreground.log',
    'mixed_receipt': owner + 'mixed001/root_execution.json', 'mixed_log': owner + 'mixed001/foreground.log',
    'mixed_calls': owner + 'mixed001/numeric_assets/candidate_calls.json',
    'mixed_finite_records': owner + 'mixed001/numeric_assets/finite_diagnostics.json',
    'original_GPU_actual': gpu + 'qualification_preparation001/actual/result.json',
    'original_GPU_receipt': gpu + 'qualification_preparation001/controller/execution_receipt.json',
    'original_GPU_closure': gpu + 'qualification_preparation001/closure001/completion.json',
    'GPU_successor_actual': gpu + 'qualification_representation_repair001/actual/result.json',
    'GPU_successor_receipt': gpu + 'qualification_representation_repair001/controller/execution_receipt.json',
    'GPU_successor_closure': gpu + 'qualification_representation_repair001/closure001/completion.json',
    'GPU_successor_preparation': gpu + 'qualification_representation_repair001/handoff.json',
    'current_shared_preparation_identity': gpu + 'qualification_representation_repair001/shared_source_at_preparation.json',
    'hour28_service': prefix + 'delivery_hour28/service_iteration_status.json',
    'hour28_original_capture': prefix + 'delivery_hour28/delivery_proof.json',
    'hour28_authorization': prefix + 'delivery_hour28/delivery_authorization_proof.json',
}
check('all roles nonnumerical files', all(Path(p).suffix in {'.json', '.py', '.log'} for p in roles.values()))
for name, path in roles.items():
    check('exact role exists/' + name, (R / path).is_file())
inputs = {p: sha(R / p) for p in roles.values()}
old, panel, proof = (read(R / roles[n]) for n in ('base_result', 'base_panel', 'base_proof'))
check('stage032 immutable result SHA', inputs[roles['base_result']] == 'c9aaacd44598f186a255cf1cc066f0cdf52d731f983e6a95d766185dcf02a896')
check('stage032 immutable panel SHA', inputs[roles['base_panel']] == '48c931fb0f114933273296ed3c600fcee2c1e12b982349ea5c3488e77ce531e8')
check('stage032 proof result and panel identities', proof['all_passed'] and proof['result'] == bound(R / roles['base_result']) and proof['panel'] == bound(R / roles['base_panel']))
registry = R / 'docs/atomx_hourly_progress_2026_10_10/progress.json'
check('stage032 actual registry baseline', sha(registry) == proof['registry']['sha256'])
for path, row in panel['inputs'].items():
    check('original renderer binding/' + path, isinstance(row, dict) and set(row) >= {'path', 'sha256'} and Path(row['path']).suffix != '.npz' and sha(R / row['path']) == row['sha256'])
note = old['card_summary'][old['card_summary'].rfind('网页v9'):]
check('same stage032 pending metadata note in panel', note.startswith('网页v9') and panel['card_summary'].endswith(note) and '仍待资格' in note)
fdr, fdc, fda = (read(R / roles[n]) for n in ('FD_receipt', 'FD_closure', 'FD_actual'))
check('FD closed receipt exact', fdr['status'] == 'foreground_closed' and fdr['exit_code'] == 0 and inputs[roles['FD_actual']] == 'b1e1f6ef10f053025873aef0fa597d8a3792b7fb5f55350ef4b65b024e40fa31')
check('FD730 actual and81 metadata closure', len(fda['checks']) == 730 and all(x['passed'] for x in fda['checks']) and fdc['metadata_checks_count'] == 81 and all(x['passed'] for x in fdc['metadata_checks']))
check('FD single direction not stationarity/fullspace', fda['single_direction_finite_measurement_only'] and not fda['full31_space_certificate'] and not fda['original_stationarity_passed'])
for name, code in [('primary', 0), ('integration', 0), ('mixed', 1)]:
    receipt = read(R / roles[name + '_receipt'])
    check('owner exact closed ' + name, receipt['status'] == 'foreground_closed' and receipt['exit_code'] == code and receipt['log_sha256'] == inputs[roles[name + '_log']])
gpur = read(R / roles['GPU_successor_receipt'])
check('closed successor before result read', gpur['status'] == 'foreground_closed' and gpur['exit_code'] == 0)
gpua, gpuc = read(R / roles['GPU_successor_actual']), read(R / roles['GPU_successor_closure'])
check('GPU actual16760 and metadata37', len(gpua['checks']) == 16760 and all(x['passed'] for x in gpua['checks']) and gpuc['checks_count'] == 37 and gpuc['all_metadata_checks_passed'])
check('GPU no speed optimization shared adoption', not gpua['full_H_CI_on_GPU'] and not gpua['controlled_speedup_qualified'] and not gpua['expanded_MCDHF_optimized'] and not gpuc['public_adoption'])
original = read(R / roles['original_GPU_actual'])
check('originalGPU196PASS1FAIL remains', original['failure'] == 'saved_actual_owner_header' and len(original['checks']) == 197 and sum(x['passed'] for x in original['checks']) == 196)
hour = read(R / roles['hour28_service'])
check('hour28 closed four actual registered probes', hour['status'] == 'completed' and len(hour['jobs']) == len(hour['job_ids']) == 4 and all(x['status'] == 'completed' for x in hour['jobs']))
check('hour28 Yb MCDHF jobs zero only', all('yb' not in x.lower() and 'mcdhf' not in x.lower() for x in hour['job_ids']))
check('original hour28 capture format failure retained', not read(R / roles['hour28_original_capture'])['passed'])
s = (B / 'refresh.py').read_text()
tree = ast.parse(s)
check('source syntax', isinstance(tree, ast.Module))
check('no scientific imports', not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(a.name.split('.')[0] in {'numpy', 'numba', 'scipy', 'atomx'} for a in n.names) for n in ast.walk(tree)))
check('root explicit execution guard', "if not options.execute:" in s and "admission['execution_authorized_by_root'] is True" in s)
check('no fake GPU pending result or early result read', "receipt.get('status') != 'foreground_closed'" in s and 'closed_GPU_successor' in s)
check('immutable all old scientific result and panel comparison', 'result[k] == v for k, v in old.items()' in s and 'panel[k] == v for k, v in old_panel.items()' in s)
check('no send API timers', not any(x in s for x in ('lark-cli', 'send-card', 'requests.', 'urllib.', 'cron', 'np.load', 'import numpy')))
check('result input objects only no nested successor map', "result['inputs']['stage033_GPU_successor']" not in s and "{'path', 'sha256'} <= v.keys() for v in result['inputs'].values()" in s)
check('old zero-auto reason preserved as history only', "result['historical_stage032_selection_reason']" in s and "result['selection_reason']" in s and '四项自动诊断已完成' in s)
probe = copy.deepcopy(panel)
probe['card_summary'] = probe['card_summary'][:-len(note)] + 'bounded separate domains'
check('metadata replacement cannot alter scientific rows', all(probe[k] == v for k, v in panel.items() if k != 'card_summary'))
contract = dict(status='SOURCE_JSON_STAGE033_FROZEN_PENDING_ROOT_METADATA_SEAL_NO_INTEGRATION', driver_sha256=sha(B/'refresh.py'), registry_before_sha256=sha(registry), roles=roles, inputs=inputs, stage032_metadata_note=note, closed_GPU_successor={k:bound(R/roles[v]) for k,v in [('result','GPU_successor_actual'),('receipt','GPU_successor_receipt'),('closure','GPU_successor_closure')]}, private_f043_source_map_sha256=fda['source_map_sha256'], mixed_expected_completed_calls=read(R/roles['mixed_calls'])['completed'], mixed_failures=[dict(name='main_H_bitwise_against_old_source002', status='FAIL_source_epoch_difference_under_audit_no_waiver'),dict(name='coarse_h1e-4_centered_direction', measured=-.00872063585143113,analytic=-.00872046743791834,error_Ha_per_unit=1.6841351279e-7, original_gate=8.72046743791834e-8,status='FAIL_no_gate_change')], fixed=dict(base_stage032_scientific_fields_rows_history_preserved=True, old3451_and_new3689_FAIL_kept=True, one_direction31FD_not_fullspace=True, private_f043_GPU_not_current_shared_adoption=True, primary63_integration26_not_mixedPASS=True, mixed42PASS2FAIL_retained=True, originalGPU196PASS1FAIL_retained=True, controlled_speedup=False, hourly_timer_unchanged=True), allowed_registry_changes=['result/panel/updated_at pointers','task0/status/detail','task5/status/detail'], exclusive_outputs=['registry_before.json','current_card_evidence.json','preview_card.json','preview_citations.json','integration_proof.json'], resources=dict(CPU=30,threads=1,RSS_bytes=1024**3,inner_seconds=60,outer_seconds=65,new_science_NPZ_GPU_API_card_timer_calls=0), stop='Explicit root exact SHA admission only. Changed source inputs, registry, closed receipt or existing outputs => fail without retry. No scientific evaluation, arrays, API, timer or publication.', preparation_execution_not_metadata_integration=True)
once(B/'contract.json',contract)
once(B/'preparation.json',dict(status='SOURCE_JSON_ONLY_PREPARED_NO_REGISTRY_WRITE',all_passed=True,checks_count=len(checks),checks=checks,driver=bound(B/'refresh.py'),contract=bound(B/'contract.json'),preparer=bound(Path(__file__)),science_NumPy_Atomx_NumBa_CUDA_NPZ_API_card_calls=0,registry_sha256=sha(registry)))
print(json.dumps(dict(preparation_checks=len(checks),driver=sha(B/'refresh.py'),contract=sha(B/'contract.json'),preparation=sha(B/'preparation.json'))))
