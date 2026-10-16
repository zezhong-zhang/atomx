"""Stored delivery plus finite-job provenance; no API or physical calls."""
from pathlib import Path
from datetime import datetime, timezone
import difflib
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIVATE = Path('/home/zzhang/.local/state/atomx-lark-hourly-aisi-codex')
KEY = 'e9f57af9-a8c4-5092-97b7-ccba601cc6e6'
MESSAGE = 'om_x100b638e098e78b0c44937fb618dd0a'
CHAT = 'oc_fb2fc627b3e98dc9555de31580878758'
BOT = 'ou_c697e48a528ad9aa6157f62531561b59'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

card, receipt, status, back = (read(HERE / n) for n in ('card.json', 'receipt.json', 'send_status.json', 'readback.json'))
execution = read(HERE / 'readback_execution.json')
assert execution['exit_code'] == 0 and execution['actual_readback_calls'] == 1
assert back['ok'] and back['identity'] == 'bot' and back['data']['total'] == 1
message = back['data']['messages'][0]
assert message['message_id'] == receipt['data']['message_id'] == status['message_id'] == MESSAGE
assert message['chat_id'] == receipt['data']['chat_id'] == CHAT
assert message['sender']['open_bot_id'] == BOT and message['sender']['sender_type'] == 'app'
assert message['sender']['name'] == 'aisi codex' and message['msg_type'] == 'interactive'
assert not message['deleted'] and not message['updated']
assert status['idempotency_key'] == KEY and status['passed'] and status['iteration_dispatch_ok']
assert card['schema'] == '2.0'
elements = card['body']['elements']
assert len(elements) == 15 and all(e['tag'] == 'markdown' for e in elements)
expected = '<card title="' + card['header']['title']['content'] + '">\n' + '\n'.join(e['content'] for e in elements) + '\n</card>'
actual = message['content']
def visible(value):
    return re.sub(r'\s+', '', re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', value))
assert visible(expected) == visible(actual)
differences = [{'operation': op, 'payload': expected[a:b], 'readback': actual[c:d]} for op, a, b, c, d in difflib.SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes() if op != 'equal']
assert differences == [*({'operation': 'delete', 'payload': '\n', 'readback': ''} for _ in range(6)), {'operation': 'insert', 'payload': '', 'readback': '['}, {'operation': 'insert', 'payload': '', 'readback': '](https://doi.org/10.1103/PhysRevA.110.L030802)'}]
assert sha(HERE / 'registry_stage019.json') == status['registry_sha256']
stage = HERE.parent / 'results_stage019.json'
assert any(e['path'] == str(stage.relative_to(REPO)) and e['sha256'] == sha(stage) for e in status['evidence'])
request_path = PRIVATE / 'iteration_requests' / (KEY + '.json')
request = read(request_path)
assert request['status'] == 'completed' and request['message_id'] == MESSAGE and request['key'] == KEY
assert len(request['jobs']) == 4 and not request['highest_level_reached']
preview = PRIVATE / 'iteration_previews' / (KEY + '.json')
decision = PRIVATE / 'iteration_decisions' / (KEY + '.json')
assert sha(preview) == request['preview_sha256'] == status['iteration_preview_sha256']
assert {j['job']['lane'] for j in request['jobs']} == {'fscc_scope', 'properties', 'sd_diagrams', 'gpu'}
scopes = {
    'fscc_scope': 'Finite12 spinors/4 core electrons,495 core/792 attachment determinants,8 model columns; unretained full attachment Bloch residual~0.00228Ha is not full-equation convergence.',
    'properties': 'Finite8x8,11 signed property matrices, fixed finite orbitals/long-wavelength scope.',
    'sd_diagrams': 'Finite10 doubles/495 four-electron determinants; signed DC/static/dynamic/phase anchors; full native scalable SD transverse remains incomplete.',
    'gpu': 'Actual ordered FP64 CPU/CUDA finite PP/HH and2688x25 density/rank0..3 kernel controls. Not MCDHF GPU or end-to-end speedup certification.'}
selected = {'nspinors','noccupied','core_determinants','attachment_determinants','model_dimension','core_unretained_rank4_max_Ha','attachment_unretained_rank34_max_Ha','full_attachment_bloch_max_Ha','full_signed_property_max_au','actual_wave_full_H_residual_Ha','native_scalable_sd_transverse_complete','realized_owned_device_bytes','highest_level_reached','source_unchanged'}
jobs = []
for job in request['jobs']:
    lane, result = job['job']['lane'], job['result']
    scientific = result['scientific_result']
    ledger = PRIVATE / 'iteration_ledger' / (job['job_id'] + '.json')
    inputs = PRIVATE / 'iteration_inputs' / (job['job_id'] + '.json')
    whole = read(ledger)
    assert whole['status'] == 'completed' and whole['job_id'] == job['job_id'] and whole['message_id'] == MESSAGE and whole['request_key'] == KEY
    assert whole['result'] == result
    assert job['status'] == 'completed' and result['execution_gate_pass'] and result['sources_and_inputs_unchanged']
    assert not result['highest_level_reached'] and not scientific['highest_level_reached']
    output = REPO / job['output']
    wrapper, probe, contract = output / 'result.json', output / 'probe/result.json', output / 'contract.json'
    assert sha(wrapper) == job['result_sha256'] and sha(probe) == result['probe_result_sha256']
    declared = (json.dumps(job['job']['sources'], sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    entry = REPO / job['job']['entry_point']
    jobs.append({'job_id':job['job_id'], 'lane':lane, 'status':'completed', 'started_at':job['process_started_at'], 'finished_at':job['process_finished_at'], 'observed_probe_subprocess_wall_seconds':result['observed_wall_seconds'], 'timing_scope':result['timing_scope'], 'execution_gate_pass':True, 'sources_and_inputs_unchanged':True, 'highest_level_reached':False, 'finite_scope':scopes[lane], 'selected_scientific_scalars':{k:v for k,v in scientific.items() if k in selected and isinstance(v,(bool,int,float))}, 'whole_immutable_job_ledger':{'path':str(ledger),'sha256':sha(ledger),'bytes':ledger.stat().st_size}, 'whole_input_ledger':{'path':str(inputs),'sha256':sha(inputs)}, 'controller_result':{'path':str(wrapper.relative_to(REPO)),'sha256':sha(wrapper)}, 'probe_result':{'path':str(probe.relative_to(REPO)),'sha256':sha(probe)}, 'probe_contract':{'path':str(contract.relative_to(REPO)),'sha256':sha(contract)}, 'entry_point_current_file':{'path':str(entry.relative_to(REPO)),'sha256':sha(entry)}, 'declared_source_map':{'entries':len(job['job']['sources']),'canonical_sorted_JSON_sha256':hashlib.sha256(declared).hexdigest(),'semantics':'Recorded source dictionary only; no new all-source numerical replay.'}, 'copied_whole_job_payload':False, 'independent_scientific_replay_executed':False})
summary = {'status':'actual_four_service_jobs_completed','request':{'path':str(request_path),'sha256':sha(request_path),'bytes':request_path.stat().st_size},'preview':{'path':str(preview),'sha256':sha(preview)},'decision':{'path':str(decision),'sha256':sha(decision)},'requested_at':request['requested_at'],'started_at':request['started_at'],'finished_at':request['finished_at'],'jobs':jobs,'highest_level_reached':False,'main_native_MCDHF_optimization_separate':True,'new_science_calls_by_delivery_audit':0,'no_MCDHF_GPU_speedup_claim':True}
write(HERE / 'service_jobs_summary.json', summary)
proof = {'proof_created_at':datetime.now(timezone.utc).isoformat(),'passed':True,'successful_send_and_readback':True,'cycle':status['cycle'],'idempotency_key':KEY,'message_id':MESSAGE,'actual_message_link':message['message_app_link'],'message_position':message['message_position'],'chat_id':CHAT,'bot_open_id':BOT,'source_stage019':{'path':str(stage.relative_to(REPO)),'sha256':sha(stage)},'registry_sha256':status['registry_sha256'],'registry_copy_provenance':'reporting_stage020/before_registry.json immutable pre-update registry','actual_card':{'payload_schema':'2.0','elements':len(elements),'title':card['header']['title']['content'],'receipt_create_time_Asia_Hong_Kong':receipt['data']['create_time'],'readback_create_time_Asia_Hong_Kong':message['create_time'],'all_visible_content_matches':True,'readback_content_characters':len(actual),'raw_readback_schema_available':False,'readback_representation':'CLI-rendered card text; actual sent rawv2 DSL archived separately','rendering_differences':differences},'service_send':{'started_at':status['started_at'],'finished_at':status['finished_at'],'passed':True,'iteration_dispatch_ok':True},'service_iteration':{'status':'completed','actual_service_jobs':4,'highest_level_reached':False,'summary':{'path':str((HERE/'service_jobs_summary.json').relative_to(REPO)),'sha256':sha(HERE/'service_jobs_summary.json')},'service_automatically_started_finite_science':True,'does_not_certify_production_MCDHF_or_GPU_speedup':True},'actual_readback_calls':1,'new_scientific_calls_by_delivery_audit':0,'cards_sent_by_delivery_audit':0,'copied_configurations_or_secrets':False,'files_sha256':{p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name not in ['delivery_proof.json','README.md']},'scope':'Actual stage019 delivery/readback plus scalar/provenance verification of four completed service jobs. Stage020 and native optimizer are separate successor reporting; production gates and old FAIL remain.'}
write(HERE / 'delivery_proof.json', proof)
print(json.dumps({'passed':True,'proof_sha256':sha(HERE/'delivery_proof.json'),'jobs_summary_sha256':sha(HERE/'service_jobs_summary.json'),'job_wall_seconds':{j['lane']:j['observed_probe_subprocess_wall_seconds'] for j in jobs},'actual_readback_calls':1,'new_science_calls':0,'link':message['message_app_link']}))
