"""Capture one existing hourly delivery and make one bridge-bound readback.

No send, scheduler creation, or science. Preserve the exact source panel and
actual closed service request separately from root-launched native iteration.
"""
import hashlib
import importlib.util
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parent
R = B.parents[2]
S = Path('/home/zzhang/.local/state/atomx-lark-hourly-aisi-codex')
REG = R / 'docs/atomx_hourly_progress_2026_10_10/progress.json'
REG_SHA = '4d1859ced2ff74646fd68a7b91b713f5bf3a9554298d202842f6d28d4d5144ec'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    status = read(S / 'last_status.json')
    assert status['cycle'] == '2026-10-11T17'
    assert status['passed'] and status['iteration_dispatch_ok']
    assert status['registry_sha256'] == sha(REG) == REG_SHA
    key = status['idempotency_key']
    request = read(S / 'iteration_requests' / (key + '.json'))
    service = read(S / 'iteration_status.json')
    assert request['key'] == service['key'] == key
    assert request['status'] == service['status'] == 'waiting_for_evidence_repair'
    assert 'physics priority uses a different current MCDHF result' in service['reason']
    assert not request.get('jobs') and not service.get('jobs')
    assert request['message_id'] == service['message_id'] == status['message_id']
    assert not (B / 'once_marker.json').exists()
    write(B / 'once_marker.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
                                 'cycle': status['cycle'], 'message_id': status['message_id']})
    files = {'latest_card.json': 'card.json', 'last_status.json': 'send_status.json',
             'receipt_' + key + '.json': 'receipt.json',
             'iteration_requests/' + key + '.json': 'iteration_request.json',
             'iteration_status.json': 'service_iteration_status.json'}
    for original, saved in files.items():
        (B / saved).write_bytes((S / original).read_bytes())
    (B / 'registry_stage024.json').write_bytes(REG.read_bytes())
    command = ['lark-cli', 'im', '+messages-mget', '--message-ids', status['message_id'],
               '--no-reactions', '--json']
    start = time.monotonic()
    with (B / 'readback.json').open('xb') as out, (B / 'readback_stderr.log').open('xb') as err:
        result = subprocess.run(command, stdout=out, stderr=err, timeout=60)
    write(B / 'readback_execution.json', {'actual_readback_calls': 1,
          'exit_code': result.returncode, 'elapsed_s': time.monotonic() - start,
          'command': command, 'readback_sha256': sha(B / 'readback.json')})
    assert result.returncode == 0
    back, card, receipt = (read(B / name) for name in ('readback.json', 'card.json', 'receipt.json'))
    utility_path = B.parent / 'delivery_stage021/verify_saved_delivery.py'
    assert sha(utility_path) == 'f8bf88f8caae2285620273f449c6f3b3d705364cc8115b2699284dd7c0bba093'
    spec = importlib.util.spec_from_file_location('frozen_card_text_utils', utility_path)
    utility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utility)
    assert back['ok'] and back['identity'] == 'bot' and back['data']['total'] == 1
    message = back['data']['messages'][0]
    assert message['message_id'] == receipt['data']['message_id'] == status['message_id']
    assert message['chat_id'] == receipt['data']['chat_id'] == utility.CHAT
    assert message['sender']['open_bot_id'] == utility.BOT and message['sender']['sender_type'] == 'app'
    assert message['msg_type'] == 'interactive' and not message['deleted'] and not message['updated']
    assert card['schema'] == '2.0' and len(card['body']['elements']) == 15
    expected = '<card title="' + card['header']['title']['content'] + '">\n' + '\n'.join(
        element['content'] for element in card['body']['elements']) + '\n</card>'
    assert utility.visible(expected) == utility.visible(message['content'])
    stage = B.parent / 'results_stage024.json'
    assert sha(stage) == '7d160c91e76beb7dfc97c1222bf85d519edea772c6547f630021f0ae214f4e34'
    assert any(row['path'] == str(stage.relative_to(R)) and row['sha256'] == sha(stage)
               for row in status['evidence'])
    jobs = request.get('jobs', [])
    assert all(job['status'] == 'completed' and job['request_key'] == key for job in jobs)
    assert len(jobs) == len(service.get('jobs', []))
    proof = {'passed': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
             'message_id': message['message_id'], 'actual_message_link': message['message_app_link'],
             'position': int(message['message_position']), 'cycle': status['cycle'],
             'source_stage': 24, 'source_stage_sha256': sha(stage), 'registry_sha256': REG_SHA,
             'actual_card': {'schema': '2.0', 'elements': 15, 'full_visible_content_matches': True},
             'actual_send': {'started_at': status['started_at'], 'finished_at': status['finished_at']},
             'service_iteration': {'status': 'waiting_for_evidence_repair', 'reason': service['reason'],
                                   'automatic_jobs': len(jobs), 'scientific_iteration_started': False,
                                   'job_ids': [job['job_id'] for job in jobs]},
             'actual_readback_calls_by_root': 1, 'new_send_calls': 0, 'new_scientific_calls': 0,
             'files_sha256': {p.name: sha(p) for p in sorted(B.glob('*.json'))},
             'verifier_sha256': sha(__file__),
             'scope': 'Existing hourly card only; root bounded native iteration is separately admitted and launched.'}
    write(B / 'delivery_proof.json', proof)
    print(json.dumps({'passed': True, 'message_id': message['message_id'], 'position': proof['position'],
                      'proof_sha256': sha(B / 'delivery_proof.json'), 'automatic_jobs': len(jobs)}))


if __name__ == '__main__':
    main()
