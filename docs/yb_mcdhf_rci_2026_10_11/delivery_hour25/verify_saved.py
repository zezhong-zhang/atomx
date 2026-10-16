"""Verify the single archived readback, retaining the original render gate failure.

Only observed table-format punctuation is normalized. Exact non-formatting
content, all row cells, actual send/source, and refusal identities stay bound.
No API call or science is performed.
"""
import difflib
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parent
R = B.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    utility_path = B.parent / 'delivery_stage021/verify_saved_delivery.py'
    assert sha(utility_path) == 'f8bf88f8caae2285620273f449c6f3b3d705364cc8115b2699284dd7c0bba093'
    spec = importlib.util.spec_from_file_location('frozen_text', utility_path)
    utility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utility)
    status, card, receipt, back, executed, request, service = [read(B / name) for name in (
        'send_status.json', 'card.json', 'receipt.json', 'readback.json',
        'readback_execution.json', 'iteration_request.json', 'service_iteration_status.json')]
    assert status['cycle'] == '2026-10-11T17' and status['passed'] and status['iteration_dispatch_ok']
    assert executed['actual_readback_calls'] == 1 and executed['exit_code'] == 0
    assert executed['readback_sha256'] == sha(B / 'readback.json')
    assert back['ok'] and back['identity'] == 'bot' and back['data']['total'] == 1
    message = back['data']['messages'][0]
    assert message['message_id'] == receipt['data']['message_id'] == status['message_id']
    assert message['chat_id'] == receipt['data']['chat_id'] == utility.CHAT
    assert message['sender']['open_bot_id'] == utility.BOT and message['sender']['sender_type'] == 'app'
    assert message['msg_type'] == 'interactive' and not message['deleted'] and not message['updated']
    assert card['schema'] == '2.0' and len(card['body']['elements']) == 15
    expected = '<card title="' + card['header']['title']['content'] + '">\n' + '\n'.join(
        element['content'] for element in card['body']['elements']) + '\n</card>'
    a, b = utility.visible(expected), utility.visible(message['content'])
    differences = [{'operation': op, 'payload': a[x:y], 'readback': b[v:w]} for op, x, y, v, w in
                   difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op != 'equal']
    observed = [('insert', '', '---'), *[('replace', ':', '---')] * 4,
                ('insert', '', '|'), ('insert', '', '||||||'), ('insert', '', '|||||')]
    assert [(r['operation'], r['payload'], r['readback']) for r in differences] == observed
    table_rows = [line.strip() for element in card['body']['elements']
                  for line in element['content'].splitlines()
                  if line.strip().startswith('|') and '---' not in line]
    assert len(table_rows) == 5
    assert all(utility.visible(row) in b for row in table_rows)
    assert sha(B / 'registry_stage024.json') == status['registry_sha256'] == (
        '4d1859ced2ff74646fd68a7b91b713f5bf3a9554298d202842f6d28d4d5144ec')
    stage = B.parent / 'results_stage024.json'
    assert sha(stage) == '7d160c91e76beb7dfc97c1222bf85d519edea772c6547f630021f0ae214f4e34'
    assert any(r['path'] == str(stage.relative_to(R)) and r['sha256'] == sha(stage) for r in status['evidence'])
    assert request['key'] == service['key'] == status['idempotency_key']
    assert request['message_id'] == service['message_id'] == message['message_id']
    assert request['status'] == service['status'] == 'waiting_for_evidence_repair'
    assert 'physics priority uses a different current MCDHF result' in service['reason']
    assert not request.get('jobs') and not service.get('jobs')
    proof = {'passed': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
             'message_id': message['message_id'], 'actual_message_link': message['message_app_link'],
             'position': int(message['message_position']), 'cycle': status['cycle'], 'source_stage': 24,
             'source_stage_sha256': sha(stage), 'registry_sha256': sha(B / 'registry_stage024.json'),
             'actual_card': {'schema': '2.0', 'elements': 15, 'raw_visible_string_equal': False,
                            'all_nonformatting_text_and_table_cells_match': True,
                            'observed_exact_formatting_differences': differences},
             'actual_send': {'started_at': status['started_at'], 'finished_at': status['finished_at']},
             'service_iteration': {'status': service['status'], 'reason': service['reason'],
                                   'automatic_jobs': 0, 'scientific_iteration_started': False},
             'actual_readback_calls_by_root': 1, 'new_API_calls_by_this_verifier': 0,
             'new_scientific_calls': 0, 'verifier_sha256': sha(__file__),
             'files_sha256': {p.name: sha(p) for p in sorted(B.glob('*.json'))},
             'scope': 'Existing hourly card. Original render gate refusal retained; exact table-format differences documented; root native iteration separately launched.'}
    target = B / 'delivery_proof.json'
    assert not target.exists()
    target.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': True, 'message_id': message['message_id'], 'position': proof['position'],
                      'proof_sha256': sha(target), 'service_status': service['status'], 'new_API_calls': 0}))


if __name__ == '__main__':
    main()
