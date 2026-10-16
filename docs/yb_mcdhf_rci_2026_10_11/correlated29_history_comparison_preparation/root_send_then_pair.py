"""One review-bound actual manual card, then exactly one native paired iteration."""
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

B = Path(__file__).resolve().parent
R = B.parents[2]
D = B.parent / 'delivery_stage023'
S = Path('/home/zzhang/.local/state/atomx-lark-hourly-aisi-codex')
REG = R / 'docs/atomx_hourly_progress_2026_10_10/progress.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, o):
    Path(p).write_text(json.dumps(o, ensure_ascii=False, indent=2) + '\n')


def main():
    expected_registry = 'f0feab13bb351a47715bfa2a16fcd7a1e54d126dabfbc7602fc2b46ba1ad2ddb'
    assert sha(REG) == expected_registry
    assert not D.exists() and not (B / 'pair_once_marker.json').exists()
    for n in ('history10', 'history40'):
        assert (B/n/'execution_contract.json').exists() and not (B/n/'once_marker.json').exists()
    D.mkdir()
    (D/'registry_stage023.json').write_bytes(REG.read_bytes())
    start = time.monotonic()
    command = ['timeout', '--signal=TERM', '--kill-after=5s', '60s', '/usr/bin/python3',
               str(R/'docs/atomx_hourly_progress_2026_10_10/sync_card.py'),
               '--config', str(S/'config.json'), '--cycle', 'mcdhf-stage023']
    with (D/'root_send_stdout.json').open('wb') as out, (D/'root_send_stderr.log').open('wb') as err:
        result = subprocess.run(command, cwd=R, stdout=out, stderr=err, check=False)
    write(D/'root_send_execution.json', {'exit_code': result.returncode, 'elapsed_s': time.monotonic()-start,
          'cycle': 'mcdhf-stage023', 'actual_send_cycle_launches': 1, 'driver_sha256': sha(__file__)})
    assert result.returncode == 0
    status = json.loads((S/'last_status.json').read_text())
    assert status['cycle'] == 'mcdhf-stage023' and status['passed'] and status['iteration_dispatch_ok']
    assert status['registry_sha256'] == expected_registry
    key = status['idempotency_key']
    files = {'latest_card.json':'card.json', 'last_status.json':'send_status.json',
             'receipt_'+key+'.json':'receipt.json', 'iteration_requests/'+key+'.json':'iteration_request.json',
             'iteration_status.json':'service_iteration_status.json'}
    for origin, target in files.items():
        (D/target).write_bytes((S/origin).read_bytes())
    write(D/'capture.json', {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'registry_sha256': expected_registry, 'message_id': status['message_id'], 'idempotency_key': key,
          'copied_files_sha256': {p.name:sha(p) for p in sorted(D.glob('*.json'))},
          'private_service_jobs_not_claimed_as_native': True})
    print(json.dumps({'card_passed':True, 'message_id':status['message_id'], 'key':key,
                      'paired_native_launch_follows':True}), flush=True)
    pair = subprocess.run(['/usr/bin/python3', str(B/'root_launch_pair.py')], cwd=R, check=False)
    return pair.returncode


if __name__ == '__main__':
    raise SystemExit(main())
