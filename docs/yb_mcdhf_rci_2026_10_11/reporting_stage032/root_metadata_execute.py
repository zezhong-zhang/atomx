"""Root admission and receipt for the bounded, metadata-only stage032 refresh."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from datetime import datetime, timezone

B = Path(__file__).resolve().parent
R = B.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(name, value):
    with (B / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))


def main():
    contract_sha = '265263f2e7468ce5c0b5751ec9b47cf4980f487176b0939c78ce2c9ba3ccd908'
    assert sha(B / 'contract.json') == contract_sha
    assert sha(B / 'handoff.json') == '0588c986328adfce120712621977bb1e5a0d6baa8607aec1c61c78a92ef19bba'
    handoff = json.loads((B / 'handoff.json').read_text())
    contract = json.loads((B / 'contract.json').read_text())
    for name, expected in handoff['files_sha256'].items():
        assert sha(B / name) == expected, name
    for name, expected in contract['inputs'].items():
        assert sha(R / name) == expected, name
    registry = R / 'docs/atomx_hourly_progress_2026_10_10/progress.json'
    assert sha(registry) == contract['registry_before_sha256']
    assert not any((B / name).exists() for name in contract['exclusive_outputs'])
    assert not (B.parent / 'results_stage032.json').exists()
    assert 30 in os.sched_getaffinity(0)
    command = ['timeout', '--kill-after=5s', '60s', 'taskset', '-c', '30',
               '/usr/bin/python3', str(B / 'refresh.py'), '--execute',
               '--contract-sha256', contract_sha]
    seal = {
        'status': 'root_source_JSON_reviewed_admitted_metadata_only',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'contract_sha256': contract_sha,
        'handoff_sha256': sha(B / 'handoff.json'),
        'launcher_sha256': sha(Path(__file__)),
        'registry_before_sha256': sha(registry),
        'command': command,
        'review': ['complete driver and exact contract read',
                   'all handoff and source_JSON identities verified',
                   'old scientific fields and failed gates preserved',
                   'GPU source preparation and native31 HF partial scopes distinct'],
        'resources': {'CPU': 30, 'threads': 1, 'maximum_seconds': 60,
                      'RLIMIT_AS_bytes': 1024 ** 3},
        'authorization': 'one metadata refresh; zero science, NPZ, API or card sends',
    }
    once('root_admission.json', seal)
    started = time.monotonic()
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    with (B / 'root_stdout.log').open('x') as output, (B / 'root_stderr.log').open('x') as errors:
        proc = subprocess.run(command, cwd=R, env=env, stdin=subprocess.DEVNULL,
                              stdout=output, stderr=errors, preexec_fn=limits,
                              timeout=70, check=False)
    receipt = {
        'status': 'foreground_closed', 'exit_code': proc.returncode,
        'wall_seconds_uncontrolled': time.monotonic() - started,
        'root_admission_sha256': sha(B / 'root_admission.json'),
        'contract_sha256': contract_sha,
        'registry_after_sha256': sha(registry),
        'stdout_sha256': sha(B / 'root_stdout.log'),
        'stderr_sha256': sha(B / 'root_stderr.log'),
        'science_NPZ_API_card_calls': 0,
    }
    for name in ('integration_proof.json', 'current_card_evidence.json'):
        if (B / name).exists():
            receipt[name + '_sha256'] = sha(B / name)
    once('root_execution.json', receipt)
    print(json.dumps(receipt))
    if proc.returncode:
        raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
