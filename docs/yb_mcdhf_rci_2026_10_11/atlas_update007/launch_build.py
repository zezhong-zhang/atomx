"""Once foreground capped local saved-JSON build and independent scalar check."""
import datetime
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    execution = json.loads((HERE / 'execution_contract.json').read_text())
    assert execution['contract_sha256'] == sha(HERE / 'contract.json')
    for name, expected in execution['scripts_sha256'].items():
        assert sha(HERE / name) == expected
    caps = execution['resources']
    assert caps['CPU'] in os.sched_getaffinity(0)
    with (HERE / 'once_marker.json').open('x') as stream:
        json.dump({'execution_contract_sha256': sha(HERE / 'execution_contract.json'), 'started_utc': now()}, stream)

    def limits():
        os.sched_setaffinity(0, {caps['CPU']})
        resource.setrlimit(resource.RLIMIT_AS, (caps['memory_bytes'],) * 2)

    env = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env[key] = '1'
    receipt = {'status': 'foreground_started', 'started_utc': now(), 'resources': caps,
               'execution_contract_sha256': sha(HERE / 'execution_contract.json'), 'launcher_sha256': sha(__file__),
               'commands': [], 'scientific_calls': 0, 'array_numerical_replays': 0, 'publication_calls': 0}
    start = time.monotonic()
    for name in execution['commands']:
        command = [str(REPO / '.venv/bin/python'), str(HERE / name)]
        with (HERE / (name + '.stdout.log')).open('xb') as out, (HERE / (name + '.stderr.log')).open('xb') as err:
            process = subprocess.run(command, cwd=REPO, env=env, preexec_fn=limits, stdout=out, stderr=err,
                                     timeout=max(1., caps['max_wall_seconds'] - (time.monotonic() - start)))
        receipt['commands'].append({'script': name, 'sha256': sha(HERE / name), 'exit_code': process.returncode,
             'stdout_sha256': sha(HERE / (name + '.stdout.log')), 'stderr_sha256': sha(HERE / (name + '.stderr.log'))})
        if process.returncode:
            break
    receipt.update(status='foreground_closed', exit_code=receipt['commands'][-1]['exit_code'],
                   finished_utc=now(), elapsed_uncontrolled_seconds=time.monotonic() - start)
    (HERE / 'build_execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    return receipt['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
