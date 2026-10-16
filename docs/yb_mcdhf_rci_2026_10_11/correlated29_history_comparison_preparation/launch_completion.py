"""Once foreground, capped stdlib-only closure; no physical imports."""
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


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    contract_path = HERE / 'completion_contract.json'
    assert sha(contract_path) == '1c4ac9f0cb95313c0fbe5038cb4bd14b8006300f8d2357b67228a30b3864b7ba'
    contract = json.loads(contract_path.read_text())
    assert sha(HERE / 'complete_saved.py') == contract['script_sha256']
    assert sha(HERE / 'completion_preparation.json') == '85e4d4be25afdce0a1ed809f28af8289f52a76663ef8ba78e1f10a393cfb97e1'
    assert not (HERE / 'completion_execution.json').exists()
    for path, expected in contract['inputs_sha256'].items():
        assert sha(REPO / path) == expected
    caps = contract['resources']
    assert caps['CPU'] in os.sched_getaffinity(0)

    def limits():
        os.sched_setaffinity(0, {caps['CPU']})
        resource.setrlimit(resource.RLIMIT_AS, (caps['max_address_space_bytes'],) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (caps['max_wall_seconds'],) * 2)

    env = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env[key] = '1'
    command = ['timeout', '--signal=TERM', '--kill-after=5s', '60s',
               str(REPO / '.venv/bin/python'), str(HERE / 'complete_saved.py')]
    receipt = {'status': 'foreground_started', 'started_utc': now(),
               'contract_sha256': sha(contract_path), 'script_sha256': sha(HERE / 'complete_saved.py'),
               'launcher_sha256': sha(__file__), 'resources': caps, 'command': command,
               'scientific_calls': 0, 'array_numerical_replays': 0}
    with (HERE / 'completion_execution.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    begin = time.monotonic()
    with (HERE / 'completion_execution.log').open('xb') as log:
        completed = subprocess.run(command, cwd=REPO, env=env, preexec_fn=limits,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=65)
    receipt.update(status='foreground_closed', exit_code=completed.returncode,
                   elapsed_uncontrolled_seconds=time.monotonic() - begin, finished_utc=now(),
                   log_sha256=sha(HERE / 'completion_execution.log'))
    receipt['completion_files_sha256'] = {str(path.relative_to(HERE)): sha(path) for path in
        (HERE / 'completion.json', HERE / 'history10/completion.json', HERE / 'history40/completion.json')
        if path.exists()}
    write(HERE / 'completion_execution.json', receipt)
    print(json.dumps(receipt))
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
