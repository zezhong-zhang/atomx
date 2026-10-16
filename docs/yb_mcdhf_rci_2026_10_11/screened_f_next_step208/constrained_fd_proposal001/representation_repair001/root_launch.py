"""One root-admitted native31 finite-difference execution, with a closed receipt."""
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parent
R = next(parent for parent in B.parents if (parent / 'pyproject.toml').is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seal = json.loads((B / 'execution_binding.json').read_text())
    for name, expected in seal['root_owned_frozen_files'].items():
        assert sha(B / name) == expected, name
    assert sha(Path(__file__)) == seal['launcher_sha256']
    assert not (B / 'actual').exists()
    caps = seal['resources']
    assert caps['CPU'] in os.sched_getaffinity(0)
    assert os.getloadavg()[0] <= caps['maximum_system_load1']
    with (B / 'root_once_marker.json').open('x') as stream:
        json.dump(dict(binding_sha256=sha(B / 'execution_binding.json'),
                       launcher_sha256=sha(Path(__file__))), stream)
        stream.flush()
        os.fsync(stream.fileno())
    env = dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                 'NUMBA_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[name] = '1'
    env['PYTHONHASHSEED'] = '0'
    command = ['timeout', '--signal=TERM', '--kill-after=5s',
               str(caps['outer_seconds']) + 's', 'taskset', '-c', str(caps['CPU']),
               caps['interpreter_literal'], str(B / 'run.py'),
               '--execution-binding', str(B / 'execution_binding.json')]
    started = datetime.now(timezone.utc).isoformat()
    wall = time.monotonic()
    with (B / 'execution.log').open('xb') as stream:
        process = subprocess.run(command, env=env, cwd=str(R),
                                 stdout=stream, stderr=subprocess.STDOUT)
    receipt = dict(status='foreground_closed', exit_code=process.returncode,
                   started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_uncontrolled_s=time.monotonic() - wall, command=command,
                   binding_sha256=sha(B / 'execution_binding.json'),
                   driver_sha256=sha(B / 'run.py'), contract_sha256=sha(B / 'contract.json'),
                   launcher_sha256=sha(Path(__file__)), log_sha256=sha(B / 'execution.log'),
                   outer_wall_seconds=caps['outer_seconds'], kill_after_seconds=5,
                   partial_outputs_preserved=True, no_retry=True, timing_claim_valid=False,
                   actual_iteration_trigger=seal['actual_iteration_trigger'])
    with (B / 'root_execution.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=receipt['status'], exit_code=process.returncode,
                          elapsed_s=receipt['elapsed_uncontrolled_s'],
                          receipt_sha256=sha(B / 'root_execution.json'))))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
