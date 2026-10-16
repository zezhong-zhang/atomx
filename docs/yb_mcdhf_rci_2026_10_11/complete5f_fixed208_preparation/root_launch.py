"""One foreground root launch and durable closed receipt, including timeout."""
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from run import authorize

B = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    binding_path = B / 'execution_binding.json'
    binding, contract, _roles, _manifest, _source = authorize(binding_path)
    assert binding['launcher_sha256'] == sha(__file__)
    assert not (B / 'actual').exists()
    assert binding['warm_benchmark_foreground_closed_before_this_launch']
    bench = B.parent / 'correlated29_radial_reuse_qualification003_preparation/warm_benchmark_preparation/execution_receipt.json'
    assert sha(bench) == binding['warm_benchmark_receipt_sha256']
    assert json.loads(bench.read_text())['status'] == 'foreground_closed'
    with (B / 'root_once_marker.json').open('x') as handle:
        handle.write(json.dumps({'binding_sha256': sha(binding_path), 'launcher_sha256': sha(__file__)}))
        handle.flush()
        os.fsync(handle.fileno())
    command = ['timeout', '--signal=TERM', '--kill-after=5s', '2055s',
               '/data/code/atomx/.venv/bin/python', str(B / 'run.py'), '--execute', '--binding', str(binding_path)]
    environment = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        environment[name] = '1'
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    with (B / 'execution.log').open('xb') as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(B.parents[2]))
    receipt = {'status': 'foreground_closed', 'exit_code': result.returncode,
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_uncontrolled_seconds': time.monotonic() - start, 'command': command,
        'binding_sha256': sha(binding_path), 'driver_sha256': sha(B / 'run.py'),
        'launcher_sha256': sha(__file__), 'contract_sha256': sha(B / 'contract.json'),
        'log_sha256': sha(B / 'execution.log'), 'partial_outputs_retained': True,
        'no_retry': True, 'timing_claim_valid': False,
        'actual_card_trigger': binding['actual_iteration_trigger'],
        'outer_wall_seconds': 2055, 'kill_after_seconds': 5}
    (B / 'root_execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'exit_code': result.returncode,
                      'elapsed_s': receipt['elapsed_uncontrolled_seconds'], 'receipt_sha256': sha(B / 'root_execution.json')}))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
